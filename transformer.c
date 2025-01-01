#include "grad.h"

Tensor* tensor_rms_norm(Tensor* x, float eps);

Tensor* tensor_masked_multihead_attention(Tensor* Q, Tensor* K, Tensor* V, Tensor* mask, int num_heads) {
    if (!Q || !K || !V || !mask || Q->ndims != 3 || K->ndims != 3 || V->ndims != 3 || mask->ndims != 4) return NULL;
    int batch_size = Q->dims[0], seq_len_q = Q->dims[1], seq_len_k = K->dims[1], d_model = Q->dims[2];
    if (d_model % num_heads != 0 || K->dims[2] != d_model || V->dims[2] != d_model || 
        seq_len_k != V->dims[1] || batch_size != K->dims[0] || batch_size != V->dims[0] ||
        mask->dims[0] != batch_size || mask->dims[1] != num_heads || 
        mask->dims[2] != seq_len_q || mask->dims[3] != seq_len_k) return NULL;
    
    int d_head = d_model/num_heads; tape_len = 0;
    int reshape_dims[] = {batch_size, -1, num_heads, d_head}, perm[] = {0, 2, 1, 3};
    
    reshape_dims[1] = seq_len_q;
    Tensor* Q_perm = tensor_permute(tensor_reshape(Q, 4, reshape_dims), perm, 4);
    reshape_dims[1] = seq_len_k;
    Tensor* K_perm = tensor_permute(tensor_reshape(K, 4, reshape_dims), perm, 4);
    Tensor* V_perm = tensor_permute(tensor_reshape(V, 4, reshape_dims), perm, 4);
    if (!Q_perm || !K_perm || !V_perm) return NULL;

    Tensor* K_transpose = tensor_permute(K_perm, (int[]){0,1,3,2}, 4);
    if (!K_transpose) return NULL;
    
    Tensor* scores = tensor_matmul(Q_perm, K_transpose);
    if (!scores) return NULL;
    
    float scale = 1.0f/sqrt(d_head);
    Tensor* scaled_scores = tensor_hadamard(scores, tensor_new(4, (int[]){1,1,1,1}, (float[]){scale}, 0));
    if (!scaled_scores) return NULL;

    // Apply mask
    Tensor* masked_scores = tensor_hadamard(scaled_scores, mask);
    if (!masked_scores) return NULL;

    Tensor* attention = tensor_matmul(tensor_softmax(masked_scores), V_perm);
    if (!attention) return NULL;

    return tensor_reshape(tensor_permute(attention, (int[]){0,2,1,3}, 4), 3, (int[]){batch_size,seq_len_q,d_model});
}

Tensor* tensor_feedforward(Tensor* input, Tensor* weights, Tensor* bias) {
    if (!input || !weights || !bias) return NULL;
    Tensor* matmul_result = tensor_matmul(input, weights);
    Tensor* bias_added = tensor_add(matmul_result, bias);
    Tensor* output = tensor_gelu(bias_added);
    return output;
}

int main() {
    // Configuration
    int batch_size = 1;
    int seq_len = 3;
    int d_model = 8;
    int num_heads = 2;
    int ff_dim = 32;  // Feed-forward dimension, typically 4x d_model
    float eps = 1e-5f;

    // Sample input sequence
    int input_dims[] = {batch_size, seq_len, d_model};
    float input_data[] = {
        // seq pos 0
        0.1f, 0.2f, -0.1f, 0.3f, 0.2f, 0.1f, -0.2f, 0.1f,
        // seq pos 1
        0.2f, 0.1f, 0.3f, -0.1f, 0.1f, 0.2f, 0.3f, -0.2f,
        // seq pos 2
        -0.1f, 0.3f, 0.2f, 0.1f, -0.2f, 0.3f, 0.1f, 0.2f
    };
    Tensor* x = tensor_new(3, input_dims, input_data, 1);

    // Create causal mask (lower triangular)
    int mask_dims[] = {batch_size, num_heads, seq_len, seq_len};
    float mask_data[] = {
        // First head
        1,0,0,
        1,1,0,
        1,1,1,
        // Second head
        1,0,0,
        1,1,0,
        1,1,1
    };
    Tensor* mask = tensor_new(4, mask_dims, mask_data, 0);

    // Initialize feed-forward weights
    int ff1_dims[] = {d_model, ff_dim};
    int ff2_dims[] = {ff_dim, d_model};
    Tensor* W_ff1 = tensor_randn(2, ff1_dims, 1);
    Tensor* W_ff2 = tensor_randn(2, ff2_dims, 1);
    
    // Initialize feed-forward biases
    int bias_ff1_dims[] = {1, ff_dim};
    int bias_ff2_dims[] = {1, d_model};
    Tensor* b_ff1 = tensor_zeros(2, bias_ff1_dims, 1);
    Tensor* b_ff2 = tensor_zeros(2, bias_ff2_dims, 1);

    printf("Processing sequence through decoder layer...\n");

    // 1. Layer Norm 1
    Tensor* norm1 = tensor_rms_norm(x, eps);

    // 2. Self-attention
    Tensor* attn_out = tensor_masked_multihead_attention(norm1, norm1, norm1, mask, num_heads);
    
    // 3. Residual connection after attention
    Tensor* post_attn = tensor_add(x, attn_out);

    // 4. Layer Norm 2
    Tensor* norm2 = tensor_rms_norm(post_attn, eps);

    // 5. Feed-forward network with residual connection
    Tensor* ff_hidden = tensor_feedforward(norm2, W_ff1, b_ff1);
    Tensor* ff_output = tensor_feedforward(ff_hidden, W_ff2, b_ff2);
    Tensor* output = tensor_add(post_attn, ff_output);

    // Print sample outputs
    printf("\nFinal output (first sequence position):\n");
    for(int j = 0; j < d_model; j++) {
        printf("%f ", output->data[j]);
    }
    printf("\n");

    // Test backward pass
    for(int i = 0; i < output->size; i++) {
        output->grad[i] = 1.0f;
    }
    backward();

    printf("\nInput gradients (first sequence position):\n");
    for(int j = 0; j < d_model; j++) {
        printf("%f ", x->grad[j]);
    }
    printf("\n");

    clean_registry();
    return 0;
}