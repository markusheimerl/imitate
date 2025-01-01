#include "transformer.h"

// Function that processes input through a transformer layer
Tensor* transformer_layer(Tensor* x, Tensor* mask, 
                         int num_heads, float eps,
                         Tensor* W_ff1, Tensor* W_ff2,
                         Tensor* b_ff1, Tensor* b_ff2) {
    
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
    Tensor* W_ff1 = tensor_randn(2, ff1_dims, 1);
    
    int ff2_dims[] = {ff_dim, d_model};
    Tensor* W_ff2 = tensor_randn(2, ff2_dims, 1);
    
    // Initialize feed-forward biases
    int bias_ff1_dims[] = {1, ff_dim};
    Tensor* b_ff1 = tensor_randn(2, bias_ff1_dims, 1);
    
    int bias_ff2_dims[] = {1, d_model};
    Tensor* b_ff2 = tensor_randn(2, bias_ff2_dims, 1);

    printf("Processing sequence through decoder layer...\n");

    // Forward pass through transformer layer
    Tensor* output = transformer_layer(x, mask, num_heads, eps, W_ff1, W_ff2, b_ff1, b_ff2);

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