#include "grad.h"

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
    {
        int batch_size = 1, seq_len = 2, d_model = 4, num_heads = 2;
        int qkv_dims[] = {batch_size, seq_len, d_model};
        int mask_dims[] = {batch_size, num_heads, seq_len, seq_len};
        
        float q_data[] = {1,1,0,0, 0,0,1,1};
        float k_data[] = {1,1,0,0, 0,0,1,1};
        float v_data[] = {1,1,2,2, 3,3,4,4};
        // Causal mask: upper triangle is 0
        float mask_data[] = {1,0, 1,1};
        
        Tensor *Q = tensor_new(3, qkv_dims, q_data, 1);
        Tensor *K = tensor_new(3, qkv_dims, k_data, 1);
        Tensor *V = tensor_new(3, qkv_dims, v_data, 1);
        Tensor *mask = tensor_new(4, mask_dims, mask_data, 0);

        printf("\nTest 1: Masked Multi-Head Attention\n\nInput values:\nQ:"); 
        for(int i = 0; i < 8; i++) printf(" %f", q_data[i]);
        printf("\nK:"); for(int i = 0; i < 8; i++) printf(" %f", k_data[i]);
        printf("\nV:"); for(int i = 0; i < 8; i++) printf(" %f", v_data[i]);
        printf("\nMask:"); for(int i = 0; i < 4; i++) printf(" %f", mask_data[i]);

        Tensor* output = tensor_masked_multihead_attention(Q, K, V, mask, num_heads);
        printf("\n\nFinal output values (with causal masking):\n");
        for (int i = 0; i < seq_len; i++) {
            printf("Seq %d:", i);
            for (int j = 0; j < d_model; j++) printf(" %6.3f", output->data[i * d_model + j]);
            printf("\n");
        }

        // Test gradients
        for (int i = 0; i < output->size; i++) output->grad[i] = 1.0f;
        backward();
        printf("\nQ gradients:\n");
        for (int i = 0; i < seq_len; i++) {
            printf("Seq %d:", i);
            for (int j = 0; j < d_model; j++) printf(" %6.3f", Q->grad[i * d_model + j]);
            printf("\n");
        }
    }

    {
        // Test with different mask patterns
        int batch_size = 1, seq_len = 3, d_model = 4, num_heads = 2;
        int qkv_dims[] = {batch_size, seq_len, d_model};
        int mask_dims[] = {batch_size, num_heads, seq_len, seq_len};
        
        float q_data[12], k_data[12], v_data[12];
        for (int i = 0; i < seq_len * d_model; i++) {
            q_data[i] = k_data[i] = 1.0f;
            v_data[i] = (i/d_model) + 1.0f;
        }
        
        // Create a mask that only allows attention to even positions
        float mask_data[] = {1,0,1, 1,0,1, 1,0,1};
        
        Tensor *Q = tensor_new(3, qkv_dims, q_data, 1);
        Tensor *K = tensor_new(3, qkv_dims, k_data, 1);
        Tensor *V = tensor_new(3, qkv_dims, v_data, 1);
        Tensor *mask = tensor_new(4, mask_dims, mask_data, 0);

        Tensor* output = tensor_masked_multihead_attention(Q, K, V, mask, num_heads);
        printf("\nTest 2: Custom Mask Pattern\nOutput:\n");
        for (int i = 0; i < seq_len; i++) {
            printf("Seq %d:", i);
            for (int j = 0; j < d_model; j++) printf(" %6.3f", output->data[i * d_model + j]);
            printf("\n");
        }
    }

    {
        // Create input tensor (batch_size x input_dim)
        int input_dims[] = {2, 4};  // 2 samples, 4 input features
        float input_data[] = {
            0.5f, -0.2f, 0.1f, 0.3f,
            -0.1f, 0.4f, -0.3f, 0.2f
        };
        Tensor* input = tensor_new(2, input_dims, input_data, 1);

        // Create weights tensor (input_dim x output_dim)
        int weight_dims[] = {4, 3};  // 4 input features, 3 output features
        float weight_data[] = {
            0.1f, 0.2f, -0.1f,
            -0.2f, 0.1f, 0.3f,
            0.0f, 0.4f, 0.1f,
            0.3f, -0.2f, 0.2f
        };
        Tensor* weights = tensor_new(2, weight_dims, weight_data, 1);

        // Create bias tensor (1 x output_dim)
        int bias_dims[] = {1, 3};
        float bias_data[] = {0.1f, -0.1f, 0.2f};
        Tensor* bias = tensor_new(2, bias_dims, bias_data, 1);

        // Apply feedforward
        Tensor* output = tensor_feedforward(input, weights, bias);

        // Print results
        printf("Input:\n");
        for (int i = 0; i < input->size; i++) {
            printf("%f ", input->data[i]);
            if ((i + 1) % input_dims[1] == 0) printf("\n");
        }

        printf("\nOutput:\n");
        for (int i = 0; i < output->size; i++) {
            printf("%f ", output->data[i]);
            if ((i + 1) % weight_dims[1] == 0) printf("\n");
        }

        // Test backward pass
        // Set gradient of output to 1.0
        for (int i = 0; i < output->size; i++) {
            output->grad[i] = 1.0f;
        }

        backward();

        printf("\nInput gradients:\n");
        for (int i = 0; i < input->size; i++) {
            printf("%f ", input->grad[i]);
            if ((i + 1) % input_dims[1] == 0) printf("\n");
        }

        printf("\nWeight gradients:\n");
        for (int i = 0; i < weights->size; i++) {
            printf("%f ", weights->grad[i]);
            if ((i + 1) % weight_dims[1] == 0) printf("\n");
        }

        printf("\nBias gradients:\n");
        for (int i = 0; i < bias->size; i++) {
            printf("%f ", bias->grad[i]);
        }
        printf("\n");
    } 

    clean_registry();
    return 0;
}