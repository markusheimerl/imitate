#include "transformer.h"

int main() {
    // Configuration
    const int batch_size = 1;
    const int seq_len = 4;
    const int d_model = 8;
    const int n_head = 2;
    const int d_head = d_model / n_head;
    const int n_layers = 2;
    
    printf("Testing decoder-only transformer (self-attention only)...\n");
    
    // Create input with controlled values
    int dims[] = {batch_size, seq_len, d_model};
    Tensor* x = tensor_new(3, dims, NULL, 1);
    for (int i = 0; i < x->size; i++) {
        x->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    
    // Initialize weights for each layer
    Tensor** W_q = malloc(n_layers * sizeof(Tensor*));
    Tensor** W_k = malloc(n_layers * sizeof(Tensor*));
    Tensor** W_v = malloc(n_layers * sizeof(Tensor*));
    Tensor** W_o = malloc(n_layers * sizeof(Tensor*));
    
    int weight_dims[] = {d_model, d_model};
    float w_scale = sqrtf(2.0f / d_model);
    
    for (int l = 0; l < n_layers; l++) {
        W_q[l] = tensor_randn(2, weight_dims, 1);
        W_k[l] = tensor_randn(2, weight_dims, 1);
        W_v[l] = tensor_randn(2, weight_dims, 1);
        W_o[l] = tensor_randn(2, weight_dims, 1);
        
        // Scale weights
        for (int i = 0; i < d_model * d_model; i++) {
            W_q[l]->data[i] *= w_scale;
            W_k[l]->data[i] *= w_scale;
            W_v[l]->data[i] *= w_scale;
            W_o[l]->data[i] *= w_scale;
        }
    }
    
    // Forward pass function
    Tensor* forward(Tensor* input) {
        Tensor* current = input;
        
        for (int l = 0; l < n_layers; l++) {
            // Self-attention
            Tensor* Q = tensor_matmul(current, W_q[l]);
            Tensor* K = tensor_matmul(current, W_k[l]);
            Tensor* V = tensor_matmul(current, W_v[l]);
            
            // Reshape for multi-head
            int qkv_dims[] = {batch_size, seq_len, n_head, d_head};
            Q = tensor_reshape(Q, 4, qkv_dims);
            K = tensor_reshape(K, 4, qkv_dims);
            V = tensor_reshape(V, 4, qkv_dims);
            
            // Transpose for attention
            int perm[] = {0, 2, 1, 3};  // [batch, head, seq, d_head]
            Q = tensor_permute(Q, perm, 4);
            K = tensor_permute(K, perm, 4);
            V = tensor_permute(V, perm, 4);
            
            // Attention scores
            int perm_k[] = {0, 1, 3, 2};
            Tensor* K_t = tensor_permute(K, perm_k, 4);
            Tensor* scores = tensor_matmul(Q, K_t);
            
            // Scale
            float scale = 1.0f / sqrtf(d_head);
            for (int i = 0; i < scores->size; i++) {
                scores->data[i] *= scale;
            }
            
            // Causal mask
            for (int b = 0; b < batch_size; b++) {
                for (int h = 0; h < n_head; h++) {
                    for (int i = 0; i < seq_len; i++) {
                        for (int j = i + 1; j < seq_len; j++) {
                            scores->data[((b * n_head + h) * seq_len + i) * seq_len + j] = -INFINITY;
                        }
                    }
                }
            }
            
            // Attention and output
            Tensor* attn = tensor_softmax(scores);
            Tensor* out = tensor_matmul(attn, V);
            
            // Reshape back
            int perm_back[] = {0, 2, 1, 3};
            out = tensor_permute(out, perm_back, 4);
            out = tensor_reshape(out, 3, dims);
            
            // Project and add residual
            Tensor* projected = tensor_matmul(out, W_o[l]);
            Tensor* residual = tensor_add(projected, current);
            current = tensor_rms_norm(residual, 1e-5f);
        }
        
        return current;
    }
    
    // Compute original output
    Tensor* output = forward(x);
    float original_output = output->data[0];
    
    // Compute analytical gradient
    output->grad[0] = 1.0f;
    backward();
    float analytical_grad = x->grad[0];
    
    // Compute numerical gradient
    float epsilon = 1e-5f;
    float saved = x->data[0];
    x->data[0] += epsilon;
    Tensor* perturbed_output = forward(x);
    float numerical_grad = (perturbed_output->data[0] - original_output) / epsilon;
    x->data[0] = saved;
    
    // Compare gradients
    float rel_error = fabsf(analytical_grad - numerical_grad) / 
                     (fabsf(analytical_grad) + fabsf(numerical_grad) + 1e-10f);
    
    printf("Gradient check:\n");
    printf("Analytical: %.6e\n", analytical_grad);
    printf("Numerical:  %.6e\n", numerical_grad);
    printf("Relative error: %.6f\n", rel_error);
    
    assert_float_eq(rel_error < 0.01f ? 1.0f : 0.0f, 1.0f, 1e-5,
                   "Gradient verification failed");
    
    printf("All tests passed!\n");
    
    // Cleanup
    free(W_q);
    free(W_k);
    free(W_v);
    free(W_o);
    clean_registry();
    
    return 0;
}