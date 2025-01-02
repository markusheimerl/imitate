#include "grad.h"

void assert_float_eq(float a, float b, float eps, const char* msg) {
    if (fabsf(a - b) > eps) {
        printf("ASSERTION FAILED: %s\n", msg);
        printf("Expected: %f, Got: %f\n", b, a);
        exit(1);
    }
}

// Feed Forward Network (simplified to match available operations)
Tensor* feed_forward(Tensor* W_in, Tensor* W_out, Tensor* x) {
    Tensor* intermediate = tensor_matmul(x, W_in);
    Tensor* activated = tensor_gelu(intermediate);
    return tensor_matmul(activated, W_out);
}

// Self-attention mechanism
Tensor* attention(Tensor* W_q, Tensor* W_k, Tensor* W_v, Tensor* W_o, 
                 Tensor* x, Tensor* scale_tensor, Tensor* causal_mask,
                 int batch_size, int seq_len, int n_head, int d_model) {
    int d_head = d_model / n_head;
    
    // Linear projections
    Tensor* Q = tensor_matmul(x, W_q);
    Tensor* K = tensor_matmul(x, W_k);
    Tensor* V = tensor_matmul(x, W_v);
    
    // Reshape for multi-head attention
    int qkv_dims[] = {batch_size, seq_len, n_head, d_head};
    Q = tensor_reshape(Q, 4, qkv_dims);
    K = tensor_reshape(K, 4, qkv_dims);
    V = tensor_reshape(V, 4, qkv_dims);
    
    // Transpose for attention
    int perm[] = {0, 2, 1, 3};
    Q = tensor_permute(Q, perm, 4);
    K = tensor_permute(K, perm, 4);
    V = tensor_permute(V, perm, 4);
    
    // Attention scores
    int perm_k[] = {0, 1, 3, 2};
    Tensor* K_t = tensor_permute(K, perm_k, 4);
    Tensor* scores = tensor_matmul(Q, K_t);
    
    // Scale scores and apply mask
    Tensor* scaled_scores = tensor_hadamard(scores, scale_tensor);
    Tensor* masked_scores = tensor_hadamard(scaled_scores, causal_mask);
    
    // Attention and output
    Tensor* attn = tensor_softmax(masked_scores);
    Tensor* out = tensor_matmul(attn, V);
    
    // Reshape back
    int perm_back[] = {0, 2, 1, 3};
    out = tensor_permute(out, perm_back, 4);
    
    int out_dims[] = {batch_size, seq_len, d_model};
    out = tensor_reshape(out, 3, out_dims);
    
    // Final projection
    return tensor_matmul(out, W_o);
}

// Transformer block
Tensor* transformer_block(Tensor* W_q, Tensor* W_k, Tensor* W_v, Tensor* W_o,
                        Tensor* W_ff1, Tensor* W_ff2, Tensor* x,
                        Tensor* scale_tensor, Tensor* causal_mask,
                        int batch_size, int seq_len, int n_head, int d_model) {
    // Self-attention with residual
    Tensor* normed = tensor_rms_norm(x, 1e-5f);
    Tensor* attn_out = attention(W_q, W_k, W_v, W_o, normed, 
                                scale_tensor, causal_mask,
                                batch_size, seq_len, n_head, d_model);
    Tensor* res1 = tensor_add(x, attn_out);
    
    // Feed-forward with residual
    Tensor* normed2 = tensor_rms_norm(res1, 1e-5f);
    Tensor* ff_out = feed_forward(W_ff1, W_ff2, normed2);
    return tensor_add(res1, ff_out);
}

int main() {
    // Configuration
    const int batch_size = 1;
    const int seq_len = 4;
    const int d_model = 64;
    const int n_head = 4;
    const int n_layers = 2;
    const int d_head = d_model / n_head;
    
    printf("Testing transformer with self-attention and feed-forward...\n");
    
    // Create input with controlled values
    int dims[] = {batch_size, seq_len, d_model};
    Tensor* x = tensor_new(3, dims, NULL, 1);
    for (int i = 0; i < x->size; i++) {
        x->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    
    // Create attention scale tensor
    float scale_val = 1.0f / sqrtf(d_head);
    int scale_dims[] = {batch_size, n_head, seq_len, seq_len};
    Tensor* scale_tensor = tensor_new(4, scale_dims, NULL, 0);
    for (int i = 0; i < scale_tensor->size; i++) {
        scale_tensor->data[i] = scale_val;
    }
    
    // Create causal mask
    Tensor* causal_mask = tensor_new(4, scale_dims, NULL, 0);
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < n_head; h++) {
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < seq_len; j++) {
                    int idx = ((b * n_head + h) * seq_len + i) * seq_len + j;
                    causal_mask->data[idx] = j <= i ? 1.0f : -1e9f;
                }
            }
        }
    }
    
    // Initialize weights for each layer
    Tensor** W_q = malloc(n_layers * sizeof(Tensor*));
    Tensor** W_k = malloc(n_layers * sizeof(Tensor*));
    Tensor** W_v = malloc(n_layers * sizeof(Tensor*));
    Tensor** W_o = malloc(n_layers * sizeof(Tensor*));
    Tensor** W_ff1 = malloc(n_layers * sizeof(Tensor*));
    Tensor** W_ff2 = malloc(n_layers * sizeof(Tensor*));
    
    int attn_dims[] = {d_model, d_model};
    int ff_dims1[] = {d_model, d_model * 4};
    int ff_dims2[] = {d_model * 4, d_model};
    
    float w_scale = sqrtf(2.0f / d_model);
    
    for (int l = 0; l < n_layers; l++) {
        W_q[l] = tensor_randn(2, attn_dims, 1);
        W_k[l] = tensor_randn(2, attn_dims, 1);
        W_v[l] = tensor_randn(2, attn_dims, 1);
        W_o[l] = tensor_randn(2, attn_dims, 1);
        W_ff1[l] = tensor_randn(2, ff_dims1, 1);
        W_ff2[l] = tensor_randn(2, ff_dims2, 1);
        
        for (int i = 0; i < d_model * d_model; i++) {
            W_q[l]->data[i] *= w_scale;
            W_k[l]->data[i] *= w_scale;
            W_v[l]->data[i] *= w_scale;
            W_o[l]->data[i] *= w_scale;
        }
        for (int i = 0; i < d_model * d_model * 4; i++) {
            W_ff1[l]->data[i] *= w_scale;
        }
        for (int i = 0; i < d_model * 4 * d_model; i++) {
            W_ff2[l]->data[i] *= w_scale;
        }
    }
    
    // Forward pass function
    Tensor* forward(Tensor* input) {
        Tensor* current = input;
        for (int l = 0; l < n_layers; l++) {
            current = transformer_block(
                W_q[l], W_k[l], W_v[l], W_o[l],
                W_ff1[l], W_ff2[l], current,
                scale_tensor, causal_mask,
                batch_size, seq_len, n_head, d_model
            );
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
    const float epsilon = 1e-4f;  // Increased epsilon for better stability
    float saved = x->data[0];
    x->data[0] += epsilon;
    Tensor* perturbed_output = forward(x);
    float numerical_grad = (perturbed_output->data[0] - original_output) / epsilon;
    x->data[0] = saved;
    
    // Compare gradients
    float abs_error = fabsf(analytical_grad - numerical_grad);
    float rel_error = abs_error / (fabsf(analytical_grad) + fabsf(numerical_grad) + 1e-6f);
    
    printf("Gradient check:\n");
    printf("Analytical: %.6e\n", analytical_grad);
    printf("Numerical:  %.6e\n", numerical_grad);
    printf("Relative error: %.6f\n", rel_error);
    
    assert_float_eq(rel_error < 0.05f ? 1.0f : 0.0f, 1.0f, 1e-5f,
                   "Gradient verification failed");
    
    printf("All tests passed!\n");
    
    // Memory cleanup
    free(W_q);
    free(W_k);
    free(W_v);
    free(W_o);
    free(W_ff1);
    free(W_ff2);
    clean_registry();
    
    return 0;
}