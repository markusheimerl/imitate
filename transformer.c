#include "grad.h"

void assert_float_eq(float a, float b, float eps, const char* msg) {
    if (fabsf(a - b) > eps) {
        printf("ASSERTION FAILED: %s\nExpected: %f, Got: %f\n", msg, b, a);
        exit(1);
    }
}

Tensor* feed_forward(Tensor* W_in, Tensor* W_out, Tensor* x) {
    return tensor_matmul(tensor_gelu(tensor_matmul(x, W_in)), W_out);
}

Tensor* attention(Tensor* W_q, Tensor* W_k, Tensor* W_v, Tensor* W_o, 
                 Tensor* x, Tensor* scale_tensor, Tensor* causal_mask,
                 int batch_size, int seq_len, int n_head, int d_model) {
    int d_head = d_model / n_head;
    int qkv_dims[] = {batch_size, seq_len, n_head, d_head};
    int perm[] = {0, 2, 1, 3}, perm_k[] = {0, 1, 3, 2}, perm_back[] = {0, 2, 1, 3};
    int out_dims[] = {batch_size, seq_len, d_model};
    
    Tensor *Q = tensor_permute(tensor_reshape(tensor_matmul(x, W_q), 4, qkv_dims), perm, 4);
    Tensor *K = tensor_permute(tensor_reshape(tensor_matmul(x, W_k), 4, qkv_dims), perm, 4);
    Tensor *V = tensor_permute(tensor_reshape(tensor_matmul(x, W_v), 4, qkv_dims), perm, 4);
    
    Tensor *out = tensor_matmul(tensor_softmax(tensor_hadamard(
        tensor_hadamard(tensor_matmul(Q, tensor_permute(K, perm_k, 4)), scale_tensor), 
        causal_mask)), V);
    
    return tensor_matmul(tensor_reshape(tensor_permute(out, perm_back, 4), 3, out_dims), W_o);
}

Tensor* transformer_block(Tensor* W_q, Tensor* W_k, Tensor* W_v, Tensor* W_o,
                        Tensor* W_ff1, Tensor* W_ff2, Tensor* x,
                        Tensor* scale_tensor, Tensor* causal_mask,
                        int batch_size, int seq_len, int n_head, int d_model) {
    Tensor *normed = tensor_rms_norm(x, 1e-5f);
    return tensor_add(tensor_add(x, attention(W_q, W_k, W_v, W_o, normed, 
        scale_tensor, causal_mask, batch_size, seq_len, n_head, d_model)),
        feed_forward(W_ff1, W_ff2, tensor_rms_norm(x, 1e-5f)));
}

int main() {
    const int batch_size = 1, seq_len = 4, d_model = 64, n_head = 4, n_layers = 2;
    const int d_head = d_model / n_head;
    printf("Testing transformer with self-attention and feed-forward...\n");
    
    int dims[] = {batch_size, seq_len, d_model};
    Tensor* x = tensor_new(3, dims, NULL, 1);
    for (int i = 0; i < x->size; i++) x->data[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
    
    int scale_dims[] = {batch_size, n_head, seq_len, seq_len};
    Tensor* scale_tensor = tensor_new(4, scale_dims, NULL, 0);
    Tensor* causal_mask = tensor_new(4, scale_dims, NULL, 0);
    float scale_val = 1.0f / sqrtf(d_head);
    
    for (int i = 0; i < scale_tensor->size; i++) scale_tensor->data[i] = scale_val;
    for (int b = 0; b < batch_size; b++)
        for (int h = 0; h < n_head; h++)
            for (int i = 0; i < seq_len; i++)
                for (int j = 0; j < seq_len; j++)
                    causal_mask->data[((b*n_head + h)*seq_len + i)*seq_len + j] = j <= i ? 1.0f : -1e9f;
    
    Tensor **W_q = malloc(n_layers * sizeof(Tensor*));
    Tensor **W_k = malloc(n_layers * sizeof(Tensor*));
    Tensor **W_v = malloc(n_layers * sizeof(Tensor*));
    Tensor **W_o = malloc(n_layers * sizeof(Tensor*));
    Tensor **W_ff1 = malloc(n_layers * sizeof(Tensor*));
    Tensor **W_ff2 = malloc(n_layers * sizeof(Tensor*));
    
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
        for (int i = 0; i < d_model * d_model * 4; i++) W_ff1[l]->data[i] *= w_scale;
        for (int i = 0; i < d_model * 4 * d_model; i++) W_ff2[l]->data[i] *= w_scale;
    }
    
    Tensor* forward(Tensor* input) {
        Tensor* current = input;
        for (int l = 0; l < n_layers; l++)
            current = transformer_block(W_q[l], W_k[l], W_v[l], W_o[l],
                W_ff1[l], W_ff2[l], current, scale_tensor, causal_mask,
                batch_size, seq_len, n_head, d_model);
        return current;
    }
    
    Tensor* output = forward(x);
    float original_output = output->data[0];
    output->grad[0] = 1.0f;
    backward();
    float analytical_grad = x->grad[0];
    
    float saved = x->data[0];
    x->data[0] += 1e-4f;
    float numerical_grad = (forward(x)->data[0] - original_output) / 1e-4f;
    x->data[0] = saved;
    
    float rel_error = fabsf(analytical_grad - numerical_grad) / 
                     (fabsf(analytical_grad) + fabsf(numerical_grad) + 1e-6f);
    
    printf("Gradient check:\nAnalytical: %.6e\nNumerical:  %.6e\nRelative error: %.6f\n",
           analytical_grad, numerical_grad, rel_error);
    
    assert_float_eq(rel_error < 0.05f ? 1.0f : 0.0f, 1.0f, 1e-5f,
                   "Gradient verification failed");
    
    printf("All tests passed!\n");
    
    free(W_q); free(W_k); free(W_v); free(W_o); free(W_ff1); free(W_ff2);
    clean_registry();
    return 0;
}