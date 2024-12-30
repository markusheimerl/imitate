#include "grad/grad.h"
#include <math.h>

// Transformer hyperparameters (small for testing)
#define BATCH_SIZE 2
#define SEQ_LENGTH 4
#define EMBED_DIM 8
#define NUM_HEADS 2
#define HEAD_DIM (EMBED_DIM / NUM_HEADS)
#define FF_DIM (4 * EMBED_DIM)

void print_tensor(const char* name, Tensor* t) {
    printf("%s shape: ", name);
    for (int i = 0; i < t->ndims; i++) {
        printf("%d", t->dims[i]);
        if (i < t->ndims - 1) printf("x");
    }
    printf("\n");
    
    if (t->size <= 32) {
        printf("First few values: ");
        for (int i = 0; i < MIN(t->size, 8); i++) {
            printf("%.4f ", t->data[i]);
        }
        printf("\n");
    }
}

Tensor* create_random_tensor(int ndims, int* dims) {
    int size = 1;
    for (int i = 0; i < ndims; i++) size *= dims[i];
    
    float* data = malloc(size * sizeof(float));
    if (!data) {
        printf("Failed to allocate memory for tensor data\n");
        return NULL;
    }
    
    // Initialize with small random values
    float scale = sqrtf(2.0f / size);  // Xavier initialization
    for (int i = 0; i < size; i++) {
        data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
    }
    
    Tensor* t = tensor_new(ndims, dims, data, 1);
    free(data);
    return t;
}

// Create causal attention mask
Tensor* create_causal_mask(int seq_length) {
    int dims[] = {seq_length, seq_length};
    float* data = malloc(seq_length * seq_length * sizeof(float));
    
    // Create lower triangular matrix (1.0 for allowed attention, 0.0 for masked)
    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < seq_length; j++) {
            data[i * seq_length + j] = j <= i ? 1.0f : 0.0f;
        }
    }
    
    Tensor* mask = tensor_new(2, dims, data, 0);  // No gradient needed for mask
    free(data);
    return mask;
}

// Causal self-attention
Tensor* causal_self_attention(Tensor* x, Tensor* w_q, Tensor* w_k, Tensor* w_v, Tensor* w_o) {
    print_tensor("Self-attention input", x);
    
    // Linear projections
    Tensor* q = tensor_matmul(x, w_q);
    Tensor* k = tensor_matmul(x, w_k);
    Tensor* v = tensor_matmul(x, w_v);
    
    print_tensor("Q", q);
    print_tensor("K", k);
    print_tensor("V", v);
    
    // Scaled dot-product attention with causal mask
    float scale = 1.0f / sqrtf(HEAD_DIM);
    Tensor* scores = tensor_matmul(q, k);  // [batch, seq, seq]
    
    // Apply scaling
    for (int i = 0; i < scores->size; i++) {
        scores->data[i] *= scale;
    }
    
    // Apply causal mask
    Tensor* mask = create_causal_mask(SEQ_LENGTH);
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int i = 0; i < SEQ_LENGTH; i++) {
            for (int j = 0; j < SEQ_LENGTH; j++) {
                int idx = b * SEQ_LENGTH * SEQ_LENGTH + i * SEQ_LENGTH + j;
                if (mask->data[i * SEQ_LENGTH + j] == 0.0f) {
                    scores->data[idx] = -INFINITY;
                }
            }
        }
    }
    
    // Softmax
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int i = 0; i < SEQ_LENGTH; i++) {
            float max_val = -INFINITY;
            float sum = 0.0f;
            
            // Find max
            for (int j = 0; j < SEQ_LENGTH; j++) {
                int idx = b * SEQ_LENGTH * SEQ_LENGTH + i * SEQ_LENGTH + j;
                max_val = fmaxf(max_val, scores->data[idx]);
            }
            
            // Compute exp and sum
            for (int j = 0; j < SEQ_LENGTH; j++) {
                int idx = b * SEQ_LENGTH * SEQ_LENGTH + i * SEQ_LENGTH + j;
                scores->data[idx] = expf(scores->data[idx] - max_val);
                sum += scores->data[idx];
            }
            
            // Normalize
            for (int j = 0; j < SEQ_LENGTH; j++) {
                int idx = b * SEQ_LENGTH * SEQ_LENGTH + i * SEQ_LENGTH + j;
                scores->data[idx] /= sum;
            }
        }
    }
    
    // Apply attention to values
    Tensor* attention = tensor_matmul(scores, v);
    
    // Project back to output space
    Tensor* output = tensor_matmul(attention, w_o);
    
    // Cleanup
    tensor_free(q);
    tensor_free(k);
    tensor_free(v);
    tensor_free(scores);
    tensor_free(mask);
    tensor_free(attention);
    
    return output;
}

// Feed-forward network
Tensor* feed_forward(Tensor* x, Tensor* w1, Tensor* b1, Tensor* w2, Tensor* b2) {
    // First linear layer + ReLU
    Tensor* hidden = tensor_matmul(x, w1);
    Tensor* hidden_bias = tensor_add(hidden, b1);
    Tensor* activated = tensor_relu(hidden_bias);
    
    // Second linear layer
    Tensor* output = tensor_matmul(activated, w2);
    Tensor* output_bias = tensor_add(output, b2);
    
    // Cleanup
    tensor_free(hidden);
    tensor_free(hidden_bias);
    tensor_free(activated);
    tensor_free(output);
    
    return output_bias;
}

// Layer normalization
Tensor* layer_norm(Tensor* x, Tensor* gamma, Tensor* beta) {
    Tensor* normalized = tensor_new(x->ndims, x->dims, NULL, 1);
    
    // Compute mean and variance along last dimension
    int last_dim = x->dims[x->ndims - 1];
    int batch_size = x->size / last_dim;
    
    for (int b = 0; b < batch_size; b++) {
        float mean = 0.0f;
        float var = 0.0f;
        
        // Compute mean
        for (int i = 0; i < last_dim; i++) {
            mean += x->data[b * last_dim + i];
        }
        mean /= last_dim;
        
        // Compute variance
        for (int i = 0; i < last_dim; i++) {
            float diff = x->data[b * last_dim + i] - mean;
            var += diff * diff;
        }
        var /= last_dim;
        
        // Normalize, scale, and shift
        for (int i = 0; i < last_dim; i++) {
            float norm = (x->data[b * last_dim + i] - mean) / sqrtf(var + 1e-5f);
            normalized->data[b * last_dim + i] = gamma->data[i] * norm + beta->data[i];
        }
    }
    
    return normalized;
}

// Decoder block
Tensor* decoder_block(Tensor* x,
                     Tensor* sa_w_q, Tensor* sa_w_k, Tensor* sa_w_v, Tensor* sa_w_o,
                     Tensor* ln1_gamma, Tensor* ln1_beta,
                     Tensor* ff_w1, Tensor* ff_b1, Tensor* ff_w2, Tensor* ff_b2,
                     Tensor* ln2_gamma, Tensor* ln2_beta) {
    // Layer norm 1
    Tensor* norm1 = layer_norm(x, ln1_gamma, ln1_beta);
    
    // Causal self-attention
    Tensor* attention = causal_self_attention(norm1, sa_w_q, sa_w_k, sa_w_v, sa_w_o);
    Tensor* residual1 = tensor_add(x, attention);
    
    // Layer norm 2
    Tensor* norm2 = layer_norm(residual1, ln2_gamma, ln2_beta);
    
    // Feed-forward
    Tensor* ff = feed_forward(norm2, ff_w1, ff_b1, ff_w2, ff_b2);
    Tensor* residual2 = tensor_add(residual1, ff);
    
    // Cleanup
    tensor_free(norm1);
    tensor_free(attention);
    tensor_free(residual1);
    tensor_free(norm2);
    tensor_free(ff);
    
    return residual2;
}

void transformer_test() {
    printf("Initializing tensors...\n");
    
    // Initialize input tensor
    int input_dims[] = {BATCH_SIZE, SEQ_LENGTH, EMBED_DIM};
    float* input_data = malloc(BATCH_SIZE * SEQ_LENGTH * EMBED_DIM * sizeof(float));
    for (int i = 0; i < BATCH_SIZE * SEQ_LENGTH * EMBED_DIM; i++) {
        input_data[i] = (float)rand() / RAND_MAX * 0.1f;
    }
    Tensor* input = tensor_new(3, input_dims, input_data, 1);
    
    // Initialize weights
    printf("Initializing weights...\n");
    
    int qkv_dims[] = {EMBED_DIM, EMBED_DIM};
    Tensor* sa_w_q = create_random_tensor(2, qkv_dims);
    Tensor* sa_w_k = create_random_tensor(2, qkv_dims);
    Tensor* sa_w_v = create_random_tensor(2, qkv_dims);
    Tensor* sa_w_o = create_random_tensor(2, qkv_dims);
    
    int ln_dims[] = {EMBED_DIM};
    Tensor* ln1_gamma = create_random_tensor(1, ln_dims);
    Tensor* ln1_beta = create_random_tensor(1, ln_dims);
    
    int ff1_dims[] = {EMBED_DIM, FF_DIM};
    int ff2_dims[] = {FF_DIM, EMBED_DIM};
    int ff1_bias_dims[] = {FF_DIM};
    int ff2_bias_dims[] = {EMBED_DIM};
    
    Tensor* ff_w1 = create_random_tensor(2, ff1_dims);
    Tensor* ff_b1 = create_random_tensor(1, ff1_bias_dims);
    Tensor* ff_w2 = create_random_tensor(2, ff2_dims);
    Tensor* ff_b2 = create_random_tensor(1, ff2_bias_dims);
    
    Tensor* ln2_gamma = create_random_tensor(1, ln_dims);
    Tensor* ln2_beta = create_random_tensor(1, ln_dims);
    
    printf("Processing through decoder block...\n");
    
    Tensor* output = decoder_block(input,
                                 sa_w_q, sa_w_k, sa_w_v, sa_w_o,
                                 ln1_gamma, ln1_beta,
                                 ff_w1, ff_b1, ff_w2, ff_b2,
                                 ln2_gamma, ln2_beta);
    
    if (output) {
        print_tensor("Final output", output);
        tensor_free(output);
    }
    
    // Cleanup
    printf("Cleaning up...\n");
    tensor_free(input);
    tensor_free(sa_w_q);
    tensor_free(sa_w_k);
    tensor_free(sa_w_v);
    tensor_free(sa_w_o);
    tensor_free(ln1_gamma);
    tensor_free(ln1_beta);
    tensor_free(ff_w1);
    tensor_free(ff_b1);
    tensor_free(ff_w2);
    tensor_free(ff_b2);
    tensor_free(ln2_gamma);
    tensor_free(ln2_beta);
    free(input_data);
    
    printf("Test completed\n");
}

int main() {
    srand(42);
    printf("Starting decoder-only transformer test\n");
    transformer_test();
    return 0;
}