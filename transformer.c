#include "grad/grad.h"
#include <math.h>

// Configuration
#define BATCH_SIZE 2
#define SEQ_LENGTH 8
#define EMBED_DIM 32
#define NUM_HEADS 4
#define HEAD_DIM (EMBED_DIM / NUM_HEADS)
#define FF_DIM (4 * EMBED_DIM)

typedef struct {
    Tensor *wq, *wk, *wv, *wo;  // Attention weights
    Tensor *w1, *w2;            // FFN weights
    Tensor *norm1, *norm2;      // RMS norm weights
} TransformerBlock;

// Helper function to create random tensor
Tensor* create_random_tensor(int ndims, const int* dims) {
    int size = 1;
    for (int i = 0; i < ndims; i++) size *= dims[i];
    
    float* data = malloc(size * sizeof(float));
    float scale = 1.0f / sqrtf(size);
    for (int i = 0; i < size; i++) {
        data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
    }
    
    Tensor* t = tensor_new(ndims, dims, data, 1);
    free(data);
    return t;
}

// RMS Normalization using only tensor operations
Tensor* rms_norm(Tensor* x, Tensor* weight) {
    // Square each element
    Tensor* squared = tensor_pow(x, 2.0f);
    
    // Sum across last dimension
    int axes[] = {x->ndims - 1};
    Tensor* sum_squared = tensor_reduce_sum(squared, axes, 1);
    
    // Get mean and add epsilon (for numerical stability)
    float eps = 1e-5f;
    Tensor* mean = tensor_pow(sum_squared, 0.5f);
    
    // Reshape mean for broadcasting
    int new_dims[MAX_DIMS];
    for(int i = 0; i < x->ndims - 1; i++) new_dims[i] = x->dims[i];
    new_dims[x->ndims - 1] = 1;
    Tensor* mean_reshaped = tensor_reshape(mean, x->ndims, new_dims);
    
    // Normalize
    Tensor* normalized = tensor_hadamard(x, tensor_pow(mean_reshaped, -1.0f));
    
    // Scale with learned weights
    Tensor* output = tensor_hadamard(normalized, weight);
    
    // Cleanup
    tensor_free(squared);
    tensor_free(sum_squared);
    tensor_free(mean);
    tensor_free(mean_reshaped);
    tensor_free(normalized);
    
    return output;
}

// Softmax implementation using tensor operations
Tensor* softmax(Tensor* x) {
    // exp(x)
    Tensor* exp_x = tensor_exp(x);
    
    // Sum across last dimension
    int axes[] = {x->ndims - 1};
    Tensor* sum_exp = tensor_reduce_sum(exp_x, axes, 1);
    
    // Reshape sum for broadcasting
    int new_dims[MAX_DIMS];
    for(int i = 0; i < x->ndims - 1; i++) new_dims[i] = x->dims[i];
    new_dims[x->ndims - 1] = 1;
    Tensor* sum_exp_reshaped = tensor_reshape(sum_exp, x->ndims, new_dims);
    
    // Divide exp(x) by sum
    Tensor* output = tensor_hadamard(exp_x, tensor_pow(sum_exp_reshaped, -1.0f));
    
    // Cleanup
    tensor_free(exp_x);
    tensor_free(sum_exp);
    tensor_free(sum_exp_reshaped);
    
    return output;
}

// Modified multi-head attention using proper softmax
Tensor* multi_head_attention(Tensor* x, TransformerBlock* block) {
    // Project to Q, K, V
    Tensor* q = tensor_matmul(x, block->wq);
    Tensor* k = tensor_matmul(x, block->wk);
    Tensor* v = tensor_matmul(x, block->wv);
    
    // Compute attention scores (scaled dot product)
    Tensor* scores = tensor_matmul(q, k);
    
    // Scale scores by 1/sqrt(d_k)
    float scale = 1.0f / sqrtf(HEAD_DIM);
    Tensor* scaled_scores = tensor_pow(scores, scale);
    
    // Apply softmax to get attention weights
    Tensor* attn_weights = softmax(scaled_scores);
    Tensor* context = tensor_matmul(attn_weights, v);
    
    // Project output
    Tensor* output = tensor_matmul(context, block->wo);
    
    // Cleanup
    tensor_free(q);
    tensor_free(k);
    tensor_free(v);
    tensor_free(scores);
    tensor_free(scaled_scores);
    tensor_free(attn_weights);
    tensor_free(context);
    
    return output;
}

// Feed-forward Network
Tensor* feed_forward(Tensor* x, TransformerBlock* block) {
    Tensor* hidden = tensor_matmul(x, block->w1);
    Tensor* activated = tensor_relu(hidden);
    Tensor* output = tensor_matmul(activated, block->w2);
    
    tensor_free(hidden);
    tensor_free(activated);
    return output;
}

// Transformer Block
Tensor* transformer_block_forward(Tensor* x, TransformerBlock* block) {
    // Self-attention sub-layer
    Tensor* norm1 = rms_norm(x, block->norm1);
    Tensor* attn = multi_head_attention(norm1, block);
    Tensor* res1 = tensor_add(x, attn);
    
    // Feed-forward sub-layer
    Tensor* norm2 = rms_norm(res1, block->norm2);
    Tensor* ff = feed_forward(norm2, block);
    Tensor* output = tensor_add(res1, ff);
    
    // Cleanup
    tensor_free(norm1);
    tensor_free(attn);
    tensor_free(res1);
    tensor_free(norm2);
    tensor_free(ff);
    
    return output;
}

// Initialize transformer block
TransformerBlock* create_transformer_block() {
    TransformerBlock* block = malloc(sizeof(TransformerBlock));
    
    int qkv_dims[] = {EMBED_DIM, EMBED_DIM};
    int ff1_dims[] = {EMBED_DIM, FF_DIM};
    int ff2_dims[] = {FF_DIM, EMBED_DIM};
    int norm_dims[] = {EMBED_DIM};
    
    block->wq = create_random_tensor(2, qkv_dims);
    block->wk = create_random_tensor(2, qkv_dims);
    block->wv = create_random_tensor(2, qkv_dims);
    block->wo = create_random_tensor(2, qkv_dims);
    
    block->w1 = create_random_tensor(2, ff1_dims);
    block->w2 = create_random_tensor(2, ff2_dims);
    
    // Initialize norm weights to ones
    float* ones = malloc(EMBED_DIM * sizeof(float));
    for(int i = 0; i < EMBED_DIM; i++) ones[i] = 1.0f;
    block->norm1 = tensor_new(1, norm_dims, ones, 1);
    block->norm2 = tensor_new(1, norm_dims, ones, 1);
    free(ones);
    
    return block;
}

void free_transformer_block(TransformerBlock* block) {
    tensor_free(block->wq);
    tensor_free(block->wk);
    tensor_free(block->wv);
    tensor_free(block->wo);
    tensor_free(block->w1);
    tensor_free(block->w2);
    tensor_free(block->norm1);
    tensor_free(block->norm2);
    free(block);
}

void test_transformer() {
    printf("Creating transformer block...\n");
    
    // Create input tensor
    int input_dims[] = {BATCH_SIZE, SEQ_LENGTH, EMBED_DIM};
    Tensor* input = create_random_tensor(3, input_dims);
    
    // Create and initialize transformer block
    TransformerBlock* block = create_transformer_block();
    
    // Forward pass
    printf("Running forward pass...\n");
    Tensor* output = transformer_block_forward(input, block);
    
    // Print some statistics
    printf("\nInput shape: %dx%dx%d\n", input->dims[0], input->dims[1], input->dims[2]);
    printf("Output shape: %dx%dx%d\n", output->dims[0], output->dims[1], output->dims[2]);
    
    // Cleanup
    tensor_free(input);
    tensor_free(output);
    free_transformer_block(block);
    cleanup_tape();
    
    printf("Test completed successfully!\n");
}

int main() {
    srand(42);
    test_transformer();
    return 0;
}