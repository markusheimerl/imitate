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

// Forward declarations
void free_transformer_block(TransformerBlock* block);
Tensor* rms_norm(Tensor* x, Tensor* weight);
Tensor* softmax(Tensor* x);
Tensor* multi_head_attention(Tensor* x, TransformerBlock* block);
Tensor* feed_forward(Tensor* x, TransformerBlock* block);
Tensor* transformer_block_forward(Tensor* x, TransformerBlock* block);

Tensor* create_random_tensor(int ndims, const int* dims) {
    int size = 1;
    for (int i = 0; i < ndims; i++) size *= dims[i];
    
    float* data = malloc(size * sizeof(float));
    if (!data) return NULL;
    
    float scale = 1.0f / sqrtf(size);
    for (int i = 0; i < size; i++) {
        data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
    }
    
    Tensor* t = tensor_new(ndims, dims, data, 1);
    free(data);
    return t;
}

Tensor* rms_norm(Tensor* x, Tensor* weight) {
    Tensor* squared = tensor_pow(x, 2.0f);
    if (!squared) return NULL;
    
    int axes[] = {x->ndims - 1};
    Tensor* sum_squared = tensor_reduce_sum(squared, axes, 1);
    if (!sum_squared) return NULL;
    
    // Add epsilon for numerical stability
    for(int i = 0; i < sum_squared->size; i++) {
        sum_squared->data[i] += 1e-5f;
    }
    
    Tensor* mean = tensor_pow(sum_squared, 0.5f);
    if (!mean) return NULL;
    
    int new_dims[MAX_DIMS];
    for(int i = 0; i < x->ndims - 1; i++) new_dims[i] = x->dims[i];
    new_dims[x->ndims - 1] = 1;
    
    Tensor* mean_reshaped = tensor_reshape(mean, x->ndims, new_dims);
    if (!mean_reshaped) return NULL;
    
    Tensor* normalized = tensor_hadamard(x, tensor_pow(mean_reshaped, -1.0f));
    if (!normalized) return NULL;
    
    return tensor_hadamard(normalized, weight);
}

Tensor* softmax(Tensor* x) {
    Tensor* exp_x = tensor_exp(x);
    if (!exp_x) return NULL;
    
    int axes[] = {x->ndims - 1};
    Tensor* sum_exp = tensor_reduce_sum(exp_x, axes, 1);
    if (!sum_exp) return NULL;
    
    int new_dims[MAX_DIMS];
    for(int i = 0; i < x->ndims - 1; i++) new_dims[i] = x->dims[i];
    new_dims[x->ndims - 1] = 1;
    
    Tensor* sum_exp_reshaped = tensor_reshape(sum_exp, x->ndims, new_dims);
    if (!sum_exp_reshaped) return NULL;
    
    return tensor_hadamard(exp_x, tensor_pow(sum_exp_reshaped, -1.0f));
}

Tensor* multi_head_attention(Tensor* x, TransformerBlock* block) {
    Tensor* q = tensor_matmul(x, block->wq);
    if (!q) return NULL;
    
    Tensor* k = tensor_matmul(x, block->wk);
    if (!k) return NULL;
    
    Tensor* v = tensor_matmul(x, block->wv);
    if (!v) return NULL;
    
    Tensor* scores = tensor_matmul(q, k);
    if (!scores) return NULL;
    
    float scale = 1.0f / sqrtf(HEAD_DIM);
    Tensor* scaled_scores = tensor_pow(scores, scale);
    if (!scaled_scores) return NULL;
    
    Tensor* attn_weights = softmax(scaled_scores);
    if (!attn_weights) return NULL;
    
    Tensor* context = tensor_matmul(attn_weights, v);
    if (!context) return NULL;
    
    return tensor_matmul(context, block->wo);
}

Tensor* feed_forward(Tensor* x, TransformerBlock* block) {
    Tensor* hidden = tensor_matmul(x, block->w1);
    if (!hidden) return NULL;
    
    Tensor* activated = tensor_relu(hidden);
    if (!activated) return NULL;
    
    return tensor_matmul(activated, block->w2);
}

Tensor* transformer_block_forward(Tensor* x, TransformerBlock* block) {
    Tensor* norm1 = rms_norm(x, block->norm1);
    if (!norm1) return NULL;
    
    Tensor* attn = multi_head_attention(norm1, block);
    if (!attn) return NULL;
    
    Tensor* res1 = tensor_add(x, attn);
    if (!res1) return NULL;
    
    Tensor* norm2 = rms_norm(res1, block->norm2);
    if (!norm2) return NULL;
    
    Tensor* ff = feed_forward(norm2, block);
    if (!ff) return NULL;
    
    return tensor_add(res1, ff);
}

TransformerBlock* create_transformer_block() {
    TransformerBlock* block = calloc(1, sizeof(TransformerBlock));
    if (!block) return NULL;
    
    int qkv_dims[] = {EMBED_DIM, EMBED_DIM};
    int ff1_dims[] = {EMBED_DIM, FF_DIM};
    int ff2_dims[] = {FF_DIM, EMBED_DIM};
    int norm_dims[] = {EMBED_DIM};
    
    if (!(block->wq = create_random_tensor(2, qkv_dims)) ||
        !(block->wk = create_random_tensor(2, qkv_dims)) ||
        !(block->wv = create_random_tensor(2, qkv_dims)) ||
        !(block->wo = create_random_tensor(2, qkv_dims)) ||
        !(block->w1 = create_random_tensor(2, ff1_dims)) ||
        !(block->w2 = create_random_tensor(2, ff2_dims))) {
        free_transformer_block(block);
        return NULL;
    }
    
    float* ones = calloc(EMBED_DIM, sizeof(float));
    if (!ones) {
        free_transformer_block(block);
        return NULL;
    }
    
    for(int i = 0; i < EMBED_DIM; i++) ones[i] = 1.0f;
    
    block->norm1 = tensor_new(1, norm_dims, ones, 1);
    block->norm2 = tensor_new(1, norm_dims, ones, 1);
    
    free(ones);
    
    if (!block->norm1 || !block->norm2) {
        free_transformer_block(block);
        return NULL;
    }
    
    return block;
}

void free_transformer_block(TransformerBlock* block) {
    if (!block) return;
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
    
    int input_dims[] = {BATCH_SIZE, SEQ_LENGTH, EMBED_DIM};
    Tensor* input = create_random_tensor(3, input_dims);
    if (!input) {
        printf("Failed to create input tensor\n");
        return;
    }
    
    TransformerBlock* block = create_transformer_block();
    if (!block) {
        printf("Failed to create transformer block\n");
        tensor_free(input);
        return;
    }
    
    printf("Running forward pass...\n");
    Tensor* output = transformer_block_forward(input, block);
    if (!output) {
        printf("Forward pass failed\n");
        tensor_free(input);
        free_transformer_block(block);
        return;
    }
    
    printf("\nInput shape: %dx%dx%d\n", input->dims[0], input->dims[1], input->dims[2]);
    printf("Output shape: %dx%dx%d\n", output->dims[0], output->dims[1], output->dims[2]);
    
    // Run backward pass
    backward();
    
    // Cleanup everything
    cleanup_tape();
    tensor_free(input);
    tensor_free(output);
    free_transformer_block(block);
    
    printf("Test completed successfully!\n");
}

int main() {
    srand(42);
    test_transformer();
    return 0;
}