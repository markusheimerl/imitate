#include "grad/grad.h"
#include <math.h>

#define BATCH_SIZE 2
#define SEQ_LENGTH 8
#define EMBED_DIM 32
#define NUM_HEADS 4
#define HEAD_DIM (EMBED_DIM / NUM_HEADS)
#define FF_DIM (4 * EMBED_DIM)

typedef struct {
    Tensor *wq, *wk, *wv, *wo;  // Attention weights
    Tensor *w1, *w2;            // FFN weights
    Tensor *norm1, *norm2;      // Layer norm weights
} TransformerBlock;

static Tensor* create_random_tensor(int ndims, const int* dims) {
    int size = 1;
    for (int i = 0; i < ndims; i++) size *= dims[i];
    
    float* data = malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) / sqrtf(size);
    }
    
    Tensor* t = tensor_new(ndims, dims, data, 1);
    free(data);
    return t;
}

static Tensor* rms_norm(Tensor* x, Tensor* weight) {
    Tensor* mean = tensor_pow(tensor_reduce_sum(tensor_pow(x, 2.0f), 
                            (int[]){x->ndims - 1}, 1), 0.5f);
    
    int reshape_dims[MAX_DIMS];
    for(int i = 0; i < x->ndims - 1; i++) reshape_dims[i] = x->dims[i];
    reshape_dims[x->ndims - 1] = 1;
    
    Tensor* normalized = tensor_hadamard(x, 
                        tensor_pow(tensor_reshape(mean, x->ndims, reshape_dims), -1.0f));
    
    return tensor_hadamard(normalized, weight);
}

static Tensor* attention(Tensor* x, TransformerBlock* block) {
    Tensor* q = tensor_matmul(x, block->wq);
    Tensor* k = tensor_matmul(x, block->wk);
    Tensor* v = tensor_matmul(x, block->wv);
    
    Tensor* scores = tensor_matmul(q, k);
    scores = tensor_pow(scores, 1.0f / sqrtf(HEAD_DIM));
    
    Tensor* attn = tensor_matmul(
        tensor_softmax(scores),
        v
    );
    
    return tensor_matmul(attn, block->wo);
}

static Tensor* ffn(Tensor* x, TransformerBlock* block) {
    return tensor_matmul(
        tensor_relu(tensor_matmul(x, block->w1)),
        block->w2
    );
}

static Tensor* transformer_forward(Tensor* x, TransformerBlock* block) {
    Tensor* a = tensor_add(x, attention(rms_norm(x, block->norm1), block));
    return tensor_add(a, ffn(rms_norm(a, block->norm2), block));
}

static TransformerBlock* create_transformer() {
    TransformerBlock* block = calloc(1, sizeof(TransformerBlock));
    
    int qkv_dims[] = {EMBED_DIM, EMBED_DIM};
    int ff1_dims[] = {EMBED_DIM, FF_DIM};
    int ff2_dims[] = {FF_DIM, EMBED_DIM};
    
    block->wq = create_random_tensor(2, qkv_dims);
    block->wk = create_random_tensor(2, qkv_dims);
    block->wv = create_random_tensor(2, qkv_dims);
    block->wo = create_random_tensor(2, qkv_dims);
    block->w1 = create_random_tensor(2, ff1_dims);
    block->w2 = create_random_tensor(2, ff2_dims);
    
    float ones[EMBED_DIM] = {1.0f};
    block->norm1 = tensor_new(1, (int[]){EMBED_DIM}, ones, 1);
    block->norm2 = tensor_new(1, (int[]){EMBED_DIM}, ones, 1);
    
    return block;
}

static void free_transformer(TransformerBlock* block) {
    tensor_free(block->wq); tensor_free(block->wk);
    tensor_free(block->wv); tensor_free(block->wo);
    tensor_free(block->w1); tensor_free(block->w2);
    tensor_free(block->norm1); tensor_free(block->norm2);
    free(block);
}

int main() {
    srand(42);
    
    int input_dims[] = {BATCH_SIZE, SEQ_LENGTH, EMBED_DIM};
    Tensor* input = create_random_tensor(3, input_dims);
    TransformerBlock* block = create_transformer();
    
    Tensor* output = transformer_forward(input, block);
    backward();
    
    cleanup_tape();
    tensor_free(input);
    tensor_free(output);
    free_transformer(block);
    
    return 0;
}