#include "grad/grad.h"

#define BATCH_SIZE 2
#define SEQ_LEN 8
#define EMBED_DIM 32
#define HEAD_DIM 8
#define NUM_HEADS 4

typedef struct {
    Tensor *wq, *wk, *wv, *wo;  // attention weights
    Tensor *w1, *w2;            // ffn weights
    Tensor *ln1, *ln2;          // layer norms
} Transformer;

static Tensor* create_weight(int nrows, int ncols) {
    float* data = malloc(nrows * ncols * sizeof(float));
    for (int i = 0; i < nrows * ncols; i++) {
        data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) / sqrtf(ncols);
    }
    Tensor* w = tensor_new(2, (int[]){nrows, ncols}, data, 1);
    free(data);
    return w;
}

static Tensor* softmax(Tensor* x) {
    Tensor* max = tensor_reduce_max(x, (int[]){x->ndims-1}, 1);
    Tensor* x_stable = tensor_add(x, tensor_hadamard(
        tensor_reshape(max, 3, (int[]){BATCH_SIZE, SEQ_LEN, 1}), 
        tensor_new(1, (int[]){1}, (float[]){-1.0f}, 1)));
    Tensor* exp = tensor_exp(x_stable);
    Tensor* sum = tensor_reduce_sum(exp, (int[]){x->ndims-1}, 1);
    return tensor_hadamard(exp, tensor_pow(
        tensor_reshape(sum, 3, (int[]){BATCH_SIZE, SEQ_LEN, 1}), -1.0f));
}

static Tensor* layer_norm(Tensor* x, Tensor* weight) {
    Tensor* var = tensor_pow(tensor_reduce_sum(tensor_pow(x, 2.0f), 
                           (int[]){x->ndims-1}, 1), 0.5f);
    return tensor_hadamard(x, tensor_hadamard(weight, 
           tensor_pow(tensor_reshape(var, 3, (int[]){BATCH_SIZE, SEQ_LEN, 1}), -1.0f)));
}

static Tensor* attention(Tensor* x, Transformer* t) {
    Tensor* q = tensor_matmul(x, t->wq);
    Tensor* k = tensor_matmul(x, t->wk);
    Tensor* v = tensor_matmul(x, t->wv);
    return tensor_matmul(tensor_matmul(
        softmax(tensor_pow(tensor_matmul(q, k), 1.0f/sqrtf(HEAD_DIM))), v), t->wo);
}

static Tensor* forward(Tensor* x, Transformer* t) {
    Tensor* a = tensor_add(x, attention(layer_norm(x, t->ln1), t));
    return tensor_add(a, tensor_matmul(
        tensor_relu(tensor_matmul(layer_norm(a, t->ln2), t->w1)), t->w2));
}

static Transformer* transformer_new() {
    Transformer* t = calloc(1, sizeof(Transformer));
    t->wq = create_weight(EMBED_DIM, EMBED_DIM);
    t->wk = create_weight(EMBED_DIM, EMBED_DIM);
    t->wv = create_weight(EMBED_DIM, EMBED_DIM);
    t->wo = create_weight(EMBED_DIM, EMBED_DIM);
    t->w1 = create_weight(EMBED_DIM, 4*EMBED_DIM);
    t->w2 = create_weight(4*EMBED_DIM, EMBED_DIM);
    
    float ones[EMBED_DIM];
    for(int i = 0; i < EMBED_DIM; i++) ones[i] = 1.0f;
    t->ln1 = tensor_new(1, (int[]){EMBED_DIM}, ones, 1);
    t->ln2 = tensor_new(1, (int[]){EMBED_DIM}, ones, 1);
    return t;
}

static void transformer_free(Transformer* t) {
    tensor_free(t->wq); tensor_free(t->wk); 
    tensor_free(t->wv); tensor_free(t->wo);
    tensor_free(t->w1); tensor_free(t->w2);
    tensor_free(t->ln1); tensor_free(t->ln2);
    free(t);
}

int main() {
    srand(42);
    
    float* data = malloc(BATCH_SIZE * SEQ_LEN * EMBED_DIM * sizeof(float));
    for(int i = 0; i < BATCH_SIZE * SEQ_LEN * EMBED_DIM; i++) {
        data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 0.1f;
    }
    Tensor* x = tensor_new(3, (int[]){BATCH_SIZE, SEQ_LEN, EMBED_DIM}, data, 1);
    free(data);
    
    Transformer* t = transformer_new();
    Tensor* out = forward(x, t);
    
    backward();
    
    tensor_free(x);
    tensor_free(out);
    transformer_free(t);
    cleanup_tape();
    
    return 0;
}