#include "grad.h"

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
    printf("Creating weight matrix %dx%d\n", nrows, ncols);
    float* data = malloc(nrows * ncols * sizeof(float));
    for (int i = 0; i < nrows * ncols; i++) {
        data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) / sqrtf(ncols);
    }
    Tensor* w = tensor_new(2, (int[]){nrows, ncols}, data, 1);
    free(data);
    return w;
}

static Tensor* softmax(Tensor* x) {
    printf("Starting softmax\n");
    Tensor* max = tensor_reduce_max(x, (int[]){x->ndims-1}, 1);
    Tensor* x_stable = tensor_add(x, tensor_hadamard(
        tensor_reshape(max, 3, (int[]){BATCH_SIZE, SEQ_LEN, 1}), 
        tensor_new(1, (int[]){1}, (float[]){-1.0f}, 1)));
    Tensor* exp = tensor_exp(x_stable);
    Tensor* sum = tensor_reduce_sum(exp, (int[]){x->ndims-1}, 1);
    Tensor* result = tensor_hadamard(exp, tensor_pow(
        tensor_reshape(sum, 3, (int[]){BATCH_SIZE, SEQ_LEN, 1}), -1.0f));
    printf("Finished softmax\n");
    return result;
}

static Tensor* layer_norm(Tensor* x, Tensor* weight) {
    printf("Starting layer norm\n");
    Tensor* var = tensor_pow(tensor_reduce_sum(tensor_pow(x, 2.0f), 
                           (int[]){x->ndims-1}, 1), 0.5f);
    Tensor* result = tensor_hadamard(x, tensor_hadamard(weight, 
           tensor_pow(tensor_reshape(var, 3, (int[]){BATCH_SIZE, SEQ_LEN, 1}), -1.0f)));
    printf("Finished layer norm\n");
    return result;
}

static Tensor* attention(Tensor* x, Transformer* t) {
    printf("Starting attention\n");
    Tensor* q = tensor_matmul(x, t->wq);
    printf("Computed Q\n");
    Tensor* k = tensor_matmul(x, t->wk);
    printf("Computed K\n");
    Tensor* v = tensor_matmul(x, t->wv);
    printf("Computed V\n");
    Tensor* qk = tensor_matmul(q, k);
    printf("Computed QK\n");
    Tensor* scaled_qk = tensor_pow(qk, 1.0f/sqrtf(HEAD_DIM));
    printf("Computed scaled QK\n");
    Tensor* attn = softmax(scaled_qk);
    printf("Computed attention weights\n");
    Tensor* attn_v = tensor_matmul(attn, v);
    printf("Computed attention output\n");
    Tensor* result = tensor_matmul(attn_v, t->wo);
    printf("Finished attention\n");
    return result;
}

static Tensor* forward(Tensor* x, Transformer* t) {
    printf("Starting forward pass\n");
    Tensor* norm1 = layer_norm(x, t->ln1);
    printf("First layer norm done\n");
    Tensor* attn = attention(norm1, t);
    printf("Attention done\n");
    Tensor* a = tensor_add(x, attn);
    printf("Residual connection 1 done\n");
    Tensor* norm2 = layer_norm(a, t->ln2);
    printf("Second layer norm done\n");
    Tensor* ff1 = tensor_matmul(norm2, t->w1);
    printf("First FFN layer done\n");
    Tensor* ff1_relu = tensor_relu(ff1);
    Tensor* ff2 = tensor_matmul(ff1_relu, t->w2);
    printf("Second FFN layer done\n");
    Tensor* result = tensor_add(a, ff2);
    printf("Finished forward pass\n");
    return result;
}

static Transformer* transformer_new() {
    printf("Creating transformer\n");
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
    printf("Finished creating transformer\n");
    return t;
}

void transformer_free(Transformer* t) {
    printf("Freeing transformer\n");
    tensor_free(t->wq);
    tensor_free(t->wk);
    tensor_free(t->wv);
    tensor_free(t->wo);
    tensor_free(t->w1);
    tensor_free(t->w2);
    tensor_free(t->ln1);
    tensor_free(t->ln2);
    free(t);
    printf("Finished freeing transformer\n");
}

int main() {
    printf("Starting program\n");
    srand(42);
    
    printf("Creating input tensor\n");
    float* data = malloc(BATCH_SIZE * SEQ_LEN * EMBED_DIM * sizeof(float));
    for(int i = 0; i < BATCH_SIZE * SEQ_LEN * EMBED_DIM; i++) {
        data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 0.1f;
    }
    Tensor* x = tensor_new(3, (int[]){BATCH_SIZE, SEQ_LEN, EMBED_DIM}, data, 1);
    free(data);
    
    Transformer* t = transformer_new();
    printf("Running forward pass\n");
    Tensor* out = forward(x, t);
    
    printf("Running backward pass\n");
    backward();
    
    printf("Cleaning up\n");
    tensor_free(x);
    tensor_free(out);
    transformer_free(t);
    cleanup_tape();
    
    printf("Program finished successfully\n");
    return 0;
}