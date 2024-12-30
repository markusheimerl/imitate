
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_DIMS 8
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

typedef enum { 
    ADD, MATMUL, RELU, SIGMOID, RESHAPE, SLICE, PERMUTE, 
    GATHER, HADAMARD, POW, EXP, REDUCE_SUM, REDUCE_MAX
} OpType;

typedef struct {
    float *data, *grad;
    int *dims, ndims, size;
    int requires_grad;
} Tensor;

typedef struct {
    OpType op;
    Tensor *result, *input1, *input2;
    int *aux_data1, *aux_data2;
    int aux_int;
} TapeEntry;

static struct { TapeEntry entries[1000]; int len; } tape;

// Core math helpers
static float sigmoid(float x) { return 1.0f / (1.0f + expf(-fmaxf(fminf(x, 88.0f), -88.0f))); }
static float d_sigmoid(float x) { float s = sigmoid(x); return s * (1 - s); }
static float relu(float x) { return x > 0 ? x : 0; }
static float d_relu(float x) { return x > 0 ? 1 : 0; }

// Index conversion utilities
static int coords_to_index(const int* coords, const int* dims, int ndims) {
    int index = 0, stride = 1;
    for (int i = ndims - 1; i >= 0; i--) {
        index += coords[i] * stride;
        stride *= dims[i];
    }
    return index;
}

static void index_to_coords(int index, int* coords, const int* dims, int ndims) {
    for (int i = ndims - 1; i >= 0; i--) {
        coords[i] = index % dims[i];
        index /= dims[i];
    }
}

Tensor* tensor_new(int ndims, const int* dims, const float* data, int requires_grad) {
    if (!dims) {
        printf("Error: NULL dims array\n");
        return NULL;
    }
    
    Tensor* t = calloc(1, sizeof(Tensor));
    if (!t) {
        printf("Error: Failed to allocate tensor\n");
        return NULL;
    }
    
    t->ndims = ndims;
    t->dims = malloc(ndims * sizeof(int));
    if (!t->dims) {
        printf("Error: Failed to allocate dims array\n");
        free(t);
        return NULL;
    }
    memcpy(t->dims, dims, ndims * sizeof(int));
    
    t->size = 1;
    for (int i = 0; i < ndims; i++) t->size *= dims[i];
    
    t->data = malloc(t->size * sizeof(float));
    if (!t->data) {
        printf("Error: Failed to allocate data array\n");
        free(t->dims);
        free(t);
        return NULL;
    }
    
    if (data) memcpy(t->data, data, t->size * sizeof(float));
    else memset(t->data, 0, t->size * sizeof(float));
    
    if ((t->requires_grad = requires_grad)) {
        t->grad = calloc(t->size, sizeof(float));
        if (!t->grad) {
            printf("Error: Failed to allocate grad array\n");
            free(t->data);
            free(t->dims);
            free(t);
            return NULL;
        }
    }
    
    return t;
}

static void record_operation(OpType op, Tensor* result, Tensor* input1, Tensor* input2, 
                           int* aux1, int* aux2, int aux_int) {
    if (!result || !result->requires_grad) return;
    if (tape.len >= 1000) {
        printf("Error: Tape overflow\n");
        return;
    }
    
    printf("Recording operation %d, tape length: %d\n", op, tape.len);
    if (!input1) {
        printf("Error: NULL input1 tensor in operation %d\n", op);
        return;
    }
    
    tape.entries[tape.len++] = (TapeEntry){
        .op = op,
        .result = result,
        .input1 = input1,
        .input2 = input2,
        .aux_data1 = aux1,
        .aux_data2 = aux2,
        .aux_int = aux_int
    };
    printf("Successfully recorded operation\n");
}

Tensor* tensor_reduce_max(Tensor* t, const int* axes, int num_axes) {
    int reduce_dims[MAX_DIMS] = {0}, new_dims[MAX_DIMS];
    int new_ndims = 0;
    
    for (int i = 0; i < num_axes; i++) reduce_dims[axes[i]] = 1;
    for (int i = 0; i < t->ndims; i++) 
        if (!reduce_dims[i]) new_dims[new_ndims++] = t->dims[i];
    
    Tensor* result = tensor_new(new_ndims, new_dims, NULL, t->requires_grad);
    for (int i = 0; i < result->size; i++) result->data[i] = -INFINITY;
    
    int* max_indices = t->requires_grad ? malloc(result->size * sizeof(int)) : NULL;
    
    int coords[MAX_DIMS], result_coords[MAX_DIMS];
    for (int i = 0; i < t->size; i++) {
        index_to_coords(i, coords, t->dims, t->ndims);
        int idx = 0;
        for (int j = 0; j < t->ndims; j++)
            if (!reduce_dims[j]) result_coords[idx++] = coords[j];
        
        int result_idx = coords_to_index(result_coords, result->dims, new_ndims);
        if (t->data[i] > result->data[result_idx]) {
            result->data[result_idx] = t->data[i];
            if (max_indices) max_indices[result_idx] = i;
        }
    }
    
    if (result->requires_grad)
        record_operation(REDUCE_MAX, result, t, NULL, max_indices, NULL, num_axes);
    return result;
}

Tensor* tensor_reduce_sum(Tensor* t, const int* axes, int num_axes) {
    int reduce_dims[MAX_DIMS] = {0}, new_dims[MAX_DIMS];
    int new_ndims = 0;
    
    for (int i = 0; i < num_axes; i++) reduce_dims[axes[i]] = 1;
    for (int i = 0; i < t->ndims; i++)
        if (!reduce_dims[i]) new_dims[new_ndims++] = t->dims[i];
    
    Tensor* result = tensor_new(new_ndims, new_dims, NULL, t->requires_grad);
    
    int coords[MAX_DIMS], result_coords[MAX_DIMS];
    for (int i = 0; i < t->size; i++) {
        index_to_coords(i, coords, t->dims, t->ndims);
        int idx = 0;
        for (int j = 0; j < t->ndims; j++)
            if (!reduce_dims[j]) result_coords[idx++] = coords[j];
        result->data[coords_to_index(result_coords, result->dims, new_ndims)] += t->data[i];
    }
    
    if (result->requires_grad) {
        int* axes_copy = malloc(num_axes * sizeof(int));
        memcpy(axes_copy, axes, num_axes * sizeof(int));
        record_operation(REDUCE_SUM, result, t, NULL, axes_copy, NULL, num_axes);
    }
    return result;
}

Tensor* tensor_gather(Tensor* t, int axis, const int* indices, int num_indices) {
    int new_dims[MAX_DIMS];
    memcpy(new_dims, t->dims, t->ndims * sizeof(int));
    new_dims[axis] = num_indices;
    
    Tensor* result = tensor_new(t->ndims, new_dims, NULL, t->requires_grad);
    
    int coords[MAX_DIMS];
    for (int i = 0; i < result->size; i++) {
        index_to_coords(i, coords, result->dims, result->ndims);
        int original_coord = coords[axis];
        coords[axis] = indices[original_coord];
        result->data[i] = t->data[coords_to_index(coords, t->dims, t->ndims)];
    }
    
    if (result->requires_grad) {
        int* indices_copy = malloc(num_indices * sizeof(int));
        memcpy(indices_copy, indices, num_indices * sizeof(int));
        record_operation(GATHER, result, t, NULL, indices_copy, NULL, axis);
    }
    return result;
}

Tensor* tensor_slice(Tensor* t, const int* start, const int* end) {
    int new_dims[MAX_DIMS];
    for (int i = 0; i < t->ndims; i++) new_dims[i] = end[i] - start[i];
    
    Tensor* result = tensor_new(t->ndims, new_dims, NULL, t->requires_grad);
    
    int coords[MAX_DIMS], src_coords[MAX_DIMS];
    for (int i = 0; i < result->size; i++) {
        index_to_coords(i, coords, result->dims, result->ndims);
        for (int j = 0; j < t->ndims; j++) src_coords[j] = coords[j] + start[j];
        result->data[i] = t->data[coords_to_index(src_coords, t->dims, t->ndims)];
    }
    
    if (result->requires_grad) {
        int *start_copy = malloc(t->ndims * sizeof(int));
        int *end_copy = malloc(t->ndims * sizeof(int));
        memcpy(start_copy, start, t->ndims * sizeof(int));
        memcpy(end_copy, end, t->ndims * sizeof(int));
        record_operation(SLICE, result, t, NULL, start_copy, end_copy, 0);
    }
    return result;
}

Tensor* tensor_reshape(Tensor* t, int new_ndims, const int* new_dims) {
    int new_size = 1;
    for (int i = 0; i < new_ndims; i++) new_size *= new_dims[i];
    if (new_size != t->size) return NULL;
    
    Tensor* result = tensor_new(new_ndims, new_dims, t->data, t->requires_grad);
    if (result->requires_grad)
        record_operation(RESHAPE, result, t, NULL, NULL, NULL, 0);
    return result;
}

Tensor* tensor_permute(Tensor* t, const int* perm) {
    int new_dims[MAX_DIMS];
    for (int i = 0; i < t->ndims; i++) new_dims[i] = t->dims[perm[i]];
    
    Tensor* result = tensor_new(t->ndims, new_dims, NULL, t->requires_grad);
    
    int coords[MAX_DIMS], new_coords[MAX_DIMS];
    for (int i = 0; i < t->size; i++) {
        index_to_coords(i, new_coords, result->dims, result->ndims);
        for (int j = 0; j < t->ndims; j++) coords[perm[j]] = new_coords[j];
        result->data[i] = t->data[coords_to_index(coords, t->dims, t->ndims)];
    }
    
    if (result->requires_grad) {
        int* perm_copy = malloc(t->ndims * sizeof(int));
        memcpy(perm_copy, perm, t->ndims * sizeof(int));
        record_operation(PERMUTE, result, t, NULL, perm_copy, NULL, 0);
    }
    return result;
}

Tensor* tensor_pow(Tensor* t, float exponent) {
    Tensor* result = tensor_new(t->ndims, t->dims, NULL, t->requires_grad);
    
    for (int i = 0; i < t->size; i++)
        result->data[i] = powf(t->data[i], exponent);
    
    if (result->requires_grad) {
        float* exp_ptr = malloc(sizeof(float));
        *exp_ptr = exponent;
        record_operation(POW, result, t, NULL, (int*)exp_ptr, NULL, 0);
    }
    return result;
}

Tensor* tensor_op(Tensor* a, Tensor* b, OpType op) {
    if (!a) {
        printf("Error: First input tensor is NULL\n");
        return NULL;
    }
    
    if ((op == ADD || op == MATMUL || op == HADAMARD) && !b) {
        printf("Error: Second input tensor is NULL for binary operation\n");
        return NULL;
    }

    if (op == RELU || op == SIGMOID || op == EXP) {
        printf("Processing unary operation %d\n", op);
        Tensor* result = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
        if (!result) return NULL;
        
        for (int i = 0; i < result->size; i++)
            result->data[i] = op == RELU ? relu(a->data[i]) : 
                             op == SIGMOID ? sigmoid(a->data[i]) : 
                             expf(fmaxf(fminf(a->data[i], 88.0f), -88.0f));
        
        if (result->requires_grad) {
            printf("Recording unary operation %d\n", op);
            record_operation(op, result, a, NULL, NULL, NULL, 0);
        }
        return result;
    }

    int out_dims[MAX_DIMS], out_ndims = op == MATMUL ? MAX(a->ndims, b->ndims) : a->ndims;
    memcpy(out_dims, a->dims, a->ndims * sizeof(int));
    if (op == MATMUL) out_dims[out_ndims-1] = b->dims[b->ndims-1];

    Tensor* result = tensor_new(out_ndims, out_dims, NULL, a->requires_grad || b->requires_grad);
    if (!result) return NULL;

    if (op == MATMUL) {
        printf("Processing MATMUL\n");
        int m = a->dims[a->ndims-2], n = a->dims[a->ndims-1], p = b->dims[b->ndims-1];
        int batch_size = result->size / (m * p);
        
        for (int batch = 0; batch < batch_size; batch++) {
            float *out = result->data + batch * m * p;
            const float *a_data = a->data + batch * m * n;
            const float *b_data = b->data + batch * n * p;
            
            for (int i = 0; i < m; i++)
                for (int k = 0; k < n; k++) {
                    float aik = a_data[i * n + k];
                    for (int j = 0; j < p; j++)
                        out[i * p + j] += aik * b_data[k * p + j];
                }
        }
    } else {
        printf("Processing binary operation %d\n", op);
        for (int i = 0; i < result->size; i++)
            result->data[i] = op == ADD ? a->data[i] + b->data[i] : a->data[i] * b->data[i];
    }

    if (result->requires_grad) {
        printf("Recording binary operation %d\n", op);
        record_operation(op, result, a, b, NULL, NULL, 0);
    }
    return result;
}

void backward() {
    printf("Starting backward pass\n");
    if (tape.len == 0) {
        printf("Error: Empty tape\n");
        return;
    }
    
    TapeEntry* e = &tape.entries[tape.len - 1];
    if (!e || !e->result) {
        printf("Error: Invalid final tape entry\n");
        return;
    }
    
    Tensor* final = e->result;
    if (!final->grad) {
        printf("Allocating gradient for final tensor\n");
        final->grad = calloc(final->size, sizeof(float));
        if (!final->grad) {
            printf("Error: Failed to allocate final gradient\n");
            return;
        }
        for (int i = 0; i < final->size; i++) final->grad[i] = 1.0f;
    }
    
    printf("Processing %d tape entries\n", tape.len);
    for (int i = tape.len - 1; i >= 0; i--) {
        e = &tape.entries[i];
        printf("Processing tape entry %d, op type: %d\n", i, e->op);
        
        if (!e->result || !e->input1) {
            printf("Error: Invalid tape entry at position %d\n", i);
            continue;
        }
        
        Tensor *t = e->result, *a = e->input1, *b = e->input2;
        
        if (a->requires_grad && !a->grad) {
            printf("Allocating gradient for input1 tensor\n");
            a->grad = calloc(a->size, sizeof(float));
            if (!a->grad) {
                printf("Error: Failed to allocate input1 gradient\n");
                return;
            }
        }
        if (b && b->requires_grad && !b->grad) {
            printf("Allocating gradient for input2 tensor\n");
            b->grad = calloc(b->size, sizeof(float));
            if (!b->grad) {
                printf("Error: Failed to allocate input2 gradient\n");
                return;
            }
        }
        
        switch (e->op) {
            case REDUCE_MAX:
                printf("Processing REDUCE_MAX\n");
                if (a->requires_grad && e->aux_data1) {
                    for (int j = 0; j < t->size; j++) {
                        if (e->aux_data1[j] >= 0 && e->aux_data1[j] < a->size) {
                            a->grad[e->aux_data1[j]] += t->grad[j];
                        }
                    }
                }
                break;
                
            case REDUCE_SUM:
                printf("Processing REDUCE_SUM\n");
                if (a->requires_grad && e->aux_data1) {
                    int coords[MAX_DIMS], result_coords[MAX_DIMS];
                    for (int j = 0; j < t->size; j++) {
                        index_to_coords(j, result_coords, t->dims, t->ndims);
                        int idx = 0;
                        for (int k = 0; k < a->ndims; k++) {
                            int is_reduced = 0;
                            for (int m = 0; m < e->aux_int; m++) {
                                if (k == e->aux_data1[m]) {
                                    is_reduced = 1;
                                    break;
                                }
                            }
                            coords[k] = is_reduced ? 0 : result_coords[idx++];
                        }
                        do {
                            int input_idx = coords_to_index(coords, a->dims, a->ndims);
                            if (input_idx >= 0 && input_idx < a->size) {
                                a->grad[input_idx] += t->grad[j];
                            }
                            
                            int done = 1;
                            for (int k = 0; k < e->aux_int; k++) {
                                int dim = e->aux_data1[k];
                                if (++coords[dim] < a->dims[dim]) {
                                    done = 0;
                                    break;
                                }
                                coords[dim] = 0;
                            }
                            if (done) break;
                        } while (1);
                    }
                }
                break;

            case RESHAPE:
            case ADD:
                printf("Processing %s\n", e->op == RESHAPE ? "RESHAPE" : "ADD");
                if (a->requires_grad) {
                    for (int j = 0; j < MIN(t->size, a->size); j++) {
                        a->grad[j] += t->grad[j];
                    }
                }
                if (b && b->requires_grad) {
                    for (int j = 0; j < MIN(t->size, b->size); j++) {
                        b->grad[j] += t->grad[j];
                    }
                }
                break;

            case SLICE:
                printf("Processing SLICE\n");
                if (a->requires_grad && e->aux_data1 && e->aux_data2) {
                    int coords[MAX_DIMS], src_coords[MAX_DIMS];
                    for (int j = 0; j < t->size; j++) {
                        index_to_coords(j, coords, t->dims, t->ndims);
                        for (int k = 0; k < a->ndims; k++) {
                            src_coords[k] = coords[k] + e->aux_data1[k];
                        }
                        int idx = coords_to_index(src_coords, a->dims, a->ndims);
                        if (idx >= 0 && idx < a->size) {
                            a->grad[idx] += t->grad[j];
                        }
                    }
                }
                break;

            case PERMUTE:
                printf("Processing PERMUTE\n");
                if (a->requires_grad && e->aux_data1) {
                    int inverse_perm[MAX_DIMS];
                    for (int j = 0; j < t->ndims; j++) {
                        inverse_perm[e->aux_data1[j]] = j;
                    }
                    
                    int old_coords[MAX_DIMS], new_coords[MAX_DIMS];
                    for (int j = 0; j < t->size; j++) {
                        index_to_coords(j, old_coords, t->dims, t->ndims);
                        for (int k = 0; k < t->ndims; k++) {
                            new_coords[inverse_perm[k]] = old_coords[k];
                        }
                        int idx = coords_to_index(new_coords, a->dims, a->ndims);
                        if (idx >= 0 && idx < a->size) {
                            a->grad[idx] += t->grad[j];
                        }
                    }
                }
                break;

            case MATMUL:
                printf("Processing MATMUL\n");
                if (a->ndims < 2 || b->ndims < 2) {
                    printf("Error: Invalid dimensions for MATMUL\n");
                    break;
                }
                {
                    int m = a->dims[a->ndims-2], n = a->dims[a->ndims-1], p = b->dims[b->ndims-1];
                    int batch_size = t->size / (m * p);
                    
                    for (int batch = 0; batch < batch_size; batch++) {
                        float *t_grad = t->grad + batch * m * p;
                        float *a_data = a->data + batch * m * n;
                        float *b_data = b->data + batch * n * p;
                        
                        if (a->requires_grad) {
                            float *a_grad = a->grad + batch * m * n;
                            for (int i = 0; i < m; i++)
                                for (int k = 0; k < n; k++)
                                    for (int j = 0; j < p; j++)
                                        a_grad[i * n + k] += t_grad[i * p + j] * b_data[k * p + j];
                        }
                        if (b->requires_grad) {
                            float *b_grad = b->grad + batch * n * p;
                            for (int k = 0; k < n; k++)
                                for (int j = 0; j < p; j++)
                                    for (int i = 0; i < m; i++)
                                        b_grad[k * p + j] += t_grad[i * p + j] * a_data[i * n + k];
                        }
                    }
                }
                break;

            case RELU:
                printf("Processing RELU\n");
                if (a->requires_grad) {
                    for (int j = 0; j < t->size; j++) {
                        a->grad[j] += t->grad[j] * d_relu(a->data[j]);
                    }
                }
                break;

            case SIGMOID:
                printf("Processing SIGMOID\n");
                if (a->requires_grad) {
                    for (int j = 0; j < t->size; j++) {
                        a->grad[j] += t->grad[j] * d_sigmoid(a->data[j]);
                    }
                }
                break;

            case EXP:
                printf("Processing EXP\n");
                if (a->requires_grad) {
                    for (int j = 0; j < t->size; j++) {
                        a->grad[j] += t->grad[j] * t->data[j];
                    }
                }
                break;

            case HADAMARD:
                printf("Processing HADAMARD\n");
                if (a->requires_grad && b) {
                    for (int j = 0; j < t->size; j++) {
                        a->grad[j] += t->grad[j] * b->data[j];
                    }
                }
                if (b && b->requires_grad) {
                    for (int j = 0; j < t->size; j++) {
                        b->grad[j] += t->grad[j] * a->data[j];
                    }
                }
                break;

            case GATHER:
                printf("Processing GATHER\n");
                if (a->requires_grad && e->aux_data1) {
                    int coords[MAX_DIMS];
                    for (int j = 0; j < t->size; j++) {
                        index_to_coords(j, coords, t->dims, t->ndims);
                        int original_coord = coords[e->aux_int];
                        coords[e->aux_int] = e->aux_data1[original_coord];
                        int idx = coords_to_index(coords, a->dims, a->ndims);
                        if (idx >= 0 && idx < a->size) {
                            a->grad[idx] += t->grad[j];
                        }
                    }
                }
                break;

            case POW:
                printf("Processing POW\n");
                if (a->requires_grad && e->aux_data1) {
                    float exponent = *(float*)e->aux_data1;
                    for (int j = 0; j < t->size; j++) {
                        if (a->data[j] != 0.0f || exponent > 0) {
                            a->grad[j] += t->grad[j] * exponent * 
                                        powf(a->data[j], exponent - 1);
                        }
                    }
                }
                break;

            default:
                printf("Error: Unknown operation type: %d\n", e->op);
                break;
        }
        printf("Finished processing tape entry %d\n", i);
    }
    printf("Finished backward pass\n");
}

void cleanup_tape() {
    for (int i = 0; i < tape.len; i++) {
        free(tape.entries[i].aux_data1);
        free(tape.entries[i].aux_data2);
    }
    tape.len = 0;
}

void tensor_free(Tensor* t) {
    if (!t) return;
    free(t->data);
    free(t->grad);
    free(t->dims);
    free(t);
}

#define tensor_add(a, b) tensor_op(a, b, ADD)
#define tensor_matmul(a, b) tensor_op(a, b, MATMUL)
#define tensor_relu(a) tensor_op(a, NULL, RELU)
#define tensor_sigmoid(a) tensor_op(a, NULL, SIGMOID)
#define tensor_hadamard(a, b) tensor_op(a, b, HADAMARD)
#define tensor_exp(a) tensor_op(a, NULL, EXP)

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