#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Error handling
typedef enum {
    ERR_NONE,
    ERR_NULL_TENSOR,
    ERR_SHAPE_MISMATCH,
    ERR_OUT_OF_MEMORY,
    ERR_INVALID_DIMS,
    ERR_INVALID_VALUE,
    ERR_TAPE_FULL,
    ERR_FILE_IO
} ErrorType;

typedef struct {
    ErrorType type;
    const char* func;
    const char* caller;
    int line;
    char msg[256];
} Error;

static __thread Error last_error;

#define RETURN_ERR(err, msg_str) do { \
    last_error = (Error){ \
        .type = err, \
        .func = __func__, \
        .caller = __func__, \
        .line = __LINE__, \
    }; \
    snprintf(last_error.msg, sizeof(last_error.msg), "%s", msg_str); \
    return NULL; \
} while(0)

#define CHECK_TENSOR(t) do { \
    if (!t) { \
        last_error.caller = __func__; \
        return NULL; \
    } \
} while(0)

// Core tensor structure
typedef struct {
    float* data;
    float* grad;
    int* dims;
    int ndims;
    size_t size;
    size_t ref_count;
} Tensor;

// High-level transformer operations
typedef enum {
    OP_ATTENTION_BLOCK,    // Full attention block including norm
    OP_FEEDFORWARD_BLOCK,  // Full FF block including norm
    OP_EMBED_FEATURES,     // Embedding of sequence and conditional features
    OP_PROJECT_OUT        // Final projection to output space
} OpType;

typedef struct {
    OpType op;
    Tensor *out;
    Tensor **inputs;  // Dynamic array of input tensors
    int n_inputs;
    void* aux_data;
} TapeEntry;

#define MAX_TAPE 1000
static TapeEntry tape[MAX_TAPE];
static int tape_len = 0;

// Reference counting
Tensor* tensor_incref(Tensor* t) {
    if (t) t->ref_count++;
    return t;
}

void tensor_decref(Tensor* t) {
    if (!t) return;
    if (--t->ref_count == 0) {
        free(t->data);
        free(t->grad);
        free(t->dims);
        free(t);
    }
}

// Tensor creation helpers
Tensor* tensor_create(int ndims, const int* dims, int requires_grad) {
    if (!dims || ndims <= 0) 
        RETURN_ERR(ERR_INVALID_DIMS, "Invalid dimensions");
    
    size_t size = 1;
    for (int i = 0; i < ndims; i++) {
        if (dims[i] <= 0) 
            RETURN_ERR(ERR_INVALID_DIMS, "Invalid dimension size");
        size *= dims[i];
    }
    
    Tensor* t = calloc(1, sizeof(Tensor));
    if (!t) RETURN_ERR(ERR_OUT_OF_MEMORY, "Failed to allocate tensor");
    
    t->data = calloc(size, sizeof(float));
    t->dims = malloc(ndims * sizeof(int));
    if (!t->data || !t->dims) {
        free(t->data);
        free(t->dims);
        free(t);
        RETURN_ERR(ERR_OUT_OF_MEMORY, "Failed to allocate tensor data/dims");
    }
    
    if (requires_grad) {
        t->grad = calloc(size, sizeof(float));
        if (!t->grad) {
            free(t->data);
            free(t->dims);
            free(t);
            RETURN_ERR(ERR_OUT_OF_MEMORY, "Failed to allocate gradients");
        }
    }
    
    memcpy(t->dims, dims, ndims * sizeof(int));
    t->ndims = ndims;
    t->size = size;
    t->ref_count = 1;
    
    return t;
}

Tensor* tensor_randn(int ndims, const int* dims, int requires_grad) {
    Tensor* t = tensor_create(ndims, dims, requires_grad);
    if (!t) return NULL;
    
    static int seed_set = 0;
    if (!seed_set) {
        srand(time(NULL));
        seed_set = 1;
    }
    
    for (size_t i = 0; i < t->size; i++) {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        t->data[i] = sqrtf(-2.0f * logf(u1 + 1e-7f)) * cosf(2.0f * M_PI * u2);
    }
    
    return t;
}

// Configuration structure
typedef struct {
    int batch_size;
    int seq_len;
    int d_model;
    int n_head;
    int n_layers;
    int d_ff;
    float learning_rate;
    int epochs;
    int condition_features;
    int sequence_features;
} Config;

// Structure to hold all transformer weights
typedef struct {
    // Embedding weights
    Tensor* W_e;    // Sequence feature embedding
    Tensor* W_cond; // Conditional feature embedding
    
    // Layer weights
    Tensor** W_q;  // Query weights for each layer
    Tensor** W_k;  // Key weights for each layer
    Tensor** W_v;  // Value weights for each layer
    Tensor** W_o;  // Output projection weights for each layer
    Tensor** W_ff1; // First feed forward weights for each layer
    Tensor** W_ff2; // Second feed forward weights for each layer
    
    // Output projection
    Tensor* W_out; // Final output projection
} TransformerWeights;

// Auxiliary data structures for backward passes
typedef struct {
    int n_head;
    float scale;
    int batch_size;
    int seq_len;
    int d_model;
    int d_head;
    float* Q;
    float* K;
    float* V;
    float* scores;
} AttentionCache;

typedef struct {
    int batch_size;
    int seq_len;
    int d_model;
    int d_ff;
    float* hidden;
    float* hidden_pre_gelu;
} FeedForwardCache;

// Direct implementation of attention block
Tensor* attention_block(Tensor* x, Tensor* Wq, Tensor* Wk, Tensor* Wv, Tensor* Wo,
                       int n_head, float eps) {
    if (!x || !Wq || !Wk || !Wv || !Wo) 
        RETURN_ERR(ERR_NULL_TENSOR, "Null input tensor");
    
    int batch_size = x->dims[0];
    int seq_len = x->dims[1];
    int d_model = x->dims[2];
    int d_head = d_model / n_head;
    
    if (d_model % n_head != 0)
        RETURN_ERR(ERR_INVALID_VALUE, "d_model must be divisible by n_head");
    
    // Layer normalization
    float* norm_x = malloc(x->size * sizeof(float));
    if (!norm_x) RETURN_ERR(ERR_OUT_OF_MEMORY, "Failed to allocate normalized input");
    
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            float* x_ptr = x->data + (b * seq_len + s) * d_model;
            float* norm_ptr = norm_x + (b * seq_len + s) * d_model;
            
            // Compute RMS
            float ms = 0.0f;
            for (int i = 0; i < d_model; i++) {
                ms += x_ptr[i] * x_ptr[i];
            }
            ms = sqrtf(ms / d_model + eps);
            
            // Normalize
            for (int i = 0; i < d_model; i++) {
                norm_ptr[i] = x_ptr[i] / ms;
            }
        }
    }
    
    // Allocate temporary buffers for attention computation
    float* Q = malloc(batch_size * n_head * seq_len * d_head * sizeof(float));
    float* K = malloc(batch_size * n_head * seq_len * d_head * sizeof(float));
    float* V = malloc(batch_size * n_head * seq_len * d_head * sizeof(float));
    float* scores = malloc(batch_size * n_head * seq_len * seq_len * sizeof(float));
    
    if (!Q || !K || !V || !scores) {
        free(norm_x);
        free(Q);
        free(K);
        free(V);
        free(scores);
        RETURN_ERR(ERR_OUT_OF_MEMORY, "Failed to allocate attention buffers");
    }

        float scale = 1.0f / sqrtf(d_head);
    
    // Compute Q, K, V for all heads in parallel
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < n_head; h++) {
            // Project to Q, K, V
            for (int i = 0; i < seq_len; i++) {
                float* x_row = norm_x + (b * seq_len + i) * d_model;
                float* q_row = Q + ((b * n_head + h) * seq_len + i) * d_head;
                float* k_row = K + ((b * n_head + h) * seq_len + i) * d_head;
                float* v_row = V + ((b * n_head + h) * seq_len + i) * d_head;
                
                for (int j = 0; j < d_head; j++) {
                    float q_sum = 0, k_sum = 0, v_sum = 0;
                    int head_offset = h * d_head;
                    for (int d = 0; d < d_model; d++) {
                        q_sum += x_row[d] * Wq->data[d * d_model + head_offset + j];
                        k_sum += x_row[d] * Wk->data[d * d_model + head_offset + j];
                        v_sum += x_row[d] * Wv->data[d * d_model + head_offset + j];
                    }
                    q_row[j] = q_sum * scale;
                    k_row[j] = k_sum;
                    v_row[j] = v_sum;
                }
            }
            
            // Compute attention scores and apply causal mask
            for (int i = 0; i < seq_len; i++) {
                float* score_row = scores + ((b * n_head + h) * seq_len + i) * seq_len;
                float max_val = -INFINITY;
                
                for (int j = 0; j <= i; j++) {  // Causal masking: only attend to past
                    float sum = 0;
                    for (int d = 0; d < d_head; d++) {
                        sum += Q[((b * n_head + h) * seq_len + i) * d_head + d] *
                              K[((b * n_head + h) * seq_len + j) * d_head + d];
                    }
                    score_row[j] = sum;
                    max_val = fmaxf(max_val, sum);
                }
                for (int j = i + 1; j < seq_len; j++) {
                    score_row[j] = -INFINITY;  // Mask future positions
                }
                
                // Apply softmax
                float sum = 0;
                for (int j = 0; j <= i; j++) {
                    score_row[j] = expf(score_row[j] - max_val);
                    sum += score_row[j];
                }
                for (int j = 0; j <= i; j++) {
                    score_row[j] /= sum;
                }
            }
        }
    }
    
    // Compute attention output and project back to d_model dimension
    int out_dims[] = {batch_size, seq_len, d_model};
    Tensor* out = tensor_create(3, out_dims, x->grad != NULL);
    if (!out) {
        free(norm_x);
        free(Q);
        free(K);
        free(V);
        free(scores);
        return NULL;
    }
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            float* tmp = malloc(d_model * sizeof(float));
            
            // Gather from all heads
            for (int h = 0; h < n_head; h++) {
                for (int d = 0; d < d_head; d++) {
                    float sum = 0;
                    for (int j = 0; j < seq_len; j++) {
                        sum += scores[((b * n_head + h) * seq_len + s) * seq_len + j] *
                              V[((b * n_head + h) * seq_len + j) * d_head + d];
                    }
                    tmp[h * d_head + d] = sum;
                }
            }
            
            // Project back to d_model dimension
            float* out_row = out->data + (b * seq_len + s) * d_model;
            for (int i = 0; i < d_model; i++) {
                float sum = 0;
                for (int j = 0; j < d_model; j++) {
                    sum += tmp[j] * Wo->data[j * d_model + i];
                }
                out_row[i] = sum + x->data[(b * seq_len + s) * d_model + i];  // Residual connection
            }
            
            free(tmp);
        }
    }
    
    // Store cache for backward pass if needed
    if (out->grad) {
        AttentionCache* cache = malloc(sizeof(AttentionCache));
        cache->n_head = n_head;
        cache->scale = scale;
        cache->batch_size = batch_size;
        cache->seq_len = seq_len;
        cache->d_model = d_model;
        cache->d_head = d_head;
        cache->Q = Q;
        cache->K = K;
        cache->V = V;
        cache->scores = scores;
        
        // Store normalized input for backward pass
        Tensor** inputs = malloc(5 * sizeof(Tensor*));
        inputs[0] = tensor_incref(x);
        inputs[1] = tensor_incref(Wq);
        inputs[2] = tensor_incref(Wk);
        inputs[3] = tensor_incref(Wv);
        inputs[4] = tensor_incref(Wo);
        
        tape[tape_len++] = (TapeEntry){
            .op = OP_ATTENTION_BLOCK,
            .out = out,
            .inputs = inputs,
            .n_inputs = 5,
            .aux_data = cache
        };
    } else {
        free(Q);
        free(K);
        free(V);
        free(scores);
    }
    
    free(norm_x);
    return out;
}

// Direct implementation of feed-forward block
Tensor* feedforward_block(Tensor* x, Tensor* W1, Tensor* W2, float eps) {
    if (!x || !W1 || !W2) 
        RETURN_ERR(ERR_NULL_TENSOR, "Null input tensor");
    
    int batch_size = x->dims[0];
    int seq_len = x->dims[1];
    int d_model = x->dims[2];
    int d_ff = W1->dims[1];
    
    // Layer normalization
    float* norm_x = malloc(x->size * sizeof(float));
    if (!norm_x) RETURN_ERR(ERR_OUT_OF_MEMORY, "Failed to allocate normalized input");
    
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            float* x_ptr = x->data + (b * seq_len + s) * d_model;
            float* norm_ptr = norm_x + (b * seq_len + s) * d_model;
            
            float ms = 0.0f;
            for (int i = 0; i < d_model; i++) {
                ms += x_ptr[i] * x_ptr[i];
            }
            ms = sqrtf(ms / d_model + eps);
            
            for (int i = 0; i < d_model; i++) {
                norm_ptr[i] = x_ptr[i] / ms;
            }
        }
    }
    
    int out_dims[] = {batch_size, seq_len, d_model};
    Tensor* out = tensor_create(3, out_dims, x->grad != NULL);
    if (!out) {
        free(norm_x);
        return NULL;
    }
    
    float* hidden = NULL;
    float* hidden_pre_gelu = NULL;
    
    if (out->grad) {
        hidden = malloc(batch_size * seq_len * d_ff * sizeof(float));
        hidden_pre_gelu = malloc(batch_size * seq_len * d_ff * sizeof(float));
        if (!hidden || !hidden_pre_gelu) {
            free(norm_x);
            free(hidden);
            free(hidden_pre_gelu);
            tensor_decref(out);
            RETURN_ERR(ERR_OUT_OF_MEMORY, "Failed to allocate FF buffers");
        }
    }
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            float* x_row = norm_x + (b * seq_len + s) * d_model;
            float* out_row = out->data + (b * seq_len + s) * d_model;
            
            // First matrix multiplication and GELU
            float* temp_hidden = malloc(d_ff * sizeof(float));
            for (int i = 0; i < d_ff; i++) {
                float sum = 0;
                for (int j = 0; j < d_model; j++) {
                    sum += x_row[j] * W1->data[j * d_ff + i];
                }
                
                if (out->grad) {
                    hidden_pre_gelu[(b * seq_len + s) * d_ff + i] = sum;
                }
                
                // GELU activation
                float x = sum;
                float cdf = 0.5f * (1.0f + tanhf(sqrtf(2.0f/M_PI) * (x + 0.044715f * x * x * x)));
                temp_hidden[i] = x * cdf;
                
                if (out->grad) {
                    hidden[(b * seq_len + s) * d_ff + i] = temp_hidden[i];
                }
            }
            
            // Second matrix multiplication
            for (int i = 0; i < d_model; i++) {
                float sum = 0;
                for (int j = 0; j < d_ff; j++) {
                    sum += temp_hidden[j] * W2->data[j * d_model + i];
                }
                out_row[i] = sum + x->data[(b * seq_len + s) * d_model + i];  // Residual connection
            }
            
            free(temp_hidden);
        }
    }
    
    if (out->grad) {
        FeedForwardCache* cache = malloc(sizeof(FeedForwardCache));
        cache->batch_size = batch_size;
        cache->seq_len = seq_len;
        cache->d_model = d_model;
        cache->d_ff = d_ff;
        cache->hidden = hidden;
        cache->hidden_pre_gelu = hidden_pre_gelu;
        
        Tensor** inputs = malloc(3 * sizeof(Tensor*));
        inputs[0] = tensor_incref(x);
        inputs[1] = tensor_incref(W1);
        inputs[2] = tensor_incref(W2);
        
        tape[tape_len++] = (TapeEntry){
            .op = OP_FEEDFORWARD_BLOCK,
            .out = out,
            .inputs = inputs,
            .n_inputs = 3,
            .aux_data = cache
        };
    }
    
    free(norm_x);
    return out;
}

// Direct implementation of feature embedding
Tensor* embed_features(float* batch_data, Tensor* W_e, Tensor* W_cond, Config* cfg) {
    int dims[] = {cfg->batch_size, cfg->seq_len, cfg->d_model};
    Tensor* out = tensor_create(3, dims, 1);
    if (!out) return NULL;
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < cfg->batch_size; b++) {
        for (int s = 0; s < cfg->seq_len; s++) {
            float* data_ptr = batch_data + (b * cfg->seq_len + s) * 
                             (cfg->condition_features + cfg->sequence_features);
            float* out_ptr = out->data + (b * cfg->seq_len + s) * cfg->d_model;
            
            // Zero initialize output
            memset(out_ptr, 0, cfg->d_model * sizeof(float));
            
            // Embed conditional features
            for (int i = 0; i < cfg->d_model; i++) {
                for (int j = 0; j < cfg->condition_features; j++) {
                    out_ptr[i] += data_ptr[j] * W_cond->data[j * cfg->d_model + i];
                }
            }
            
            // Add sequence feature embeddings
            for (int i = 0; i < cfg->d_model; i++) {
                for (int j = 0; j < cfg->sequence_features; j++) {
                    out_ptr[i] += data_ptr[cfg->condition_features + j] * 
                                 W_e->data[j * cfg->d_model + i];
                }
            }
        }
    }
    
    if (out->grad) {
        Tensor** inputs = malloc(2 * sizeof(Tensor*));
        inputs[0] = tensor_incref(W_e);
        inputs[1] = tensor_incref(W_cond);
        
        float* data_copy = malloc(cfg->batch_size * cfg->seq_len * 
                                (cfg->condition_features + cfg->sequence_features) * 
                                sizeof(float));
        memcpy(data_copy, batch_data, cfg->batch_size * cfg->seq_len * 
               (cfg->condition_features + cfg->sequence_features) * sizeof(float));
        
        tape[tape_len++] = (TapeEntry){
            .op = OP_EMBED_FEATURES,
            .out = out,
            .inputs = inputs,
            .n_inputs = 2,
            .aux_data = data_copy
        };
    }
    
    return out;
}

// Direct implementation of output projection
Tensor* project_out(Tensor* x, Tensor* W_out, Config* cfg) {
    if (!x || !W_out) RETURN_ERR(ERR_NULL_TENSOR, "Null input tensor");
    
    int out_dims[] = {cfg->batch_size, cfg->seq_len, cfg->sequence_features};
    Tensor* out = tensor_create(3, out_dims, x->grad != NULL);
    if (!out) return NULL;
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < cfg->batch_size; b++) {
        for (int s = 0; s < cfg->seq_len; s++) {
            float* x_ptr = x->data + (b * cfg->seq_len + s) * cfg->d_model;
            float* out_ptr = out->data + (b * cfg->seq_len + s) * cfg->sequence_features;
            
            for (int i = 0; i < cfg->sequence_features; i++) {
                float sum = 0;
                for (int j = 0; j < cfg->d_model; j++) {
                    sum += x_ptr[j] * W_out->data[j * cfg->sequence_features + i];
                }
                out_ptr[i] = sum;
            }
        }
    }
    
    if (out->grad) {
        Tensor** inputs = malloc(2 * sizeof(Tensor*));
        inputs[0] = tensor_incref(x);
        inputs[1] = tensor_incref(W_out);
        
        tape[tape_len++] = (TapeEntry){
            .op = OP_PROJECT_OUT,
            .out = out,
            .inputs = inputs,
            .n_inputs = 2,
            .aux_data = NULL
        };
    }
    
    return out;
}

// Backward pass implementations
static void backward_attention_block(TapeEntry* e) {
    AttentionCache* cache = e->aux_data;
    Tensor* x = e->inputs[0];
    Tensor* Wq = e->inputs[1];
    Tensor* Wk = e->inputs[2];
    Tensor* Wv = e->inputs[3];
    Tensor* out = e->out;
    
    float* dQ = calloc(cache->batch_size * cache->n_head * cache->seq_len * cache->d_head, sizeof(float));
    float* dK = calloc(cache->batch_size * cache->n_head * cache->seq_len * cache->d_head, sizeof(float));
    float* dV = calloc(cache->batch_size * cache->n_head * cache->seq_len * cache->d_head, sizeof(float));
    
    if (!dQ || !dK || !dV) {
        free(dQ);
        free(dK);
        free(dV);
        return;  // Memory allocation failed
    }
    
    // Backward pass through attention mechanism
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < cache->batch_size; b++) {
        for (int h = 0; h < cache->n_head; h++) {
            for (int i = 0; i < cache->seq_len; i++) {
                // Gradient w.r.t. V
                for (int j = 0; j < cache->seq_len; j++) {
                    for (int d = 0; d < cache->d_head; d++) {
                        float grad = 0;
                        for (int k = j; k < cache->seq_len; k++) {
                            grad += cache->scores[((b * cache->n_head + h) * cache->seq_len + k) * cache->seq_len + j] *
                                   out->grad[(b * cache->seq_len + k) * cache->d_model + h * cache->d_head + d];
                        }
                        dV[((b * cache->n_head + h) * cache->seq_len + j) * cache->d_head + d] = grad;
                    }
                }
                
                // Gradient w.r.t. Q and K
                for (int j = 0; j <= i; j++) {  // Only consider attended positions (causal mask)
                    float sum_dv = 0;
                    for (int d = 0; d < cache->d_head; d++) {
                        sum_dv += out->grad[(b * cache->seq_len + i) * cache->d_model + h * cache->d_head + d] *
                                 cache->V[((b * cache->n_head + h) * cache->seq_len + j) * cache->d_head + d];
                    }
                    
                    float dscore = sum_dv;
                    float dqk = dscore * cache->scale;
                    
                    // Accumulate gradients for Q and K
                    for (int d = 0; d < cache->d_head; d++) {
                        dQ[((b * cache->n_head + h) * cache->seq_len + i) * cache->d_head + d] +=
                            dqk * cache->K[((b * cache->n_head + h) * cache->seq_len + j) * cache->d_head + d];
                        
                        dK[((b * cache->n_head + h) * cache->seq_len + j) * cache->d_head + d] +=
                            dqk * cache->Q[((b * cache->n_head + h) * cache->seq_len + i) * cache->d_head + d];
                    }
                }
            }
        }
    }
    
    // Accumulate gradients for weight matrices
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < cache->batch_size; b++) {
        for (int s = 0; s < cache->seq_len; s++) {
            float* x_ptr = x->data + (b * cache->seq_len + s) * cache->d_model;
            
            // Gradients for Q, K, V weight matrices
            for (int h = 0; h < cache->n_head; h++) {
                for (int d = 0; d < cache->d_head; d++) {
                    float dq = dQ[((b * cache->n_head + h) * cache->seq_len + s) * cache->d_head + d];
                    float dk = dK[((b * cache->n_head + h) * cache->seq_len + s) * cache->d_head + d];
                    float dv = dV[((b * cache->n_head + h) * cache->seq_len + s) * cache->d_head + d];
                    
                    int head_offset = h * cache->d_head;
                    for (int j = 0; j < cache->d_model; j++) {
                        #pragma omp atomic
                        Wq->grad[j * cache->d_model + head_offset + d] += x_ptr[j] * dq;
                        #pragma omp atomic
                        Wk->grad[j * cache->d_model + head_offset + d] += x_ptr[j] * dk;
                        #pragma omp atomic
                        Wv->grad[j * cache->d_model + head_offset + d] += x_ptr[j] * dv;
                    }
                }
            }
        }
    }
    
    // Clean up
    free(dQ);
    free(dK);
    free(dV);
    free(cache->Q);
    free(cache->K);
    free(cache->V);
    free(cache->scores);
    free(cache);
}

static void backward_feedforward_block(TapeEntry* e) {
    FeedForwardCache* cache = e->aux_data;
    Tensor* x = e->inputs[0];
    Tensor* W1 = e->inputs[1];
    Tensor* W2 = e->inputs[2];
    Tensor* out = e->out;
    
    float* d_hidden = calloc(cache->batch_size * cache->seq_len * cache->d_ff, sizeof(float));
    
    // Backward through second linear layer
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < cache->batch_size; b++) {
        for (int s = 0; s < cache->seq_len; s++) {
            float* out_grad = out->grad + (b * cache->seq_len + s) * cache->d_model;
            float* hidden_ptr = cache->hidden + (b * cache->seq_len + s) * cache->d_ff;
            
            for (int i = 0; i < cache->d_ff; i++) {
                float grad = 0;
                for (int j = 0; j < cache->d_model; j++) {
                    grad += out_grad[j] * W2->data[i * cache->d_model + j];
                    W2->grad[i * cache->d_model + j] += out_grad[j] * hidden_ptr[i];
                }
                d_hidden[(b * cache->seq_len + s) * cache->d_ff + i] = grad;
            }
        }
    }
    
    // Backward through GELU
    #pragma omp parallel for
    for (int i = 0; i < cache->batch_size * cache->seq_len * cache->d_ff; i++) {
        float x = cache->hidden_pre_gelu[i];
        float cdf = 0.5f * (1.0f + tanhf(sqrtf(2.0f/M_PI) * (x + 0.044715f * x * x * x)));
        float pdf = exp(-0.5f * x * x) / sqrtf(2.0f * M_PI);
        float dgelu = cdf + x * pdf;
        d_hidden[i] *= dgelu;
    }
    
    // Backward through first linear layer
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < cache->batch_size; b++) {
        for (int s = 0; s < cache->seq_len; s++) {
            float* x_ptr = x->data + (b * cache->seq_len + s) * cache->d_model;
            float* d_hidden_ptr = d_hidden + (b * cache->seq_len + s) * cache->d_ff;
            
            for (int i = 0; i < cache->d_model; i++) {
                for (int j = 0; j < cache->d_ff; j++) {
                    W1->grad[i * cache->d_ff + j] += x_ptr[i] * d_hidden_ptr[j];
                }
            }
        }
    }
    
    free(d_hidden);
    free(cache->hidden);
    free(cache->hidden_pre_gelu);
    free(cache);
}

static void backward_embed_features(TapeEntry* e) {
    Config* cfg = e->aux_data;
    Tensor* W_e = e->inputs[0];
    Tensor* W_cond = e->inputs[1];
    float* batch_data = e->aux_data;
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < cfg->batch_size; b++) {
        for (int s = 0; s < cfg->seq_len; s++) {
            float* data_ptr = batch_data + (b * cfg->seq_len + s) * 
                             (cfg->condition_features + cfg->sequence_features);
            float* grad_ptr = e->out->grad + (b * cfg->seq_len + s) * cfg->d_model;
            
            // Gradients for conditional embedding
            for (int i = 0; i < cfg->d_model; i++) {
                for (int j = 0; j < cfg->condition_features; j++) {
                    W_cond->grad[j * cfg->d_model + i] += data_ptr[j] * grad_ptr[i];
                }
            }
            
            // Gradients for sequence embedding
            for (int i = 0; i < cfg->d_model; i++) {
                for (int j = 0; j < cfg->sequence_features; j++) {
                    W_e->grad[j * cfg->d_model + i] += 
                        data_ptr[cfg->condition_features + j] * grad_ptr[i];
                }
            }
        }
    }
    
    free(batch_data);
}

static void backward_project_out(TapeEntry* e) {
    Tensor* x = e->inputs[0];
    Tensor* W_out = e->inputs[1];
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < x->dims[0]; b++) {
        for (int s = 0; s < x->dims[1]; s++) {
            float* x_ptr = x->data + (b * x->dims[1] + s) * x->dims[2];
            float* grad_ptr = e->out->grad + (b * x->dims[1] + s) * W_out->dims[1];
            
            for (int i = 0; i < x->dims[2]; i++) {
                for (int j = 0; j < W_out->dims[1]; j++) {
                    W_out->grad[i * W_out->dims[1] + j] += x_ptr[i] * grad_ptr[j];
                }
            }
        }
    }
}

void backward() {
    for (int t = tape_len - 1; t >= 0; t--) {
        TapeEntry* e = &tape[t];
        
        switch (e->op) {
            case OP_ATTENTION_BLOCK:
                backward_attention_block(e);
                break;
            case OP_FEEDFORWARD_BLOCK:
                backward_feedforward_block(e);
                break;
            case OP_EMBED_FEATURES:
                backward_embed_features(e);
                break;
            case OP_PROJECT_OUT:
                backward_project_out(e);
                break;
        }
        
        for (int i = 0; i < e->n_inputs; i++) {
            tensor_decref(e->inputs[i]);
        }
        free(e->inputs);
    }
    
    tape_len = 0;
}

// Full transformer implementation
Tensor* transformer_forward(Tensor* x, TransformerWeights* w, Config* cfg) {
    if (!x || !w) RETURN_ERR(ERR_NULL_TENSOR, "Null input tensor or weights");
    
    Tensor* current = x;
    
    // Process each transformer layer
    for (int i = 0; i < cfg->n_layers; i++) {
        // Attention block
        Tensor* attn = attention_block(current, w->W_q[i], w->W_k[i], w->W_v[i], 
                                     w->W_o[i], cfg->n_head, 1e-5f);
        if (!attn) {
            if (current != x) tensor_decref(current);
            return NULL;
        }
        
        // Feed-forward block
        Tensor* ff = feedforward_block(attn, w->W_ff1[i], w->W_ff2[i], 1e-5f);
        tensor_decref(attn);
        
        if (!ff) {
            if (current != x) tensor_decref(current);
            return NULL;
        }
        
        if (current != x) tensor_decref(current);
        current = ff;
    }
    
    return current;
}

void free_transformer_weights(TransformerWeights* w, Config* cfg) {
    if (!w) return;
    
    tensor_decref(w->W_e);
    tensor_decref(w->W_cond);
    tensor_decref(w->W_out);
    
    if (w->W_q) {
        for (int i = 0; i < cfg->n_layers; i++) {
            tensor_decref(w->W_q[i]);
            tensor_decref(w->W_k[i]);
            tensor_decref(w->W_v[i]);
            tensor_decref(w->W_o[i]);
            tensor_decref(w->W_ff1[i]);
            tensor_decref(w->W_ff2[i]);
        }
        free(w->W_q);
        free(w->W_k);
        free(w->W_v);
        free(w->W_o);
        free(w->W_ff1);
        free(w->W_ff2);
    }
    
    free(w);
}

// Weight initialization
TransformerWeights* create_transformer_weights(Config* cfg) {
    TransformerWeights* w = malloc(sizeof(TransformerWeights));
    if (!w) return NULL;
    memset(w, 0, sizeof(TransformerWeights));  // Initialize all pointers to NULL
    
    // Create embedding weights
    int dims_e[] = {cfg->sequence_features, cfg->d_model};
    int dims_cond[] = {cfg->condition_features, cfg->d_model};
    w->W_e = tensor_randn(2, dims_e, 1);
    w->W_cond = tensor_randn(2, dims_cond, 1);
    
    if (!w->W_e || !w->W_cond) goto cleanup;
    
    // Initialize layer weights
    w->W_q = malloc(cfg->n_layers * sizeof(Tensor*));
    w->W_k = malloc(cfg->n_layers * sizeof(Tensor*));
    w->W_v = malloc(cfg->n_layers * sizeof(Tensor*));
    w->W_o = malloc(cfg->n_layers * sizeof(Tensor*));
    w->W_ff1 = malloc(cfg->n_layers * sizeof(Tensor*));
    w->W_ff2 = malloc(cfg->n_layers * sizeof(Tensor*));
    
    if (!w->W_q || !w->W_k || !w->W_v || !w->W_o || !w->W_ff1 || !w->W_ff2) 
        goto cleanup;
    
    int dims_attn[] = {cfg->d_model, cfg->d_model};
    int dims_ff1[] = {cfg->d_model, cfg->d_ff};
    int dims_ff2[] = {cfg->d_ff, cfg->d_model};
    
    for (int i = 0; i < cfg->n_layers; i++) {
        w->W_q[i] = tensor_randn(2, dims_attn, 1);
        w->W_k[i] = tensor_randn(2, dims_attn, 1);
        w->W_v[i] = tensor_randn(2, dims_attn, 1);
        w->W_o[i] = tensor_randn(2, dims_attn, 1);
        w->W_ff1[i] = tensor_randn(2, dims_ff1, 1);
        w->W_ff2[i] = tensor_randn(2, dims_ff2, 1);
        
        if (!w->W_q[i] || !w->W_k[i] || !w->W_v[i] || !w->W_o[i] || 
            !w->W_ff1[i] || !w->W_ff2[i]) 
            goto cleanup;
    }
    
    // Create output projection
    int dims_out[] = {cfg->d_model, cfg->sequence_features};
    w->W_out = tensor_randn(2, dims_out, 1);
    if (!w->W_out) goto cleanup;
    
    // Scale initial weights
    float w_scale = sqrtf(2.0f / cfg->d_model);
    void scale_tensor(Tensor* t) {
        for (size_t j = 0; j < t->size; j++) {
            t->data[j] *= w_scale;
        }
    }
    
    scale_tensor(w->W_e);
    scale_tensor(w->W_cond);
    scale_tensor(w->W_out);
    
    for (int i = 0; i < cfg->n_layers; i++) {
        scale_tensor(w->W_q[i]);
        scale_tensor(w->W_k[i]);
        scale_tensor(w->W_v[i]);
        scale_tensor(w->W_o[i]);
        scale_tensor(w->W_ff1[i]);
        scale_tensor(w->W_ff2[i]);
    }
    
    return w;

cleanup:
    free_transformer_weights(w, cfg);
    return NULL;
}

// Training implementation
typedef struct {
    float* data;
    int rows;
    int cols;
} Dataset;

Dataset* load_csv(const char* filename, Config* cfg) {
    Dataset* dataset = malloc(sizeof(Dataset));
    if (!dataset) return NULL;
    dataset->data = NULL;
    
    FILE* file = fopen(filename, "r");
    if (!file) {
        free(dataset);
        return NULL;
    }
    
    // Skip header
    char line[4096];
    if (!fgets(line, sizeof(line), file)) {
        fclose(file);
        free(dataset);
        return NULL;
    }
    
    // Count lines and allocate memory
    int capacity = 1000;
    dataset->cols = cfg->condition_features + cfg->sequence_features;
    dataset->data = malloc(capacity * dataset->cols * sizeof(float));
    if (!dataset->data) {
        fclose(file);
        free(dataset);
        return NULL;
    }
    
    // Read and normalize data
    float* mins = calloc(dataset->cols, sizeof(float));
    float* maxs = calloc(dataset->cols, sizeof(float));
    for (int i = 0; i < dataset->cols; i++) {
        mins[i] = INFINITY;
        maxs[i] = -INFINITY;
    }
    
    dataset->rows = 0;
    while (fgets(line, sizeof(line), file)) {
        if (dataset->rows >= capacity) {
            capacity *= 2;
            float* new_data = realloc(dataset->data, capacity * dataset->cols * sizeof(float));
            if (!new_data) {
                free(dataset->data);
                free(mins);
                free(maxs);
                free(dataset);
                fclose(file);
                return NULL;
            }
            dataset->data = new_data;
        }
        
        char* token = strtok(line, ",");
        for (int i = 0; i < dataset->cols; i++) {
            if (!token) {
                free(dataset->data);
                free(mins);
                free(maxs);
                free(dataset);
                fclose(file);
                return NULL;
            }
            
            float val = atof(token);
            dataset->data[dataset->rows * dataset->cols + i] = val;
            mins[i] = fminf(mins[i], val);
            maxs[i] = fmaxf(maxs[i], val);
            token = strtok(NULL, ",");
        }
        dataset->rows++;
    }
    
    // Normalize data to [-1, 1]
    for (int i = 0; i < dataset->rows; i++) {
        for (int j = 0; j < dataset->cols; j++) {
            float range = maxs[j] - mins[j];
            if (range > 0) {
                dataset->data[i * dataset->cols + j] = 
                    2.0f * (dataset->data[i * dataset->cols + j] - mins[j]) / range - 1.0f;
            }
        }
    }
    
    free(mins);
    free(maxs);
    fclose(file);
    return dataset;
}

float train_batch(float* batch_data, float* target_data, 
                 TransformerWeights* w, Config* cfg) {
    // Embed input features
    Tensor* embedded = embed_features(batch_data, w->W_e, w->W_cond, cfg);
    if (!embedded) return INFINITY;
    
    // Forward pass through transformer
    Tensor* transformer_out = transformer_forward(embedded, w, cfg);
    tensor_decref(embedded);
    if (!transformer_out) return INFINITY;
    
    // Project to output space
    Tensor* output = project_out(transformer_out, w->W_out, cfg);
    tensor_decref(transformer_out);
    if (!output) return INFINITY;
    
    // Compute MSE loss and gradients
    float total_loss = 0.0f;
    int target_offset = cfg->condition_features;
    int features_per_step = cfg->condition_features + cfg->sequence_features;
    
    for (int b = 0; b < cfg->batch_size; b++) {
        for (int s = 0; s < cfg->seq_len; s++) {
            for (int f = 0; f < cfg->sequence_features; f++) {
                int out_idx = (b * cfg->seq_len + s) * cfg->sequence_features + f;
                int target_idx = (b * cfg->seq_len + s) * features_per_step + 
                                target_offset + f;
                float diff = output->data[out_idx] - target_data[target_idx];
                total_loss += diff * diff;
                output->grad[out_idx] = 2.0f * diff;
            }
        }
    }
    
    // Backward pass
    backward();
    
    // Update weights with gradient descent
    float max_grad = 1.0f;  // Gradient clipping threshold
    
    void update_weights(Tensor* w) {
        for (size_t i = 0; i < w->size; i++) {
            float grad = fmaxf(fminf(w->grad[i], max_grad), -max_grad);
            w->data[i] -= cfg->learning_rate * grad;
            w->grad[i] = 0.0f;
        }
    }
    
    update_weights(w->W_e);
    update_weights(w->W_cond);
    update_weights(w->W_out);
    
    for (int i = 0; i < cfg->n_layers; i++) {
        update_weights(w->W_q[i]);
        update_weights(w->W_k[i]);
        update_weights(w->W_v[i]);
        update_weights(w->W_o[i]);
        update_weights(w->W_ff1[i]);
        update_weights(w->W_ff2[i]);
    }
    
    float avg_loss = total_loss / (cfg->batch_size * cfg->seq_len * cfg->sequence_features);
    tensor_decref(output);
    return avg_loss;
}

void train(Dataset* dataset, TransformerWeights* w, Config* cfg) {
    int n_batches = (dataset->rows - cfg->seq_len) / cfg->batch_size;
    
    for (int epoch = 0; epoch < cfg->epochs; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < n_batches; batch++) {
            // Prepare batch data
            float* batch_data = malloc(cfg->batch_size * cfg->seq_len * 
                                     (cfg->condition_features + cfg->sequence_features) * 
                                     sizeof(float));
            float* target_data = malloc(cfg->batch_size * cfg->seq_len * 
                                      (cfg->condition_features + cfg->sequence_features) * 
                                      sizeof(float));
            
            for (int b = 0; b < cfg->batch_size; b++) {
                int start_idx = (batch * cfg->batch_size + b) % 
                              (dataset->rows - cfg->seq_len - 1);
                
                memcpy(batch_data + b * cfg->seq_len * dataset->cols,
                       dataset->data + start_idx * dataset->cols,
                       cfg->seq_len * dataset->cols * sizeof(float));
                       
                memcpy(target_data + b * cfg->seq_len * dataset->cols,
                       dataset->data + (start_idx + 1) * dataset->cols,
                       cfg->seq_len * dataset->cols * sizeof(float));
            }
            
            float batch_loss = train_batch(batch_data, target_data, w, cfg);
            epoch_loss += batch_loss;
            
            if (batch % 10 == 0) {
                printf("Epoch %d/%d, Batch %d/%d, Loss: %f\n", 
                       epoch + 1, cfg->epochs, batch + 1, n_batches, batch_loss);
            }
            
            free(batch_data);
            free(target_data);
        }
        
        printf("Epoch %d/%d complete, Average Loss: %f\n", 
               epoch + 1, cfg->epochs, epoch_loss / n_batches);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <data_file>\n", argv[0]);
        return 1;
    }
    
    Config cfg = {
        .batch_size = 32,
        .seq_len = 64,
        .d_model = 128,
        .n_head = 4,
        .n_layers = 3,
        .d_ff = 512,
        .learning_rate = 0.0001f,
        .epochs = 100,
        .condition_features = 4,
        .sequence_features = 10
    };
    
    // Load dataset
    Dataset* dataset = load_csv(argv[1], &cfg);
    if (!dataset) {
        printf("Failed to load dataset: %s\n", last_error.msg);
        return 1;
    }
    
    // Initialize transformer weights
    TransformerWeights* weights = create_transformer_weights(&cfg);
    if (!weights) {
        printf("Failed to create transformer weights: %s\n", last_error.msg);
        free(dataset->data);
        free(dataset);
        return 1;
    }
    
    // Train the model
    train(dataset, weights, &cfg);
    
    // Cleanup
    free_transformer_weights(weights, &cfg);
    free(dataset->data);
    free(dataset);
    
    return 0;
}