#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_LINE_LENGTH 1024
#define CONDITION_FEATURES 4
#define SEQUENCE_FEATURES 10
#define INPUT_FEATURES (CONDITION_FEATURES + SEQUENCE_FEATURES)
#define LEARNING_RATE 0.0001f
#define BATCH_SIZE 4
#define SEQ_LENGTH 64
#define D_MODEL 128
#define N_HEAD 4
#define N_LAYERS 2
#define EPOCHS 100

typedef struct {
    double *data, *grad;
    int *dims, ndims, size;
} Tensor;

typedef struct {
    double* data;
    int rows;
    int cols;
} Dataset;

// Helper functions
double randn() {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

void rmsnorm(Tensor *output, const Tensor *input, const Tensor *weight) {
    double eps = 1e-5;
    int features = D_MODEL;
    
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int s = 0; s < SEQ_LENGTH; s++) {
            // Calculate sum of squares
            double ss = 0.0;
            for (int i = 0; i < features; i++) {
                int idx = (b * SEQ_LENGTH * features) + (s * features) + i;
                ss += input->data[idx] * input->data[idx];
            }
            double rms = sqrt(ss / features + eps);
            
            // Normalize and scale
            for (int i = 0; i < features; i++) {
                int idx = (b * SEQ_LENGTH * features) + (s * features) + i;
                output->data[idx] = (input->data[idx] / rms) * weight->data[i];
            }
        }
    }
}

double gelu(double x) {
    return 0.5 * x * (1.0 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * pow(x, 3))));
}

void feedforward(Tensor *output, const Tensor *w1, const Tensor *w2, const Tensor *input) {
    int hidden_dim = D_MODEL * 4;
    
    double *intermediate = (double*)malloc(BATCH_SIZE * SEQ_LENGTH * hidden_dim * sizeof(double));
    
    // First linear layer + GELU
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int s = 0; s < SEQ_LENGTH; s++) {
            for (int h = 0; h < hidden_dim; h++) {
                double sum = 0.0;
                for (int d = 0; d < D_MODEL; d++) {
                    int in_idx = (b * SEQ_LENGTH * D_MODEL) + (s * D_MODEL) + d;
                    sum += input->data[in_idx] * w1->data[h * D_MODEL + d];
                }
                int out_idx = (b * SEQ_LENGTH * hidden_dim) + (s * hidden_dim) + h;
                intermediate[out_idx] = gelu(sum);
            }
        }
    }
    
    // Second linear layer
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int s = 0; s < SEQ_LENGTH; s++) {
            for (int d = 0; d < D_MODEL; d++) {
                double sum = 0.0;
                for (int h = 0; h < hidden_dim; h++) {
                    int in_idx = (b * SEQ_LENGTH * hidden_dim) + (s * hidden_dim) + h;
                    sum += intermediate[in_idx] * w2->data[d * hidden_dim + h];
                }
                int out_idx = (b * SEQ_LENGTH * D_MODEL) + (s * D_MODEL) + d;
                output->data[out_idx] = sum;
            }
        }
    }
    
    free(intermediate);
}

void multihead_attention(Tensor *output, const Tensor *q, const Tensor *k, const Tensor *v,
                        const Tensor *wq, const Tensor *wk, const Tensor *wv, const Tensor *wo) {
    int head_dim = D_MODEL / N_HEAD;
    
    double *q_heads = (double*)malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double *k_heads = (double*)malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double *v_heads = (double*)malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double *attn_scores = (double*)malloc(BATCH_SIZE * N_HEAD * SEQ_LENGTH * SEQ_LENGTH * sizeof(double));
    
    // Linear projections
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int s = 0; s < SEQ_LENGTH; s++) {
            for (int h = 0; h < N_HEAD; h++) {
                for (int d = 0; d < head_dim; d++) {
                    int head_offset = h * head_dim;
                    int out_idx = (b * SEQ_LENGTH * D_MODEL) + (s * D_MODEL) + head_offset + d;
                    
                    double sum_q = 0.0, sum_k = 0.0, sum_v = 0.0;
                    for (int i = 0; i < D_MODEL; i++) {
                        int in_idx = (b * SEQ_LENGTH * D_MODEL) + (s * D_MODEL) + i;
                        sum_q += q->data[in_idx] * wq->data[head_offset * D_MODEL + d * D_MODEL + i];
                        sum_k += k->data[in_idx] * wk->data[head_offset * D_MODEL + d * D_MODEL + i];
                        sum_v += v->data[in_idx] * wv->data[head_offset * D_MODEL + d * D_MODEL + i];
                    }
                    q_heads[out_idx] = sum_q;
                    k_heads[out_idx] = sum_k;
                    v_heads[out_idx] = sum_v;
                }
            }
        }
    }

       // Compute attention scores with ALiBi
    double scale = 1.0 / sqrt(head_dim);
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int h = 0; h < N_HEAD; h++) {
            // ALiBi slope for this head
            double m = pow(2.0, -(8.0 * (h + 1) / N_HEAD));
            
            for (int q_pos = 0; q_pos < SEQ_LENGTH; q_pos++) {
                for (int k_pos = 0; k_pos < SEQ_LENGTH; k_pos++) {
                    if (k_pos > q_pos) {  // Causal masking
                        attn_scores[(b * N_HEAD * SEQ_LENGTH * SEQ_LENGTH) + 
                                  (h * SEQ_LENGTH * SEQ_LENGTH) + 
                                  (q_pos * SEQ_LENGTH) + k_pos] = -INFINITY;
                        continue;
                    }
                    
                    double score = 0.0;
                    for (int d = 0; d < head_dim; d++) {
                        int q_idx = (b * SEQ_LENGTH * D_MODEL) + (q_pos * D_MODEL) + (h * head_dim) + d;
                        int k_idx = (b * SEQ_LENGTH * D_MODEL) + (k_pos * D_MODEL) + (h * head_dim) + d;
                        score += q_heads[q_idx] * k_heads[k_idx];
                    }
                    
                    // Apply scaling and ALiBi bias
                    score = score * scale - m * (q_pos - k_pos);
                    
                    attn_scores[(b * N_HEAD * SEQ_LENGTH * SEQ_LENGTH) + 
                               (h * SEQ_LENGTH * SEQ_LENGTH) + 
                               (q_pos * SEQ_LENGTH) + k_pos] = score;
                }
                
                // Softmax
                double max_score = -INFINITY;
                for (int k_pos = 0; k_pos <= q_pos; k_pos++) {
                    int idx = (b * N_HEAD * SEQ_LENGTH * SEQ_LENGTH) + 
                             (h * SEQ_LENGTH * SEQ_LENGTH) + 
                             (q_pos * SEQ_LENGTH) + k_pos;
                    if (attn_scores[idx] > max_score) max_score = attn_scores[idx];
                }
                
                double sum_exp = 0.0;
                for (int k_pos = 0; k_pos <= q_pos; k_pos++) {
                    int idx = (b * N_HEAD * SEQ_LENGTH * SEQ_LENGTH) + 
                             (h * SEQ_LENGTH * SEQ_LENGTH) + 
                             (q_pos * SEQ_LENGTH) + k_pos;
                    attn_scores[idx] = exp(attn_scores[idx] - max_score);
                    sum_exp += attn_scores[idx];
                }
                
                for (int k_pos = 0; k_pos <= q_pos; k_pos++) {
                    int idx = (b * N_HEAD * SEQ_LENGTH * SEQ_LENGTH) + 
                             (h * SEQ_LENGTH * SEQ_LENGTH) + 
                             (q_pos * SEQ_LENGTH) + k_pos;
                    attn_scores[idx] /= sum_exp;
                }
            }
        }
    }
    
    // Apply attention scores to values and project back
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int s = 0; s < SEQ_LENGTH; s++) {
            for (int d = 0; d < D_MODEL; d++) {
                double sum = 0.0;
                for (int h = 0; h < N_HEAD; h++) {
                    for (int s2 = 0; s2 <= s; s2++) {
                        int attn_idx = (b * N_HEAD * SEQ_LENGTH * SEQ_LENGTH) + 
                                     (h * SEQ_LENGTH * SEQ_LENGTH) + 
                                     (s * SEQ_LENGTH) + s2;
                        int v_idx = (b * SEQ_LENGTH * D_MODEL) + 
                                  (s2 * D_MODEL) + 
                                  (h * head_dim) + (d % head_dim);
                        sum += attn_scores[attn_idx] * v_heads[v_idx];
                    }
                }
                
                // Project back with Wo
                double final_sum = 0.0;
                for (int i = 0; i < D_MODEL; i++) {
                    final_sum += sum * wo->data[d * D_MODEL + i];
                }
                output->data[(b * SEQ_LENGTH * D_MODEL) + (s * D_MODEL) + d] = final_sum;
            }
        }
    }
    
    free(q_heads);
    free(k_heads);
    free(v_heads);
    free(attn_scores);
}

Dataset load_csv(const char* filename) {
    printf("Loading CSV file: %s\n", filename);
    Dataset dataset = {NULL, 0, INPUT_FEATURES};
    char line[MAX_LINE_LENGTH];
    double* temp = malloc(1000 * INPUT_FEATURES * sizeof(double));
    int capacity = 1000;
    
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Failed to open file\n");
        exit(1);
    }
    
    // Skip header
    if (fgets(line, MAX_LINE_LENGTH, file) == NULL) {
        printf("Failed to read header line\n");
        fclose(file);
        exit(1);
    }
    
    double* mins = calloc(INPUT_FEATURES, sizeof(double));
    double* maxs = calloc(INPUT_FEATURES, sizeof(double));
    for (int i = 0; i < INPUT_FEATURES; i++) {
        mins[i] = INFINITY;
        maxs[i] = -INFINITY;
    }
    
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        if (dataset.rows >= capacity) {
            capacity *= 2;
            temp = realloc(temp, capacity * INPUT_FEATURES * sizeof(double));
        }
        
        char* token = strtok(line, ",");
        for (int i = 0; i < INPUT_FEATURES; i++) {
            double val = atof(token);
            temp[dataset.rows * INPUT_FEATURES + i] = val;
            mins[i] = fmin(mins[i], val);
            maxs[i] = fmax(maxs[i], val);
            token = strtok(NULL, ",");
        }
        dataset.rows++;
    }
    
    // Normalize data
    for (int i = 0; i < dataset.rows; i++) {
        for (int j = 0; j < INPUT_FEATURES; j++) {
            double range = maxs[j] - mins[j];
            if (range > 0) {
                temp[i * INPUT_FEATURES + j] = 2.0 * (temp[i * INPUT_FEATURES + j] - mins[j]) / range - 1.0;
            }
        }
    }
    
    free(mins);
    free(maxs);
    fclose(file);
    dataset.data = temp;
    printf("Loaded %d rows\n", dataset.rows);
    return dataset;
}

int main() {
    srand(time(NULL));
    
    // Load data
    Dataset dataset = load_csv("2024-12-29_6-25-1_control_data.csv");
    
    // Initialize weights
    double w_scale = sqrt(2.0 / D_MODEL);
    
    // Embedding weights
    int dims_e[] = {SEQUENCE_FEATURES, D_MODEL};
    int dims_cond[] = {CONDITION_FEATURES, D_MODEL};
    Tensor W_e = {malloc(SEQUENCE_FEATURES * D_MODEL * sizeof(double)), 
                  malloc(SEQUENCE_FEATURES * D_MODEL * sizeof(double)), 
                  dims_e, 2, SEQUENCE_FEATURES * D_MODEL};
    Tensor W_cond = {malloc(CONDITION_FEATURES * D_MODEL * sizeof(double)), 
                     malloc(CONDITION_FEATURES * D_MODEL * sizeof(double)), 
                     dims_cond, 2, CONDITION_FEATURES * D_MODEL};
    
    // Initialize embeddings
    for (int i = 0; i < W_e.size; i++) W_e.data[i] = randn() * w_scale;
    for (int i = 0; i < W_cond.size; i++) W_cond.data[i] = randn() * w_scale;
    
    // Initialize transformer layers
    int attn_dims[] = {D_MODEL, D_MODEL};
    int ff_dims1[] = {D_MODEL, D_MODEL * 4};
    int ff_dims2[] = {D_MODEL * 4, D_MODEL};
    
    Tensor W_q[N_LAYERS], W_k[N_LAYERS], W_v[N_LAYERS], W_o[N_LAYERS];
    Tensor W_ff1[N_LAYERS], W_ff2[N_LAYERS];
    
    for (int l = 0; l < N_LAYERS; l++) {
        // Initialize attention weights
        W_q[l] = (Tensor){malloc(D_MODEL * D_MODEL * sizeof(double)), 
                          malloc(D_MODEL * D_MODEL * sizeof(double)), 
                          attn_dims, 2, D_MODEL * D_MODEL};
        W_k[l] = (Tensor){malloc(D_MODEL * D_MODEL * sizeof(double)), 
                          malloc(D_MODEL * D_MODEL * sizeof(double)), 
                          attn_dims, 2, D_MODEL * D_MODEL};
        W_v[l] = (Tensor){malloc(D_MODEL * D_MODEL * sizeof(double)), 
                          malloc(D_MODEL * D_MODEL * sizeof(double)), 
                          attn_dims, 2, D_MODEL * D_MODEL};
        W_o[l] = (Tensor){malloc(D_MODEL * D_MODEL * sizeof(double)), 
                          malloc(D_MODEL * D_MODEL * sizeof(double)), 
                          attn_dims, 2, D_MODEL * D_MODEL};
        
        // Initialize feedforward weights
        W_ff1[l] = (Tensor){malloc(D_MODEL * (D_MODEL * 4) * sizeof(double)), 
                            malloc(D_MODEL * (D_MODEL * 4) * sizeof(double)), 
                            ff_dims1, 2, D_MODEL * (D_MODEL * 4)};
        W_ff2[l] = (Tensor){malloc((D_MODEL * 4) * D_MODEL * sizeof(double)), 
                            malloc((D_MODEL * 4) * D_MODEL * sizeof(double)), 
                            ff_dims2, 2, (D_MODEL * 4) * D_MODEL};
        
        // Initialize weights
        for (int i = 0; i < D_MODEL * D_MODEL; i++) {
            W_q[l].data[i] = randn() * w_scale;
            W_k[l].data[i] = randn() * w_scale;
            W_v[l].data[i] = randn() * w_scale;
            W_o[l].data[i] = randn() * w_scale;
        }
        for (int i = 0; i < W_ff1[l].size; i++) W_ff1[l].data[i] = randn() * w_scale;
        for (int i = 0; i < W_ff2[l].size; i++) W_ff2[l].data[i] = randn() * w_scale;
    }
    
    // Process a single batch
    int batch_dims[] = {BATCH_SIZE, SEQ_LENGTH, D_MODEL};
    Tensor current = {malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double)), 
                     malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double)), 
                     batch_dims, 3, BATCH_SIZE * SEQ_LENGTH * D_MODEL};
    
    // Forward pass through the model
    for (int l = 0; l < N_LAYERS; l++) {
        Tensor temp = {malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double)), 
                      malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double)), 
                      batch_dims, 3, BATCH_SIZE * SEQ_LENGTH * D_MODEL};
        
        // Layer norm + attention
        rmsnorm(&temp, &current, &W_e);
        multihead_attention(&temp, &temp, &temp, &temp,
                          &W_q[l], &W_k[l], &W_v[l], &W_o[l]);
        
        // Add residual
        for (int i = 0; i < current.size; i++) {
            current.data[i] += temp.data[i];
        }
        
        // Layer norm + feedforward
        rmsnorm(&temp, &current, &W_e);
        feedforward(&temp, &W_ff1[l], &W_ff2[l], &temp);
        
        // Add residual
        for (int i = 0; i < current.size; i++) {
            current.data[i] += temp.data[i];
        }
        
        free(temp.data);
        free(temp.grad);
    }
    
    // Clean up
    free(dataset.data);
    free(current.data);
    free(current.grad);
    // ... free all other allocated memory ...
    
    return 0;
}