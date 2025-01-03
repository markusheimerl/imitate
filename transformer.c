#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_LINE_LENGTH 1024
#define CONDITION_FEATURES 4
#define SEQUENCE_FEATURES 10
#define INPUT_FEATURES (CONDITION_FEATURES + SEQUENCE_FEATURES)
#define BATCH_SIZE 4
#define SEQ_LENGTH 64
#define D_MODEL 128
#define N_HEAD 4
#define N_LAYERS 2
#define EPSILON 1e-4
#define LEARNING_RATE 0.01
#define TRAINING_STEPS 100

typedef struct {
    double *data, *grad;
    int *dims, ndims, size;
} Tensor;

typedef struct {
    double* data;
    int rows, cols;
    double *mins, *maxs;
} Dataset;

double normalize(double value, double min_val, double max_val) {
    double range = max_val - min_val;
    return range == 0 ? 0 : 2.0 * (value - min_val) / range - 1.0;
}

double denormalize(double normalized_value, double min_val, double max_val) {
    return (normalized_value + 1.0) * (max_val - min_val) / 2.0 + min_val;
}

double randn() {
    double u1 = (double)rand() / RAND_MAX, u2 = (double)rand() / RAND_MAX;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

void rmsnorm(Tensor *output, const Tensor *input) {
    double eps = 1e-5;
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int s = 0; s < SEQ_LENGTH; s++) {
            double ss = 0.0;
            for (int i = 0; i < D_MODEL; i++) {
                int idx = (b * SEQ_LENGTH * D_MODEL) + (s * D_MODEL) + i;
                ss += input->data[idx] * input->data[idx];
            }
            double rms = sqrt(ss / D_MODEL + eps);
            for (int i = 0; i < D_MODEL; i++) {
                int idx = (b * SEQ_LENGTH * D_MODEL) + (s * D_MODEL) + i;
                output->data[idx] = input->data[idx] / rms;
            }
        }
    }
}

double gelu(double x) {
    return 0.5 * x * (1.0 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * pow(x, 3))));
}

void feedforward(Tensor *output, const Tensor *w1, const Tensor *w2, const Tensor *input) {
    double *intermediate = malloc(BATCH_SIZE * SEQ_LENGTH * (D_MODEL * 4) * sizeof(double));
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int s = 0; s < SEQ_LENGTH; s++) {
            for (int h = 0; h < D_MODEL * 4; h++) {
                double sum = 0.0;
                for (int d = 0; d < D_MODEL; d++)
                    sum += input->data[(b * SEQ_LENGTH * D_MODEL) + (s * D_MODEL) + d] * w1->data[h * D_MODEL + d];
                intermediate[(b * SEQ_LENGTH * (D_MODEL * 4)) + (s * (D_MODEL * 4)) + h] = gelu(sum);
            }
        }
    }
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int s = 0; s < SEQ_LENGTH; s++) {
            for (int d = 0; d < D_MODEL; d++) {
                double sum = 0.0;
                for (int h = 0; h < D_MODEL * 4; h++)
                    sum += intermediate[(b * SEQ_LENGTH * (D_MODEL * 4)) + (s * (D_MODEL * 4)) + h] * w2->data[d * (D_MODEL * 4) + h];
                output->data[(b * SEQ_LENGTH * D_MODEL) + (s * D_MODEL) + d] = sum;
            }
        }
    }
    free(intermediate);
}

void multihead_attention(Tensor *output, const Tensor *input, const Tensor *wq, const Tensor *wk, const Tensor *wv, const Tensor *wo) {
    int head_dim = D_MODEL / N_HEAD;
    double scale = 1.0 / sqrt(head_dim);
    double *q_heads = malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double *k_heads = malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double *v_heads = malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double *scores = malloc(BATCH_SIZE * N_HEAD * SEQ_LENGTH * SEQ_LENGTH * sizeof(double));
    
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int s = 0; s < SEQ_LENGTH; s++) {
            for (int h = 0; h < N_HEAD; h++) {
                for (int d = 0; d < head_dim; d++) {
                    double sum_q = 0.0, sum_k = 0.0, sum_v = 0.0;
                    for (int i = 0; i < D_MODEL; i++) {
                        int in_idx = (b * SEQ_LENGTH * D_MODEL) + (s * D_MODEL) + i;
                        int w_idx = (h * head_dim + d) * D_MODEL + i;
                        sum_q += input->data[in_idx] * wq->data[w_idx];
                        sum_k += input->data[in_idx] * wk->data[w_idx];
                        sum_v += input->data[in_idx] * wv->data[w_idx];
                    }
                    int out_idx = (b * SEQ_LENGTH * D_MODEL) + (s * D_MODEL) + (h * head_dim) + d;
                    q_heads[out_idx] = sum_q;
                    k_heads[out_idx] = sum_k;
                    v_heads[out_idx] = sum_v;
                }
            }
        }
    }
    
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int h = 0; h < N_HEAD; h++) {
            double m = pow(2.0, -(8.0 * (h + 1) / N_HEAD));
            for (int q_pos = 0; q_pos < SEQ_LENGTH; q_pos++) {
                for (int k_pos = 0; k_pos < SEQ_LENGTH; k_pos++) {
                    if (k_pos > q_pos) {
                        scores[(b * N_HEAD * SEQ_LENGTH * SEQ_LENGTH) + (h * SEQ_LENGTH * SEQ_LENGTH) + 
                               (q_pos * SEQ_LENGTH) + k_pos] = -INFINITY;
                        continue;
                    }
                    double score = 0.0;
                    for (int d = 0; d < head_dim; d++) {
                        int q_idx = (b * SEQ_LENGTH * D_MODEL) + (q_pos * D_MODEL) + (h * head_dim) + d;
                        int k_idx = (b * SEQ_LENGTH * D_MODEL) + (k_pos * D_MODEL) + (h * head_dim) + d;
                        score += q_heads[q_idx] * k_heads[k_idx];
                    }
                    score = score * scale - m * (q_pos - k_pos);
                    scores[(b * N_HEAD * SEQ_LENGTH * SEQ_LENGTH) + (h * SEQ_LENGTH * SEQ_LENGTH) + 
                           (q_pos * SEQ_LENGTH) + k_pos] = score;
                }
                
                double max_score = -INFINITY;
                for (int k_pos = 0; k_pos <= q_pos; k_pos++) {
                    int idx = (b * N_HEAD * SEQ_LENGTH * SEQ_LENGTH) + (h * SEQ_LENGTH * SEQ_LENGTH) + 
                             (q_pos * SEQ_LENGTH) + k_pos;
                    if (scores[idx] > max_score) max_score = scores[idx];
                }
                
                double sum_exp = 0.0;
                for (int k_pos = 0; k_pos <= q_pos; k_pos++) {
                    int idx = (b * N_HEAD * SEQ_LENGTH * SEQ_LENGTH) + (h * SEQ_LENGTH * SEQ_LENGTH) + 
                             (q_pos * SEQ_LENGTH) + k_pos;
                    scores[idx] = exp(scores[idx] - max_score);
                    sum_exp += scores[idx];
                }
                
                for (int k_pos = 0; k_pos <= q_pos; k_pos++) {
                    int idx = (b * N_HEAD * SEQ_LENGTH * SEQ_LENGTH) + (h * SEQ_LENGTH * SEQ_LENGTH) + 
                             (q_pos * SEQ_LENGTH) + k_pos;
                    scores[idx] /= sum_exp;
                }
            }
        }
    }
    
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int s = 0; s < SEQ_LENGTH; s++) {
            for (int d = 0; d < D_MODEL; d++) {
                double sum = 0.0;
                for (int h = 0; h < N_HEAD; h++) {
                    double head_sum = 0.0;
                    for (int s2 = 0; s2 <= s; s2++) {
                        int score_idx = (b * N_HEAD * SEQ_LENGTH * SEQ_LENGTH) + (h * SEQ_LENGTH * SEQ_LENGTH) + 
                                      (s * SEQ_LENGTH) + s2;
                        int v_idx = (b * SEQ_LENGTH * D_MODEL) + (s2 * D_MODEL) + (h * head_dim) + (d % head_dim);
                        head_sum += scores[score_idx] * v_heads[v_idx];
                    }
                    sum += head_sum;
                }
                double final_sum = 0.0;
                for (int i = 0; i < D_MODEL; i++) final_sum += sum * wo->data[d * D_MODEL + i];
                output->data[(b * SEQ_LENGTH * D_MODEL) + (s * D_MODEL) + d] = final_sum;
            }
        }
    }
    free(q_heads); free(k_heads); free(v_heads); free(scores);
}

Dataset load_csv(const char* filename) {
    Dataset dataset = {NULL, 0, INPUT_FEATURES, calloc(INPUT_FEATURES, sizeof(double)), 
                      calloc(INPUT_FEATURES, sizeof(double))};
    char line[MAX_LINE_LENGTH];
    double* temp = malloc(1000 * INPUT_FEATURES * sizeof(double));
    int capacity = 1000;
    FILE* file = fopen(filename, "r");
    if (!file) { printf("Failed to open file\n"); exit(1); }
    if (fgets(line, MAX_LINE_LENGTH, file) == NULL) {
        printf("Failed to read header line\n"); fclose(file); exit(1);
    }
    for (int i = 0; i < INPUT_FEATURES; i++) {
        dataset.mins[i] = INFINITY;
        dataset.maxs[i] = -INFINITY;
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
            dataset.mins[i] = fmin(dataset.mins[i], val);
            dataset.maxs[i] = fmax(dataset.maxs[i], val);
            token = strtok(NULL, ",");
        }
        dataset.rows++;
    }
    for (int i = 0; i < dataset.rows; i++)
        for (int j = 0; j < INPUT_FEATURES; j++)
            temp[i * INPUT_FEATURES + j] = normalize(temp[i * INPUT_FEATURES + j], dataset.mins[j], dataset.maxs[j]);
    dataset.data = temp;
    return dataset;
}

void embed_sequence(Tensor* output, const double* input, const Tensor* W_seq, const Tensor* W_cond) {
    for (int b = 0; b < BATCH_SIZE; b++)
        for (int s = 0; s < SEQ_LENGTH; s++)
            for (int d = 0; d < D_MODEL; d++) {
                double sum = 0.0;
                for (int f = 0; f < SEQUENCE_FEATURES; f++)
                    sum += input[(b * SEQ_LENGTH + s) * INPUT_FEATURES + f + CONDITION_FEATURES] * 
                           W_seq->data[f * D_MODEL + d];
                for (int f = 0; f < CONDITION_FEATURES; f++)
                    sum += input[(b * SEQ_LENGTH + s) * INPUT_FEATURES + f] * W_cond->data[f * D_MODEL + d];
                output->data[(b * SEQ_LENGTH * D_MODEL) + (s * D_MODEL) + d] = sum;
            }
}

double forward_pass(const Dataset* dataset, Tensor* output, Tensor* hidden, Tensor* temp,
                   const Tensor* W_seq, const Tensor* W_cond, 
                   const Tensor* W_q, const Tensor* W_k, const Tensor* W_v, const Tensor* W_o,
                   const Tensor* W_ff1, const Tensor* W_ff2, const Tensor* W_out) {
    
    embed_sequence(hidden, dataset->data, W_seq, W_cond);
    
    for (int l = 0; l < N_LAYERS; l++) {
        rmsnorm(temp, hidden);
        multihead_attention(temp, temp, &W_q[l], &W_k[l], &W_v[l], &W_o[l]);
        for (int i = 0; i < hidden->size; i++) 
            hidden->data[i] += temp->data[i];
        
        rmsnorm(temp, hidden);
        feedforward(temp, &W_ff1[l], &W_ff2[l], temp);
        for (int i = 0; i < hidden->size; i++) 
            hidden->data[i] += temp->data[i];
    }
    
    // Compute output
    for (int b = 0; b < BATCH_SIZE; b++)
        for (int s = 0; s < SEQ_LENGTH; s++)
            for (int f = 0; f < SEQUENCE_FEATURES; f++) {
                double sum = 0.0;
                for (int d = 0; d < D_MODEL; d++)
                    sum += hidden->data[(b * SEQ_LENGTH * D_MODEL) + (s * D_MODEL) + d] * 
                           W_out->data[d * SEQUENCE_FEATURES + f];
                output->data[(b * SEQ_LENGTH * SEQUENCE_FEATURES) + (s * SEQUENCE_FEATURES) + f] = sum;
            }
    
    // Compute loss
    double total_loss = 0.0;
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int s = 0; s < SEQ_LENGTH; s++) {
            for (int f = 0; f < SEQUENCE_FEATURES; f++) {
                int pred_idx = (b * SEQ_LENGTH * SEQUENCE_FEATURES) + (s * SEQUENCE_FEATURES) + f;
                int actual_idx = ((b * SEQ_LENGTH + s + 1) * INPUT_FEATURES) + f + CONDITION_FEATURES;
                double diff = output->data[pred_idx] - dataset->data[actual_idx];
                total_loss += diff * diff;
            }
        }
    }
    
    return total_loss / (BATCH_SIZE * SEQ_LENGTH * SEQUENCE_FEATURES);
}

void train_finite_diff(Dataset* dataset, Tensor* output, Tensor* hidden, Tensor* temp,
                      Tensor* W_seq, Tensor* W_cond, 
                      Tensor* W_q, Tensor* W_k, Tensor* W_v, Tensor* W_o,
                      Tensor* W_ff1, Tensor* W_ff2, Tensor* W_out) {
    
    for (int step = 0; step < TRAINING_STEPS; step++) {
        double base_loss = forward_pass(dataset, output, hidden, temp,
                                      W_seq, W_cond, W_q, W_k, W_v, W_o,
                                      W_ff1, W_ff2, W_out);
        
        printf("Step %d, Loss: %f\n", step, base_loss);
        
        // Update each parameter using finite differences
        for (int l = 0; l < N_LAYERS; l++) {
            // Update W_q
            for (int i = 0; i < W_q[l].size; i++) {
                W_q[l].data[i] += EPSILON;
                double new_loss = forward_pass(dataset, output, hidden, temp,
                                             W_seq, W_cond, W_q, W_k, W_v, W_o,
                                             W_ff1, W_ff2, W_out);
                W_q[l].data[i] -= EPSILON;
                double grad = (new_loss - base_loss) / EPSILON;
                W_q[l].data[i] -= LEARNING_RATE * grad;
            }
            
            // Update W_k
            for (int i = 0; i < W_k[l].size; i++) {
                W_k[l].data[i] += EPSILON;
                double new_loss = forward_pass(dataset, output, hidden, temp,
                                             W_seq, W_cond, W_q, W_k, W_v, W_o,
                                             W_ff1, W_ff2, W_out);
                W_k[l].data[i] -= EPSILON;
                double grad = (new_loss - base_loss) / EPSILON;
                W_k[l].data[i] -= LEARNING_RATE * grad;
            }
            
            // Update W_v
            for (int i = 0; i < W_v[l].size; i++) {
                W_v[l].data[i] += EPSILON;
                double new_loss = forward_pass(dataset, output, hidden, temp,
                                             W_seq, W_cond, W_q, W_k, W_v, W_o,
                                             W_ff1, W_ff2, W_out);
                W_v[l].data[i] -= EPSILON;
                double grad = (new_loss - base_loss) / EPSILON;
                W_v[l].data[i] -= LEARNING_RATE * grad;
            }
            
            // Update W_o
            for (int i = 0; i < W_o[l].size; i++) {
                W_o[l].data[i] += EPSILON;
                double new_loss = forward_pass(dataset, output, hidden, temp,
                                             W_seq, W_cond, W_q, W_k, W_v, W_o,
                                             W_ff1, W_ff2, W_out);
                W_o[l].data[i] -= EPSILON;
                double grad = (new_loss - base_loss) / EPSILON;
                W_o[l].data[i] -= LEARNING_RATE * grad;
            }
            
            // Update W_ff1
            for (int i = 0; i < W_ff1[l].size; i++) {
                W_ff1[l].data[i] += EPSILON;
                double new_loss = forward_pass(dataset, output, hidden, temp,
                                             W_seq, W_cond, W_q, W_k, W_v, W_o,
                                             W_ff1, W_ff2, W_out);
                W_ff1[l].data[i] -= EPSILON;
                double grad = (new_loss - base_loss) / EPSILON;
                W_ff1[l].data[i] -= LEARNING_RATE * grad;
            }
            
            // Update W_ff2
            for (int i = 0; i < W_ff2[l].size; i++) {
                W_ff2[l].data[i] += EPSILON;
                double new_loss = forward_pass(dataset, output, hidden, temp,
                                             W_seq, W_cond, W_q, W_k, W_v, W_o,
                                             W_ff1, W_ff2, W_out);
                W_ff2[l].data[i] -= EPSILON;
                double grad = (new_loss - base_loss) / EPSILON;
                W_ff2[l].data[i] -= LEARNING_RATE * grad;
            }
        }
        
        // Update embedding weights (W_seq)
        for (int i = 0; i < W_seq->size; i++) {
            W_seq->data[i] += EPSILON;
            double new_loss = forward_pass(dataset, output, hidden, temp,
                                         W_seq, W_cond, W_q, W_k, W_v, W_o,
                                         W_ff1, W_ff2, W_out);
            W_seq->data[i] -= EPSILON;
            double grad = (new_loss - base_loss) / EPSILON;
            W_seq->data[i] -= LEARNING_RATE * grad;
        }
        
        // Update condition weights (W_cond)
        for (int i = 0; i < W_cond->size; i++) {
            W_cond->data[i] += EPSILON;
            double new_loss = forward_pass(dataset, output, hidden, temp,
                                         W_seq, W_cond, W_q, W_k, W_v, W_o,
                                         W_ff1, W_ff2, W_out);
            W_cond->data[i] -= EPSILON;
            double grad = (new_loss - base_loss) / EPSILON;
            W_cond->data[i] -= LEARNING_RATE * grad;
        }
        
        // Update output weights (W_out)
        for (int i = 0; i < W_out->size; i++) {
            W_out->data[i] += EPSILON;
            double new_loss = forward_pass(dataset, output, hidden, temp,
                                         W_seq, W_cond, W_q, W_k, W_v, W_o,
                                         W_ff1, W_ff2, W_out);
            W_out->data[i] -= EPSILON;
            double grad = (new_loss - base_loss) / EPSILON;
            W_out->data[i] -= LEARNING_RATE * grad;
        }
        
        // Optional: Print predictions every N steps
        if (step % 10 == 0) {
            printf("\nPredictions at step %d:\n", step);
            for (int b = 0; b < 1; b++) {  // Just print first batch
                for (int s = 0; s < 5; s++) {  // Just print first 5 steps
                    printf("Step %d: ", s);
                    for (int f = 0; f < SEQUENCE_FEATURES; f++) {
                        int pred_idx = (b * SEQ_LENGTH * SEQUENCE_FEATURES) + (s * SEQUENCE_FEATURES) + f;
                        int actual_idx = ((b * SEQ_LENGTH + s + 1) * INPUT_FEATURES) + f + CONDITION_FEATURES;
                        double pred = denormalize(output->data[pred_idx], 
                                                dataset->mins[f + CONDITION_FEATURES],
                                                dataset->maxs[f + CONDITION_FEATURES]);
                        double actual = denormalize(dataset->data[actual_idx],
                                                  dataset->mins[f + CONDITION_FEATURES],
                                                  dataset->maxs[f + CONDITION_FEATURES]);
                        printf("F%d(P:%.2f,A:%.2f) ", f, pred, actual);
                    }
                    printf("\n");
                }
                printf("\n");
            }
        }
    }
}

int main() {
    srand(time(NULL));
    Dataset dataset = load_csv("2024-12-29_6-25-1_control_data.csv");
    double w_scale = sqrt(2.0 / D_MODEL);
    
    // Initialize model parameters
    int dims_e[] = {SEQUENCE_FEATURES, D_MODEL};
    int dims_cond[] = {CONDITION_FEATURES, D_MODEL};
    Tensor W_seq = {malloc(SEQUENCE_FEATURES * D_MODEL * sizeof(double)), NULL, dims_e, 2, 
                    SEQUENCE_FEATURES * D_MODEL};
    Tensor W_cond = {malloc(CONDITION_FEATURES * D_MODEL * sizeof(double)), NULL, dims_cond, 2, 
                     CONDITION_FEATURES * D_MODEL};
    
    for (int i = 0; i < W_seq.size; i++) W_seq.data[i] = randn() * w_scale;
    for (int i = 0; i < W_cond.size; i++) W_cond.data[i] = randn() * w_scale;
    
    int attn_dims[] = {D_MODEL, D_MODEL};
    int ff_dims1[] = {D_MODEL, D_MODEL * 4};
    int ff_dims2[] = {D_MODEL * 4, D_MODEL};
    
    Tensor W_q[N_LAYERS], W_k[N_LAYERS], W_v[N_LAYERS], W_o[N_LAYERS];
    Tensor W_ff1[N_LAYERS], W_ff2[N_LAYERS];
    
    for (int l = 0; l < N_LAYERS; l++) {
        W_q[l] = (Tensor){malloc(D_MODEL * D_MODEL * sizeof(double)), NULL, attn_dims, 2, D_MODEL * D_MODEL};
        W_k[l] = (Tensor){malloc(D_MODEL * D_MODEL * sizeof(double)), NULL, attn_dims, 2, D_MODEL * D_MODEL};
        W_v[l] = (Tensor){malloc(D_MODEL * D_MODEL * sizeof(double)), NULL, attn_dims, 2, D_MODEL * D_MODEL};
        W_o[l] = (Tensor){malloc(D_MODEL * D_MODEL * sizeof(double)), NULL, attn_dims, 2, D_MODEL * D_MODEL};
        W_ff1[l] = (Tensor){malloc(D_MODEL * (D_MODEL * 4) * sizeof(double)), NULL, ff_dims1, 2, D_MODEL * (D_MODEL * 4)};
        W_ff2[l] = (Tensor){malloc((D_MODEL * 4) * D_MODEL * sizeof(double)), NULL, ff_dims2, 2, (D_MODEL * 4) * D_MODEL};
        
        for (int i = 0; i < D_MODEL * D_MODEL; i++) {
            W_q[l].data[i] = randn() * w_scale;
            W_k[l].data[i] = randn() * w_scale;
            W_v[l].data[i] = randn() * w_scale;
            W_o[l].data[i] = randn() * w_scale;
        }
        for (int i = 0; i < W_ff1[l].size; i++) W_ff1[l].data[i] = randn() * w_scale;
        for (int i = 0; i < W_ff2[l].size; i++) W_ff2[l].data[i] = randn() * w_scale;
    }
    
    Tensor hidden = {malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double)), NULL,
                    (int[]){BATCH_SIZE, SEQ_LENGTH, D_MODEL}, 3, BATCH_SIZE * SEQ_LENGTH * D_MODEL};
    Tensor temp = {malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double)), NULL,
                  (int[]){BATCH_SIZE, SEQ_LENGTH, D_MODEL}, 3, BATCH_SIZE * SEQ_LENGTH * D_MODEL};
    
    int out_dims[] = {BATCH_SIZE, SEQ_LENGTH, SEQUENCE_FEATURES};
    Tensor output = {malloc(BATCH_SIZE * SEQ_LENGTH * SEQUENCE_FEATURES * sizeof(double)), NULL,
                    out_dims, 3, BATCH_SIZE * SEQ_LENGTH * SEQUENCE_FEATURES};
    Tensor W_out = {malloc(D_MODEL * SEQUENCE_FEATURES * sizeof(double)), NULL,
                    (int[]){D_MODEL, SEQUENCE_FEATURES}, 2, D_MODEL * SEQUENCE_FEATURES};
    
    for (int i = 0; i < W_out.size; i++) W_out.data[i] = randn() * w_scale;
    
    // Train the model
    train_finite_diff(&dataset, &output, &hidden, &temp,
                     &W_seq, &W_cond, W_q, W_k, W_v, W_o,
                     W_ff1, W_ff2, &W_out);
    
    // Cleanup
    free(dataset.data); free(dataset.mins); free(dataset.maxs);
    free(hidden.data); free(temp.data); free(output.data); free(W_out.data);
    for (int l = 0; l < N_LAYERS; l++) {
        free(W_q[l].data); free(W_k[l].data); free(W_v[l].data); free(W_o[l].data);
        free(W_ff1[l].data); free(W_ff2[l].data);
    }
    free(W_seq.data); free(W_cond.data);
    return 0;
}