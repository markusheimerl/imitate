#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <omp.h>
#include "transformer.h"

// MSE Loss = mean((pred - target)^2) over sequence and features
// target is shifted by 1 in sequence dimension (predicting next vector in sequence)
double compute_loss(const double* out, const double* seq_data) {
    double loss = 0.0;
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* pred = out + s * OUTPUT_FEATURES;
        const double* target = seq_data + (s + 1) * (CONDITION_FEATURES + SEQUENCE_FEATURES) + CONDITION_FEATURES + ROTOR_OFFSET;
        for (int r = 0; r < OUTPUT_FEATURES; r++) {
            double error = pred[r] - target[r];
            loss += error * error;
        }
    }
    return loss / (SEQ_LENGTH * OUTPUT_FEATURES);
}

// dx = dy * scale - x * scale^3 * (sum(dy * x))/(2*d*ss)
// where scale = 1/sqrt(mean(x^2) + eps)
void rmsnorm_backward(double* d_x, const double* d_y, const double* x, const double* norm_x, int size) {
    const double inv_d = 1.0 / D_MODEL;
    #pragma omp parallel for
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* x_s = x + s * size;
        const double* d_y_s = d_y + s * size;
        double* d_x_s = d_x + s * size;
        double ss = 0.0, d_scale = 0.0;
        
        for (int i = 0; i < size; i++) ss += x_s[i] * x_s[i];
        double scale = 1.0 / sqrt(ss * inv_d + 1e-5);
        for (int i = 0; i < size; i++) d_scale += d_y_s[i] * x_s[i];
        for (int i = 0; i < size; i++) d_x_s[i] = d_y_s[i] * scale - x_s[i] * scale * scale * scale * inv_d * d_scale / (2.0 * size);
    }
}

// Backward pass through feedforward network:
// 1. d_w2 = d_y * G(U)^T, d_G(U) = d_y * W2^T
// 2. d_U = d_G(U) * G'(U) where G'(U) is GELU derivative
// 3. d_w1 = d_U * X^T, d_x = d_U * W1^T
void feedforward_backward(double* d_x, double* d_w1, double* d_w2, const double* d_y, const double* x, const double* w1, const double* w2, double* mid, double* d_mid) {
    #pragma omp parallel for
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* x_s = x + s * D_MODEL;
        const double* d_y_s = d_y + s * D_MODEL;
        double* d_x_s = d_x + s * D_MODEL;
        double* m_s = mid + s * (D_MODEL * 4);
        double* d_m_s = d_mid + s * (D_MODEL * 4);
        
        for (int h = 0; h < D_MODEL * 4; h++) {
            d_m_s[h] = 0.0;
            for (int d = 0; d < D_MODEL; d++) {
                d_m_s[h] += d_y_s[d] * w2[d * (D_MODEL * 4) + h];
                d_w2[d * (D_MODEL * 4) + h] += d_y_s[d] * m_s[h];
            }
        }
        
        for (int h = 0; h < D_MODEL * 4; h++) {
            double sum = 0.0;
            for (int d = 0; d < D_MODEL; d++) sum += x_s[d] * w1[h * D_MODEL + d];
            double t = sum + 0.044715 * sum * sum * sum;
            double tanh_t = tanh(sqrt(2.0/M_PI) * t);
            double d_gelu = d_m_s[h] * 0.5 * (1.0 + tanh_t + sum * sqrt(2.0/M_PI) * (1.0 - tanh_t * tanh_t) * (1.0 + 0.134145 * sum * sum));
            for (int d = 0; d < D_MODEL; d++) {
                d_w1[h * D_MODEL + d] += d_gelu * x_s[d];
                d_x_s[d] += d_gelu * w1[h * D_MODEL + d];
            }
        }
    }
}

// Backward pass through attention:
// 1. Output projection: d_v = d_y * Wo^T, d_Wo = d_y * v^T
// 2. Attention scores with ALiBi:
//    d_v += d_v * A, d_A = d_v * V^T
//    d_score = d_A * softmax'(score)
//    d_q = d_score * K/sqrt(d), d_k = d_score * Q/sqrt(d)
// 3. QKV projection:
//    d_Wq = d_q * X^T, d_Wk = d_k * X^T, d_Wv = d_v * X^T
//    d_x = d_q * Wq^T + d_k * Wk^T + d_v * Wv^T
void attention_backward(double* d_q, double* d_k, double* d_v, double* d_wq, double* d_wk, double* d_wv, 
                       double* d_wo, double* d_x, const double* d_out, const double* q, const double* k, 
                       const double* v, const double* s, const double* x, const double* wq, const double* wk, 
                       const double* wv, const double* wo) {
    const double scale = 1.0 / sqrt(D_MODEL);
    const double alibi_slope = 0.125;

    // Step 1: Output projection gradients
    #pragma omp parallel for
    for (int pos = 0; pos < SEQ_LENGTH; pos++) {
        double d_weighted[D_MODEL] = {0};
        for (int d = 0; d < D_MODEL; d++) {
            for (int i = 0; i < D_MODEL; i++) {
                d_weighted[i] += d_out[pos * D_MODEL + d] * wo[d * D_MODEL + i];
                d_wo[d * D_MODEL + i] += d_out[pos * D_MODEL + d] * v[pos * D_MODEL + i];
            }
        }

        // Step 2: Attention and value gradients
        for (int j = 0; j <= pos; j++) {
            // Gradient for values
            for (int d = 0; d < D_MODEL; d++) d_v[j * D_MODEL + d] += d_weighted[d] * s[pos * SEQ_LENGTH + j];

            // Gradient for attention scores (through softmax)
            double d_score = 0.0;
            for (int d = 0; d < D_MODEL; d++) {
                d_score += d_weighted[d] * v[j * D_MODEL + d];
            }
            
            // Softmax derivative and ALiBi
            const double ds = s[pos * SEQ_LENGTH + j] * ((j == pos) ? (1 - s[pos * SEQ_LENGTH + j]) : -s[pos * SEQ_LENGTH + j]) * (d_score * scale - alibi_slope * (pos - j));

            // Gradients for Q and K
            for (int d = 0; d < D_MODEL; d++) {
                d_q[pos * D_MODEL + d] += ds * k[j * D_MODEL + d];
                d_k[j * D_MODEL + d] += ds * q[pos * D_MODEL + d];
            }
        }
    }

    // Step 3: QKV projection gradients
    #pragma omp parallel for
    for (int pos = 0; pos < SEQ_LENGTH; pos++) {
        const int x_idx = pos * D_MODEL;
        for (int d = 0; d < D_MODEL; d++) {
            const int w_idx = d * D_MODEL;
            for (int i = 0; i < D_MODEL; i++) {
                d_wq[w_idx + i] += d_q[x_idx + d] * x[x_idx + i];
                d_wk[w_idx + i] += d_k[x_idx + d] * x[x_idx + i];
                d_wv[w_idx + i] += d_v[x_idx + d] * x[x_idx + i];
                d_x[x_idx + i] += d_q[x_idx + d] * wq[w_idx + i] + d_k[x_idx + d] * wk[w_idx + i] + d_v[x_idx + d] * wv[w_idx + i];
            }
        }
    }
}

// Backward pass through transformer:
// 1. Output projection gradients
// 2. For each layer from top to bottom:
//    d_x = d_x + ffn_backward(rmsnorm_backward(d_x))
//    d_x = d_x + attention_backward(rmsnorm_backward(d_x))
// 3. Input embedding gradients
void backward_pass(double* grads, const double* seq_data, const double* out, const double* hidden, 
                  const double* ws, const double* wc, const double* wq, const double* wk, const double* wv, 
                  const double* wo, const double* wf1, const double* wf2, const double* wout, 
                  double* d_hidden, double* d_temp, double* q_buf, double* k_buf, double* v_buf, 
                  double* s_buf, double* mid_buf, double* d_mid) {
    // Clear gradient buffers
    const size_t grad_size = (SEQUENCE_FEATURES + CONDITION_FEATURES) * D_MODEL + D_MODEL * OUTPUT_FEATURES + N_LAYERS * (4 * D_MODEL * D_MODEL + D_MODEL * (D_MODEL * 4) + (D_MODEL * 4) * D_MODEL);
    memset(grads, 0, grad_size * sizeof(double));
    memset(d_hidden, 0, SEQ_LENGTH * D_MODEL * sizeof(double));
    
    // Step 1: Output projection gradients
    for (int pos = 0; pos < SEQ_LENGTH; pos++) {
        const double* pred = out + pos * OUTPUT_FEATURES;
        const double* target = seq_data + (pos + 1) * (CONDITION_FEATURES + SEQUENCE_FEATURES) + CONDITION_FEATURES + ROTOR_OFFSET;
        for (int f = 0; f < OUTPUT_FEATURES; f++) {
            const double d_out = 2.0 * (pred[f] - target[f]) / (SEQ_LENGTH * OUTPUT_FEATURES);
            const size_t wout_offset = SEQUENCE_FEATURES * D_MODEL + CONDITION_FEATURES * D_MODEL + f * D_MODEL;
            for (int d = 0; d < D_MODEL; d++) {
                grads[wout_offset + d] += d_out * hidden[pos * D_MODEL + d];
                d_hidden[pos * D_MODEL + d] += d_out * wout[f * D_MODEL + d];
            }
        }
    }
    
    // Step 2: Layer gradients from top to bottom
    size_t offset = (SEQUENCE_FEATURES + CONDITION_FEATURES) * D_MODEL + D_MODEL * OUTPUT_FEATURES;
    for (int l = N_LAYERS - 1; l >= 0; l--) {
        const size_t layer_offset = l * (4 * D_MODEL * D_MODEL + D_MODEL * (D_MODEL * 4) + (D_MODEL * 4) * D_MODEL);
        
        // Feedforward backward
        memcpy(d_temp, d_hidden, SEQ_LENGTH * D_MODEL * sizeof(double));
        feedforward_backward(d_hidden, grads + offset + layer_offset + 4 * D_MODEL * D_MODEL, grads + offset + layer_offset + 4 * D_MODEL * D_MODEL + D_MODEL * (D_MODEL * 4), d_temp, hidden, wf1 + l * D_MODEL * (D_MODEL * 4), wf2 + l * (D_MODEL * 4) * D_MODEL, mid_buf, d_mid);
        
        // RMSNorm backward
        rmsnorm_backward(d_temp, d_hidden, hidden, hidden, D_MODEL);
        memcpy(d_hidden, d_temp, SEQ_LENGTH * D_MODEL * sizeof(double));
        
        // Attention backward
        attention_backward(q_buf, k_buf, v_buf,grads + offset + layer_offset, grads + offset + layer_offset + D_MODEL * D_MODEL, grads + offset + layer_offset + 2 * D_MODEL * D_MODEL, grads + offset + layer_offset + 3 * D_MODEL * D_MODEL, d_temp, d_hidden, q_buf, k_buf, v_buf, s_buf, hidden, wq + l * D_MODEL * D_MODEL, wk + l * D_MODEL * D_MODEL, wv + l * D_MODEL * D_MODEL, wo + l * D_MODEL * D_MODEL);
        rmsnorm_backward(d_hidden, d_temp, hidden, hidden, D_MODEL);
    }
    
    // Step 3: Input embedding gradients
    for (int pos = 0; pos < SEQ_LENGTH; pos++) {
        const double* x = seq_data + pos * (CONDITION_FEATURES + SEQUENCE_FEATURES);
        const double* d_h = d_hidden + pos * D_MODEL;
        for (int d = 0; d < D_MODEL; d++) {
            for (int f = 0; f < SEQUENCE_FEATURES; f++) {
                grads[f * D_MODEL + d] += d_h[d] * x[f + CONDITION_FEATURES];
            }
            for (int f = 0; f < CONDITION_FEATURES; f++) {
                grads[SEQUENCE_FEATURES * D_MODEL + f * D_MODEL + d] += d_h[d] * x[f];
            }
        }
    }
}

// Train transformer using backpropagation and Adam optimizer:
// 1. Forward pass to compute loss
// 2. Backward pass to compute gradients
// 3. Update weights using Adam:
//    m = beta1 * m + (1-beta1) * g
//    v = beta2 * v + (1-beta2) * g^2
//    w = w - lr * m_hat / (sqrt(v_hat) + eps)
void train_backprop(Dataset* ds, double* ws, double* wc, double* wq, double* wk, double* wv, double* wo, double* wf1, double* wf2, double* wout) {
    double lr = LEARNING_RATE, prev_loss = 1e9;
    
    // Setup logging
    time_t t = time(NULL);
    struct tm *tm = localtime(&t);
    char loss_name[100];
    sprintf(loss_name, "%d-%d-%d_%d-%d-%d_loss.csv", tm->tm_year+1900, tm->tm_mon+1, tm->tm_mday, tm->tm_hour, tm->tm_min, tm->tm_sec);
    FILE* f = fopen(loss_name, "w");
    if (f) fprintf(f, "step,loss\n");

    // Allocate buffers
    const size_t grad_size = (SEQUENCE_FEATURES + CONDITION_FEATURES) * D_MODEL + D_MODEL * OUTPUT_FEATURES + N_LAYERS * (4 * D_MODEL * D_MODEL + D_MODEL * (D_MODEL * 4) + (D_MODEL * 4) * D_MODEL);
    
    // Training buffers
    double *seq_data = malloc((SEQ_LENGTH + 1) * (CONDITION_FEATURES + SEQUENCE_FEATURES) * sizeof(double));
    double *grads = malloc(grad_size * sizeof(double));
    double *hidden = malloc(SEQ_LENGTH * D_MODEL * sizeof(double));
    double *temp = malloc(SEQ_LENGTH * D_MODEL * sizeof(double));
    double *output = malloc(SEQ_LENGTH * OUTPUT_FEATURES * sizeof(double));
    double *d_hidden = malloc(SEQ_LENGTH * D_MODEL * sizeof(double));
    double *d_temp = malloc(SEQ_LENGTH * D_MODEL * sizeof(double));
    double *q_buf = malloc(SEQ_LENGTH * D_MODEL * sizeof(double));
    double *k_buf = malloc(SEQ_LENGTH * D_MODEL * sizeof(double));
    double *v_buf = malloc(SEQ_LENGTH * D_MODEL * sizeof(double));
    double *s_buf = malloc(SEQ_LENGTH * SEQ_LENGTH * sizeof(double));
    double *mid_buf = malloc(SEQ_LENGTH * (D_MODEL * 4) * sizeof(double));
    double *d_mid = malloc(SEQ_LENGTH * (D_MODEL * 4) * sizeof(double));

    // Adam momentum buffers
    double *ws_m = calloc(SEQUENCE_FEATURES * D_MODEL, sizeof(double));
    double *wc_m = calloc(CONDITION_FEATURES * D_MODEL, sizeof(double));
    double *wout_m = calloc(D_MODEL * OUTPUT_FEATURES, sizeof(double));
    double *wq_m = calloc(N_LAYERS * D_MODEL * D_MODEL, sizeof(double));
    double *wk_m = calloc(N_LAYERS * D_MODEL * D_MODEL, sizeof(double));
    double *wv_m = calloc(N_LAYERS * D_MODEL * D_MODEL, sizeof(double));
    double *wo_m = calloc(N_LAYERS * D_MODEL * D_MODEL, sizeof(double));
    double *wf1_m = calloc(N_LAYERS * D_MODEL * (D_MODEL * 4), sizeof(double));
    double *wf2_m = calloc(N_LAYERS * (D_MODEL * 4) * D_MODEL, sizeof(double));
    
    // Adam variance buffers
    double *ws_v = calloc(SEQUENCE_FEATURES * D_MODEL, sizeof(double));
    double *wc_v = calloc(CONDITION_FEATURES * D_MODEL, sizeof(double));
    double *wout_v = calloc(D_MODEL * OUTPUT_FEATURES, sizeof(double));
    double *wq_v = calloc(N_LAYERS * D_MODEL * D_MODEL, sizeof(double));
    double *wk_v = calloc(N_LAYERS * D_MODEL * D_MODEL, sizeof(double));
    double *wv_v = calloc(N_LAYERS * D_MODEL * D_MODEL, sizeof(double));
    double *wo_v = calloc(N_LAYERS * D_MODEL * D_MODEL, sizeof(double));
    double *wf1_v = calloc(N_LAYERS * D_MODEL * (D_MODEL * 4), sizeof(double));
    double *wf2_v = calloc(N_LAYERS * (D_MODEL * 4) * D_MODEL, sizeof(double));

    const double beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.01;

    // Training loop
    for (int step = 0; step < TRAINING_STEPS; step++) {
        // Sample random sequence
        const size_t seq_offset = (rand() % (ds->rows - SEQ_LENGTH - 1)) * (CONDITION_FEATURES + SEQUENCE_FEATURES);
        memcpy(seq_data, ds->data + seq_offset, (SEQ_LENGTH + 1) * (CONDITION_FEATURES + SEQUENCE_FEATURES) * sizeof(double));

        // Forward pass and loss computation
        forward_pass(seq_data, output, hidden, temp, ws, wc, wq, wk, wv, wo, wf1, wf2, wout, q_buf, k_buf, v_buf, s_buf, mid_buf);
        double loss = compute_loss(output, seq_data);
        
        // Adaptive learning rate
        if (step > 0) lr = fmax(1e-9, fmin(1e-3, lr * ((loss > prev_loss * 1.1) ? 0.95 : (loss < prev_loss * 0.95) ? 1.05 : 1.0)));
        prev_loss = loss;
        
        // Logging
        if (step % 1000 == 0) {
            printf("Step %d, Loss: %f, LR: %e\n", step, loss, lr);
            const int last_seq = SEQ_LENGTH - 1;
            const double* pred = output + last_seq * OUTPUT_FEATURES;
            const double* target = seq_data + (last_seq + 1) * (CONDITION_FEATURES + SEQUENCE_FEATURES) + CONDITION_FEATURES + ROTOR_OFFSET;
            printf("Desired velocity: [%.3f, %.3f, %.3f]\n", seq_data[last_seq * (CONDITION_FEATURES + SEQUENCE_FEATURES) + 0], seq_data[last_seq * (CONDITION_FEATURES + SEQUENCE_FEATURES) + 1], seq_data[last_seq * (CONDITION_FEATURES + SEQUENCE_FEATURES) + 2]);
            printf("Rotor speeds - Pred: [%.3f, %.3f, %.3f, %.3f] Ground Truth: [%.3f, %.3f, %.3f, %.3f]\n\n", pred[0], pred[1], pred[2], pred[3], target[0], target[1], target[2], target[3]);
        }
        if (f) fprintf(f, "%d,%f\n", step, loss);

        // Backward pass
        backward_pass(grads, seq_data, output, hidden, ws, wc, wq, wk, wv, wo, wf1, wf2, wout, d_hidden, d_temp, q_buf, k_buf, v_buf, s_buf, mid_buf, d_mid);
        
        // Update embedding and output weights
        size_t offset = 0;
        const size_t sizes[] = {SEQUENCE_FEATURES * D_MODEL, CONDITION_FEATURES * D_MODEL, OUTPUT_FEATURES * D_MODEL};
        double *weights[] = {ws, wc, wout}, *m[] = {ws_m, wc_m, wout_m}, *v[] = {ws_v, wc_v, wout_v};
        
        for (int w = 0; w < 3; w++) {
            for (int i = 0; i < sizes[w]; i++) {
                double grad = grads[offset + i] + weight_decay * weights[w][i];
                m[w][i] = beta1 * m[w][i] + (1 - beta1) * grad;
                v[w][i] = beta2 * v[w][i] + (1 - beta2) * grad * grad;
                double m_hat = m[w][i] / (1.0 - pow(beta1, step + 1));
                double v_hat = v[w][i] / (1.0 - pow(beta2, step + 1));
                weights[w][i] -= lr * m_hat / (sqrt(v_hat) + eps);
            }
            offset += sizes[w];
        }
        
        // Update layer weights
        for (int l = 0; l < N_LAYERS; l++) {
            const size_t layer_sizes[] = {D_MODEL * D_MODEL, D_MODEL * D_MODEL, D_MODEL * D_MODEL, D_MODEL * D_MODEL, D_MODEL * (D_MODEL * 4), (D_MODEL * 4) * D_MODEL};
            double *layer_weights[] = {wq + l * layer_sizes[0], wk + l * layer_sizes[1], wv + l * layer_sizes[2], wo + l * layer_sizes[3], wf1 + l * layer_sizes[4], wf2 + l * layer_sizes[5]};
            double *layer_m[] = {wq_m + l * layer_sizes[0], wk_m + l * layer_sizes[1], wv_m + l * layer_sizes[2], wo_m + l * layer_sizes[3], wf1_m + l * layer_sizes[4], wf2_m + l * layer_sizes[5]};
            double *layer_v[] = {wq_v + l * layer_sizes[0], wk_v + l * layer_sizes[1], wv_v + l * layer_sizes[2], wo_v + l * layer_sizes[3], wf1_v + l * layer_sizes[4], wf2_v + l * layer_sizes[5]};
            
            for (int w = 0; w < 6; w++) {
                for (int i = 0; i < layer_sizes[w]; i++) {
                    double grad = grads[offset + i] + weight_decay * layer_weights[w][i];
                    layer_m[w][i] = beta1 * layer_m[w][i] + (1 - beta1) * grad;
                    layer_v[w][i] = beta2 * layer_v[w][i] + (1 - beta2) * grad * grad;
                    double m_hat = layer_m[w][i] / (1.0 - pow(beta1, step + 1));
                    double v_hat = layer_v[w][i] / (1.0 - pow(beta2, step + 1));
                    layer_weights[w][i] -= lr * m_hat / (sqrt(v_hat) + eps);
                }
                offset += layer_sizes[w];
            }
        }
    }

    // Cleanup
    free(seq_data); free(grads); free(hidden); free(temp); free(output);
    free(d_hidden); free(d_temp); free(q_buf); free(k_buf); free(v_buf);
    free(s_buf); free(mid_buf); free(d_mid);
    free(ws_m); free(ws_v); free(wc_m); free(wc_v);
    free(wout_m); free(wout_v); free(wq_m); free(wq_v);
    free(wk_m); free(wk_v); free(wv_m); free(wv_v);
    free(wo_m); free(wo_v); free(wf1_m); free(wf1_v);
    free(wf2_m); free(wf2_v);
    if (f) fclose(f);
}

int main(int argc, char *argv[]) {
    // Parse command line arguments
    if (argc < 2 || argc > 3 || !strstr(argv[1], ".csv")) {printf("Usage: %s <training_data.csv> [weights.bin]\n", argv[0]); return 1;}
    char *weights_file = argc > 2 ? (strstr(argv[2], ".bin") ? argv[2] : NULL) : NULL;
    if (argc > 2 && !weights_file) {printf("Error: Invalid file '%s'\n", argv[2]); return 1;}

    // Initialize random seed and load data
    srand(time(NULL));
    Dataset ds = load_csv(argv[1]);

    // Allocate model weights
    double *W_seq = malloc(SEQUENCE_FEATURES * D_MODEL * sizeof(double));
    double *W_cond = malloc(CONDITION_FEATURES * D_MODEL * sizeof(double));
    double *W_out = malloc(D_MODEL * OUTPUT_FEATURES * sizeof(double));
    double *W_q = malloc(N_LAYERS * D_MODEL * D_MODEL * sizeof(double));
    double *W_k = malloc(N_LAYERS * D_MODEL * D_MODEL * sizeof(double));
    double *W_v = malloc(N_LAYERS * D_MODEL * D_MODEL * sizeof(double));
    double *W_o = malloc(N_LAYERS * D_MODEL * D_MODEL * sizeof(double));
    double *W_ff1 = malloc(N_LAYERS * D_MODEL * (D_MODEL * 4) * sizeof(double));
    double *W_ff2 = malloc(N_LAYERS * (D_MODEL * 4) * D_MODEL * sizeof(double));

    // Initialize or load weights
    if (!weights_file || !load_weights(weights_file, W_seq, W_cond, W_q, W_k, W_v, W_o, W_ff1, W_ff2, W_out)) {
        if (weights_file) printf("Failed to load weights, initializing randomly\n");
        
        // Initialize embedding and output weights
        for (int i = 0; i < SEQUENCE_FEATURES * D_MODEL; i++) W_seq[i] = randn() * sqrt(2.0 / D_MODEL);
        for (int i = 0; i < CONDITION_FEATURES * D_MODEL; i++) W_cond[i] = randn() * sqrt(2.0 / D_MODEL);
        for (int i = 0; i < D_MODEL * OUTPUT_FEATURES; i++) W_out[i] = randn() * sqrt(2.0 / D_MODEL);
        
        // Initialize attention weights
        for (int i = 0; i < N_LAYERS * D_MODEL * D_MODEL; i++) {
            W_q[i] = randn() * sqrt(2.0 / D_MODEL);
            W_k[i] = randn() * sqrt(2.0 / D_MODEL);
            W_v[i] = randn() * sqrt(2.0 / D_MODEL);
            W_o[i] = randn() * sqrt(2.0 / D_MODEL);
        }
        
        // Initialize feedforward weights
        for (int i = 0; i < N_LAYERS * D_MODEL * (D_MODEL * 4); i++) W_ff1[i] = randn() * sqrt(2.0 / D_MODEL);
        for (int i = 0; i < N_LAYERS * (D_MODEL * 4) * D_MODEL; i++) W_ff2[i] = randn() * sqrt(2.0 / D_MODEL);
    } else {
        printf("Successfully loaded weights\n");
    }

    // Train model
    train_backprop(&ds, W_seq, W_cond, W_q, W_k, W_v, W_o, W_ff1, W_ff2, W_out);
    
    // Save trained weights
    save_weights(W_seq, W_cond, W_q, W_k, W_v, W_o, W_ff1, W_ff2, W_out);

    // Cleanup
    free(ds.data);
    free(W_seq); free(W_cond); free(W_out);
    free(W_q); free(W_k); free(W_v); free(W_o);
    free(W_ff1); free(W_ff2);
    
    return 0;
}