#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <omp.h>

#define CONDITION_FEATURES 4
#define SEQUENCE_FEATURES 10
#define INPUT_FEATURES (CONDITION_FEATURES + SEQUENCE_FEATURES)
#define SEQ_LENGTH 64
#define D_MODEL 64
#define N_HEAD 4
#define N_LAYERS 4
#define LEARNING_RATE 0.00001
#define TRAINING_STEPS 10000

typedef struct { double *data; int rows, cols; } Dataset;
typedef struct { double *data; double *m, *v; int size; } Tensor;

double randn() { return sqrt(-2.0 * log((double)rand() / RAND_MAX)) * cos(2.0 * M_PI * (double)rand() / RAND_MAX); }

Dataset load_csv(const char* filename) {
    Dataset ds = {malloc(1000 * INPUT_FEATURES * sizeof(double)), 0, INPUT_FEATURES};
    char line[1024]; FILE* f = fopen(filename, "r");
    if (!f || !fgets(line, 1024, f)) { printf("File error\n"); exit(1); }
    
    while (fgets(line, 1024, f)) {
        if (ds.rows >= 1000) ds.data = realloc(ds.data, (ds.rows*2) * INPUT_FEATURES * sizeof(double));
        char* tok = strtok(line, ",");
        for (int i = 0; i < INPUT_FEATURES && tok; i++, tok = strtok(NULL, ",")) ds.data[ds.rows * INPUT_FEATURES + i] = atof(tok);
        ds.rows++;
    }
    fclose(f);
    return ds;
}

void save_weights(const char* filename, const Tensor* ws, const Tensor* wc,
                 const Tensor* wq, const Tensor* wk, const Tensor* wv, const Tensor* wo,
                 const Tensor* wf1, const Tensor* wf2, const Tensor* wout) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;
    
    fwrite(ws->data, sizeof(double), ws->size, f);
    fwrite(wc->data, sizeof(double), wc->size, f);
    for (int l = 0; l < N_LAYERS; l++)
        fwrite(wq[l].data, sizeof(double), wq[l].size, f),
        fwrite(wk[l].data, sizeof(double), wk[l].size, f),
        fwrite(wv[l].data, sizeof(double), wv[l].size, f),
        fwrite(wo[l].data, sizeof(double), wo[l].size, f),
        fwrite(wf1[l].data, sizeof(double), wf1[l].size, f),
        fwrite(wf2[l].data, sizeof(double), wf2[l].size, f);
    fwrite(wout->data, sizeof(double), wout->size, f);
    fclose(f);
}

int load_weights(const char* filename, Tensor* ws, Tensor* wc,
                Tensor* wq, Tensor* wk, Tensor* wv, Tensor* wo,
                Tensor* wf1, Tensor* wf2, Tensor* wout) {
    FILE* f = fopen(filename, "rb");
    if (!f) return 0;
    
    size_t read = fread(ws->data, sizeof(double), ws->size, f), expected = ws->size + wc->size + wout->size;
    read += fread(wc->data, sizeof(double), wc->size, f);
    
    for (int l = 0; l < N_LAYERS; l++)
        read += fread(wq[l].data, sizeof(double), wq[l].size, f),
        read += fread(wk[l].data, sizeof(double), wk[l].size, f),
        read += fread(wv[l].data, sizeof(double), wv[l].size, f),
        read += fread(wo[l].data, sizeof(double), wo[l].size, f),
        read += fread(wf1[l].data, sizeof(double), wf1[l].size, f),
        read += fread(wf2[l].data, sizeof(double), wf2[l].size, f),
        expected += wq[l].size + wk[l].size + wv[l].size + wo[l].size + wf1[l].size + wf2[l].size;
    
    read += fread(wout->data, sizeof(double), wout->size, f);
    fclose(f);
    return read == expected;
}

// Given input X of shape [seq_len, d_model]
// RMSNorm(X)_s,d = X_s,d / sqrt(1/D * sum_i(X_s,i^2) + eps)
// where s=sequence index, d=dimension index
void rmsnorm(Tensor *out, const Tensor *in) {
    #pragma omp parallel for
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* x = in->data + s * D_MODEL;
        double* y = out->data + s * D_MODEL;
        double ss = 0.0;
        for (int i = 0; i < D_MODEL; i++) ss += x[i] * x[i];
        double scale = 1.0 / sqrt(ss / D_MODEL + 1e-5);
        for (int i = 0; i < D_MODEL; i++) y[i] = x[i] * scale;
    }
}

// Given input X of shape [seq_len, d_model]
// For each position in sequence:
// 1. Linear: U = X*W1 where X:[1,d_model], W1:[d_model,4*d_model] -> U:[1,4*d_model]
// 2. GELU (elementwise): G(U) = 0.5 * U * (1 + tanh(sqrt(2/pi) * (U + 0.044715 * U^3)))
// 3. Linear: Y = G(U)*W2 where G(U):[1,4*d_model], W2:[4*d_model,d_model] -> Y:[1,d_model]
void feedforward(Tensor *out, const Tensor *w1, const Tensor *w2, const Tensor *in, double *mid) {
    #pragma omp parallel for
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* x = in->data + s * D_MODEL;
        double* y = out->data + s * D_MODEL;
        double* u = mid + s * (D_MODEL * 4);
        
        for (int i = 0; i < D_MODEL * 4; i++) {
            u[i] = 0.0;
            for (int j = 0; j < D_MODEL; j++) u[i] += x[j] * w1->data[i * D_MODEL + j];
            u[i] = 0.5 * u[i] * (1.0 + tanh(sqrt(2.0/M_PI) * u[i] + 0.044715 * u[i] * u[i] * u[i]));
        }
        
        for (int i = 0; i < D_MODEL; i++) {
            y[i] = 0.0;
            for (int j = 0; j < D_MODEL * 4; j++) y[i] += u[j] * w2->data[i * (D_MODEL * 4) + j];
        }
    }
}

// Given input X of shape [seq_len, d_model]
// 1. QKV projection for each head h:
//    Q_h = X * Wq_h, K_h = X * Wk_h, V_h = X * Wv_h
// 2. Scaled dot-product attention with ALiBi bias per head:
//    score = (Q_h * K_h^T)/sqrt(d_head) - ALiBi_slope_h * distance_matrix
//    A_h = softmax(score) * V_h  where softmax is causal (upper triangle masked)
// 3. Concatenate heads and project:
//    MultiHead(X) = concat(A_1,...,A_h) * Wo
void multihead_attention(Tensor *out, const Tensor *in, const Tensor *wq, const Tensor *wk, 
                        const Tensor *wv, const Tensor *wo,
                        double *q, double *k, double *v, double *s) {
    const int hd = D_MODEL / N_HEAD;
    const double scale = 1.0 / sqrt(hd);

    // QKV Transform
    #pragma omp parallel for
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* x = in->data + s * D_MODEL;
        for (int h = 0; h < N_HEAD; h++)
            for (int d = 0; d < hd; d++) {
                double sq = 0.0, sk = 0.0, sv = 0.0;
                const int w_idx = (h * hd + d) * D_MODEL;
                for (int i = 0; i < D_MODEL; i++) 
                    sq += x[i] * wq->data[w_idx + i],
                    sk += x[i] * wk->data[w_idx + i],
                    sv += x[i] * wv->data[w_idx + i];
                const int qkv_idx = s * D_MODEL + h * hd + d;
                q[qkv_idx] = sq, k[qkv_idx] = sk, v[qkv_idx] = sv;
            }
    }

    // Attention with alibi mask
    #pragma omp parallel for
    for (int h = 0; h < N_HEAD; h++) {
        const double slope = pow(2.0, -(8.0 * (h + 1) / N_HEAD));
        for (int i = 0; i < SEQ_LENGTH; i++) {
            double max = -1e9, sum = 0.0;
            for (int j = 0; j <= i; j++) {
                double dot = 0.0;
                for (int d = 0; d < hd; d++) 
                    dot += q[i * D_MODEL + h * hd + d] * k[j * D_MODEL + h * hd + d];
                s[(h * SEQ_LENGTH + i) * SEQ_LENGTH + j] = dot * scale - slope * (i - j);
                max = fmax(max, s[(h * SEQ_LENGTH + i) * SEQ_LENGTH + j]);
            }
            for (int j = 0; j <= i; j++) {
                s[(h * SEQ_LENGTH + i) * SEQ_LENGTH + j] = exp(s[(h * SEQ_LENGTH + i) * SEQ_LENGTH + j] - max);
                sum += s[(h * SEQ_LENGTH + i) * SEQ_LENGTH + j];
            }
            for (int j = 0; j <= i; j++) 
                s[(h * SEQ_LENGTH + i) * SEQ_LENGTH + j] /= (sum + 1e-10);
        }
    }

    // Output projection
    #pragma omp parallel for
    for (int t = 0; t < SEQ_LENGTH; t++) {
        double tmp[D_MODEL] = {0};
        for (int h = 0; h < N_HEAD; h++)
            for (int d = 0; d < hd; d++) {
                double sum = 0.0;
                for (int j = 0; j <= t; j++) 
                    sum += s[(h * SEQ_LENGTH + t) * SEQ_LENGTH + j] * 
                          v[j * D_MODEL + h * hd + d];
                tmp[h * hd + d] = sum;
            }
        for (int d = 0; d < D_MODEL; d++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++) 
                sum += tmp[i] * wo->data[d * D_MODEL + i];
            out->data[t * D_MODEL + d] = sum;
        }
    }
}

// Forward pass through the transformer:
// 1. Input embedding: sequence_features * Ws + condition_features * Wc
// 2. N transformer layers of:
//    x = x + attention(rmsnorm(x))
//    x = x + ffn(rmsnorm(x))
// 3. Output projection to sequence_features
void forward_pass(const double* seq_data, Tensor* out, Tensor* hidden, Tensor* temp,
                 const Tensor* ws, const Tensor* wc, const Tensor* wq, const Tensor* wk, 
                 const Tensor* wv, const Tensor* wo, const Tensor* wf1, const Tensor* wf2, 
                 const Tensor* wout,
                 double* q_buf, double* k_buf, double* v_buf, double* s_buf, double* mid_buf) {
    
    #pragma omp parallel for
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* x = seq_data + s * INPUT_FEATURES;
        double* y = hidden->data + s * D_MODEL;
        for (int d = 0; d < D_MODEL; d++) {
            double sum = 0.0;
            for (int f = 0; f < SEQUENCE_FEATURES; f++) 
                sum += x[f + CONDITION_FEATURES] * ws->data[f * D_MODEL + d];
            for (int f = 0; f < CONDITION_FEATURES; f++) 
                sum += x[f] * wc->data[f * D_MODEL + d];
            y[d] = sum;
        }
    }

    for (int l = 0; l < N_LAYERS; l++) {
        rmsnorm(temp, hidden);
        multihead_attention(temp, temp, &wq[l], &wk[l], &wv[l], &wo[l], 
                          q_buf, k_buf, v_buf, s_buf);
        for (int i = 0; i < hidden->size; i++) 
            hidden->data[i] += temp->data[i];
        
        rmsnorm(temp, hidden);
        feedforward(temp, &wf1[l], &wf2[l], temp, mid_buf);
        for (int i = 0; i < hidden->size; i++) 
            hidden->data[i] += temp->data[i];
    }

    #pragma omp parallel for
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* h = hidden->data + s * D_MODEL;
        double* o = out->data + s * SEQUENCE_FEATURES;
        for (int f = 0; f < SEQUENCE_FEATURES; f++) {
            double sum = 0.0;
            for (int d = 0; d < D_MODEL; d++) 
                sum += h[d] * wout->data[f * D_MODEL + d];
            o[f] = sum;
        }
    }
}

// MSE Loss = mean((pred - target)^2) over sequence and features
// target is shifted by 1 in sequence dimension (predicting next vector in sequence)
double compute_loss(const Tensor* out, const double* seq_data) {
    double loss = 0.0;
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* pred = out->data + s * SEQUENCE_FEATURES;
        const double* target = seq_data + (s + 1) * INPUT_FEATURES + CONDITION_FEATURES;
        for (int f = 0; f < SEQUENCE_FEATURES; f++) 
            loss += (pred[f] - target[f]) * (pred[f] - target[f]);
    }
    return loss / (SEQ_LENGTH * SEQUENCE_FEATURES);
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
        for (int i = 0; i < size; i++) 
            d_x_s[i] = d_y_s[i] * scale - x_s[i] * scale * scale * scale * inv_d * d_scale / (2.0 * size);
    }
}

void feedforward_backward(double* d_x, double* d_w1, double* d_w2, const double* d_y, const double* x,
                         const double* w1, const double* w2, double* mid, double* d_mid) {
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
            for (int d = 0; d < D_MODEL; d++) 
                sum += x_s[d] * w1[h * D_MODEL + d];
            double t = sum + 0.044715 * sum * sum * sum;
            double tanh_t = tanh(sqrt(2.0/M_PI) * t);
            double d_gelu = d_m_s[h] * 0.5 * (1.0 + tanh_t + sum * sqrt(2.0/M_PI) * 
                           (1.0 - tanh_t * tanh_t) * (1.0 + 0.134145 * sum * sum));
            for (int d = 0; d < D_MODEL; d++) {
                d_w1[h * D_MODEL + d] += d_gelu * x_s[d];
                d_x_s[d] += d_gelu * w1[h * D_MODEL + d];
            }
        }
    }
}

void attention_backward(double* d_q, double* d_k, double* d_v, double* d_wq, double* d_wk, 
                       double* d_wv, double* d_wo, double* d_x, const double* d_out, 
                       const double* q, const double* k, const double* v, const double* s, 
                       const double* x, const double* wq, const double* wk, 
                       const double* wv, const double* wo) {
    const int hd = D_MODEL / N_HEAD;
    const double scale = 1.0 / sqrt(hd);
    
    // Output projection backward
    #pragma omp parallel for
    for (int t = 0; t < SEQ_LENGTH; t++) {
        double d_tmp[D_MODEL] = {0};
        const int out_idx = t * D_MODEL;
        for (int d = 0; d < D_MODEL; d++)
            for (int i = 0; i < D_MODEL; i++) {
                d_tmp[i] += d_out[out_idx + d] * wo[d * D_MODEL + i];
                d_wo[d * D_MODEL + i] += d_out[out_idx + d] * v[out_idx + i];
            }
        
        // Attention + ALiBi backward
        for (int h = 0; h < N_HEAD; h++) {
            const double slope = pow(2.0, -(8.0 * (h + 1) / N_HEAD));
            const int s_off = (h * SEQ_LENGTH + t) * SEQ_LENGTH;
            
            for (int j = 0; j <= t; j++) {
                double d_score = 0.0;
                for (int d = 0; d < hd; d++) {
                    const int qt_idx = t * D_MODEL + h * hd + d;
                    const int qj_idx = j * D_MODEL + h * hd + d;
                    d_score += d_tmp[h * hd + d] * v[qj_idx];
                    d_v[qj_idx] += d_tmp[h * hd + d] * s[s_off + j];
                }
                
                d_score = (d_score - slope * (t - j)) * scale;
                const double d_s = s[s_off + j] * ((j == t) ? (1 - s[s_off + j]) : -s[s_off + j]) * d_score;
                
                for (int d = 0; d < hd; d++) {
                    const int qt_idx = t * D_MODEL + h * hd + d;
                    const int qj_idx = j * D_MODEL + h * hd + d;
                    d_q[qt_idx] += d_s * k[qj_idx];
                    d_k[qj_idx] += d_s * q[qt_idx];
                }
            }
        }
    }

    // QKV projection backward
    #pragma omp parallel for
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const int x_idx = s * D_MODEL;
        for (int h = 0; h < N_HEAD; h++)
            for (int d = 0; d < hd; d++) {
                const int w_idx = (h * hd + d) * D_MODEL;
                const int qkv_idx = x_idx + h * hd + d;
                for (int i = 0; i < D_MODEL; i++) {
                    d_wq[w_idx + i] += d_q[qkv_idx] * x[x_idx + i];
                    d_wk[w_idx + i] += d_k[qkv_idx] * x[x_idx + i];
                    d_wv[w_idx + i] += d_v[qkv_idx] * x[x_idx + i];
                    d_x[x_idx + i] += d_q[qkv_idx] * wq[w_idx + i] + 
                                     d_k[qkv_idx] * wk[w_idx + i] + 
                                     d_v[qkv_idx] * wv[w_idx + i];
                }
            }
    }
}

void backward_pass(double* grads, const double* seq_data, const Tensor* out, const Tensor* hidden,
                  const Tensor* ws, const Tensor* wc, const Tensor* wq, const Tensor* wk, 
                  const Tensor* wv, const Tensor* wo, const Tensor* wf1, const Tensor* wf2, 
                  const Tensor* wout, double* d_hidden, double* d_temp, double* q_buf, 
                  double* k_buf, double* v_buf, double* s_buf, double* mid_buf, double* d_mid) {
    
    memset(grads, 0, (ws->size + wc->size + wout->size + 
           N_LAYERS * (wq[0].size + wk[0].size + wv[0].size + 
           wo[0].size + wf1[0].size + wf2[0].size)) * sizeof(double));
    memset(d_hidden, 0, SEQ_LENGTH * D_MODEL * sizeof(double));
    
    // Output layer backward
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* pred = out->data + s * SEQUENCE_FEATURES;
        const double* target = seq_data + (s + 1) * INPUT_FEATURES + CONDITION_FEATURES;
        for (int f = 0; f < SEQUENCE_FEATURES; f++) {
            const double d_out = 2.0 * (pred[f] - target[f]) / (SEQ_LENGTH * SEQUENCE_FEATURES);
            for (int d = 0; d < D_MODEL; d++) {
                grads[ws->size + wc->size + f * D_MODEL + d] += 
                    d_out * hidden->data[s * D_MODEL + d];
                d_hidden[s * D_MODEL + d] += d_out * wout->data[f * D_MODEL + d];
            }
        }
    }
    
    double* layer_grads = grads + ws->size + wc->size + wout->size;
    for (int l = N_LAYERS - 1; l >= 0; l--) {
        const size_t l_off = l * (wq[0].size + wk[0].size + wv[0].size + 
                                 wo[0].size + wf1[0].size + wf2[0].size);
        
        memcpy(d_temp, d_hidden, SEQ_LENGTH * D_MODEL * sizeof(double));
        feedforward_backward(d_hidden, 
                           layer_grads + l_off + wq[0].size + wk[0].size + wv[0].size + wo[0].size,
                           layer_grads + l_off + wq[0].size + wk[0].size + wv[0].size + wo[0].size + wf1[0].size,
                           d_temp, hidden->data, wf1[l].data, wf2[l].data, mid_buf, d_mid);
        
        rmsnorm_backward(d_temp, d_hidden, hidden->data, hidden->data, D_MODEL);
        memcpy(d_hidden, d_temp, SEQ_LENGTH * D_MODEL * sizeof(double));
        
        attention_backward(q_buf, k_buf, v_buf,
                         layer_grads + l_off,
                         layer_grads + l_off + wq[0].size,
                         layer_grads + l_off + wq[0].size + wk[0].size,
                         layer_grads + l_off + wq[0].size + wk[0].size + wv[0].size,
                         d_temp, d_hidden, q_buf, k_buf, v_buf, s_buf, hidden->data,
                         wq[l].data, wk[l].data, wv[l].data, wo[l].data);
        
        rmsnorm_backward(d_hidden, d_temp, hidden->data, hidden->data, D_MODEL);
    }
    
    // Input embedding backward
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* x = seq_data + s * INPUT_FEATURES;
        const double* d_h = d_hidden + s * D_MODEL;
        for (int d = 0; d < D_MODEL; d++) {
            for (int f = 0; f < SEQUENCE_FEATURES; f++)
                grads[f * D_MODEL + d] += d_h[d] * x[f + CONDITION_FEATURES];
            for (int f = 0; f < CONDITION_FEATURES; f++)
                grads[ws->size + f * D_MODEL + d] += d_h[d] * x[f];
        }
    }
}

void train_backprop(Dataset* ds, Tensor* out, Tensor* hidden, Tensor* temp,
                   Tensor* ws, Tensor* wc, Tensor* wq, Tensor* wk, Tensor* wv, 
                   Tensor* wo, Tensor* wf1, Tensor* wf2, Tensor* wout) {
    double lr = LEARNING_RATE, prev_loss = 1e9;
    time_t t = time(NULL);
    struct tm *tm = localtime(&t);
    char loss_name[100];
    sprintf(loss_name, "%d-%d-%d_%d-%d-%d_loss.csv", tm->tm_year+1900, tm->tm_mon+1, 
            tm->tm_mday, tm->tm_hour, tm->tm_min, tm->tm_sec);
    FILE* f = fopen(loss_name, "w");
    if (f) fprintf(f, "step,loss\n");

    const size_t total_params = ws->size + wc->size + wout->size + 
                               N_LAYERS * (wq[0].size + wk[0].size + wv[0].size + 
                                         wo[0].size + wf1[0].size + wf2[0].size);
    
    double *seq_data = malloc((SEQ_LENGTH + 1) * INPUT_FEATURES * sizeof(double));
    double *grads = malloc(total_params * sizeof(double));
    double *d_hidden = malloc(SEQ_LENGTH * D_MODEL * sizeof(double));
    double *d_temp = malloc(SEQ_LENGTH * D_MODEL * sizeof(double));
    double *q_buf = malloc(SEQ_LENGTH * D_MODEL * sizeof(double));
    double *k_buf = malloc(SEQ_LENGTH * D_MODEL * sizeof(double));
    double *v_buf = malloc(SEQ_LENGTH * D_MODEL * sizeof(double));
    double *s_buf = malloc(N_HEAD * SEQ_LENGTH * SEQ_LENGTH * sizeof(double));
    double *mid_buf = malloc(SEQ_LENGTH * (D_MODEL * 4) * sizeof(double));
    double *d_mid = malloc(SEQ_LENGTH * (D_MODEL * 4) * sizeof(double));

    const double beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.01;

    for (int step = 0; step < TRAINING_STEPS; step++) {
        int seq_start = rand() % (ds->rows - SEQ_LENGTH - 1);
        memcpy(seq_data, ds->data + seq_start * INPUT_FEATURES,
               (SEQ_LENGTH + 1) * INPUT_FEATURES * sizeof(double));

        forward_pass(seq_data, out, hidden, temp, ws, wc, wq, wk, wv, wo, wf1, wf2, wout,
                    q_buf, k_buf, v_buf, s_buf, mid_buf);
        double loss = compute_loss(out, seq_data);
        
        if (step > 0) {
            lr *= (loss > prev_loss * 1.1) ? 0.95 : (loss < prev_loss * 0.95) ? 1.05 : 1.0;
            lr = fmax(1e-6, fmin(1e-3, lr));
        }
        prev_loss = loss;
        
        printf("Step %d, Loss: %f, LR: %e\n", step, loss, lr);
        if (f) fprintf(f, "%d,%f\n", step, loss);

        backward_pass(grads, seq_data, out, hidden, ws, wc, wq, wk, wv, wo, wf1, wf2, wout,
                     d_hidden, d_temp, q_buf, k_buf, v_buf, s_buf, mid_buf, d_mid);
        
        // Update weights with Adam optimizer
        size_t offset = 0;
        Tensor* weights[] = {ws, wc, wout};
        for (int w = 0; w < 3; w++) {
            for (int i = 0; i < weights[w]->size; i++) {
                double grad = grads[offset + i] + weight_decay * weights[w]->data[i];
                weights[w]->m[i] = beta1 * weights[w]->m[i] + (1 - beta1) * grad;
                weights[w]->v[i] = beta2 * weights[w]->v[i] + (1 - beta2) * grad * grad;
                double m_hat = weights[w]->m[i] / (1.0 - pow(beta1, step + 1));
                double v_hat = weights[w]->v[i] / (1.0 - pow(beta2, step + 1));
                weights[w]->data[i] -= lr * m_hat / (sqrt(v_hat) + eps);
            }
            offset += weights[w]->size;
        }
        
        for (int l = 0; l < N_LAYERS; l++) {
            Tensor* layer_weights[] = {&wq[l], &wk[l], &wv[l], &wo[l], &wf1[l], &wf2[l]};
            for (int w = 0; w < 6; w++) {
                for (int i = 0; i < layer_weights[w]->size; i++) {
                    double grad = grads[offset + i] + weight_decay * layer_weights[w]->data[i];
                    layer_weights[w]->m[i] = beta1 * layer_weights[w]->m[i] + (1 - beta1) * grad;
                    layer_weights[w]->v[i] = beta2 * layer_weights[w]->v[i] + (1 - beta2) * grad * grad;
                    double m_hat = layer_weights[w]->m[i] / (1.0 - pow(beta1, step + 1));
                    double v_hat = layer_weights[w]->v[i] / (1.0 - pow(beta2, step + 1));
                    layer_weights[w]->data[i] -= lr * m_hat / (sqrt(v_hat) + eps);
                }
                offset += layer_weights[w]->size;
            }
        }
    }
    
    free(seq_data); free(grads); free(d_hidden); free(d_temp);
    free(q_buf); free(k_buf); free(v_buf); free(s_buf); 
    free(mid_buf); free(d_mid);
    if (f) fclose(f);
}

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 3 || !strstr(argv[1], ".csv")) { printf("Usage: %s <training_data.csv> [weights.bin]\n", argv[0]); return 1; }
    char *csv_file = argv[1], *weights_file = argc > 2 ? (strstr(argv[2], ".bin") ? argv[2] : NULL) : NULL;
    if (argc > 2 && !weights_file) { printf("Error: Invalid file '%s'\n", argv[2]); return 1; }

    srand(time(NULL));
    Dataset ds = load_csv(csv_file);
    const double ws = sqrt(2.0 / D_MODEL);
    
    Tensor W_seq = {malloc(SEQUENCE_FEATURES * D_MODEL * sizeof(double)), calloc(SEQUENCE_FEATURES * D_MODEL, sizeof(double)), calloc(SEQUENCE_FEATURES * D_MODEL, sizeof(double)), SEQUENCE_FEATURES * D_MODEL};
    Tensor W_cond = {malloc(CONDITION_FEATURES * D_MODEL * sizeof(double)), calloc(CONDITION_FEATURES * D_MODEL, sizeof(double)), calloc(CONDITION_FEATURES * D_MODEL, sizeof(double)), CONDITION_FEATURES * D_MODEL};
    Tensor W_out = {malloc(D_MODEL * SEQUENCE_FEATURES * sizeof(double)), calloc(D_MODEL * SEQUENCE_FEATURES, sizeof(double)), calloc(D_MODEL * SEQUENCE_FEATURES, sizeof(double)), D_MODEL * SEQUENCE_FEATURES};
    
    Tensor W_q[N_LAYERS], W_k[N_LAYERS], W_v[N_LAYERS], W_o[N_LAYERS], W_ff1[N_LAYERS], W_ff2[N_LAYERS];
    for (int l = 0; l < N_LAYERS; l++) {
        const int attn_size = D_MODEL * D_MODEL, ff_size1 = D_MODEL * (D_MODEL * 4), ff_size2 = (D_MODEL * 4) * D_MODEL;
        W_q[l] = (Tensor){malloc(attn_size * sizeof(double)), calloc(attn_size, sizeof(double)), calloc(attn_size, sizeof(double)), attn_size};
        W_k[l] = (Tensor){malloc(attn_size * sizeof(double)), calloc(attn_size, sizeof(double)), calloc(attn_size, sizeof(double)), attn_size};
        W_v[l] = (Tensor){malloc(attn_size * sizeof(double)), calloc(attn_size, sizeof(double)), calloc(attn_size, sizeof(double)), attn_size};
        W_o[l] = (Tensor){malloc(attn_size * sizeof(double)), calloc(attn_size, sizeof(double)), calloc(attn_size, sizeof(double)), attn_size};
        W_ff1[l] = (Tensor){malloc(ff_size1 * sizeof(double)), calloc(ff_size1, sizeof(double)), calloc(ff_size1, sizeof(double)), ff_size1};
        W_ff2[l] = (Tensor){malloc(ff_size2 * sizeof(double)), calloc(ff_size2, sizeof(double)), calloc(ff_size2, sizeof(double)), ff_size2};
    }

    Tensor hidden = {malloc(SEQ_LENGTH * D_MODEL * sizeof(double)), NULL, NULL, SEQ_LENGTH * D_MODEL};
    Tensor temp = {malloc(SEQ_LENGTH * D_MODEL * sizeof(double)), NULL, NULL, SEQ_LENGTH * D_MODEL};
    Tensor output = {malloc(SEQ_LENGTH * SEQUENCE_FEATURES * sizeof(double)), NULL, NULL, SEQ_LENGTH * SEQUENCE_FEATURES};

    if (weights_file && load_weights(weights_file, &W_seq, &W_cond, W_q, W_k, W_v, W_o, W_ff1, W_ff2, &W_out)) {
        printf("Successfully loaded weights\n");
    } else {
        if (weights_file) printf("Failed to load weights, initializing randomly\n");
        for (int i = 0; i < W_seq.size; i++) W_seq.data[i] = randn() * ws;
        for (int i = 0; i < W_cond.size; i++) W_cond.data[i] = randn() * ws;
        for (int i = 0; i < W_out.size; i++) W_out.data[i] = randn() * ws;
        for (int l = 0; l < N_LAYERS; l++) {
            for (int i = 0; i < W_q[l].size; i++) W_q[l].data[i] = W_k[l].data[i] = W_v[l].data[i] = W_o[l].data[i] = randn() * ws;
            for (int i = 0; i < W_ff1[l].size; i++) W_ff1[l].data[i] = randn() * ws;
            for (int i = 0; i < W_ff2[l].size; i++) W_ff2[l].data[i] = randn() * ws;
        }
    }

    train_backprop(&ds, &output, &hidden, &temp, &W_seq, &W_cond, W_q, W_k, W_v, W_o, W_ff1, W_ff2, &W_out);
    
    time_t t = time(NULL); struct tm tm = *localtime(&t);
    char filename[100]; sprintf(filename, "%d-%d-%d_%d-%d-%d_weights.bin", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    save_weights(filename, &W_seq, &W_cond, W_q, W_k, W_v, W_o, W_ff1, W_ff2, &W_out);
    printf("Saved weights to: %s\n", filename);

    free(ds.data); free(hidden.data); free(temp.data); free(output.data);
    free(W_seq.data); free(W_seq.m); free(W_seq.v);
    free(W_cond.data); free(W_cond.m); free(W_cond.v);
    free(W_out.data); free(W_out.m); free(W_out.v);
    for (int l = 0; l < N_LAYERS; l++) {
        free(W_q[l].data); free(W_q[l].m); free(W_q[l].v);
        free(W_k[l].data); free(W_k[l].m); free(W_k[l].v);
        free(W_v[l].data); free(W_v[l].m); free(W_v[l].v);
        free(W_o[l].data); free(W_o[l].m); free(W_o[l].v);
        free(W_ff1[l].data); free(W_ff1[l].m); free(W_ff1[l].v);
        free(W_ff2[l].data); free(W_ff2[l].m); free(W_ff2[l].v);
    }
    return 0;
}