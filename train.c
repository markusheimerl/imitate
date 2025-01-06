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

void save_weights(const char* filename, const double* ws, const double* wc, const double* wq, const double* wk, const double* wv, const double* wo, const double* wf1, const double* wf2, const double* wout) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;
    
    fwrite(ws, sizeof(double), SEQUENCE_FEATURES * D_MODEL, f);
    fwrite(wc, sizeof(double), CONDITION_FEATURES * D_MODEL, f);
    for (int l = 0; l < N_LAYERS; l++) {
        const int layer_offset = l * D_MODEL * D_MODEL;
        const int ff_offset1 = l * D_MODEL * (D_MODEL * 4);
        const int ff_offset2 = l * (D_MODEL * 4) * D_MODEL;
        fwrite(wq + layer_offset, sizeof(double), D_MODEL * D_MODEL, f);
        fwrite(wk + layer_offset, sizeof(double), D_MODEL * D_MODEL, f);
        fwrite(wv + layer_offset, sizeof(double), D_MODEL * D_MODEL, f);
        fwrite(wo + layer_offset, sizeof(double), D_MODEL * D_MODEL, f);
        fwrite(wf1 + ff_offset1, sizeof(double), D_MODEL * (D_MODEL * 4), f);
        fwrite(wf2 + ff_offset2, sizeof(double), (D_MODEL * 4) * D_MODEL, f);
    }
    fwrite(wout, sizeof(double), D_MODEL * SEQUENCE_FEATURES, f);
    fclose(f);
}

int load_weights(const char* filename, double* ws, double* wc, double* wq, double* wk, double* wv, double* wo, double* wf1, double* wf2, double* wout) {
    FILE* f = fopen(filename, "rb");
    if (!f) return 0;
    
    size_t read = fread(ws, sizeof(double), SEQUENCE_FEATURES * D_MODEL, f);
    read += fread(wc, sizeof(double), CONDITION_FEATURES * D_MODEL, f);
    
    for (int l = 0; l < N_LAYERS; l++) {
        const int layer_offset = l * D_MODEL * D_MODEL;
        const int ff_offset1 = l * D_MODEL * (D_MODEL * 4);
        const int ff_offset2 = l * (D_MODEL * 4) * D_MODEL;
        read += fread(wq + layer_offset, sizeof(double), D_MODEL * D_MODEL, f);
        read += fread(wk + layer_offset, sizeof(double), D_MODEL * D_MODEL, f);
        read += fread(wv + layer_offset, sizeof(double), D_MODEL * D_MODEL, f);
        read += fread(wo + layer_offset, sizeof(double), D_MODEL * D_MODEL, f);
        read += fread(wf1 + ff_offset1, sizeof(double), D_MODEL * (D_MODEL * 4), f);
        read += fread(wf2 + ff_offset2, sizeof(double), (D_MODEL * 4) * D_MODEL, f);
    }
    
    read += fread(wout, sizeof(double), D_MODEL * SEQUENCE_FEATURES, f);
    fclose(f);
    
    size_t expected = SEQUENCE_FEATURES * D_MODEL + CONDITION_FEATURES * D_MODEL + D_MODEL * SEQUENCE_FEATURES + N_LAYERS * (4 * D_MODEL * D_MODEL + D_MODEL * (D_MODEL * 4) + (D_MODEL * 4) * D_MODEL);
    return read == expected;
}

void rmsnorm(double *out, const double *in) {
    #pragma omp parallel for
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* x = in + s * D_MODEL;
        double* y = out + s * D_MODEL;
        double ss = 0.0;
        for (int i = 0; i < D_MODEL; i++) {
            ss += x[i] * x[i];
        }
        double scale = 1.0 / sqrt(ss / D_MODEL + 1e-5);
        for (int i = 0; i < D_MODEL; i++) {
            y[i] = x[i] * scale;
        }
    }
}

void feedforward(double *out, const double *w1, const double *w2, const double *in, double *mid) {
    #pragma omp parallel for
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* x = in + s * D_MODEL;
        double* y = out + s * D_MODEL;
        double* u = mid + s * (D_MODEL * 4);
        
        for (int i = 0; i < D_MODEL * 4; i++) {
            u[i] = 0.0;
            for (int j = 0; j < D_MODEL; j++) {
                u[i] += x[j] * w1[i * D_MODEL + j];
            }
            u[i] = 0.5 * u[i] * (1.0 + tanh(sqrt(2.0/M_PI) * u[i] + 0.044715 * u[i] * u[i] * u[i]));
        }
        
        for (int i = 0; i < D_MODEL; i++) {
            y[i] = 0.0;
            for (int j = 0; j < D_MODEL * 4; j++) {
                y[i] += u[j] * w2[i * (D_MODEL * 4) + j];
            }
        }
    }
}

void multihead_attention(double *out, const double *in, const double *wq, const double *wk, const double *wv, const double *wo, double *q, double *k, double *v, double *s) {
    const int hd = D_MODEL / N_HEAD;
    const double scale = 1.0 / sqrt(hd);

    #pragma omp parallel for
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* x = in + s * D_MODEL;
        for (int h = 0; h < N_HEAD; h++) {
            for (int d = 0; d < hd; d++) {
                double sq = 0.0, sk = 0.0, sv = 0.0;
                const int w_idx = (h * hd + d) * D_MODEL;
                for (int i = 0; i < D_MODEL; i++) {
                    sq += x[i] * wq[w_idx + i];
                    sk += x[i] * wk[w_idx + i];
                    sv += x[i] * wv[w_idx + i];
                }
                const int qkv_idx = s * D_MODEL + h * hd + d;
                q[qkv_idx] = sq;
                k[qkv_idx] = sk;
                v[qkv_idx] = sv;
            }
        }
    }

    #pragma omp parallel for
    for (int h = 0; h < N_HEAD; h++) {
        const double slope = pow(2.0, -(8.0 * (h + 1) / N_HEAD));
        for (int i = 0; i < SEQ_LENGTH; i++) {
            double max = -1e9, sum = 0.0;
            for (int j = 0; j <= i; j++) {
                double dot = 0.0;
                for (int d = 0; d < hd; d++) {
                    dot += q[i * D_MODEL + h * hd + d] * k[j * D_MODEL + h * hd + d];
                }
                s[(h * SEQ_LENGTH + i) * SEQ_LENGTH + j] = dot * scale - slope * (i - j);
                max = fmax(max, s[(h * SEQ_LENGTH + i) * SEQ_LENGTH + j]);
            }
            for (int j = 0; j <= i; j++) {
                s[(h * SEQ_LENGTH + i) * SEQ_LENGTH + j] = exp(s[(h * SEQ_LENGTH + i) * SEQ_LENGTH + j] - max);
                sum += s[(h * SEQ_LENGTH + i) * SEQ_LENGTH + j];
            }
            for (int j = 0; j <= i; j++) {
                s[(h * SEQ_LENGTH + i) * SEQ_LENGTH + j] /= (sum + 1e-10);
            }
        }
    }

    #pragma omp parallel for
    for (int t = 0; t < SEQ_LENGTH; t++) {
        double tmp[D_MODEL] = {0};
        for (int h = 0; h < N_HEAD; h++) {
            for (int d = 0; d < hd; d++) {
                double sum = 0.0;
                for (int j = 0; j <= t; j++) {
                    sum += s[(h * SEQ_LENGTH + t) * SEQ_LENGTH + j] * v[j * D_MODEL + h * hd + d];
                }
                tmp[h * hd + d] = sum;
            }
        }
        for (int d = 0; d < D_MODEL; d++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++) {
                sum += tmp[i] * wo[d * D_MODEL + i];
            }
            out[t * D_MODEL + d] = sum;
        }
    }
}

void forward_pass(const double* seq_data, double* out, double* hidden, double* temp, const double* ws, const double* wc, const double* wq, const double* wk, const double* wv, const double* wo, const double* wf1, const double* wf2, const double* wout, double* q_buf, double* k_buf, double* v_buf, double* s_buf, double* mid_buf) {
    #pragma omp parallel for
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* x = seq_data + s * INPUT_FEATURES;
        double* y = hidden + s * D_MODEL;
        for (int d = 0; d < D_MODEL; d++) {
            double sum = 0.0;
            for (int f = 0; f < SEQUENCE_FEATURES; f++) {
                sum += x[f + CONDITION_FEATURES] * ws[f * D_MODEL + d];
            }
            for (int f = 0; f < CONDITION_FEATURES; f++) {
                sum += x[f] * wc[f * D_MODEL + d];
            }
            y[d] = sum;
        }
    }

    for (int l = 0; l < N_LAYERS; l++) {
        const int layer_offset = l * D_MODEL * D_MODEL;
        const int ff_offset1 = l * D_MODEL * (D_MODEL * 4);
        const int ff_offset2 = l * (D_MODEL * 4) * D_MODEL;
        
        rmsnorm(temp, hidden);
        multihead_attention(temp, temp, wq + layer_offset, wk + layer_offset, wv + layer_offset, wo + layer_offset, q_buf, k_buf, v_buf, s_buf);
        
        for (int i = 0; i < SEQ_LENGTH * D_MODEL; i++) {
            hidden[i] += temp[i];
        }
        
        rmsnorm(temp, hidden);
        feedforward(temp, wf1 + ff_offset1, wf2 + ff_offset2, temp, mid_buf);
        
        for (int i = 0; i < SEQ_LENGTH * D_MODEL; i++) {
            hidden[i] += temp[i];
        }
    }

    #pragma omp parallel for
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* h = hidden + s * D_MODEL;
        double* o = out + s * SEQUENCE_FEATURES;
        for (int f = 0; f < SEQUENCE_FEATURES; f++) {
            double sum = 0.0;
            for (int d = 0; d < D_MODEL; d++) {
                sum += h[d] * wout[f * D_MODEL + d];
            }
            o[f] = sum;
        }
    }
}

double compute_loss(const double* out, const double* seq_data) {
    double loss = 0.0;
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* pred = out + s * SEQUENCE_FEATURES;
        const double* target = seq_data + (s + 1) * INPUT_FEATURES + CONDITION_FEATURES;
        for (int f = 0; f < SEQUENCE_FEATURES; f++) {
            loss += (pred[f] - target[f]) * (pred[f] - target[f]);
        }
    }
    return loss / (SEQ_LENGTH * SEQUENCE_FEATURES);
}

void rmsnorm_backward(double* d_x, const double* d_y, const double* x, const double* norm_x, int size) {
    const double inv_d = 1.0 / D_MODEL;
    #pragma omp parallel for
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* x_s = x + s * size;
        const double* d_y_s = d_y + s * size;
        double* d_x_s = d_x + s * size;
        double ss = 0.0, d_scale = 0.0;
        
        for (int i = 0; i < size; i++) {
            ss += x_s[i] * x_s[i];
        }
        double scale = 1.0 / sqrt(ss * inv_d + 1e-5);
        for (int i = 0; i < size; i++) {
            d_scale += d_y_s[i] * x_s[i];
        }
        for (int i = 0; i < size; i++) {
            d_x_s[i] = d_y_s[i] * scale - x_s[i] * scale * scale * scale * inv_d * d_scale / (2.0 * size);
        }
    }
}

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
            for (int d = 0; d < D_MODEL; d++) {
                sum += x_s[d] * w1[h * D_MODEL + d];
            }
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

void attention_backward(double* d_q, double* d_k, double* d_v, double* d_wq, double* d_wk, double* d_wv, double* d_wo, double* d_x, const double* d_out, const double* q, const double* k, const double* v, const double* s, const double* x, const double* wq, const double* wk, const double* wv, const double* wo) {
    const int hd = D_MODEL / N_HEAD;
    const double scale = 1.0 / sqrt(hd);
    
    #pragma omp parallel for
    for (int t = 0; t < SEQ_LENGTH; t++) {
        double d_tmp[D_MODEL] = {0};
        const int out_idx = t * D_MODEL;
        for (int d = 0; d < D_MODEL; d++) {
            for (int i = 0; i < D_MODEL; i++) {
                d_tmp[i] += d_out[out_idx + d] * wo[d * D_MODEL + i];
                d_wo[d * D_MODEL + i] += d_out[out_idx + d] * v[out_idx + i];
            }
        }
        
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

    #pragma omp parallel for
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const int x_idx = s * D_MODEL;
        for (int h = 0; h < N_HEAD; h++) {
            for (int d = 0; d < hd; d++) {
                const int w_idx = (h * hd + d) * D_MODEL;
                const int qkv_idx = x_idx + h * hd + d;
                for (int i = 0; i < D_MODEL; i++) {
                    d_wq[w_idx + i] += d_q[qkv_idx] * x[x_idx + i];
                    d_wk[w_idx + i] += d_k[qkv_idx] * x[x_idx + i];
                    d_wv[w_idx + i] += d_v[qkv_idx] * x[x_idx + i];
                    d_x[x_idx + i] += d_q[qkv_idx] * wq[w_idx + i] + d_k[qkv_idx] * wk[w_idx + i] + d_v[qkv_idx] * wv[w_idx + i];
                }
            }
        }
    }
}

void backward_pass(double* grads, const double* seq_data, const double* out, const double* hidden, const double* ws, const double* wc, const double* wq, const double* wk, const double* wv, const double* wo, const double* wf1, const double* wf2, const double* wout, double* d_hidden, double* d_temp, double* q_buf, double* k_buf, double* v_buf, double* s_buf, double* mid_buf, double* d_mid) {
    memset(grads, 0, ((SEQUENCE_FEATURES + CONDITION_FEATURES) * D_MODEL + D_MODEL * SEQUENCE_FEATURES + N_LAYERS * (4 * D_MODEL * D_MODEL + D_MODEL * (D_MODEL * 4) + (D_MODEL * 4) * D_MODEL)) * sizeof(double));
    memset(d_hidden, 0, SEQ_LENGTH * D_MODEL * sizeof(double));
    
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* pred = out + s * SEQUENCE_FEATURES;
        const double* target = seq_data + (s + 1) * INPUT_FEATURES + CONDITION_FEATURES;
        for (int f = 0; f < SEQUENCE_FEATURES; f++) {
            const double d_out = 2.0 * (pred[f] - target[f]) / (SEQ_LENGTH * SEQUENCE_FEATURES);
            for (int d = 0; d < D_MODEL; d++) {
                grads[SEQUENCE_FEATURES * D_MODEL + CONDITION_FEATURES * D_MODEL + f * D_MODEL + d] += d_out * hidden[s * D_MODEL + d];
                d_hidden[s * D_MODEL + d] += d_out * wout[f * D_MODEL + d];
            }
        }
    }
    
    size_t offset = (SEQUENCE_FEATURES + CONDITION_FEATURES) * D_MODEL + D_MODEL * SEQUENCE_FEATURES;
    for (int l = N_LAYERS - 1; l >= 0; l--) {
        const size_t l_off = l * (D_MODEL * D_MODEL + D_MODEL * D_MODEL + D_MODEL * D_MODEL + D_MODEL * D_MODEL + D_MODEL * (D_MODEL * 4) + (D_MODEL * 4) * D_MODEL);
        
        memcpy(d_temp, d_hidden, SEQ_LENGTH * D_MODEL * sizeof(double));
        feedforward_backward(d_hidden, grads + offset + l_off + 4 * D_MODEL * D_MODEL, grads + offset + l_off + 4 * D_MODEL * D_MODEL + D_MODEL * (D_MODEL * 4), d_temp, hidden, wf1 + l * D_MODEL * (D_MODEL * 4), wf2 + l * (D_MODEL * 4) * D_MODEL, mid_buf, d_mid);
        
        rmsnorm_backward(d_temp, d_hidden, hidden, hidden, D_MODEL);
        memcpy(d_hidden, d_temp, SEQ_LENGTH * D_MODEL * sizeof(double));
        
        attention_backward(q_buf, k_buf, v_buf, grads + offset + l_off, grads + offset + l_off + D_MODEL * D_MODEL, grads + offset + l_off + 2 * D_MODEL * D_MODEL, grads + offset + l_off + 3 * D_MODEL * D_MODEL, d_temp, d_hidden, q_buf, k_buf, v_buf, s_buf, hidden, wq + l * D_MODEL * D_MODEL, wk + l * D_MODEL * D_MODEL, wv + l * D_MODEL * D_MODEL, wo + l * D_MODEL * D_MODEL);
        
        rmsnorm_backward(d_hidden, d_temp, hidden, hidden, D_MODEL);
    }
    
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* x = seq_data + s * INPUT_FEATURES;
        const double* d_h = d_hidden + s * D_MODEL;
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

void train_backprop(Dataset* ds, double* out, double* hidden, double* temp, double* ws, double* wc, double* wq, double* wk, double* wv, double* wo, double* wf1, double* wf2, double* wout, double* ws_m, double* ws_v, double* wc_m, double* wc_v, double* wq_m, double* wq_v, double* wk_m, double* wk_v, double* wv_m, double* wv_v, double* wo_m, double* wo_v, double* wf1_m, double* wf1_v, double* wf2_m, double* wf2_v, double* wout_m, double* wout_v) {
    double lr = LEARNING_RATE, prev_loss = 1e9;
    time_t t = time(NULL);
    struct tm *tm = localtime(&t);
    char loss_name[100];
    sprintf(loss_name, "%d-%d-%d_%d-%d-%d_loss.csv", tm->tm_year+1900, tm->tm_mon+1, tm->tm_mday, tm->tm_hour, tm->tm_min, tm->tm_sec);
    FILE* f = fopen(loss_name, "w");
    if (f) fprintf(f, "step,loss\n");

    const size_t total_params = (SEQUENCE_FEATURES + CONDITION_FEATURES + SEQUENCE_FEATURES) * D_MODEL + N_LAYERS * (4 * D_MODEL * D_MODEL + D_MODEL * (D_MODEL * 4) + (D_MODEL * 4) * D_MODEL);
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
        memcpy(seq_data, ds->data + seq_start * INPUT_FEATURES, (SEQ_LENGTH + 1) * INPUT_FEATURES * sizeof(double));

        forward_pass(seq_data, out, hidden, temp, ws, wc, wq, wk, wv, wo, wf1, wf2, wout, q_buf, k_buf, v_buf, s_buf, mid_buf);
        double loss = compute_loss(out, seq_data);
        
        if (step > 0) {
            lr *= (loss > prev_loss * 1.1) ? 0.95 : (loss < prev_loss * 0.95) ? 1.05 : 1.0;
            lr = fmax(1e-6, fmin(1e-3, lr));
        }
        prev_loss = loss;
        
        printf("Step %d, Loss: %f, LR: %e\n", step, loss, lr);
        if (f) fprintf(f, "%d,%f\n", step, loss);

        backward_pass(grads, seq_data, out, hidden, ws, wc, wq, wk, wv, wo, wf1, wf2, wout, d_hidden, d_temp, q_buf, k_buf, v_buf, s_buf, mid_buf, d_mid);
        
        size_t offset = 0;
        const size_t sizes[] = {SEQUENCE_FEATURES * D_MODEL, CONDITION_FEATURES * D_MODEL, SEQUENCE_FEATURES * D_MODEL};
        double *weights[] = {ws, wc, wout};
        double *m[] = {ws_m, wc_m, wout_m};
        double *v[] = {ws_v, wc_v, wout_v};
        
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
    
    double *W_seq = malloc(SEQUENCE_FEATURES * D_MODEL * sizeof(double));
    double *W_seq_m = calloc(SEQUENCE_FEATURES * D_MODEL, sizeof(double));
    double *W_seq_v = calloc(SEQUENCE_FEATURES * D_MODEL, sizeof(double));
    
    double *W_cond = malloc(CONDITION_FEATURES * D_MODEL * sizeof(double));
    double *W_cond_m = calloc(CONDITION_FEATURES * D_MODEL, sizeof(double));
    double *W_cond_v = calloc(CONDITION_FEATURES * D_MODEL, sizeof(double));
    
    double *W_out = malloc(D_MODEL * SEQUENCE_FEATURES * sizeof(double));
    double *W_out_m = calloc(D_MODEL * SEQUENCE_FEATURES, sizeof(double));
    double *W_out_v = calloc(D_MODEL * SEQUENCE_FEATURES, sizeof(double));
    
    double *W_q = malloc(N_LAYERS * D_MODEL * D_MODEL * sizeof(double));
    double *W_k = malloc(N_LAYERS * D_MODEL * D_MODEL * sizeof(double));
    double *W_v = malloc(N_LAYERS * D_MODEL * D_MODEL * sizeof(double));
    double *W_o = malloc(N_LAYERS * D_MODEL * D_MODEL * sizeof(double));
    double *W_ff1 = malloc(N_LAYERS * D_MODEL * (D_MODEL * 4) * sizeof(double));
    double *W_ff2 = malloc(N_LAYERS * (D_MODEL * 4) * D_MODEL * sizeof(double));
    
    double *W_q_m = calloc(N_LAYERS * D_MODEL * D_MODEL, sizeof(double));
    double *W_k_m = calloc(N_LAYERS * D_MODEL * D_MODEL, sizeof(double));
    double *W_v_m = calloc(N_LAYERS * D_MODEL * D_MODEL, sizeof(double));
    double *W_o_m = calloc(N_LAYERS * D_MODEL * D_MODEL, sizeof(double));
    double *W_ff1_m = calloc(N_LAYERS * D_MODEL * (D_MODEL * 4), sizeof(double));
    double *W_ff2_m = calloc(N_LAYERS * (D_MODEL * 4) * D_MODEL, sizeof(double));
    
    double *W_q_v = calloc(N_LAYERS * D_MODEL * D_MODEL, sizeof(double));
    double *W_k_v = calloc(N_LAYERS * D_MODEL * D_MODEL, sizeof(double));
    double *W_v_v = calloc(N_LAYERS * D_MODEL * D_MODEL, sizeof(double));
    double *W_o_v = calloc(N_LAYERS * D_MODEL * D_MODEL, sizeof(double));
    double *W_ff1_v = calloc(N_LAYERS * D_MODEL * (D_MODEL * 4), sizeof(double));
    double *W_ff2_v = calloc(N_LAYERS * (D_MODEL * 4) * D_MODEL, sizeof(double));

    double *hidden = malloc(SEQ_LENGTH * D_MODEL * sizeof(double));
    double *temp = malloc(SEQ_LENGTH * D_MODEL * sizeof(double));
    double *output = malloc(SEQ_LENGTH * SEQUENCE_FEATURES * sizeof(double));

    if (weights_file && load_weights(weights_file, W_seq, W_cond, W_q, W_k, W_v, W_o, W_ff1, W_ff2, W_out)) {
        printf("Successfully loaded weights\n");
    } else {
        if (weights_file) printf("Failed to load weights, initializing randomly\n");
        for (int i = 0; i < SEQUENCE_FEATURES * D_MODEL; i++) W_seq[i] = randn() * ws;
        for (int i = 0; i < CONDITION_FEATURES * D_MODEL; i++) W_cond[i] = randn() * ws;
        for (int i = 0; i < D_MODEL * SEQUENCE_FEATURES; i++) W_out[i] = randn() * ws;
        for (int i = 0; i < N_LAYERS * D_MODEL * D_MODEL; i++) W_q[i] = W_k[i] = W_v[i] = W_o[i] = randn() * ws;
        for (int i = 0; i < N_LAYERS * D_MODEL * (D_MODEL * 4); i++) W_ff1[i] = randn() * ws;
        for (int i = 0; i < N_LAYERS * (D_MODEL * 4) * D_MODEL; i++) W_ff2[i] = randn() * ws;
    }

    train_backprop(&ds, output, hidden, temp, W_seq, W_cond, W_q, W_k, W_v, W_o, W_ff1, W_ff2, W_out, W_seq_m, W_seq_v, W_cond_m, W_cond_v, W_q_m, W_q_v, W_k_m, W_k_v, W_v_m, W_v_v, W_o_m, W_o_v, W_ff1_m, W_ff1_v, W_ff2_m, W_ff2_v, W_out_m, W_out_v);
    
    time_t t = time(NULL); struct tm tm = *localtime(&t);
    char filename[100]; sprintf(filename, "%d-%d-%d_%d-%d-%d_weights.bin", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    save_weights(filename, W_seq, W_cond, W_q, W_k, W_v, W_o, W_ff1, W_ff2, W_out);
    printf("Saved weights to: %s\n", filename);

    free(ds.data); free(hidden); free(temp); free(output);
    free(W_seq); free(W_seq_m); free(W_seq_v);
    free(W_cond); free(W_cond_m); free(W_cond_v);
    free(W_out); free(W_out_m); free(W_out_v);
    free(W_q); free(W_q_m); free(W_q_v);
    free(W_k); free(W_k_m); free(W_k_v);
    free(W_v); free(W_v_m); free(W_v_v);
    free(W_o); free(W_o_m); free(W_o_v);
    free(W_ff1); free(W_ff1_m); free(W_ff1_v);
    free(W_ff2); free(W_ff2_m); free(W_ff2_v);
    return 0;
}