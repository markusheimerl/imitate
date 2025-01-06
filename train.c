#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#define CONDITION_FEATURES 4
#define SEQUENCE_FEATURES 10
#define INPUT_FEATURES (CONDITION_FEATURES + SEQUENCE_FEATURES)
#define BATCH_SIZE 2
#define SEQ_LENGTH 32
#define D_MODEL 32
#define N_HEAD 2
#define N_LAYERS 2
#define EPSILON 1e-4
#define LEARNING_RATE 0.00001
#define TRAINING_STEPS 10000

typedef struct { double *data; int rows, cols; } Dataset;
typedef struct { double *data; double *m, *v; int size; } Tensor;

double randn() { return sqrt(-2.0 * log((double)rand() / RAND_MAX)) * cos(2.0 * M_PI * (double)rand() / RAND_MAX); }

Dataset load_csv(const char* filename) {
    Dataset ds = {NULL, 0, INPUT_FEATURES};
    char line[1024];
    double* tmp = malloc(1000 * INPUT_FEATURES * sizeof(double));
    FILE* f = fopen(filename, "r");
    if (!f || !fgets(line, 1024, f)) { printf("File error\n"); exit(1); }
    
    while (fgets(line, 1024, f)) {
        if (ds.rows >= 1000) tmp = realloc(tmp, (ds.rows*2) * INPUT_FEATURES * sizeof(double));
        char* tok = strtok(line, ",");
        for (int i = 0; i < INPUT_FEATURES && tok; i++, tok = strtok(NULL, ",")) {
            double val = atof(tok);
            tmp[ds.rows * INPUT_FEATURES + i] = val;
        }
        ds.rows++;
    }
    
    ds.data = tmp;
    fclose(f);
    return ds;
}

void save_weights(const char* filename, const Tensor* ws, const Tensor* wc,
                 const Tensor* wq, const Tensor* wk, const Tensor* wv, const Tensor* wo,
                 const Tensor* wf1, const Tensor* wf2, const Tensor* wout) {
    FILE* f = fopen(filename, "wb");
    if (f) {
        fwrite(ws->data, sizeof(double), ws->size, f);
        fwrite(wc->data, sizeof(double), wc->size, f);
        for (int l = 0; l < N_LAYERS; l++) {
            fwrite(wq[l].data, sizeof(double), wq[l].size, f);
            fwrite(wk[l].data, sizeof(double), wk[l].size, f);
            fwrite(wv[l].data, sizeof(double), wv[l].size, f);
            fwrite(wo[l].data, sizeof(double), wo[l].size, f);
            fwrite(wf1[l].data, sizeof(double), wf1[l].size, f);
            fwrite(wf2[l].data, sizeof(double), wf2[l].size, f);
        }
        fwrite(wout->data, sizeof(double), wout->size, f);
        fclose(f);
    }
}

int load_weights(const char* filename, Tensor* ws, Tensor* wc,
                Tensor* wq, Tensor* wk, Tensor* wv, Tensor* wo,
                Tensor* wf1, Tensor* wf2, Tensor* wout) {
    FILE* f = fopen(filename, "rb");
    if (!f) return 0;

    size_t read = fread(ws->data, sizeof(double), ws->size, f) +
                  fread(wc->data, sizeof(double), wc->size, f);
    
    for (int l = 0; l < N_LAYERS; l++)
        read += fread(wq[l].data, sizeof(double), wq[l].size, f) +
                fread(wk[l].data, sizeof(double), wk[l].size, f) +
                fread(wv[l].data, sizeof(double), wv[l].size, f) +
                fread(wo[l].data, sizeof(double), wo[l].size, f) +
                fread(wf1[l].data, sizeof(double), wf1[l].size, f) +
                fread(wf2[l].data, sizeof(double), wf2[l].size, f);
    
    read += fread(wout->data, sizeof(double), wout->size, f);
    fclose(f);

    size_t expected = ws->size + wc->size + wout->size;
    for (int l = 0; l < N_LAYERS; l++)
        expected += wq[l].size + wk[l].size + wv[l].size + 
                   wo[l].size + wf1[l].size + wf2[l].size;

    return read == expected;
}

void rmsnorm(Tensor *out, const Tensor *in) {
    const double inv_d = 1.0 / D_MODEL;
    for (int b = 0; b < BATCH_SIZE * SEQ_LENGTH; b++) {
        const double* x = in->data + b * D_MODEL;
        double* y = out->data + b * D_MODEL;
        double ss = 0.0;
        for (int i = 0; i < D_MODEL; i++) ss += x[i] * x[i];
        double scale = 1.0 / sqrt(ss * inv_d + 1e-5);
        for (int i = 0; i < D_MODEL; i++) y[i] = x[i] * scale;
    }
}

void feedforward(Tensor *out, const Tensor *w1, const Tensor *w2, const Tensor *in, double *mid) {
    const double sqrt_2_pi = sqrt(2.0/M_PI);
    for (int b = 0; b < BATCH_SIZE * SEQ_LENGTH; b++) {
        const double* x = in->data + b * D_MODEL;
        double* y = out->data + b * D_MODEL;
        double* m = mid + b * (D_MODEL * 4);
        
        for (int h = 0; h < D_MODEL * 4; h++) {
            double sum = 0.0;
            for (int d = 0; d < D_MODEL; d++) sum += x[d] * w1->data[h * D_MODEL + d];
            double t = sum + 0.044715 * sum * sum * sum;
            m[h] = 0.5 * sum * (1.0 + tanh(sqrt_2_pi * t));
        }
        
        for (int d = 0; d < D_MODEL; d++) {
            double sum = 0.0;
            for (int h = 0; h < D_MODEL * 4; h++) sum += m[h] * w2->data[d * (D_MODEL * 4) + h];
            y[d] = sum;
        }
    }
}

void multihead_attention(Tensor *out, const Tensor *in, const Tensor *wq, const Tensor *wk, 
                        const Tensor *wv, const Tensor *wo,
                        double *q, double *k, double *v, double *s) {
    const int hd = D_MODEL / N_HEAD;
    const double scale = 1.0 / sqrt(hd);

    // QKV Transform
    for (int b = 0; b < BATCH_SIZE * SEQ_LENGTH; b++) {
        const double* x = in->data + b * D_MODEL;
        for (int h = 0; h < N_HEAD; h++) {
            for (int d = 0; d < hd; d++) {
                double sq = 0.0, sk = 0.0, sv = 0.0;
                const int w_idx = (h * hd + d) * D_MODEL;
                for (int i = 0; i < D_MODEL; i++) {
                    sq += x[i] * wq->data[w_idx + i];
                    sk += x[i] * wk->data[w_idx + i];
                    sv += x[i] * wv->data[w_idx + i];
                }
                const int qkv_idx = b * D_MODEL + h * hd + d;
                q[qkv_idx] = sq; k[qkv_idx] = sk; v[qkv_idx] = sv;
            }
        }
    }

    // Attention with alibi mask
    for (int b = 0; b < BATCH_SIZE; b++)
        for (int h = 0; h < N_HEAD; h++) {
            const double slope = pow(2.0, -(8.0 * (h + 1) / N_HEAD));
            for (int i = 0; i < SEQ_LENGTH; i++) {
                double max = -1e9, sum = 0.0;
                for (int j = 0; j <= i; j++) {
                    double dot = 0.0;
                    for (int d = 0; d < hd; d++) dot += q[(b * SEQ_LENGTH + i) * D_MODEL + h * hd + d] * k[(b * SEQ_LENGTH + j) * D_MODEL + h * hd + d];
                    s[(b * N_HEAD * SEQ_LENGTH + h * SEQ_LENGTH + i) * SEQ_LENGTH + j] = dot * scale - slope * (i - j);
                    max = fmax(max, s[(b * N_HEAD * SEQ_LENGTH + h * SEQ_LENGTH + i) * SEQ_LENGTH + j]);
                }
                
                for (int j = 0; j <= i; j++) {
                    s[(b * N_HEAD * SEQ_LENGTH + h * SEQ_LENGTH + i) * SEQ_LENGTH + j] = exp(s[(b * N_HEAD * SEQ_LENGTH + h * SEQ_LENGTH + i) * SEQ_LENGTH + j] - max);
                    sum += s[(b * N_HEAD * SEQ_LENGTH + h * SEQ_LENGTH + i) * SEQ_LENGTH + j];
                }
                for (int j = 0; j <= i; j++) s[(b * N_HEAD * SEQ_LENGTH + h * SEQ_LENGTH + i) * SEQ_LENGTH + j] /= (sum + 1e-10);
            }
        }

    // Output projection
    for (int b = 0; b < BATCH_SIZE; b++)
        for (int t = 0; t < SEQ_LENGTH; t++) {
            double tmp[D_MODEL] = {0};
            for (int h = 0; h < N_HEAD; h++)
                for (int d = 0; d < hd; d++) {
                    double sum = 0.0;
                    for (int j = 0; j <= t; j++) sum += s[(b * N_HEAD * SEQ_LENGTH + h * SEQ_LENGTH + t) * SEQ_LENGTH + j] * v[(b * SEQ_LENGTH + j) * D_MODEL + h * hd + d];
                    tmp[h * hd + d] = sum;
                }
            
            for (int d = 0; d < D_MODEL; d++) {
                double sum = 0.0;
                for (int i = 0; i < D_MODEL; i++) sum += tmp[i] * wo->data[d * D_MODEL + i];
                out->data[(b * SEQ_LENGTH + t) * D_MODEL + d] = sum;
            }
        }
}

void embed_sequence(Tensor* out, const double* in, const Tensor* ws, const Tensor* wc) {
    for (int b = 0; b < BATCH_SIZE * SEQ_LENGTH; b++) {
        const double* x = in + b * INPUT_FEATURES;
        double* y = out->data + b * D_MODEL;
        
        for (int d = 0; d < D_MODEL; d++) {
            double sum = 0.0;
            for (int f = 0; f < SEQUENCE_FEATURES; f++) sum += x[f + CONDITION_FEATURES] * ws->data[f * D_MODEL + d];
            for (int f = 0; f < CONDITION_FEATURES; f++) sum += x[f] * wc->data[f * D_MODEL + d];
            y[d] = sum;
        }
    }
}

void forward_pass(const double* batch_data, Tensor* out, Tensor* hidden, Tensor* temp,
                 const Tensor* ws, const Tensor* wc, const Tensor* wq, const Tensor* wk, 
                 const Tensor* wv, const Tensor* wo, const Tensor* wf1, const Tensor* wf2, 
                 const Tensor* wout,
                 double* q_buf, double* k_buf, double* v_buf, double* s_buf, double* mid_buf) {
    embed_sequence(hidden, batch_data, ws, wc);

    for (int l = 0; l < N_LAYERS; l++) {
        rmsnorm(temp, hidden);
        multihead_attention(temp, temp, &wq[l], &wk[l], &wv[l], &wo[l], q_buf, k_buf, v_buf, s_buf);
        for (int i = 0; i < hidden->size; i++) hidden->data[i] += temp->data[i];
        rmsnorm(temp, hidden);
        feedforward(temp, &wf1[l], &wf2[l], temp, mid_buf);
        for (int i = 0; i < hidden->size; i++) hidden->data[i] += temp->data[i];
    }

    for (int b = 0; b < BATCH_SIZE * SEQ_LENGTH; b++) {
        const double* h = hidden->data + b * D_MODEL;
        double* o = out->data + b * SEQUENCE_FEATURES;
        for (int f = 0; f < SEQUENCE_FEATURES; f++) {
            double sum = 0.0;
            const double* w = wout->data + f * D_MODEL;
            for (int d = 0; d < D_MODEL; d++) sum += h[d] * w[d];
            o[f] = sum;
        }
    }
}

double compute_loss(const Tensor* out, const double* batch_data) {
    double loss = 0.0;
    for (int b = 0; b < BATCH_SIZE * SEQ_LENGTH; b++) {
        const double* pred = out->data + b * SEQUENCE_FEATURES;
        const double* target = batch_data + ((b / SEQ_LENGTH) * (SEQ_LENGTH + 1) + (b % SEQ_LENGTH + 1)) * INPUT_FEATURES + CONDITION_FEATURES;
        for (int f = 0; f < SEQUENCE_FEATURES; f++) {
            double diff = pred[f] - target[f];
            loss += diff * diff;
        }
    }
    return loss / (BATCH_SIZE * SEQ_LENGTH * SEQUENCE_FEATURES);
}

void rmsnorm_backward(double* d_x, const double* d_y, const double* x, const double* norm_x, int size) {
    const double inv_d = 1.0 / D_MODEL;
    for (int b = 0; b < BATCH_SIZE * SEQ_LENGTH; b++) {
        const double* x_b = x + b * size, *y_b = norm_x + b * size, *d_y_b = d_y + b * size;
        double* d_x_b = d_x + b * size;
        double ss = 0.0;
        for (int i = 0; i < size; i++) ss += x_b[i] * x_b[i];
        double scale = 1.0 / sqrt(ss * inv_d + EPSILON);
        double d_scale = 0.0;
        for (int i = 0; i < size; i++) d_scale += d_y_b[i] * x_b[i];
        for (int i = 0; i < size; i++) 
            d_x_b[i] = d_y_b[i] * scale - x_b[i] * scale * scale * scale * inv_d * d_scale / (2.0 * size);
    }
}

void feedforward_backward(double* d_x, double* d_w1, double* d_w2, const double* d_y, const double* x,
                         const double* w1, const double* w2, double* mid, double* d_mid) {
    const double sqrt_2_pi = sqrt(2.0/M_PI);
    for (int b = 0; b < BATCH_SIZE * SEQ_LENGTH; b++) {
        const double* x_b = x + b * D_MODEL, *d_y_b = d_y + b * D_MODEL;
        double* d_x_b = d_x + b * D_MODEL, *m_b = mid + b * (D_MODEL * 4), *d_m_b = d_mid + b * (D_MODEL * 4);
        
        for (int h = 0; h < D_MODEL * 4; h++) {
            d_m_b[h] = 0.0;
            for (int d = 0; d < D_MODEL; d++) {
                d_m_b[h] += d_y_b[d] * w2[d * (D_MODEL * 4) + h];
                d_w2[d * (D_MODEL * 4) + h] += d_y_b[d] * m_b[h];
            }
        }
        
        for (int h = 0; h < D_MODEL * 4; h++) {
            double sum = 0.0;
            for (int d = 0; d < D_MODEL; d++) sum += x_b[d] * w1[h * D_MODEL + d];
            double t = sum + 0.044715 * sum * sum * sum;
            double tanh_t = tanh(sqrt_2_pi * t);
            double d_gelu = d_m_b[h] * 0.5 * (1.0 + tanh_t + sum * sqrt_2_pi * (1.0 - tanh_t * tanh_t) * 
                                             (1.0 + 0.134145 * sum * sum));
            for (int d = 0; d < D_MODEL; d++) {
                d_w1[h * D_MODEL + d] += d_gelu * x_b[d];
                d_x_b[d] += d_gelu * w1[h * D_MODEL + d];
            }
        }
    }
}

void attention_backward(double* d_q, double* d_k, double* d_v, double* d_wq, double* d_wk, double* d_wv, 
                       double* d_wo, double* d_x, const double* d_out, const double* q, const double* k, 
                       const double* v, const double* s, const double* x, const double* wq, const double* wk, 
                       const double* wv, const double* wo) {
    const int hd = D_MODEL / N_HEAD;
    const double scale = 1.0 / sqrt(hd);
    
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int t = 0; t < SEQ_LENGTH; t++) {
            double d_tmp[D_MODEL] = {0};
            const double* d_o = d_out + (b * SEQ_LENGTH + t) * D_MODEL;
            
            for (int d = 0; d < D_MODEL; d++) {
                for (int i = 0; i < D_MODEL; i++) {
                    d_tmp[i] += d_o[d] * wo[d * D_MODEL + i];
                    d_wo[d * D_MODEL + i] += d_o[d] * ((b * SEQ_LENGTH + t) * D_MODEL + i < 
                        D_MODEL * BATCH_SIZE * SEQ_LENGTH ? v[(b * SEQ_LENGTH + t) * D_MODEL + i] : 0);
                }
            }
            
            for (int h = 0; h < N_HEAD; h++) {
                for (int d = 0; d < hd; d++) {
                    double d_v_sum = 0.0;
                    for (int j = 0; j <= t; j++) {
                        double d_score = d_tmp[h * hd + d] * v[(b * SEQ_LENGTH + j) * D_MODEL + h * hd + d] * scale;
                        if (j < t) d_score *= (1.0 - s[(b * N_HEAD * SEQ_LENGTH + h * SEQ_LENGTH + t) * SEQ_LENGTH + j]);
                        
                        for (int i = 0; i < D_MODEL; i++) {
                            d_wq[(h * hd + d) * D_MODEL + i] += d_score * x[(b * SEQ_LENGTH + t) * D_MODEL + i];
                            if (j > 0) d_wk[(h * hd + d) * D_MODEL + i] += d_score * 
                                x[(b * SEQ_LENGTH + j - 1) * D_MODEL + i];
                        }
                        d_v_sum += s[(b * N_HEAD * SEQ_LENGTH + h * SEQ_LENGTH + t) * SEQ_LENGTH + j] * 
                            d_tmp[h * hd + d];
                    }
                    for (int i = 0; i < D_MODEL; i++)
                        d_wv[(h * hd + d) * D_MODEL + i] += d_v_sum * x[(b * SEQ_LENGTH + t) * D_MODEL + i];
                }
            }
        }
    }
}

void backward_pass(double* grads, const double* batch_data, const Tensor* out, const Tensor* hidden,
                  const Tensor* ws, const Tensor* wc, const Tensor* wq, const Tensor* wk, const Tensor* wv, 
                  const Tensor* wo, const Tensor* wf1, const Tensor* wf2, const Tensor* wout,
                  double* d_hidden, double* d_temp, double* q_buf, double* k_buf, double* v_buf, 
                  double* s_buf, double* mid_buf, double* d_mid) {
    
    memset(grads, 0, (ws->size + wc->size + wout->size + N_LAYERS * (wq[0].size + wk[0].size + 
           wv[0].size + wo[0].size + wf1[0].size + wf2[0].size)) * sizeof(double));
    memset(d_hidden, 0, BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    
    for (int b = 0; b < BATCH_SIZE * SEQ_LENGTH; b++) {
        const double* pred = out->data + b * SEQUENCE_FEATURES;
        const double* target = batch_data + ((b / SEQ_LENGTH) * (SEQ_LENGTH + 1) + 
                             (b % SEQ_LENGTH + 1)) * INPUT_FEATURES + CONDITION_FEATURES;
        for (int f = 0; f < SEQUENCE_FEATURES; f++) {
            double d_out = 2.0 * (pred[f] - target[f]) / (BATCH_SIZE * SEQ_LENGTH * SEQUENCE_FEATURES);
            double* w_grad = grads + ws->size + wc->size + f * D_MODEL;
            for (int d = 0; d < D_MODEL; d++) {
                w_grad[d] += d_out * hidden->data[b * D_MODEL + d];
                d_hidden[b * D_MODEL + d] += d_out * wout->data[f * D_MODEL + d];
            }
        }
    }
    
    double* layer_grads = grads + ws->size + wc->size + wout->size;
    for (int l = N_LAYERS - 1; l >= 0; l--) {
        memcpy(d_temp, d_hidden, BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
        feedforward_backward(d_hidden, layer_grads + l * (wf1[0].size + wf2[0].size),
                           layer_grads + l * (wf1[0].size + wf2[0].size) + wf1[0].size,
                           d_temp, hidden->data, wf1[l].data, wf2[l].data, mid_buf, d_mid);
        
        rmsnorm_backward(d_temp, d_hidden, hidden->data, hidden->data, D_MODEL);
        
        memcpy(d_hidden, d_temp, BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
        attention_backward(q_buf, k_buf, v_buf,
                         layer_grads + l * (wq[0].size + wk[0].size + wv[0].size + wo[0].size),
                         layer_grads + l * (wq[0].size + wk[0].size + wv[0].size + wo[0].size) + wq[0].size,
                         layer_grads + l * (wq[0].size + wk[0].size + wv[0].size + wo[0].size) + 
                             wq[0].size + wk[0].size,
                         layer_grads + l * (wq[0].size + wk[0].size + wv[0].size + wo[0].size) + 
                             wq[0].size + wk[0].size + wv[0].size,
                         d_temp, d_hidden, q_buf, k_buf, v_buf, s_buf, hidden->data,
                         wq[l].data, wk[l].data, wv[l].data, wo[l].data);
        
        rmsnorm_backward(d_hidden, d_temp, hidden->data, hidden->data, D_MODEL);
    }
    
    for (int b = 0; b < BATCH_SIZE * SEQ_LENGTH; b++) {
        const double* x = batch_data + b * INPUT_FEATURES;
        double* w_grad_s = grads, *w_grad_c = grads + ws->size;
        for (int d = 0; d < D_MODEL; d++) {
            for (int f = 0; f < SEQUENCE_FEATURES; f++)
                w_grad_s[f * D_MODEL + d] += d_hidden[b * D_MODEL + d] * x[f + CONDITION_FEATURES];
            for (int f = 0; f < CONDITION_FEATURES; f++)
                w_grad_c[f * D_MODEL + d] += d_hidden[b * D_MODEL + d] * x[f];
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
    
    double *batch_data = malloc(BATCH_SIZE * (SEQ_LENGTH + 1) * INPUT_FEATURES * sizeof(double));
    double *grads = malloc(total_params * sizeof(double));
    double *d_hidden = malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double *d_temp = malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double *q_buf = malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double *k_buf = malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double *v_buf = malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double *s_buf = malloc(BATCH_SIZE * N_HEAD * SEQ_LENGTH * SEQ_LENGTH * sizeof(double));
    double *mid_buf = malloc(BATCH_SIZE * SEQ_LENGTH * (D_MODEL * 4) * sizeof(double));
    double *d_mid = malloc(BATCH_SIZE * SEQ_LENGTH * (D_MODEL * 4) * sizeof(double));

    const double beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.01;

    for (int step = 0; step < TRAINING_STEPS; step++) {
        for (int b = 0; b < BATCH_SIZE; b++) {
            int seq_start = rand() % (ds->rows - SEQ_LENGTH - 1);
            memcpy(batch_data + b * (SEQ_LENGTH + 1) * INPUT_FEATURES,
                   ds->data + seq_start * INPUT_FEATURES,
                   (SEQ_LENGTH + 1) * INPUT_FEATURES * sizeof(double));
        }

        forward_pass(batch_data, out, hidden, temp, ws, wc, wq, wk, wv, wo, wf1, wf2, wout,
                    q_buf, k_buf, v_buf, s_buf, mid_buf);
        double loss = compute_loss(out, batch_data);
        
        if (step > 0) {
            lr *= (loss > prev_loss * 1.1) ? 0.95 : (loss < prev_loss * 0.95) ? 1.05 : 1.0;
            lr = fmax(1e-6, fmin(1e-3, lr));
        }
        prev_loss = loss;
        
        printf("Step %d, Loss: %f, LR: %e\n", step, loss, lr);
        if (f) fprintf(f, "%d,%f\n", step, loss);

        backward_pass(grads, batch_data, out, hidden, ws, wc, wq, wk, wv, wo, wf1, wf2, wout,
                     d_hidden, d_temp, q_buf, k_buf, v_buf, s_buf, mid_buf, d_mid);
        
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
    
    free(batch_data); free(grads); free(d_hidden); free(d_temp);
    free(q_buf); free(k_buf); free(v_buf); free(s_buf); free(mid_buf); free(d_mid);
    if (f) fclose(f);
}

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 3 || !strstr(argv[1], ".csv")) {
        printf("Usage: %s <training_data.csv> [weights.bin]\n", argv[0]);
        return 1;
    }

    char *csv_file = argv[1];
    char *weights_file = argc > 2 ? (strstr(argv[2], ".bin") ? argv[2] : NULL) : NULL;
    if (argc > 2 && !weights_file) {
        printf("Error: Invalid file '%s'\n", argv[2]);
        return 1;
    }

    srand(time(NULL));
    Dataset ds = load_csv(csv_file);
    const double ws = sqrt(2.0 / D_MODEL);
    
    Tensor W_seq = {malloc(SEQUENCE_FEATURES * D_MODEL * sizeof(double)), calloc(SEQUENCE_FEATURES * D_MODEL, sizeof(double)), calloc(SEQUENCE_FEATURES * D_MODEL, sizeof(double)), SEQUENCE_FEATURES * D_MODEL};
    Tensor W_cond = {malloc(CONDITION_FEATURES * D_MODEL * sizeof(double)), calloc(CONDITION_FEATURES * D_MODEL, sizeof(double)), calloc(CONDITION_FEATURES * D_MODEL, sizeof(double)), CONDITION_FEATURES * D_MODEL};
    Tensor W_out = {malloc(D_MODEL * SEQUENCE_FEATURES * sizeof(double)), calloc(D_MODEL * SEQUENCE_FEATURES, sizeof(double)), calloc(D_MODEL * SEQUENCE_FEATURES, sizeof(double)), D_MODEL * SEQUENCE_FEATURES};
    
    Tensor W_q[N_LAYERS], W_k[N_LAYERS], W_v[N_LAYERS], W_o[N_LAYERS], W_ff1[N_LAYERS], W_ff2[N_LAYERS];
    for (int l = 0; l < N_LAYERS; l++) {
        const int attn_size = D_MODEL * D_MODEL;
        const int ff_size1 = D_MODEL * (D_MODEL * 4);
        const int ff_size2 = (D_MODEL * 4) * D_MODEL;
        
        W_q[l] = (Tensor){malloc(attn_size * sizeof(double)), calloc(attn_size, sizeof(double)), calloc(attn_size, sizeof(double)), attn_size};
        W_k[l] = (Tensor){malloc(attn_size * sizeof(double)), calloc(attn_size, sizeof(double)), calloc(attn_size, sizeof(double)), attn_size};
        W_v[l] = (Tensor){malloc(attn_size * sizeof(double)), calloc(attn_size, sizeof(double)), calloc(attn_size, sizeof(double)), attn_size};
        W_o[l] = (Tensor){malloc(attn_size * sizeof(double)), calloc(attn_size, sizeof(double)), calloc(attn_size, sizeof(double)), attn_size};
        W_ff1[l] = (Tensor){malloc(ff_size1 * sizeof(double)), calloc(ff_size1, sizeof(double)), calloc(ff_size1, sizeof(double)), ff_size1};
        W_ff2[l] = (Tensor){malloc(ff_size2 * sizeof(double)), calloc(ff_size2, sizeof(double)), calloc(ff_size2, sizeof(double)), ff_size2};
    }

    if (weights_file && load_weights(weights_file, &W_seq, &W_cond, W_q, W_k, W_v, W_o, W_ff1, W_ff2, &W_out)) {
        printf("Successfully loaded weights\n");
    } else {
        if (weights_file) printf("Failed to load weights, initializing randomly\n");
        for (int i = 0; i < W_seq.size; i++) W_seq.data[i] = randn() * ws;
        for (int i = 0; i < W_cond.size; i++) W_cond.data[i] = randn() * ws;
        for (int i = 0; i < W_out.size; i++) W_out.data[i] = randn() * ws;
        
        for (int l = 0; l < N_LAYERS; l++) {
            for (int i = 0; i < W_q[l].size; i++) {
                W_q[l].data[i] = randn() * ws;
                W_k[l].data[i] = randn() * ws;
                W_v[l].data[i] = randn() * ws;
                W_o[l].data[i] = randn() * ws;
            }
            for (int i = 0; i < W_ff1[l].size; i++) W_ff1[l].data[i] = randn() * ws;
            for (int i = 0; i < W_ff2[l].size; i++) W_ff2[l].data[i] = randn() * ws;
        }
    }

    Tensor hidden = {malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double)), NULL, NULL, BATCH_SIZE * SEQ_LENGTH * D_MODEL};
    Tensor temp = {malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double)), NULL, NULL, BATCH_SIZE * SEQ_LENGTH * D_MODEL};
    Tensor output = {malloc(BATCH_SIZE * SEQ_LENGTH * SEQUENCE_FEATURES * sizeof(double)), NULL, NULL, BATCH_SIZE * SEQ_LENGTH * SEQUENCE_FEATURES};
    
    train_backprop(&ds, &output, &hidden, &temp, &W_seq, &W_cond, W_q, W_k, W_v, W_o, W_ff1, W_ff2, &W_out);
    
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    char filename[100];
    sprintf(filename, "%d-%d-%d_%d-%d-%d_weights.bin", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    
    save_weights(filename, &W_seq, &W_cond, W_q, W_k, W_v, W_o, W_ff1, W_ff2, &W_out);
    printf("Saved weights to: %s\n", filename);

    free(ds.data);
    free(hidden.data); free(temp.data); free(output.data);
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