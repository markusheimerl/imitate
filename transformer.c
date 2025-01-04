#include "transformer.h"

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
                double max = -INFINITY, sum = 0.0;
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
                for (int j = 0; j <= i; j++) s[(b * N_HEAD * SEQ_LENGTH + h * SEQ_LENGTH + i) * SEQ_LENGTH + j] /= sum;
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

void update_weights(Tensor* w, double base_loss, double lr, const double* batch_data, 
                   Tensor* out, Tensor* hidden, Tensor* temp,
                   const Tensor* ws, const Tensor* wc, const Tensor* wq, const Tensor* wk, 
                   const Tensor* wv, const Tensor* wo, const Tensor* wf1, const Tensor* wf2, 
                   const Tensor* wout,
                   double* q_buf, double* k_buf, double* v_buf, double* s_buf, 
                   double* mid_buf) {
    for (int i = 0; i < w->size; i++) {
        w->data[i] += EPSILON;
        forward_pass(batch_data, out, hidden, temp, ws, wc, wq, wk, wv, wo, wf1, wf2, wout, q_buf, k_buf, v_buf, s_buf, mid_buf);
        double new_loss = compute_loss(out, batch_data);
        w->data[i] -= EPSILON;
        
        if (!isnan(new_loss)) {
            double grad = (new_loss - base_loss) / EPSILON;
            grad = fmax(-10.0, fmin(10.0, grad));
            w->data[i] -= lr * grad;
        }
    }
}

void train_finite_diff(Dataset* ds, Tensor* out, Tensor* hidden, Tensor* temp,
                      Tensor* ws, Tensor* wc, Tensor* wq, Tensor* wk, Tensor* wv, 
                      Tensor* wo, Tensor* wf1, Tensor* wf2, Tensor* wout) {
    const double lr = LEARNING_RATE;
    
    // Allocate buffers
    double* batch_data = malloc(BATCH_SIZE * (SEQ_LENGTH + 1) * INPUT_FEATURES * sizeof(double));
    double* q_buf = malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double* k_buf = malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double* v_buf = malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double* s_buf = malloc(BATCH_SIZE * N_HEAD * SEQ_LENGTH * SEQ_LENGTH * sizeof(double));
    double* mid_buf = malloc(BATCH_SIZE * SEQ_LENGTH * (D_MODEL * 4) * sizeof(double));
    
    Tensor* global_weights[] = {ws, wc, wout};
    
    for (int step = 0; step < TRAINING_STEPS; step++) {
        // Sample random batches
        for (int b = 0; b < BATCH_SIZE; b++) {
            int seq_start = rand() % (ds->rows - SEQ_LENGTH - 1);
            for (int s = 0; s < SEQ_LENGTH + 1; s++)
                for (int f = 0; f < INPUT_FEATURES; f++)
                    batch_data[(b * (SEQ_LENGTH + 1) + s) * INPUT_FEATURES + f] = 
                        ds->data[(seq_start + s) * INPUT_FEATURES + f];
        }

        forward_pass(batch_data, out, hidden, temp, ws, wc, wq, wk, wv, wo, wf1, wf2, wout, q_buf, k_buf, v_buf, s_buf, mid_buf);
        double base_loss = compute_loss(out, batch_data);
        
        if (isnan(base_loss)) {
            printf("NaN detected at step %d, skipping update\n", step);
            continue;
        }
        printf("Step %d, Loss: %f\n", step, base_loss);

        // Update per-layer weights
        for (int l = 0; l < N_LAYERS; l++) {
            Tensor* layer_weights[] = {&wq[l], &wk[l], &wv[l], &wo[l], &wf1[l], &wf2[l]};
            for (int w = 0; w < 6; w++) {
                update_weights(layer_weights[w], base_loss, lr, batch_data, out, hidden, temp, ws, wc, wq, wk, wv, wo, wf1, wf2, wout, q_buf, k_buf, v_buf, s_buf, mid_buf);
            }
        }
        
        // Update global weights
        for (int w = 0; w < 3; w++) {
            update_weights(global_weights[w], base_loss, lr, batch_data, out, hidden, temp, ws, wc, wq, wk, wv, wo, wf1, wf2, wout, q_buf, k_buf, v_buf, s_buf, mid_buf);
        }

        // Print predictions periodically
        if (step > 0 && step % 100 == 0) {
            printf("\nPredictions at step %d:\n", step);
            for (int s = 0; s < 5; s++) {
                printf("Step %d: ", s);
                for (int f = 0; f < SEQUENCE_FEATURES; f++) {
                    double pred = denormalize(out->data[s * SEQUENCE_FEATURES + f], ds->mins[f + CONDITION_FEATURES], ds->maxs[f + CONDITION_FEATURES]);
                    double actual = denormalize(batch_data[(s + 1) * INPUT_FEATURES + f + CONDITION_FEATURES], ds->mins[f + CONDITION_FEATURES], ds->maxs[f + CONDITION_FEATURES]);
                    printf("F%d(P:%.2f,A:%.2f) ", f, pred, actual);
                }
                printf("\n");
            }
            printf("\n");
        }
    }
    
    free(batch_data); free(q_buf); free(k_buf); free(v_buf); free(s_buf); free(mid_buf);
}

int main() {
    srand(time(NULL));
    Dataset ds = load_csv("2024-12-29_6-25-1_control_data.csv");
    const double ws = sqrt(2.0 / D_MODEL);
    
    // Initialize global weights
    Tensor W_seq = {malloc(SEQUENCE_FEATURES * D_MODEL * sizeof(double)), SEQUENCE_FEATURES * D_MODEL};
    Tensor W_cond = {malloc(CONDITION_FEATURES * D_MODEL * sizeof(double)), CONDITION_FEATURES * D_MODEL};
    Tensor W_out = {malloc(D_MODEL * SEQUENCE_FEATURES * sizeof(double)), D_MODEL * SEQUENCE_FEATURES};
    
    // Initialize per-layer weights
    Tensor W_q[N_LAYERS], W_k[N_LAYERS], W_v[N_LAYERS], W_o[N_LAYERS], W_ff1[N_LAYERS], W_ff2[N_LAYERS];
    for (int l = 0; l < N_LAYERS; l++) {
        const int attn_size = D_MODEL * D_MODEL;
        const int ff_size1 = D_MODEL * (D_MODEL * 4);
        const int ff_size2 = (D_MODEL * 4) * D_MODEL;
        
        W_q[l] = (Tensor){malloc(attn_size * sizeof(double)), attn_size};
        W_k[l] = (Tensor){malloc(attn_size * sizeof(double)), attn_size};
        W_v[l] = (Tensor){malloc(attn_size * sizeof(double)), attn_size};
        W_o[l] = (Tensor){malloc(attn_size * sizeof(double)), attn_size};
        W_ff1[l] = (Tensor){malloc(ff_size1 * sizeof(double)), ff_size1};
        W_ff2[l] = (Tensor){malloc(ff_size2 * sizeof(double)), ff_size2};
        
        // Initialize attention weights
        for (int i = 0; i < attn_size; i++) W_q[l].data[i] = W_k[l].data[i] = W_v[l].data[i] = W_o[l].data[i] = randn() * ws;
        
        // Initialize feedforward weights
        for (int i = 0; i < ff_size1; i++) W_ff1[l].data[i] = randn() * ws;
        for (int i = 0; i < ff_size2; i++) W_ff2[l].data[i] = randn() * ws;
    }
    
    // Initialize global weights
    for (int i = 0; i < W_seq.size; i++) W_seq.data[i] = randn() * ws;
    for (int i = 0; i < W_cond.size; i++) W_cond.data[i] = randn() * ws;
    for (int i = 0; i < W_out.size; i++) W_out.data[i] = randn() * ws;
    
    // Initialize buffers
    Tensor hidden = {malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double)), BATCH_SIZE * SEQ_LENGTH * D_MODEL};
    Tensor temp = {malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double)), BATCH_SIZE * SEQ_LENGTH * D_MODEL};
    Tensor output = {malloc(BATCH_SIZE * SEQ_LENGTH * SEQUENCE_FEATURES * sizeof(double)), BATCH_SIZE * SEQ_LENGTH * SEQUENCE_FEATURES};

    // Train model
    train_finite_diff(&ds, &output, &hidden, &temp, &W_seq, &W_cond, W_q, W_k, W_v, W_o, W_ff1, W_ff2, &W_out);
    
    // Save weights
    save_weights("weights.bin", &W_seq, &W_cond, W_q, W_k, W_v, W_o, W_ff1, W_ff2, &W_out);

    // Cleanup
    free(ds.data); free(ds.mins); free(ds.maxs);
    free(hidden.data); free(temp.data); free(output.data);
    free(W_seq.data); free(W_cond.data); free(W_out.data);
    
    for (int l = 0; l < N_LAYERS; l++) {
        free(W_q[l].data); free(W_k[l].data); free(W_v[l].data);
        free(W_o[l].data); free(W_ff1[l].data); free(W_ff2[l].data);
    }
    
    return 0;
}