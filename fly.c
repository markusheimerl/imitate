#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <stdbool.h>
#include "quad.h"
#include "gif.h"
#include "rasterizer.h"

#define CONDITION_FEATURES 4
#define SEQUENCE_FEATURES 10
#define INPUT_FEATURES (CONDITION_FEATURES + SEQUENCE_FEATURES)
#define BATCH_SIZE 8
#define SEQ_LENGTH 16
#define D_MODEL 16
#define N_HEAD 2
#define N_LAYERS 2
#define EPSILON 1e-6
#define LEARNING_RATE 0.01
#define TRAINING_STEPS 10

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

void update_weights(Tensor* w, double base_loss, int step, double lr, const double* batch_data, 
                   Tensor* out, Tensor* hidden, Tensor* temp,
                   const Tensor* ws, const Tensor* wc, const Tensor* wq, const Tensor* wk, 
                   const Tensor* wv, const Tensor* wo, const Tensor* wf1, const Tensor* wf2, 
                   const Tensor* wout,
                   double* q_buf, double* k_buf, double* v_buf, double* s_buf, 
                   double* mid_buf) {
    const double beta1 = 0.9;
    const double beta2 = 0.999;
    const double eps = 1e-8;
    const double weight_decay = 0.01;

    #pragma omp parallel for
    for (int i = 0; i < w->size; i++) {
        w->data[i] += EPSILON;
        forward_pass(batch_data, out, hidden, temp, ws, wc, wq, wk, wv, wo, wf1, wf2, wout, q_buf, k_buf, v_buf, s_buf, mid_buf);
        double new_loss = compute_loss(out, batch_data);
        w->data[i] -= EPSILON;
        
        double grad = (new_loss - base_loss) / EPSILON;
        grad = fmax(-10.0, fmin(10.0, grad));
        w->m[i] = beta1 * w->m[i] + (1 - beta1) * grad;
        w->v[i] = beta2 * w->v[i] + (1 - beta2) * grad * grad;
        double m_hat = w->m[i] / (1.0 - pow(beta1, step + 1));
        double v_hat = w->v[i] / (1.0 - pow(beta2, step + 1));
        w->data[i] = w->data[i] * (1.0 - lr * weight_decay) - lr * m_hat / (sqrt(v_hat) + eps);
    }
}

void train_finite_diff(Dataset* ds, Tensor* out, Tensor* hidden, Tensor* temp,
                      Tensor* ws, Tensor* wc, Tensor* wq, Tensor* wk, Tensor* wv, 
                      Tensor* wo, Tensor* wf1, Tensor* wf2, Tensor* wout) {
    const double lr = LEARNING_RATE;

    time_t t = time(NULL);
    struct tm *tm = localtime(&t);
    char loss_name[100];
    sprintf(loss_name, "%d-%d-%d_%d-%d-%d_loss.csv", tm->tm_year+1900, tm->tm_mon+1, tm->tm_mday, tm->tm_hour, tm->tm_min, tm->tm_sec);
    FILE* f = fopen(loss_name, "w");

    // Allocate buffers
    double* batch_data = malloc(BATCH_SIZE * (SEQ_LENGTH + 1) * INPUT_FEATURES * sizeof(double));
    double* q_buf = malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double* k_buf = malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double* v_buf = malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double* s_buf = malloc(BATCH_SIZE * N_HEAD * SEQ_LENGTH * SEQ_LENGTH * sizeof(double));
    double* mid_buf = malloc(BATCH_SIZE * SEQ_LENGTH * (D_MODEL * 4) * sizeof(double));
    
    Tensor* global_weights[] = {ws, wc, wout};
    
    for (int step = 0; step < TRAINING_STEPS; step++) {
        for (int b = 0; b < BATCH_SIZE; b++) {
            int seq_start = rand() % (ds->rows - SEQ_LENGTH - 1);
            for (int s = 0; s < SEQ_LENGTH + 1; s++)
                for (int f = 0; f < INPUT_FEATURES; f++)
                    batch_data[(b * (SEQ_LENGTH + 1) + s) * INPUT_FEATURES + f] = 
                        ds->data[(seq_start + s) * INPUT_FEATURES + f];
        }

        forward_pass(batch_data, out, hidden, temp, ws, wc, wq, wk, wv, wo, wf1, wf2, wout, q_buf, k_buf, v_buf, s_buf, mid_buf);
        double base_loss = compute_loss(out, batch_data);
        
        printf("Step %d, Loss: %f\n", step, base_loss);
        if (f) fprintf(f, "%d,%f\n", step, base_loss);

        // Update per-layer weights
        for (int l = 0; l < N_LAYERS; l++) {
            Tensor* layer_weights[] = {&wq[l], &wk[l], &wv[l], &wo[l], &wf1[l], &wf2[l]};
            for (int w = 0; w < 6; w++) {
                update_weights(layer_weights[w], base_loss, step, lr, batch_data, out, hidden, temp, ws, wc, wq, wk, wv, wo, wf1, wf2, wout, q_buf, k_buf, v_buf, s_buf, mid_buf);
            }
        }
        
        // Update global weights
        for (int w = 0; w < 3; w++) {
            update_weights(global_weights[w], base_loss, step, lr, batch_data, out, hidden, temp, ws, wc, wq, wk, wv, wo, wf1, wf2, wout, q_buf, k_buf, v_buf, s_buf, mid_buf);
        }

        // Print predictions periodically
        if (step > 0 && step % 100 == 0) {
            printf("\nPredictions at step %d:\n", step);
            for (int s = 0; s < 5; s++) {
                printf("Step %d: ", s);
                for (int f = 0; f < SEQUENCE_FEATURES; f++) {
                    double pred = out->data[s * SEQUENCE_FEATURES + f];
                    double actual = batch_data[(s + 1) * INPUT_FEATURES + f + CONDITION_FEATURES];
                    printf("F%d(P:%.2f,A:%.2f) ", f, pred, actual);
                }
                printf("\n");
            }
            printf("\n");
        }
    }
    
    free(batch_data); free(q_buf); free(k_buf); free(v_buf); free(s_buf); free(mid_buf); fclose(f);
}

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define DT_RENDER   (1.0 / 30.0)

static double history[SEQ_LENGTH][INPUT_FEATURES];
static int history_len = 0;

static bool is_stable(void) {
    for (int i = 0; i < 3; i++) if (fabs(angular_velocity_B[i]) > 0.005) return false;
    return true;
}

static bool is_at_target_position(void) {
    for (int i = 0; i < 3; i++) if (fabs(linear_position_W[i] - linear_position_d_W[i]) > 0.1) return false;
    return true;
}

static bool check_divergence(void) {
    for (int i = 0; i < 3; i++) {
        if (fabs(linear_position_W[i]) > 1000.0 || fabs(linear_velocity_W[i]) > 100.0 || 
            fabs(angular_velocity_B[i]) > 100.0) return true;
    }
    for (int i = 0; i < 4; i++) if (omega_next[i] < 0 || omega_next[i] > 1000) return true;
    return false;
}

static void update_history(void) {
    if (history_len == SEQ_LENGTH) {
        for (int i = 0; i < SEQ_LENGTH - 1; i++)
            for (int j = 0; j < INPUT_FEATURES; j++)
                history[i][j] = history[i + 1][j];
    } else history_len++;
    
    int idx = 0;
    for (int i = 0; i < 3; i++) history[history_len-1][idx++] = linear_position_d_W[i];
    history[history_len-1][idx++] = yaw_d;
    for (int i = 0; i < 3; i++) history[history_len-1][idx++] = angular_velocity_B[i];
    for (int i = 0; i < 3; i++) history[history_len-1][idx++] = linear_acceleration_B[i];
    for (int i = 0; i < 4; i++) history[history_len-1][idx++] = omega[i];
}

static void prepare_transformer_input(double* input) {
    for (int s = 0; s < SEQ_LENGTH; s++) {
        if (s < history_len) {
            for (int f = 0; f < INPUT_FEATURES; f++)
                input[s * INPUT_FEATURES + f] = history[s][f];
        } else {
            for (int f = 0; f < INPUT_FEATURES; f++)
                input[s * INPUT_FEATURES + f] = 0.0;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) { printf("Usage: %s <weights_file>\n", argv[0]); return 1; }

    Tensor W_seq = {malloc(SEQUENCE_FEATURES * D_MODEL * sizeof(double)), NULL, NULL, SEQUENCE_FEATURES * D_MODEL};
    Tensor W_cond = {malloc(CONDITION_FEATURES * D_MODEL * sizeof(double)), NULL, NULL, CONDITION_FEATURES * D_MODEL};
    Tensor W_out = {malloc(D_MODEL * SEQUENCE_FEATURES * sizeof(double)), NULL, NULL, D_MODEL * SEQUENCE_FEATURES};
    
    Tensor W_q[N_LAYERS], W_k[N_LAYERS], W_v[N_LAYERS], W_o[N_LAYERS], W_ff1[N_LAYERS], W_ff2[N_LAYERS];
    for (int l = 0; l < N_LAYERS; l++) {
        const int attn_size = D_MODEL * D_MODEL;
        const int ff_size1 = D_MODEL * (D_MODEL * 4);
        const int ff_size2 = (D_MODEL * 4) * D_MODEL;
        W_q[l] = (Tensor){malloc(attn_size * sizeof(double)), NULL, NULL, attn_size};
        W_k[l] = (Tensor){malloc(attn_size * sizeof(double)), NULL, NULL, attn_size};
        W_v[l] = (Tensor){malloc(attn_size * sizeof(double)), NULL, NULL, attn_size};
        W_o[l] = (Tensor){malloc(attn_size * sizeof(double)), NULL, NULL, attn_size};
        W_ff1[l] = (Tensor){malloc(ff_size1 * sizeof(double)), NULL, NULL, ff_size1};
        W_ff2[l] = (Tensor){malloc(ff_size2 * sizeof(double)), NULL, NULL, ff_size2};
    }

    if (!load_weights(argv[1], &W_seq, &W_cond, W_q, W_k, W_v, W_o, W_ff1, W_ff2, &W_out)) {
        printf("Failed to load weights from %s\n", argv[1]); return 1;
    }

    double *transformer_input = malloc(BATCH_SIZE * (SEQ_LENGTH + 1) * INPUT_FEATURES * sizeof(double));
    Tensor hidden = {malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double)), NULL, NULL, BATCH_SIZE * SEQ_LENGTH * D_MODEL};
    Tensor temp = {malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double)), NULL, NULL, BATCH_SIZE * SEQ_LENGTH * D_MODEL};
    Tensor output = {malloc(BATCH_SIZE * SEQ_LENGTH * SEQUENCE_FEATURES * sizeof(double)), NULL, NULL, BATCH_SIZE * SEQ_LENGTH * SEQUENCE_FEATURES};
    double *q_buf = malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double *k_buf = malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double *v_buf = malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double *s_buf = malloc(BATCH_SIZE * N_HEAD * SEQ_LENGTH * SEQ_LENGTH * sizeof(double));
    double *mid_buf = malloc(BATCH_SIZE * SEQ_LENGTH * (D_MODEL * 4) * sizeof(double));

    Mesh* meshes[] = {create_mesh("sim/rasterizer/drone.obj", "sim/rasterizer/drone.bmp"), 
                      create_mesh("sim/rasterizer/ground.obj", "sim/rasterizer/ground.bmp")};
    uint8_t *frame_buffer = calloc(WIDTH * HEIGHT * 3, sizeof(uint8_t));
    time_t t = time(NULL);
    struct tm *tm = localtime(&t);
    char gif_name[32];
    sprintf(gif_name, "%d-%d-%d_%d-%d-%d_flight.gif", tm->tm_year+1900, tm->tm_mon+1, tm->tm_mday, tm->tm_hour, tm->tm_min, tm->tm_sec);
    ge_GIF *gif = ge_new_gif(gif_name, WIDTH, HEIGHT, 4, -1, 0);
    transform_mesh(meshes[1], (double[3]){0.0, -0.5, 0.0}, 1.0, (double[9]){1,0,0, 0,1,0, 0,0,1});

    double t_physics = 0.0, t_control = 0.0, t_render = 0.0;
    linear_position_d_W[0] = 2.0; linear_position_d_W[1] = 2.0; linear_position_d_W[2] = 2.0; yaw_d = M_PI / 4;

    while (!is_stable() || !is_at_target_position()) {
        if (check_divergence()) { printf("Simulation diverged.\n"); return 1; }

        update_drone_physics(DT_PHYSICS);
        t_physics += DT_PHYSICS;
        
        if (t_control <= t_physics) {
            update_history();
            if (history_len == SEQ_LENGTH) {
                prepare_transformer_input(transformer_input);
                forward_pass(transformer_input, &output, &hidden, &temp,
                           &W_seq, &W_cond, W_q, W_k, W_v, W_o, W_ff1, W_ff2, &W_out,
                           q_buf, k_buf, v_buf, s_buf, mid_buf);
                int last_seq = (SEQ_LENGTH - 1) * SEQUENCE_FEATURES;
                for (int i = 0; i < 4; i++) omega_next[i] = output.data[last_seq + i + 6];
            }
            update_rotor_speeds();
            t_control += DT_CONTROL;
        }

        if (t_render <= t_physics) {
            transform_mesh(meshes[0], linear_position_W, 0.5, R_W_B);
            memset(frame_buffer, 0, WIDTH * HEIGHT * 3);
            vertex_shader(meshes, 2, (double[3]){-2.0, 2.0, -2.0}, (double[3]){0.0, 0.0, 0.0});
            rasterize(frame_buffer, meshes, 2);
            ge_add_frame(gif, frame_buffer, 6);
            t_render += DT_RENDER;
        }
        printf("Position: [%.3f, %.3f, %.3f]\nTarget: [%.3f, %.3f, %.3f]\nRotor speeds: [%.1f, %.1f, %.1f, %.1f]\n---\n",
               linear_position_W[0], linear_position_W[1], linear_position_W[2],
               linear_position_d_W[0], linear_position_d_W[1], linear_position_d_W[2],
               omega[0], omega[1], omega[2], omega[3]);
    }

    free(frame_buffer); free_meshes(meshes, 2); ge_close_gif(gif);
    free(transformer_input); free(hidden.data); free(temp.data); free(output.data);
    free(q_buf); free(k_buf); free(v_buf); free(s_buf); free(mid_buf);
    free(W_seq.data); free(W_cond.data); free(W_out.data);
    for (int l = 0; l < N_LAYERS; l++) {
        free(W_q[l].data); free(W_k[l].data); free(W_v[l].data); free(W_o[l].data);
        free(W_ff1[l].data); free(W_ff2[l].data);
    }
    return 0;
}