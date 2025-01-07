#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <omp.h>

#define CONDITION_FEATURES 4
#define SEQUENCE_FEATURES 10
#define SEQ_LENGTH 64
#define D_MODEL 64
#define N_HEAD 4
#define N_LAYERS 4
#define LEARNING_RATE 0.00001
#define TRAINING_STEPS 10000

typedef struct { double *data; int rows, cols; } Dataset;

double randn() { return sqrt(-2.0 * log((double)rand() / RAND_MAX)) * cos(2.0 * M_PI * (double)rand() / RAND_MAX); }

Dataset load_csv(const char* filename) {
    Dataset ds = {malloc(1000 * (CONDITION_FEATURES + SEQUENCE_FEATURES) * sizeof(double)), 0, (CONDITION_FEATURES + SEQUENCE_FEATURES)};
    char line[1024]; FILE* f = fopen(filename, "r");
    if (!f || !fgets(line, 1024, f)) { printf("File error\n"); exit(1); }
    
    while (fgets(line, 1024, f)) {
        if (ds.rows >= 1000) ds.data = realloc(ds.data, (ds.rows*2) * (CONDITION_FEATURES + SEQUENCE_FEATURES) * sizeof(double));
        char* tok = strtok(line, ",");
        for (int i = 0; i < (CONDITION_FEATURES + SEQUENCE_FEATURES) && tok; i++, tok = strtok(NULL, ",")) ds.data[ds.rows * (CONDITION_FEATURES + SEQUENCE_FEATURES) + i] = atof(tok);
        ds.rows++;
    }
    fclose(f);
    return ds;
}

void save_weights(const double* ws, const double* wc, const double* wq, const double* wk, const double* wv, const double* wo, const double* wf1, const double* wf2, const double* wout) {
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    char filename[100]; 
    sprintf(filename, "%d-%d-%d_%d-%d-%d_weights.bin", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    
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
    printf("Saved weights to: %s\n", filename);
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

// Given input X of shape [seq_len, d_model]
// RMSNorm(X)_s,d = X_s,d / sqrt(1/D * sum_i(X_s,i^2) + eps)
// where s=sequence index, d=dimension index
void rmsnorm(double *out, const double *in) {
    #pragma omp parallel for
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* x = in + s * D_MODEL;
        double* y = out + s * D_MODEL;
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
void feedforward(double *out, const double *w1, const double *w2, const double *in, double *mid) {
    #pragma omp parallel for
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* x = in + s * D_MODEL;
        double* y = out + s * D_MODEL;
        double* u = mid + s * (D_MODEL * 4);
        
        for (int i = 0; i < D_MODEL * 4; i++) {
            u[i] = 0.0;
            for (int j = 0; j < D_MODEL; j++) u[i] += x[j] * w1[i * D_MODEL + j];
            u[i] = 0.5 * u[i] * (1.0 + tanh(sqrt(2.0/M_PI) * u[i] + 0.044715 * u[i] * u[i] * u[i]));
        }
        
        for (int i = 0; i < D_MODEL; i++) {
            y[i] = 0.0;
            for (int j = 0; j < D_MODEL * 4; j++) y[i] += u[j] * w2[i * (D_MODEL * 4) + j];
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
void multihead_attention(double *out, const double *in, const double *wq, const double *wk, const double *wv, const double *wo, double *q, double *k, double *v, double *s) {
    const int hd = D_MODEL / N_HEAD;
    const double scale = 1.0 / sqrt(hd);

    #pragma omp parallel for
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* x = in + s * D_MODEL;
        for (int h = 0; h < N_HEAD; h++)
            for (int d = 0; d < hd; d++) {
                double sq = 0.0, sk = 0.0, sv = 0.0;
                const int w_idx = (h * hd + d) * D_MODEL;
                for (int i = 0; i < D_MODEL; i++) sq += x[i] * wq[w_idx + i], sk += x[i] * wk[w_idx + i], sv += x[i] * wv[w_idx + i];
                const int qkv_idx = s * D_MODEL + h * hd + d;
                q[qkv_idx] = sq, k[qkv_idx] = sk, v[qkv_idx] = sv;
            }
    }

    #pragma omp parallel for
    for (int h = 0; h < N_HEAD; h++) {
        const double slope = pow(2.0, -(8.0 * (h + 1) / N_HEAD));
        for (int i = 0; i < SEQ_LENGTH; i++) {
            double max = -1e9, sum = 0.0;
            for (int j = 0; j <= i; j++) {
                double dot = 0.0;
                for (int d = 0; d < hd; d++) dot += q[i * D_MODEL + h * hd + d] * k[j * D_MODEL + h * hd + d];
                s[(h * SEQ_LENGTH + i) * SEQ_LENGTH + j] = dot * scale - slope * (i - j);
                max = fmax(max, s[(h * SEQ_LENGTH + i) * SEQ_LENGTH + j]);
            }
            for (int j = 0; j <= i; j++) {
                s[(h * SEQ_LENGTH + i) * SEQ_LENGTH + j] = exp(s[(h * SEQ_LENGTH + i) * SEQ_LENGTH + j] - max);
                sum += s[(h * SEQ_LENGTH + i) * SEQ_LENGTH + j];
            }
            for (int j = 0; j <= i; j++) s[(h * SEQ_LENGTH + i) * SEQ_LENGTH + j] /= (sum + 1e-10);
        }
    }

    #pragma omp parallel for
    for (int t = 0; t < SEQ_LENGTH; t++) {
        double tmp[D_MODEL] = {0};
        for (int h = 0; h < N_HEAD; h++)
            for (int d = 0; d < hd; d++) {
                double sum = 0.0;
                for (int j = 0; j <= t; j++) sum += s[(h * SEQ_LENGTH + t) * SEQ_LENGTH + j] * v[j * D_MODEL + h * hd + d];
                tmp[h * hd + d] = sum;
            }
        for (int d = 0; d < D_MODEL; d++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++) sum += tmp[i] * wo[d * D_MODEL + i];
            out[t * D_MODEL + d] = sum;
        }
    }
}

// Forward pass through the transformer:
// 1. Input embedding: sequence_features * Ws + condition_features * Wc
// 2. N transformer layers of:
//    x = x + attention(rmsnorm(x))
//    x = x + ffn(rmsnorm(x))
// 3. Output projection to sequence_features
void forward_pass(const double* seq_data, double* out, double* hidden, double* temp, const double* ws, const double* wc, const double* wq, const double* wk, const double* wv, const double* wo, const double* wf1, const double* wf2, const double* wout, double* q_buf, double* k_buf, double* v_buf, double* s_buf, double* mid_buf) {
    #pragma omp parallel for
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* x = seq_data + s * (CONDITION_FEATURES + SEQUENCE_FEATURES);
        double* y = hidden + s * D_MODEL;
        for (int d = 0; d < D_MODEL; d++) {
            double sum = 0.0;
            for (int f = 0; f < SEQUENCE_FEATURES; f++) sum += x[f + CONDITION_FEATURES] * ws[f * D_MODEL + d];
            for (int f = 0; f < CONDITION_FEATURES; f++) sum += x[f] * wc[f * D_MODEL + d];
            y[d] = sum;
        }
    }

    for (int l = 0; l < N_LAYERS; l++) {
        const int layer_offset = l * D_MODEL * D_MODEL;
        const int ff_offset1 = l * D_MODEL * (D_MODEL * 4);
        const int ff_offset2 = l * (D_MODEL * 4) * D_MODEL;
        
        rmsnorm(temp, hidden);
        multihead_attention(temp, temp, wq + layer_offset, wk + layer_offset, wv + layer_offset, wo + layer_offset, q_buf, k_buf, v_buf, s_buf);
        for (int i = 0; i < SEQ_LENGTH * D_MODEL; i++) hidden[i] += temp[i];
        
        rmsnorm(temp, hidden);
        feedforward(temp, wf1 + ff_offset1, wf2 + ff_offset2, temp, mid_buf);
        for (int i = 0; i < SEQ_LENGTH * D_MODEL; i++) hidden[i] += temp[i];
    }

    #pragma omp parallel for
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* h = hidden + s * D_MODEL;
        double* o = out + s * SEQUENCE_FEATURES;
        for (int f = 0; f < SEQUENCE_FEATURES; f++) {
            double sum = 0.0;
            for (int d = 0; d < D_MODEL; d++) sum += h[d] * wout[f * D_MODEL + d];
            o[f] = sum;
        }
    }
}

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define DT_RENDER   (1.0 / 30.0)
#define VEC3_MAG2(v) ((v)[0]*(v)[0] + (v)[1]*(v)[1] + (v)[2]*(v)[2])

int main(int argc, char *argv[]) {
    if (argc != 2) { printf("Usage: %s <weights_file>\n", argv[0]); return 1; }

    srand(time(NULL));

    // Initialize transformer weights
    Tensor W_seq = {malloc(SEQUENCE_FEATURES * D_MODEL * sizeof(double)), NULL, NULL, SEQUENCE_FEATURES * D_MODEL};
    Tensor W_cond = {malloc(CONDITION_FEATURES * D_MODEL * sizeof(double)), NULL, NULL, CONDITION_FEATURES * D_MODEL};
    Tensor W_out = {malloc(D_MODEL * SEQUENCE_FEATURES * sizeof(double)), NULL, NULL, D_MODEL * SEQUENCE_FEATURES};
    
    Tensor W_q[N_LAYERS], W_k[N_LAYERS], W_v[N_LAYERS], W_o[N_LAYERS], W_ff1[N_LAYERS], W_ff2[N_LAYERS];
    for (int l = 0; l < N_LAYERS; l++) {
        const int attn_size = D_MODEL * D_MODEL, ff_size1 = D_MODEL * (D_MODEL * 4), ff_size2 = (D_MODEL * 4) * D_MODEL;
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

    // Allocate transformer buffers
    double *transformer_input = malloc(BATCH_SIZE * (SEQ_LENGTH + 1) * INPUT_FEATURES * sizeof(double));
    Tensor hidden = {malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double)), NULL, NULL, BATCH_SIZE * SEQ_LENGTH * D_MODEL};
    Tensor temp = {malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double)), NULL, NULL, BATCH_SIZE * SEQ_LENGTH * D_MODEL};
    Tensor output = {malloc(BATCH_SIZE * SEQ_LENGTH * SEQUENCE_FEATURES * sizeof(double)), NULL, NULL, BATCH_SIZE * SEQ_LENGTH * SEQUENCE_FEATURES};
    double *q_buf = malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double *k_buf = malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double *v_buf = malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double *s_buf = malloc(BATCH_SIZE * N_HEAD * SEQ_LENGTH * SEQ_LENGTH * sizeof(double));
    double *mid_buf = malloc(BATCH_SIZE * SEQ_LENGTH * (D_MODEL * 4) * sizeof(double));

    // Initialize visualization
    Mesh* meshes[] = {create_mesh("sim/rasterizer/drone.obj", "sim/rasterizer/drone.bmp"), create_mesh("sim/rasterizer/ground.obj", "sim/rasterizer/ground.bmp")};
    uint8_t *frame_buffer = calloc(WIDTH * HEIGHT * 3, sizeof(uint8_t));
    
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    char filename[100];
    sprintf(filename, "%d-%d-%d_%d-%d-%d_flight.gif", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    ge_GIF *gif = ge_new_gif(filename, WIDTH, HEIGHT, 4, -1, 0);
    transform_mesh(meshes[1], (double[3]){0.0, -0.5, 0.0}, 1.0, (double[9]){1,0,0, 0,1,0, 0,0,1});

    // Initialize simulation
    double t_physics = 0.0, t_control = 0.0, t_render = 0.0;
    int history_len = 0, target_count = 0;
    double history[SEQ_LENGTH][INPUT_FEATURES] = {0};

    while (target_count < 2) {
        if (history_len == SEQ_LENGTH) {
            bool velocity_achieved = true;
            for (int i = 0; i < 3; i++) {
                if (fabs(angular_velocity_B[i]) > 0.01 || fabs(linear_velocity_B[i] - linear_velocity_d_B[i]) > 0.1) {
                    velocity_achieved = false;
                    break;
                }
            }
            double current_yaw = fmod(atan2(R_W_B[2], R_W_B[8]) + M_PI, 2*M_PI) - M_PI;
            double desired_yaw = fmod(yaw_d + M_PI, 2*M_PI) - M_PI;
            if (fabs(current_yaw - desired_yaw) > 0.1) velocity_achieved = false;

            if (velocity_achieved) {
                for (int i = 0; i < 3; i++) linear_velocity_d_B[i] = (double)rand() / RAND_MAX * 1.0 - 0.5;
                linear_velocity_d_B[1] *= 0.2;
                yaw_d = (double)rand() / RAND_MAX * 2 * M_PI;
                target_count++;
                printf("\n=== New Target %d ===\nDesired velocity (body): [%.3f, %.3f, %.3f], yaw: %.3f\n", target_count, linear_velocity_d_B[0], linear_velocity_d_B[1], linear_velocity_d_B[2], yaw_d);
            }
        }

        if (VEC3_MAG2(linear_position_W) > 1000.0*1000.0 || VEC3_MAG2(linear_velocity_W) > 100.0*100.0 || VEC3_MAG2(angular_velocity_B) > 100.0*100.0) {
            printf("\nSimulation diverged.\n"); return 1;
        }

        update_drone_physics(DT_PHYSICS);
        t_physics += DT_PHYSICS;
        
        if (t_control <= t_physics) {
            if (history_len == SEQ_LENGTH) memmove(history[0], history[1], (SEQ_LENGTH-1) * INPUT_FEATURES * sizeof(double));
            else history_len++;
            
            double *current = history[history_len-1];
            memcpy(current, linear_velocity_d_B, 3 * sizeof(double));
            current[3] = yaw_d;
            memcpy(current + 4, angular_velocity_B, 3 * sizeof(double));
            memcpy(current + 7, linear_acceleration_B, 3 * sizeof(double));
            memcpy(current + 10, omega, 4 * sizeof(double));

            if (history_len == SEQ_LENGTH) {
                memcpy(transformer_input, history, SEQ_LENGTH * INPUT_FEATURES * sizeof(double));
                forward_pass(transformer_input, &output, &hidden, &temp, &W_seq, &W_cond, W_q, W_k, W_v, W_o, W_ff1, W_ff2, &W_out, q_buf, k_buf, v_buf, s_buf, mid_buf);
                memcpy(omega_next, &output.data[(SEQ_LENGTH-1) * SEQUENCE_FEATURES + 6], 4 * sizeof(double));
                for (int i = 0; i < 4; i++) omega_next[i] = fmax(OMEGA_MIN, fmin(OMEGA_MAX, omega_next[i]));
            }
            
            update_rotor_speeds();
            t_control += DT_CONTROL;

            printf("\rPos: [%5.2f, %5.2f, %5.2f] Vel (body): [%5.2f, %5.2f, %5.2f] Rotors: [%5.2f, %5.2f, %5.2f, %5.2f]", linear_position_W[0], linear_position_W[1], linear_position_W[2], linear_velocity_B[0], linear_velocity_B[1], linear_velocity_B[2], omega[0], omega[1], omega[2], omega[3]);
            fflush(stdout);
        }

        if (t_render <= t_physics) {
            transform_mesh(meshes[0], linear_position_W, 0.5, R_W_B);
            memset(frame_buffer, 0, WIDTH * HEIGHT * 3);
            vertex_shader(meshes, 2, (double[3]){-2.0, 2.0, -2.0}, (double[3]){0.0, 0.0, 0.0});
            rasterize(frame_buffer, meshes, 2);
            ge_add_frame(gif, frame_buffer, 6);
            t_render += DT_RENDER;
        }
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