#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define N_EPOCHS 1
#define S 32    // Sequence length
#define D 256   // Hidden dimension
#define M 6     // Input dimension (3 ang_vel + 3 acc)
#define B1 0.9
#define B2 0.999
#define E 1e-8
#define DC 0.01
#define C 1.0
#define TEMPORAL_DECAY 0.125

double g_prev_loss = 1e30, g_lr = 0.0001;

void save_weights(const char* filename, double *W_in, double *b_in, double *W_q, double *W_k, double *W_v, double *W_out, double *b_out) {
    FILE* f = fopen(filename, "wb");
    if (!f) { printf("Error: Could not open file %s for writing\n", filename); return; }
    fwrite(W_in, sizeof(double), D * M, f);
    fwrite(b_in, sizeof(double), D, f);
    fwrite(W_q, sizeof(double), D * D, f);
    fwrite(W_k, sizeof(double), D * D, f);
    fwrite(W_v, sizeof(double), D * D, f);
    fwrite(W_out, sizeof(double), D, f);
    fwrite(b_out, sizeof(double), 1, f);
    fclose(f);
}

void forward(double *W_in, double *b_in, double *W_q, double *W_k, double *W_v, double *W_out, double *b_out, 
            double *hidden, double *q, double *k, double *v, double *attn_scores, double *attn_probs, 
            double *context, double (*seq)[M], double *out) {
    for(int s = 0; s < S; s++) {
        for(int d = 0; d < D; d++) {
            double sum = b_in[d];
            for(int m = 0; m < M; m++) sum += W_in[d * M + m] * seq[s][m];
            hidden[s * D + d] = fmax(0.0, sum);
        }
    }

    for(int s = 0; s < S; s++) {
        for(int d = 0; d < D; d++) {
            q[s * D + d] = k[s * D + d] = v[s * D + d] = 0;
            for(int j = 0; j < D; j++) {
                q[s * D + d] += W_q[d * D + j] * hidden[s * D + j];
                k[s * D + d] += W_k[d * D + j] * hidden[s * D + j];
                v[s * D + d] += W_v[d * D + j] * hidden[s * D + j];
            }
        }
    }

    for(int i = 0; i < S; i++) {
        double max_val = -1e30;
        for(int j = 0; j < S; j++) {
            double score = 0;
            for(int d = 0; d < D; d++) score += q[i * D + d] * k[j * D + d];
            max_val = fmax(max_val, (attn_scores[i * S + j] = score/sqrt(D) - TEMPORAL_DECAY * (i-j)));
        }
        double sum = 0;
        for(int j = 0; j < S; j++) sum += (attn_probs[i * S + j] = exp(attn_scores[i * S + j] - max_val));
        for(int j = 0; j < S; j++) attn_probs[i * S + j] /= sum;
    }

    for(int i = 0; i < S; i++) {
        double rms = 0;
        for(int d = 0; d < D; d++) {
            double sum = 0;
            for(int j = 0; j < S; j++) sum += attn_probs[i * S + j] * v[j * D + d];
            context[i * D + d] = sum + hidden[i * D + d];
            rms += context[i * D + d] * context[i * D + d];
        }
        rms = sqrt(rms / D + E);
        for(int d = 0; d < D; d++) context[i * D + d] /= rms;
    }

    *out = b_out[0];
    for(int d = 0; d < D; d++) *out += W_out[d] * context[(S-1) * D + d];
}

double backward(double *W_in, double *W_q, double *W_k, double *W_v, double *W_out,
               double *hidden, double *q, double *k, double *v, double *attn_scores, double *attn_probs, double *context,
               double *d_W_in, double *d_b_in, double *d_W_q, double *d_W_k, double *d_W_v, double *d_W_out, double *d_b_out,
               double *d_context, double *d_v, double *d_q, double *d_k, double *d_attn_probs, double *d_hidden, 
               double *d_attn_scores, double (*seq)[M], double *out, double target) {
    double loss = 0;
    memset(d_context, 0, S * D * sizeof(double));
    memset(d_v, 0, S * D * sizeof(double));
    memset(d_q, 0, S * D * sizeof(double));
    memset(d_k, 0, S * D * sizeof(double));
    memset(d_attn_probs, 0, S * S * sizeof(double));
    memset(d_hidden, 0, S * D * sizeof(double));
    memset(d_attn_scores, 0, S * S * sizeof(double));

    double d_out = 2 * (*out - target);
    loss = (*out - target) * (*out - target);
    d_b_out[0] += d_out;
    for(int d = 0; d < D; d++) {
        d_W_out[d] += d_out * context[(S-1) * D + d];
        d_context[(S-1) * D + d] += d_out * W_out[d];
    }

    for(int i = 0; i < S; i++) {
        double rms = 0;
        for(int d = 0; d < D; d++) rms += context[i * D + d] * context[i * D + d];
        rms = sqrt(rms / D + E);
        double sum_grad = 0;
        for(int d = 0; d < D; d++) sum_grad += d_context[i * D + d] * context[i * D + d];
        for(int d = 0; d < D; d++) {
            double grad = d_context[i * D + d] / rms - context[i * D + d] * sum_grad / (D * rms * rms * rms);
            d_hidden[i * D + d] += grad;
            for(int j = 0; j < S; j++) d_v[j * D + d] += grad * attn_probs[i * S + j];
        }
    }

    for(int i = 0; i < S; i++) {
        for(int j = 0; j < S; j++) {
            double sum = 0;
            for(int k = 0; k < S; k++) sum += attn_probs[i * S + k] * d_attn_probs[i * S + k];
            d_attn_scores[i * S + j] = attn_probs[i * S + j] * (d_attn_probs[i * S + j] - sum);
        }
    }

    for(int i = 0; i < S; i++) {
        for(int d = 0; d < D; d++) {
            for(int j = 0; j < D; j++) {
                d_W_q[d * D + j] += d_q[i * D + d] * hidden[i * D + j];
                d_W_k[d * D + j] += d_k[i * D + d] * hidden[i * D + j];
                d_W_v[d * D + j] += d_v[i * D + d] * hidden[i * D + j];
                d_hidden[i * D + j] += d_q[i * D + d] * W_q[d * D + j] + d_k[i * D + d] * W_k[d * D + j] + d_v[i * D + d] * W_v[d * D + j];
            }
        }
    }

    for(int s = 0; s < S; s++) {
        for(int d = 0; d < D; d++) {
            if(hidden[s * D + d] > 0) {
                d_b_in[d] += d_hidden[s * D + d];
                for(int m = 0; m < M; m++) d_W_in[d * M + m] += d_hidden[s * D + d] * seq[s][m];
            }
        }
    }

    return loss;
}

void adam(double *p, double *g, double *m, double *v, int size, int t) {
    double lr_t = g_lr * sqrt(1.0 - pow(B2, t)) / (1.0 - pow(B1, t)), norm = 0;
    for(int i = 0; i < size; i++) norm += g[i] * g[i];
    if(sqrt(norm) > C) for(int i = 0; i < size; i++) g[i] *= C / sqrt(norm);
    for(int i = 0; i < size; i++) {
        m[i] = B1 * m[i] + (1-B1) * g[i];
        v[i] = B2 * v[i] + (1-B2) * g[i] * g[i];
        p[i] -= lr_t * (m[i]/(sqrt(v[i]) + E) + DC * p[i]);
    }
}

int main(int argc, char **argv) {
    if(argc != 2) { printf("Usage: %s <data_file>\n", argv[0]); return 1; }
    srand(time(NULL));
    
    double *W_in = malloc(D * M * sizeof(double)), *b_in = malloc(D * sizeof(double));
    double *W_q = malloc(D * D * sizeof(double)), *W_k = malloc(D * D * sizeof(double));
    double *W_v = malloc(D * D * sizeof(double)), *W_out = malloc(D * sizeof(double));
    double *b_out = malloc(sizeof(double)), *hidden = malloc(S * D * sizeof(double));
    double *q = malloc(S * D * sizeof(double)), *k = malloc(S * D * sizeof(double));
    double *v = malloc(S * D * sizeof(double)), *attn_scores = malloc(S * S * sizeof(double));
    double *attn_probs = malloc(S * S * sizeof(double)), *context = malloc(S * D * sizeof(double));
    double *d_W_in = calloc(D * M, sizeof(double)), *d_b_in = calloc(D, sizeof(double));
    double *d_W_q = calloc(D * D, sizeof(double)), *d_W_k = calloc(D * D, sizeof(double));
    double *d_W_v = calloc(D * D, sizeof(double)), *d_W_out = calloc(D, sizeof(double));
    double *d_b_out = calloc(1, sizeof(double)), *d_context = malloc(S * D * sizeof(double));
    double *d_v = malloc(S * D * sizeof(double)), *d_q = malloc(S * D * sizeof(double));
    double *d_k = malloc(S * D * sizeof(double)), *d_attn_probs = malloc(S * S * sizeof(double));
    double *d_hidden = malloc(S * D * sizeof(double)), *d_attn_scores = malloc(S * S * sizeof(double));
    double *m_W_in = calloc(D * M, sizeof(double)), *m_b_in = calloc(D, sizeof(double));
    double *m_W_q = calloc(D * D, sizeof(double)), *m_W_k = calloc(D * D, sizeof(double));
    double *m_W_v = calloc(D * D, sizeof(double)), *m_W_out = calloc(D, sizeof(double));
    double *m_b_out = calloc(1, sizeof(double)), *v_W_in = calloc(D * M, sizeof(double));
    double *v_b_in = calloc(D, sizeof(double)), *v_W_q = calloc(D * D, sizeof(double));
    double *v_W_k = calloc(D * D, sizeof(double)), *v_W_v = calloc(D * D, sizeof(double));
    double *v_W_out = calloc(D, sizeof(double)), *v_b_out = calloc(1, sizeof(double));

    double scale = sqrt(2.0/M);
    for(int i = 0; i < D * M; i++) W_in[i] = ((double)rand()/RAND_MAX - 0.5) * scale;
    memset(b_in, 0, D * sizeof(double));
    for(int i = 0; i < D * D; i++) W_q[i] = W_k[i] = W_v[i] = ((double)rand()/RAND_MAX - 0.5) * scale;
    for(int i = 0; i < D; i++) W_out[i] = ((double)rand()/RAND_MAX - 0.5) * scale;
    *b_out = 0;

    FILE *f = fopen(argv[1], "r");
    if(!f) { printf("Error: Could not open file %s\n", argv[1]); return 1; }
    char line[1024];
    int rows = -1;
    while(fgets(line, sizeof(line), f)) rows++;
    rewind(f);
    fgets(line, sizeof(line), f);

    double **sensor_data = malloc(rows * sizeof(double*));
    double *reward_data = malloc(rows * sizeof(double));
    for(int i = 0; i < rows; i++) {
        sensor_data[i] = malloc(M * sizeof(double));
        if(!fgets(line, sizeof(line), f)) break;
        char *token = strtok(line, ",");
        for(int j = 0; j < 4; j++) token = strtok(NULL, ",");
        for(int j = 0; j < M; j++) {
            sensor_data[i][j] = atof(token);
            token = strtok(NULL, ",");
        }
        for(int j = 0; j < 5; j++) token = strtok(NULL, ",");
        reward_data[i] = atof(token);
    }
    fclose(f);

    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    char loss_filename[100];
    sprintf(loss_filename, "%d-%d-%d_%d-%d-%d_loss.csv", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    FILE *loss_file = fopen(loss_filename, "w");
    fprintf(loss_file, "step,loss\n");

    int max_start = rows - S, step = 1;
    int *positions = malloc(max_start * sizeof(int));
    for(int i = 0; i < max_start; i++) positions[i] = i;
    double running_loss = 0, out;

    for(int epoch = 0; epoch < N_EPOCHS; epoch++) {
        for(int i = max_start - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = positions[i];
            positions[i] = positions[j];
            positions[j] = temp;
        }

        for(int i = 0; i < max_start; i++) {
            double seq[S][M];
            for(int j = 0; j < S; j++) 
                memcpy(seq[j], sensor_data[positions[i] + j], M * sizeof(double));
            
            forward(W_in, b_in, W_q, W_k, W_v, W_out, b_out, hidden, q, k, v, attn_scores, attn_probs, context, seq, &out);
            running_loss += backward(W_in, W_q, W_k, W_v, W_out, hidden, q, k, v, attn_scores, attn_probs, context,
                                   d_W_in, d_b_in, d_W_q, d_W_k, d_W_v, d_W_out, d_b_out,
                                   d_context, d_v, d_q, d_k, d_attn_probs, d_hidden, d_attn_scores,
                                   seq, &out, reward_data[positions[i] + S]);

            adam(W_in, d_W_in, m_W_in, v_W_in, D * M, step);
            adam(b_in, d_b_in, m_b_in, v_b_in, D, step);
            adam(W_q, d_W_q, m_W_q, v_W_q, D * D, step);
            adam(W_k, d_W_k, m_W_k, v_W_k, D * D, step);
            adam(W_v, d_W_v, m_W_v, v_W_v, D * D, step);
            adam(W_out, d_W_out, m_W_out, v_W_out, D, step);
            adam(b_out, d_b_out, m_b_out, v_b_out, 1, step);

            g_lr *= ((running_loss/i) > g_prev_loss) ? 0.95 : 1.05;
            g_lr = fmax(1e-10, fmin(1e-3, g_lr));
            g_prev_loss = (running_loss/i);
            fprintf(loss_file, "%d,%f\n", step, running_loss/100);

            if(step % 100 == 0) {
                printf("Step %d (Epoch %d), Average Loss: %f, LR: %e\n", step, epoch, running_loss/100, g_lr);
                running_loss = 0;
            }
            if(step % 1000 == 0) {
                printf("\nDiagnostic at step %d:\n", step);
                printf("Input sequence (last 5 steps shown):\n");
                for(int j = S-5; j < S; j++) {
                    printf("  Step %d: ang_vel[%.3f, %.3f, %.3f] acc[%.3f, %.3f, %.3f]\n", 
                           j, seq[j][0], seq[j][1], seq[j][2], seq[j][3], seq[j][4], seq[j][5]);
                }
                printf("Predicted cumulative reward: %.3f\n", out);
                printf("Actual cumulative reward: %.3f\n\n", reward_data[positions[i] + S]);
            }
            step++;
        }
    }

    fclose(loss_file);

    char weights_filename[100];
    sprintf(weights_filename, "%d-%d-%d_%d-%d-%d_weights.bin", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    save_weights(weights_filename, W_in, b_in, W_q, W_k, W_v, W_out, b_out);

    // Cleanup
    free(W_in); free(b_in); free(W_q); free(W_k); free(W_v); free(W_out); free(b_out);
    free(hidden); free(q); free(k); free(v); free(attn_scores); free(attn_probs); free(context);
    free(d_W_in); free(d_b_in); free(d_W_q); free(d_W_k); free(d_W_v); free(d_W_out); free(d_b_out);
    free(d_context); free(d_v); free(d_q); free(d_k); free(d_attn_probs); free(d_hidden); free(d_attn_scores);
    free(m_W_in); free(m_b_in); free(m_W_q); free(m_W_k); free(m_W_v); free(m_W_out); free(m_b_out);
    free(v_W_in); free(v_b_in); free(v_W_q); free(v_W_k); free(v_W_v); free(v_W_out); free(v_b_out);
    for(int i = 0; i < rows; i++) free(sensor_data[i]);
    free(sensor_data); free(reward_data); free(positions);
    return 0;
}