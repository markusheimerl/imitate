#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define D 2048   // Hidden dimension
#define M 6     // Input dimension (3 ang_vel + 3 acc)
#define B1 0.9  // Adam beta1
#define B2 0.999// Adam beta2
#define E 1e-8  // Adam epsilon
#define DC 0.01 // Weight decay

double g_prev_loss = 1e30, g_lr = 1e-4;

void forward(double *W_in, double *b_in, double *W_out, double *b_out, double *hidden, double *input, double *out) {
    for(int d = 0; d < D; d++) {
        double sum = b_in[d];
        for(int m = 0; m < M; m++) sum += W_in[d * M + m] * input[m];
        hidden[d] = fmax(0.0, sum);
    }
    *out = *b_out;
    for(int d = 0; d < D; d++) *out += W_out[d] * hidden[d];
}

void adam(double *p, double *g, double *m, double *v, int size, int t) {
    double lr_t = g_lr * sqrt(1.0 - pow(B2, t)) / (1.0 - pow(B1, t));
    for(int i = 0; i < size; i++) {
        m[i] = B1 * m[i] + (1-B1) * g[i];
        v[i] = B2 * v[i] + (1-B2) * g[i] * g[i];
        p[i] -= lr_t * (m[i] / (sqrt(v[i]) + E) + DC * p[i]);
    }
}

double backward(double *W_in, double *b_in, double *W_out, double *b_out, double *hidden, double *input, double *out, double target, double *m_W_in, double *m_b_in, double *m_W_out, double *m_b_out, double *v_W_in, double *v_b_in, double *v_W_out, double *v_b_out, int step) {
    double d_out = 2 * (*out - target);
    double *d_W_in = calloc(D * M, sizeof(double)), *d_b_in = calloc(D, sizeof(double)), *d_W_out = calloc(D, sizeof(double)), d_b_out = d_out;

    for(int d = 0; d < D; d++) {
        d_W_out[d] = d_out * hidden[d];
        if(hidden[d] > 0) {
            for(int m = 0; m < M; m++) d_W_in[d * M + m] = d_out * W_out[d] * input[m];
            d_b_in[d] = d_out * W_out[d];
        }
    }

    adam(W_in, d_W_in, m_W_in, v_W_in, D * M, step);
    adam(b_in, d_b_in, m_b_in, v_b_in, D, step);
    adam(W_out, d_W_out, m_W_out, v_W_out, D, step);
    adam(b_out, &d_b_out, m_b_out, v_b_out, 1, step);

    double loss = (*out - target) * (*out - target);
    free(d_W_in); free(d_b_in); free(d_W_out);
    return loss;
}

int main(int argc, char **argv) {
    if(argc != 2) { printf("Usage: %s <data_file>\n", argv[0]); return 1; }
    
    double *W_in = malloc(D * M * sizeof(double)), *b_in = calloc(D, sizeof(double)), *W_out = malloc(D * sizeof(double)), *b_out = calloc(1, sizeof(double)), *hidden = malloc(D * sizeof(double));
    double *m_W_in = calloc(D * M, sizeof(double)), *m_b_in = calloc(D, sizeof(double)), *m_W_out = calloc(D, sizeof(double)), *m_b_out = calloc(1, sizeof(double));
    double *v_W_in = calloc(D * M, sizeof(double)), *v_b_in = calloc(D, sizeof(double)), *v_W_out = calloc(D, sizeof(double)), *v_b_out = calloc(1, sizeof(double));
    
    double scale = sqrt(2.0/M);
    for(int i = 0; i < D * M; i++) W_in[i] = ((double)rand()/RAND_MAX - 0.5) * scale;
    for(int i = 0; i < D; i++) W_out[i] = ((double)rand()/RAND_MAX - 0.5) * scale;

    FILE *f = fopen(argv[1], "r");
    char line[1024];
    int rows = -1;
    while(fgets(line, sizeof(line), f)) rows++;
    rewind(f); fgets(line, sizeof(line), f);

    double **data = malloc(rows * sizeof(double*)), *targets = malloc(rows * sizeof(double));
    int *indices = malloc(rows * sizeof(int));
    for(int i = 0; i < rows; i++) {
        indices[i] = i;
        data[i] = malloc(M * sizeof(double));
        fgets(line, sizeof(line), f);
        char *token = strtok(line, ",");
        for(int j = 0; j < 4; j++) token = strtok(NULL, ",");
        for(int j = 0; j < M; j++) { data[i][j] = atof(token); token = strtok(NULL, ","); }
        for(int j = 0; j < 5; j++) token = strtok(NULL, ",");
        targets[i] = atof(token);
    }
    fclose(f);

    double out, running_loss = 0;
    int step = 1;

    for(int epoch = 0; epoch < 10; epoch++) {
        for(int i = rows-1; i > 0; i--) { int j = rand() % (i + 1); int temp = indices[i]; indices[i] = indices[j]; indices[j] = temp; }
        
        for(int i = 0; i < rows; i++, step++) {
            forward(W_in, b_in, W_out, b_out, hidden, data[indices[i]], &out);
            running_loss += backward(W_in, b_in, W_out, b_out, hidden, data[indices[i]], &out, targets[indices[i]], m_W_in, m_b_in, m_W_out, m_b_out, v_W_in, v_b_in, v_W_out, v_b_out, step);

            if(step % 10000 == 0) {
                double avg_loss = running_loss/10000;
                g_lr *= (avg_loss > g_prev_loss) ? 0.95 : 1.05;
                g_lr = fmax(1e-6, fmin(1e-3, g_lr));
                printf("Epoch %d, Step %d, Loss: %f, LR: %e\n", epoch, step, avg_loss, g_lr);
                if(step % 100000 == 0) printf("Sample: input[%6.3f, %6.3f, %6.3f, %6.3f, %6.3f, %6.3f] pred: %8.3f true: %8.3f\n\n", data[indices[i]][0], data[indices[i]][1], data[indices[i]][2], data[indices[i]][3], data[indices[i]][4], data[indices[i]][5], out, targets[indices[i]]);
                g_prev_loss = avg_loss;
                running_loss = 0;
            }
        }
    }

    char filename[100];
    time_t t = time(NULL); struct tm tm = *localtime(&t);
    sprintf(filename, "%d-%d-%d_%d-%d-%d_weights.bin", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    FILE *wf = fopen(filename, "wb");
    fwrite(W_in, sizeof(double), D * M, wf); fwrite(b_in, sizeof(double), D, wf); fwrite(W_out, sizeof(double), D, wf); fwrite(b_out, sizeof(double), 1, wf);
    fclose(wf);

    free(W_in); free(b_in); free(W_out); free(b_out); free(hidden); free(m_W_in); free(m_b_in); free(m_W_out); free(m_b_out);
    free(v_W_in); free(v_b_in); free(v_W_out); free(v_b_out); free(indices);
    for(int i = 0; i < rows; i++) free(data[i]);
    free(data); free(targets);
    return 0;
}