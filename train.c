#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define S 32
#define D 256
#define M 4
#define B1 0.9f
#define B2 0.999f
#define E 1e-8f
#define DC 0.01f
#define C 1.0f

double g_prev_loss = INFINITY, g_lr = 0.0001;

void forward(double *W_up, double *b_up, double *W_down, double *b_down, double *hidden, double (*seq)[M], double *out) {
    memset(hidden, 0, D * sizeof(double));
    #pragma omp parallel for
    for(int s = 0; s < S; s++) for(int i = 0; i < D; i++) {
        double sum = b_up[i];
        for(int j = 0; j < M; j++) sum += W_up[i * M + j] * seq[s][j];
        hidden[i] += fmax(0.0, sum);
    }
    for(int i = 0; i < M; i++) {
        double sum = b_down[i];
        for(int j = 0; j < D; j++) sum += W_down[i * D + j] * hidden[j];
        out[i] = sum;
    }
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

double train(double *W_up, double *b_up, double *W_down, double *b_down, double *hidden, double *d_W_up, double *d_b_up, double *d_W_down, double *d_b_down, double *m_W_up, double *m_b_up, double *m_W_down, double *m_b_down, double *v_W_up, double *v_b_up, double *v_W_down, double *v_b_down, double (*seq)[M], double *target, int step) {
    double out[M], loss = 0;
    forward(W_up, b_up, W_down, b_down, hidden, seq, out);
    memset(d_W_up, 0, D * M * sizeof(double));
    memset(d_b_up, 0, D * sizeof(double));
    memset(d_W_down, 0, M * D * sizeof(double));
    memset(d_b_down, 0, M * sizeof(double));
    
    for(int i = 0; i < M; i++) {
        double d_out = 2 * (out[i] - target[i]);
        loss += pow(out[i] - target[i], 2);
        d_b_down[i] = d_out;
        for(int j = 0; j < D; j++) {
            d_W_down[i * D + j] = d_out * hidden[j];
            double d_hidden = d_out * W_down[i * D + j];
            for(int s = 0; s < S; s++) {
                double sum = b_up[j];
                for(int k = 0; k < M; k++) sum += W_up[j * M + k] * seq[s][k];
                if(sum > 0) {
                    d_b_up[j] += d_hidden;
                    for(int k = 0; k < M; k++) d_W_up[j * M + k] += d_hidden * seq[s][k];
                }
            }
        }
    }
    adam(W_up, d_W_up, m_W_up, v_W_up, D * M, step);
    adam(b_up, d_b_up, m_b_up, v_b_up, D, step);
    adam(W_down, d_W_down, m_W_down, v_W_down, M * D, step);
    adam(b_down, d_b_down, m_b_down, v_b_down, M, step);
    g_lr *= (loss > g_prev_loss) ? 0.95 : 1.05;
    g_lr = fmax(1e-9, fmin(1e-3, g_lr));
    g_prev_loss = loss;
    return loss;
}

int main(int argc, char **argv) {
    srand(time(NULL));
    double scale = sqrt(2.0/M), *ptrs[17], seq[S][M], running_loss = 0;
    int sizes[] = {D * M, D, M * D, M}, step = 1;
    
    // Initialize arrays
    for(int i = 0; i < 17; i++) {
        ptrs[i] = i < 5 ? malloc(sizes[i % 4] * sizeof(double)) : calloc(sizes[i % 4], sizeof(double));
        if(i < 2) for(int j = 0; j < sizes[i]; j++) ptrs[i][j] = ((double)rand()/RAND_MAX - 0.5) * scale;
    }
    
    // Read data
    FILE *f = fopen(argv[1], "r");
    if(!f) { printf("Error: Could not open file %s\n", argv[1]); return 1; }
    
    char line[1024];
    int rows = -1;
    while(fgets(line, sizeof(line), f)) rows++;
    rewind(f);
    fgets(line, sizeof(line), f);
    
    double **data = malloc(rows * sizeof(double*));
    for(int i = 0; i < rows; i++) {
        data[i] = malloc(M * sizeof(double));
        if(!fgets(line, sizeof(line), f)) break;
        char *token = strtok(line, ",");
        for(int j = 0; j < 10; j++) token = strtok(NULL, ",");
        for(int j = 0; j < M; j++) if((token = strtok(NULL, ","))) data[i][j] = atof(token);
    }
    fclose(f);
    
    // Training loop
    int max_start = rows - S;
    int *positions = malloc(max_start * sizeof(int));
    for(int i = 0; i < max_start; i++) positions[i] = i;
    
    for(int epoch = 0; epoch < 1000; epoch++) {
        for(int i = max_start - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = positions[i];
            positions[i] = positions[j];
            positions[j] = temp;
        }
        
        for(int i = 0; i < max_start; i++) {
            for(int j = 0; j < S; j++) memcpy(seq[j], data[positions[i] + j], M * sizeof(double));
            running_loss += train(ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7], ptrs[8], ptrs[9], ptrs[10], ptrs[11], ptrs[12], ptrs[13], ptrs[14], ptrs[15], ptrs[16], seq, data[positions[i] + S], step);
            if(step++ % 10000 == 0) {
                printf("Step %d (Epoch %d), Average Loss: %f, LR: %e\n", step-1, epoch, running_loss/1000, g_lr);
                running_loss = 0;
            }
        }
    }
    
    // Cleanup
    for(int i = 0; i < rows; i++) free(data[i]);
    for(int i = 0; i < 17; i++) free(ptrs[i]);
    free(positions);
    free(data);
    return 0;
}