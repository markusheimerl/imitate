#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#define D1 64
#define D2 32
#define D3 16
#define M 18
#define B1 0.9
#define B2 0.999
#define E 1e-8
#define DC 0.01

double g_prev_loss = 1e30, g_lr = 1e-4;

bool load_weights(const char* filename, double *W1, double *b1, double *W2, double *b2, double *W3, double *b3, double *W4, double *b4) {
    FILE *f = fopen(filename, "rb");
    if (!f) return false;
    size_t items_read = fread(W1, sizeof(double), D1*M, f) + fread(b1, sizeof(double), D1, f) + 
                        fread(W2, sizeof(double), D2*D1, f) + fread(b2, sizeof(double), D2, f) + 
                        fread(W3, sizeof(double), D3*D2, f) + fread(b3, sizeof(double), D3, f) + 
                        fread(W4, sizeof(double), D3, f) + fread(b4, sizeof(double), 1, f);
    fclose(f);
    return items_read == (D1*M + D1 + D2*D1 + D2 + D3*D2 + D3 + D3 + 1);
}

void save_weights(const char* filename, double *W1, double *b1, double *W2, double *b2, double *W3, double *b3, double *W4, double *b4) {
    FILE *f = fopen(filename, "wb");
    fwrite(W1, sizeof(double), D1*M, f); fwrite(b1, sizeof(double), D1, f);
    fwrite(W2, sizeof(double), D2*D1, f); fwrite(b2, sizeof(double), D2, f);
    fwrite(W3, sizeof(double), D3*D2, f); fwrite(b3, sizeof(double), D3, f);
    fwrite(W4, sizeof(double), D3, f); fwrite(b4, sizeof(double), 1, f);
    fclose(f);
}

void forward(double *W1, double *b1, double *W2, double *b2, double *W3, double *b3, double *W4, double *b4,
            double *input, double *h1, double *h2, double *h3, double *out) {
    for(int i = 0; i < D1; i++) {
        double sum = b1[i];
        for(int j = 0; j < M; j++) sum += W1[i*M + j] * input[j];
        h1[i] = sum > 0 ? sum : sum * 0.1;
    }
    for(int i = 0; i < D2; i++) {
        double sum = b2[i];
        for(int j = 0; j < D1; j++) sum += W2[i*D1 + j] * h1[j];
        h2[i] = sum > 0 ? sum : sum * 0.1;
    }
    for(int i = 0; i < D3; i++) {
        double sum = b3[i];
        for(int j = 0; j < D2; j++) sum += W3[i*D2 + j] * h2[j];
        h3[i] = sum > 0 ? sum : sum * 0.1;
    }
    *out = b4[0];
    for(int i = 0; i < D3; i++) *out += W4[i] * h3[i];
}

void adam(double *p, double *g, double *m, double *v, int size, int t) {
    double lr_t = g_lr * sqrt(1.0 - pow(B2, t)) / (1.0 - pow(B1, t));
    for(int i = 0; i < size; i++) {
        m[i] = B1 * m[i] + (1-B1) * g[i];
        v[i] = B2 * v[i] + (1-B2) * g[i] * g[i];
        p[i] -= lr_t * (m[i] / (sqrt(v[i]) + E) + DC * p[i]);
    }
}

double backward(double *W1, double *b1, double *W2, double *b2, double *W3, double *b3, double *W4, double *b4,
               double *m_W1, double *m_b1, double *m_W2, double *m_b2, double *m_W3, double *m_b3, double *m_W4, double *m_b4,
               double *v_W1, double *v_b1, double *v_W2, double *v_b2, double *v_W3, double *v_b3, double *v_W4, double *v_b4,
               double *input, double *h1, double *h2, double *h3, double *out, double target, int step) {
    double d_out = 2 * (*out - target);
    double *d_W1 = calloc(D1*M, sizeof(double)), *d_b1 = calloc(D1, sizeof(double));
    double *d_W2 = calloc(D2*D1, sizeof(double)), *d_b2 = calloc(D2, sizeof(double));
    double *d_W3 = calloc(D3*D2, sizeof(double)), *d_b3 = calloc(D3, sizeof(double));
    double *d_W4 = calloc(D3, sizeof(double)), d_b4 = d_out;
    
    for(int i = 0; i < D3; i++) {
        d_W4[i] = d_out * h3[i];
        double d_h3 = d_out * W4[i] * (h3[i] > 0 ? 1.0 : 0.1);
        for(int j = 0; j < D2; j++) d_W3[i*D2 + j] = d_h3 * h2[j];
        d_b3[i] = d_h3;
    }
    for(int i = 0; i < D2; i++) {
        double d_h2 = 0;
        for(int j = 0; j < D3; j++) d_h2 += d_b3[j] * W3[j*D2 + i];
        d_h2 *= (h2[i] > 0 ? 1.0 : 0.1);
        for(int j = 0; j < D1; j++) d_W2[i*D1 + j] = d_h2 * h1[j];
        d_b2[i] = d_h2;
    }
    for(int i = 0; i < D1; i++) {
        double d_h1 = 0;
        for(int j = 0; j < D2; j++) d_h1 += d_b2[j] * W2[j*D1 + i];
        d_h1 *= (h1[i] > 0 ? 1.0 : 0.1);
        for(int j = 0; j < M; j++) d_W1[i*M + j] = d_h1 * input[j];
        d_b1[i] = d_h1;
    }

    adam(W1, d_W1, m_W1, v_W1, D1*M, step); adam(b1, d_b1, m_b1, v_b1, D1, step);
    adam(W2, d_W2, m_W2, v_W2, D2*D1, step); adam(b2, d_b2, m_b2, v_b2, D2, step);
    adam(W3, d_W3, m_W3, v_W3, D3*D2, step); adam(b3, d_b3, m_b3, v_b3, D3, step);
    adam(W4, d_W4, m_W4, v_W4, D3, step); adam(b4, &d_b4, m_b4, v_b4, 1, step);

    free(d_W1), free(d_b1), free(d_W2), free(d_b2), free(d_W3), free(d_b3), free(d_W4);
    return (*out - target) * (*out - target);
}

int main(int argc, char **argv) {
    if(argc < 2 || argc > 3) { printf("Usage: %s <data_file> [weights_file]\n", argv[0]); return 1; }
    
    double *W1 = malloc(D1*M*sizeof(double)), *b1 = calloc(D1, sizeof(double));
    double *W2 = malloc(D2*D1*sizeof(double)), *b2 = calloc(D2, sizeof(double));
    double *W3 = malloc(D3*D2*sizeof(double)), *b3 = calloc(D3, sizeof(double));
    double *W4 = malloc(D3*sizeof(double)), *b4 = calloc(1, sizeof(double));
    double *h1 = malloc(D1*sizeof(double)), *h2 = malloc(D2*sizeof(double)), *h3 = malloc(D3*sizeof(double));

    if (argc == 3) {
        if (!load_weights(argv[2], W1, b1, W2, b2, W3, b3, W4, b4)) { printf("Failed to load weights\n"); return 1; }
    } else {
        srand(time(NULL));
        for(int i = 0; i < D1*M; i++) W1[i] = ((double)rand()/RAND_MAX - 0.5) * sqrt(2.0/M);
        for(int i = 0; i < D2*D1; i++) W2[i] = ((double)rand()/RAND_MAX - 0.5) * sqrt(2.0/D1);
        for(int i = 0; i < D3*D2; i++) W3[i] = ((double)rand()/RAND_MAX - 0.5) * sqrt(2.0/D2);
        for(int i = 0; i < D3; i++) W4[i] = ((double)rand()/RAND_MAX - 0.5) * sqrt(2.0/D3);
    }

    double *m_W1 = calloc(D1*M, sizeof(double)), *m_b1 = calloc(D1, sizeof(double));
    double *m_W2 = calloc(D2*D1, sizeof(double)), *m_b2 = calloc(D2, sizeof(double));
    double *m_W3 = calloc(D3*D2, sizeof(double)), *m_b3 = calloc(D3, sizeof(double));
    double *m_W4 = calloc(D3, sizeof(double)), *m_b4 = calloc(1, sizeof(double));
    double *v_W1 = calloc(D1*M, sizeof(double)), *v_b1 = calloc(D1, sizeof(double));
    double *v_W2 = calloc(D2*D1, sizeof(double)), *v_b2 = calloc(D2, sizeof(double));
    double *v_W3 = calloc(D3*D2, sizeof(double)), *v_b3 = calloc(D3, sizeof(double));
    double *v_W4 = calloc(D3, sizeof(double)), *v_b4 = calloc(1, sizeof(double));

    FILE *f = fopen(argv[1], "r");
    if (!f) { printf("Failed to open file\n"); return 1; }
    char line[1024];
    int rows = -1;
    while(fgets(line, sizeof(line), f)) rows++;
    rewind(f); fgets(line, sizeof(line), f);

    printf("Loading %d rows of data...\n", rows);

    double **data = malloc(rows * sizeof(double*)), *targets = malloc(rows * sizeof(double));
    int *indices = malloc(rows * sizeof(int));
    for(int i = 0; i < rows; i++) {
        indices[i] = i;
        data[i] = malloc(M * sizeof(double));
        fgets(line, sizeof(line), f);
        char *token = strtok(line, ",");
        // Read first 18 values (pos, vel, ang_vel, R)
        for(int j = 0; j < M; j++) {
            data[i][j] = atof(token);
            token = strtok(NULL, ",");
        }
        // Skip 10 columns (acc_s[3], gyro_s[3], omega[4])
        for(int j = 0; j < 10; j++) token = strtok(NULL, ",");
        // Read discounted_return (last column)
        token = strtok(NULL, ",");
        if (token) targets[i] = atof(token);
    }
    fclose(f);

    printf("First target value: %f\n", targets[0]);

    double out, running_loss = 0;
    int step = 1;

    for(int epoch = 0; epoch < 10000; epoch++) {
        for(int i = rows-1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = indices[i]; indices[i] = indices[j]; indices[j] = temp;
        }
        for(int i = 0; i < rows; i++, step++) {
            forward(W1, b1, W2, b2, W3, b3, W4, b4, data[indices[i]], h1, h2, h3, &out);
            running_loss += backward(W1, b1, W2, b2, W3, b3, W4, b4, m_W1, m_b1, m_W2, m_b2, m_W3, m_b3, m_W4, m_b4,
                                   v_W1, v_b1, v_W2, v_b2, v_W3, v_b3, v_W4, v_b4, data[indices[i]], h1, h2, h3, &out, targets[indices[i]], step);
            if(step % 100000 == 0) {
                double avg_loss = running_loss/100000;
                g_lr *= (avg_loss > g_prev_loss) ? 0.95 : 1.05;
                g_lr = fmax(1e-6, fmin(1e-3, g_lr));
                printf("Epoch %d, Step %d, Loss: %f, LR: %e\n", epoch, step, avg_loss, g_lr);
                if(step % 1000000 == 0) printf("Sample pred: %8.3f true: %8.3f\n\n", out, targets[indices[i]]);
                g_prev_loss = avg_loss;
                running_loss = 0;
            }
        }
    }

    char filename[100];
    time_t t = time(NULL); struct tm tm = *localtime(&t);
    sprintf(filename, "%d-%d-%d_%d-%d-%d_value_weights.bin", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    save_weights(filename, W1, b1, W2, b2, W3, b3, W4, b4);

    for(int i = 0; i < rows; i++) free(data[i]);
    free(data), free(targets), free(indices);
    free(W1), free(b1), free(W2), free(b2), free(W3), free(b3), free(W4), free(b4);
    free(h1), free(h2), free(h3);
    free(m_W1), free(m_b1), free(m_W2), free(m_b2), free(m_W3), free(m_b3), free(m_W4), free(m_b4);
    free(v_W1), free(v_b1), free(v_W2), free(v_b2), free(v_W3), free(v_b3), free(v_W4), free(v_b4);
    return 0;
}