#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#define D1 2048
#define D2 1024
#define D3 512
#define M_IN 6
#define M_OUT 4
#define B1 0.9
#define B2 0.999
#define E 1e-8
#define DC 0.01

double g_lr = 1e-5;

bool load_weights(const char* filename, double *W1, double *b1, double *W2, double *b2, 
                 double *W3, double *b3, double *W4, double *b4) {
    FILE* f = fopen(filename, "rb");
    if (!f) return false;
    size_t items_read = fread(W1, sizeof(double), D1*M_IN, f) + fread(b1, sizeof(double), D1, f) +
                        fread(W2, sizeof(double), D2*D1, f) + fread(b2, sizeof(double), D2, f) +
                        fread(W3, sizeof(double), D3*D2, f) + fread(b3, sizeof(double), D3, f) +
                        fread(W4, sizeof(double), M_OUT*D3, f) + fread(b4, sizeof(double), M_OUT, f);
    fclose(f);
    return items_read == (D1*M_IN + D1 + D2*D1 + D2 + D3*D2 + D3 + M_OUT*D3 + M_OUT);
}

void save_weights(const char* filename, double *W1, double *b1, double *W2, double *b2,
                 double *W3, double *b3, double *W4, double *b4) {
    FILE* f = fopen(filename, "wb");
    fwrite(W1, sizeof(double), D1*M_IN, f); fwrite(b1, sizeof(double), D1, f);
    fwrite(W2, sizeof(double), D2*D1, f); fwrite(b2, sizeof(double), D2, f);
    fwrite(W3, sizeof(double), D3*D2, f); fwrite(b3, sizeof(double), D3, f);
    fwrite(W4, sizeof(double), M_OUT*D3, f); fwrite(b4, sizeof(double), M_OUT, f);
    fclose(f);
}

void forward(double *W1, double *b1, double *W2, double *b2, double *W3, double *b3, 
            double *W4, double *b4, double *input, double *h1, double *h2, double *h3, double *output) {
    for(int i = 0; i < D1; i++) {
        double sum = b1[i];
        for(int j = 0; j < M_IN; j++) sum += W1[i*M_IN + j] * input[j];
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
    for(int i = 0; i < M_OUT; i++) {
        double sum = b4[i];
        for(int j = 0; j < D3; j++) sum += W4[i*D3 + j] * h3[j];
        output[i] = 50.0 + 50.0 / (1.0 + exp(-sum));
    }
}

void adam(double *p, double *g, double *m, double *v, int size, int t, bool ascend) {
    double lr_t = g_lr * sqrt(1.0 - pow(B2, t)) / (1.0 - pow(B1, t));
    for(int i = 0; i < size; i++) {
        m[i] = B1 * m[i] + (1-B1) * g[i];
        v[i] = B2 * v[i] + (1-B2) * g[i] * g[i];
        if (ascend)
            p[i] += lr_t * (m[i] / (sqrt(v[i]) + E) - DC * p[i]);
        else
            p[i] -= lr_t * (m[i] / (sqrt(v[i]) + E) + DC * p[i]);
    }
}

void backward(double *W1, double *b1, double *W2, double *b2, double *W3, double *b3, double *W4, double *b4,
             double *m_W1, double *m_b1, double *m_W2, double *m_b2, double *m_W3, double *m_b3, double *m_W4, double *m_b4,
             double *v_W1, double *v_b1, double *v_W2, double *v_b2, double *v_W3, double *v_b3, double *v_W4, double *v_b4,
             double *input, double *h1, double *h2, double *h3, double *output, double advantage, int step) {
    
    double *d_W1 = calloc(D1*M_IN, sizeof(double)), *d_b1 = calloc(D1, sizeof(double));
    double *d_W2 = calloc(D2*D1, sizeof(double)), *d_b2 = calloc(D2, sizeof(double));
    double *d_W3 = calloc(D3*D2, sizeof(double)), *d_b3 = calloc(D3, sizeof(double));
    double *d_W4 = calloc(M_OUT*D3, sizeof(double)), *d_b4 = calloc(M_OUT, sizeof(double));
    
    // Output layer gradients
    for(int i = 0; i < M_OUT; i++) {
        double sigmoid = (output[i] - 50.0) / 50.0;
        d_b4[i] = advantage * sigmoid * (1 - sigmoid);
        for(int j = 0; j < D3; j++) {
            d_W4[i*D3 + j] = d_b4[i] * h3[j];
        }
    }

    // Hidden layer gradients
    for(int i = 0; i < D3; i++) {
        double d_h3 = 0;
        for(int j = 0; j < M_OUT; j++) d_h3 += d_b4[j] * W4[j*D3 + i];
        d_h3 *= (h3[i] > 0 ? 1.0 : 0.1);
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
        for(int j = 0; j < M_IN; j++) d_W1[i*M_IN + j] = d_h1 * input[j];
        d_b1[i] = d_h1;
    }

    bool ascend = advantage > 0;
    adam(W1, d_W1, m_W1, v_W1, D1*M_IN, step, ascend);
    adam(b1, d_b1, m_b1, v_b1, D1, step, ascend);
    adam(W2, d_W2, m_W2, v_W2, D2*D1, step, ascend);
    adam(b2, d_b2, m_b2, v_b2, D2, step, ascend);
    adam(W3, d_W3, m_W3, v_W3, D3*D2, step, ascend);
    adam(b3, d_b3, m_b3, v_b3, D3, step, ascend);
    adam(W4, d_W4, m_W4, v_W4, M_OUT*D3, step, ascend);
    adam(b4, d_b4, m_b4, v_b4, M_OUT, step, ascend);

    free(d_W1); free(d_b1); free(d_W2); free(d_b2);
    free(d_W3); free(d_b3); free(d_W4); free(d_b4);
}

int main(int argc, char **argv) {
    if(argc != 3) {
        printf("Usage: %s <trajectory_csv> <policy_weights>\n", argv[0]);
        return 1;
    }

    // Allocate memory for network
    double *W1 = malloc(D1*M_IN*sizeof(double)), *b1 = calloc(D1, sizeof(double));
    double *W2 = malloc(D2*D1*sizeof(double)), *b2 = calloc(D2, sizeof(double));
    double *W3 = malloc(D3*D2*sizeof(double)), *b3 = calloc(D3, sizeof(double));
    double *W4 = malloc(M_OUT*D3*sizeof(double)), *b4 = calloc(M_OUT, sizeof(double));
    double *h1 = malloc(D1*sizeof(double)), *h2 = malloc(D2*sizeof(double));
    double *h3 = malloc(D3*sizeof(double));

    // Adam optimizer memory
    double *m_W1 = calloc(D1*M_IN, sizeof(double)), *m_b1 = calloc(D1, sizeof(double));
    double *m_W2 = calloc(D2*D1, sizeof(double)), *m_b2 = calloc(D2, sizeof(double));
    double *m_W3 = calloc(D3*D2, sizeof(double)), *m_b3 = calloc(D3, sizeof(double));
    double *m_W4 = calloc(M_OUT*D3, sizeof(double)), *m_b4 = calloc(M_OUT, sizeof(double));
    double *v_W1 = calloc(D1*M_IN, sizeof(double)), *v_b1 = calloc(D1, sizeof(double));
    double *v_W2 = calloc(D2*D1, sizeof(double)), *v_b2 = calloc(D2, sizeof(double));
    double *v_W3 = calloc(D3*D2, sizeof(double)), *v_b3 = calloc(D3, sizeof(double));
    double *v_W4 = calloc(M_OUT*D3, sizeof(double)), *v_b4 = calloc(M_OUT, sizeof(double));

    if (!load_weights(argv[2], W1, b1, W2, b2, W3, b3, W4, b4)) {
        printf("Failed to load weights\n");
        return 1;
    }

    FILE *f = fopen(argv[1], "r");
    if (!f) { printf("Failed to open CSV\n"); return 1; }

    char line[4096];
    fgets(line, sizeof(line), f); // Skip header

    int step = 1;
    double running_advantage = 0;
    
    while (fgets(line, sizeof(line), f)) {
        double input[M_IN], output[M_OUT];
        char *token = strtok(line, ",");
        
        // Skip rollout number
        token = strtok(NULL, ",");
        
        // Skip to acc_s (19th column now, due to rollout column)
        for(int i = 0; i < 18; i++) {
            token = strtok(NULL, ",");
            if (!token) { printf("Error: skipping to acc_s, column %d\n", i); return 1; }
        }
        
        // Read acc_s and gyro_s (6 values)
        for(int i = 0; i < M_IN; i++) {
            if (!token) { printf("Error: reading sensors, column %d\n", i); return 1; }
            input[i] = atof(token);
            token = strtok(NULL, ",");
        }

        // Skip omega[4], reward, discounted_return (6 values)
        for(int i = 0; i < 6; i++) {
            if (!token) { printf("Error: skipping to advantage, column %d\n", i); return 1; }
            token = strtok(NULL, ",");
        }
        
        if (!token) { printf("Error: reading advantage\n"); return 1; }
        double advantage = atof(token);

        forward(W1, b1, W2, b2, W3, b3, W4, b4, input, h1, h2, h3, output);
        backward(W1, b1, W2, b2, W3, b3, W4, b4,
                m_W1, m_b1, m_W2, m_b2, m_W3, m_b3, m_W4, m_b4,
                v_W1, v_b1, v_W2, v_b2, v_W3, v_b3, v_W4, v_b4,
                input, h1, h2, h3, output, advantage, step);

        running_advantage += advantage;
        if(step % 100 == 0) {
            printf("Step %d, Average advantage: %f\n", step, running_advantage/100);
            running_advantage = 0;
        }
        step++;
    }

    save_weights(argv[2], W1, b1, W2, b2, W3, b3, W4, b4);

    // Cleanup
    fclose(f);
    free(W1); free(b1); free(W2); free(b2); free(W3); free(b3); free(W4); free(b4);
    free(h1); free(h2); free(h3);
    free(m_W1); free(m_b1); free(m_W2); free(m_b2);
    free(m_W3); free(m_b3); free(m_W4); free(m_b4);
    free(v_W1); free(v_b1); free(v_W2); free(v_b2);
    free(v_W3); free(v_b3); free(v_W4); free(v_b4);

    return 0;
}