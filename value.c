#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define D1 64  // First hidden layer
#define D2 32  // Second hidden layer
#define D3 16   // Third hidden layer
#define M 18     // Input dimension (3 pos + 3 vel + 3 ang_vel + 9 rotation)
#define B1 0.9   // Adam beta1
#define B2 0.999 // Adam beta2
#define E 1e-8   // Adam epsilon
#define DC 0.01  // Weight decay

double g_prev_loss = 1e30, g_lr = 1e-4;

typedef struct {
    double *W1, *b1, *W2, *b2, *W3, *b3, *W4, *b4;
    double *h1, *h2, *h3;
} Network;

typedef struct {
    double *m_W1, *m_b1, *m_W2, *m_b2, *m_W3, *m_b3, *m_W4, *m_b4;
    double *v_W1, *v_b1, *v_W2, *v_b2, *v_v_W3, *v_b3, *v_W4, *v_b4;
} Adam;

void forward(Network *net, double *input, double *out) {
    // First layer
    for(int i = 0; i < D1; i++) {
        double sum = net->b1[i];
        for(int j = 0; j < M; j++) sum += net->W1[i * M + j] * input[j];
        net->h1[i] = sum > 0 ? sum : sum * 0.1; // LeakyReLU
    }
    
    // Second layer
    for(int i = 0; i < D2; i++) {
        double sum = net->b2[i];
        for(int j = 0; j < D1; j++) sum += net->W2[i * D1 + j] * net->h1[j];
        net->h2[i] = sum > 0 ? sum : sum * 0.1;
    }
    
    // Third layer
    for(int i = 0; i < D3; i++) {
        double sum = net->b3[i];
        for(int j = 0; j < D2; j++) sum += net->W3[i * D2 + j] * net->h2[j];
        net->h3[i] = sum > 0 ? sum : sum * 0.1;
    }
    
    // Output layer
    *out = net->b4[0];
    for(int i = 0; i < D3; i++) *out += net->W4[i] * net->h3[i];
}

void adam(double *p, double *g, double *m, double *v, int size, int t) {
    double lr_t = g_lr * sqrt(1.0 - pow(B2, t)) / (1.0 - pow(B1, t));
    for(int i = 0; i < size; i++) {
        m[i] = B1 * m[i] + (1-B1) * g[i];
        v[i] = B2 * v[i] + (1-B2) * g[i] * g[i];
        p[i] -= lr_t * (m[i] / (sqrt(v[i]) + E) + DC * p[i]);
    }
}

double backward(Network *net, Adam *opt, double *input, double *out, double target, int step) {
    // Compute gradients
    double d_out = 2 * (*out - target);
    double *d_W1 = calloc(D1 * M, sizeof(double)), *d_b1 = calloc(D1, sizeof(double));
    double *d_W2 = calloc(D2 * D1, sizeof(double)), *d_b2 = calloc(D2, sizeof(double));
    double *d_W3 = calloc(D3 * D2, sizeof(double)), *d_b3 = calloc(D3, sizeof(double));
    double *d_W4 = calloc(D3, sizeof(double)), d_b4 = d_out;
    
    // Output layer
    for(int i = 0; i < D3; i++) {
        d_W4[i] = d_out * net->h3[i];
        double d_h3 = d_out * net->W4[i] * (net->h3[i] > 0 ? 1.0 : 0.1);
        for(int j = 0; j < D2; j++) d_W3[i * D2 + j] = d_h3 * net->h2[j];
        d_b3[i] = d_h3;
    }
    
    // Hidden layer 2
    for(int i = 0; i < D2; i++) {
        double d_h2 = 0;
        for(int j = 0; j < D3; j++) d_h2 += d_b3[j] * net->W3[j * D2 + i];
        d_h2 *= (net->h2[i] > 0 ? 1.0 : 0.1);
        for(int j = 0; j < D1; j++) d_W2[i * D1 + j] = d_h2 * net->h1[j];
        d_b2[i] = d_h2;
    }
    
    // Hidden layer 1
    for(int i = 0; i < D1; i++) {
        double d_h1 = 0;
        for(int j = 0; j < D2; j++) d_h1 += d_b2[j] * net->W2[j * D1 + i];
        d_h1 *= (net->h1[i] > 0 ? 1.0 : 0.1);
        for(int j = 0; j < M; j++) d_W1[i * M + j] = d_h1 * input[j];
        d_b1[i] = d_h1;
    }

    // Update weights
    adam(net->W1, d_W1, opt->m_W1, opt->v_W1, D1 * M, step);
    adam(net->b1, d_b1, opt->m_b1, opt->v_b1, D1, step);
    adam(net->W2, d_W2, opt->m_W2, opt->v_W2, D2 * D1, step);
    adam(net->b2, d_b2, opt->m_b2, opt->v_b2, D2, step);
    adam(net->W3, d_W3, opt->m_W3, opt->v_v_W3, D3 * D2, step);
    adam(net->b3, d_b3, opt->m_b3, opt->v_b3, D3, step);
    adam(net->W4, d_W4, opt->m_W4, opt->v_W4, D3, step);
    adam(net->b4, &d_b4, opt->m_b4, opt->v_b4, 1, step);

    free(d_W1); free(d_b1); free(d_W2); free(d_b2);
    free(d_W3); free(d_b3); free(d_W4);
    return (*out - target) * (*out - target);
}

int main(int argc, char **argv) {
    if(argc != 2) { printf("Usage: %s <data_file>\n", argv[0]); return 1; }
    
    srand(time(NULL));
    Network net = {
        .W1 = malloc(D1 * M * sizeof(double)), .b1 = calloc(D1, sizeof(double)),
        .W2 = malloc(D2 * D1 * sizeof(double)), .b2 = calloc(D2, sizeof(double)),
        .W3 = malloc(D3 * D2 * sizeof(double)), .b3 = calloc(D3, sizeof(double)),
        .W4 = malloc(D3 * sizeof(double)), .b4 = calloc(1, sizeof(double)),
        .h1 = malloc(D1 * sizeof(double)),
        .h2 = malloc(D2 * sizeof(double)),
        .h3 = malloc(D3 * sizeof(double))
    };
    
    Adam opt = {
        .m_W1 = calloc(D1 * M, sizeof(double)), .m_b1 = calloc(D1, sizeof(double)),
        .m_W2 = calloc(D2 * D1, sizeof(double)), .m_b2 = calloc(D2, sizeof(double)),
        .m_W3 = calloc(D3 * D2, sizeof(double)), .m_b3 = calloc(D3, sizeof(double)),
        .m_W4 = calloc(D3, sizeof(double)), .m_b4 = calloc(1, sizeof(double)),
        .v_W1 = calloc(D1 * M, sizeof(double)), .v_b1 = calloc(D1, sizeof(double)),
        .v_W2 = calloc(D2 * D1, sizeof(double)), .v_b2 = calloc(D2, sizeof(double)),
        .v_v_W3 = calloc(D3 * D2, sizeof(double)), .v_b3 = calloc(D3, sizeof(double)),
        .v_W4 = calloc(D3, sizeof(double)), .v_b4 = calloc(1, sizeof(double))
    };

    // Xavier initialization
    for(int i = 0; i < D1 * M; i++) net.W1[i] = ((double)rand()/RAND_MAX - 0.5) * sqrt(2.0/M);
    for(int i = 0; i < D2 * D1; i++) net.W2[i] = ((double)rand()/RAND_MAX - 0.5) * sqrt(2.0/D1);
    for(int i = 0; i < D3 * D2; i++) net.W3[i] = ((double)rand()/RAND_MAX - 0.5) * sqrt(2.0/D2);
    for(int i = 0; i < D3; i++) net.W4[i] = ((double)rand()/RAND_MAX - 0.5) * sqrt(2.0/D3);

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
        for(int j = 0; j < M; j++) {
            data[i][j] = atof(token);
            token = strtok(NULL, ",");
        }
        for(int j = 0; j < 4; j++) token = strtok(NULL, ","); // Skip to discounted return
        targets[i] = atof(token);
    }
    fclose(f);

    double out, running_loss = 0;
    int step = 1;

    for(int epoch = 0; epoch < 10; epoch++) {
        for(int i = rows-1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        
        for(int i = 0; i < rows; i++, step++) {
            forward(&net, data[indices[i]], &out);
            running_loss += backward(&net, &opt, data[indices[i]], &out, targets[indices[i]], step);

            if(step % 10000 == 0) {
                double avg_loss = running_loss/10000;
                g_lr *= (avg_loss > g_prev_loss) ? 0.95 : 1.05;
                g_lr = fmax(1e-6, fmin(1e-3, g_lr));
                printf("Epoch %d, Step %d, Loss: %f, LR: %e\n", epoch, step, avg_loss, g_lr);
                if(step % 100000 == 0) {
                    printf("Sample pred: %8.3f true: %8.3f\n\n", out, targets[indices[i]]);
                }
                g_prev_loss = avg_loss;
                running_loss = 0;
            }
        }
    }

    char filename[100];
    time_t t = time(NULL); struct tm tm = *localtime(&t);
    sprintf(filename, "%d-%d-%d_%d-%d-%d_weights.bin", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    FILE *wf = fopen(filename, "wb");
    fwrite(net.W1, sizeof(double), D1 * M, wf);
    fwrite(net.b1, sizeof(double), D1, wf);
    fwrite(net.W2, sizeof(double), D2 * D1, wf);
    fwrite(net.b2, sizeof(double), D2, wf);
    fwrite(net.W3, sizeof(double), D3 * D2, wf);
    fwrite(net.b3, sizeof(double), D3, wf);
    fwrite(net.W4, sizeof(double), D3, wf);
    fwrite(net.b4, sizeof(double), 1, wf);
    fclose(wf);

    // Cleanup
    free(net.W1); free(net.b1); free(net.W2); free(net.b2);
    free(net.W3); free(net.b3); free(net.W4); free(net.b4);
    free(net.h1); free(net.h2); free(net.h3);
    free(opt.m_W1); free(opt.m_b1); free(opt.m_W2); free(opt.m_b2);
    free(opt.m_W3); free(opt.m_b3); free(opt.m_W4); free(opt.m_b4);
    free(opt.v_W1); free(opt.v_b1); free(opt.v_W2); free(opt.v_b2);
    free(opt.v_v_W3); free(opt.v_b3); free(opt.v_W4); free(opt.v_b4);
    for(int i = 0; i < rows; i++) free(data[i]);
    free(data); free(targets); free(indices);
    return 0;
}