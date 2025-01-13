#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include "grad.h"

#define D1 64
#define D2 32
#define D3 16
#define M_IN 9  // pos[3], vel[3], ang_vel[3]
#define M_OUT 1 // Estimated cumulative discounted reward

static double prev_loss = 1e30;
static double learning_rate = 1e-4;

void forward_value(double *W1, double *b1, double *W2, double *b2, double *W3, double *b3, double *W4, double *b4, double *input, double *h1, double *h2, double *h3, double *output) {
    for (int i = 0; i < D1; i++) h1[i] = l_relu(b1[i] + dot(&W1[i * M_IN], input, M_IN));
    for (int i = 0; i < D2; i++) h2[i] = l_relu(b2[i] + dot(&W2[i * D1], h1, D1));
    for (int i = 0; i < D3; i++) h3[i] = l_relu(b3[i] + dot(&W3[i * D2], h2, D2));
    for (int i = 0; i < M_OUT; i++) output[i] = b4[i] + dot(&W4[i * D3], h3, D3);
}

double backward_value(double *W1, double *b1, double *W2, double *b2, double *W3, double *b3,
               double *W4, double *b4, double *m_W1, double *m_b1, double *m_W2, double *m_b2,
               double *m_W3, double *m_b3, double *m_W4, double *m_b4, double *v_W1, double *v_b1,
               double *v_W2, double *v_b2, double *v_W3, double *v_b3, double *v_W4, double *v_b4,
               double *input, double *h1, double *h2, double *h3, double *output, double target, 
               double *d_W1, double *d_b1, double *d_W2, double *d_b2, double *d_W3, double *d_b3, 
               double *d_W4, double *d_h1, double *d_h2, double *d_h3, int step) {
    
    memset(d_W1, 0, D1*M_IN*sizeof(double));
    memset(d_b1, 0, D1*sizeof(double));
    memset(d_W2, 0, D2*D1*sizeof(double));
    memset(d_b2, 0, D2*sizeof(double));
    memset(d_W3, 0, D3*D2*sizeof(double));
    memset(d_b3, 0, D3*sizeof(double));
    memset(d_W4, 0, D3*sizeof(double));

    double d_out = 2.0 * (*output - target);
    double d_b4 = d_out;
    for(int i = 0; i < D3; i++) d_W4[i] = d_out * h3[i];
    for(int i = 0; i < D3; i++) d_h3[i] = d_out * W4[i] * d_l_relu(h3[i]);
    d_dot_left(d_W3, d_h3, h2, D2, D3);
    memcpy(d_b3, d_h3, D3 * sizeof(double));
    d_dot_right(d_h2, d_h3, W3, D2, D3);
    for(int i = 0; i < D2; i++) d_h2[i] *= d_l_relu(h2[i]);
    d_dot_left(d_W2, d_h2, h1, D1, D2);
    memcpy(d_b2, d_h2, D2 * sizeof(double));
    d_dot_right(d_h1, d_h2, W2, D1, D2);
    for(int i = 0; i < D1; i++) d_h1[i] *= d_l_relu(h1[i]);
    d_dot_left(d_W1, d_h1, input, M_IN, D1);
    memcpy(d_b1, d_h1, D1 * sizeof(double));

    adam_update(W1, d_W1, m_W1, v_W1, D1*M_IN, step, learning_rate);
    adam_update(b1, d_b1, m_b1, v_b1, D1, step, learning_rate);
    adam_update(W2, d_W2, m_W2, v_W2, D2*D1, step, learning_rate);
    adam_update(b2, d_b2, m_b2, v_b2, D2, step, learning_rate);
    adam_update(W3, d_W3, m_W3, v_W3, D3*D2, step, learning_rate);
    adam_update(b3, d_b3, m_b3, v_b3, D3, step, learning_rate);
    adam_update(W4, d_W4, m_W4, v_W4, D3, step, learning_rate);
    adam_update(b4, &d_b4, m_b4, v_b4, 1, step, learning_rate);
    
    return (*output - target) * (*output - target);
}

int main(int argc, char **argv) {
    if(argc < 2) {
        printf("Usage: %s <data_file> [weights_file]\n", argv[0]);
        return 1;
    }

    // Initialize network parameters
    double *W1, *b1, *W2, *b2, *W3, *b3, *W4, *b4;
    double *h1 = malloc(D1*sizeof(double));
    double *h2 = malloc(D2*sizeof(double));
    double *h3 = malloc(D3*sizeof(double));

    double *m_W1 = calloc(D1*M_IN, sizeof(double)), *m_b1 = calloc(D1, sizeof(double));
    double *m_W2 = calloc(D2*D1, sizeof(double)), *m_b2 = calloc(D2, sizeof(double));
    double *m_W3 = calloc(D3*D2, sizeof(double)), *m_b3 = calloc(D3, sizeof(double));
    double *m_W4 = calloc(D3, sizeof(double)), *m_b4 = calloc(1, sizeof(double));
    
    double *v_W1 = calloc(D1*M_IN, sizeof(double)), *v_b1 = calloc(D1, sizeof(double));
    double *v_W2 = calloc(D2*D1, sizeof(double)), *v_b2 = calloc(D2, sizeof(double));
    double *v_W3 = calloc(D3*D2, sizeof(double)), *v_b3 = calloc(D3, sizeof(double));
    double *v_W4 = calloc(D3, sizeof(double)), *v_b4 = calloc(1, sizeof(double));

    double *d_W1 = malloc(D1*M_IN*sizeof(double));
    double *d_b1 = malloc(D1*sizeof(double));
    double *d_W2 = malloc(D2*D1*sizeof(double));
    double *d_b2 = malloc(D2*sizeof(double));
    double *d_W3 = malloc(D3*D2*sizeof(double));
    double *d_b3 = malloc(D3*sizeof(double));
    double *d_W4 = malloc(D3*sizeof(double));
    double *d_h1 = malloc(D1*sizeof(double));
    double *d_h2 = malloc(D2*sizeof(double));
    double *d_h3 = malloc(D3*sizeof(double));

    if (argc > 2) {
        load_weights(argv[2], (double*[]){W1, b1, W2, b2, W3, b3, W4, b4}, (int[]){M_IN * D1, D1, D1 * D2, D2, D2 * D3, D3, D3 * M_OUT, M_OUT}, 8);
    } else {
        srand(time(NULL));
        init_linear(&W1, &b1, M_IN, D1);
        init_linear(&W2, &b2, D1, D2);
        init_linear(&W3, &b3, D2, D3);
        init_linear(&W4, &b4, D3, M_OUT);
    }
    
    Logger* logger = init_logger();
    read_trajectories(logger, argv[1]);

    // Training loop
    double output, running_loss = 0;
    int step = 1;

    for(int epoch = 0; epoch < 3; epoch++) {
        // Shuffle indices
        for(int i = logger->rows-1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = logger->indices[i];
            logger->indices[i] = logger->indices[j];
            logger->indices[j] = temp;
        }

        for(int i = 0; i < logger->rows; i++, step++) {
            forward_value(W1, b1, W2, b2, W3, b3, W4, b4, logger->data[logger->indices[i]], h1, h2, h3, &output);
            running_loss += backward_value(W1, b1, W2, b2, W3, b3, W4, b4, m_W1, m_b1, m_W2, m_b2, m_W3, m_b3, m_W4, m_b4, v_W1, v_b1, v_W2, v_b2, v_W3, v_b3, v_W4, v_b4, logger->data[logger->indices[i]], h1, h2, h3, &output, logger->targets[logger->indices[i]], d_W1, d_b1, d_W2, d_b2, d_W3, d_b3, d_W4, d_h1, d_h2, d_h3, step);

            if(step % 30000 == 0) {
                double avg_loss = (running_loss/30000);
                learning_rate *= (avg_loss > prev_loss) ? 0.95 : 1.05;
                learning_rate = fmax(1e-6, fmin(1e-2, learning_rate));
                printf("%3d | %6d | %.6f | %.2e | %.6f | %.6f\n", epoch, step, avg_loss, learning_rate, output, targets[indices[i]]);
                prev_loss = avg_loss;
                running_loss = 0;
            }
        }
    }

    if (argc > 2){
        save_weights(argv[2], (double*[]){W1, b1, W2, b2, W3, b3, W4, b4}, (int[]){M_IN * D1, D1, D1 * D2, D2, D2 * D3, D3, D3 * M_OUT, M_OUT}, 8);
    }else{
        char filename[100];
        strftime(filename, 100, "%Y-%m-%d_%H-%M-%S_value_weights.bin", localtime(&(time_t){time(NULL)}));
        save_weights(filename, (double*[]){W1, b1, W2, b2, W3, b3, W4, b4}, (int[]){M_IN * D1, D1, D1 * D2, D2, D2 * D3, D3, D3 * M_OUT, M_OUT}, 8);
    }

    // Cleanup
    free_logger(logger);
    free(data); free(targets); free(indices);
    free(W1); free(b1); free(W2); free(b2); free(W3); free(b3); free(W4); free(b4);
    free(h1); free(h2); free(h3);
    free(m_W1); free(m_b1); free(m_W2); free(m_b2);
    free(m_W3); free(m_b3); free(m_W4); free(m_b4);
    free(v_W1); free(v_b1); free(v_W2); free(v_b2);
    free(v_W3); free(v_b3); free(v_W4); free(v_b4);
    free(d_W1); free(d_b1); free(d_W2); free(d_b2);
    free(d_W3); free(d_b3); free(d_W4);
    free(d_h1); free(d_h2); free(d_h3);

    return 0;
}