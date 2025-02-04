#ifndef NET_H
#define NET_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define INPUT_DIM 96   // 6 states * 16 history length
#define HIDDEN_DIM 64  // Power of 2
#define OUTPUT_DIM 8   // Padded from 4 (4 means + 4 stds)

typedef struct {
    // Weight matrices
    double W1[HIDDEN_DIM][INPUT_DIM];
    double W2[OUTPUT_DIM][HIDDEN_DIM];
    
    // Layer activations
    double h[3][HIDDEN_DIM];  // h[0] is input, h[1] is hidden, h[2] is output
    
    // Gradient accumulation
    double dW1[HIDDEN_DIM][INPUT_DIM];
    double dW2[OUTPUT_DIM][HIDDEN_DIM];
    
    double lr;
} Net;

static double gelu(double x) {
    return 0.5 * x * (1 + tanh(sqrt(2/M_PI) * (x + 0.044715 * pow(x, 3))));
}

static double gelu_derivative(double x) {
    double cdf = 0.5 * (1 + tanh(sqrt(2/M_PI) * (x + 0.044715 * pow(x, 3))));
    double pdf = exp(-0.5 * x * x) / sqrt(2 * M_PI);
    return cdf + x * pdf;
}

Net* create_net(double learning_rate) {
    Net* net = (Net*)calloc(1, sizeof(Net));
    if (!net) return NULL;

    net->lr = learning_rate;

    // Xavier initialization
    double scale1 = sqrt(2.0 / (INPUT_DIM + HIDDEN_DIM));
    double scale2 = sqrt(2.0 / (HIDDEN_DIM + OUTPUT_DIM));

    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < INPUT_DIM; j++) {
            net->W1[i][j] = ((double)rand()/RAND_MAX * 2 - 1) * scale1;
        }
    }

    for (int i = 0; i < OUTPUT_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            net->W2[i][j] = ((double)rand()/RAND_MAX * 2 - 1) * scale2;
        }
    }

    return net;
}

void forward_net(Net* net, const double* input) {
    // Copy input
    memcpy(net->h[0], input, INPUT_DIM * sizeof(double));

    // Hidden layer
    memset(net->h[1], 0, HIDDEN_DIM * sizeof(double));
    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < INPUT_DIM; j++) {
            net->h[1][i] += net->W1[i][j] * net->h[0][j];
        }
        net->h[1][i] = gelu(net->h[1][i]);
    }

    // Output layer
    memset(net->h[2], 0, OUTPUT_DIM * sizeof(double));
    for (int i = 0; i < OUTPUT_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            net->h[2][i] += net->W2[i][j] * net->h[1][j];
        }
    }
}

void backward_net(Net* net, const double* output_gradient) {
    double delta[HIDDEN_DIM];  // Hidden layer deltas

    // Output layer gradients
    for (int i = 0; i < OUTPUT_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            net->dW2[i][j] += output_gradient[i] * net->h[1][j];
        }
    }

    // Hidden layer gradients
    memset(delta, 0, HIDDEN_DIM * sizeof(double));
    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < OUTPUT_DIM; j++) {
            delta[i] += output_gradient[j] * net->W2[j][i];
        }
        delta[i] *= gelu_derivative(net->h[1][i]);

        for (int j = 0; j < INPUT_DIM; j++) {
            net->dW1[i][j] += delta[i] * net->h[0][j];
        }
    }
}

void zero_gradients(Net* net) {
    memset(net->dW1, 0, sizeof(net->dW1));
    memset(net->dW2, 0, sizeof(net->dW2));
}

void update_net(Net* net) {
    // Update first layer weights
    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < INPUT_DIM; j++) {
            net->W1[i][j] -= net->lr * net->dW1[i][j];
        }
    }

    // Update second layer weights
    for (int i = 0; i < OUTPUT_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            net->W2[i][j] -= net->lr * net->dW2[i][j];
        }
    }
}

bool save_net(const char* filename, const Net* net) {
    FILE* file = fopen(filename, "wb");
    if (!file) return false;
    
    // Save learning rate and weights
    fwrite(&net->lr, sizeof(double), 1, file);
    fwrite(net->W1, sizeof(net->W1), 1, file);
    fwrite(net->W2, sizeof(net->W2), 1, file);
    
    fclose(file);
    return true;
}

Net* load_net(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) return NULL;
    
    Net* net = (Net*)calloc(1, sizeof(Net));
    if (!net) {
        fclose(file);
        return NULL;
    }
    
    // Load learning rate and weights
    if (fread(&net->lr, sizeof(double), 1, file) != 1 ||
        fread(net->W1, sizeof(net->W1), 1, file) != 1 ||
        fread(net->W2, sizeof(net->W2), 1, file) != 1) {
        free(net);
        fclose(file);
        return NULL;
    }
    
    fclose(file);
    return net;
}

void free_net(Net* net) {
    free(net);
}

#endif // NET_H