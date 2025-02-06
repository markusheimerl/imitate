#ifndef NET_H
#define NET_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define INPUT_DIM 6 // STATE_DIM
#define HIDDEN_DIM 64
#define OUTPUT_DIM 8 // ACTION_DIM

typedef struct {
    // Weight matrices
    double W1[HIDDEN_DIM][INPUT_DIM];
    double W2[OUTPUT_DIM][HIDDEN_DIM];
    
    // Layer activations
    double h[3][HIDDEN_DIM];  // h[0] is input, h[1] is hidden, h[2] is output
    
    // AdamW parameters
    double m1[HIDDEN_DIM][INPUT_DIM];  // First moment for W1
    double m2[OUTPUT_DIM][HIDDEN_DIM];  // First moment for W2
    double v1[HIDDEN_DIM][INPUT_DIM];  // Second moment for W1
    double v2[OUTPUT_DIM][HIDDEN_DIM];  // Second moment for W2
    
    // Gradient accumulation
    double dW1[HIDDEN_DIM][INPUT_DIM];
    double dW2[OUTPUT_DIM][HIDDEN_DIM];
    
    // Hyperparameters
    double lr;        // Learning rate
    double beta1;     // First moment decay
    double beta2;     // Second moment decay
    double epsilon;   // Small constant for numerical stability
    double weight_decay; // L2 regularization
    unsigned long step;  // Number of optimization steps
} Net;

__device__ __host__ double gelu(double x) {
    return 0.5 * x * (1 + tanh(sqrt(2/M_PI) * (x + 0.044715 * pow(x, 3))));
}

__device__ __host__ double gelu_derivative(double x) {
    double cdf = 0.5 * (1 + tanh(sqrt(2/M_PI) * (x + 0.044715 * pow(x, 3))));
    double pdf = exp(-0.5 * x * x) / sqrt(2 * M_PI);
    return cdf + x * pdf;
}

Net* create_net(double learning_rate) {
    Net* net;
    cudaMallocManaged(&net, sizeof(Net));
    if (!net) return NULL;

    net->lr = learning_rate;
    net->beta1 = 0.9;
    net->beta2 = 0.999;
    net->epsilon = 1e-8;
    net->weight_decay = 0.01;
    net->step = 0;

    // Xavier initialization
    double scale1 = sqrt(2.0 / (INPUT_DIM + HIDDEN_DIM));
    double scale2 = sqrt(2.0 / (HIDDEN_DIM + OUTPUT_DIM));

    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < INPUT_DIM; j++) {
            net->W1[i][j] = ((double)rand()/RAND_MAX * 2 - 1) * scale1;
            net->m1[i][j] = 0.0;
            net->v1[i][j] = 0.0;
        }
    }

    for (int i = 0; i < OUTPUT_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            net->W2[i][j] = ((double)rand()/RAND_MAX * 2 - 1) * scale2;
            net->m2[i][j] = 0.0;
            net->v2[i][j] = 0.0;
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

void backward_net(Net* net, const double* output_gradients) {
    double delta[HIDDEN_DIM];
    
    // Output layer gradients
    for (int i = 0; i < OUTPUT_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            net->dW2[i][j] = output_gradients[i] * net->h[1][j];
        }
    }

    // Hidden layer gradients
    memset(delta, 0, HIDDEN_DIM * sizeof(double));
    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < OUTPUT_DIM; j++) {
            delta[i] += output_gradients[j] * net->W2[j][i];
        }
        delta[i] *= gelu_derivative(net->h[1][i]);

        for (int j = 0; j < INPUT_DIM; j++) {
            net->dW1[i][j] = delta[i] * net->h[0][j];
        }
    }
}

void zero_gradients(Net* net) {
    memset(net->dW1, 0, sizeof(net->dW1));
    memset(net->dW2, 0, sizeof(net->dW2));
}

void update_net(Net* net) {
    net->step++;
    double beta1_t = pow(net->beta1, net->step);
    double beta2_t = pow(net->beta2, net->step);
    
    // Update W1
    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < INPUT_DIM; j++) {
            // Add L2 regularization gradient
            net->dW1[i][j] += net->weight_decay * net->W1[i][j];
            
            // Update moments
            net->m1[i][j] = net->beta1 * net->m1[i][j] + (1 - net->beta1) * net->dW1[i][j];
            net->v1[i][j] = net->beta2 * net->v1[i][j] + (1 - net->beta2) * net->dW1[i][j] * net->dW1[i][j];
            
            // Bias correction
            double m_hat = net->m1[i][j] / (1 - beta1_t);
            double v_hat = net->v1[i][j] / (1 - beta2_t);
            
            // Update weights
            net->W1[i][j] -= net->lr * m_hat / (sqrt(v_hat) + net->epsilon);
        }
    }
    
    // Update W2
    for (int i = 0; i < OUTPUT_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            // Add L2 regularization gradient
            net->dW2[i][j] += net->weight_decay * net->W2[i][j];
            
            // Update moments
            net->m2[i][j] = net->beta1 * net->m2[i][j] + (1 - net->beta1) * net->dW2[i][j];
            net->v2[i][j] = net->beta2 * net->v2[i][j] + (1 - net->beta2) * net->dW2[i][j] * net->dW2[i][j];
            
            // Bias correction
            double m_hat = net->m2[i][j] / (1 - beta1_t);
            double v_hat = net->v2[i][j] / (1 - beta2_t);
            
            // Update weights
            net->W2[i][j] -= net->lr * m_hat / (sqrt(v_hat) + net->epsilon);
        }
    }
}

bool save_net(const char* filename, const Net* net) {
    FILE* file = fopen(filename, "wb");
    if (!file) return false;
    
    // Save hyperparameters
    fwrite(&net->lr, sizeof(double), 1, file);
    fwrite(&net->beta1, sizeof(double), 1, file);
    fwrite(&net->beta2, sizeof(double), 1, file);
    fwrite(&net->epsilon, sizeof(double), 1, file);
    fwrite(&net->weight_decay, sizeof(double), 1, file);
    fwrite(&net->step, sizeof(unsigned long), 1, file);
    
    // Save weights and optimizer state
    fwrite(net->W1, sizeof(net->W1), 1, file);
    fwrite(net->W2, sizeof(net->W2), 1, file);
    fwrite(net->m1, sizeof(net->m1), 1, file);
    fwrite(net->m2, sizeof(net->m2), 1, file);
    fwrite(net->v1, sizeof(net->v1), 1, file);
    fwrite(net->v2, sizeof(net->v2), 1, file);
    
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
    
    // Load hyperparameters
    if (fread(&net->lr, sizeof(double), 1, file) != 1 ||
        fread(&net->beta1, sizeof(double), 1, file) != 1 ||
        fread(&net->beta2, sizeof(double), 1, file) != 1 ||
        fread(&net->epsilon, sizeof(double), 1, file) != 1 ||
        fread(&net->weight_decay, sizeof(double), 1, file) != 1 ||
        fread(&net->step, sizeof(unsigned long), 1, file) != 1) {
        free(net);
        fclose(file);
        return NULL;
    }
    
    // Load weights and optimizer state
    if (fread(net->W1, sizeof(net->W1), 1, file) != 1 ||
        fread(net->W2, sizeof(net->W2), 1, file) != 1 ||
        fread(net->m1, sizeof(net->m1), 1, file) != 1 ||
        fread(net->m2, sizeof(net->m2), 1, file) != 1 ||
        fread(net->v1, sizeof(net->v1), 1, file) != 1 ||
        fread(net->v2, sizeof(net->v2), 1, file) != 1) {
        free(net);
        fclose(file);
        return NULL;
    }
    
    fclose(file);
    return net;
}

#endif // NET_H