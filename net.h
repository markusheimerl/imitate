#ifndef NET_H
#define NET_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define DT_PHYSICS (1.0/1000.0)
#define DT_CONTROL (1.0/60.0)

#define STATE_DIM 6      // 3 accel + 3 gyro
#define ACTION_DIM 8     // 4 means + 4 stds
#define MAX_STEPS 256
#define NUM_ROLLOUTS 1024

#define GAMMA 0.999
#define MAX_STD 3.0
#define MIN_STD 1e-5

#define MAX_MEAN (OMEGA_MAX - 4.0 * MAX_STD)
#define MIN_MEAN (OMEGA_MIN + 4.0 * MAX_STD)

#define HIDDEN_DIM 64

typedef struct {
    // Weight matrices
    double W1[HIDDEN_DIM][STATE_DIM];
    double W2[ACTION_DIM][HIDDEN_DIM];
    
    // Layer activations
    double h[3][HIDDEN_DIM];  // h[0] is input, h[1] is hidden, h[2] is output
    
    // AdamW parameters
    double m1[HIDDEN_DIM][STATE_DIM];  // First moment for W1
    double m2[ACTION_DIM][HIDDEN_DIM];  // First moment for W2
    double v1[HIDDEN_DIM][STATE_DIM];  // Second moment for W1
    double v2[ACTION_DIM][HIDDEN_DIM];  // Second moment for W2
    
    // Gradient accumulation
    double dW1[HIDDEN_DIM][STATE_DIM];
    double dW2[ACTION_DIM][HIDDEN_DIM];
    
    // Hyperparameters
    double lr;        // Learning rate
    double beta1;     // First moment decay
    double beta2;     // Second moment decay
    double epsilon;   // Small constant for numerical stability
    double weight_decay; // L2 regularization
    unsigned long step;  // Number of optimization steps

    // State normalization parameters
    double state_mean[STATE_DIM];
    double state_var[STATE_DIM];
} Net;

__device__ __host__ double swish(double x) {
    return x / (1.0 + exp(-x));
}

__device__ __host__ double swish_derivative(double x) {
    double sigmoid = 1.0 / (1.0 + exp(-x));
    return sigmoid + x * sigmoid * (1.0 - sigmoid);
}

__device__ __host__ double squash(double x, double min, double max) { 
    return ((max + min) / 2.0) + ((max - min) / 2.0) * tanh(x); 
}

__device__ __host__ double dsquash(double x, double min, double max) { 
    return ((max - min) / 2.0) * (1.0 - tanh(x) * tanh(x)); 
}

__device__ __host__ void normalize_state(Net* net, const double* input, double* normalized) {
    // Welford's online algorithm for streaming mean/variance calculation
    // For each new value x:
    // δ = x - μ₍ₙ₎          (distance from current mean)
    // μ₍ₙ₊₁₎ = μ₍ₙ₎ + α*δ    (update mean, α=0.01 learning rate)
    // σ²₍ₙ₊₁₎ = σ²₍ₙ₎ + α*(δ*(x - μ₍ₙ₊₁₎) - σ²₍ₙ₎)
    for (int i = 0; i < STATE_DIM; i++) {
        // Step 1: Calculate deviation from current mean
        double delta = input[i] - net->state_mean[i];

        // Step 2: Update running mean: μ₍ₙ₊₁₎ = μ₍ₙ₎ + α*(x - μ₍ₙ₎)
        net->state_mean[i] += 0.01 * delta;

        // Step 3: Update running variance using Welford's formula
        // Note: δ*(x - μ₍ₙ₊₁₎) is the corrected sum of squares
        net->state_var[i] += 0.01 * (delta * (input[i] - net->state_mean[i]) - net->state_var[i]);
        
        // Step 4: Standardize to N(0,1): z = (x - μ)/σ
        // Add ε=1e-8 for numerical stability when σ ≈ 0
        normalized[i] = (input[i] - net->state_mean[i]) / 
                       (sqrt(net->state_var[i] + 1e-8));
    }
}

Net* create_net(double learning_rate) {
    Net* net = (Net*)calloc(1, sizeof(Net));
    if (!net) return NULL;

    net->lr = learning_rate;
    net->beta1 = 0.9;
    net->beta2 = 0.999;
    net->epsilon = 1e-8;
    net->weight_decay = 0.01;
    net->step = 0;

    // Xavier initialization
    double scale1 = sqrt(2.0 / (STATE_DIM + HIDDEN_DIM));
    double scale2 = sqrt(2.0 / (HIDDEN_DIM + ACTION_DIM));

    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < STATE_DIM; j++) {
            net->W1[i][j] = ((double)rand()/RAND_MAX * 2 - 1) * scale1;
            net->m1[i][j] = 0.0;
            net->v1[i][j] = 0.0;
        }
    }

    for (int i = 0; i < ACTION_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            net->W2[i][j] = ((double)rand()/RAND_MAX * 2 - 1) * scale2;
            net->m2[i][j] = 0.0;
            net->v2[i][j] = 0.0;
        }
    }

    // Initialize normalization parameters
    for (int i = 0; i < STATE_DIM; i++) {
        net->state_mean[i] = 0.0;
        net->state_var[i] = 1.0;
    }

    return net;
}

__device__ __host__ void forward_net(Net* net, const double* input) {
    // Normalize input
    double normalized_input[STATE_DIM];
    normalize_state(net, input, normalized_input);
    
    // Copy normalized input
    memcpy(net->h[0], normalized_input, STATE_DIM * sizeof(double));

    // Hidden layer
    memset(net->h[1], 0, HIDDEN_DIM * sizeof(double));
    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < STATE_DIM; j++) {
            net->h[1][i] += net->W1[i][j] * net->h[0][j];
        }
        net->h[1][i] = swish(net->h[1][i]);
    }

    // Output layer
    memset(net->h[2], 0, ACTION_DIM * sizeof(double));
    for (int i = 0; i < ACTION_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            net->h[2][i] += net->W2[i][j] * net->h[1][j];
        }
    }
}

__device__ __host__ void backward_net(Net* net, const double* output_gradients) {
    double delta[HIDDEN_DIM];
    
    // Output layer gradients
    for (int i = 0; i < ACTION_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            net->dW2[i][j] = output_gradients[i] * net->h[1][j];
        }
    }

    // Hidden layer gradients
    memset(delta, 0, HIDDEN_DIM * sizeof(double));
    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < ACTION_DIM; j++) {
            delta[i] += output_gradients[j] * net->W2[j][i];
        }
        delta[i] *= swish_derivative(net->h[1][i]);

        for (int j = 0; j < STATE_DIM; j++) {
            net->dW1[i][j] = delta[i] * net->h[0][j];
        }
    }
}

__device__ __host__ void zero_gradients(Net* net) {
    memset(net->dW1, 0, sizeof(net->dW1));
    memset(net->dW2, 0, sizeof(net->dW2));
}

__device__ __host__ void update_net(Net* net) {
    net->step++;
    double beta1_t = pow(net->beta1, net->step);
    double beta2_t = pow(net->beta2, net->step);
    
    // Update W1
    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < STATE_DIM; j++) {
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
    for (int i = 0; i < ACTION_DIM; i++) {
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
    
    // Save normalization parameters
    fwrite(net->state_mean, sizeof(double), STATE_DIM, file);
    fwrite(net->state_var, sizeof(double), STATE_DIM, file);
    
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
    
    // Load normalization parameters
    if (fread(net->state_mean, sizeof(double), STATE_DIM, file) != STATE_DIM ||
        fread(net->state_var, sizeof(double), STATE_DIM, file) != STATE_DIM) {
        free(net);
        fclose(file);
        return NULL;
    }
    
    fclose(file);
    return net;
}

#endif // NET_H