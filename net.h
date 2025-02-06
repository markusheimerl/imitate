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
    float W1[HIDDEN_DIM][INPUT_DIM];
    float W2[OUTPUT_DIM][HIDDEN_DIM];
    
    // Layer activations
    float h[3][HIDDEN_DIM];  // h[0] is input, h[1] is hidden, h[2] is output
    
    // Gradient accumulation
    float dW1[HIDDEN_DIM][INPUT_DIM];
    float dW2[OUTPUT_DIM][HIDDEN_DIM];
    
    // Sophia parameters
    float m1[HIDDEN_DIM][INPUT_DIM];  // Momentum
    float m2[OUTPUT_DIM][HIDDEN_DIM];
    float h1[HIDDEN_DIM][INPUT_DIM];  // Hessian estimates
    float h2[OUTPUT_DIM][HIDDEN_DIM];
    float prev_dW1[HIDDEN_DIM][INPUT_DIM];  // Previous gradients
    float prev_dW2[OUTPUT_DIM][HIDDEN_DIM];
    unsigned long long t;
    
    float lr;
    float beta1;    // Momentum decay
    float beta2;    // Hessian decay
    float rho;      // Hessian clipping
    float epsilon;
} Net;

static float gelu(float x) {
    return 0.5 * x * (1 + tanh(sqrt(2/M_PI) * (x + 0.044715 * pow(x, 3))));
}

static float gelu_derivative(float x) {
    float cdf = 0.5 * (1 + tanh(sqrt(2/M_PI) * (x + 0.044715 * pow(x, 3))));
    float pdf = exp(-0.5 * x * x) / sqrt(2 * M_PI);
    return cdf + x * pdf;
}

Net* create_net(float learning_rate) {
    Net* net = (Net*)calloc(1, sizeof(Net));
    if (!net) return NULL;

    net->lr = learning_rate;
    net->beta1 = 0.9;     // Momentum decay
    net->beta2 = 0.999;   // Hessian decay
    net->rho = 0.01;      // Hessian clipping
    net->epsilon = 1e-8;
    net->t = 0;

    // Xavier initialization
    float scale1 = sqrt(2.0 / (INPUT_DIM + HIDDEN_DIM));
    float scale2 = sqrt(2.0 / (HIDDEN_DIM + OUTPUT_DIM));

    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < INPUT_DIM; j++) {
            net->W1[i][j] = ((float)rand()/RAND_MAX * 2 - 1) * scale1;
        }
    }

    for (int i = 0; i < OUTPUT_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            net->W2[i][j] = ((float)rand()/RAND_MAX * 2 - 1) * scale2;
        }
    }

    return net;
}

void forward_net(Net* net, const float* input) {
    // Copy input
    memcpy(net->h[0], input, INPUT_DIM * sizeof(float));

    // Hidden layer
    memset(net->h[1], 0, HIDDEN_DIM * sizeof(float));
    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < INPUT_DIM; j++) {
            net->h[1][i] += net->W1[i][j] * net->h[0][j];
        }
        net->h[1][i] = gelu(net->h[1][i]);
    }

    // Output layer
    memset(net->h[2], 0, OUTPUT_DIM * sizeof(float));
    for (int i = 0; i < OUTPUT_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            net->h[2][i] += net->W2[i][j] * net->h[1][j];
        }
    }
}

void backward_net(Net* net, const float* output_gradients) {
    float delta[HIDDEN_DIM];
    
    // Output layer gradients
    for (int i = 0; i < OUTPUT_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            net->dW2[i][j] = output_gradients[i] * net->h[1][j];
        }
    }

    // Hidden layer gradients
    memset(delta, 0, HIDDEN_DIM * sizeof(float));
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
    net->t++;
    
    // Update first layer weights using Sophia
    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < INPUT_DIM; j++) {
            // Update momentum
            net->m1[i][j] = net->beta1 * net->m1[i][j] + (1.0 - net->beta1) * net->dW1[i][j];
            
            // Approximate second derivative using consecutive gradients
            float grad_diff = net->dW1[i][j] - net->prev_dW1[i][j];
            float hessian_approx = fabs(grad_diff / (net->lr + net->epsilon));
            
            // Update and clip Hessian estimate
            net->h1[i][j] = fmax(net->rho, 
                                net->beta2 * net->h1[i][j] + 
                                (1.0 - net->beta2) * hessian_approx);
            
            // Store current gradient for next iteration
            net->prev_dW1[i][j] = net->dW1[i][j];
            
            // Update weights using Sophia rule
            net->W1[i][j] -= net->lr * net->m1[i][j] / sqrt(net->h1[i][j] + net->epsilon);
        }
    }

    // Update second layer weights using Sophia
    for (int i = 0; i < OUTPUT_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            // Update momentum
            net->m2[i][j] = net->beta1 * net->m2[i][j] + (1.0 - net->beta1) * net->dW2[i][j];
            
            // Approximate second derivative using consecutive gradients
            float grad_diff = net->dW2[i][j] - net->prev_dW2[i][j];
            float hessian_approx = fabs(grad_diff / (net->lr + net->epsilon));
            
            // Update and clip Hessian estimate
            net->h2[i][j] = fmax(net->rho, 
                                net->beta2 * net->h2[i][j] + 
                                (1.0 - net->beta2) * hessian_approx);
            
            // Store current gradient for next iteration
            net->prev_dW2[i][j] = net->dW2[i][j];
            
            // Update weights using Sophia rule
            net->W2[i][j] -= net->lr * net->m2[i][j] / sqrt(net->h2[i][j] + net->epsilon);
        }
    }
}

bool save_net(const char* filename, const Net* net) {
    FILE* file = fopen(filename, "wb");
    if (!file) return false;
    
    // Save parameters
    fwrite(&net->lr, sizeof(float), 1, file);
    fwrite(&net->beta1, sizeof(float), 1, file);
    fwrite(&net->beta2, sizeof(float), 1, file);
    fwrite(&net->rho, sizeof(float), 1, file);
    fwrite(&net->epsilon, sizeof(float), 1, file);
    fwrite(&net->t, sizeof(unsigned long long), 1, file);
    
    // Save weights and Sophia states
    fwrite(net->W1, sizeof(net->W1), 1, file);
    fwrite(net->W2, sizeof(net->W2), 1, file);
    fwrite(net->m1, sizeof(net->m1), 1, file);
    fwrite(net->m2, sizeof(net->m2), 1, file);
    fwrite(net->h1, sizeof(net->h1), 1, file);
    fwrite(net->h2, sizeof(net->h2), 1, file);
    fwrite(net->prev_dW1, sizeof(net->prev_dW1), 1, file);
    fwrite(net->prev_dW2, sizeof(net->prev_dW2), 1, file);
    
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
    
    // Load parameters
    if (fread(&net->lr, sizeof(float), 1, file) != 1 ||
        fread(&net->beta1, sizeof(float), 1, file) != 1 ||
        fread(&net->beta2, sizeof(float), 1, file) != 1 ||
        fread(&net->rho, sizeof(float), 1, file) != 1 ||
        fread(&net->epsilon, sizeof(float), 1, file) != 1 ||
        fread(&net->t, sizeof(unsigned long long), 1, file) != 1 ||
        
        // Load weights and Sophia states
        fread(net->W1, sizeof(net->W1), 1, file) != 1 ||
        fread(net->W2, sizeof(net->W2), 1, file) != 1 ||
        fread(net->m1, sizeof(net->m1), 1, file) != 1 ||
        fread(net->m2, sizeof(net->m2), 1, file) != 1 ||
        fread(net->h1, sizeof(net->h1), 1, file) != 1 ||
        fread(net->h2, sizeof(net->h2), 1, file) != 1 ||
        fread(net->prev_dW1, sizeof(net->prev_dW1), 1, file) != 1 ||
        fread(net->prev_dW2, sizeof(net->prev_dW2), 1, file) != 1) {
        
        free(net);
        fclose(file);
        return NULL;
    }
    
    fclose(file);
    return net;
}

#endif // NET_H