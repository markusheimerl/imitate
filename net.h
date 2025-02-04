#ifndef NET_H
#define NET_H

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define HISTORY_LENGTH 3
#define STATE_DIM 6     // 3 accel + 3 gyro
#define INPUT_DIM (STATE_DIM * HISTORY_LENGTH)  // 6 states * 16 history length
#define HIDDEN_DIM 64
#define OUTPUT_DIM 8    // 4 means + 4 stds

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

__device__ __host__ inline double gelu(double x) {
    return 0.5 * x * (1 + tanh(sqrt(2/M_PI) * (x + 0.044715 * pow(x, 3))));
}

__device__ __host__ inline double gelu_derivative(double x) {
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

__global__ void backward_kernel(
    double* d_W1, double* d_W2,
    const double* d_output_gradients, 
    const double* d_stored_inputs,
    const double* d_stored_hidden,
    int num_steps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread handles one weight update
    if (idx < HIDDEN_DIM * INPUT_DIM) {
        int i = idx / INPUT_DIM;
        int j = idx % INPUT_DIM;
        double grad = 0.0;
        
        for(int step = 0; step < num_steps; step++) {
            double delta = 0.0;
            const double* output_gradient = &d_output_gradients[step * OUTPUT_DIM];
            const double* step_hidden = &d_stored_hidden[step * HIDDEN_DIM];
            const double* step_input = &d_stored_inputs[step * INPUT_DIM];
            
            // Compute hidden layer delta
            for (int k = 0; k < OUTPUT_DIM; k++) {
                delta += output_gradient[k] * d_W2[k * HIDDEN_DIM + i];
            }
            delta *= gelu_derivative(step_hidden[i]);
            
            grad += delta * step_input[j];
        }
        d_W1[idx] = grad;
    }
    else if (idx < (HIDDEN_DIM * INPUT_DIM + OUTPUT_DIM * HIDDEN_DIM)) {
        // Handle W2 gradients
        idx -= HIDDEN_DIM * INPUT_DIM;
        int i = idx / HIDDEN_DIM;
        int j = idx % HIDDEN_DIM;
        double grad = 0.0;
        
        for(int step = 0; step < num_steps; step++) {
            const double* output_gradient = &d_output_gradients[step * OUTPUT_DIM];
            const double* step_hidden = &d_stored_hidden[step * HIDDEN_DIM];
            grad += output_gradient[i] * step_hidden[j];
        }
        d_W2[idx] = grad;
    }
}

void backward_net(Net* net, const double* output_gradients, 
                 const double* stored_inputs, const double* stored_hidden,
                 int num_steps) 
{
    // Allocate device memory
    double *d_W1, *d_W2;
    double *d_output_gradients, *d_stored_inputs, *d_stored_hidden;
    
    cudaMalloc(&d_W1, HIDDEN_DIM * INPUT_DIM * sizeof(double));
    cudaMalloc(&d_W2, OUTPUT_DIM * HIDDEN_DIM * sizeof(double));
    cudaMalloc(&d_output_gradients, num_steps * OUTPUT_DIM * sizeof(double));
    cudaMalloc(&d_stored_inputs, num_steps * INPUT_DIM * sizeof(double));
    cudaMalloc(&d_stored_hidden, num_steps * HIDDEN_DIM * sizeof(double));
    
    // Copy data to device
    cudaMemcpy(d_output_gradients, output_gradients, 
               num_steps * OUTPUT_DIM * sizeof(double), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_stored_inputs, stored_inputs,
               num_steps * INPUT_DIM * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_stored_hidden, stored_hidden,
               num_steps * HIDDEN_DIM * sizeof(double),
               cudaMemcpyHostToDevice);
    
    // Launch kernel
    int total_weights = HIDDEN_DIM * INPUT_DIM + OUTPUT_DIM * HIDDEN_DIM;
    int block_size = 256;
    int num_blocks = (total_weights + block_size - 1) / block_size;
    
    backward_kernel<<<num_blocks, block_size>>>(
        d_W1, d_W2,
        d_output_gradients,
        d_stored_inputs,
        d_stored_hidden,
        num_steps
    );
    
    // Copy results back
    cudaMemcpy(net->dW1, d_W1, 
               HIDDEN_DIM * INPUT_DIM * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(net->dW2, d_W2,
               OUTPUT_DIM * HIDDEN_DIM * sizeof(double), 
               cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_output_gradients);
    cudaFree(d_stored_inputs);
    cudaFree(d_stored_hidden);
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