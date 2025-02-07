#ifndef NET_H
#define NET_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define DT_PHYSICS (1.0/1000.0)
#define DT_CONTROL (1.0/60.0)
#define DT_RENDER (1.0/24.0)

#define STATE_DIM 15
#define ACTION_DIM 8
#define MAX_STEPS 1000
#define NUM_ROLLOUTS 128

#define GAMMA 0.999
#define MAX_STD 3.0
#define MIN_STD 1e-5

#define MAX_MEAN (OMEGA_MAX - 4.0 * MAX_STD)
#define MIN_MEAN (OMEGA_MIN + 4.0 * MAX_STD)

#define MIN_DISTANCE 0.01
#define MAX_DISTANCE 0.5

typedef struct {
    // Network architecture
    int n_layers;
    int* sizes;  // Array of layer sizes
    
    // Network state
    double** x;  // Layer activations
    double** dx; // Layer gradients
    
    // Weights and biases
    double** W;  // Weights between layers
    double** b;  // Biases for each layer
    
    // Gradients
    double** dW; // Weight gradients
    double** db; // Bias gradients
    
    // AdamW state
    double** m_W;  // First moment for weights
    double** v_W;  // Second moment for weights
    double** m_b;  // First moment for biases
    double** v_b;  // Second moment for biases
    
    // Optimizer state
    int t;          // Timestep for AdamW
    double lr;      // Learning rate
} Net;

static double gelu(double x) {
    return 0.5 * x * (1 + tanh(sqrt(2/M_PI) * (x + 0.044715 * pow(x, 3))));
}

static double gelu_derivative(double x) {
    double cdf = 0.5 * (1 + tanh(sqrt(2/M_PI) * (x + 0.044715 * pow(x, 3))));
    double pdf = exp(-0.5 * x * x) / sqrt(2 * M_PI);
    return cdf + x * pdf;
}

static double get_learning_rate(int epoch, int total_epochs, double initial_lr) {
    return initial_lr * (1 + cos(M_PI * epoch / total_epochs)) / 2;
}

static double get_weight_decay(int epoch, int total_epochs) {
    return 0.01 * (1 - epoch / (double)total_epochs);
}

static double get_warmup_lr(int epoch, int warmup_epochs, double initial_lr) {
    if (epoch < warmup_epochs) {
        return initial_lr * epoch / warmup_epochs;
    }
    return initial_lr;
}

Net* init_net(int n_layers, int* sizes, double lr) {
    Net* net = (Net*)malloc(sizeof(Net));
    net->n_layers = n_layers;
    net->sizes = (int*)malloc(n_layers * sizeof(int));
    memcpy(net->sizes, sizes, n_layers * sizeof(int));
    
    net->t = 1;
    net->lr = lr;
    
    // Allocate arrays for layer states
    net->x = (double**)malloc(n_layers * sizeof(double*));
    net->dx = (double**)malloc(n_layers * sizeof(double*));
    for(int i = 0; i < n_layers; i++) {
        net->x[i] = (double*)calloc(sizes[i], sizeof(double));
        net->dx[i] = (double*)calloc(sizes[i], sizeof(double));
    }
    
    // Allocate arrays for weights, biases, and their gradients
    net->W = (double**)malloc((n_layers-1) * sizeof(double*));
    net->b = (double**)malloc((n_layers-1) * sizeof(double*));
    net->dW = (double**)malloc((n_layers-1) * sizeof(double*));
    net->db = (double**)malloc((n_layers-1) * sizeof(double*));
    net->m_W = (double**)malloc((n_layers-1) * sizeof(double*));
    net->v_W = (double**)malloc((n_layers-1) * sizeof(double*));
    net->m_b = (double**)malloc((n_layers-1) * sizeof(double*));
    net->v_b = (double**)malloc((n_layers-1) * sizeof(double*));
    
    for(int i = 0; i < n_layers-1; i++) {
        int rows = sizes[i+1];
        int cols = sizes[i];
        
        net->W[i] = (double*)malloc(rows * cols * sizeof(double));
        net->b[i] = (double*)calloc(rows, sizeof(double));
        net->dW[i] = (double*)calloc(rows * cols, sizeof(double));
        net->db[i] = (double*)calloc(rows, sizeof(double));
        net->m_W[i] = (double*)calloc(rows * cols, sizeof(double));
        net->v_W[i] = (double*)calloc(rows * cols, sizeof(double));
        net->m_b[i] = (double*)calloc(rows, sizeof(double));
        net->v_b[i] = (double*)calloc(rows, sizeof(double));
        
        // Xavier initialization
        double scale = sqrt(2.0 / (sizes[i] + sizes[i+1]));
        for(int j = 0; j < rows * cols; j++) {
            net->W[i][j] = ((double)rand()/RAND_MAX * 2 - 1) * scale;
        }
    }
    
    return net;
}

void zero_gradients(Net* net) {
    for(int i = 0; i < net->n_layers-1; i++) {
        int rows = net->sizes[i+1];
        int cols = net->sizes[i];
        memset(net->dW[i], 0, rows * cols * sizeof(double));
        memset(net->db[i], 0, rows * sizeof(double));
    }
    for(int i = 0; i < net->n_layers; i++) {
        memset(net->dx[i], 0, net->sizes[i] * sizeof(double));
    }
}

void forward_net(Net* net, double* input) {
    // Copy input
    memcpy(net->x[0], input, net->sizes[0] * sizeof(double));
    
    // Forward propagation
    for(int i = 0; i < net->n_layers-1; i++) {
        int curr_size = net->sizes[i];
        int next_size = net->sizes[i+1];
        
        // Reset next layer
        memset(net->x[i+1], 0, next_size * sizeof(double));
        
        // Compute weighted sum
        for(int j = 0; j < next_size; j++) {
            for(int k = 0; k < curr_size; k++) {
                net->x[i+1][j] += net->x[i][k] * net->W[i][j * curr_size + k];
            }
            net->x[i+1][j] += net->b[i][j];
            
            // Apply GELU activation (except for output layer)
            if(i < net->n_layers-2) {
                net->x[i+1][j] = gelu(net->x[i+1][j]);
            }
        }
    }
}

void backward_net(Net* net, double* output_gradient, int epoch, int total_epochs) {
    int last = net->n_layers-1;
    
    // Set output layer gradient
    memcpy(net->dx[last], output_gradient, net->sizes[last] * sizeof(double));
    
    // Backward propagation
    for(int i = last-1; i >= 0; i--) {
        int curr_size = net->sizes[i];
        int next_size = net->sizes[i+1];
        
        // Compute gradients for weights and biases
        for(int j = 0; j < next_size; j++) {
            for(int k = 0; k < curr_size; k++) {
                net->dW[i][j * curr_size + k] = net->dx[i+1][j] * net->x[i][k];
            }
            net->db[i][j] = net->dx[i+1][j];
        }
        
        // Compute gradients for current layer
        if(i > 0) {
            memset(net->dx[i], 0, curr_size * sizeof(double));
            for(int j = 0; j < curr_size; j++) {
                for(int k = 0; k < next_size; k++) {
                    net->dx[i][j] += net->dx[i+1][k] * net->W[i][k * curr_size + j];
                }
                net->dx[i][j] *= gelu_derivative(net->x[i][j]);
            }
        }
    }
    
}

void update_net(Net* net, int epoch, int total_epochs) {
    // Update weights and biases using AdamW
    const double beta1 = 0.9;
    const double beta2 = 0.999;
    const double eps = 1e-8;
    
    const int warmup_epochs = 5;
    double current_lr = get_warmup_lr(epoch, warmup_epochs, net->lr);
    current_lr = get_learning_rate(epoch, total_epochs, current_lr);
    double weight_decay = get_weight_decay(epoch, total_epochs);
    
    for(int i = 0; i < net->n_layers-1; i++) {
        int curr_size = net->sizes[i];
        int next_size = net->sizes[i+1];
        int w_size = next_size * curr_size;
        
        // Update weights
        for(int j = 0; j < w_size; j++) {
            net->m_W[i][j] = beta1 * net->m_W[i][j] + (1 - beta1) * net->dW[i][j];
            net->v_W[i][j] = beta2 * net->v_W[i][j] + (1 - beta2) * net->dW[i][j] * net->dW[i][j];
            
            double m_hat = net->m_W[i][j] / (1 - pow(beta1, net->t));
            double v_hat = net->v_W[i][j] / (1 - pow(beta2, net->t));
            
            net->W[i][j] -= current_lr * (m_hat / (sqrt(v_hat) + eps) + 
                                        weight_decay * net->W[i][j]);
        }
        
        // Update biases
        for(int j = 0; j < next_size; j++) {
            net->m_b[i][j] = beta1 * net->m_b[i][j] + (1 - beta1) * net->db[i][j];
            net->v_b[i][j] = beta2 * net->v_b[i][j] + (1 - beta2) * net->db[i][j] * net->db[i][j];
            
            double m_hat = net->m_b[i][j] / (1 - pow(beta1, net->t));
            double v_hat = net->v_b[i][j] / (1 - pow(beta2, net->t));
            
            net->b[i][j] -= current_lr * m_hat / (sqrt(v_hat) + eps);
        }
    }
    
    net->t++;
}

void save_net(const char* filename, Net* net) {
    FILE* fp = fopen(filename, "wb");
    if(!fp) return;
    
    fwrite(&net->n_layers, sizeof(int), 1, fp);
    fwrite(&net->lr, sizeof(double), 1, fp);
    fwrite(&net->t, sizeof(int), 1, fp);
    
    for(int i = 0; i < net->n_layers; i++) {
        fwrite(&net->sizes[i], sizeof(int), 1, fp);
    }
    
    for(int i = 0; i < net->n_layers-1; i++) {
        int rows = net->sizes[i+1];
        int cols = net->sizes[i];
        int w_size = rows * cols;
        
        fwrite(net->W[i], sizeof(double), w_size, fp);
        fwrite(net->m_W[i], sizeof(double), w_size, fp);
        fwrite(net->v_W[i], sizeof(double), w_size, fp);
        fwrite(net->b[i], sizeof(double), rows, fp);
        fwrite(net->m_b[i], sizeof(double), rows, fp);
        fwrite(net->v_b[i], sizeof(double), rows, fp);
    }
    
    fclose(fp);
}

Net* load_net(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if(!fp) return NULL;
    
    int n_layers;
    double learning_rate;
    int timestep;
    
    fread(&n_layers, sizeof(int), 1, fp);
    fread(&learning_rate, sizeof(double), 1, fp);
    fread(&timestep, sizeof(int), 1, fp);
    
    int* sizes = (int*)malloc(n_layers * sizeof(int));
    for(int i = 0; i < n_layers; i++) {
        fread(&sizes[i], sizeof(int), 1, fp);
    }
    
    Net* net = init_net(n_layers, sizes, learning_rate);
    net->t = timestep;
    
    for(int i = 0; i < n_layers-1; i++) {
        int rows = sizes[i+1];
        int cols = sizes[i];
        int w_size = rows * cols;
        
        fread(net->W[i], sizeof(double), w_size, fp);
        fread(net->m_W[i], sizeof(double), w_size, fp);
        fread(net->v_W[i], sizeof(double), w_size, fp);
        fread(net->b[i], sizeof(double), rows, fp);
        fread(net->m_b[i], sizeof(double), rows, fp);
        fread(net->v_b[i], sizeof(double), rows, fp);
    }
    
    free(sizes);
    fclose(fp);
    return net;
}

void free_net(Net* net) {
    for(int i = 0; i < net->n_layers; i++) {
        free(net->x[i]);
        free(net->dx[i]);
    }
    for(int i = 0; i < net->n_layers-1; i++) {
        free(net->W[i]);
        free(net->b[i]);
        free(net->dW[i]);
        free(net->db[i]);
        free(net->m_W[i]);
        free(net->v_W[i]);
        free(net->m_b[i]);
        free(net->v_b[i]);
    }
    
    free(net->x);
    free(net->dx);
    free(net->W);
    free(net->b);
    free(net->dW);
    free(net->db);
    free(net->m_W);
    free(net->v_W);
    free(net->m_b);
    free(net->v_b);
    free(net->sizes);
    free(net);
}

#endif // NET_H