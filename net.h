#ifndef NET_H
#define NET_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

// Constants for optimization
#define BETA1 0.9     // Momentum factor for AdamW
#define BETA2 0.999   // Velocity factor for AdamW
#define EPSILON 1e-8  // Small constant for numerical stability
#define WARMUP_EPOCHS 5

// Forward declarations
typedef struct Layer Layer;
typedef struct Net Net;

// Structure definitions
struct Layer {
    double* values;     // Layer activation values
    double* gradients;  // Layer gradients
    int size;          // Number of neurons
};

struct Net {
    Layer* layers;      // Array of layers
    double** weights;   // Weight matrices
    double** biases;    // Bias vectors
    
    // Gradient matrices
    double** weight_gradients;
    double** bias_gradients;
    
    // AdamW optimizer states
    double** weight_momentum;
    double** weight_velocity;
    double** bias_momentum;
    double** bias_velocity;
    
    int num_layers;     // Total number of layers
    int timestep;       // Current optimization step
    double learning_rate;
};

// Activation functions
static double gelu(double x) {
    return 0.5 * x * (1 + tanh(sqrt(2/M_PI) * (x + 0.044715 * pow(x, 3))));
}

static double gelu_derivative(double x) {
    double cdf = 0.5 * (1 + tanh(sqrt(2/M_PI) * (x + 0.044715 * pow(x, 3))));
    double pdf = exp(-0.5 * x * x) / sqrt(2 * M_PI);
    return cdf + x * pdf;
}

// Learning rate scheduling functions
static double compute_warmup_lr(int epoch, int warmup_epochs, double initial_lr) {
    return (epoch < warmup_epochs) ? initial_lr * epoch / warmup_epochs : initial_lr;
}

static double compute_cosine_lr(int epoch, int total_epochs, double initial_lr) {
    return initial_lr * (1 + cos(M_PI * epoch / total_epochs)) / 2;
}

static double compute_weight_decay(int epoch, int total_epochs) {
    return 0.01 * (1 - epoch / (double)total_epochs);
}

// Net initialization
Net* create_net(int num_layers, const int* layer_sizes, double learning_rate) {
    Net* net = (Net*)malloc(sizeof(Net));
    if (!net) return NULL;

    net->num_layers = num_layers;
    net->timestep = 1;
    net->learning_rate = learning_rate;
    
    // Allocate layers
    net->layers = (Layer*)malloc(num_layers * sizeof(Layer));
    if (!net->layers) {
        free(net);
        return NULL;
    }

    // Initialize layers
    for (int i = 0; i < num_layers; i++) {
        net->layers[i].size = layer_sizes[i];
        net->layers[i].values = (double*)calloc(layer_sizes[i], sizeof(double));
        net->layers[i].gradients = (double*)calloc(layer_sizes[i], sizeof(double));
    }

    // Allocate matrices
    net->weights = (double**)malloc((num_layers - 1) * sizeof(double*));
    net->biases = (double**)malloc((num_layers - 1) * sizeof(double*));
    net->weight_gradients = (double**)malloc((num_layers - 1) * sizeof(double*));
    net->bias_gradients = (double**)malloc((num_layers - 1) * sizeof(double*));
    net->weight_momentum = (double**)malloc((num_layers - 1) * sizeof(double*));
    net->weight_velocity = (double**)malloc((num_layers - 1) * sizeof(double*));
    net->bias_momentum = (double**)malloc((num_layers - 1) * sizeof(double*));
    net->bias_velocity = (double**)malloc((num_layers - 1) * sizeof(double*));

        // Initialize weights and related matrices
    for (int i = 0; i < num_layers - 1; i++) {
        int rows = layer_sizes[i + 1];
        int cols = layer_sizes[i];
        double scale = sqrt(2.0 / (layer_sizes[i] + layer_sizes[i + 1]));  // Xavier initialization

        net->weights[i] = (double*)malloc(rows * cols * sizeof(double));
        net->biases[i] = (double*)calloc(rows, sizeof(double));
        net->weight_gradients[i] = (double*)calloc(rows * cols, sizeof(double));
        net->bias_gradients[i] = (double*)calloc(rows, sizeof(double));
        net->weight_momentum[i] = (double*)calloc(rows * cols, sizeof(double));
        net->weight_velocity[i] = (double*)calloc(rows * cols, sizeof(double));
        net->bias_momentum[i] = (double*)calloc(rows, sizeof(double));
        net->bias_velocity[i] = (double*)calloc(rows, sizeof(double));

        // Initialize weights with Xavier initialization
        for (int j = 0; j < rows * cols; j++) {
            net->weights[i][j] = ((double)rand()/RAND_MAX * 2 - 1) * scale;
        }
    }

    return net;
}

// Forward propagation
void forward_net(Net* net, const double* input) {
    memcpy(net->layers[0].values, input, net->layers[0].size * sizeof(double));

    for (int i = 0; i < net->num_layers - 1; i++) {
        Layer* current = &net->layers[i];
        Layer* next = &net->layers[i + 1];
        memset(next->values, 0, next->size * sizeof(double));

        // Compute weighted sum and activation
        for (int j = 0; j < next->size; j++) {
            for (int k = 0; k < current->size; k++) {
                next->values[j] += current->values[k] * net->weights[i][j * current->size + k];
            }
            next->values[j] += net->biases[i][j];
            
            if (i < net->num_layers - 2) {
                next->values[j] = gelu(next->values[j]);
            }
        }
    }
}

// AdamW optimizer update
static void network_adamw_update(Net* net, int epoch, int total_epochs) {
    double current_lr = compute_warmup_lr(epoch, WARMUP_EPOCHS, net->learning_rate);
    current_lr = compute_cosine_lr(epoch, total_epochs, current_lr);
    double weight_decay = compute_weight_decay(epoch, total_epochs);
    
    for (int i = 0; i < net->num_layers - 1; i++) {
        Layer* current = &net->layers[i];
        Layer* next = &net->layers[i + 1];
        int weight_size = next->size * current->size;
        
        // Update weights
        for (int j = 0; j < weight_size; j++) {
            // Update momentum and velocity
            net->weight_momentum[i][j] = BETA1 * net->weight_momentum[i][j] + 
                                       (1 - BETA1) * net->weight_gradients[i][j];
            net->weight_velocity[i][j] = BETA2 * net->weight_velocity[i][j] + 
                                       (1 - BETA2) * net->weight_gradients[i][j] * net->weight_gradients[i][j];
            
            // Bias correction
            double m_hat = net->weight_momentum[i][j] / (1 - pow(BETA1, net->timestep));
            double v_hat = net->weight_velocity[i][j] / (1 - pow(BETA2, net->timestep));
            
            // AdamW update
            net->weights[i][j] -= current_lr * (m_hat / (sqrt(v_hat) + EPSILON) + 
                                              weight_decay * net->weights[i][j]);
        }
        
        // Update biases
        for (int j = 0; j < next->size; j++) {
            // Update momentum and velocity
            net->bias_momentum[i][j] = BETA1 * net->bias_momentum[i][j] + 
                                     (1 - BETA1) * net->bias_gradients[i][j];
            net->bias_velocity[i][j] = BETA2 * net->bias_velocity[i][j] + 
                                     (1 - BETA2) * net->bias_gradients[i][j] * net->bias_gradients[i][j];
            
            // Bias correction
            double m_hat = net->bias_momentum[i][j] / (1 - pow(BETA1, net->timestep));
            double v_hat = net->bias_velocity[i][j] / (1 - pow(BETA2, net->timestep));
            
            // Update bias
            net->biases[i][j] -= current_lr * m_hat / (sqrt(v_hat) + EPSILON);
        }
    }
    net->timestep++;
}

// Backward propagation
void backward_net(Net* net, const double* output_gradient, int epoch, int total_epochs) {
    int last_layer = net->num_layers - 1;
    memcpy(net->layers[last_layer].gradients, output_gradient, 
           net->layers[last_layer].size * sizeof(double));
    
    // Backward propagation through layers
    for (int i = last_layer - 1; i >= 0; i--) {
        Layer* current = &net->layers[i];
        Layer* next = &net->layers[i + 1];
        
        // Compute weight and bias gradients
        for (int j = 0; j < next->size; j++) {
            for (int k = 0; k < current->size; k++) {
                net->weight_gradients[i][j * current->size + k] = 
                    next->gradients[j] * current->values[k];
            }
            net->bias_gradients[i][j] = next->gradients[j];
        }
        
        // Compute gradients for current layer
        if (i > 0) {
            memset(current->gradients, 0, current->size * sizeof(double));
            for (int j = 0; j < current->size; j++) {
                for (int k = 0; k < next->size; k++) {
                    current->gradients[j] += next->gradients[k] * 
                        net->weights[i][k * current->size + j];
                }
                current->gradients[j] *= gelu_derivative(current->values[j]);
            }
        }
    }
    
    network_adamw_update(net, epoch, total_epochs);
}

// Save network to file
bool save_net(const char* filename, const Net* net) {
    FILE* file = fopen(filename, "wb");
    if (!file) return false;
    
    // Save network architecture and parameters
    fwrite(&net->num_layers, sizeof(int), 1, file);
    fwrite(&net->learning_rate, sizeof(double), 1, file);
    fwrite(&net->timestep, sizeof(int), 1, file);
    
    // Save layer sizes
    for (int i = 0; i < net->num_layers; i++) {
        fwrite(&net->layers[i].size, sizeof(int), 1, file);
    }
    
    // Save weights, biases, and optimizer states
    for (int i = 0; i < net->num_layers - 1; i++) {
        int rows = net->layers[i + 1].size;
        int cols = net->layers[i].size;
        int weight_size = rows * cols;
        
        fwrite(net->weights[i], sizeof(double), weight_size, file);
        fwrite(net->weight_momentum[i], sizeof(double), weight_size, file);
        fwrite(net->weight_velocity[i], sizeof(double), weight_size, file);
        
        fwrite(net->biases[i], sizeof(double), rows, file);
        fwrite(net->bias_momentum[i], sizeof(double), rows, file);
        fwrite(net->bias_velocity[i], sizeof(double), rows, file);
    }
    
    fclose(file);
    return true;
}

// Load network from file
Net* load_net(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) return NULL;
    
    // Load network architecture
    int num_layers;
    double learning_rate;
    int timestep;
    
    fread(&num_layers, sizeof(int), 1, file);
    fread(&learning_rate, sizeof(double), 1, file);
    fread(&timestep, sizeof(int), 1, file);
    
    // Load layer sizes
    int* layer_sizes = (int*)malloc(num_layers * sizeof(int));
    for (int i = 0; i < num_layers; i++) {
        fread(&layer_sizes[i], sizeof(int), 1, file);
    }
    
    // Create network
    Net* net = create_net(num_layers, layer_sizes, learning_rate);
    if (!net) {
        free(layer_sizes);
        fclose(file);
        return NULL;
    }
    
    net->timestep = timestep;
    
    // Load weights, biases, and optimizer states
    for (int i = 0; i < num_layers - 1; i++) {
        int rows = net->layers[i + 1].size;
        int cols = net->layers[i].size;
        int weight_size = rows * cols;
        
        fread(net->weights[i], sizeof(double), weight_size, file);
        fread(net->weight_momentum[i], sizeof(double), weight_size, file);
        fread(net->weight_velocity[i], sizeof(double), weight_size, file);
        
        fread(net->biases[i], sizeof(double), rows, file);
        fread(net->bias_momentum[i], sizeof(double), rows, file);
        fread(net->bias_velocity[i], sizeof(double), rows, file);
    }
    
    free(layer_sizes);
    fclose(file);
    return net;
}

// Free network resources
void free_net(Net* net) {
    if (!net) return;
    
    for (int i = 0; i < net->num_layers; i++) {
        free(net->layers[i].values);
        free(net->layers[i].gradients);
    }
    
    for (int i = 0; i < net->num_layers - 1; i++) {
        free(net->weights[i]);
        free(net->biases[i]);
        free(net->weight_gradients[i]);
        free(net->bias_gradients[i]);
        free(net->weight_momentum[i]);
        free(net->weight_velocity[i]);
        free(net->bias_momentum[i]);
        free(net->bias_velocity[i]);
    }
    
    free(net->weights);
    free(net->biases);
    free(net->weight_gradients);
    free(net->bias_gradients);
    free(net->weight_momentum);
    free(net->weight_velocity);
    free(net->bias_momentum);
    free(net->bias_velocity);
    free(net->layers);
    free(net);
}

#endif // NET_H