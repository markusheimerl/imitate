#ifndef NET_H
#define NET_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

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
    double** weight_gradients;  // Weight gradients
    double** bias_gradients;    // Bias gradients
    int num_layers;     // Total number of layers
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

// Net initialization
Net* create_net(int num_layers, const int* layer_sizes, double learning_rate) {
    Net* net = (Net*)malloc(sizeof(Net));
    if (!net) return NULL;

    net->num_layers = num_layers;
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

    // Initialize weights and related matrices
    for (int i = 0; i < num_layers - 1; i++) {
        int rows = layer_sizes[i + 1];
        int cols = layer_sizes[i];
        double scale = sqrt(2.0 / (layer_sizes[i] + layer_sizes[i + 1]));  // Xavier initialization

        net->weights[i] = (double*)malloc(rows * cols * sizeof(double));
        net->biases[i] = (double*)calloc(rows, sizeof(double));
        net->weight_gradients[i] = (double*)calloc(rows * cols, sizeof(double));
        net->bias_gradients[i] = (double*)calloc(rows, sizeof(double));

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

// Compute gradients through backpropagation
void backward_net(Net* net, const double* output_gradient) {
    int last_layer = net->num_layers - 1;
    memcpy(net->layers[last_layer].gradients, output_gradient, 
           net->layers[last_layer].size * sizeof(double));
    
    // Backward propagation through layers
    for (int i = last_layer - 1; i >= 0; i--) {
        Layer* current = &net->layers[i];
        Layer* next = &net->layers[i + 1];
        
        // Accumulate weight and bias gradients
        for (int j = 0; j < next->size; j++) {
            for (int k = 0; k < current->size; k++) {
                int idx = j * current->size + k;
                net->weight_gradients[i][idx] += next->gradients[j] * current->values[k];
            }
            net->bias_gradients[i][j] += next->gradients[j];
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
}

void zero_gradients(Net* net) {
    for (int i = 0; i < net->num_layers - 1; i++) {
        Layer* current = &net->layers[i];
        Layer* next = &net->layers[i + 1];
        
        int weight_size = next->size * current->size;
        memset(net->weight_gradients[i], 0, weight_size * sizeof(double));
        memset(net->bias_gradients[i], 0, next->size * sizeof(double));
    }
}

// Apply SGD update using computed gradients
void update_net(Net* net) {
    for (int i = 0; i < net->num_layers - 1; i++) {
        Layer* current = &net->layers[i];
        Layer* next = &net->layers[i + 1];
        
        // Update weights and biases using gradients
        for (int j = 0; j < next->size; j++) {
            for (int k = 0; k < current->size; k++) {
                int idx = j * current->size + k;
                net->weights[i][idx] -= net->learning_rate * net->weight_gradients[i][idx];
            }
            net->biases[i][j] -= net->learning_rate * net->bias_gradients[i][j];
        }
    }
}

// Save network to file
bool save_net(const char* filename, const Net* net) {
    FILE* file = fopen(filename, "wb");
    if (!file) return false;
    
    fwrite(&net->num_layers, sizeof(int), 1, file);
    fwrite(&net->learning_rate, sizeof(double), 1, file);
    
    for (int i = 0; i < net->num_layers; i++) {
        fwrite(&net->layers[i].size, sizeof(int), 1, file);
    }
    
    for (int i = 0; i < net->num_layers - 1; i++) {
        int rows = net->layers[i + 1].size;
        int cols = net->layers[i].size;
        fwrite(net->weights[i], sizeof(double), rows * cols, file);
        fwrite(net->biases[i], sizeof(double), rows, file);
    }
    
    fclose(file);
    return true;
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
    }
    
    free(net->weights);
    free(net->biases);
    free(net->weight_gradients);
    free(net->bias_gradients);
    free(net->layers);
    free(net);
}

// Load network from file
Net* load_net(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) return NULL;
    
    int num_layers;
    double learning_rate;
    
    if (fread(&num_layers, sizeof(int), 1, file) != 1 ||
        fread(&learning_rate, sizeof(double), 1, file) != 1) {
        fclose(file);
        return NULL;
    }
    
    int* layer_sizes = (int*)malloc(num_layers * sizeof(int));
    for (int i = 0; i < num_layers; i++) {
        if (fread(&layer_sizes[i], sizeof(int), 1, file) != 1) {
            free(layer_sizes);
            fclose(file);
            return NULL;
        }
    }
    
    Net* net = create_net(num_layers, layer_sizes, learning_rate);
    if (!net) {
        free(layer_sizes);
        fclose(file);
        return NULL;
    }
    
    for (int i = 0; i < num_layers - 1; i++) {
        int rows = net->layers[i + 1].size;
        int cols = net->layers[i].size;
        
        if (fread(net->weights[i], sizeof(double), rows * cols, file) != (size_t)(rows * cols) ||
            fread(net->biases[i], sizeof(double), rows, file) != (size_t)rows) {
            free_net(net);
            free(layer_sizes);
            fclose(file);
            return NULL;
        }
    }
    
    free(layer_sizes);
    fclose(file);
    return net;
}

#endif // NET_H