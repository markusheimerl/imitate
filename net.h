#ifndef NET_H
#define NET_H

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
    double *x;      // layer values
    double *dx;     // gradients
    int size;       // size of this layer
} Layer;

typedef struct {
    Layer *layers;  // array of layers
    double **W;     // weights between layers
    double **dW;    // weight gradients
    double **b;     // biases for each layer
    double **db;    // bias gradients
    double **m_W;   // momentum for weights (AdamW)
    double **v_W;   // velocity for weights (AdamW)
    double **m_b;   // momentum for biases (AdamW)
    double **v_b;   // velocity for biases (AdamW)
    int n_layers;   // number of layers
    int t;          // timestep for AdamW
    double lr;      // initial learning rate
} Net;

// Initialize network with given architecture
Net* init_net(int n_layers, int* sizes, double lr) {
    Net* net = (Net*)malloc(sizeof(Net));
    net->n_layers = n_layers;
    net->t = 1;  // Initialize timestep
    net->lr = lr;  // Set initial learning rate
    net->layers = (Layer*)malloc(n_layers * sizeof(Layer));
    
    // Initialize layers
    for(int i = 0; i < n_layers; i++) {
        net->layers[i].size = sizes[i];
        net->layers[i].x = (double*)calloc(sizes[i], sizeof(double));
        net->layers[i].dx = (double*)calloc(sizes[i], sizeof(double));
    }
    
    // Initialize weights and biases
    net->W = (double**)malloc((n_layers-1) * sizeof(double*));
    net->dW = (double**)malloc((n_layers-1) * sizeof(double*));
    net->b = (double**)malloc((n_layers-1) * sizeof(double*));
    net->db = (double**)malloc((n_layers-1) * sizeof(double*));
    net->m_W = (double**)malloc((n_layers-1) * sizeof(double*));
    net->v_W = (double**)malloc((n_layers-1) * sizeof(double*));
    net->m_b = (double**)malloc((n_layers-1) * sizeof(double*));
    net->v_b = (double**)malloc((n_layers-1) * sizeof(double*));
    
    for(int i = 0; i < n_layers-1; i++) {
        int rows = sizes[i+1];
        int cols = sizes[i];
        net->W[i] = (double*)malloc(rows * cols * sizeof(double));
        net->dW[i] = (double*)calloc(rows * cols, sizeof(double));
        net->b[i] = (double*)malloc(rows * sizeof(double));
        net->db[i] = (double*)calloc(rows, sizeof(double));
        net->m_W[i] = (double*)calloc(rows * cols, sizeof(double));
        net->v_W[i] = (double*)calloc(rows * cols, sizeof(double));
        net->m_b[i] = (double*)calloc(rows, sizeof(double));
        net->v_b[i] = (double*)calloc(rows, sizeof(double));
        
        // Xavier initialization
        double scale = sqrt(2.0 / (sizes[i] + sizes[i+1]));
        for(int j = 0; j < rows * cols; j++)
            net->W[i][j] = ((double)rand()/RAND_MAX * 2 - 1) * scale;
        for(int j = 0; j < rows; j++)
            net->b[i][j] = 0;
    }
    return net;
}

double gelu(double x) {
    return 0.5 * x * (1 + tanh(sqrt(2/M_PI) * (x + 0.044715 * pow(x, 3))));
}

double gelu_derivative(double x) {
    double cdf = 0.5 * (1 + tanh(sqrt(2/M_PI) * (x + 0.044715 * pow(x, 3))));
    double pdf = exp(-0.5 * x * x) / sqrt(2 * M_PI);
    return cdf + x * pdf;
}

// Forward pass
void forward(Net* net, double* input) {
    // Set input layer
    memcpy(net->layers[0].x, input, net->layers[0].size * sizeof(double));
    
    // Forward propagation
    for(int i = 0; i < net->n_layers-1; i++) {
        Layer *curr = &net->layers[i];
        Layer *next = &net->layers[i+1];
        
        // Reset next layer
        memset(next->x, 0, next->size * sizeof(double));
        
        // Compute weighted sum
        for(int j = 0; j < next->size; j++) {
            for(int k = 0; k < curr->size; k++) {
                next->x[j] += curr->x[k] * net->W[i][j * curr->size + k];
            }
            next->x[j] += net->b[i][j];
            // Apply gelu activation (except for output layer)
            if(i < net->n_layers-2) {
                next->x[j] = gelu(next->x[j]);
            }
        }
    }
}

double get_learning_rate(int epoch, int total_epochs, double initial_lr) {
    return initial_lr * (1 + cos(M_PI * epoch / total_epochs)) / 2;
}

double get_weight_decay(int epoch, int total_epochs) {
    return 0.01 * (1 - epoch / (double)total_epochs);
}

double get_warmup_lr(int epoch, int warmup_epochs, double initial_lr) {
    if (epoch < warmup_epochs) {
        return initial_lr * epoch / warmup_epochs;
    }
    return initial_lr;
}

// AdamW update function with learning rate scheduling
void adamw_update(Net* net, int epoch, int total_epochs) {
    const double beta1 = 0.9;    // Momentum factor
    const double beta2 = 0.999;  // Velocity factor
    const double eps = 1e-8;     // Small constant for numerical stability
    
    // Adaptive learning rate and weight decay scheduling
    const int warmup_epochs = 5;
    double current_lr = get_warmup_lr(epoch, warmup_epochs, net->lr);
    current_lr = get_learning_rate(epoch, total_epochs, current_lr);
    double weight_decay = get_weight_decay(epoch, total_epochs);
    
    for(int i = 0; i < net->n_layers-1; i++) {
        Layer *curr = &net->layers[i];
        Layer *next = &net->layers[i+1];
        int w_size = next->size * curr->size;
        
        // Update weights
        for(int j = 0; j < w_size; j++) {
            // Momentum update
            net->m_W[i][j] = beta1 * net->m_W[i][j] + (1 - beta1) * net->dW[i][j];
            // Velocity update
            net->v_W[i][j] = beta2 * net->v_W[i][j] + (1 - beta2) * net->dW[i][j] * net->dW[i][j];
            
            // Bias correction
            double m_hat = net->m_W[i][j] / (1 - pow(beta1, net->t));
            double v_hat = net->v_W[i][j] / (1 - pow(beta2, net->t));
            
            // AdamW update (including weight decay)
            net->W[i][j] -= current_lr * (m_hat / (sqrt(v_hat) + eps) + 
                                        weight_decay * net->W[i][j]);
        }
        
        // Update biases
        for(int j = 0; j < next->size; j++) {
            net->m_b[i][j] = beta1 * net->m_b[i][j] + (1 - beta1) * net->db[i][j];
            net->v_b[i][j] = beta2 * net->v_b[i][j] + (1 - beta2) * net->db[i][j] * net->db[i][j];
            
            double m_hat = net->m_b[i][j] / (1 - pow(beta1, net->t));
            double v_hat = net->v_b[i][j] / (1 - pow(beta2, net->t));
            
            net->b[i][j] -= current_lr * m_hat / (sqrt(v_hat) + eps);
        }
    }
    net->t++;  // Increment time step
}

// Backward pass
void bwd(Net* net, double* output_gradient, int epoch, int total_epochs) {
    int last = net->n_layers-1;
    
    // Set output layer gradient
    memcpy(net->layers[last].dx, output_gradient, net->layers[last].size * sizeof(double));
    
    // Backward propagation
    for(int i = last-1; i >= 0; i--) {
        Layer *curr = &net->layers[i];
        Layer *next = &net->layers[i+1];
        
        // Compute gradients for weights and biases
        for(int j = 0; j < next->size; j++) {
            for(int k = 0; k < curr->size; k++) {
                net->dW[i][j * curr->size + k] = next->dx[j] * curr->x[k];
            }
            net->db[i][j] = next->dx[j];
        }
        
        // Compute gradients for current layer
        if(i > 0) {
            memset(curr->dx, 0, curr->size * sizeof(double));
            for(int j = 0; j < curr->size; j++) {
                for(int k = 0; k < next->size; k++) {
                    curr->dx[j] += next->dx[k] * net->W[i][k * curr->size + j];
                }
                curr->dx[j] *= gelu_derivative(curr->x[j]);
            }
        }
    }
    
    // Update weights and biases using AdamW
    adamw_update(net, epoch, total_epochs);
}

void backward(Net* net, double* target, int epoch, int total_epochs) {
    int last = net->n_layers-1;
    double output_gradient[net->layers[last].size];
    
    for(int i = 0; i < net->layers[last].size; i++) {
        output_gradient[i] = net->layers[last].x[i] - target[i];
    }
    
    bwd(net, output_gradient, epoch, total_epochs);
}

void save_net(const char* f, Net* net) {
    FILE* fp = fopen(f, "wb");
    if(!fp) return;
    
    // Save network architecture, learning rate, and timestep
    fwrite(&net->n_layers, sizeof(int), 1, fp);
    fwrite(&net->lr, sizeof(double), 1, fp);
    fwrite(&net->t, sizeof(int), 1, fp);
    
    // Save layer sizes
    for(int i = 0; i < net->n_layers; i++) {
        fwrite(&net->layers[i].size, sizeof(int), 1, fp);
    }
    
    // Save weights, biases, and optimizer states
    for(int i = 0; i < net->n_layers-1; i++) {
        int rows = net->layers[i+1].size;
        int cols = net->layers[i].size;
        int w_size = rows * cols;
        
        // Save weights and related optimizer states
        fwrite(net->W[i], sizeof(double), w_size, fp);
        fwrite(net->m_W[i], sizeof(double), w_size, fp);
        fwrite(net->v_W[i], sizeof(double), w_size, fp);
        
        // Save biases and related optimizer states
        fwrite(net->b[i], sizeof(double), rows, fp);
        fwrite(net->m_b[i], sizeof(double), rows, fp);
        fwrite(net->v_b[i], sizeof(double), rows, fp);
    }
    
    fclose(fp);
}

Net* load_net(const char* f) {
    FILE* fp = fopen(f, "rb");
    if(!fp) return NULL;
    
    // Load network architecture and learning rate
    int n_layers;
    double learning_rate;
    int timestep;
    fread(&n_layers, sizeof(int), 1, fp);
    fread(&learning_rate, sizeof(double), 1, fp);
    fread(&timestep, sizeof(int), 1, fp);  // Load timestep
    
    // Load layer sizes
    int* sizes = (int*)malloc(n_layers * sizeof(int));
    for(int i = 0; i < n_layers; i++) {
        fread(&sizes[i], sizeof(int), 1, fp);
    }
    
    // Initialize network
    Net* net = init_net(n_layers, sizes, learning_rate);
    net->t = timestep;  // Restore timestep
    free(sizes);
    
    // Load weights, biases, and optimizer states
    for(int i = 0; i < n_layers-1; i++) {
        int rows = net->layers[i+1].size;
        int cols = net->layers[i].size;
        int w_size = rows * cols;
        
        // Load weights and related optimizer states
        fread(net->W[i], sizeof(double), w_size, fp);
        fread(net->m_W[i], sizeof(double), w_size, fp);
        fread(net->v_W[i], sizeof(double), w_size, fp);
        
        // Load biases and related optimizer states
        fread(net->b[i], sizeof(double), rows, fp);
        fread(net->m_b[i], sizeof(double), rows, fp);
        fread(net->v_b[i], sizeof(double), rows, fp);
    }
    
    fclose(fp);
    return net;
}

void free_net(Net* net) {
    for(int i = 0; i < net->n_layers; i++) {
        free(net->layers[i].x);
        free(net->layers[i].dx);
    }
    for(int i = 0; i < net->n_layers-1; i++) {
        free(net->W[i]);
        free(net->dW[i]);
        free(net->b[i]);
        free(net->db[i]);
        free(net->m_W[i]);
        free(net->v_W[i]);
        free(net->m_b[i]);
        free(net->v_b[i]);
    }
    free(net->W);
    free(net->dW);
    free(net->b);
    free(net->db);
    free(net->m_W);
    free(net->v_W);
    free(net->m_b);
    free(net->v_b);
    free(net->layers);
    free(net);
}

#endif // NET_H