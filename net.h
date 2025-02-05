#ifndef NET_H
#define NET_H

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

typedef struct {
    // Dimensions
    int input_dim;
    int hidden_dim;
    int output_dim;
    
    // Weight matrices
    double* W1;    // hidden_dim x input_dim
    double* W2;    // output_dim x hidden_dim
    
    // Layer activations
    double* h;     // 3 x hidden_dim
    
    // Gradient accumulation
    double* dW1;   // hidden_dim x input_dim
    double* dW2;   // output_dim x hidden_dim
    
    // Adam parameters
    double* m1;    // hidden_dim x input_dim
    double* m2;    // output_dim x hidden_dim
    double* v1;    // hidden_dim x input_dim
    double* v2;    // output_dim x hidden_dim
    
    unsigned long long t;
    double lr;
    double beta1;
    double beta2;
    double epsilon;
} Net;

static double gelu(double x) {
    return 0.5 * x * (1 + tanh(sqrt(2/M_PI) * (x + 0.044715 * pow(x, 3))));
}

static double gelu_derivative(double x) {
    double cdf = 0.5 * (1 + tanh(sqrt(2/M_PI) * (x + 0.044715 * pow(x, 3))));
    double pdf = exp(-0.5 * x * x) / sqrt(2 * M_PI);
    return cdf + x * pdf;
}

void free_net(Net* net) {
    if (!net) return;
    free(net->W1);
    free(net->W2);
    free(net->h);
    free(net->dW1);
    free(net->dW2);
    free(net->m1);
    free(net->m2);
    free(net->v1);
    free(net->v2);
    free(net);
}

Net* create_net(int input_dim, int hidden_dim, int output_dim, double learning_rate) {
    Net* net = (Net*)calloc(1, sizeof(Net));
    if (!net) return NULL;

    net->input_dim = input_dim;
    net->hidden_dim = hidden_dim;
    net->output_dim = output_dim;

    // Allocate memory
    size_t w1_size = hidden_dim * input_dim;
    size_t w2_size = output_dim * hidden_dim;
    size_t h_size = 3 * hidden_dim;

    net->W1 = calloc(w1_size, sizeof(double));
    net->W2 = calloc(w2_size, sizeof(double));
    net->h = calloc(h_size, sizeof(double));
    net->dW1 = calloc(w1_size, sizeof(double));
    net->dW2 = calloc(w2_size, sizeof(double));
    net->m1 = calloc(w1_size, sizeof(double));
    net->m2 = calloc(w2_size, sizeof(double));
    net->v1 = calloc(w1_size, sizeof(double));
    net->v2 = calloc(w2_size, sizeof(double));

    if (!net->W1 || !net->W2 || !net->h || !net->dW1 || !net->dW2 || 
        !net->m1 || !net->m2 || !net->v1 || !net->v2) {
        free_net(net);
        return NULL;
    }

    net->lr = learning_rate;
    net->beta1 = 0.9;
    net->beta2 = 0.999;
    net->epsilon = 1e-8;
    net->t = 0;

    // Xavier initialization
    double scale1 = sqrt(2.0 / (input_dim + hidden_dim));
    double scale2 = sqrt(2.0 / (hidden_dim + output_dim));

    for (int i = 0; i < hidden_dim; i++) {
        for (int j = 0; j < input_dim; j++) {
            net->W1[i * input_dim + j] = ((double)rand()/RAND_MAX * 2 - 1) * scale1;
        }
    }

    for (int i = 0; i < output_dim; i++) {
        for (int j = 0; j < hidden_dim; j++) {
            net->W2[i * hidden_dim + j] = ((double)rand()/RAND_MAX * 2 - 1) * scale2;
        }
    }

    return net;
}

void forward_net(Net* net, const double* input) {
    // Copy input
    memcpy(net->h, input, net->input_dim * sizeof(double));

    // Hidden layer
    memset(net->h + net->hidden_dim, 0, net->hidden_dim * sizeof(double));
    for (int i = 0; i < net->hidden_dim; i++) {
        for (int j = 0; j < net->input_dim; j++) {
            net->h[net->hidden_dim + i] += net->W1[i * net->input_dim + j] * net->h[j];
        }
        net->h[net->hidden_dim + i] = gelu(net->h[net->hidden_dim + i]);
    }

    // Output layer
    memset(net->h + 2 * net->hidden_dim, 0, net->output_dim * sizeof(double));
    for (int i = 0; i < net->output_dim; i++) {
        for (int j = 0; j < net->hidden_dim; j++) {
            net->h[2 * net->hidden_dim + i] += net->W2[i * net->hidden_dim + j] * net->h[net->hidden_dim + j];
        }
    }
}

void backward_net(Net* net, const double* output_gradients) {
    double* delta = malloc(net->hidden_dim * sizeof(double));
    
    // Output layer gradients
    for (int i = 0; i < net->output_dim; i++) {
        for (int j = 0; j < net->hidden_dim; j++) {
            net->dW2[i * net->hidden_dim + j] = output_gradients[i] * net->h[net->hidden_dim + j];
        }
    }

    // Hidden layer gradients
    memset(delta, 0, net->hidden_dim * sizeof(double));
    for (int i = 0; i < net->hidden_dim; i++) {
        for (int j = 0; j < net->output_dim; j++) {
            delta[i] += output_gradients[j] * net->W2[j * net->hidden_dim + i];
        }
        delta[i] *= gelu_derivative(net->h[net->hidden_dim + i]);

        for (int j = 0; j < net->input_dim; j++) {
            net->dW1[i * net->input_dim + j] = delta[i] * net->h[j];
        }
    }

    free(delta);
}

void zero_gradients(Net* net) {
    memset(net->dW1, 0, net->hidden_dim * net->input_dim * sizeof(double));
    memset(net->dW2, 0, net->output_dim * net->hidden_dim * sizeof(double));
}

void update_net(Net* net) {
    net->t++;
    
    double beta1_t = pow(net->beta1, net->t);
    double beta2_t = pow(net->beta2, net->t);
    
    // Update first layer weights
    for (int i = 0; i < net->hidden_dim; i++) {
        for (int j = 0; j < net->input_dim; j++) {
            int idx = i * net->input_dim + j;
            net->m1[idx] = net->beta1 * net->m1[idx] + (1.0 - net->beta1) * net->dW1[idx];
            net->v1[idx] = net->beta2 * net->v1[idx] + (1.0 - net->beta2) * net->dW1[idx] * net->dW1[idx];
            double m_hat = net->m1[idx] / (1.0 - beta1_t);
            double v_hat = net->v1[idx] / (1.0 - beta2_t);
            net->W1[idx] -= net->lr * m_hat / (sqrt(v_hat) + net->epsilon);
        }
    }

    // Update second layer weights
    for (int i = 0; i < net->output_dim; i++) {
        for (int j = 0; j < net->hidden_dim; j++) {
            int idx = i * net->hidden_dim + j;
            net->m2[idx] = net->beta1 * net->m2[idx] + (1.0 - net->beta1) * net->dW2[idx];
            net->v2[idx] = net->beta2 * net->v2[idx] + (1.0 - net->beta2) * net->dW2[idx] * net->dW2[idx];
            double m_hat = net->m2[idx] / (1.0 - beta1_t);
            double v_hat = net->v2[idx] / (1.0 - beta2_t);
            net->W2[idx] -= net->lr * m_hat / (sqrt(v_hat) + net->epsilon);
        }
    }
}

bool save_net(const char* filename, const Net* net) {
    FILE* file = fopen(filename, "wb");
    if (!file) return false;
    
    // Save dimensions and parameters
    fwrite(&net->input_dim, sizeof(int), 1, file);
    fwrite(&net->hidden_dim, sizeof(int), 1, file);
    fwrite(&net->output_dim, sizeof(int), 1, file);
    fwrite(&net->lr, sizeof(double), 1, file);
    fwrite(&net->beta1, sizeof(double), 1, file);
    fwrite(&net->beta2, sizeof(double), 1, file);
    fwrite(&net->epsilon, sizeof(double), 1, file);
    fwrite(&net->t, sizeof(unsigned long long), 1, file);
    
    // Save weights and Adam states
    size_t w1_size = net->hidden_dim * net->input_dim;
    size_t w2_size = net->output_dim * net->hidden_dim;
    
    fwrite(net->W1, sizeof(double), w1_size, file);
    fwrite(net->W2, sizeof(double), w2_size, file);
    fwrite(net->m1, sizeof(double), w1_size, file);
    fwrite(net->m2, sizeof(double), w2_size, file);
    fwrite(net->v1, sizeof(double), w1_size, file);
    fwrite(net->v2, sizeof(double), w2_size, file);
    
    fclose(file);
    return true;
}

Net* load_net(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) return NULL;
    
    int input_dim, hidden_dim, output_dim;
    double learning_rate;
    
    // Read dimensions
    if (fread(&input_dim, sizeof(int), 1, file) != 1 ||
        fread(&hidden_dim, sizeof(int), 1, file) != 1 ||
        fread(&output_dim, sizeof(int), 1, file) != 1 ||
        fread(&learning_rate, sizeof(double), 1, file) != 1) {
        fclose(file);
        return NULL;
    }
    
    Net* net = create_net(input_dim, hidden_dim, output_dim, learning_rate);
    if (!net) {
        fclose(file);
        return NULL;
    }
    
    // Read parameters
    if (fread(&net->beta1, sizeof(double), 1, file) != 1 ||
        fread(&net->beta2, sizeof(double), 1, file) != 1 ||
        fread(&net->epsilon, sizeof(double), 1, file) != 1 ||
        fread(&net->t, sizeof(unsigned long long), 1, file) != 1) {
        free_net(net);
        fclose(file);
        return NULL;
    }
    
    // Read weights and Adam states
    size_t w1_size = hidden_dim * input_dim;
    size_t w2_size = output_dim * hidden_dim;
    
    if (fread(net->W1, sizeof(double), w1_size, file) != w1_size ||
        fread(net->W2, sizeof(double), w2_size, file) != w2_size ||
        fread(net->m1, sizeof(double), w1_size, file) != w1_size ||
        fread(net->m2, sizeof(double), w2_size, file) != w2_size ||
        fread(net->v1, sizeof(double), w1_size, file) != w1_size ||
        fread(net->v2, sizeof(double), w2_size, file) != w2_size) {
        free_net(net);
        fclose(file);
        return NULL;
    }
    
    fclose(file);
    return net;
}

#endif // NET_H