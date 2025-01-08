#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <omp.h>

#define INPUT_FEATURES 10
#define HIDDEN_SIZE 128
#define N_HIDDEN 4
#define OUTPUT_FEATURES 4
#define LEARNING_RATE 0.0001
#define TRAINING_STEPS 100000
#define CLIP_VALUE 1.0

typedef struct { double *data; int rows, cols; } Dataset;

double randn() { return sqrt(-2.0 * log((double)rand() / RAND_MAX)) * cos(2.0 * M_PI * (double)rand() / RAND_MAX); }

Dataset load_csv(const char* filename) {
    Dataset ds = {malloc(1000 * (INPUT_FEATURES + OUTPUT_FEATURES) * sizeof(double)), 0, (INPUT_FEATURES + OUTPUT_FEATURES)};
    char line[1024]; FILE* f = fopen(filename, "r");
    if (!f || !fgets(line, 1024, f)) { printf("File error\n"); exit(1); }
    
    while (fgets(line, 1024, f)) {
        if (ds.rows >= 1000) ds.data = realloc(ds.data, (ds.rows * 2) * (INPUT_FEATURES + OUTPUT_FEATURES) * sizeof(double));
        char* tok = strtok(line, ",");
        for (int i = 0; i < (INPUT_FEATURES + OUTPUT_FEATURES) && tok; i++, tok = strtok(NULL, ",")) 
            ds.data[ds.rows * (INPUT_FEATURES + OUTPUT_FEATURES) + i] = atof(tok);
        ds.rows++;
    }
    fclose(f);
    return ds;
}

void save_weights(double** weights, double** biases) {
    time_t t = time(NULL); struct tm tm = *localtime(&t);
    char filename[100];
    sprintf(filename, "%d-%d-%d_%d-%d-%d_weights.bin", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    FILE* f = fopen(filename, "wb");
    if (!f) return;
    
    fwrite(weights[0], sizeof(double), INPUT_FEATURES * HIDDEN_SIZE, f);
    fwrite(biases[0], sizeof(double), HIDDEN_SIZE, f);
    for (int i = 1; i < N_HIDDEN; i++) {
        fwrite(weights[i], sizeof(double), HIDDEN_SIZE * HIDDEN_SIZE, f);
        fwrite(biases[i], sizeof(double), HIDDEN_SIZE, f);
    }
    fwrite(weights[N_HIDDEN], sizeof(double), HIDDEN_SIZE * OUTPUT_FEATURES, f);
    fwrite(biases[N_HIDDEN], sizeof(double), OUTPUT_FEATURES, f);
    fclose(f);
    printf("Saved weights to: %s\n", filename);
}

int load_weights(const char* filename, double** weights, double** biases) {
    FILE* f = fopen(filename, "rb");
    if (!f) return 0;
    
    size_t read = fread(weights[0], sizeof(double), INPUT_FEATURES * HIDDEN_SIZE, f);
    read += fread(biases[0], sizeof(double), HIDDEN_SIZE, f);
    for (int i = 1; i < N_HIDDEN; i++) {
        read += fread(weights[i], sizeof(double), HIDDEN_SIZE * HIDDEN_SIZE, f);
        read += fread(biases[i], sizeof(double), HIDDEN_SIZE, f);
    }
    read += fread(weights[N_HIDDEN], sizeof(double), HIDDEN_SIZE * OUTPUT_FEATURES, f);
    read += fread(biases[N_HIDDEN], sizeof(double), OUTPUT_FEATURES, f);
    fclose(f);
    return read == (INPUT_FEATURES * HIDDEN_SIZE + HIDDEN_SIZE + (N_HIDDEN - 1) * (HIDDEN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE) + HIDDEN_SIZE * OUTPUT_FEATURES + OUTPUT_FEATURES);
}

double relu(double x) { return x > 0 ? x : 0; }
double relu_derivative(double x) { return x > 0 ? 1 : 0; }

void forward_pass(const double* input, double** activations, double** weights, double** biases) {
    #pragma omp parallel for
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        double sum = biases[0][i];
        for (int j = 0; j < INPUT_FEATURES; j++) sum += input[j] * weights[0][i * INPUT_FEATURES + j];
        activations[0][i] = relu(sum);
    }
    
    for (int layer = 1; layer < N_HIDDEN; layer++) {
        #pragma omp parallel for
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            double sum = biases[layer][i];
            for (int j = 0; j < HIDDEN_SIZE; j++) sum += activations[layer-1][j] * weights[layer][i * HIDDEN_SIZE + j];
            activations[layer][i] = relu(sum);
        }
    }
    
    #pragma omp parallel for
    for (int i = 0; i < OUTPUT_FEATURES; i++) {
        double sum = biases[N_HIDDEN][i];
        for (int j = 0; j < HIDDEN_SIZE; j++) sum += activations[N_HIDDEN-1][j] * weights[N_HIDDEN][i * HIDDEN_SIZE + j];
        activations[N_HIDDEN][i] = sum;
    }
}

double compute_loss(const double* pred, const double* target) {
    double loss = 0.0;
    #pragma omp parallel for reduction(+:loss)
    for (int i = 0; i < OUTPUT_FEATURES; i++) {
        double error = pred[i] - target[i];
        loss += error * error;
    }
    return loss / OUTPUT_FEATURES;
}

void clip_gradients(double* grad, int size) {
    double norm = 0.0;
    #pragma omp parallel for reduction(+:norm)
    for (int i = 0; i < size; i++) norm += grad[i] * grad[i];
    norm = sqrt(norm);
    if (norm > CLIP_VALUE) {
        double scale = CLIP_VALUE / norm;
        #pragma omp parallel for
        for (int i = 0; i < size; i++) grad[i] *= scale;
    }
}

void backward_pass(const double* input, const double* target, double** activations, double** weights, double** biases, double learning_rate) {
    double* output_deltas = malloc(OUTPUT_FEATURES * sizeof(double));
    double** hidden_deltas = malloc(N_HIDDEN * sizeof(double*));
    for (int i = 0; i < N_HIDDEN; i++) hidden_deltas[i] = calloc(HIDDEN_SIZE, sizeof(double));
    
    #pragma omp parallel for
    for (int i = 0; i < OUTPUT_FEATURES; i++) output_deltas[i] = 2 * (activations[N_HIDDEN][i] - target[i]) / OUTPUT_FEATURES;
    clip_gradients(output_deltas, OUTPUT_FEATURES);
    
    #pragma omp parallel for
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_FEATURES; j++) hidden_deltas[N_HIDDEN-1][i] += output_deltas[j] * weights[N_HIDDEN][j * HIDDEN_SIZE + i];
        hidden_deltas[N_HIDDEN-1][i] *= relu_derivative(activations[N_HIDDEN-1][i]);
    }
    clip_gradients(hidden_deltas[N_HIDDEN-1], HIDDEN_SIZE);
    
    for (int layer = N_HIDDEN-2; layer >= 0; layer--) {
        #pragma omp parallel for
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) hidden_deltas[layer][i] += hidden_deltas[layer+1][j] * weights[layer+1][j * HIDDEN_SIZE + i];
            hidden_deltas[layer][i] *= relu_derivative(activations[layer][i]);
        }
        clip_gradients(hidden_deltas[layer], HIDDEN_SIZE);
    }
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_FEATURES; j++)
            weights[0][i * INPUT_FEATURES + j] -= learning_rate * hidden_deltas[0][i] * input[j];
    
    #pragma omp parallel for
    for (int i = 0; i < HIDDEN_SIZE; i++) biases[0][i] -= learning_rate * hidden_deltas[0][i];
    
    for (int layer = 1; layer < N_HIDDEN; layer++) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < HIDDEN_SIZE; i++)
            for (int j = 0; j < HIDDEN_SIZE; j++)
                weights[layer][i * HIDDEN_SIZE + j] -= learning_rate * hidden_deltas[layer][i] * activations[layer-1][j];
        
        #pragma omp parallel for
        for (int i = 0; i < HIDDEN_SIZE; i++) biases[layer][i] -= learning_rate * hidden_deltas[layer][i];
    }
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < OUTPUT_FEATURES; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            weights[N_HIDDEN][i * HIDDEN_SIZE + j] -= learning_rate * output_deltas[i] * activations[N_HIDDEN-1][j];
    
    #pragma omp parallel for
    for (int i = 0; i < OUTPUT_FEATURES; i++) biases[N_HIDDEN][i] -= learning_rate * output_deltas[i];
    
    free(output_deltas);
    for (int i = 0; i < N_HIDDEN; i++) free(hidden_deltas[i]);
    free(hidden_deltas);
}

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 3) { printf("Usage: %s <training_data.csv> [weights.bin]\n", argv[0]); return 1; }
    
    srand(time(NULL));
    Dataset ds = load_csv(argv[1]);
    
    double** weights = malloc((N_HIDDEN + 1) * sizeof(double*));
    double** biases = malloc((N_HIDDEN + 1) * sizeof(double*));
    double** activations = malloc((N_HIDDEN + 1) * sizeof(double*));
    
    weights[0] = malloc(INPUT_FEATURES * HIDDEN_SIZE * sizeof(double));
    biases[0] = malloc(HIDDEN_SIZE * sizeof(double));
    activations[0] = malloc(HIDDEN_SIZE * sizeof(double));
    
    for (int i = 1; i < N_HIDDEN; i++) {
        weights[i] = malloc(HIDDEN_SIZE * HIDDEN_SIZE * sizeof(double));
        biases[i] = malloc(HIDDEN_SIZE * sizeof(double));
        activations[i] = malloc(HIDDEN_SIZE * sizeof(double));
    }
    
    weights[N_HIDDEN] = malloc(HIDDEN_SIZE * OUTPUT_FEATURES * sizeof(double));
    biases[N_HIDDEN] = malloc(OUTPUT_FEATURES * sizeof(double));
    activations[N_HIDDEN] = malloc(OUTPUT_FEATURES * sizeof(double));
    
    if (argc < 3 || !load_weights(argv[2], weights, biases)) {
        double scale = sqrt(2.0 / INPUT_FEATURES);
        for (int i = 0; i < INPUT_FEATURES * HIDDEN_SIZE; i++) weights[0][i] = randn() * scale;
        
        scale = sqrt(2.0 / HIDDEN_SIZE);
        for (int layer = 1; layer < N_HIDDEN; layer++)
            for (int i = 0; i < HIDDEN_SIZE * HIDDEN_SIZE; i++) weights[layer][i] = randn() * scale;
        
        for (int i = 0; i < HIDDEN_SIZE * OUTPUT_FEATURES; i++) weights[N_HIDDEN][i] = randn() * scale;
        
        for (int layer = 0; layer <= N_HIDDEN; layer++)
            for (int i = 0; i < (layer == N_HIDDEN ? OUTPUT_FEATURES : HIDDEN_SIZE); i++) biases[layer][i] = 0.0;
    }
    
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    char loss_filename[100];
    sprintf(loss_filename, "%d-%d-%d_%d-%d-%d_loss.csv", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    
    FILE* loss_file = fopen(loss_filename, "w");
    if (loss_file) fprintf(loss_file, "step,loss\n");
    
    double prev_loss = 1e9;
    double learning_rate = LEARNING_RATE;
    
    for (int step = 0; step < TRAINING_STEPS; step++) {
        int idx = rand() % ds.rows;
        double* input = ds.data + idx * (INPUT_FEATURES + OUTPUT_FEATURES);
        double* target = input + INPUT_FEATURES;
        
        forward_pass(input, activations, weights, biases);
        double loss = compute_loss(activations[N_HIDDEN], target);
        
        if (step > 0) {
            learning_rate *= (loss > prev_loss * 1.1) ? 0.95 : (loss < prev_loss * 0.95) ? 1.05 : 1.0;
            learning_rate = fmax(1e-6, fmin(1e-3, learning_rate));
        }
        prev_loss = loss;
        
        if (step % 1000 == 0) {
            printf("Step %d, Loss: %f, LR: %e\n", step, loss, learning_rate);
            printf("Target: [%.3f, %.3f, %.3f, %.3f]\n", target[0], target[1], target[2], target[3]);
            printf("Prediction: [%.3f, %.3f, %.3f, %.3f]\n\n", 
                   activations[N_HIDDEN][0], activations[N_HIDDEN][1], 
                   activations[N_HIDDEN][2], activations[N_HIDDEN][3]);
        }
        if (loss_file) fprintf(loss_file, "%d,%f\n", step, loss);
        
        backward_pass(input, target, activations, weights, biases, learning_rate);
    }
    
    if (loss_file) fclose(loss_file);
    save_weights(weights, biases);
    
    free(ds.data);
    for (int i = 0; i <= N_HIDDEN; i++) {
        free(weights[i]);
        free(biases[i]);
        free(activations[i]);
    }
    free(weights);
    free(biases);
    free(activations);
    
    return 0;
}