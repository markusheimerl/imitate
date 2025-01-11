#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#define D1 64
#define D2 32
#define D3 16
#define M_IN 18  // pos[3], vel[3], ang_vel[3], R[9]
#define BETA1 0.9
#define BETA2 0.999
#define EPSILON 1e-8
#define DECAY 0.01

static double prev_loss = 1e30;
static double learning_rate = 1e-4;

void save_weights(const char* filename, double *W1, double *b1, double *W2, double *b2, 
                 double *W3, double *b3, double *W4, double *b4) {
    FILE *f = fopen(filename, "wb");
    fwrite(W1, sizeof(double), D1*M_IN, f);
    fwrite(b1, sizeof(double), D1, f);
    fwrite(W2, sizeof(double), D2*D1, f);
    fwrite(b2, sizeof(double), D2, f);
    fwrite(W3, sizeof(double), D3*D2, f);
    fwrite(b3, sizeof(double), D3, f);
    fwrite(W4, sizeof(double), D3, f);
    fwrite(b4, sizeof(double), 1, f);
    fclose(f);
}

bool load_weights(const char* filename, double *W1, double *b1, double *W2, double *b2,
                 double *W3, double *b3, double *W4, double *b4) {
    FILE *f = fopen(filename, "rb");
    if (!f) return false;
    
    size_t read = fread(W1, sizeof(double), D1*M_IN, f) +
                  fread(b1, sizeof(double), D1, f) +
                  fread(W2, sizeof(double), D2*D1, f) +
                  fread(b2, sizeof(double), D2, f) +
                  fread(W3, sizeof(double), D3*D2, f) +
                  fread(b3, sizeof(double), D3, f) +
                  fread(W4, sizeof(double), D3, f) +
                  fread(b4, sizeof(double), 1, f);
    fclose(f);
    return read == (D1*M_IN + D1 + D2*D1 + D2 + D3*D2 + D3 + D3 + 1);
}

void forward(double *W1, double *b1, double *W2, double *b2, double *W3, double *b3,
            double *W4, double *b4, double *input, double *h1, double *h2, double *h3, double *output) {
    // First layer
    for(int i = 0; i < D1; i++) {
        double sum = b1[i];
        for(int j = 0; j < M_IN; j++) {
            sum += W1[i*M_IN + j] * input[j];
        }
        h1[i] = sum > 0 ? sum : sum * 0.1;  // LeakyReLU
    }

    // Second layer
    for(int i = 0; i < D2; i++) {
        double sum = b2[i];
        for(int j = 0; j < D1; j++) {
            sum += W2[i*D1 + j] * h1[j];
        }
        h2[i] = sum > 0 ? sum : sum * 0.1;
    }

    // Third layer
    for(int i = 0; i < D3; i++) {
        double sum = b3[i];
        for(int j = 0; j < D2; j++) {
            sum += W3[i*D2 + j] * h2[j];
        }
        h3[i] = sum > 0 ? sum : sum * 0.1;
    }

    // Output layer (linear)
    *output = b4[0];
    for(int i = 0; i < D3; i++) {
        *output += W4[i] * h3[i];
    }
}

void adam_update(double *param, double *grad, double *m, double *v, int size, int step) {
    double lr_t = learning_rate * sqrt(1.0 - pow(BETA2, step)) / (1.0 - pow(BETA1, step));
    
    for(int i = 0; i < size; i++) {
        m[i] = BETA1 * m[i] + (1-BETA1) * grad[i];
        v[i] = BETA2 * v[i] + (1-BETA2) * grad[i] * grad[i];
        param[i] -= lr_t * (m[i] / (sqrt(v[i]) + EPSILON) + DECAY * param[i]);
    }
}

double backward(double *W1, double *b1, double *W2, double *b2, double *W3, double *b3,
               double *W4, double *b4, double *m_W1, double *m_b1, double *m_W2, double *m_b2,
               double *m_W3, double *m_b3, double *m_W4, double *m_b4, double *v_W1, double *v_b1,
               double *v_W2, double *v_b2, double *v_W3, double *v_b3, double *v_W4, double *v_b4,
               double *input, double *h1, double *h2, double *h3, double *output, double target, 
               double *d_W1, double *d_b1, double *d_W2, double *d_b2, double *d_W3, double *d_b3, 
               double *d_W4, double *d_h1, double *d_h2, double *d_h3, int step) {
    
    // Clear gradients
    memset(d_W1, 0, D1*M_IN*sizeof(double));
    memset(d_b1, 0, D1*sizeof(double));
    memset(d_W2, 0, D2*D1*sizeof(double));
    memset(d_b2, 0, D2*sizeof(double));
    memset(d_W3, 0, D3*D2*sizeof(double));
    memset(d_b3, 0, D3*sizeof(double));
    memset(d_W4, 0, D3*sizeof(double));

    double d_out = 2.0 * (*output - target);
    double d_b4 = d_out;

    // Output layer gradients
    for(int i = 0; i < D3; i++) {
        d_W4[i] = d_out * h3[i];
    }

    // Third layer gradients
    for(int i = 0; i < D3; i++) {
        d_h3[i] = d_out * W4[i] * (h3[i] > 0 ? 1.0 : 0.1);
        for(int j = 0; j < D2; j++) {
            d_W3[i*D2 + j] = d_h3[i] * h2[j];
        }
        d_b3[i] = d_h3[i];
    }

    // Second layer gradients
    for(int i = 0; i < D2; i++) {
        d_h2[i] = 0;
        for(int j = 0; j < D3; j++) {
            d_h2[i] += d_h3[j] * W3[j*D2 + i];
        }
        d_h2[i] *= (h2[i] > 0 ? 1.0 : 0.1);
        for(int j = 0; j < D1; j++) {
            d_W2[i*D1 + j] = d_h2[i] * h1[j];
        }
        d_b2[i] = d_h2[i];
    }

    // First layer gradients
    for(int i = 0; i < D1; i++) {
        d_h1[i] = 0;
        for(int j = 0; j < D2; j++) {
            d_h1[i] += d_h2[j] * W2[j*D1 + i];
        }
        d_h1[i] *= (h1[i] > 0 ? 1.0 : 0.1);
        for(int j = 0; j < M_IN; j++) {
            d_W1[i*M_IN + j] = d_h1[i] * input[j];
        }
        d_b1[i] = d_h1[i];
    }

    // Update weights using Adam
    adam_update(W1, d_W1, m_W1, v_W1, D1*M_IN, step);
    adam_update(b1, d_b1, m_b1, v_b1, D1, step);
    adam_update(W2, d_W2, m_W2, v_W2, D2*D1, step);
    adam_update(b2, d_b2, m_b2, v_b2, D2, step);
    adam_update(W3, d_W3, m_W3, v_W3, D3*D2, step);
    adam_update(b3, d_b3, m_b3, v_b3, D3, step);
    adam_update(W4, d_W4, m_W4, v_W4, D3, step);
    adam_update(b4, &d_b4, m_b4, v_b4, 1, step);
    
    return (*output - target) * (*output - target);
}

int main(int argc, char **argv) {
    if(argc < 2) {
        printf("Usage: %s <data_file> [weights_file]\n", argv[0]);
        return 1;
    }

    printf("Initializing network...\n");
    fflush(stdout);

    printf("Allocating network parameters...\n");
    fflush(stdout);
    // Initialize weights and buffers
    double *W1 = malloc(D1*M_IN*sizeof(double)), *b1 = calloc(D1, sizeof(double));
    double *W2 = malloc(D2*D1*sizeof(double)), *b2 = calloc(D2, sizeof(double));
    double *W3 = malloc(D3*D2*sizeof(double)), *b3 = calloc(D3, sizeof(double));
    double *W4 = malloc(D3*sizeof(double)), *b4 = calloc(1, sizeof(double));
    
    // Hidden layer activations
    double *h1 = malloc(D1*sizeof(double));
    double *h2 = malloc(D2*sizeof(double));
    double *h3 = malloc(D3*sizeof(double));

    // Adam optimizer momentum buffers
    double *m_W1 = calloc(D1*M_IN, sizeof(double)), *m_b1 = calloc(D1, sizeof(double));
    double *m_W2 = calloc(D2*D1, sizeof(double)), *m_b2 = calloc(D2, sizeof(double));
    double *m_W3 = calloc(D3*D2, sizeof(double)), *m_b3 = calloc(D3, sizeof(double));
    double *m_W4 = calloc(D3, sizeof(double)), *m_b4 = calloc(1, sizeof(double));
    
    // Adam optimizer variance buffers
    double *v_W1 = calloc(D1*M_IN, sizeof(double)), *v_b1 = calloc(D1, sizeof(double));
    double *v_W2 = calloc(D2*D1, sizeof(double)), *v_b2 = calloc(D2, sizeof(double));
    double *v_W3 = calloc(D3*D2, sizeof(double)), *v_b3 = calloc(D3, sizeof(double));
    double *v_W4 = calloc(D3, sizeof(double)), *v_b4 = calloc(1, sizeof(double));

    // Gradient buffers
    double *d_W1 = malloc(D1*M_IN*sizeof(double));
    double *d_b1 = malloc(D1*sizeof(double));
    double *d_W2 = malloc(D2*D1*sizeof(double));
    double *d_b2 = malloc(D2*sizeof(double));
    double *d_W3 = malloc(D3*D2*sizeof(double));
    double *d_b3 = malloc(D3*sizeof(double));
    double *d_W4 = malloc(D3*sizeof(double));
    double *d_h1 = malloc(D1*sizeof(double));
    double *d_h2 = malloc(D2*sizeof(double));
    double *d_h3 = malloc(D3*sizeof(double));
    printf("Network parameters allocated\n");
    fflush(stdout);

    // Load or initialize weights
    if (argc > 2) {
        printf("Loading weights from %s...\n", argv[2]);
        fflush(stdout);
        if (!load_weights(argv[2], W1, b1, W2, b2, W3, b3, W4, b4)) {
            printf("Failed to load weights\n");
            return 1;
        }
    } else {
        printf("Initializing weights randomly...\n");
        fflush(stdout);
        srand(time(NULL));
        for(int i = 0; i < D1*M_IN; i++) W1[i] = ((double)rand()/RAND_MAX - 0.5) * sqrt(2.0/M_IN);
        for(int i = 0; i < D2*D1; i++) W2[i] = ((double)rand()/RAND_MAX - 0.5) * sqrt(2.0/D1);
        for(int i = 0; i < D3*D2; i++) W3[i] = ((double)rand()/RAND_MAX - 0.5) * sqrt(2.0/D2);
        for(int i = 0; i < D3; i++) W4[i] = ((double)rand()/RAND_MAX - 0.5) * sqrt(2.0/D3);
    }

    printf("Loading training data from %s...\n", argv[1]);
    fflush(stdout);

    printf("Opening file...\n");
    fflush(stdout);
    FILE *f = fopen(argv[1], "r");
    if (!f) {
        printf("Failed to open file\n");
        return 1;
    }

    printf("Counting rows...\n");
    fflush(stdout);
    char line[1024];
    int rows = -1;  // -1 to account for header
    while(fgets(line, sizeof(line), f)) rows++;
    printf("Found %d training examples\n", rows);
    fflush(stdout);

    rewind(f);
    fgets(line, sizeof(line), f);  // Skip header

    printf("Allocating data arrays...\n");
    fflush(stdout);
    double **data = malloc(rows * sizeof(double*));
    double *targets = malloc(rows * sizeof(double));
    int *indices = malloc(rows * sizeof(int));

    printf("Allocating individual data rows...\n");
    fflush(stdout);
    for(int i = 0; i < rows; i++) {
        data[i] = malloc(M_IN * sizeof(double));
        indices[i] = i;
        if (i % 10000 == 0) {
            printf("Allocated %d rows\n", i);
            fflush(stdout);
        }
    }

    printf("Reading data...\n");
    fflush(stdout);
    for(int i = 0; i < rows; i++) {
        if (i % 10000 == 0) {
            printf("Read %d rows\n", i);
            fflush(stdout);
        }
        
        if (!fgets(line, sizeof(line), f)) {
            printf("Unexpected end of file at row %d\n", i);
            return 1;
        }
        
        // Debug print for first row
        if (i == 0) {
            printf("First row content: %s\n", line);
        }

        // Remove newline character if present
        line[strcspn(line, "\n")] = 0;
        
        char *token = strtok(line, ",");  // rollout
        if (!token) {
            printf("Failed to get first token in row %d\n", i);
            return 1;
        }

        // Read 18 values: pos[3], vel[3], ang_vel[3], R[9]
        for(int j = 0; j < M_IN; j++) {
            token = strtok(NULL, ",");
            if (!token) {
                printf("Missing data in row %d, column %d\n", i, j);
                return 1;
            }
            data[i][j] = atof(token);
            if (i == 0) {
                printf("Column %d: %f\n", j, data[i][j]);
            }
        }

        // Skip acc_s[3], gyro_s[3], omega[4]
        for(int j = 0; j < 10; j++) {
            token = strtok(NULL, ",");
            if (!token) {
                printf("Missing data while skipping column %d in row %d\n", j, i);
                return 1;
            }
        }
        
        // Get reward
        token = strtok(NULL, ",");
        if (!token) {
            printf("Missing reward in row %d\n", i);
            return 1;
        }
        
        // Get discounted_return (final column)
        token = strtok(NULL, "\n\r,");
        if (!token) {
            printf("Missing discounted_return in row %d\n", i);
            return 1;
        }
        
        if (i == 0) {
            printf("First row discounted_return: %s\n", token);
        }
        
        targets[i] = atof(token);
    }

    printf("Starting training...\n");
    printf("Epoch | Step | Loss | Learning Rate | Prediction | Target\n");
    printf("------------------------------------------------\n");
    fflush(stdout);

    // Training loop
    double output, running_loss = 0;
    int step = 1;

    for(int epoch = 0; epoch < 500; epoch++) {
        // Shuffle data
        for(int i = rows-1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }

        for(int i = 0; i < rows; i++, step++) {
            forward(W1, b1, W2, b2, W3, b3, W4, b4, data[indices[i]], h1, h2, h3, &output);
            
            running_loss += backward(W1, b1, W2, b2, W3, b3, W4, b4,
                                   m_W1, m_b1, m_W2, m_b2, m_W3, m_b3, m_W4, m_b4,
                                   v_W1, v_b1, v_W2, v_b2, v_W3, v_b3, v_W4, v_b4,
                                   data[indices[i]], h1, h2, h3, &output, targets[indices[i]],
                                   d_W1, d_b1, d_W2, d_b2, d_W3, d_b3, d_W4,
                                   d_h1, d_h2, d_h3, step);

            if(step % 10000 == 0) {
                double avg_loss = running_loss/10000;
                learning_rate *= (avg_loss > prev_loss) ? 0.95 : 1.05;
                learning_rate = fmax(1e-6, fmin(1e-2, learning_rate));
                printf("%3d | %6d | %.6f | %.2e | %.6f | %.6f\n", epoch, step, avg_loss, learning_rate, output, targets[indices[i]]);
                fflush(stdout);
                prev_loss = avg_loss;
                running_loss = 0;
            }
        }
    }

    // Save weights
    char filename[100];
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    sprintf(filename, "%d-%d-%d_%d-%d-%d_value_weights.bin", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    
    if (argc > 2) {
        save_weights(argv[2], W1, b1, W2, b2, W3, b3, W4, b4);
    } else {
        save_weights(filename, W1, b1, W2, b2, W3, b3, W4, b4);
    }

    printf("Training complete. Weights saved to %s\n", argc > 2 ? argv[2] : filename);
    fflush(stdout);

    // Cleanup
    for(int i = 0; i < rows; i++) free(data[i]);
    free(data); free(targets); free(indices);
    free(W1); free(b1); free(W2); free(b2); free(W3); free(b3); free(W4); free(b4);
    free(h1); free(h2); free(h3);
    free(m_W1); free(m_b1); free(m_W2); free(m_b2);
    free(m_W3); free(m_b3); free(m_W4); free(m_b4);
    free(v_W1); free(v_b1); free(v_W2); free(v_b2);
    free(v_W3); free(v_b3); free(v_W4); free(v_b4);
    free(d_W1); free(d_b1); free(d_W2); free(d_b2);
    free(d_W3); free(d_b3); free(d_W4);
    free(d_h1); free(d_h2); free(d_h3);

    return 0;
}