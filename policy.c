#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#define D1 64
#define D2 32
#define D3 16
#define M_IN 6
#define M_OUT 8  // 4 means, 4 variances
#define BETA1 0.9
#define BETA2 0.999
#define EPSILON 1e-8
#define DECAY 0.01
#define LEARNING_RATE 1e-5
#define PPO_EPSILON 0.2  // PPO clipping parameter
#define PI 3.14159265358979323846

bool load_weights(const char* filename, double *W1, double *b1, double *W2, double *b2, double *W3, double *b3, double *W4, double *b4) {
    FILE* f = fopen(filename, "rb");
    if (!f) return false;
    size_t items_read = fread(W1, sizeof(double), D1*M_IN, f) + fread(b1, sizeof(double), D1, f) + 
                       fread(W2, sizeof(double), D2*D1, f) + fread(b2, sizeof(double), D2, f) + 
                       fread(W3, sizeof(double), D3*D2, f) + fread(b3, sizeof(double), D3, f) + 
                       fread(W4, sizeof(double), M_OUT*D3, f) + fread(b4, sizeof(double), M_OUT, f);
    fclose(f);
    return items_read == (D1*M_IN + D1 + D2*D1 + D2 + D3*D2 + D3 + M_OUT*D3 + M_OUT);
}

void save_weights(const char* filename, double *W1, double *b1, double *W2, double *b2, double *W3, double *b3, double *W4, double *b4) {
    FILE* f = fopen(filename, "wb");
    fwrite(W1, sizeof(double), D1*M_IN, f); fwrite(b1, sizeof(double), D1, f);
    fwrite(W2, sizeof(double), D2*D1, f); fwrite(b2, sizeof(double), D2, f);
    fwrite(W3, sizeof(double), D3*D2, f); fwrite(b3, sizeof(double), D3, f);
    fwrite(W4, sizeof(double), M_OUT*D3, f); fwrite(b4, sizeof(double), M_OUT, f);
    fclose(f);
}

void forward(double *W1, double *b1, double *W2, double *b2, double *W3, double *b3, double *W4, double *b4, 
            double *input, double *h1, double *h2, double *h3, double *output) {
    for(int i = 0; i < D1; i++) {
        double sum = b1[i];
        for(int j = 0; j < M_IN; j++) sum += W1[i*M_IN + j] * input[j];
        h1[i] = sum > 0 ? sum : sum * 0.1;
    }
    for(int i = 0; i < D2; i++) {
        double sum = b2[i];
        for(int j = 0; j < D1; j++) sum += W2[i*D1 + j] * h1[j];
        h2[i] = sum > 0 ? sum : sum * 0.1;
    }
    for(int i = 0; i < D3; i++) {
        double sum = b3[i];
        for(int j = 0; j < D2; j++) sum += W3[i*D2 + j] * h2[j];
        h3[i] = sum > 0 ? sum : sum * 0.1;
    }
    for(int i = 0; i < M_OUT/2; i++) {  // First 4 outputs are means
        double sum = b4[i];
        for(int j = 0; j < D3; j++) sum += W4[i*D3 + j] * h3[j];
        output[i] = 50.0 + 50.0 / (1.0 + exp(-sum));  // Mean in [0,100]
    }
    for(int i = M_OUT/2; i < M_OUT; i++) {  // Last 4 outputs are variances
        double sum = b4[i];
        for(int j = 0; j < D3; j++) sum += W4[i*D3 + j] * h3[j];
        output[i] = 10.0 / (1.0 + exp(-sum));  // Variance in [0,10]
    }
}

double gaussian_log_prob(double x, double mean, double var) {
    return -0.5 * (log(2.0 * PI * var) + (x - mean) * (x - mean) / var);
}

void adam_update(double *params, double *grads, double *m, double *v, int size, int step) {
    double lr_t = LEARNING_RATE * sqrt(1.0 - pow(BETA2, step)) / (1.0 - pow(BETA1, step));
    for(int i = 0; i < size; i++) {
        m[i] = BETA1 * m[i] + (1-BETA1) * grads[i];
        v[i] = BETA2 * v[i] + (1-BETA2) * grads[i] * grads[i];
        params[i] += lr_t * (m[i] / (sqrt(v[i]) + EPSILON) - DECAY * params[i]);
    }
}

void backward(double *W1, double *b1, double *W2, double *b2, double *W3, double *b3, double *W4, double *b4,
             double *m_W1, double *m_b1, double *m_W2, double *m_b2, double *m_W3, double *m_b3, double *m_W4, double *m_b4,
             double *v_W1, double *v_b1, double *v_W2, double *v_b2, double *v_W3, double *v_b3, double *v_W4, double *v_b4,
             double *input, double *h1, double *h2, double *h3, double *output, double *actions,
             double *old_means, double *old_vars, double advantage, int step,
             double *dW1, double *db1, double *dW2, double *db2, double *dW3, double *db3, double *dW4, double *db4) {

    // Compute probability ratio
    double log_prob_new = 0, log_prob_old = 0;
    for(int i = 0; i < 4; i++) {
        log_prob_new += gaussian_log_prob(actions[i], output[i], output[i+4]);
        log_prob_old += gaussian_log_prob(actions[i], old_means[i], old_vars[i]);
    }
    double prob_ratio = exp(log_prob_new - log_prob_old);
    
    // Compute clipped surrogate objective
    double clipped_ratio = fmax(fmin(prob_ratio, 1.0 + PPO_EPSILON), 1.0 - PPO_EPSILON);
    double surrogate = -fmin(prob_ratio * advantage, clipped_ratio * advantage);

    memset(dW1, 0, D1*M_IN*sizeof(double)); memset(db1, 0, D1*sizeof(double));
    memset(dW2, 0, D2*D1*sizeof(double)); memset(db2, 0, D2*sizeof(double));
    memset(dW3, 0, D3*D2*sizeof(double)); memset(db3, 0, D3*sizeof(double));
    memset(dW4, 0, M_OUT*D3*sizeof(double)); memset(db4, 0, M_OUT*sizeof(double));

    // Backpropagate through Gaussian distributions
    for(int i = 0; i < 4; i++) {
        double d_mean = surrogate * (actions[i] - output[i]) / output[i+4];
        double d_var = surrogate * (pow(actions[i] - output[i], 2) / (2.0 * pow(output[i+4], 2)) - 0.5 / output[i+4]);
        
        db4[i] = d_mean;
        db4[i+4] = d_var;
        
        for(int j = 0; j < D3; j++) {
            dW4[i*D3 + j] = d_mean * h3[j];
            dW4[(i+4)*D3 + j] = d_var * h3[j];
        }
    }

    for(int i = 0; i < D3; i++) {
        double dh3 = 0;
        for(int j = 0; j < M_OUT; j++) dh3 += db4[j] * W4[j*D3 + i];
        dh3 *= (h3[i] > 0 ? 1.0 : 0.1);
        for(int j = 0; j < D2; j++) dW3[i*D2 + j] = dh3 * h2[j];
        db3[i] = dh3;
    }

    for(int i = 0; i < D2; i++) {
        double dh2 = 0;
        for(int j = 0; j < D3; j++) dh2 += db3[j] * W3[j*D2 + i];
        dh2 *= (h2[i] > 0 ? 1.0 : 0.1);
        for(int j = 0; j < D1; j++) dW2[i*D1 + j] = dh2 * h1[j];
        db2[i] = dh2;
    }

    for(int i = 0; i < D1; i++) {
        double dh1 = 0;
        for(int j = 0; j < D2; j++) dh1 += db2[j] * W2[j*D1 + i];
        dh1 *= (h1[i] > 0 ? 1.0 : 0.1);
        for(int j = 0; j < M_IN; j++) dW1[i*M_IN + j] = dh1 * input[j];
        db1[i] = dh1;
    }

    adam_update(W1, dW1, m_W1, v_W1, D1*M_IN, step);
    adam_update(b1, db1, m_b1, v_b1, D1, step);
    adam_update(W2, dW2, m_W2, v_W2, D2*D1, step);
    adam_update(b2, db2, m_b2, v_b2, D2, step);
    adam_update(W3, dW3, m_W3, v_W3, D3*D2, step);
    adam_update(b3, db3, m_b3, v_b3, D3, step);
    adam_update(W4, dW4, m_W4, v_W4, M_OUT*D3, step);
    adam_update(b4, db4, m_b4, v_b4, M_OUT, step);
}

int main(int argc, char **argv) {
    if(argc != 3) {
        printf("Usage: %s <trajectory_csv> <policy_weights>\n", argv[0]);
        return 1;
    }

    // Network weights and biases
    double *W1 = malloc(D1*M_IN*sizeof(double)), *b1 = calloc(D1, sizeof(double));
    double *W2 = malloc(D2*D1*sizeof(double)), *b2 = calloc(D2, sizeof(double));
    double *W3 = malloc(D3*D2*sizeof(double)), *b3 = calloc(D3, sizeof(double));
    double *W4 = malloc(M_OUT*D3*sizeof(double)), *b4 = calloc(M_OUT, sizeof(double));

    // Layer activations
    double *h1 = malloc(D1*sizeof(double)), *h2 = malloc(D2*sizeof(double));
    double *h3 = malloc(D3*sizeof(double));

    // Adam optimizer states
    double *m_W1 = calloc(D1*M_IN, sizeof(double)), *m_b1 = calloc(D1, sizeof(double));
    double *m_W2 = calloc(D2*D1, sizeof(double)), *m_b2 = calloc(D2, sizeof(double));
    double *m_W3 = calloc(D3*D2, sizeof(double)), *m_b3 = calloc(D3, sizeof(double));
    double *m_W4 = calloc(M_OUT*D3, sizeof(double)), *m_b4 = calloc(M_OUT, sizeof(double));
    double *v_W1 = calloc(D1*M_IN, sizeof(double)), *v_b1 = calloc(D1, sizeof(double));
    double *v_W2 = calloc(D2*D1, sizeof(double)), *v_b2 = calloc(D2, sizeof(double));
    double *v_W3 = calloc(D3*D2, sizeof(double)), *v_b3 = calloc(D3, sizeof(double));
    double *v_W4 = calloc(M_OUT*D3, sizeof(double)), *v_b4 = calloc(M_OUT, sizeof(double));

    // Gradient buffers
    double *dW1 = malloc(D1*M_IN*sizeof(double)), *db1 = malloc(D1*sizeof(double));
    double *dW2 = malloc(D2*D1*sizeof(double)), *db2 = malloc(D2*sizeof(double));
    double *dW3 = malloc(D3*D2*sizeof(double)), *db3 = malloc(D3*sizeof(double));
    double *dW4 = malloc(M_OUT*D3*sizeof(double)), *db4 = malloc(M_OUT*sizeof(double));

    if (!load_weights(argv[2], W1, b1, W2, b2, W3, b3, W4, b4)) {
        printf("Failed to load weights\n");
        return 1;
    }

    // First pass: count lines
    FILE *f = fopen(argv[1], "r");
    if (!f) {
        printf("Failed to open CSV\n");
        return 1;
    }
    char line[4096];
    fgets(line, sizeof(line), f);  // Skip header
    int num_rows = 0;
    while (fgets(line, sizeof(line), f)) num_rows++;
    rewind(f);
    fgets(line, sizeof(line), f);  // Skip header again

    // Read all data into memory
    double *all_inputs = malloc(num_rows * M_IN * sizeof(double));
    double *all_actions = malloc(num_rows * 4 * sizeof(double));
    double *all_old_means = malloc(num_rows * 4 * sizeof(double));
    double *all_old_vars = malloc(num_rows * 4 * sizeof(double));
    double *all_advantages = malloc(num_rows * sizeof(double));
    int *indices = malloc(num_rows * sizeof(int));
    
    for(int i = 0; i < num_rows; i++) {
        fgets(line, sizeof(line), f);
        char *token = strtok(line, ",");
        token = strtok(NULL, ",");  // Skip rollout number
        
        for(int j = 0; j < 18; j++) token = strtok(NULL, ",");  // Skip to acc_s
        
        for(int j = 0; j < M_IN; j++) {
            all_inputs[i * M_IN + j] = atof(token);
            token = strtok(NULL, ",");
        }

        // Read means and variances
        for(int j = 0; j < 4; j++) {
            all_old_means[i * 4 + j] = atof(token);
            token = strtok(NULL, ",");
        }
        for(int j = 0; j < 4; j++) {
            all_old_vars[i * 4 + j] = atof(token);
            token = strtok(NULL, ",");
        }

        // Read actions
        for(int j = 0; j < 4; j++) {
            all_actions[i * 4 + j] = atof(token);
            token = strtok(NULL, ",");
        }

        // Skip to advantage
        for(int j = 0; j < 2; j++) token = strtok(NULL, ",");
        all_advantages[i] = atof(token);
        indices[i] = i;
    }
    fclose(f);

    // Training loop
    double output[M_OUT];
    srand(time(NULL));
    int epochs = 10;
    
    for(int epoch = 0; epoch < epochs; epoch++) {
        // Shuffle indices
        for(int i = num_rows - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }

        double running_loss = 0;
        for(int step = 0; step < num_rows; step++) {
            int idx = indices[step];
            double *input = &all_inputs[idx * M_IN];
            double *actions = &all_actions[idx * 4];
            double *old_means = &all_old_means[idx * 4];
            double *old_vars = &all_old_vars[idx * 4];
            double advantage = all_advantages[idx];

            forward(W1, b1, W2, b2, W3, b3, W4, b4, input, h1, h2, h3, output);
            backward(W1, b1, W2, b2, W3, b3, W4, b4,
                    m_W1, m_b1, m_W2, m_b2, m_W3, m_b3, m_W4, m_b4,
                    v_W1, v_b1, v_W2, v_b2, v_W3, v_b3, v_W4, v_b4,
                    input, h1, h2, h3, output, actions, old_means, old_vars,
                    advantage, step + 1,
                    dW1, db1, dW2, db2, dW3, db3, dW4, db4);

            if((step + 1) % 1000 == 0) {
                printf("\rEpoch %d/%d, Step %d/%d", 
                       epoch + 1, epochs, step + 1, num_rows);
                fflush(stdout);
            }
        }
        printf("\n");
    }

    save_weights(argv[2], W1, b1, W2, b2, W3, b3, W4, b4);

    // Cleanup
    free(W1); free(b1); free(W2); free(b2); free(W3); free(b3); free(W4); free(b4);
    free(h1); free(h2); free(h3);
    free(m_W1); free(m_b1); free(m_W2); free(m_b2);
    free(m_W3); free(m_b3); free(m_W4); free(m_b4);
    free(v_W1); free(v_b1); free(v_W2); free(v_b2);
    free(v_W3); free(v_b3); free(v_W4); free(v_b4);
    free(dW1); free(db1); free(dW2); free(db2);
    free(dW3); free(db3); free(dW4); free(db4);
    free(all_inputs);
    free(all_actions);
    free(all_old_means);
    free(all_old_vars);
    free(all_advantages);
    free(indices);

    return 0;
}