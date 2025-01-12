#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <float.h>

#define D1 64
#define D2 32
#define D3 16
#define M_IN 6
#define M_OUT 8  // 4 means, 4 variances

#define BETA1 0.9
#define BETA2 0.999
#define EPSILON 1e-8
#define DECAY 0.01
#define LEARNING_RATE 1e-6
#define PPO_EPSILON 0.2
#define PI 3.14159265358979323846
#define ENTROPY_COEF 0.01

bool load_weights(const char* filename, double *W1, double *b1, double *W2, double *b2, double *W3, double *b3, double *W4, double *b4) {
    FILE *f = fopen(filename, "rb");
    if (!f) return false;
    size_t items_read = fread(W1, sizeof(double), D1*M_IN, f) +
                       fread(b1, sizeof(double), D1, f) +
                       fread(W2, sizeof(double), D2*D1, f) +
                       fread(b2, sizeof(double), D2, f) +
                       fread(W3, sizeof(double), D3*D2, f) +
                       fread(b3, sizeof(double), D3, f) +
                       fread(W4, sizeof(double), M_OUT*D3, f) +
                       fread(b4, sizeof(double), M_OUT, f);
    fclose(f);
    return items_read == (D1*M_IN + D1 + D2*D1 + D2 + D3*D2 + D3 + M_OUT*D3 + M_OUT);
}

void save_weights(const char* filename, double *W1, double *b1, double *W2, double *b2, double *W3, double *b3, double *W4, double *b4) {
    FILE *f = fopen(filename, "wb");
    fwrite(W1, sizeof(double), D1*M_IN, f);
    fwrite(b1, sizeof(double), D1, f);
    fwrite(W2, sizeof(double), D2*D1, f);
    fwrite(b2, sizeof(double), D2, f);
    fwrite(W3, sizeof(double), D3*D2, f);
    fwrite(b3, sizeof(double), D3, f);
    fwrite(W4, sizeof(double), M_OUT*D3, f);
    fwrite(b4, sizeof(double), M_OUT, f);
    fclose(f);
}

void forward(double *W1, double *b1, double *W2, double *b2, double *W3, double *b3,
            double *W4, double *b4, double *input, double *h1, double *h2, double *h3, double *output) {
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

    for(int i = 0; i < M_OUT/2; i++) {
        double sum = b4[i];
        for(int j = 0; j < D3; j++) sum += W4[i*D3 + j] * h3[j];
        output[i] = sum;  // Raw mean output
    }
    
    for(int i = M_OUT/2; i < M_OUT; i++) {
        double sum = b4[i];
        for(int j = 0; j < D3; j++) sum += W4[i*D3 + j] * h3[j];
        output[i] = exp(sum);  // Log-variance to variance
    }
}

double gaussian_log_prob(double x, double mean, double var) {
    const double min_var = 1e-6;
    var = fmax(var, min_var);
    double diff = x - mean;
    return -0.5 * (log(2.0 * PI * var) + diff * diff / var);
}

void adam_update(double *param, double *grad, double *m, double *v, int size, int step) {
    double lr_t = LEARNING_RATE * sqrt(1.0 - pow(BETA2, step)) / (1.0 - pow(BETA1, step));
    for(int i = 0; i < size; i++) {
        m[i] = BETA1 * m[i] + (1-BETA1) * grad[i];
        v[i] = BETA2 * v[i] + (1-BETA2) * grad[i] * grad[i];
        param[i] -= lr_t * (m[i] / (sqrt(v[i]) + EPSILON) + DECAY * param[i]);
    }
}

void backward(double *W1, double *b1, double *W2, double *b2, double *W3, double *b3, double *W4, double *b4,
             double *m_W1, double *m_b1, double *m_W2, double *m_b2, double *m_W3, double *m_b3, double *m_W4, double *m_b4,
             double *v_W1, double *v_b1, double *v_W2, double *v_b2, double *v_W3, double *v_b3, double *v_W4, double *v_b4,
             double *input, double *h1, double *h2, double *h3, double *output,
             double *actions, double *old_means, double *old_vars, double advantage, int step) {
    
    double d_W1[D1*M_IN] = {0}, d_b1[D1] = {0};
    double d_W2[D2*D1] = {0}, d_b2[D2] = {0};
    double d_W3[D3*D2] = {0}, d_b3[D3] = {0};
    double d_W4[M_OUT*D3] = {0}, d_b4[M_OUT] = {0};
    double d_h1[D1], d_h2[D2], d_h3[D3];

    // Calculate log probabilities in a numerically stable way
    double log_prob_new = 0, log_prob_old = 0;
    for(int i = 0; i < 4; i++) {
        // Compute individual action log probs for debugging
        double lp_new = gaussian_log_prob(actions[i], output[i], output[i+4]);
        double lp_old = gaussian_log_prob(actions[i], old_means[i], old_vars[i]);
        
        if (step % 10000 == 0) {
            printf("Action %d: action=%.3f, new_mean=%.3f, new_std=%.3f, old_mean=%.3f, old_std=%.3f, lp_new=%.3f, lp_old=%.3f\n",
                   i, actions[i], output[i], sqrt(output[i+4]), old_means[i], sqrt(old_vars[i]), lp_new, lp_old);
        }
        
        log_prob_new += lp_new;
        log_prob_old += lp_old;
    }

    // Compute probability ratio in a numerically stable way
    double ratio = exp(log_prob_new - log_prob_old);
    double clipped_ratio = fmax(fmin(ratio, 1.0 + PPO_EPSILON), 1.0 - PPO_EPSILON);
    
    // Compute surrogate loss
    double surrogate_1 = ratio * advantage;
    double surrogate_2 = clipped_ratio * advantage;
    double surrogate_loss = -fmin(surrogate_1, surrogate_2);  // Negative because we're minimizing

    if (step % 10000 == 0) {
        printf("ratio=%.3f, clipped_ratio=%.3f, advantage=%.3f, loss=%.3f\n",
               ratio, clipped_ratio, advantage, surrogate_loss);
    }

    // Compute gradients for each action dimension
    for(int i = 0; i < 4; i++) {
        double mean = output[i];
        double var = output[i+4];
        double action = actions[i];
        
        // Determine if we should use the clipped or unclipped objective
        bool use_clipped = (advantage >= 0 && ratio > 1.0 + PPO_EPSILON) || 
                          (advantage < 0 && ratio < 1.0 - PPO_EPSILON);
        double grad_scale = use_clipped ? 0.0 : ratio;

        // Gradient for mean
        double d_mean = grad_scale * advantage * (action - mean) / var;
        d_b4[i] = d_mean;
        
        // Gradient for variance with minimum variance constraint
        double d_var = grad_scale * advantage * (
            (action - mean) * (action - mean) / (2.0 * var * var) - 1.0 / (2.0 * var)
        );
        
        // Apply chain rule through exp (since variance is parameterized as log variance)
        d_b4[i+4] = d_var * var;
        
        // Weight gradients
        for(int j = 0; j < D3; j++) {
            d_W4[i*D3 + j] = d_mean * h3[j];
            d_W4[(i+4)*D3 + j] = d_var * var * h3[j];
        }
    }

    // Backpropagate through hidden layers
    for(int i = 0; i < D3; i++) {
        d_h3[i] = 0;
        for(int j = 0; j < M_OUT; j++) {
            d_h3[i] += d_b4[j] * W4[j*D3 + i];
        }
        d_h3[i] *= (h3[i] > 0 ? 1.0 : 0.1);  // Leaky ReLU derivative
        for(int j = 0; j < D2; j++) {
            d_W3[i*D2 + j] = d_h3[i] * h2[j];
        }
        d_b3[i] = d_h3[i];
    }

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

    // Apply updates with Adam
    adam_update(W1, d_W1, m_W1, v_W1, D1*M_IN, step);
    adam_update(b1, d_b1, m_b1, v_b1, D1, step);
    adam_update(W2, d_W2, m_W2, v_W2, D2*D1, step);
    adam_update(b2, d_b2, m_b2, v_b2, D2, step);
    adam_update(W3, d_W3, m_W3, v_W3, D3*D2, step);
    adam_update(b3, d_b3, m_b3, v_b3, D3, step);
    adam_update(W4, d_W4, m_W4, v_W4, M_OUT*D3, step);
    adam_update(b4, d_b4, m_b4, v_b4, M_OUT, step);
}

int main(int argc, char **argv) {
    if(argc != 3) {
        printf("Usage: %s <trajectory_csv> <policy_weights>\n", argv[0]);
        return 1;
    }

    double *W1 = malloc(D1*M_IN*sizeof(double)), *b1 = calloc(D1, sizeof(double));
    double *W2 = malloc(D2*D1*sizeof(double)), *b2 = calloc(D2, sizeof(double));
    double *W3 = malloc(D3*D2*sizeof(double)), *b3 = calloc(D3, sizeof(double));
    double *W4 = malloc(M_OUT*D3*sizeof(double)), *b4 = calloc(M_OUT, sizeof(double));

    double *m_W1 = calloc(D1*M_IN, sizeof(double)), *m_b1 = calloc(D1, sizeof(double));
    double *m_W2 = calloc(D2*D1, sizeof(double)), *m_b2 = calloc(D2, sizeof(double));
    double *m_W3 = calloc(D3*D2, sizeof(double)), *m_b3 = calloc(D3, sizeof(double));
    double *m_W4 = calloc(M_OUT*D3, sizeof(double)), *m_b4 = calloc(M_OUT, sizeof(double));
    
    double *v_W1 = calloc(D1*M_IN, sizeof(double)), *v_b1 = calloc(D1, sizeof(double));
    double *v_W2 = calloc(D2*D1, sizeof(double)), *v_b2 = calloc(D2, sizeof(double));
    double *v_W3 = calloc(D3*D2, sizeof(double)), *v_b3 = calloc(D3, sizeof(double));
    double *v_W4 = calloc(M_OUT*D3, sizeof(double)), *v_b4 = calloc(M_OUT, sizeof(double));

    if (!load_weights(argv[2], W1, b1, W2, b2, W3, b3, W4, b4)) {
        printf("Failed to load weights\n");
        return 1;
    }

    FILE *f = fopen(argv[1], "r");
    if (!f) { printf("Failed to open CSV\n"); return 1; }

    char line[4096];
    fgets(line, sizeof(line), f);
    int num_rows = 0;
    while (fgets(line, sizeof(line), f)) num_rows++;
    rewind(f);
    fgets(line, sizeof(line), f);

    double *all_inputs = malloc(num_rows * M_IN * sizeof(double));
    double *all_actions = malloc(num_rows * 4 * sizeof(double));
    double *all_old_means = malloc(num_rows * 4 * sizeof(double));
    double *all_old_vars = malloc(num_rows * 4 * sizeof(double));
    double *all_advantages = malloc(num_rows * sizeof(double));
    int *indices = malloc(num_rows * sizeof(int));

    printf("Loading %d training samples...\n", num_rows);

    for(int i = 0; i < num_rows; i++) {
        fgets(line, sizeof(line), f);
        char *token = strtok(line, ",");
        
        // Read position (3), velocity (3), and angular velocity (3)
        for(int j = 0; j < 9; j++) token = strtok(NULL, ",");
        
        // Read accelerometer and gyroscope readings (policy inputs)
        for(int j = 0; j < M_IN; j++) {
            all_inputs[i * M_IN + j] = atof(token);
            token = strtok(NULL, ",");
        }

        // Read means (4)
        for(int j = 0; j < 4; j++) {
            all_old_means[i * 4 + j] = atof(token);
            token = strtok(NULL, ",");
        }
        
        // Read variances (4)
        for(int j = 0; j < 4; j++) {
            all_old_vars[i * 4 + j] = atof(token);
            token = strtok(NULL, ",");
        }

        // Read actions (4)
        for(int j = 0; j < 4; j++) {
            all_actions[i * 4 + j] = atof(token);
            token = strtok(NULL, ",");
        }

        // Skip reward and read advantage
        token = strtok(NULL, ",");  // skip reward
        all_advantages[i] = atof(token);  // read advantage
        indices[i] = i;
    }
    fclose(f);

    double adv_mean = 0, adv_std = 0;
    for(int i = 0; i < num_rows; i++) adv_mean += all_advantages[i];
    adv_mean /= num_rows;
    
    for(int i = 0; i < num_rows; i++) {
        double diff = all_advantages[i] - adv_mean;
        adv_std += diff * diff;
    }
    adv_std = sqrt(adv_std / num_rows);
    
    for(int i = 0; i < num_rows; i++) {
        all_advantages[i] = (all_advantages[i] - adv_mean) / (adv_std + 1e-8);
    }

    double output[M_OUT];
    double h1[D1], h2[D2], h3[D3];
    int epochs = 10;
    
    printf("\nStarting PPO training for %d epochs...\n", epochs);
    
    for(int epoch = 0; epoch < epochs; epoch++) {
        for(int i = num_rows-1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }

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
                    advantage, step + 1);

            if((step + 1) % 1000 == 0) {
                printf("\rEpoch %d/%d, Step %d/%d", epoch + 1, epochs, step + 1, num_rows);
                fflush(stdout);
            }
        }
        printf("\n");
    }

    printf("\nSaving final weights...\n");
    save_weights(argv[2], W1, b1, W2, b2, W3, b3, W4, b4);

    free(W1); free(b1); free(W2); free(b2); free(W3); free(b3); free(W4); free(b4);
    free(m_W1); free(m_b1); free(m_W2); free(m_b2);
    free(m_W3); free(m_b3); free(m_W4); free(m_b4);
    free(v_W1); free(v_b1); free(v_W2); free(v_b2);
    free(v_W3); free(v_b3); free(v_W4); free(v_b4);
    free(all_inputs);
    free(all_actions);
    free(all_old_means);
    free(all_old_vars);
    free(all_advantages);
    free(indices);

    return 0;
}