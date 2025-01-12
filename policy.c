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
#define LEARNING_RATE 1e-5
#define PPO_EPSILON 0.2
#define PI 3.14159265358979323846
#define ENTROPY_COEF 0.01

typedef struct {
    double min_mean, max_mean;
    double min_var, max_var;
    double min_ratio, max_ratio;
    double min_advantage, max_advantage;
    double avg_entropy;
    double avg_kl;
    double grad_norm;
    double effective_lr;
    int clipped_count;
    int invalid_count;
    double advantage_std;
} TrainStats;

void init_train_stats(TrainStats *stats) {
    stats->min_mean = DBL_MAX;
    stats->max_mean = -DBL_MAX;
    stats->min_var = DBL_MAX;
    stats->max_var = -DBL_MAX;
    stats->min_ratio = DBL_MAX;
    stats->max_ratio = -DBL_MAX;
    stats->min_advantage = DBL_MAX;
    stats->max_advantage = -DBL_MAX;
    stats->avg_entropy = 0;
    stats->avg_kl = 0;
    stats->grad_norm = 0;
    stats->effective_lr = 0;
    stats->clipped_count = 0;
    stats->invalid_count = 0;
    stats->advantage_std = 0;
}

void print_train_stats(const TrainStats *stats, int total_steps) {
    printf("\nTraining Statistics:\n");
    printf("Policy:\n");
    printf("  Mean Range: [%.3f, %.3f]\n", stats->min_mean, stats->max_mean);
    printf("  Variance Range: [%.3f, %.3f]\n", stats->min_var, stats->max_var);
    printf("  Average Entropy: %.3f\n", stats->avg_entropy / total_steps);
    printf("PPO Metrics:\n");
    printf("  Probability Ratio Range: [%.3f, %.3f]\n", stats->min_ratio, stats->max_ratio);
    printf("  Clipped Updates: %.1f%%\n", 100.0 * stats->clipped_count / total_steps);
    printf("  Average KL Divergence: %.3f\n", stats->avg_kl / total_steps);
    printf("Training Dynamics:\n");
    printf("  Advantage Range: [%.3f, %.3f] (std: %.3f)\n", 
           stats->min_advantage, stats->max_advantage, stats->advantage_std);
    printf("  Gradient Norm: %.3f\n", stats->grad_norm);
    printf("  Effective Learning Rate: %.2e\n", stats->effective_lr);
    printf("Stability:\n");
    printf("  Invalid Updates: %.1f%%\n", 100.0 * stats->invalid_count / total_steps);
}

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
    
    for(int i = 0; i < M_OUT/2; i++) {
        double sum = b4[i];
        for(int j = 0; j < D3; j++) sum += W4[i*D3 + j] * h3[j];
        output[i] = 50.0 + 50.0 / (1.0 + exp(-sum));
    }
    
    for(int i = M_OUT/2; i < M_OUT; i++) {
        double sum = b4[i];
        for(int j = 0; j < D3; j++) sum += W4[i*D3 + j] * h3[j];
        output[i] = 1.0 + 9.0 / (1.0 + exp(-sum));
    }
}

double gaussian_log_prob(double x, double mean, double var) {
    const double min_var = 1e-6;
    var = fmax(var, min_var);
    double diff = x - mean;
    if (fabs(diff) > 100.0) return -1e6;
    return -0.5 * (log(2.0 * PI * var) + diff * diff / var);
}

double compute_prob_ratio(double log_prob_new, double log_prob_old) {
    double diff = fmin(log_prob_new - log_prob_old, 20.0);
    return fmax(1e-6, fmin(exp(diff), 1e6));
}

void adam_update(double *params, double *grads, double *m, double *v, int size, int step, double *grad_norm) {
    double lr_t = LEARNING_RATE * sqrt(1.0 - pow(BETA2, step)) / (1.0 - pow(BETA1, step));
    *grad_norm = 0;
    
    for(int i = 0; i < size; i++) {
        grads[i] = fmax(fmin(grads[i], 1.0), -1.0);
        *grad_norm += grads[i] * grads[i];
        
        m[i] = BETA1 * m[i] + (1-BETA1) * grads[i];
        v[i] = BETA2 * v[i] + (1-BETA2) * grads[i] * grads[i];
        params[i] += lr_t * (m[i] / (sqrt(v[i]) + EPSILON) - DECAY * params[i]);
    }
    *grad_norm = sqrt(*grad_norm);
}

void backward(double *W1, double *b1, double *W2, double *b2, double *W3, double *b3, double *W4, double *b4,
             double *m_W1, double *m_b1, double *m_W2, double *m_b2, double *m_W3, double *m_b3, double *m_W4, double *m_b4,
             double *v_W1, double *v_b1, double *v_W2, double *v_b2, double *v_W3, double *v_b3, double *v_W4, double *v_b4,
             double *input, double *h1, double *h2, double *h3, double *output, double *actions,
             double *old_means, double *old_vars, double advantage, int step,
             double *dW1, double *db1, double *dW2, double *db2, double *dW3, double *db3, double *dW4, double *db4,
             TrainStats *stats) {

    double log_prob_new = 0, log_prob_old = 0;
    double entropy = 0;
    
    for(int i = 0; i < 4; i++) {
        log_prob_new += gaussian_log_prob(actions[i], output[i], output[i+4]);
        log_prob_old += gaussian_log_prob(actions[i], old_means[i], old_vars[i]);
        entropy += 0.5 * (log(2.0 * PI * output[i+4]) + 1.0);
        
        stats->min_mean = fmin(stats->min_mean, output[i]);
        stats->max_mean = fmax(stats->max_mean, output[i]);
        stats->min_var = fmin(stats->min_var, output[i+4]);
        stats->max_var = fmax(stats->max_var, output[i+4]);
    }
    
    double prob_ratio = compute_prob_ratio(log_prob_new, log_prob_old);
    double clipped_ratio = fmax(fmin(prob_ratio, 1.0 + PPO_EPSILON), 1.0 - PPO_EPSILON);
    double surrogate = -fmin(prob_ratio * advantage, clipped_ratio * advantage);
    double total_loss = surrogate - ENTROPY_COEF * entropy;
    
    stats->min_ratio = fmin(stats->min_ratio, prob_ratio);
    stats->max_ratio = fmax(stats->max_ratio, prob_ratio);
    stats->min_advantage = fmin(stats->min_advantage, advantage);
    stats->max_advantage = fmax(stats->max_advantage, advantage);
    stats->avg_entropy += entropy;
    if (prob_ratio != clipped_ratio) stats->clipped_count++;
    if (!isfinite(prob_ratio)) stats->invalid_count++;

    memset(dW1, 0, D1*M_IN*sizeof(double));
    memset(db1, 0, D1*sizeof(double));
    memset(dW2, 0, D2*D1*sizeof(double));
    memset(db2, 0, D2*sizeof(double));
    memset(dW3, 0, D3*D2*sizeof(double));
    memset(db3, 0, D3*sizeof(double));
    memset(dW4, 0, M_OUT*D3*sizeof(double));
    memset(db4, 0, M_OUT*sizeof(double));

    for(int i = 0; i < 4; i++) {
        double d_mean = total_loss * (actions[i] - output[i]) / fmax(output[i+4], 1e-6);
        double d_var = total_loss * (pow(actions[i] - output[i], 2) / (2.0 * pow(output[i+4], 2)) - 0.5 / output[i+4]);
        
        d_mean = fmax(fmin(d_mean, 1.0), -1.0);
        d_var = fmax(fmin(d_var, 1.0), -1.0);
        
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
        dh3 = fmax(fmin(dh3, 1.0), -1.0);
        for(int j = 0; j < D2; j++) dW3[i*D2 + j] = dh3 * h2[j];
        db3[i] = dh3;
    }

    for(int i = 0; i < D2; i++) {
        double dh2 = 0;
        for(int j = 0; j < D3; j++) dh2 += db3[j] * W3[j*D2 + i];
        dh2 *= (h2[i] > 0 ? 1.0 : 0.1);
        dh2 = fmax(fmin(dh2, 1.0), -1.0);
        for(int j = 0; j < D1; j++) dW2[i*D1 + j] = dh2 * h1[j];
        db2[i] = dh2;
    }

    for(int i = 0; i < D1; i++) {
        double dh1 = 0;
        for(int j = 0; j < D2; j++) dh1 += db2[j] * W2[j*D1 + i];
        dh1 *= (h1[i] > 0 ? 1.0 : 0.1);
        dh1 = fmax(fmin(dh1, 1.0), -1.0);
        for(int j = 0; j < M_IN; j++) dW1[i*M_IN + j] = dh1 * input[j];
        db1[i] = dh1;
    }

    double grad_norm;
    adam_update(W1, dW1, m_W1, v_W1, D1*M_IN, step, &grad_norm);
    adam_update(b1, db1, m_b1, v_b1, D1, step, &grad_norm);
    adam_update(W2, dW2, m_W2, v_W2, D2*D1, step, &grad_norm);
    adam_update(b2, db2, m_b2, v_b2, D2, step, &grad_norm);
    adam_update(W3, dW3, m_W3, v_W3, D3*D2, step, &grad_norm);
    adam_update(b3, db3, m_b3, v_b3, D3, step, &grad_norm);
    adam_update(W4, dW4, m_W4, v_W4, M_OUT*D3, step, &grad_norm);
    adam_update(b4, db4, m_b4, v_b4, M_OUT, step, &grad_norm);
    
    stats->grad_norm = grad_norm;
    stats->effective_lr = LEARNING_RATE * sqrt(1.0 - pow(BETA2, step)) / (1.0 - pow(BETA1, step));
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

    double *h1 = malloc(D1*sizeof(double));
    double *h2 = malloc(D2*sizeof(double));
    double *h3 = malloc(D3*sizeof(double));

    double *m_W1 = calloc(D1*M_IN, sizeof(double)), *m_b1 = calloc(D1, sizeof(double));
    double *m_W2 = calloc(D2*D1, sizeof(double)), *m_b2 = calloc(D2, sizeof(double));
    double *m_W3 = calloc(D3*D2, sizeof(double)), *m_b3 = calloc(D3, sizeof(double));
    double *m_W4 = calloc(M_OUT*D3, sizeof(double)), *m_b4 = calloc(M_OUT, sizeof(double));
    
    double *v_W1 = calloc(D1*M_IN, sizeof(double)), *v_b1 = calloc(D1, sizeof(double));
    double *v_W2 = calloc(D2*D1, sizeof(double)), *v_b2 = calloc(D2, sizeof(double));
    double *v_W3 = calloc(D3*D2, sizeof(double)), *v_b3 = calloc(D3, sizeof(double));
    double *v_W4 = calloc(M_OUT*D3, sizeof(double)), *v_b4 = calloc(M_OUT, sizeof(double));

    double *dW1 = malloc(D1*M_IN*sizeof(double)), *db1 = malloc(D1*sizeof(double));
    double *dW2 = malloc(D2*D1*sizeof(double)), *db2 = malloc(D2*sizeof(double));
    double *dW3 = malloc(D3*D2*sizeof(double)), *db3 = malloc(D3*sizeof(double));
    double *dW4 = malloc(M_OUT*D3*sizeof(double)), *db4 = malloc(M_OUT*sizeof(double));

    if (!load_weights(argv[2], W1, b1, W2, b2, W3, b3, W4, b4)) {
        printf("Failed to load weights\n");
        return 1;
    }

    FILE *f = fopen(argv[1], "r");
    if (!f) {
        printf("Failed to open CSV\n");
        return 1;
    }

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
        token = strtok(NULL, ",");
        
        for(int j = 0; j < 18; j++) token = strtok(NULL, ",");
        
        for(int j = 0; j < M_IN; j++) {
            all_inputs[i * M_IN + j] = atof(token);
            token = strtok(NULL, ",");
        }

        for(int j = 0; j < 4; j++) {
            all_old_means[i * 4 + j] = atof(token);
            token = strtok(NULL, ",");
        }
        for(int j = 0; j < 4; j++) {
            all_old_vars[i * 4 + j] = atof(token);
            token = strtok(NULL, ",");
        }

        for(int j = 0; j < 4; j++) {
            all_actions[i * 4 + j] = atof(token);
            token = strtok(NULL, ",");
        }

        for(int j = 0; j < 2; j++) token = strtok(NULL, ",");
        all_advantages[i] = atof(token);
        indices[i] = i;
    }
    fclose(f);

    // Normalize advantages
    double adv_mean = 0, adv_std = 0;
    for(int i = 0; i < num_rows; i++) {
        adv_mean += all_advantages[i];
    }
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
    int epochs = 10;
    int print_interval = 1000;
    
    printf("\nStarting PPO training for %d epochs...\n", epochs);
    
    for(int epoch = 0; epoch < epochs; epoch++) {
        TrainStats stats;
        init_train_stats(&stats);
        stats.advantage_std = adv_std;
        
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
                    advantage, step + 1,
                    dW1, db1, dW2, db2, dW3, db3, dW4, db4,
                    &stats);

            if((step + 1) % print_interval == 0) {
                printf("\rEpoch %d/%d, Step %d/%d", 
                       epoch + 1, epochs, step + 1, num_rows);
                fflush(stdout);
            }
        }
        printf("\n");
        print_train_stats(&stats, num_rows);
    }

    printf("\nSaving final weights...\n");
    save_weights(argv[2], W1, b1, W2, b2, W3, b3, W4, b4);

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