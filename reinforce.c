#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include "grad/grad.h"
#include "grad/data.h"

#define STATE_DIM 12
#define ACTION_DIM 8
#define HIDDEN_DIM 64
#define NUM_ROLLOUTS 10
#define LEARNING_RATE 0.0001
#define MAX_GRAD_NORM 1.0

// Load a single rollout file
Data* load_rollout(const char* filename) {
    Data* data = load_csv(filename);
    if(!data) {
        printf("Failed to load %s\n", filename);
        return NULL;
    }
    return data;
}

// Compute statistics of returns across all rollouts
void compute_return_stats(Data** rollouts, int num_rollouts, 
                         double* mean, double* std) {
    int total_steps = 0;
    *mean = 0.0;
    *std = 0.0;
    
    // Calculate mean
    for(int r = 0; r < num_rollouts; r++) {
        for(int t = 0; t < rollouts[r]->n; t++) {
            *mean += rollouts[r]->y[t][ACTION_DIM + 1];
            total_steps++;
        }
    }
    *mean /= total_steps;
    
    // Calculate standard deviation
    for(int r = 0; r < num_rollouts; r++) {
        for(int t = 0; t < rollouts[r]->n; t++) {
            *std += pow(rollouts[r]->y[t][ACTION_DIM + 1] - *mean, 2);
        }
    }
    *std = sqrt(*std / total_steps);
    
    // Prevent division by zero
    if(*std < 1e-6) *std = 1.0;
}

// Clip gradients to prevent explosive updates
void clip_gradients(double* grad, int size, double max_norm) {
    double norm = 0.0;
    for(int i = 0; i < size; i++) {
        norm += grad[i] * grad[i];
    }
    norm = sqrt(norm);
    
    if(norm > max_norm) {
        double scale = max_norm / norm;
        for(int i = 0; i < size; i++) {
            grad[i] *= scale;
        }
    }
}

// Update policy using REINFORCE
void update_policy(Net* policy, Data** rollouts, int num_rollouts) {
    double** act = malloc(4 * sizeof(double*));
    double** grad = malloc(4 * sizeof(double*));
    for(int i = 0; i < 4; i++) {
        act[i] = malloc(policy->sz[i] * sizeof(double));
        grad[i] = malloc(policy->sz[i] * sizeof(double));
    }
    
    // Compute return statistics for normalization
    double mean_return, std_return;
    compute_return_stats(rollouts, num_rollouts, &mean_return, &std_return);
    printf("Return stats - Mean: %.3f, Std: %.3f\n", mean_return, std_return);
    
    // Process all timesteps from all rollouts
    for(int r = 0; r < num_rollouts; r++) {
        for(int t = 0; t < rollouts[r]->n; t++) {
            // Forward pass
            fwd(policy, rollouts[r]->X[t], act);
            
            // Get normalized return for this timestep
            double G = (rollouts[r]->y[t][ACTION_DIM + 1] - mean_return) / std_return;
            
            // Zero out gradients
            memset(grad[3], 0, policy->sz[3] * sizeof(double));
            
            // Compute gradients for means and logvars
            for(int i = 0; i < 4; i++) {
                double mean = act[3][i];
                double logvar = act[3][i + 4];
                double action = rollouts[r]->y[t][i];
                double var = exp(logvar);
                
                // Mean gradient
                grad[3][i] = (action - mean) / var * G;
                
                // Logvar gradient
                grad[3][i + 4] = 0.5 * (pow(action - mean, 2) / var - 1.0) * G;
            }
            
            // Clip gradients
            clip_gradients(grad[3], policy->sz[3], MAX_GRAD_NORM);
            
            // Backward pass with learning rate
            bwd(policy, act, grad[3], grad, LEARNING_RATE);
        }
    }
    
    for(int i = 0; i < 4; i++) {
        free(act[i]);
        free(grad[i]);
    }
    free(act);
    free(grad);
}

int main(int argc, char** argv) {
    if(argc != 2) {
        printf("Usage: %s <policy.bin>\n", argv[0]);
        return 1;
    }
    
    // Load policy
    Net* policy = load_weights(argv[1]);
    if(!policy) {
        printf("Failed to load policy from %s\n", argv[1]);
        return 1;
    }
    
    // Load all rollout files
    Data* rollouts[NUM_ROLLOUTS];
    int loaded_rollouts = 0;
    
    DIR *dir;
    struct dirent *ent;
    dir = opendir(".");
    if(!dir) {
        printf("Failed to open directory\n");
        return 1;
    }
    
    while((ent = readdir(dir)) != NULL && loaded_rollouts < NUM_ROLLOUTS) {
        if(strstr(ent->d_name, "rollout.csv")) {
            rollouts[loaded_rollouts] = load_rollout(ent->d_name);
            if(rollouts[loaded_rollouts]) {
                loaded_rollouts++;
            }
        }
    }
    closedir(dir);
    
    if(loaded_rollouts == 0) {
        printf("No rollout files found\n");
        return 1;
    }
    
    printf("Loaded %d rollouts\n", loaded_rollouts);
    
    // Update policy
    update_policy(policy, rollouts, loaded_rollouts);
    
    // Save updated policy
    save_weights(policy, argv[1]);
    printf("Updated policy saved to %s\n", argv[1]);
    
    // Cleanup
    for(int i = 0; i < loaded_rollouts; i++) {
        free_data(rollouts[i]);
    }
    free_net(policy);
    
    return 0;
}