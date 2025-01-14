#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <math.h>
#include "grad/grad.h"
#include "grad/data.h"

#define STATE_DIM 12
#define ACTION_DIM 8  // 4 means + 4 logvars
#define HIDDEN_DIM 64
#define NUM_ROLLOUTS 10
#define LEARNING_RATE 0.001
#define MAX_GRAD_NORM 1.0
#define ENTROPY_COEF 0.01

// Forward declare
void clip_gradients(double* grad, int size, double max_norm);

// Load a single rollout file
Data* load_rollout(const char* filename) {
    Data* data = load_csv(filename);
    if(!data) {
        printf("Failed to load %s\n", filename);
        return NULL;
    }
    return data;
}

// Update policy using REINFORCE
void update_policy(Net* policy, Data** rollouts, int num_rollouts) {
    double** act = malloc(4 * sizeof(double*));
    double** grad = malloc(4 * sizeof(double*));
    for(int i = 0; i < 4; i++) {
        act[i] = malloc(policy->sz[i] * sizeof(double));
        grad[i] = malloc(policy->sz[i] * sizeof(double));
    }
    
    // Compute mean return for baseline
    double total_return = 0.0;
    int total_steps = 0;
    for(int r = 0; r < num_rollouts; r++) {
        for(int t = 0; t < rollouts[r]->n; t++) {
            total_return += rollouts[r]->y[t][ACTION_DIM + 1];  // Last column is return
            total_steps++;
        }
    }
    double baseline = total_return / total_steps;
    printf("Baseline return: %.3f\n", baseline);
    
    // Process all timesteps from all rollouts
    double total_loss = 0.0;
    for(int r = 0; r < num_rollouts; r++) {
        for(int t = 0; t < rollouts[r]->n; t++) {
            // Forward pass
            fwd(policy, rollouts[r]->X[t], act);
            
            // Zero gradients
            memset(grad[3], 0, policy->sz[3] * sizeof(double));
            
            // Get return for this timestep
            double G = rollouts[r]->y[t][ACTION_DIM + 1];  // Last column is return
            double advantage = G - baseline;
            
            // Compute gradients for means and logvars
            for(int i = 0; i < 4; i++) {
                double mean = act[3][i];
                double logvar = act[3][i + 4];
                double std = exp(0.5 * logvar);
                double action = rollouts[r]->y[t][i];  // First 4 columns are actions
                
                // Compute policy gradient
                double delta = (action - mean) / (std + 1e-8);
                grad[3][i] = -delta * advantage / (std + 1e-8);
                
                // Compute logvar gradient with entropy bonus
                grad[3][i + 4] = (-0.5 * (delta * delta - 1.0) * advantage + 
                                 ENTROPY_COEF) / (std + 1e-8);
                
                // Accumulate loss
                total_loss += -0.5 * (delta * delta + logvar);
            }
            
            // Clip gradients
            clip_gradients(grad[3], policy->sz[3], MAX_GRAD_NORM);
            
            // Update policy
            bwd(policy, act, grad[3], grad, LEARNING_RATE);
        }
    }
    
    printf("Average loss: %.3f\n", total_loss / total_steps);
    
    for(int i = 0; i < 4; i++) {
        free(act[i]);
        free(grad[i]);
    }
    free(act);
    free(grad);
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