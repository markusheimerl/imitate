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
#define LEARNING_RATE 0.001

// Load a single rollout file
Data* load_rollout(const char* filename) {
    Data* data = load_csv(filename);
    if(!data) {
        printf("Failed to load %s\n", filename);
        return NULL;
    }
    return data;
}

// Compute log probability of action under Gaussian policy
double log_prob(double action, double mean, double logvar) {
    double var = exp(logvar);
    return -0.5 * (log(2 * M_PI) + logvar + pow(action - mean, 2) / var);
}

// Update policy using REINFORCE
void update_policy(Net* policy, Data** rollouts, int num_rollouts) {
    double** act = malloc(4 * sizeof(double*));
    double** grad = malloc(4 * sizeof(double*));
    for(int i = 0; i < 4; i++) {
        act[i] = malloc(policy->sz[i] * sizeof(double));
        grad[i] = malloc(policy->sz[i] * sizeof(double));
    }
    
    // Process all timesteps from all rollouts
    for(int r = 0; r < num_rollouts; r++) {
        for(int t = 0; t < rollouts[r]->n; t++) {
            // Forward pass
            fwd(policy, rollouts[r]->X[t], act);
            
            // Get return for this timestep
            double G = rollouts[r]->y[t][ACTION_DIM + 1];
            
            // Compute gradients for means and logvars
            for(int i = 0; i < 4; i++) {
                double mean = rollouts[r]->y[t][i];
                double logvar = rollouts[r]->y[t][i + 4];
                double action = rollouts[r]->y[t][i];  // Sampled action
                double var = exp(logvar);
                
                // Mean gradient: (action - mean) / var * return
                grad[3][i] = (action - mean) / var * G;
                
                // Logvar gradient: (0.5 * ((action - mean)^2 / var - 1)) * return
                grad[3][i + 4] = 0.5 * (pow(action - mean, 2) / var - 1.0) * G;
            }
            
            // Backward pass
            bwd(policy, act, grad[3], grad, 0.0);
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
    
    // Print average return across rollouts
    double avg_return = 0.0;
    for(int i = 0; i < loaded_rollouts; i++) {
        avg_return += rollouts[i]->y[0][ACTION_DIM + 1];
    }
    avg_return /= loaded_rollouts;
    printf("Average return: %.3f\n", avg_return);
    
    // Update policy
    update_policy(policy, rollouts, loaded_rollouts);
    
    // Overwrite existing policy file
    save_weights(policy, argv[1]);
    printf("Updated policy saved to %s\n", argv[1]);
    
    // Cleanup
    for(int i = 0; i < loaded_rollouts; i++) {
        free_data(rollouts[i]);
    }
    free_net(policy);
    
    return 0;
}