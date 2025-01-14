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

// Compute mean baseline from returns
double compute_baseline(Data** rollouts, int num_rollouts) {
    double sum = 0;
    int count = 0;
    
    for(int i = 0; i < num_rollouts; i++) {
        for(int t = 0; t < rollouts[i]->n; t++) {
            sum += rollouts[i]->y[t][ACTION_DIM + 1]; // Return column
            count++;
        }
    }
    
    return sum / count;
}

void update_policy(Net* policy, Data** rollouts, int num_rollouts) {
    // Compute baseline
    double baseline = compute_baseline(rollouts, num_rollouts);
    printf("Baseline return: %.3f\n", baseline);
    
    // Allocate temporary arrays
    double* state = malloc(STATE_DIM * sizeof(double));
    double* action_means = malloc(4 * sizeof(double));
    double* action_logvars = malloc(4 * sizeof(double));
    double* gradients = malloc(policy->sz[3] * sizeof(double));
    double** activations = malloc(4 * sizeof(double*));
    for(int i = 0; i < 4; i++) {
        activations[i] = malloc(policy->sz[i] * sizeof(double));
    }
    
    // Process all trajectories
    for(int i = 0; i < num_rollouts; i++) {
        for(int t = 0; t < rollouts[i]->n; t++) {
            // Get state and policy output
            memcpy(state, rollouts[i]->X[t], STATE_DIM * sizeof(double));
            fwd(policy, state, activations);
            
            // Get actual actions and advantage
            double advantage = rollouts[i]->y[t][ACTION_DIM + 1] - baseline;
            
            // Zero gradients
            memset(gradients, 0, policy->sz[3] * sizeof(double));
            
            // Compute gradients for means and logvars
            for(int j = 0; j < 4; j++) {
                double mean = activations[3][j];
                double logvar = activations[3][j + 4];
                double std = exp(0.5 * logvar);
                double action = rollouts[i]->y[t][j];
                
                // Mean gradient
                gradients[j] = (action - mean) * advantage / (std * std);
                
                // Logvar gradient 
                gradients[j + 4] = 0.5 * ((action - mean) * (action - mean) / (std * std) - 1.0) * advantage;
            }
            
            // Update policy
            bwd(policy, activations, gradients, NULL, LEARNING_RATE);
        }
    }
    
    // Cleanup
    free(state);
    free(action_means);
    free(action_logvars);
    free(gradients);
    for(int i = 0; i < 4; i++) {
        free(activations[i]);
    }
    free(activations);
}

int main(int argc, char** argv) {
    if(argc != 2) {
        printf("Usage: %s <policy.bin>\n", argv[0]);
        return 1;
    }
    
    // Load policy
    Net* policy = load_weights(argv[1]);
    if(!policy) return 1;
    
    // Load rollouts
    Data* rollouts[100];  // Arbitrary max
    int num_rollouts = 0;
    
    DIR *dir = opendir(".");
    if(!dir) return 1;
    
    struct dirent *ent;
    while((ent = readdir(dir)) != NULL) {
        if(strstr(ent->d_name, "rollout.csv")) {
            rollouts[num_rollouts] = load_rollout(ent->d_name);
            if(rollouts[num_rollouts]) num_rollouts++;
        }
    }
    closedir(dir);
    
    printf("Loaded %d rollouts\n", num_rollouts);
    if(num_rollouts == 0) return 1;
    
    // Update policy
    update_policy(policy, rollouts, num_rollouts);
    
    // Save updated policy
    save_weights(policy, argv[1]);
    
    // Cleanup
    for(int i = 0; i < num_rollouts; i++) {
        free_data(rollouts[i]);
    }
    free_net(policy);
    
    return 0;
}