#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dirent.h>
#include "grad/grad.h"

#define STATE_DIM 12
#define ACTION_DIM 8
#define HIDDEN_DIM 64
#define BATCH_SIZE 32

double compute_log_prob(double x, double mean, double logvar) {
    const double LOG_2PI = 1.8378770664093453;
    double std = exp(0.5 * logvar);
    double z = (x - mean) / std;
    return -0.5 * (LOG_2PI + logvar + z * z);
}

void process_rollout(const char* filename, Net* policy, double** act, double** grad) {
    Data* rollout = load_csv(filename, STATE_DIM + 4, 1);
    if(!rollout) return;
    
    // Forward pass through the network for each state
    for(int t = 0; t < rollout->n; t++) {
        fwd(policy, rollout->X[t], act);
        
        // Compute policy gradients for means and logvars
        double G = rollout->y[t][0];  // Return value
        
        for(int i = 0; i < 4; i++) {
            double mean = fabs(act[4][i]) * 50.0;
            double logvar = act[4][i + 4];
            double action = rollout->X[t][STATE_DIM + i];
            
            // Gradient for mean
            double dmean = (action - mean) / (exp(logvar));
            grad[4][i] = G * dmean * (action > 0 ? 50.0 : -50.0);
            
            // Gradient for logvar
            double dlogvar = 0.5 * ((action - mean) * (action - mean) / exp(logvar) - 1.0);
            grad[4][i + 4] = G * dlogvar;
        }
        
        bwd(policy, act, grad);
    }
    
    free_data(rollout);
}

int main(int argc, char** argv) {
    if(argc != 2) {
        printf("Usage: %s <policy_file>\n", argv[0]);
        return 1;
    }
    
    // Initialize or load policy network
    int layers[] = {STATE_DIM, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, ACTION_DIM};
    Net* policy = load_weights(argv[1]);
    if(!policy) {
        printf("Failed to load policy from %s\n", argv[1]);
        return 1;
    }
    policy->lr = 1e-4;
    
    // Allocate memory for activations and gradients
    double** act = malloc(5 * sizeof(double*));
    double** grad = malloc(5 * sizeof(double*));
    for(int i = 0; i < 5; i++) {
        act[i] = malloc(policy->sz[i] * sizeof(double));
        grad[i] = calloc(policy->sz[i], sizeof(double));
    }
    
    // Process all rollout files in current directory
    DIR* dir = opendir(".");
    struct dirent* entry;
    int processed = 0;
    
    while((entry = readdir(dir)) != NULL) {
        if(strstr(entry->d_name, "_rollout.csv")) {
            process_rollout(entry->d_name, policy, act, grad);
            processed++;
            
            if(processed % BATCH_SIZE == 0) {
                printf("Processed %d rollouts\n", processed);
            }
        }
    }
    closedir(dir);
    
    // Save updated policy
    save_weights(argv[1], policy);
    printf("Updated policy saved to %s\n", argv[1]);
    
    // Cleanup
    for(int i = 0; i < 5; i++) {
        free(act[i]);
        free(grad[i]);
    }
    free(act);
    free(grad);
    free_net(policy);
    
    return 0;
}