#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include "grad/grad.h"
#include "grad/data.h"

#define STATE_DIM 12
#define ACTION_DIM 8  // 4 means + 4 logvars

Data* load_rollout(const char* filename) {
    Data* data = load_csv(filename);
    if(!data) printf("Failed to load %s\n", filename);
    return data;
}

void update_policy(Net* policy, Data** rollouts, int num_rollouts) {
    double** act = malloc(4 * sizeof(double*));
    double* grad = malloc(ACTION_DIM * sizeof(double));
    
    for(int i = 0; i < 4; i++) {
        act[i] = malloc(policy->sz[i] * sizeof(double));
    }
    
    // Process all trajectories
    for(int i = 0; i < num_rollouts; i++) {
        for(int t = 0; t < rollouts[i]->n; t++) {
            // Get return (discounted reward-to-go)
            double G = rollouts[i]->y[t][ACTION_DIM + 1];
            
            // Forward pass
            fwd(policy, rollouts[i]->X[t], act);
            
            // Compute policy gradients
            for(int j = 0; j < 4; j++) {
                double mean = act[3][j];
                double logvar = act[3][j + 4];
                double std = exp(0.5 * logvar);
                double action = rollouts[i]->y[t][j];
                
                // Mean gradient
                grad[j] = (action - mean) * G / (std * std);
                
                // Logvar gradient
                grad[j + 4] = 0.5 * ((action - mean) * (action - mean) / 
                             (std * std) - 1.0) * G;
            }
            
            // Update policy
            bwd(policy, act, grad);
        }
    }
    
    for(int i = 0; i < 4; i++) free(act[i]);
    free(act);
    free(grad);
}

int main(int argc, char** argv) {
    if(argc != 2) {
        printf("Usage: %s <policy.bin>\n", argv[0]);
        return 1;
    }
    
    Net* policy = load_weights(argv[1]);
    if(!policy) return 1;
    
    Data* rollouts[100];
    int num_rollouts = 0;
    
    DIR *dir = opendir(".");
    struct dirent *ent;
    while((ent = readdir(dir)) && num_rollouts < 100) {
        if(strstr(ent->d_name, "rollout.csv")) {
            if((rollouts[num_rollouts] = load_rollout(ent->d_name)))
                num_rollouts++;
        }
    }
    closedir(dir);
    
    printf("Loaded %d rollouts\n", num_rollouts);
    if(num_rollouts == 0) return 1;
    
    update_policy(policy, rollouts, num_rollouts);
    save_weights(policy, argv[1]);
    
    for(int i = 0; i < num_rollouts; i++) free_data(rollouts[i]);
    free_net(policy);
    
    return 0;
}