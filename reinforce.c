#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include "grad/grad.h"
#include "grad/data.h"

#define STATE_DIM 12
#define ACTION_DIM 8

void update_policy(Net* policy, Data** rollouts, int num_rollouts) {
    double** act = malloc(5 * sizeof(double*));
    double** grad = malloc(5 * sizeof(double*));
    
    for(int i = 0; i < 5; i++) {
        act[i] = malloc(policy->sz[i] * sizeof(double));
        grad[i] = malloc(policy->sz[i] * sizeof(double));
    }
    
    for(int r = 0; r < num_rollouts; r++) {
        for(int t = 0; t < rollouts[r]->n; t++) {
            double G = rollouts[r]->y[t][ACTION_DIM];
            
            fwd(policy, rollouts[r]->X[t], act);
            
            // Compute policy gradients
            for(int j = 0; j < 4; j++) {
                double mean = fabs(act[4][j]) * 50.0;
                double logvar = act[4][j + 4];
                double std = exp(0.5 * logvar);
                double action = rollouts[r]->y[t][j];
                
                grad[4][j] = (action - mean) * G / (std * std) / 50.0 * (act[4][j] >= 0 ? 1.0 : -1.0);
                grad[4][j + 4] = 0.5 * ((action - mean) * (action - mean) / (std * std) - 1.0) * G;
            }
            
            bwd(policy, act, grad);
        }
    }
    
    for(int i = 0; i < 5; i++) {
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
    
    Net* policy = load_weights(argv[1]);
    if(!policy) return 1;

    Data* rollouts[100];
    int num_rollouts = 0;
    
    DIR *dir = opendir(".");
    struct dirent *ent;
    while((ent = readdir(dir)) && num_rollouts < 1000) {
        if(strstr(ent->d_name, "rollout.csv")) {
            rollouts[num_rollouts] = load_csv(ent->d_name);
            if(rollouts[num_rollouts]) num_rollouts++;
        }
    }
    closedir(dir);
    
    printf("Loaded %d rollouts\n", num_rollouts);
    if(num_rollouts == 0) return 1;
    
    update_policy(policy, rollouts, num_rollouts);
    save_weights(argv[1], policy);
    
    for(int i = 0; i < num_rollouts; i++) free_data(rollouts[i]);
    free_net(policy);
    
    return 0;
}