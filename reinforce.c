#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include "grad/grad.h"

#define STATE_DIM 12
#define ACTION_DIM 8
#define MAX_STEPS 1000
#define MAX_ROLLOUTS 100

typedef struct {
    int n;                    // number of timesteps
    double** actions;         // actual actions taken
    double** means;          // means from policy
    double** logvars;        // logvars from policy
    double* returns;         // returns for each timestep
} Rollout;

Rollout* load_rollout(const char* filename) {
    FILE* f = fopen(filename, "r");
    if(!f) return NULL;

    Rollout* r = malloc(sizeof(Rollout));
    r->actions = malloc(MAX_STEPS * sizeof(double*));
    r->means = malloc(MAX_STEPS * sizeof(double*));
    r->logvars = malloc(MAX_STEPS * sizeof(double*));
    r->returns = malloc(MAX_STEPS * sizeof(double));
    
    char line[512];
    fgets(line, sizeof(line), f);  // Skip header
    
    r->n = 0;
    while(r->n < MAX_STEPS && fgets(line, sizeof(line), f)) {
        r->actions[r->n] = malloc(4 * sizeof(double));
        r->means[r->n] = malloc(4 * sizeof(double));
        r->logvars[r->n] = malloc(4 * sizeof(double));
        
        char* token = strtok(line, ",");
        for(int i = 0; i < 4; i++) {
            r->actions[r->n][i] = atof(token);
            token = strtok(NULL, ",");
        }
        for(int i = 0; i < 4; i++) {
            r->means[r->n][i] = atof(token);
            token = strtok(NULL, ",");
        }
        for(int i = 0; i < 4; i++) {
            r->logvars[r->n][i] = atof(token);
            token = strtok(NULL, ",");
        }
        r->returns[r->n] = atof(token);
        r->n++;
    }
    fclose(f);
    return r;
}

void free_rollout(Rollout* r) {
    for(int i = 0; i < r->n; i++) {
        free(r->actions[i]);
        free(r->means[i]);
        free(r->logvars[i]);
    }
    free(r->actions);
    free(r->means);
    free(r->logvars);
    free(r->returns);
    free(r);
}

void update_policy(Net* policy, Rollout** rollouts, int num_rollouts) {
    double** act = malloc(5 * sizeof(double*));
    double** grad = malloc(5 * sizeof(double*));
    double** grad_acc = malloc(5 * sizeof(double*));
    
    for(int i = 0; i < 5; i++) {
        act[i] = malloc(policy->sz[i] * sizeof(double));
        grad[i] = malloc(policy->sz[i] * sizeof(double));
        grad_acc[i] = malloc(policy->sz[i] * sizeof(double));
    }
    
    double total_grad_norm = 0.0;  // Track gradient magnitude
    
    for(int r = 0; r < num_rollouts; r++) {
        for(int i = 0; i < 5; i++) {
            memset(grad_acc[i], 0, policy->sz[i] * sizeof(double));
        }
        
        for(int t = 0; t < rollouts[r]->n; t++) {
            for(int j = 0; j < 4; j++) {
                double mean = rollouts[r]->means[t][j];
                double logvar = rollouts[r]->logvars[t][j];
                double std = exp(0.5 * logvar);
                double action = rollouts[r]->actions[t][j];
                double G = rollouts[r]->returns[t];
                
                grad_acc[4][j] += (action - mean) * G / (std * std) / 50.0;
                grad_acc[4][j + 4] += 0.5 * ((action - mean) * (action - mean) / (std * std) - 1.0) * G;
            }
        }
        
        // Calculate gradient norm before applying update
        double grad_norm = 0.0;
        for(int j = 0; j < policy->sz[4]; j++) {
            grad_norm += grad_acc[4][j] * grad_acc[4][j];
        }
        total_grad_norm += sqrt(grad_norm);
        
        memcpy(grad[4], grad_acc[4], policy->sz[4] * sizeof(double));
        bwd(policy, act, grad);
    }
    
    printf("Average gradient norm during policy update: %.6f\n", total_grad_norm / num_rollouts);
    
    for(int i = 0; i < 5; i++) {
        free(act[i]);
        free(grad[i]);
        free(grad_acc[i]);
    }
    free(act);
    free(grad);
    free(grad_acc);
}

int main(int argc, char** argv) {
    if(argc != 2) {
        printf("Usage: %s <policy.bin>\n", argv[0]);
        return 1;
    }
    
    Net* policy = load_weights(argv[1]);
    if(!policy) return 1;
    policy->lr = 1e-6;

    Rollout* rollouts[MAX_ROLLOUTS];
    int num_rollouts = 0;
    
    DIR *dir = opendir(".");
    struct dirent *ent;
    while((ent = readdir(dir)) && num_rollouts < MAX_ROLLOUTS) {
        if(strstr(ent->d_name, "rollout.csv")) {
            rollouts[num_rollouts] = load_rollout(ent->d_name);
            if(rollouts[num_rollouts]) num_rollouts++;
        }
    }
    closedir(dir);
    
    printf("Updating policy with %d rollouts\n", num_rollouts);
    if(num_rollouts == 0) return 1;
    
    update_policy(policy, rollouts, num_rollouts);
    save_weights(argv[1], policy);
    
    for(int i = 0; i < num_rollouts; i++) free_rollout(rollouts[i]);
    free_net(policy);
    
    return 0;
}