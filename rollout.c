#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "grad/grad.h"
#include "sim/sim.h"

#define STATE_DIM 12
#define ACTION_DIM 8
#define HIDDEN_DIM 64
#define MAX_STEPS 1000
#define DT_PHYSICS (1.0/1000.0)
#define DT_CONTROL (1.0/60.0)
#define MAX_DISTANCE 2.0
#define MAX_VELOCITY 5.0
#define MAX_ANGULAR_VELOCITY 5.0
#define NUM_ROLLOUTS 100
#define GAMMA 0.99

const double TARGET_POS[3] = {0.0, 1.0, 0.0};

void get_state(Quad* q, double* state) {
    memcpy(state, q->linear_position_W, 3 * sizeof(double));
    memcpy(state + 3, q->linear_velocity_W, 3 * sizeof(double));
    memcpy(state + 6, q->angular_velocity_B, 3 * sizeof(double));
    state[9] = q->R_W_B[0];   // Roll
    state[10] = q->R_W_B[4];  // Pitch
    state[11] = q->R_W_B[8];  // Yaw
}

double compute_reward(Quad* q) {
    double pos_error = 0.0, vel_error = 0.0, ang_error = 0.0;
    for(int i = 0; i < 3; i++) {
        pos_error += pow(q->linear_position_W[i] - TARGET_POS[i], 2);
        vel_error += pow(q->linear_velocity_W[i], 2);
        ang_error += pow(q->angular_velocity_B[i], 2);
    }
    return -pos_error - 0.1 * vel_error - 0.1 * ang_error + 5.0 * q->R_W_B[4];
}

bool is_terminated(Quad* q) {
    double dist = 0.0, vel = 0.0, ang_vel = 0.0;
    for(int i = 0; i < 3; i++) {
        dist += pow(q->linear_position_W[i] - TARGET_POS[i], 2);
        vel += pow(q->linear_velocity_W[i], 2);
        ang_vel += pow(q->angular_velocity_B[i], 2);
    }
    return sqrt(dist) > MAX_DISTANCE || 
           sqrt(vel) > MAX_VELOCITY || 
           sqrt(ang_vel) > MAX_ANGULAR_VELOCITY ||
           q->R_W_B[4] < 0.0;
}

void collect_rollout(Sim* sim, Net* policy, int rollout_num) {
    double** actions = malloc(MAX_STEPS * sizeof(double*));
    double** means = malloc(MAX_STEPS * sizeof(double*));
    double** logvars = malloc(MAX_STEPS * sizeof(double*));
    double* rewards = malloc(MAX_STEPS * sizeof(double));
    double* states = malloc(STATE_DIM * sizeof(double));
    double** act = malloc(5 * sizeof(double*));
    
    for(int i = 0; i < MAX_STEPS; i++) {
        actions[i] = malloc(4 * sizeof(double));
        means[i] = malloc(4 * sizeof(double));
        logvars[i] = malloc(4 * sizeof(double));
    }
    for(int i = 0; i < 5; i++) act[i] = malloc(policy->sz[i] * sizeof(double));
    
    reset_quad(sim->quad, 
              TARGET_POS[0] + ((double)rand()/RAND_MAX - 0.5) * 0.2,
              TARGET_POS[1] + ((double)rand()/RAND_MAX - 0.5) * 0.2,
              TARGET_POS[2] + ((double)rand()/RAND_MAX - 0.5) * 0.2);
    
    double t_physics = 0.0, t_control = 0.0;
    int step = 0;
    double total_reward = 0.0;
    
    while(step < MAX_STEPS && !is_terminated(sim->quad)) {
        update_quad(sim->quad, DT_PHYSICS);
        t_physics += DT_PHYSICS;
        
        if(t_control <= t_physics) {
            get_state(sim->quad, states);
            fwd(policy, states, act);
            
            for(int i = 0; i < 4; i++) {
                double mean = fabs(act[4][i]) * 50.0;
                double logvar = act[4][i + 4];
                double std = exp(0.5 * logvar);
                double noise = sqrt(-2.0 * log((double)rand()/RAND_MAX)) * cos(2.0 * M_PI * (double)rand()/RAND_MAX);
                sim->quad->omega_next[i] = mean + std * noise;
                
                means[step][i] = mean;
                logvars[step][i] = logvar;
                actions[step][i] = sim->quad->omega_next[i];
            }
            
            rewards[step] = compute_reward(sim->quad);
            total_reward += rewards[step];
            step++;
            t_control += DT_CONTROL;
        }
    }
    
    char filename[64];
    sprintf(filename, "%d_rollout.csv", rollout_num);
    FILE* f = fopen(filename, "w");

    // Write header
    fprintf(f, "action1,action2,action3,action4,");
    fprintf(f, "mean1,mean2,mean3,mean4,");
    fprintf(f, "logvar1,logvar2,logvar3,logvar4,");
    fprintf(f, "return\n");

    // Calculate returns first
    double* returns = malloc(step * sizeof(double));
    double G = 0;
    for(int i = step-1; i >= 0; i--) {
        G = rewards[i] + GAMMA * G;
        returns[i] = G;
    }

    // Write data in chronological order
    for(int i = 0; i < step; i++) {
        for(int j = 0; j < 4; j++) fprintf(f, "%.6f,", actions[i][j]);
        for(int j = 0; j < 4; j++) fprintf(f, "%.6f,", means[i][j]);
        for(int j = 0; j < 4; j++) fprintf(f, "%.6f,", logvars[i][j]);
        fprintf(f, "%.6f\n", returns[i]);
    }
    fclose(f);
    free(returns);

    printf("\rRollout %d: %d steps, reward: %.3f", rollout_num, step, total_reward);
    
    for(int i = 0; i < MAX_STEPS; i++) {
        free(actions[i]); free(means[i]); free(logvars[i]);
    }
    free(actions); free(means); free(logvars); free(rewards); free(states);
    for(int i = 0; i < 5; i++) free(act[i]);
    free(act);
}

int main(int argc, char** argv) {
    srand(time(NULL));
    
    int layers[] = {STATE_DIM, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, ACTION_DIM};
    Net* policy = argc > 1 ? load_weights(argv[1]) : init_net(5, layers);
    if(!policy) return 1;
    
    Sim* sim = init_sim(false);
    for(int i = 0; i < NUM_ROLLOUTS; i++) collect_rollout(sim, policy, i);
    printf("\n");
    
    char filename[64];
    if(argc > 1) save_weights(argv[1], policy);
    else {
        strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_policy.bin", localtime(&(time_t){time(NULL)}));
        save_weights(filename, policy);
    }
    
    free_net(policy);
    free_sim(sim);
    return 0;
}