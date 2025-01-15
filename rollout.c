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

double collect_rollout(Sim* sim, Net* policy, int rollout_num) {
    double* states = malloc(STATE_DIM * sizeof(double));
    double** act = malloc(5 * sizeof(double*));
    for(int i = 0; i < 5; i++) {
        act[i] = malloc(policy->sz[i] * sizeof(double));
    }
    
    // Initialize Data structure for rollout
    Data* rollout = malloc(sizeof(Data));
    rollout->fx = STATE_DIM + 4;  // 12 state dimensions + 4 actions
    rollout->fy = 1;              // return value
    rollout->n = 0;               // will be filled during collection
    rollout->X = malloc(MAX_STEPS * sizeof(double*));
    rollout->y = malloc(MAX_STEPS * sizeof(double*));
    rollout->headers = malloc((rollout->fx + rollout->fy) * sizeof(char*));
    const char* header_names[] = {
        "pos_x", "pos_y", "pos_z",
        "vel_x", "vel_y", "vel_z",
        "ang_vel_x", "ang_vel_y", "ang_vel_z",
        "roll", "pitch", "yaw",
        "action1", "action2", "action3", "action4",
        "return"
    };
    for(int i = 0; i < rollout->fx + rollout->fy; i++) {
        rollout->headers[i] = strdup(header_names[i]);
    }

    reset_quad(sim->quad, 
              TARGET_POS[0] + ((double)rand()/RAND_MAX - 0.5) * 0.2,
              TARGET_POS[1] + ((double)rand()/RAND_MAX - 0.5) * 0.2,
              TARGET_POS[2] + ((double)rand()/RAND_MAX - 0.5) * 0.2);
    
    double t_physics = 0.0, t_control = 0.0;
    double* rewards = malloc(MAX_STEPS * sizeof(double));
    
    while(rollout->n < MAX_STEPS && !is_terminated(sim->quad)) {
        update_quad(sim->quad, DT_PHYSICS);
        t_physics += DT_PHYSICS;
        
        if(t_control <= t_physics) {
            rollout->X[rollout->n] = malloc(rollout->fx * sizeof(double));
            rollout->y[rollout->n] = malloc(rollout->fy * sizeof(double));
            
            // Store state
            get_state(sim->quad, states);
            memcpy(rollout->X[rollout->n], states, STATE_DIM * sizeof(double));
            
            // Get and store actions
            fwd(policy, states, act);
            for(int i = 0; i < 4; i++) {
                double mean = fabs(act[4][i]) * 50.0;
                double logvar = act[4][i + 4];
                double std = exp(0.5 * logvar);
                double noise = sqrt(-2.0 * log((double)rand()/RAND_MAX)) * cos(2.0 * M_PI * (double)rand()/RAND_MAX);
                double action = mean + std * noise;
                
                sim->quad->omega_next[i] = action;
                rollout->X[rollout->n][STATE_DIM + i] = action;
            }
            
            rewards[rollout->n] = compute_reward(sim->quad);
            rollout->n++;
            t_control += DT_CONTROL;
        }
    }
    
    // Compute returns and store in y
    double G = 0.0;
    double initial_return = 0.0;
    for(int i = rollout->n-1; i >= 0; i--) {
        G = rewards[i] + GAMMA * G;
        rollout->y[i][0] = G;
        if(i == 0) initial_return = G;
    }
    
    // Save rollout to CSV
    char filename[64];
    sprintf(filename, "%d_rollout.csv", rollout_num);
    save_csv(filename, rollout);
    
    // Cleanup
    free(states);
    free(rewards);
    for(int i = 0; i < 5; i++) free(act[i]);
    free(act);
    free_data(rollout);
    
    return initial_return;
}

int main(int argc, char** argv) {
    srand(time(NULL));
    
    int layers[] = {STATE_DIM, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, ACTION_DIM};
    Net* policy = argc > 1 ? load_weights(argv[1]) : init_net(5, layers);
    if(!policy) return 1;
    
    Sim* sim = init_sim(false);
    
    double sum_returns = 0.0, sum_squared = 0.0;
    double min_return = 1e9, max_return = -1e9;
    
    for(int i = 0; i < NUM_ROLLOUTS; i++) {
        double ret = collect_rollout(sim, policy, i);
        sum_returns += ret;
        sum_squared += ret * ret;
        min_return = fmin(min_return, ret);
        max_return = fmax(max_return, ret);
    }
    
    double mean = sum_returns / NUM_ROLLOUTS;
    double std = sqrt(sum_squared/NUM_ROLLOUTS - mean*mean);
    printf("Rollouts [n=%d]: %.2f ± %.2f (min: %.2f, max: %.2f)\n", 
           NUM_ROLLOUTS, mean, std, min_return, max_return);
    
    char filename[64];
    if(argc > 1) {
        save_weights(argv[1], policy);
    } else {
        strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_policy.bin", localtime(&(time_t){time(NULL)}));
        save_weights(filename, policy);
    }
    
    free_net(policy);
    free_sim(sim);
    return 0;
}