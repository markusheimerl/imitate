#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "grad/grad.h"
#include "grad/data.h"
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
#define NUM_ROLLOUTS 10
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
    
    // Weighted reward components
    double position_reward = -pos_error;
    double velocity_penalty = -0.1 * vel_error;
    double angular_penalty = -0.1 * ang_error;
    double upright_reward = 5.0 * q->R_W_B[4];  // Reward for being upright
    
    return position_reward + velocity_penalty + angular_penalty + upright_reward;
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
           q->R_W_B[4] < 0.0;  // Terminate if flipped
}

Data* collect_rollout(Sim* sim, Net* policy, int rollout_num) {
    Data* data = malloc(sizeof(Data));
    data->fx = STATE_DIM;
    data->fy = ACTION_DIM + 2;  // +1 for reward, +1 for return
    data->n = 0;
    data->X = malloc(MAX_STEPS * sizeof(double*));
    data->y = malloc(MAX_STEPS * sizeof(double*));
    
    double** act = malloc(4 * sizeof(double*));
    for(int i = 0; i < 4; i++) act[i] = malloc(policy->sz[i] * sizeof(double));
    for(int i = 0; i < MAX_STEPS; i++) {
        data->X[i] = malloc(STATE_DIM * sizeof(double));
        data->y[i] = malloc((ACTION_DIM + 2) * sizeof(double));
    }
    
    // Random initial position near target
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
            // Get current state
            get_state(sim->quad, data->X[step]);
            
            // Get policy output
            fwd(policy, data->X[step], act);
            
            // Store means and logvars
            memcpy(data->y[step], act[3], ACTION_DIM * sizeof(double));
            
            // Sample actions from Gaussian
            for(int i = 0; i < 4; i++) {
                double mean = act[3][i];
                double logvar = act[3][i + 4];
                double std = exp(0.5 * logvar);
                double u1 = (double)rand() / RAND_MAX;
                double u2 = (double)rand() / RAND_MAX;
                sim->quad->omega_next[i] = mean + std * 
                    sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
            }
            
            // Compute and store reward
            double reward = compute_reward(sim->quad);
            data->y[step][ACTION_DIM] = reward;
            total_reward += reward;
            
            step++;
            t_control += DT_CONTROL;
        }
    }
    
    data->n = step;
    
    // Calculate returns (backwards)
    double* returns = malloc(step * sizeof(double));
    returns[step - 1] = data->y[step - 1][ACTION_DIM];
    
    for(int i = step - 2; i >= 0; i--) {
        returns[i] = data->y[i][ACTION_DIM] + GAMMA * returns[i + 1];
    }
    
    // Add returns as last column
    for(int i = 0; i < step; i++) {
        data->y[i][ACTION_DIM + 1] = returns[i];
    }
    
    // Save numbered rollout
    char filename[64];
    sprintf(filename, "%d_rollout.csv", rollout_num);
    const char* header = "px,py,pz,vx,vy,vz,wx,wy,wz,r11,r22,r33,"
                        "m1_mean,m2_mean,m3_mean,m4_mean,"
                        "m1_logvar,m2_logvar,m3_logvar,m4_logvar,reward,return";
    save_csv(filename, data, header);
    
    printf("Rollout %d completed: %d steps, total reward: %.3f\n", 
           rollout_num, step, total_reward);
    
    free(returns);
    for(int i = 0; i < 4; i++) free(act[i]);
    free(act);
    
    return data;
}

int main(int argc, char** argv) {
    srand(time(NULL));
    
    Net* policy;
    if(argc > 1) {
        printf("Loading weights from %s...\n", argv[1]);
        policy = load_weights(argv[1]);
    } else {
        printf("Initializing policy network...\n");
        policy = init_net(4, (int[]){STATE_DIM, HIDDEN_DIM, HIDDEN_DIM, ACTION_DIM});
    }
    if(!policy) return 1;
    
    Sim* sim = init_sim(false);
    
    printf("\nCollecting %d rollouts...\n", NUM_ROLLOUTS);
    for(int i = 0; i < NUM_ROLLOUTS; i++) {
        Data* data = collect_rollout(sim, policy, i);
        free_data(data);
    }

    if(argc > 1) {
        save_weights(policy, argv[1]);
    } else {
        char* policy_filename = get_timestamp_filename("policy.bin");
        save_weights(policy, policy_filename);
        free(policy_filename);
    }
    
    free_net(policy);
    free_sim(sim);
    return 0;
}