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
#define NUM_ITERATIONS 2000

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
    return sqrt(dist) > MAX_DISTANCE || sqrt(vel) > MAX_VELOCITY || sqrt(ang_vel) > MAX_ANGULAR_VELOCITY || q->R_W_B[4] < 0.0;
}

double compute_log_prob(double x, double mean, double logvar) {
    const double LOG_2PI = 1.8378770664093453;
    double std = exp(0.5 * logvar);
    double z = (x - mean) / std;
    return -0.5 * (LOG_2PI + logvar + z * z);
}

int collect_rollout(Sim* sim, Net* policy, double** act, double** states, double** actions, double* rewards) {
    reset_quad(sim->quad, TARGET_POS[0] + ((double)rand()/RAND_MAX - 0.5) * 0.2, TARGET_POS[1] + ((double)rand()/RAND_MAX - 0.5) * 0.2, TARGET_POS[2] + ((double)rand()/RAND_MAX - 0.5) * 0.2);
    
    double t_physics = 0.0, t_control = 0.0;
    int steps = 0;
    
    while(steps < MAX_STEPS && !is_terminated(sim->quad)) {
        update_quad(sim->quad, DT_PHYSICS);
        t_physics += DT_PHYSICS;
        
        if(t_control <= t_physics) {
            get_state(sim->quad, states[steps]);
            
            fwd(policy, states[steps], act);
            for(int i = 0; i < 4; i++) {
                double mean = 50.0 + 20.0 * tanh(act[4][i]);
                double logvar = fmax(fmin(act[4][i + 4], 2.0), -20.0);
                double std = exp(0.5 * logvar);
                double noise = sqrt(-2.0 * log((double)rand()/RAND_MAX)) * cos(2.0 * M_PI * (double)rand()/RAND_MAX);
                double action = mean + std * noise;
                
                sim->quad->omega_next[i] = action;
                actions[steps][i] = action;
            }
            
            rewards[steps] = compute_reward(sim->quad);
            steps++;
            t_control += DT_CONTROL;
        }
    }
    
    double G = 0.0;
    for(int i = steps-1; i >= 0; i--) {
        G = rewards[i] + GAMMA * G;
        rewards[i] = G;
    }
    
    return steps;
}

void update_policy(Net* policy, double** states, double** actions, double* returns, int steps, double** act, double** grad) {
    for(int t = 0; t < steps; t++) {
        fwd(policy, states[t], act);
        
        for(int i = 0; i < 4; i++) {
            double tanh_x = tanh(act[4][i]);
            double mean = 50.0 + 20.0 * tanh_x;
            double logvar = fmax(fmin(act[4][i + 4], 2.0), -20.0);
            double action = actions[t][i];
            
            double log_prob = compute_log_prob(action, mean, logvar);
            
            double dmean = (action - mean) / exp(logvar);
            double dtanh = 1.0 - tanh_x * tanh_x;
            grad[4][i] = log_prob * returns[t] * dmean * 20.0 * dtanh;
            
            double entropy_bonus = -0.01 * (logvar + 1.0);
            grad[4][i + 4] = (log_prob * returns[t] * 0.5 * ((action - mean) * (action - mean) / exp(logvar) - 1.0) + entropy_bonus) * 0.1;
        }
        
        bwd(policy, act, grad);
    }
}

int main(int argc, char** argv) {
    srand(time(NULL));
    
    int layers[] = {STATE_DIM, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, ACTION_DIM};
    Net* policy = init_net(5, layers);
    if(!policy) return 1;
    policy->lr = 5e-7;
    
    Sim* sim = init_sim(false);
    
    double** act = malloc(5 * sizeof(double*));
    double** grad = malloc(5 * sizeof(double*));
    for(int i = 0; i < 5; i++) {
        act[i] = malloc(policy->sz[i] * sizeof(double));
        grad[i] = calloc(policy->sz[i], sizeof(double));
    }
    
    // Allocate memory for each rollout separately
    double*** all_states = malloc(NUM_ROLLOUTS * sizeof(double**));
    double*** all_actions = malloc(NUM_ROLLOUTS * sizeof(double**));
    double** all_rewards = malloc(NUM_ROLLOUTS * sizeof(double*));
    int* steps_per_rollout = malloc(NUM_ROLLOUTS * sizeof(int));
    
    for(int r = 0; r < NUM_ROLLOUTS; r++) {
        all_states[r] = malloc(MAX_STEPS * sizeof(double*));
        all_actions[r] = malloc(MAX_STEPS * sizeof(double*));
        all_rewards[r] = malloc(MAX_STEPS * sizeof(double));
        for(int i = 0; i < MAX_STEPS; i++) {
            all_states[r][i] = malloc(STATE_DIM * sizeof(double));
            all_actions[r][i] = malloc(4 * sizeof(double));
        }
    }

    for(int iteration = 1; iteration <= NUM_ITERATIONS; iteration++) {
        double sum_returns = 0.0, sum_squared = 0.0;
        double min_return = 1e9, max_return = -1e9;
        
        // First collect all rollouts
        for(int r = 0; r < NUM_ROLLOUTS; r++) {
            steps_per_rollout[r] = collect_rollout(sim, policy, act, 
                                                 all_states[r], all_actions[r], all_rewards[r]);
            double ret = all_rewards[r][0];
            
            sum_returns += ret;
            sum_squared += ret * ret;
            min_return = fmin(min_return, ret);
            max_return = fmax(max_return, ret);
        }
        
        // Then update policy for each rollout independently
        for(int r = 0; r < NUM_ROLLOUTS; r++) {
            update_policy(policy, all_states[r], all_actions[r], all_rewards[r], 
                        steps_per_rollout[r], act, grad);
        }
        
        double mean = sum_returns / NUM_ROLLOUTS;
        double std = sqrt(sum_squared/NUM_ROLLOUTS - mean*mean);
        printf("Iteration %d/%d [n=%d]: %.2f ± %.2f (min: %.2f, max: %.2f)\n", 
               iteration, NUM_ITERATIONS, NUM_ROLLOUTS, mean, std, min_return, max_return);
    }
    
    char filename[64];
    strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_policy.bin", localtime(&(time_t){time(NULL)}));
    save_weights(filename, policy);
    
    // Cleanup
    for(int r = 0; r < NUM_ROLLOUTS; r++) {
        for(int i = 0; i < MAX_STEPS; i++) {
            free(all_states[r][i]);
            free(all_actions[r][i]);
        }
        free(all_states[r]);
        free(all_actions[r]);
        free(all_rewards[r]);
    }
    free(all_states);
    free(all_actions);
    free(all_rewards);
    free(steps_per_rollout);
    
    for(int i = 0; i < 5; i++) {
        free(act[i]);
        free(grad[i]);
    }
    free(act); free(grad); free_net(policy); free_sim(sim);
    return 0;
}