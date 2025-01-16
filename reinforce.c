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
#define NUM_ROLLOUTS 100
#define NUM_ITERATIONS 2000
#define GAMMA 0.99
#define ALPHA 0.01

#define DT_PHYSICS (1.0/1000.0)
#define DT_CONTROL (1.0/60.0)
#define MAX_DISTANCE 2.0
#define MAX_VELOCITY 5.0
#define MAX_ANGULAR_VELOCITY 5.0

const double TARGET_POS[3] = {0.0, 1.0, 0.0};

void get_state(Quad* q, double* state) {
    memcpy(state, q->linear_position_W, 3 * sizeof(double));
    memcpy(state + 3, q->linear_velocity_W, 3 * sizeof(double));
    memcpy(state + 6, q->angular_velocity_B, 3 * sizeof(double));
    state[9] = q->R_W_B[0];
    state[10] = q->R_W_B[4];
    state[11] = q->R_W_B[8];
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

int collect_rollout(Sim* sim, Net* policy, double** act, double** states, double** actions, double* rewards) {
    // Initialize quadcopter at slightly random position near target
    reset_quad(sim->quad, 
        TARGET_POS[0] + ((double)rand()/RAND_MAX - 0.5) * 0.2,
        TARGET_POS[1] + ((double)rand()/RAND_MAX - 0.5) * 0.2, 
        TARGET_POS[2] + ((double)rand()/RAND_MAX - 0.5) * 0.2
    );
    
    double t_physics = 0.0, t_control = 0.0;
    int steps = 0;
    
    while(steps < MAX_STEPS && !is_terminated(sim->quad)) {
        // 1. Update physics simulation
        update_quad(sim->quad, DT_PHYSICS);
        t_physics += DT_PHYSICS;
        
        // 2. Control loop runs at lower frequency than physics
        if(t_control <= t_physics) {
            // 3. Get current state observation
            get_state(sim->quad, states[steps]);
            
            // 4. Forward pass through policy network
            fwd(policy, states[steps], act);
            
            // 5. Sample actions from policy distribution
            for(int i = 0; i < 4; i++) {
                // Get mean (constrained between 30 and 70)
                double mean = 50.0 + 20.0 * tanh(act[4][i]);
                
                // Get standard deviation from log variance (using direct log variance with soft constraint)
                double logvar = -5.0 * tanh(act[4][i + 4]);
                double std = exp(0.5 * logvar);
                
                // Sample from normal distribution using Box-Muller transform
                double u1 = (double)rand()/RAND_MAX;
                double u2 = (double)rand()/RAND_MAX;
                double noise = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                
                // Store and apply sampled action
                actions[steps][i] = mean + std * noise;
                sim->quad->omega_next[i] = actions[steps][i];
            }
            
            // 6. Compute immediate reward
            rewards[steps] = compute_reward(sim->quad);
            
            steps++;
            t_control += DT_CONTROL;
        }
    }
    
    // 7. Compute discounted returns
    double G = 0.0;
    for(int i = steps-1; i >= 0; i--) {
        rewards[i] = G = rewards[i] + GAMMA * G;
    }
    
    return steps;
}

void update_policy(Net* policy, double** states, double** actions, double* returns, int steps, double** act, double** grad) {
    for(int t = 0; t < steps; t++) {
        // Forward pass through policy network
        fwd(policy, states[t], act);
        
        for(int i = 0; i < 4; i++) {
            // 1. Get mean of action distribution (constrained between 30 and 70)
            double tanh_mean = tanh(act[4][i]);
            double mean = 50.0 + 20.0 * tanh_mean;
            
            // 2. Get log variance of action distribution (using direct log variance with soft constraint)
            double tanh_logvar = tanh(act[4][i + 4]);
            double logvar = -5.0 * tanh_logvar;
            double std = exp(0.5 * logvar);
            
            // 3. Compute normalized action (z-score)
            double z = (actions[t][i] - mean) / std;
            
            // 4. Compute log probability of the action
            // log(p(x)) = -0.5 * (log(2π) + logvar + z²)
            double log_prob = -0.5 * (1.8378770664093453 + logvar + z * z);
            
            // 5. Compute entropy of the Gaussian distribution
            // H = 0.5 * (log(2πe) + logvar)
            double entropy = 0.5 * (2.837877066 + logvar);
            
            // 6. Compute gradient for mean
            // ∂log_prob/∂mean = z/std
            // ∂mean/∂θ = 20 * (1 - tanh²)
            double dmean = z / std;
            double dtanh_mean = 1.0 - tanh_mean * tanh_mean;
            grad[4][i] = (returns[t] * log_prob + ALPHA * entropy) * dmean * 20.0 * dtanh_mean;
            
            // 7. Compute gradient for log variance
            // ∂log_prob/∂logvar = 0.5 * (z² - 1)
            // ∂entropy/∂logvar = 0.5
            // ∂logvar/∂θ = -5.0 * (1 - tanh²)
            double dlogvar = 0.5 * (z * z - 1.0);
            double dtanh_logvar = 1.0 - tanh_logvar * tanh_logvar;
            grad[4][i + 4] = (returns[t] * log_prob * dlogvar + ALPHA * 0.5) * -5.0 * dtanh_logvar;
        }
        
        // Backward pass to update policy parameters
        bwd(policy, act, grad);
    }
}

int main(int argc, char** argv) {
    srand(time(NULL));
    
    int layers[] = {STATE_DIM, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, ACTION_DIM};
    Net* policy = init_net(5, layers, sgd);
    if(!policy) return 1;
    policy->lr = 1e-2;
    
    Sim* sim = init_sim(false);
    
    double** act = malloc(5 * sizeof(double*));
    double** grad = malloc(5 * sizeof(double*));
    double*** states = malloc(NUM_ROLLOUTS * sizeof(double**));
    double*** actions = malloc(NUM_ROLLOUTS * sizeof(double**));
    double** rewards = malloc(NUM_ROLLOUTS * sizeof(double*));
    int* steps = malloc(NUM_ROLLOUTS * sizeof(int));
    
    for(int i = 0; i < 5; i++) {
        act[i] = malloc(policy->sz[i] * sizeof(double));
        grad[i] = calloc(policy->sz[i], sizeof(double));
    }
    
    for(int r = 0; r < NUM_ROLLOUTS; r++) {
        states[r] = malloc(MAX_STEPS * sizeof(double*));
        actions[r] = malloc(MAX_STEPS * sizeof(double*));
        rewards[r] = malloc(MAX_STEPS * sizeof(double));
        for(int i = 0; i < MAX_STEPS; i++) {
            states[r][i] = malloc(STATE_DIM * sizeof(double));
            actions[r][i] = malloc(4 * sizeof(double));
        }
    }

    for(int iter = 1; iter <= NUM_ITERATIONS; iter++) {
        double sum_returns = 0.0, sum_squared = 0.0;
        double min_return = 1e9, max_return = -1e9;
        
        for(int r = 0; r < NUM_ROLLOUTS; r++) {
            steps[r] = collect_rollout(sim, policy, act, states[r], actions[r], rewards[r]);
            double ret = rewards[r][0];
            
            sum_returns += ret;
            sum_squared += ret * ret;
            min_return = fmin(min_return, ret);
            max_return = fmax(max_return, ret);
        }
        
        for(int r = 0; r < NUM_ROLLOUTS; r++) 
            update_policy(policy, states[r], actions[r], rewards[r], steps[r], act, grad);
        
        printf("Iteration %d/%d [n=%d]: %.2f ± %.2f (min: %.2f, max: %.2f)\n", 
               iter, NUM_ITERATIONS, NUM_ROLLOUTS, 
               sum_returns / NUM_ROLLOUTS,
               sqrt(sum_squared/NUM_ROLLOUTS - pow(sum_returns/NUM_ROLLOUTS, 2)),
               min_return, max_return);
    }
    
    char filename[64];
    strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_policy.bin", localtime(&(time_t){time(NULL)}));
    save_weights(filename, policy);
    
    for(int r = 0; r < NUM_ROLLOUTS; r++) {
        for(int i = 0; i < MAX_STEPS; i++) {
            free(states[r][i]);
            free(actions[r][i]);
        }
        free(states[r]); free(actions[r]); free(rewards[r]);
    }
    
    for(int i = 0; i < 5; i++) free(act[i]), free(grad[i]);
    free(states); free(actions); free(rewards); free(steps);
    free(act); free(grad); free_net(policy); free_sim(sim);
    return 0;
}