#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include "grad/grad.h"
#include "sim/sim.h"

// Time steps for physics and control loops
#define DT_PHYSICS (1.0/1000.0)  // 1kHz physics simulation
#define DT_CONTROL (1.0/60.0)    // 60Hz control loop

// Termination conditions
#define MAX_DISTANCE 2.0
#define MAX_VELOCITY 5.0
#define MAX_ANGULAR_VELOCITY 5.0

// Network architecture
#define STATE_DIM 12             // [pos(3), vel(3), ang_vel(3), R_diag(3)]
#define HIDDEN_DIM 64
#define ACTION_DIM 8             // [mean(4), log_std(4)] for rotor speeds

// Training parameters
#define MAX_STEPS 1000
#define NUM_ROLLOUTS 50

// RL and policy parameters
#define GAMMA 0.999              // Discount factor for future rewards
#define ALPHA 1e-9              // Entropy bonus coefficient
#define MAX_STD 3.0             // Maximum policy standard deviation
#define MIN_STD 1e-5            // Minimum policy standard deviation
#define TARGET_KL 1e-3          // Target KL divergence between old and new policy
#define KL_PENALTY_COEFFICIENT 1e-5  // Coefficient for KL penalty when exceeding target
#define MAX_LR 2e-3            // Maximum learning rate

const double TARGET_POS[3] = {0.0, 1.0, 0.0};  // Target hover position

// Extract state vector from quadrotor state
// state = [position, velocity, angular_velocity, R_diagonal]
void get_state(Quad* q, double* state) {
    memcpy(state, q->linear_position_W, 3 * sizeof(double));
    memcpy(state + 3, q->linear_velocity_W, 3 * sizeof(double));
    memcpy(state + 6, q->angular_velocity_B, 3 * sizeof(double));
    state[9] = q->R_W_B[0];   // R11
    state[10] = q->R_W_B[4];  // R22
    state[11] = q->R_W_B[8];  // R33
}

// Compute reward based on:
// - Distance from target position: ||p - p_target||
// - Linear velocity (should be zero for hovering): ||v||
// - Angular velocity (should be zero for hovering): ||ω||
// - Orientation error (should be upright): |1 - R22|
// R = exp(-error) ensures reward is always positive and maximized at perfect hover
double compute_reward(Quad* q) {
    double pos_error = 0.0;
    for(int i = 0; i < 3; i++) {
        pos_error += pow(q->linear_position_W[i] - TARGET_POS[i], 2);
    }
    pos_error = sqrt(pos_error);
    
    double vel_magnitude = 0.0;
    for(int i = 0; i < 3; i++) {
        vel_magnitude += pow(q->linear_velocity_W[i], 2);
    }
    vel_magnitude = sqrt(vel_magnitude);
    
    double ang_vel_magnitude = 0.0;
    for(int i = 0; i < 3; i++) {
        ang_vel_magnitude += pow(q->angular_velocity_B[i], 2);
    }
    ang_vel_magnitude = sqrt(ang_vel_magnitude);
    
    // R22 should be 1.0 for upright orientation
    double orientation_error = fabs(1.0 - q->R_W_B[4]);
    
    double total_error = (pos_error * 2.0) + 
                        (vel_magnitude * 1.0) + 
                        (ang_vel_magnitude * 0.5) + 
                        (orientation_error * 2.0);
    
    return exp(-total_error);
}

// Check termination conditions:
// - Distance from target exceeds threshold
// - Velocity exceeds threshold
// - Angular velocity exceeds threshold
// - Quadrotor is inverted (R22 < 0)
bool is_terminated(Quad* q) {
    double dist = 0.0, vel = 0.0, ang_vel = 0.0;
    for(int i = 0; i < 3; i++) {
        dist += pow(q->linear_position_W[i] - TARGET_POS[i], 2);
        vel += pow(q->linear_velocity_W[i], 2);
        ang_vel += pow(q->angular_velocity_B[i], 2);
    }
    return sqrt(dist) > MAX_DISTANCE || sqrt(vel) > MAX_VELOCITY || 
           sqrt(ang_vel) > MAX_ANGULAR_VELOCITY || q->R_W_B[4] < 0.0;
}

// Compute KL divergence between two Gaussian distributions
// KL(p||q) = log(σ_q/σ_p) + (σ_p² + (μ_p - μ_q)²)/(2σ_q²) - 1/2
// This measures how different the new policy is from the old policy
double compute_kl_divergence(double old_mean, double old_std, double new_mean, double new_std) {
    const double eps = 1e-9;
    const double max_kl = 10.0;  // Clip KL to reasonable range
    
    double var_old = old_std * old_std + eps;
    double var_new = new_std * new_std + eps;
    double kl = log(new_std/old_std) + 
                (var_old + (old_mean - new_mean) * (old_mean - new_mean)) / (2.0 * var_new) - 
                0.5;
    return fmin(max_kl, fmax(-max_kl, kl));
}

// Collect a single rollout using the current policy
// Returns number of steps before termination
int collect_rollout(Sim* sim, Net* policy, double** act, double** states, double** actions, double* rewards) {
    // Initialize quadrotor near target with small random offset
    reset_quad(sim->quad, 
        TARGET_POS[0] + ((double)rand()/RAND_MAX - 0.5) * 0.2,
        TARGET_POS[1] + ((double)rand()/RAND_MAX - 0.5) * 0.2, 
        TARGET_POS[2] + ((double)rand()/RAND_MAX - 0.5) * 0.2
    );
    
    double t_physics = 0.0, t_control = 0.0;
    int steps = 0;
    
    while(steps < MAX_STEPS && !is_terminated(sim->quad)) {
        update_quad(sim->quad, DT_PHYSICS);
        t_physics += DT_PHYSICS;
        
        if(t_control <= t_physics) {
            get_state(sim->quad, states[steps]);
            fwd(policy, states[steps], act);
            
            // Sample actions from Gaussian policy
            // π(a|s) = N(μ(s), σ(s)²)
            for(int i = 0; i < 4; i++) {
                // Get standard deviation using squash function:
                // σ = ((max + min)/2) + ((max - min)/2) * tanh(x)
                double std = squash(act[4][i + 4], MIN_STD, MAX_STD);
                
                // Compute dynamic bounds for mean based on current std
                // Ensures x = μ ± 4σ stays within [OMEGA_MIN, OMEGA_MAX]
                // P(|x - μ| > 4σ) ≈ 6.334e-5 for Gaussian
                double safe_margin = 4.0 * std;
                double mean_min = OMEGA_MIN + safe_margin;
                double mean_max = OMEGA_MAX - safe_margin;
                
                // Get mean using squash with dynamic bounds
                double mean = squash(act[4][i], mean_min, mean_max);
                
                // Sample from N(μ, σ²) using Box-Muller transform:
                // If U₁,U₂ ~ Uniform(0,1)
                // Then √(-2ln(U₁))cos(2πU₂) ~ N(0,1)
                // And μ + σ√(-2ln(U₁))cos(2πU₂) ~ N(μ,σ²)
                double u1 = (double)rand()/RAND_MAX;
                double u2 = (double)rand()/RAND_MAX;
                double noise = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                
                actions[steps][i] = mean + std * noise;
                sim->quad->omega_next[i] = actions[steps][i];
            }
            
            rewards[steps] = compute_reward(sim->quad);
            steps++;
            t_control += DT_CONTROL;
        }
    }
    
    // Compute discounted returns
    // G_t = r_t + γr_{t+1} + γ²r_{t+2} + ...
    // Using backward recursion: G_t = r_t + γG_{t+1}
    double G = 0.0;
    for(int i = steps-1; i >= 0; i--) {
        rewards[i] = G = rewards[i] + GAMMA * G;
    }
    
    return steps;
}

// Update policy using collected rollout data
// Uses KL divergence penalty to prevent too large policy updates
void update_policy(Net* policy, double** states, double** actions, double* returns, int steps, 
                  double** act, double** grad, double* mean_kl, double* max_kl) {
    // Store old policy parameters for KL computation
    double old_means[steps][4];
    double old_stds[steps][4];
    double total_kl = 0.0;
    *max_kl = 0.0;
    
    // First pass to store old policy parameters
    for(int t = 0; t < steps; t++) {
        fwd(policy, states[t], act);
        for(int i = 0; i < 4; i++) {
            old_stds[t][i] = squash(act[4][i + 4], MIN_STD, MAX_STD);
            double safe_margin = 4.0 * old_stds[t][i];
            double mean_min = OMEGA_MIN + safe_margin;
            double mean_max = OMEGA_MAX - safe_margin;
            old_means[t][i] = squash(act[4][i], mean_min, mean_max);
        }
    }

    // Policy update pass
    for(int t = 0; t < steps; t++) {
        fwd(policy, states[t], act);
        
        double step_kl = 0.0;
        for(int i = 0; i < 4; i++) {
            double std = squash(act[4][i + 4], MIN_STD, MAX_STD);
            double safe_margin = 4.0 * std;
            double mean_min = OMEGA_MIN + safe_margin;
            double mean_max = OMEGA_MAX - safe_margin;
            double mean = squash(act[4][i], mean_min, mean_max);
            
            // Compute KL divergence from old to new policy
            double kl = compute_kl_divergence(old_means[t][i], old_stds[t][i], mean, std);
            step_kl += kl;

            // Compute log probability of taken action under current policy
            // log(p(x)) = -0.5(log(2π) + 2log(σ) + ((x-μ)/σ)²)
            double z = (actions[t][i] - mean) / std;
            double log_prob = -0.5 * (1.8378770664093453 + 2.0 * log(std) + z * z);
            
            // Compute entropy bonus
            // H = 0.5(log(2πe) + 2log(σ))
            double entropy = 0.5 * (2.837877066 + 2.0 * log(std));
            
            // Add KL penalty if divergence exceeds target
            double kl_penalty = KL_PENALTY_COEFFICIENT * fmax(0.0, (step_kl - TARGET_KL) / TARGET_KL);
            if (step_kl < 1e-6) {  // Add small push when KL is too low
                kl_penalty = -KL_PENALTY_COEFFICIENT * 0.1;  // Negative penalty to encourage exploration
            }
            
            // Policy gradient with entropy bonus and KL penalty
            // ∇_θ J = E[∇_θ log π_θ(a|s) * (R - b)]
            double dmean = z / std;  // ∂log π/∂μ
            grad[4][i] = (returns[t] * log_prob + ALPHA * entropy - kl_penalty * kl) * 
                        dmean * dsquash(act[4][i], mean_min, mean_max);
            
            // Standard deviation gradient includes direct terms and mean coupling
            double dstd_direct = (z * z - 1.0) / std;  // ∂log π/∂σ
            double dmean_dstd = -4.0 * dsquash(act[4][i], mean_min, mean_max);
            double dstd = dstd_direct + (z / std) * dmean_dstd;
            
            grad[4][i + 4] = (returns[t] * log_prob * dstd + 
                             ALPHA * (1.0 / std) - 
                             kl_penalty * kl) * 
                            dsquash(act[4][i + 4], MIN_STD, MAX_STD);
        }

        total_kl += step_kl;
        *max_kl = fmax(*max_kl, step_kl);

        bwd(policy, act, grad);
    }
    
    *mean_kl = total_kl / steps;
}

int main(int argc, char** argv) {
    if(argc != 2 && argc != 3) {
        printf("Usage: %s <num_iterations> [initial_weights.bin]\n", argv[0]);
        return 1;
    }

    srand(time(NULL));
    
    Net* net;
    if(argc == 3) {
        net = load_weights(argv[2], adamw);
    } else {
        int layers[] = {STATE_DIM, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, ACTION_DIM};
        net = init_net(5, layers, adamw);
    }
    net->lr = 5e-4;
    
    Sim* sim = init_sim("", false);
    double** act = malloc(5 * sizeof(double*));
    double** grad = malloc(5 * sizeof(double*));
    
    for(int i = 0; i < 5; i++) {
        act[i] = malloc(net->sz[i] * sizeof(double));
        grad[i] = calloc(net->sz[i], sizeof(double));
    }

    int iterations = atoi(argv[1]);
    double best_return = -1e30;
    double initial_best = -1e30;
    struct timeval start_time, current_time;
    gettimeofday(&start_time, NULL);
    
    // Maximum possible return for perfect hovering
    // Sum of geometric series: 1 + γ + γ² + ... + γ^(T-1)
    double theoretical_max = (1.0 - pow(GAMMA + 1e-15, MAX_STEPS))/(1.0 - (GAMMA + 1e-15));
    
    for(int iter = 0; iter < iterations; iter++) {
        double sum_returns = 0.0;
        double iter_mean_kl = 0.0;
        double iter_max_kl = 0.0;

        // Allocate memory for rollout data
        double* all_states[NUM_ROLLOUTS][MAX_STEPS];
        double* all_actions[NUM_ROLLOUTS][MAX_STEPS];
        double all_rewards[NUM_ROLLOUTS][MAX_STEPS];
        int rollout_steps[NUM_ROLLOUTS];

        for(int r = 0; r < NUM_ROLLOUTS; r++) {
            for(int i = 0; i < MAX_STEPS; i++) {
                all_states[r][i] = malloc(STATE_DIM * sizeof(double));
                all_actions[r][i] = malloc(4 * sizeof(double));
            }
        }

        // Collect rollouts using current policy
        for(int r = 0; r < NUM_ROLLOUTS; r++) {
            rollout_steps[r] = collect_rollout(sim, net, act, 
                                             all_states[r], 
                                             all_actions[r], 
                                             all_rewards[r]);
            sum_returns += all_rewards[r][0];
        }

        // Update policy using collected data
        for(int r = 0; r < NUM_ROLLOUTS; r++) {
            double rollout_mean_kl = 0.0;
            double rollout_max_kl = 0.0;
            update_policy(net, all_states[r], all_actions[r], 
                         all_rewards[r], rollout_steps[r], act, grad,
                         &rollout_mean_kl, &rollout_max_kl);
            iter_mean_kl += rollout_mean_kl;
            iter_max_kl = fmax(iter_max_kl, rollout_max_kl);
        }
        iter_mean_kl /= NUM_ROLLOUTS;

        // Adapt learning rate based on mean KL across all rollouts
        if(iter_mean_kl > 2.0 * TARGET_KL) {
            net->lr *= 0.95;
        } else if(iter_mean_kl < 0.5 * TARGET_KL) {
            net->lr *= 1.02;
        }
        net->lr = fmax(1e-6, fmin(MAX_LR, net->lr));

        // Cleanup rollout data
        for(int r = 0; r < NUM_ROLLOUTS; r++) {
            for(int i = 0; i < MAX_STEPS; i++) {
                free(all_states[r][i]);
                free(all_actions[r][i]);
            }
        }

        // Update statistics
        double mean_return = sum_returns / NUM_ROLLOUTS;
        if(mean_return > best_return) {
            best_return = mean_return;
        }

        if(iter == 0) {
            initial_best = best_return;
        }

        // Compute training progress metrics
        gettimeofday(&current_time, NULL);
        double elapsed = (current_time.tv_sec - start_time.tv_sec) + 
                        (current_time.tv_usec - start_time.tv_usec) / 1000000.0;
        
        double initial_percentage = (initial_best / theoretical_max) * 100.0;
        double current_percentage = (best_return / theoretical_max) * 100.0;
        double percentage_rate = (current_percentage - initial_percentage) / elapsed;

        // Display progress
        printf("Iter %d/%d | Return: %.2f/%.2f (%.1f%%) | Best: %.2f | KL: %.4f/%.4f | Rate: %.3f %%/s | lr: %.2e\n", 
               iter+1, iterations, mean_return, theoretical_max, 
               (mean_return/theoretical_max) * 100.0, best_return,
               iter_mean_kl, iter_max_kl, percentage_rate, net->lr);
    }
    printf("\n");

    // Save final policy
    char final_weights[64];
    strftime(final_weights, sizeof(final_weights), "%Y%m%d_%H%M%S_policy.bin", 
             localtime(&(time_t){time(NULL)}));
    save_weights(final_weights, net);
    printf("Final weights saved to: %s\n", final_weights);

    // Cleanup
    for(int i = 0; i < 5; i++) {
        free(act[i]);
        free(grad[i]);
    }
    free(act);
    free(grad);
    free_net(net);
    free_sim(sim);

    return 0;
}