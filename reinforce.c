#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include "net.h"
#include "quad.h"

typedef struct {
    double states[MAX_STEPS][STATE_DIM];
    double actions[MAX_STEPS][ACTION_DIM];
    double rewards[MAX_STEPS];
    double returns[MAX_STEPS];
    int length;
} Rollout;

__device__ __host__ double compute_reward(const Quad* q) {
    double distance = sqrt(
        pow(q->linear_position_W[0] - 1.0, 2) +
        pow(q->linear_position_W[1] - 2.0, 2) +
        pow(q->linear_position_W[2] - 1.0, 2)
    );
    return exp(-1.0 * distance);
}

void collect_rollouts(Net* policy, Rollout* rollouts, uint32_t* rng_state) {
    for(int r = 0; r < NUM_ROLLOUTS; r++) {
        // Initialize environment (quadcopter) with starting state s_0
        Quad quad = create_quad(0.0, 1.0, 0.0);
        double t_control = 0.0;
        rollouts[r].length = 0;

        while(rollouts[r].length < MAX_STEPS) {
            // Terminal condition: quadcopter too far from goal
            if (sqrt(
                pow(quad.linear_position_W[0] - 1.0, 2) +
                pow(quad.linear_position_W[1] - 2.0, 2) +
                pow(quad.linear_position_W[2] - 1.0, 2)) > 4.0) {
                break;
            }

            // Physics simulation steps
            update_quad(&quad, DT_PHYSICS);
            t_control += DT_PHYSICS;
            
            // Control at lower frequency
            if (t_control >= DT_CONTROL) {
                int step = rollouts[r].length;
                
                // Get state s_t
                memcpy(rollouts[r].states[step], quad.linear_position_W, 3 * sizeof(double));
                memcpy(rollouts[r].states[step] + 3, quad.angular_velocity_B, 3 * sizeof(double));

                // Forward pass through policy network
                forward_net(policy, rollouts[r].states[step]);
                
                // Sample actions from Gaussian policy
                for(int i = 0; i < 4; i++) {
                    // Get policy parameters
                    double mu = squash(policy->h[2][i], MIN_MEAN, MAX_MEAN);
                    double sigma = squash(policy->h[2][i + 4], MIN_STD, MAX_STD);

                    // Sample from N(0,1) using Box-Muller transform
                    double u1 = random_uniform(rng_state);
                    double u2 = random_uniform(rng_state);
                    double epsilon = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                    
                    // Transform to N(μ, σ²): a = μ + σε
                    rollouts[r].actions[step][i] = mu + sigma * epsilon;
                    
                    // Execute action
                    quad.omega_next[i] = rollouts[r].actions[step][i];
                }
                
                // Get reward
                rollouts[r].rewards[step] = compute_reward(&quad);
                rollouts[r].length++;
                t_control = 0.0;
            }
        }
        
        // Compute discounted returns
        double G = 0.0;
        for(int i = rollouts[r].length-1; i >= 0; i--) {
            G = rollouts[r].rewards[i] + GAMMA * G;
            rollouts[r].returns[i] = G;
        }
    }
}

// Policy gradient update using vanilla REINFORCE algorithm
//
// For a Gaussian policy π_θ(a|s) = N(μ_θ(s), σ_θ(s)²):
// The policy gradient theorem states:
// ∇_θ J(θ) = E_τ[Σ_t ∇_θ log π_θ(a_t|s_t) * R_t]
//
// For a Gaussian policy, the log probability is:
// log π_θ(a|s) = -1/2 * ((a-μ)²/σ² + 2log(σ) + log(2π))
//
// Taking derivatives w.r.t. θ gives two terms:
// ∇_θ log π_θ(a|s) = [
//   ∇_θ μ * (a-μ)/σ² +           (mean gradient)
//   ∇_θ σ * ((a-μ)²-σ²)/σ³       (std gradient)
// ]
void update_policy(Net* policy, Rollout* rollouts) {
    double output_gradients[ACTION_DIM];
    
    for(int r = 0; r < NUM_ROLLOUTS; r++) {
        zero_gradients(policy);

        for(int step = 0; step < rollouts[r].length; step++) {
            // Forward pass to get policy parameters
            forward_net(policy, rollouts[r].states[step]);
            
            for(int i = 0; i < 4; i++) {
                // Get raw network outputs before squashing
                double mean_raw = policy->h[2][i];
                double std_raw = policy->h[2][i + 4];
                
                // Apply squashing functions to get actual policy parameters:
                // μ = ((MAX+MIN)/2) + ((MAX-MIN)/2)*tanh(mean_raw)
                // σ = ((MAX_STD+MIN_STD)/2) + ((MAX_STD-MIN_STD)/2)*tanh(std_raw)
                double mean = squash(mean_raw, MIN_MEAN, MAX_MEAN);
                double std_val = squash(std_raw, MIN_STD, MAX_STD);
                
                // Action deviation from mean
                double delta = rollouts[r].actions[step][i] - mean;

                // Mean gradient term:
                // ∂/∂θ log π = (a-μ)/σ² * ∂μ/∂θ
                // Where ∂μ/∂θ includes the squashing function derivative
                // The negative sign is because we're doing gradient ascent
                output_gradients[i] = -(delta / (std_val * std_val)) * 
                    dsquash(mean_raw, MIN_MEAN, MAX_MEAN) * 
                    rollouts[r].returns[step];

                // Standard deviation gradient term:
                // ∂/∂θ log π = ((a-μ)²-σ²)/σ³ * ∂σ/∂θ
                // This comes from differentiating the log probability of a 
                // Gaussian w.r.t. its standard deviation
                // The ((a-μ)²-σ²) term acts as a natural baseline:
                // - If (a-μ)² > σ², increase σ to make the policy more exploratory
                // - If (a-μ)² < σ², decrease σ to make the policy more deterministic
                output_gradients[i + 4] = -((delta * delta - std_val * std_val) / 
                    (std_val * std_val * std_val)) * 
                    dsquash(std_raw, MIN_STD, MAX_STD) * 
                    rollouts[r].returns[step];
            }
            backward_net(policy, output_gradients);
        }
        update_net(policy);
    }
}

int main(int argc, char** argv) {
    if(argc != 2 && argc != 3) {
        printf("Usage: %s <num_epochs> [initial_weights.bin]\n", argv[0]);
        return 1;
    }

    uint32_t rng_state = time(NULL) ^ getpid();
    Net* net = (Net*)calloc(1, sizeof(Net));
    if(argc == 3){load_net(net, argv[2]);}else{create_net(net, 3e-5, &rng_state);}
    Rollout* rollouts = (Rollout*)calloc(NUM_ROLLOUTS, sizeof(Rollout));

    int num_epochs = atoi(argv[1]);
    double best_return = -1e30;
    double theoretical_max = (1.0 - pow(GAMMA + 1e-15, MAX_STEPS))/(1.0 - (GAMMA + 1e-15));
    struct timeval start_time;
    gettimeofday(&start_time, NULL);
    
    for(int epoch = 0; epoch < num_epochs; epoch++) {
        // Collect rollouts
        collect_rollouts(net, rollouts, &rng_state);

        // Calculate mean and best return
        double mean_return = 0.0;
        for(int r = 0; r < NUM_ROLLOUTS; r++) {
            mean_return += rollouts[r].returns[0];
        }
        mean_return /= NUM_ROLLOUTS;
        best_return = fmax(mean_return, best_return);
        
        // Update policy
        update_policy(net, rollouts);

        // Print progress
        struct timeval now;
        gettimeofday(&now, NULL);
        double elapsed = (now.tv_sec - start_time.tv_sec) + 
                        (now.tv_usec - start_time.tv_usec) / 1e6;
        
        printf("Epoch %d/%d | Return: %.2f/%.2f (%.1f%%) | Best: %.2f | Rate: %.3f %%/s\n", 
            epoch+1, num_epochs,
            mean_return, theoretical_max, 
            (mean_return/theoretical_max) * 100.0, best_return,
            ((best_return/theoretical_max) * 100.0 / elapsed));
    }

    // Save final policy
    char filename[64];
    time_t current_time = time(NULL);
    strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_policy.bin", localtime(&current_time));
    save_net(filename, net);

    free(rollouts);
    free(net);
    return 0;
}