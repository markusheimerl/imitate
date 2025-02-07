#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include "net.h"
#include "quad.h"

typedef struct {
    double states[MAX_STEPS][STATE_DIM];
    double actions[MAX_STEPS][ACTION_DIM];
    double rewards[MAX_STEPS];
    double returns[MAX_STEPS];
    int length;
} Rollout;

double compute_reward(const Quad* q) {
    double distance = sqrt(
        pow(q->linear_position_W[0] - 1.0, 2) +
        pow(q->linear_position_W[1] - 2.0, 2) +
        pow(q->linear_position_W[2] - 1.0, 2)
    );
    return exp(-1.0 * distance);
}

void collect_rollouts(Net* policy, Rollout* rollouts) {
    // Create an array of quadcopter simulations (one per rollout)
    Quad* quads = (Quad*)calloc(NUM_ROLLOUTS, sizeof(Quad));
    double* t_controls = (double*)calloc(NUM_ROLLOUTS, sizeof(double));
    // Active flags indicate if the rollout is still running.
    bool* active = (bool*)calloc(NUM_ROLLOUTS, sizeof(bool));
    
    for (int r = 0; r < NUM_ROLLOUTS; r++) {
        quads[r] = create_quad(0.0, 1.0, 0.0);
        t_controls[r] = 0.0;
        rollouts[r].length = 0;
        active[r] = true;
    }
    
    // Outer loop over control steps
    for (int step = 0; step < MAX_STEPS; step++) {
        // Inner loop over rollouts
        for (int r = 0; r < NUM_ROLLOUTS; r++) {
            // Skip a rollout if it has already terminated.
            if (!active[r]) continue;
            
            // Run the physics simulation until it's time for a control update.
            while (t_controls[r] < DT_CONTROL) {
                // Terminal condition: if quad gets too far from goal, mark rollout inactive.
                double dist = sqrt(
                    pow(quads[r].linear_position_W[0] - 1.0, 2) +
                    pow(quads[r].linear_position_W[1] - 2.0, 2) +
                    pow(quads[r].linear_position_W[2] - 1.0, 2)
                );
                if (dist > 4.0) {
                    active[r] = false;
                    break;
                }
                update_quad(&quads[r], DT_PHYSICS);
                t_controls[r] += DT_PHYSICS;
            }
            
            // If the rollout ended during the simulation loop, skip recording this step.
            if (!active[r]) continue;
            
            // Save sensor readings into the rollout for control step "step"
            memcpy(rollouts[r].states[step], quads[r].linear_acceleration_B_s, 3 * sizeof(double));
            memcpy(rollouts[r].states[step] + 3, quads[r].angular_velocity_B_s, 3 * sizeof(double));
            
            // Run the policy network to get control commands.
            forward_net(policy, rollouts[r].states[step]);
            for (int i = 0; i < 4; i++) {
                double mu = squash(policy->h[2][i], MIN_MEAN, MAX_MEAN);
                double sigma = squash(policy->h[2][i + 4], MIN_STD, MAX_STD);
                
                // Sample action using the Box-Muller transform.
                double u1 = (double)rand()/RAND_MAX;
                double u2 = (double)rand()/RAND_MAX;
                double epsilon = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                rollouts[r].actions[step][i] = mu + sigma * epsilon;
                
                // Set next rotor speed.
                quads[r].omega_next[i] = rollouts[r].actions[step][i];
            }
            
            // Record reward.
            rollouts[r].rewards[step] = compute_reward(&quads[r]);
            
            // Increment the rollout's length.
            rollouts[r].length = step + 1;
            
            // Reset the control timer for this rollout.
            t_controls[r] = 0.0;
        }
    }
    
    // Postprocess: compute discounted returns for each rollout.
    for (int r = 0; r < NUM_ROLLOUTS; r++) {
        double G = 0.0;
        for (int i = rollouts[r].length - 1; i >= 0; i--) {
            G = rollouts[r].rewards[i] + GAMMA * G;
            rollouts[r].returns[i] = G;
        }
    }
    
    free(quads);
    free(t_controls);
    free(active);
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
    
    // Process all rollouts for each timestep
    for(int step = 0; step < MAX_STEPS; step++) {
        zero_gradients(policy);
        
        // Process all rollouts for this timestep
        for(int r = 0; r < NUM_ROLLOUTS; r++) {
            if(step >= rollouts[r].length) continue;
            
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

        // Monitor gradients
        double grad_norm = 0.0;
        for (int i = 0; i < HIDDEN_DIM; i++) {
            for (int j = 0; j < STATE_DIM; j++) {
                grad_norm += policy->dW1[i][j] * policy->dW1[i][j];
            }
        }
        for (int i = 0; i < ACTION_DIM; i++) {
            for (int j = 0; j < HIDDEN_DIM; j++) {
                grad_norm += policy->dW2[i][j] * policy->dW2[i][j];
            }
        }
        grad_norm = sqrt(grad_norm);

        if (grad_norm > 10.0) {
            //printf("Warning: Large gradient norm at step %d: %.6f\n", step, grad_norm);
        }

        update_net(policy);
    }
}

int main(int argc, char** argv) {
    if(argc != 2 && argc != 3) {
        printf("Usage: %s <num_epochs> [initial_weights.bin]\n", argv[0]);
        return 1;
    }

    srand(time(NULL));
    
    Net* net = (argc == 3) ? load_net(argv[2]) : create_net(3e-5);
    Rollout* rollouts = (Rollout*)calloc(NUM_ROLLOUTS, sizeof(Rollout));

    int num_epochs = atoi(argv[1]);
    double best_return = -1e30;
    double theoretical_max = (1.0 - pow(GAMMA + 1e-15, MAX_STEPS))/(1.0 - (GAMMA + 1e-15));
    struct timeval start_time;
    gettimeofday(&start_time, NULL);
    
    for(int epoch = 0; epoch < num_epochs; epoch++) {
        // Collect rollouts
        collect_rollouts(net, rollouts);

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