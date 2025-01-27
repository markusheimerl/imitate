#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include "grad.h"
#include "quad.h"
#include "rollout.h"

double squash(double x, double min, double max) { 
    return ((max + min) / 2.0) + ((max - min) / 2.0) * tanh(x); 
}

double dsquash(double x, double min, double max) { 
    return ((max - min) / 2.0) * (1.0 - tanh(x) * tanh(x)); 
}

double compute_reward(Quad q, double* target_pos) {
    double pos_error = 0.0;
    for(int i = 0; i < 3; i++) {
        pos_error += pow(q.linear_position_W[i] - target_pos[i], 2);
    }
    pos_error = sqrt(pos_error);
    
    double vel_magnitude = 0.0;
    for(int i = 0; i < 3; i++) {
        vel_magnitude += pow(q.linear_velocity_W[i], 2);
    }
    vel_magnitude = sqrt(vel_magnitude);
    
    double ang_vel_magnitude = 0.0;
    for(int i = 0; i < 3; i++) {
        ang_vel_magnitude += pow(q.angular_velocity_B[i], 2);
    }
    ang_vel_magnitude = sqrt(ang_vel_magnitude);
    
    double orientation_error = fabs(1.0 - q.R_W_B[4]);
    
    double total_error = (pos_error * 2.0) + 
                        (vel_magnitude * 1.0) + 
                        (ang_vel_magnitude * 0.5) + 
                        (orientation_error * 2.0);
    
    return exp(-total_error);
}

void collect_rollout(Net* policy, Rollout* rollout, int epoch, int epochs) {
    // Linear curriculum learning: increase max distance over time
    double max_current_distance = MIN_DISTANCE + 
        (MAX_DISTANCE - MIN_DISTANCE) * ((double)epoch / epochs);
    
    // Generate two random positions within a sphere of radius max_current_distance
    double positions[2][3];
    for(int i = 0; i < 2; i++) {
        double r = max_current_distance * ((double)rand() / RAND_MAX);
        double theta = 2 * M_PI * ((double)rand() / RAND_MAX);
        double phi = acos(2 * ((double)rand() / RAND_MAX) - 1);
        
        positions[i][0] = r * sin(phi) * cos(theta);
        positions[i][1] = r * sin(phi) * sin(theta) + 1.0;
        positions[i][2] = r * cos(phi);
    }

    double* start = positions[0];
    double* target = positions[1];
    
    // Create a new quad for this rollout
    Quad quad = create_quad(start[0], start[1], start[2]);
    
    // Initialize timers
    double t_physics = 0.0;
    double t_control = 0.0;
    rollout->length = 0;

    while(rollout->length < MAX_STEPS) {
        if (dotVec3f(quad.linear_position_W, quad.linear_position_W) > 16.0 || // 4 meters squared
            dotVec3f(quad.linear_velocity_W, quad.linear_velocity_W) > 25.0 || // 5 m/s squared
            dotVec3f(quad.angular_velocity_B, quad.angular_velocity_B) > 25.0 || // ~5 rad/s squared
            quad.R_W_B[4] < 0.0 /* more than 90° tilt */) break;
            
        // Physics update
        if (t_physics >= DT_PHYSICS) {
            update_quad(&quad, DT_PHYSICS);
            t_physics = 0.0;
        }
        
        // Control update
        if (t_control >= DT_CONTROL) {
            get_quad_state(quad, rollout->states[rollout->length]);
            memcpy(rollout->states[rollout->length] + 12, target, 3 * sizeof(double));
            
            forward(policy, rollout->states[rollout->length]);
            
            for(int i = 0; i < 4; i++) {
                double std = squash(policy->layers[policy->n_layers-1].x[i + 4], 
                                  MIN_STD, MAX_STD);
                
                double safe_margin = 4.0 * std;
                double mean_min = OMEGA_MIN + safe_margin;
                double mean_max = OMEGA_MAX - safe_margin;
                double mean = squash(policy->layers[policy->n_layers-1].x[i], 
                                   mean_min, mean_max);
                
                double u1 = (double)rand()/RAND_MAX;
                double u2 = (double)rand()/RAND_MAX;
                double noise = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                
                rollout->actions[rollout->length][i] = mean + std * noise;
                quad.omega_next[i] = rollout->actions[rollout->length][i];
            }
            
            rollout->rewards[rollout->length] = compute_reward(quad, target);
            rollout->length++;
            t_control = 0.0;
        }
        
        // Increment timers
        t_physics += DT_PHYSICS;
        t_control += DT_PHYSICS;
    }
    
    // Compute returns
    double G = 0.0;
    for(int i = rollout->length-1; i >= 0; i--) {
        rollout->rewards[i] = G = rollout->rewards[i] + GAMMA * G;
    }
}

// ∇J(θ) = E[∇log π_θ(a|s) * R] ≈ (1/N) Σᵢ[∇log π_θ(aᵢ|sᵢ) * Rᵢ] - REINFORCE algorithm
// For Gaussian policy π_θ(a|s) = N(μ_θ(s), σ_θ(s))
// where μ_θ(s) is the mean action and σ_θ(s) is the standard deviation
void update_policy(Net* policy, Rollout* rollout, int epoch, int epochs) {
    double output_gradient[ACTION_DIM];
    
    for(int t = 0; t < rollout->length; t++) {
        forward(policy, rollout->states[t]);
        Layer *output = &policy->layers[policy->n_layers-1];
        
        for(int i = 0; i < 4; i++) {
            // 1. Transform network outputs to valid std and mean ranges
            // σ ∈ [MIN_STD, MAX_STD] via squashing function
            double std = squash(output->x[i + 4], MIN_STD, MAX_STD);
            
            // Ensure actions stay within bounds with high probability (4σ = 99.994%)
            // μ ∈ [OMEGA_MIN + 4σ, OMEGA_MAX - 4σ]
            double safe_margin = 4.0 * std;
            double mean_min = OMEGA_MIN + safe_margin;
            double mean_max = OMEGA_MAX - safe_margin;
            double mean = squash(output->x[i], mean_min, mean_max);
            
            // 2. Compute normalized action
            // z = (a - μ)/σ
            double z = (rollout->actions[t][i] - mean) / std;
            
            // 3. Compute gradients for Gaussian policy
            // For mean: ∂log π/∂μ = (a-μ)/σ² = z/σ
            // Chain rule: ∂log π/∂θμ = (∂log π/∂μ)(∂μ/∂θμ)
            // where ∂μ/∂θμ = dsquash(θμ, mean_min, mean_max)
            double dmean_dtheta = dsquash(output->x[i], mean_min, mean_max);
            output_gradient[i] = -rollout->rewards[t] * (z / std) * dmean_dtheta;
            
            // 4. Gradient for standard deviation
            // ∂log π/∂σ = (z² - 1)/σ
            // Indirect effect through mean bounds: ∂μ/∂σ = -4.0 * dsquash
            // Total effect: ∂log π/∂σ = (z² - 1)/σ + (z/σ) * (-4.0 * dsquash)
            double dstd_direct = (z * z - 1.0) / std;
            double dmean_dstd = -4.0 * dsquash(output->x[i], mean_min, mean_max);
            double dstd = dstd_direct + (z / std) * dmean_dstd;
            
            // Chain rule through squashing: ∂log π/∂θσ = (∂log π/∂σ)(∂σ/∂θσ)
            // where ∂σ/∂θσ = dsquash(θσ, MIN_STD, MAX_STD)
            double dstd_dtheta = dsquash(output->x[i + 4], MIN_STD, MAX_STD);
            output_gradient[i + 4] = -rollout->rewards[t] * dstd * dstd_dtheta;
        }

        bwd(policy, output_gradient, epoch, epochs);
    }
}

int main(int argc, char** argv) {
    if(argc != 2 && argc != 3) {
        printf("Usage: %s <num_epochs> [initial_weights.bin]\n", argv[0]);
        return 1;
    }

    srand(time(NULL) ^ getpid());
    Net* net = (argc == 3) ? load_net(argv[2]) : init_net(3, (int[]){STATE_DIM, 64, ACTION_DIM}, 1e-5);
    
    Rollout* rollouts[NUM_ROLLOUTS];
    for(int r = 0; r < NUM_ROLLOUTS; r++) rollouts[r] = create_rollout();

    int epochs = atoi(argv[1]);
    double best_return = -1e30;
    double theoretical_max = (1.0 - pow(GAMMA + 1e-15, MAX_STEPS))/(1.0 - (GAMMA + 1e-15));
    struct timeval start_time;
    gettimeofday(&start_time, NULL);
    
    for(int epoch = 0; epoch < epochs; epoch++) {
        double sum_returns = 0.0;
        for(int r = 0; r < NUM_ROLLOUTS; r++) {
            collect_rollout(net, rollouts[r], epoch, epochs);
            sum_returns += rollouts[r]->rewards[0];
        }
        
        for(int r = 0; r < NUM_ROLLOUTS; r++) {
            update_policy(net, rollouts[r], epoch, epochs);
        }

        double mean_return = sum_returns / NUM_ROLLOUTS;
        best_return = fmax(mean_return, best_return);

        struct timeval now;
        gettimeofday(&now, NULL);
        double elapsed = (now.tv_sec - start_time.tv_sec) + (now.tv_usec - start_time.tv_usec) / 1e6;
        
        printf("epoch %d/%d | Distance: %.3f | Return: %.2f/%.2f (%.1f%%) | Best: %.2f | Rate: %.3f %%/s\n", 
               epoch+1, epochs, 
               MIN_DISTANCE + (MAX_DISTANCE - MIN_DISTANCE) * ((double)epoch / epochs),
               mean_return, theoretical_max, 
               (mean_return/theoretical_max) * 100.0, best_return,
               ((best_return/theoretical_max) * 100.0 / elapsed));
    }

    char filename[64];
    strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_policy.bin", localtime(&(time_t){time(NULL)}));
    save_net(filename, net);

    for(int r = 0; r < NUM_ROLLOUTS; r++) free_rollout(rollouts[r]);
    free_net(net);
    return 0;
}