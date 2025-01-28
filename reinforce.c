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

double compute_reward(Quad q) {
    // 1. Acceleration stability (want close to 0 for hover)
    double accel_error = 0.0;
    for(int i = 0; i < 3; i++) {
        double accel = q.linear_acceleration_B_s[i];
        if(i == 1) accel += GRAVITY;  // Compensate for gravity in Y axis
        accel_error += accel * accel;
    }
    accel_error = sqrt(accel_error);
    
    // 2. Angular velocity stability (want close to 0 for hover)
    double ang_vel_magnitude = 0.0;
    for(int i = 0; i < 3; i++) {
        ang_vel_magnitude += q.angular_velocity_B_s[i] * q.angular_velocity_B_s[i];
    }
    ang_vel_magnitude = sqrt(ang_vel_magnitude);
    
    // 3. Orientation stability (want upright)
    double orientation_error = fabs(1.0 - q.R_W_B[4]);
    
    // 4. Position drift (3D distance from hover point)
    double drift = sqrt(
        pow(q.linear_position_W[0], 2) +
        pow(q.linear_position_W[1] - 1.0, 2) +
        pow(q.linear_position_W[2], 2)
    );
    
    // Only penalize drift beyond 0.25m radius
    double position_error = fmax(0.0, drift - 0.25);
    
    // Combine all factors with strong emphasis on stability
    double stability_error = (accel_error * 1.0) +         // Acceleration stability
                           (ang_vel_magnitude * 2.0) +      // Angular velocity stability
                           (orientation_error * 4.0) +      // Upright orientation (most important)
                           (position_error * 2.0);          // Position drift beyond allowed radius
    
    return exp(-stability_error);
}

void collect_rollout(Net* policy, Rollout* rollout) {
    Quad quad = create_quad(0.0, 1.0, 0.0);
    
    double t_physics = 0.0;
    double t_control = 0.0;
    rollout->length = 0;

    while(rollout->length < MAX_STEPS) {
        // Calculate 3D distance from hover point
        double drift = sqrt(
            pow(quad.linear_position_W[0], 2) +
            pow(quad.linear_position_W[1] - 1.0, 2) +
            pow(quad.linear_position_W[2], 2)
        );

        if (dotVec3f(quad.linear_velocity_W, quad.linear_velocity_W) > 4.0 ||  // ~2 m/s
            dotVec3f(quad.angular_velocity_B, quad.angular_velocity_B) > 9.0 || // ~3 rad/s
            quad.R_W_B[4] < 0.0 ||    // More than 90° tilt
            drift > 1.0) {            // More than 1m from hover point
            break;
        }
            
        if (t_physics >= DT_PHYSICS) {
            update_quad(&quad, DT_PHYSICS);
            t_physics = 0.0;
        }
        
        if (t_control >= DT_CONTROL) {
            // Use only sensor readings as state
            memcpy(rollout->states[rollout->length], quad.linear_acceleration_B_s, 3 * sizeof(double));
            memcpy(rollout->states[rollout->length] + 3, quad.angular_velocity_B_s, 3 * sizeof(double));
            
            forward(policy, rollout->states[rollout->length]);
            
            for(int i = 0; i < 4; i++) {
                double mean = squash(policy->layers[policy->n_layers-1].x[i], MIN_MEAN, MAX_MEAN);
                double std = squash(policy->layers[policy->n_layers-1].x[i + 4], MIN_STD, MAX_STD);

                double u1 = (double)rand()/RAND_MAX;
                double u2 = (double)rand()/RAND_MAX;
                double noise = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                
                rollout->actions[rollout->length][i] = mean + std * noise;
                quad.omega_next[i] = rollout->actions[rollout->length][i];
            }
            
            rollout->rewards[rollout->length] = compute_reward(quad);
            rollout->length++;
            t_control = 0.0;
        }
        
        t_physics += DT_PHYSICS;
        t_control += DT_PHYSICS;
    }
    
    double G = 0.0;
    for(int i = rollout->length-1; i >= 0; i--) {
        rollout->rewards[i] = G = rollout->rewards[i] + GAMMA * G;
    }
}

// ∇J(θ) = E[∇_θ log π_θ(a|s) * R] ≈ 1/N Σ_t [∇_θ log π_θ(a_t|s_t) * R_t]
// Where:
// J(θ) - Policy objective function
// π_θ(a|s) - Gaussian policy parameterized by θ (network weights)
// R_t - Discounted return from time step t
void update_policy(Net* policy, Rollout* rollout, int epoch, int epochs) {
    double output_gradient[ACTION_DIM];
    
    for(int t = 0; t < rollout->length; t++) {
        forward(policy, rollout->states[t]);
        
        for(int i = 0; i < 4; i++) {
            // Network outputs raw parameters before squashing
            double mean_raw = policy->layers[policy->n_layers-1].x[i];
            double std_raw = policy->layers[policy->n_layers-1].x[i + 4];
            
            // Squashed parameters using tanh-based scaling
            // μ = ((MAX+MIN)/2) + ((MAX-MIN)/2)*tanh(mean_raw)
            // σ = ((MAX_STD+MIN_STD)/2) + ((MAX_STD-MIN_STD)/2)*tanh(std_raw)
            double mean = squash(mean_raw, MIN_MEAN, MAX_MEAN);
            double std_val = squash(std_raw, MIN_STD, MAX_STD);
            
            // Sampled action and its deviation from mean
            double action = rollout->actions[t][i];
            double delta = action - mean;

            // Gradient for mean parameter:
            // ∇_{μ_raw} log π = [ (a - μ)/σ² ] * dμ/dμ_raw
            // Where:
            // (a - μ)/σ² = derivative of log N(a; μ, σ²) w.r.t μ
            // dμ/dμ_raw = derivative of squashing function (dsquash)
            output_gradient[i] = -(delta / (std_val * std_val)) * 
                dsquash(mean_raw, MIN_MEAN, MAX_MEAN) * 
                rollout->rewards[t];

            // Gradient for standard deviation parameter:
            // ∇_{σ_raw} log π = [ ( (a-μ)^2 - σ² ) / σ³ ] * dσ/dσ_raw
            // Where:
            // ( (a-μ)^2 - σ² ) / σ³ = derivative of log N(a; μ, σ²) w.r.t σ
            // dσ/dσ_raw = derivative of squashing function (dsquash)
            output_gradient[i + 4] = -((delta * delta - std_val * std_val) / 
                (std_val * std_val * std_val)) * 
                dsquash(std_raw, MIN_STD, MAX_STD) * 
                rollout->rewards[t];
        }

        // Backpropagate gradients through network
        // Negative sign converts gradient ascent (policy improvement) 
        // to gradient descent (standard optimization framework)
        bwd(policy, output_gradient, epoch, epochs);
    }
}

int main(int argc, char** argv) {
    if(argc != 2 && argc != 3) {
        printf("Usage: %s <num_epochs> [initial_weights.bin]\n", argv[0]);
        return 1;
    }

    srand(time(NULL) ^ getpid());
    Net* net = (argc == 3) ? load_net(argv[2]) : init_net(3, (int[]){STATE_DIM, 64, ACTION_DIM}, 5e-6);
    
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
            collect_rollout(net, rollouts[r]);
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
        
        printf("epoch %d/%d | Return: %.2f/%.2f (%.1f%%) | Best: %.2f | Rate: %.3f %%/s\n", 
            epoch+1, epochs,
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