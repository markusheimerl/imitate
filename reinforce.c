#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include "grad/grad.h"
#include "sim/sim.h"

#define DT_PHYSICS (1.0/1000.0)
#define DT_CONTROL (1.0/60.0)

#define STATE_DIM 10  // 3 accel + 3 gyro + 3 desired vel + 1 desired yaw rate
#define ACTION_DIM 8

#define MAX_STEPS 1000 // 1000 * 60Hz = 16.66s
#define NUM_ROLLOUTS 64

#define GAMMA 0.999
#define MAX_STD 3.0
#define MIN_STD 1e-5

double squash(double x, double min, double max) { return ((max + min) / 2.0) + ((max - min) / 2.0) * tanh(x); }
double dsquash(double x, double min, double max) { return ((max - min) / 2.0) * (1.0 - tanh(x) * tanh(x)); }

void velocity_controller(Quad* q, double* target_pos, double* desired_vel_B, double* desired_yaw_rate) {
    // 1. Calculate position error in world frame
    double error_p[3];
    subVec3f(q->linear_position_W, target_pos, error_p);
    
    // 2. Calculate desired velocity in world frame
    double desired_vel_W[3];
    multScalVec3f(-K_P, error_p, desired_vel_W);
    
    // 3. Calculate desired heading (yaw) based on velocity direction
    double desired_heading = atan2(desired_vel_W[2], desired_vel_W[0]);
    
    // Get current heading from rotation matrix
    double current_heading = atan2(q->R_W_B[2], q->R_W_B[0]);
    
    // Calculate heading error (considering circular nature of angles)
    double heading_error = desired_heading - current_heading;
    if (heading_error > M_PI) heading_error -= 2*M_PI;
    if (heading_error < -M_PI) heading_error += 2*M_PI;
    
    // Calculate desired yaw rate (proportional control)
    *desired_yaw_rate = K_R * heading_error;  // You can tune this gain
    
    // 4. Transform desired velocity to body frame
    double R_B_W[9];
    transpMat3f(q->R_W_B, R_B_W);
    multMatVec3f(R_B_W, desired_vel_W, desired_vel_B);
}

double get_task_radius(int epoch, int total_epochs) {
    const double INITIAL_TASK_RADIUS = 0.01;  // Start with easier tasks
    const double FINAL_TASK_RADIUS = 5.0;     // End with harder tasks
    const int CURRICULUM_WARMUP = 100;        // Initial warmup period
    
    if (epoch < CURRICULUM_WARMUP) {
        return INITIAL_TASK_RADIUS;
    }
    
    // Linear progress from 0 to 1 after warmup
    double progress = (double)(epoch - CURRICULUM_WARMUP) / 
                     (double)(total_epochs - CURRICULUM_WARMUP);
    progress = fmin(1.0, fmax(0.0, progress));  // Clamp to [0,1]
    
    // Simple linear interpolation between initial and final radius
    return INITIAL_TASK_RADIUS + 
           (FINAL_TASK_RADIUS - INITIAL_TASK_RADIUS) * progress;
}

void get_random_position(double pos[3], double center[3], double radius) {
    double theta = ((double)rand()/RAND_MAX) * 2.0 * M_PI;
    double phi = acos(2.0 * ((double)rand()/RAND_MAX) - 1.0);
    double r = radius * ((double)rand()/RAND_MAX);

    pos[0] = center[0] + r * sin(phi) * cos(theta);
    pos[1] = center[1] + r * sin(phi) * sin(theta);
    pos[2] = center[2] + r * cos(phi);

    pos[1] = fmax(pos[1], 0.5); // MIN_HEIGHT
    pos[1] = fmin(pos[1], 1.5); // MAX_HEIGHT
}

void get_state(Quad* q, double* state, double* desired_vel_B, double desired_yaw_rate) {
    // Accelerometer readings (body frame)
    memcpy(state, q->linear_acceleration_B_s, 3 * sizeof(double));
    
    // Gyroscope readings (body frame)
    memcpy(state + 3, q->angular_velocity_B_s, 3 * sizeof(double));
    
    // Desired velocities (body frame)
    memcpy(state + 6, desired_vel_B, 3 * sizeof(double));
    
    // Desired yaw rate
    state[9] = desired_yaw_rate;
}

double compute_reward(Quad* q, double* desired_vel_B, double desired_yaw_rate) {
    // Calculate velocity error in body frame
    double current_vel_B[3];
    double R_B_W[9];
    transpMat3f(q->R_W_B, R_B_W);
    multMatVec3f(R_B_W, q->linear_velocity_W, current_vel_B);
    
    double vel_error = 0.0;
    for(int i = 0; i < 3; i++) {
        vel_error += pow(current_vel_B[i] - desired_vel_B[i], 2);
    }
    
    // Calculate yaw rate error
    double yaw_rate_error = pow(q->angular_velocity_B[1] - desired_yaw_rate, 2);
    
    // Calculate other angular velocities (should be minimized)
    double other_ang_vel = pow(q->angular_velocity_B[0], 2) + 
                          pow(q->angular_velocity_B[2], 2);
    
    double orientation_error = fabs(1.0 - q->R_W_B[4]);  // Maintain upright
    
    double total_error = (vel_error * 2.0) + 
                        (yaw_rate_error * 1.0) +
                        (other_ang_vel * 0.5) + 
                        (orientation_error * 2.0);
    
    return exp(-total_error);
}

bool is_terminated(Quad* q, double* target_pos) {
    static const double MAX_DISTANCE = 2.0;
    static const double MAX_VELOCITY = 5.0;
    static const double MAX_ANGULAR_VELOCITY = 5.0;
    static const double MIN_HEIGHT = 0.1;
    static const double MIN_ORIENTATION = 0.0;

    double dist = 0.0, vel = 0.0, ang_vel = 0.0;
    for(int i = 0; i < 3; i++) {
        dist += pow(q->linear_position_W[i] - target_pos[i], 2);
        vel += pow(q->linear_velocity_W[i], 2);
        ang_vel += pow(q->angular_velocity_B[i], 2);
    }
    
    // Check all termination conditions
    return q->linear_position_W[1] < MIN_HEIGHT ||    // Too low
           sqrt(dist) > MAX_DISTANCE ||               // Too far from target
           sqrt(vel) > MAX_VELOCITY ||               // Moving too fast
           sqrt(ang_vel) > MAX_ANGULAR_VELOCITY ||   // Spinning too fast
           q->R_W_B[4] < MIN_ORIENTATION;           // Not upright
}

int collect_rollout(Sim* sim, Net* policy, double** states, double** actions, double* rewards, int epoch, int total_epochs) {
    double start_pos[3];
    double target_pos[3];
    
    // Get current task radius based on curriculum learning progress
    double current_radius = get_task_radius(epoch, total_epochs);

    get_random_position(start_pos, (double[3]){0, 1, 0}, current_radius);
    get_random_position(target_pos, start_pos, current_radius);
    
    reset_quad(sim->quad, start_pos[0], start_pos[1], start_pos[2]);
    
    double t_physics = 0.0, t_control = 0.0;
    int steps = 0;
    
    double desired_vel_B[3] = {0};
    double desired_yaw_rate = 0.0;
    
    while(steps < MAX_STEPS && !is_terminated(sim->quad, target_pos)) {
        update_quad(sim->quad, DT_PHYSICS);
        t_physics += DT_PHYSICS;
        
        if(t_control <= t_physics) {
            // Get desired body-frame velocities and yaw rate from high-level controller
            velocity_controller(sim->quad, target_pos, desired_vel_B, &desired_yaw_rate);
            
            // Get state using sensor data, desired velocities, and desired yaw rate
            get_state(sim->quad, states[steps], desired_vel_B, desired_yaw_rate);
            
            // Run policy network
            forward(policy, states[steps]);
            
            for(int i = 0; i < 4; i++) {
                // 1. Transform network outputs to valid std and mean ranges
                // σ ∈ [MIN_STD, MAX_STD] via squashing function
                double std = squash(policy->layers[policy->n_layers-1].x[i + 4], MIN_STD, MAX_STD);
                
                // Ensure actions stay within bounds with high probability (4σ = 99.994%)
                // μ ∈ [OMEGA_MIN + 4σ, OMEGA_MAX - 4σ]
                double safe_margin = 4.0 * std;
                double mean_min = OMEGA_MIN + safe_margin;
                double mean_max = OMEGA_MAX - safe_margin;
                double mean = squash(policy->layers[policy->n_layers-1].x[i], mean_min, mean_max);
                
                // 2. Sample action from Gaussian distribution
                double u1 = (double)rand()/RAND_MAX;
                double u2 = (double)rand()/RAND_MAX;
                double noise = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                
                actions[steps][i] = mean + std * noise;
                sim->quad->omega_next[i] = actions[steps][i];
            }
            
            rewards[steps] = compute_reward(sim->quad, desired_vel_B, desired_yaw_rate);
            steps++;
            t_control += DT_CONTROL;
        }
    }
    
    // Compute discounted returns: Gₜ = Σₖ₌₀ γᵏ rₜ₊ₖ₊₁
    double G = 0.0;
    for(int i = steps-1; i >= 0; i--) {
        rewards[i] = G = rewards[i] + GAMMA * G;
    }
    
    return steps;
}

// ∇J(θ) = E[∇log π_θ(a|s) * R] ≈ (1/N) Σᵢ[∇log π_θ(aᵢ|sᵢ) * Rᵢ] - REINFORCE algorithm
// where π_θ(a|s) is a Gaussian policy with state-dependent mean μ(s) and std σ(s)
void update_policy(Net* policy, double** states, double** actions, double* returns, int steps, int epoch, int epochs) {
    double output_gradient[ACTION_DIM];
    
    for(int t = 0; t < steps; t++) {
        forward(policy, states[t]);
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
            
            // 2. Normalize action to compute probability
            // z = (a - μ)/σ where a is the actual action taken
            double z = (actions[t][i] - mean) / std;
            
            // 3. Log probability of Gaussian distribution
            // log π(a|s) = -1/2(log(2π) + 2log(σ) + ((a-μ)/σ)²)
            double log_2pi = log(2.0 * M_PI);
            double log_var = 2.0 * log(std);
            double squared_error = z * z;
            double log_prob = -0.5 * (log_2pi + log_var + squared_error);

            // 4. Gradient for mean parameter
            // ∂log π/∂μ = (a-μ)/σ² = z/σ
            // Chain rule through squashing: ∂log π/∂θμ = (∂log π/∂μ)(∂μ/∂θμ)
            // where ∂μ/∂θμ = dsquash(θμ, mean_min, mean_max)
            double dmean = z / std;
            double dmean_dtheta = dsquash(output->x[i], mean_min, mean_max);
            output_gradient[i] = returns[t] * log_prob * dmean * dmean_dtheta;
            
            // 5. Gradient for standard deviation parameter
            // Direct effect: ∂log π/∂σ = (z² - 1)/σ
            // Indirect effect through mean bounds: ∂μ/∂σ = -4.0 * dsquash
            // Total effect: ∂log π/∂σ = (z² - 1)/σ + (z/σ) * (-4.0 * dsquash)
            double dstd_direct = (squared_error - 1.0) / std;
            double dmean_dstd = -4.0 * dsquash(output->x[i], mean_min, mean_max);
            double dstd = dstd_direct + (z / std) * dmean_dstd;
            
            // Chain rule through squashing: ∂log π/∂θσ = (∂log π/∂σ)(∂σ/∂θσ)
            // where ∂σ/∂θσ = dsquash(θσ, MIN_STD, MAX_STD)
            double dstd_dtheta = dsquash(output->x[i + 4], MIN_STD, MAX_STD);
            output_gradient[i + 4] = returns[t] * log_prob * dstd * dstd_dtheta;
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
    
    Net* net;
    if(argc == 3) {
        net = load_net(argv[2]);
    } else {
        int layers[] = {STATE_DIM, 64, ACTION_DIM};
        net = init_net(3, layers, 1e-4);
    }
    
    Sim* sim = init_sim("", false);
    double theoretical_max = (1.0 - pow(GAMMA + 1e-15, MAX_STEPS))/(1.0 - (GAMMA + 1e-15));
    double elapsed = 0.0;

    // Heap allocations for large arrays
    double*** all_states = malloc(NUM_ROLLOUTS * sizeof(double**));
    double*** all_actions = malloc(NUM_ROLLOUTS * sizeof(double**));
    double** all_rewards = malloc(NUM_ROLLOUTS * sizeof(double*));
    int* rollout_steps = malloc(NUM_ROLLOUTS * sizeof(int));

    // Allocate memory for each rollout
    for(int r = 0; r < NUM_ROLLOUTS; r++) {
        all_states[r] = malloc(MAX_STEPS * sizeof(double*));
        all_actions[r] = malloc(MAX_STEPS * sizeof(double*));
        all_rewards[r] = malloc(MAX_STEPS * sizeof(double));
        
        for(int i = 0; i < MAX_STEPS; i++) {
            all_states[r][i] = malloc(STATE_DIM * sizeof(double));
            all_actions[r][i] = malloc(4 * sizeof(double));
        }
    }

    int epochs = atoi(argv[1]);
    double best_return = -1e30;
    double initial_best = -1e30;
    struct timeval start_time, current_time;
    gettimeofday(&start_time, NULL);
    
    for(int epoch = 0; epoch < epochs; epoch++) {
        double sum_returns = 0.0;
        double current_radius = get_task_radius(epoch, epochs);

        for(int r = 0; r < NUM_ROLLOUTS; r++) {
            rollout_steps[r] = collect_rollout(sim, net, 
                                             all_states[r], 
                                             all_actions[r], 
                                             all_rewards[r],
                                             epoch, epochs);
            sum_returns += all_rewards[r][0];
        }

        for(int r = 0; r < NUM_ROLLOUTS; r++) {
            update_policy(net, all_states[r], all_actions[r], 
                         all_rewards[r], rollout_steps[r], epoch, epochs);
        }

        double mean_return = sum_returns / NUM_ROLLOUTS;
        if(mean_return > best_return) {
            best_return = mean_return;
        }

        if(epoch == 0) {
            initial_best = best_return;
        }

        gettimeofday(&current_time, NULL);
        elapsed = (current_time.tv_sec - start_time.tv_sec) + 
                        (current_time.tv_usec - start_time.tv_usec) / 1000000.0;
        
        double initial_percentage = (initial_best / theoretical_max) * 100.0;
        double current_percentage = (best_return / theoretical_max) * 100.0;
        double percentage_rate = (current_percentage - initial_percentage) / elapsed;

        printf("epoch %d/%d | Radius: %.3f | Return: %.2f/%.2f (%.1f%%) | Best: %.2f | Rate: %.3f %%/s\n", 
               epoch+1, epochs, current_radius, mean_return, theoretical_max, 
               (mean_return/theoretical_max) * 100.0, best_return,
               percentage_rate);
    }
    printf("\n");

    char final_weights[64];
    strftime(final_weights, sizeof(final_weights), "%Y%m%d_%H%M%S_policy.bin", 
             localtime(&(time_t){time(NULL)}));
    save_net(final_weights, net);
    printf("Final weights saved to: %s\n", final_weights);

    int hours = (int)elapsed / 3600;
    int minutes = ((int)elapsed % 3600) / 60;
    int seconds = (int)elapsed % 60;
    printf("\nTotal training time: %dh %dm %ds\n", hours, minutes, seconds);

    // Free all allocated memory
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
    free(rollout_steps);

    free_net(net);
    free_sim(sim);

    return 0;
}