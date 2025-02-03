#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include "net.h"
#include "quad.h"

#define DT_PHYSICS (1.0/1000.0)
#define DT_CONTROL (1.0/60.0)
#define DT_RENDER (1.0/24.0)

#define STATE_DIM 6      // 3 accel + 3 gyro
#define ACTION_DIM 8     // 4 means + 4 stds
#define MAX_STEPS 256
#define NUM_ROLLOUTS 128

#define GAMMA 0.999
#define MAX_STD 3.0
#define MIN_STD 1e-5

#define MAX_MEAN (OMEGA_MAX - 4.0 * MAX_STD)
#define MIN_MEAN (OMEGA_MIN + 4.0 * MAX_STD)

typedef struct {
    double states[NUM_ROLLOUTS][MAX_STEPS][STATE_DIM];
    double actions[NUM_ROLLOUTS][MAX_STEPS][ACTION_DIM];
    double rewards[NUM_ROLLOUTS][MAX_STEPS];
    double returns[NUM_ROLLOUTS][MAX_STEPS];
    int lengths[NUM_ROLLOUTS];
    Quad quads[NUM_ROLLOUTS];
} Rollout;

double squash(double x, double min, double max) { 
    return ((max + min) / 2.0) + ((max - min) / 2.0) * tanh(x); 
}

double dsquash(double x, double min, double max) { 
    return ((max - min) / 2.0) * (1.0 - tanh(x) * tanh(x)); 
}

double compute_reward(const Quad* q) {
    double distance = sqrt(
        pow(q->linear_position_W[0] - 0.0, 2) +  // x distance from 0
        pow(q->linear_position_W[1] - 1.0, 2) +  // y distance from 1
        pow(q->linear_position_W[2] - 0.0, 2)    // z distance from 0
    );
    return exp(-4.0 * distance);
}

void collect_rollouts(Net* policy, Rollout* rollout) {
    for(int r = 0; r < NUM_ROLLOUTS; r++) {
        rollout->quads[r] = create_quad(0.0, 1.0, 0.0);
        Quad* quad = &rollout->quads[r];
        
        double t_control = 0.0;
        rollout->lengths[r] = 0;

        while(rollout->lengths[r] < MAX_STEPS) {
            if (sqrt(
                pow(quad->linear_position_W[0], 2) +
                pow(quad->linear_position_W[1] - 1.0, 2) +
                pow(quad->linear_position_W[2], 2)) > 1.0) {
                break;
            }

            update_quad(quad, DT_PHYSICS);
            t_control += DT_PHYSICS;
            
            if (t_control >= DT_CONTROL) {
                int step = rollout->lengths[r];
                
                memcpy(rollout->states[r][step], quad->linear_acceleration_B_s, 3 * sizeof(double));
                memcpy(rollout->states[r][step] + 3, quad->angular_velocity_B_s, 3 * sizeof(double));
                
                forward_net(policy, rollout->states[r][step]);
                
                for(int i = 0; i < 4; i++) {
                    double mean = squash(policy->layers[policy->num_layers-1].values[i], MIN_MEAN, MAX_MEAN);
                    double std = squash(policy->layers[policy->num_layers-1].values[i + 4], MIN_STD, MAX_STD);

                    double u1 = (double)rand()/RAND_MAX;
                    double u2 = (double)rand()/RAND_MAX;
                    double noise = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                    
                    rollout->actions[r][step][i] = mean + std * noise;
                    quad->omega_next[i] = rollout->actions[r][step][i];
                }
                
                rollout->rewards[r][step] = compute_reward(quad);
                rollout->lengths[r]++;
                t_control = 0.0;
            }
        }
        
        // Compute discounted returns
        double G = 0.0;
        for(int i = rollout->lengths[r]-1; i >= 0; i--) {
            G = rollout->rewards[r][i] + GAMMA * G;
            rollout->returns[r][i] = G;
        }
    }
}

// ∇J(θ) = E[∇_θ log π_θ(a|s) * R] ≈ 1/N Σ_t [∇_θ log π_θ(a_t|s_t) * R_t]
// Where:
// J(θ) - Policy objective function
// π_θ(a|s) - Gaussian policy parameterized by θ (network weights)
// R_t - Discounted return from time step t
void update_policy(Net* policy, Rollout* rollout) {
    double output_gradient[ACTION_DIM];
    zero_gradients(policy);
    
    for(int r = 0; r < NUM_ROLLOUTS; r++) {
        for(int t = 0; t < rollout->lengths[r]; t++) {
            forward_net(policy, rollout->states[r][t]);
            
            for(int i = 0; i < 4; i++) {
                // Network outputs raw parameters before squashing
                double mean_raw = policy->layers[policy->num_layers-1].values[i];
                double std_raw = policy->layers[policy->num_layers-1].values[i + 4];
                
                // Squashed parameters using tanh-based scaling
                // μ = ((MAX+MIN)/2) + ((MAX-MIN)/2)*tanh(mean_raw)
                // σ = ((MAX_STD+MIN_STD)/2) + ((MAX_STD-MIN_STD)/2)*tanh(std_raw)
                double mean = squash(mean_raw, MIN_MEAN, MAX_MEAN);
                double std_val = squash(std_raw, MIN_STD, MAX_STD);
                
                // Sampled action and its deviation from mean
                double delta = rollout->actions[r][t][i] - mean;

                // Gradient for mean parameter:
                // ∇_{μ_raw} log π = [ (a - μ)/σ² ] * dμ/dμ_raw
                // Where:
                // (a - μ)/σ² = derivative of log N(a; μ, σ²) w.r.t μ
                // dμ/dμ_raw = derivative of squashing function (dsquash)
                output_gradient[i] = -(delta / (std_val * std_val)) * 
                    dsquash(mean_raw, MIN_MEAN, MAX_MEAN) * 
                    rollout->returns[r][t];

                // Gradient for standard deviation parameter:
                // ∇_{σ_raw} log π = [ ( (a-μ)^2 - σ² ) / σ³ ] * dσ/dσ_raw
                // Where:
                // ( (a-μ)^2 - σ² ) / σ³ = derivative of log N(a; μ, σ²) w.r.t σ
                // dσ/dσ_raw = derivative of squashing function (dsquash)
                output_gradient[i + 4] = -((delta * delta - std_val * std_val) / 
                    (std_val * std_val * std_val)) * 
                    dsquash(std_raw, MIN_STD, MAX_STD) * 
                    rollout->returns[r][t];
            }

            backward_net(policy, output_gradient);
        }
    }
    update_net(policy);
}

int main(int argc, char** argv) {
    if(argc != 2 && argc != 3) {
        printf("Usage: %s <num_epochs> [initial_weights.bin]\n", argv[0]);
        return 1;
    }

    srand(time(NULL) ^ getpid());
    
    static const int layer_sizes[] = {STATE_DIM, 64, ACTION_DIM};
    Net* net = (argc == 3) ? load_net(argv[2]) : create_net(3, layer_sizes, 9e-8);
    
    Rollout rollout = {0};

    int epochs = atoi(argv[1]);
    double best_return = -1e30;
    double theoretical_max = (1.0 - pow(GAMMA + 1e-15, MAX_STEPS))/(1.0 - (GAMMA + 1e-15));
    struct timeval start_time;
    gettimeofday(&start_time, NULL);
    
    for(int epoch = 0; epoch < epochs; epoch++) {
        collect_rollouts(net, &rollout);
        update_policy(net, &rollout);

        double mean_return = 0.0;
        for(int r = 0; r < NUM_ROLLOUTS; r++) {
            mean_return += rollout.returns[r][0];
        }
        mean_return /= NUM_ROLLOUTS;
        
        best_return = fmax(mean_return, best_return);

        struct timeval now;
        gettimeofday(&now, NULL);
        double elapsed = (now.tv_sec - start_time.tv_sec) + 
                        (now.tv_usec - start_time.tv_usec) / 1e6;
        
        printf("epoch %d/%d | Return: %.2f/%.2f (%.1f%%) | Best: %.2f | Rate: %.3f %%/s\n", 
            epoch+1, epochs,
            mean_return, theoretical_max, 
            (mean_return/theoretical_max) * 100.0, best_return,
            ((best_return/theoretical_max) * 100.0 / elapsed));
    }

    char filename[64];
    time_t current_time = time(NULL);
    strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_policy.bin", 
             localtime(&current_time));
    save_net(filename, net);

    free_net(net);
    return 0;
}