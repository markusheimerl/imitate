#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include "grad.h"
#include "quad.h"
#include <stdbool.h>

#define DT_PHYSICS (1.0/1000.0)
#define DT_CONTROL (1.0/60.0)

#define STATE_DIM 15
#define ACTION_DIM 8
#define MAX_STEPS 1000
#define NUM_ROLLOUTS 64

#define GAMMA 0.999
#define MAX_STD 3.0
#define MIN_STD 1e-5

typedef struct {
    double** states;    // [MAX_STEPS][STATE_DIM]
    double** actions;   // [MAX_STEPS][NUM_ROTORS]
    double* rewards;    // [MAX_STEPS]
    int length;
} Rollout;

Rollout* create_rollout() {
    Rollout* r = malloc(sizeof(Rollout));
    r->states = malloc(MAX_STEPS * sizeof(double*));
    r->actions = malloc(MAX_STEPS * sizeof(double*));
    r->rewards = malloc(MAX_STEPS * sizeof(double));
    
    for(int i = 0; i < MAX_STEPS; i++) {
        r->states[i] = malloc(STATE_DIM * sizeof(double));
        r->actions[i] = malloc(4 * sizeof(double));
    }
    return r;
}

void free_rollout(Rollout* r) {
    for(int i = 0; i < MAX_STEPS; i++) {
        free(r->states[i]);
        free(r->actions[i]);
    }
    free(r->states);
    free(r->actions);
    free(r->rewards);
    free(r);
}

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
    // Get random start/target with curriculum
    const double max_training_distance = 2.0;
    double r = (epoch < 100) ? 0.01 : 
               0.01 + (max_training_distance - 0.01) * fmin(1.0, (epoch - 100.0)/(epochs - 100.0));
    
    double angle = 2 * M_PI * ((double)rand() / RAND_MAX);
    double dist = r * ((double)rand() / RAND_MAX);
    
    double start[3] = {0, 1, 0};
    double target[3] = {
        dist * cos(angle),
        1 + 0.2 * ((double)rand()/RAND_MAX - 0.5),
        dist * sin(angle)
    };
    
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
    
    Net* net;
    if(argc == 3) {
        net = load_net(argv[2]);
    } else {
        int layers[] = {STATE_DIM, 64, ACTION_DIM};
        net = init_net(3, layers, 1e-5);
    }
    
    double theoretical_max = (1.0 - pow(GAMMA + 1e-15, MAX_STEPS))/(1.0 - (GAMMA + 1e-15));
    
    Rollout* rollouts[NUM_ROLLOUTS];
    for(int r = 0; r < NUM_ROLLOUTS; r++) {
        rollouts[r] = create_rollout();
    }

    int epochs = atoi(argv[1]);
    double best_return = -1e30;
    double initial_best = -1e30;
    struct timeval start_time, current_time;
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
        if(mean_return > best_return) {
            best_return = mean_return;
        }

        if(epoch == 0) {
            initial_best = best_return;
        }

        gettimeofday(&current_time, NULL);
        double elapsed = (current_time.tv_sec - start_time.tv_sec) + 
                        (current_time.tv_usec - start_time.tv_usec) / 1000000.0;
        
        double initial_percentage = (initial_best / theoretical_max) * 100.0;
        double current_percentage = (best_return / theoretical_max) * 100.0;
        double percentage_rate = (current_percentage - initial_percentage) / elapsed;

        printf("epoch %d/%d | Radius: %.3f | Return: %.2f/%.2f (%.1f%%) | Best: %.2f | Rate: %.3f %%/s\n", 
               epoch+1, epochs, 
               (epoch < 100) ? 0.01 : 0.01 + 0.99 * fmin(1.0, (epoch - 100.0)/(epochs - 100.0)),
               mean_return, theoretical_max, 
               (mean_return/theoretical_max) * 100.0, best_return,
               percentage_rate);
    }

    char final_weights[64];
    strftime(final_weights, sizeof(final_weights), "%Y%m%d_%H%M%S_policy.bin", 
             localtime(&(time_t){time(NULL)}));
    save_net(final_weights, net);
    printf("\nFinal weights saved to: %s\n", final_weights);

    for(int r = 0; r < NUM_ROLLOUTS; r++) {
        free_rollout(rollouts[r]);
    }

    free_net(net);

    return 0;
}