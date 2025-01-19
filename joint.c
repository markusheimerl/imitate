#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include "grad/grad.h"
#include "sim/sim.h"

#define DT_PHYSICS (1.0/1000.0)
#define DT_CONTROL (1.0/60.0)
#define MAX_DISTANCE 2.0
#define MAX_VELOCITY 5.0
#define MAX_ANGULAR_VELOCITY 5.0

#define STATE_DIM 12
#define HIDDEN_DIM 64
#define ACTION_DIM 8

#define MAX_STEPS 1000
#define NUM_ROLLOUTS 50
#define NUM_ITERATIONS 10

// Hyperparameters
#define GAMMA 0.999
#define ALPHA 0.001
#define MAX_STD 3.0
#define MIN_STD 0.0001

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
    // Calculate position error (distance from target hover point)
    double pos_error = 0.0;
    for(int i = 0; i < 3; i++) {
        pos_error += pow(q->linear_position_W[i] - TARGET_POS[i], 2);
    }
    pos_error = sqrt(pos_error);  // Convert to actual distance
    
    // Calculate velocity magnitude (should be 0 for perfect hover)
    double vel_magnitude = 0.0;
    for(int i = 0; i < 3; i++) {
        vel_magnitude += pow(q->linear_velocity_W[i], 2);
    }
    vel_magnitude = sqrt(vel_magnitude);
    
    // Calculate angular velocity magnitude (should be 0 for perfect hover)
    double ang_vel_magnitude = 0.0;
    for(int i = 0; i < 3; i++) {
        ang_vel_magnitude += pow(q->angular_velocity_B[i], 2);
    }
    ang_vel_magnitude = sqrt(ang_vel_magnitude);
    
    // Calculate orientation error (should be upright, so R_W_B[4] should be 1.0)
    double orientation_error = 1.0 - q->R_W_B[4];  // Will be 0 when perfectly upright
    
    // Combine all errors with appropriate scaling
    double total_error = (pos_error * 2.0) +          // Position has highest weight
                        (vel_magnitude * 1.0) +        // Velocity is second
                        (ang_vel_magnitude * 0.5) +    // Angular velocity third
                        (orientation_error * 2.0);     // Orientation also important
    
    // Convert error to reward (1.0 at perfect hover, decreasing but always positive)
    double reward = exp(-total_error);
    
    return reward;
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
                // Get standard deviation first using squash:
                // σ = ((max + min)/2) + ((max - min)/2) * tanh(x)
                // MAX_STD ensures ±4σ stays within action range
                // MIN_STD allows for precise control when needed
                double std = squash(act[4][i + 4], MIN_STD, MAX_STD);
                
                // Compute dynamic bounds for mean based on current std
                // To ensure x = μ ± 4σ stays within [OMEGA_MIN, OMEGA_MAX]:
                // OMEGA_MIN ≤ μ - 4σ and μ + 4σ ≤ OMEGA_MAX
                // Therefore: OMEGA_MIN + 4σ ≤ μ ≤ OMEGA_MAX - 4σ
                double safe_margin = 4.0 * std; // 99.994% within ±4σ
                double mean_min = OMEGA_MIN + safe_margin;
                double mean_max = OMEGA_MAX - safe_margin;
                
                // Get mean using squash with dynamic bounds:
                // μ = ((max + min)/2) + ((max - min)/2) * tanh(x)
                double mean = squash(act[4][i], mean_min, mean_max);
                
                // Sample from N(μ, σ²) using Box-Muller transform:
                // If U₁,U₂ ~ Uniform(0,1)
                // Then √(-2ln(U₁))cos(2πU₂) ~ N(0,1)
                // And μ + σ√(-2ln(U₁))cos(2πU₂) ~ N(μ,σ²)
                double u1 = (double)rand()/RAND_MAX;
                double u2 = (double)rand()/RAND_MAX;
                double noise = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                
                // Store and apply sampled action
                // Guaranteed to be within [OMEGA_MIN, OMEGA_MAX] by construction
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
    // G_t = r_t + γr_{t+1} + γ²r_{t+2} + ...
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
            // 1. Get std and compute dynamic mean bounds
            // std range: [MIN_STD, MAX_STD]
            double std = squash(act[4][i + 4], MIN_STD, MAX_STD);
            double safe_margin = 4.0 * std; // 99.994% within ±4σ
            double mean_min = OMEGA_MIN + safe_margin;
            double mean_max = OMEGA_MAX - safe_margin;
            double mean = squash(act[4][i], mean_min, mean_max);
            
            // 2. Compute normalized action (z-score)
            // z = (x - μ)/σ
            double z = (actions[t][i] - mean) / std;
            
            // 3. Compute log probability of the action
            // log(p(x)) = -0.5 * (log(2π) + 2log(σ) + z²)
            double log_prob = -0.5 * (1.8378770664093453 + 2.0 * log(std) + z * z);
            
            // 4. Compute entropy of the Gaussian distribution
            // H = 0.5 * (log(2πe) + 2log(σ))
            double entropy = 0.5 * (2.837877066 + 2.0 * log(std));
            
            // 5. Compute gradient for mean
            // Direct effect: ∂log_prob/∂μ = z/σ
            // Through squash: ∂μ/∂x = dsquash(x, mean_min, mean_max)
            double dmean = z / std;
            grad[4][i] = (returns[t] * log_prob + ALPHA * entropy) * dmean * dsquash(act[4][i], mean_min, mean_max);
            
            // 6. Compute gradient for std
            // Direct effect on log_prob: ∂log_prob/∂σ = (z² - 1)/σ
            // Effect through mean bounds:
            // ∂mean/∂σ = -4.0 * dsquash(x, mean_min, mean_max)
            // Total: ∂log_prob/∂σ = (z² - 1)/σ + (z/σ) * (-4.0 * dsquash)
            double dstd_direct = (z * z - 1.0) / std;
            double dmean_dstd = -4.0 * dsquash(act[4][i], mean_min, mean_max);
            double dstd = dstd_direct + (z / std) * dmean_dstd;
            
            // Effect through entropy and squash function
            grad[4][i + 4] = (returns[t] * log_prob * dstd + ALPHA * (1.0 / std)) * dsquash(act[4][i + 4], MIN_STD, MAX_STD);
        }
        
        // Backward pass to update policy parameters
        bwd(policy, act, grad);
    }
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <sys/time.h>
#include "grad/grad.h"

#define NUM_PROCESSES 8
#define ELITE_COUNT 2

typedef struct {
    double mean_return;
    double std_return;
    pid_t pid;
    char weights_file[64];
} ProcessResult;

void interpolate_weights(Net* net, Net* elite, double alpha) {
    for(int i = 0; i < net->n; i++) {
        int in = net->sz[i], out = net->sz[i+1];
        for(int j = 0; j < in*out; j++) {
            net->w[i][j] = alpha * net->w[i][j] + (1.0 - alpha) * elite->w[i][j];
        }
        for(int j = 0; j < out; j++) {
            net->b[i][j] = alpha * net->b[i][j] + (1.0 - alpha) * elite->b[i][j];
        }
    }
}

int compare_results(const void* a, const void* b) {
    double diff = ((ProcessResult*)b)->mean_return - ((ProcessResult*)a)->mean_return;
    return diff > 0 ? 1 : (diff < 0 ? -1 : 0);
}

int run_reinforce(int argc, char** argv) {
    srand(time(NULL) ^ getpid());
    
    int layers[] = {STATE_DIM, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, ACTION_DIM};
    Net* policy = init_net(5, layers, adamw);
    if(!policy) return 1;
    if(argc > 1) {
        Net* loaded = load_weights(argv[1], adamw);
        if(loaded) {
            free_net(policy);
            policy = loaded;
        }
    }
    policy->lr = 1e-4;
    
    Sim* sim = init_sim("", false);
    
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
    
    if(argc > 1) {
        save_weights(argv[1], policy);  // Save back to the same file
    } else {
        char filename[64];
        strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_policy.bin", 
                localtime(&(time_t){time(NULL)}));
        save_weights(filename, policy);
    }
    
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

int main(int argc, char** argv) {
    if(argc != 2 && argc != 3) {
        printf("Usage: %s <num_generations> [initial_weights.bin]\n", argv[0]);
        return 1;
    }
    
    srand(time(NULL) ^ getpid());
    
    ProcessResult* results = mmap(NULL, NUM_PROCESSES * sizeof(ProcessResult),
                                PROT_READ | PROT_WRITE,
                                MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    
    for(int i = 0; i < NUM_PROCESSES; i++) {
        sprintf(results[i].weights_file, "weights_%d.bin", i);
    }
    
    char final_weights[64];
    strftime(final_weights, sizeof(final_weights), "%Y%m%d_%H%M%S_policy.bin", 
             localtime(&(time_t){time(NULL)}));
    
    if(argc == 3) {
        // If initial weights provided, copy them to our new file
        char command[256];
        sprintf(command, "cp %s %s", argv[2], final_weights);
        if(system(command) != 0) {
            fprintf(stderr, "Failed to copy initial weights from %s\n", argv[2]);
            return 1;
        }
        printf("Starting from weights in %s\n", argv[2]);
        printf("Will save results to %s\n", final_weights);
    } else {
        // Create fresh initial weights
        Net* initial_net = init_net(5, (int[]){12, 64, 64, 64, 8}, adamw);
        if(!initial_net) return 1;
        save_weights(final_weights, initial_net);
        free_net(initial_net);
        printf("Starting fresh, will save results to %s\n", final_weights);
    }
    
    int generations = atoi(argv[1]);
    for(int gen = 0; gen < generations; gen++) {
        printf("\nGeneration %d/%d\n", gen + 1, generations);
        
        int pipes[NUM_PROCESSES][2];
        
        for(int i = 0; i < NUM_PROCESSES; i++) {
            if(pipe(pipes[i]) == -1) {
                perror("pipe creation failed");
                exit(1);
            }
            
            if(fork() == 0) {
                close(pipes[i][0]);
                dup2(pipes[i][1], STDOUT_FILENO);
                close(pipes[i][1]);
                
                Net* net;
                if(gen == 0) {
                    // First generation: everyone starts from initial weights
                    net = load_weights(final_weights, adamw);
                    if(!net) {
                        fprintf(stderr, "Failed to load weights from %s\n", final_weights);
                        exit(1);
                    }
                } else {
                    // Everyone loads their previous weights
                    net = load_weights(results[i].weights_file, adamw);
                    if(i >= ELITE_COUNT) {
                        // Non-elites interpolate with the best
                        Net* elite = load_weights(final_weights, adamw);
                        interpolate_weights(net, elite, 0.5);  // Keep 50% of own weights
                        free_net(elite);
                    }
                }
                
                save_weights(results[i].weights_file, net);
                free_net(net);
                
                char* reinforce_args[] = {"reinforce", results[i].weights_file, NULL};
                exit(run_reinforce(2, reinforce_args));
            }
            
            close(pipes[i][1]);
        }
        
        char buf[256], last_iteration[256] = {0};
        for(int i = 0; i < NUM_PROCESSES; i++) {
            FILE* fp = fdopen(pipes[i][0], "r");
            while(fgets(buf, sizeof(buf), fp)) {
                if(strstr(buf, "Iteration")) {
                    strcpy(last_iteration, buf);
                }
            }
            if(last_iteration[0]) {
                sscanf(last_iteration, "Iteration %*d/%*d [n=%*d]: %lf ± %lf",
                       &results[i].mean_return, &results[i].std_return);
            }
            fclose(fp);
            wait(NULL);
        }
        
        qsort(results, NUM_PROCESSES, sizeof(ProcessResult), compare_results);
        
        printf("\nGeneration Results:\n");
        for(int i = 0; i < NUM_PROCESSES; i++) {
            printf("Agent %d: %.2f ± %.2f\n", i, 
                   results[i].mean_return, results[i].std_return);
        }
        
        char command[256];
        sprintf(command, "cp %s %s", results[0].weights_file, final_weights);
        system(command);
    }
    
    for(int i = 0; i < NUM_PROCESSES; i++) {
        remove(results[i].weights_file);
    }
    
    munmap(results, NUM_PROCESSES * sizeof(ProcessResult));
    return 0;
}