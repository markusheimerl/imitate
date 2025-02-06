#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>
#include "net.h"
#include "quad.h"

#define DT_PHYSICS (1.0/1000.0)
#define DT_CONTROL (1.0/60.0)

#define STATE_DIM 6      // 3 accel + 3 gyro
#define ACTION_DIM 8     // 4 means + 4 stds
#define MAX_STEPS 256
#define NUM_ROLLOUTS 1024

#define GAMMA 0.999
#define MAX_STD 3.0
#define MIN_STD 1e-5

#define MAX_MEAN (OMEGA_MAX - 4.0 * MAX_STD)
#define MIN_MEAN (OMEGA_MIN + 4.0 * MAX_STD)

typedef struct {
    double states[MAX_STEPS][STATE_DIM];
    double actions[MAX_STEPS][ACTION_DIM];
    double rewards[MAX_STEPS];
    double returns[MAX_STEPS];
    int length;
} Rollout;

__device__ __host__ double squash(double x, double min, double max) { 
    return ((max + min) / 2.0) + ((max - min) / 2.0) * tanh(x); 
}

__device__ __host__ double dsquash(double x, double min, double max) { 
    return ((max - min) / 2.0) * (1.0 - tanh(x) * tanh(x)); 
}

double compute_reward(const Quad* q) {
    double distance = sqrt(
        pow(q->linear_position_W[0] - 0.0, 2) +
        pow(q->linear_position_W[1] - 1.0, 2) +
        pow(q->linear_position_W[2] - 0.0, 2)
    );
    return exp(-4.0 * distance);
}

void collect_rollout(Net* policy, Rollout* rollout) {
    // Initialize environment (quadcopter) with starting state s_0
    Quad quad = create_quad(0.0, 1.0, 0.0);
    double t_control = 0.0;
    rollout->length = 0;

    // Terminate episode if:
    // 1. Max timesteps reached (T = MAX_STEPS)
    // 2. Terminal state reached (quadcopter too far from goal)
    while(rollout->length < MAX_STEPS) {
        // Check if terminal state reached
        if (sqrt(
            pow(quad.linear_position_W[0], 2) +
            pow(quad.linear_position_W[1] - 1.0, 2) +
            pow(quad.linear_position_W[2], 2)) > 1.0) {
            break;
        }

        // Environment step with physics simulation
        update_quad(&quad, DT_PHYSICS);
        t_control += DT_PHYSICS;
        
        // Control at lower frequency than physics simulation
        if (t_control >= DT_CONTROL) {
            int step = rollout->length;
            
            // Get state s_t (accelerometer and gyroscope readings)
            memcpy(rollout->states[step], quad.linear_acceleration_B_s, 3 * sizeof(double));
            memcpy(rollout->states[step] + 3, quad.angular_velocity_B_s, 3 * sizeof(double));

            // Get policy distribution parameters μ(s_t), σ(s_t)
            forward_net(policy, rollout->states[step]);
            
            // Sample actions a_t ~ π_θ(a|s) for each rotor
            // Using Gaussian policy: a_t = μ(s_t) + σ(s_t) * ε, where ε ~ N(0,1)
            for(int i = 0; i < 4; i++) {
                // Get μ(s_t) and σ(s_t) for this action dimension
                double mu = squash(policy->h[2][i], MIN_MEAN, MAX_MEAN);
                double sigma = squash(policy->h[2][i + 4], MIN_STD, MAX_STD);

                // Sample ε ~ N(0,1) using Box-Muller transform
                double u1 = (double)rand()/RAND_MAX;
                double u2 = (double)rand()/RAND_MAX;
                double epsilon = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                
                // Sample action: a_t = μ(s_t) + σ(s_t) * ε
                rollout->actions[step][i] = mu + sigma * epsilon;
                
                // Apply action to environment
                quad.omega_next[i] = rollout->actions[step][i];
            }
            
            // Get reward r_t from environment
            rollout->rewards[step] = compute_reward(&quad);
            rollout->length++;
            t_control = 0.0;
        }
    }
    
    // Compute returns (discounted sum of rewards)
    // R_t = Σ_{k=t}^T γ^{k-t} * r_k
    double G = 0.0;  // G represents R_t (return)
    for(int i = rollout->length-1; i >= 0; i--) {
        // G = r_t + γ * G
        G = rollout->rewards[i] + GAMMA * G;
        rollout->returns[i] = G;
    }
}

void* collection_thread(void* arg) {
    srand(time(NULL) ^ getpid());

    void** args = (void**)arg;
    Net* shared_net = (Net*)args[0];
    Rollout* shared_rollouts = (Rollout*)args[1];
    volatile bool* sync = (volatile bool*)args[2];
    
    Rollout* local_rollouts;
    cudaMallocManaged(&local_rollouts, NUM_ROLLOUTS * sizeof(Rollout));
    
    while(1) {
        for(int r = 0; r < NUM_ROLLOUTS; r++) {
            collect_rollout(shared_net, &local_rollouts[r]);
        }
        
        while(*sync);
        
        memcpy(shared_rollouts, local_rollouts, NUM_ROLLOUTS * sizeof(Rollout));
        memset(local_rollouts, 0, NUM_ROLLOUTS * sizeof(Rollout));
        *sync = true;
    }
}

// ∇J(θ) = E[∇_θ log π_θ(a|s) * R] ≈ 1/N Σ_t [∇_θ log π_θ(a_t|s_t) * R_t]
// Where:
// J(θ) - Policy objective function
// π_θ(a|s) - Gaussian policy parameterized by θ (network weights)
// R_t - Discounted return from time step t
void update_policy(Net* policy, Rollout* rollouts) {
    double output_gradients[ACTION_DIM];
    
    // For each timestep t
    for(int step = 0; step < MAX_STEPS; step++) {
        zero_gradients(policy);
        
        // Compute value baseline V(s_t) as mean of returns across rollouts
        double V_s = 0;
        int valid_rollouts = 0;
        for(int r = 0; r < NUM_ROLLOUTS; r++) {
            if(step < rollouts[r].length) {
                V_s += rollouts[r].returns[step];
                valid_rollouts++;
            }
        }
        V_s /= valid_rollouts;
        
        // For each rollout (sampling from policy π_θ)
        for(int r = 0; r < NUM_ROLLOUTS; r++) {
            if(step >= rollouts[r].length) continue;
            
            // Get policy distribution parameters for state s_t
            forward_net(policy, rollouts[r].states[step]);
            
            // Compute advantage estimate A(s_t,a_t) = R_t - V(s_t)
            // Where R_t is the return (discounted sum of rewards) from time t
            double advantage = rollouts[r].returns[step] - V_s;
            
            // For each action dimension (quadcopter has 4 rotors)
            for(int i = 0; i < 4; i++) {
                // Raw network outputs before squashing
                double mean_raw = policy->h[2][i];
                double std_raw = policy->h[2][i + 4];
                
                // Policy parameters μ(s_t) and σ(s_t) for Gaussian policy π_θ(a|s)
                double mu = squash(mean_raw, MIN_MEAN, MAX_MEAN);
                double sigma = squash(std_raw, MIN_STD, MAX_STD);
                
                // a_t - μ(s_t) for this action dimension
                double action_diff = rollouts[r].actions[step][i] - mu;

                // Compute ∇_θ log π_θ(a|s) for mean parameter
                // For Gaussian: ∂/∂μ log π_θ(a|s) = (a-μ)/σ²
                double score_mu = (action_diff / (sigma * sigma)) * 
                    dsquash(mean_raw, MIN_MEAN, MAX_MEAN);
                
                // Compute ∇_θ log π_θ(a|s) for standard deviation parameter
                // For Gaussian: ∂/∂σ log π_θ(a|s) = ((a-μ)²-σ²)/σ³
                double score_sigma = ((action_diff * action_diff - sigma * sigma) / 
                    (sigma * sigma * sigma)) * 
                    dsquash(std_raw, MIN_STD, MAX_STD);

                // Policy gradient theorem: ∇_θ J(θ) = E[∇_θ log π_θ(a|s) * A(s,a)]
                // Update = learning_rate * score_function * advantage
                output_gradients[i] = score_mu * advantage;
                output_gradients[i + 4] = score_sigma * advantage;
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

    srand(time(NULL) ^ getpid());
    
    Net* net = (argc == 3) ? load_net(argv[2]) : create_net(3e-7);
    Rollout* shared_rollouts;
    cudaMallocManaged(&shared_rollouts, NUM_ROLLOUTS * sizeof(Rollout));
    volatile bool sync = false;
    
    pthread_t collector;
    pthread_create(&collector, NULL, collection_thread, (void*[3]){net, shared_rollouts, (void*)&sync});

    int num_epochs = atoi(argv[1]);
    double best_return = -1e30;
    double theoretical_max = (1.0 - pow(GAMMA + 1e-15, MAX_STEPS))/(1.0 - (GAMMA + 1e-15));
    struct timeval start_time;
    gettimeofday(&start_time, NULL);
    
    Rollout* local_rollouts;
    cudaMallocManaged(&local_rollouts, NUM_ROLLOUTS * sizeof(Rollout));
    
    for(int epoch = 0; epoch < num_epochs; epoch++) {
        while(!sync);
        
        memcpy(local_rollouts, shared_rollouts, NUM_ROLLOUTS * sizeof(Rollout));
        sync = false;

        double mean_return = 0.0;
        for(int r = 0; r < NUM_ROLLOUTS; r++) {
            mean_return += local_rollouts[r].returns[0];
        }
        mean_return /= NUM_ROLLOUTS;
        
        update_policy(net, local_rollouts);
        best_return = fmax(mean_return, best_return);

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

    char filename[64];
    time_t current_time = time(NULL);
    strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_policy.bin", localtime(&current_time));
    save_net(filename, net);

    return 0;
}