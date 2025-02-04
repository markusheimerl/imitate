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
    double states[MAX_STEPS][STATE_DIM];
    double actions[MAX_STEPS][ACTION_DIM];
    double rewards[MAX_STEPS];
    double returns[MAX_STEPS];
    int length;
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

void collect_rollout(Net* policy, Rollout* rollout) {
    Quad quad = create_quad(0.0, 1.0, 0.0);
    double t_control = 0.0;
    rollout->length = 0;
    double input[INPUT_DIM] = {0};

    while(rollout->length < MAX_STEPS) {
        if (sqrt(
            pow(quad.linear_position_W[0], 2) +
            pow(quad.linear_position_W[1] - 1.0, 2) +
            pow(quad.linear_position_W[2], 2)) > 1.0) {
            break;
        }

        update_quad(&quad, DT_PHYSICS);
        t_control += DT_PHYSICS;
        
        if (t_control >= DT_CONTROL) {
            int step = rollout->length;
            
            memcpy(rollout->states[step], quad.linear_acceleration_B_s, 3 * sizeof(double));
            memcpy(rollout->states[step] + 3, quad.angular_velocity_B_s, 3 * sizeof(double));
            
            // Fill input array with history using exponential decay
            for(int i = 0; i < 16 && step - i >= 0; i++)
                for(int j = 0; j < 6; j++) 
                    input[i * 6 + j] = rollout->states[step - i][j] * pow(0.85, i);
                
            forward_net(policy, input);
            
            for(int i = 0; i < 4; i++) {
                double mean = squash(policy->h[2][i], MIN_MEAN, MAX_MEAN);
                double std = squash(policy->h[2][i + 4], MIN_STD, MAX_STD);

                double u1 = (double)rand()/RAND_MAX;
                double u2 = (double)rand()/RAND_MAX;
                double noise = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                
                rollout->actions[step][i] = mean + std * noise;
                quad.omega_next[i] = rollout->actions[step][i];
            }
            
            rollout->rewards[step] = compute_reward(&quad);
            rollout->length++;
            t_control = 0.0;
        }
    }
    
    // Compute discounted returns
    double G = 0.0;
    for(int i = rollout->length-1; i >= 0; i--) {
        G = rollout->rewards[i] + GAMMA * G;
        rollout->returns[i] = G;
    }
}

// ∇J(θ) = E[∇_θ log π_θ(a|s) * R] ≈ 1/N Σ_t [∇_θ log π_θ(a_t|s_t) * R_t]
// Where:
// J(θ) - Policy objective function
// π_θ(a|s) - Gaussian policy parameterized by θ (network weights)
// R_t - Discounted return from time step t
void update_policy(Net* policy, Rollout* rollout) {
    double output_gradient[ACTION_DIM];
    double input[INPUT_DIM] = {0};
    
    for(int step = 0; step < rollout->length; step++) {
        // Fill input array with history using exponential decay
        for(int i = 0; i < 16 && step - i >= 0; i++) 
            for(int j = 0; j < 6; j++) 
                input[i * 6 + j] = rollout->states[step - i][j] * pow(0.85, i);
            
        forward_net(policy, input);
        
        for(int i = 0; i < 4; i++) {
            // Network outputs raw parameters before squashing
            double mean_raw = policy->h[2][i];
            double std_raw = policy->h[2][i + 4];
            
            // Squashed parameters using tanh-based scaling
            // μ = ((MAX+MIN)/2) + ((MAX-MIN)/2)*tanh(mean_raw)
            // σ = ((MAX_STD+MIN_STD)/2) + ((MAX_STD-MIN_STD)/2)*tanh(std_raw)
            double mean = squash(mean_raw, MIN_MEAN, MAX_MEAN);
            double std_val = squash(std_raw, MIN_STD, MAX_STD);
            
            // Sampled action and its deviation from mean
            double delta = rollout->actions[step][i] - mean;

            // Gradient for mean parameter:
            // ∇_{μ_raw} log π = [ (a - μ)/σ² ] * dμ/dμ_raw
            // Where:
            // (a - μ)/σ² = derivative of log N(a; μ, σ²) w.r.t μ
            // dμ/dμ_raw = derivative of squashing function (dsquash)
            output_gradient[i] = -(delta / (std_val * std_val)) * 
                dsquash(mean_raw, MIN_MEAN, MAX_MEAN) * 
                rollout->returns[step];

            // Gradient for standard deviation parameter:
            // ∇_{σ_raw} log π = [ ( (a-μ)^2 - σ² ) / σ³ ] * dσ/dσ_raw
            // Where:
            // ( (a-μ)^2 - σ² ) / σ³ = derivative of log N(a; μ, σ²) w.r.t σ
            // dσ/dσ_raw = derivative of squashing function (dsquash)
            output_gradient[i + 4] = -((delta * delta - std_val * std_val) / 
                (std_val * std_val * std_val)) * 
                dsquash(std_raw, MIN_STD, MAX_STD) * 
                rollout->returns[step];
        }

        backward_net(policy, output_gradient);
    }
}

void* collection_thread(void* arg) {
    srand(time(NULL) ^ getpid());

    void** args = (void**)arg;
    Net* shared_net = (Net*)args[0];
    Rollout* shared_rollouts = (Rollout*)args[1];
    volatile bool* sync = (volatile bool*)args[2];
    
    Rollout local_rollouts[NUM_ROLLOUTS];
    
    while(1) {
        for(int r = 0; r < NUM_ROLLOUTS; r++) collect_rollout(shared_net, &local_rollouts[r]);
        
        unsigned long long local_count = 0;
        while(*sync) local_count++;
        
        memcpy(shared_rollouts, local_rollouts, sizeof(Rollout) * NUM_ROLLOUTS);
        *sync = true;
    }
    return NULL;
}

void* update_thread(void* arg) {
    srand(time(NULL) ^ getpid());

    void** args = (void**)arg;
    Net* shared_net = (Net*)args[0];
    Rollout* shared_rollouts = (Rollout*)args[1];
    volatile bool* sync = (volatile bool*)args[2];
    double* shared_mean_return = (double*)args[3];

    Net* local_net = create_net(shared_net->lr);
    memcpy(local_net, shared_net, sizeof(Net));
    
    Rollout local_rollouts[NUM_ROLLOUTS];
    
    while(1) {
        double local_mean_return = 0.0;
        for(int r = 0; r < NUM_ROLLOUTS; r++) local_mean_return += local_rollouts[r].returns[0];
        local_mean_return /= NUM_ROLLOUTS;

        unsigned long long local_count = 0;
        while(!(*sync)) local_count++;

        memcpy(local_rollouts, shared_rollouts, sizeof(Rollout) * NUM_ROLLOUTS);
        memcpy(shared_net, local_net, sizeof(Net));
        memcpy(shared_mean_return, &local_mean_return, sizeof(double));
        *sync = false;

        for(int r = 0; r < NUM_ROLLOUTS; r++) {
            zero_gradients(local_net);
            update_policy(local_net, &local_rollouts[r]);
            update_net(local_net);
        }
    }
    
    free_net(local_net);
    return NULL;
}

int main(int argc, char** argv) {
    if(argc != 2 && argc != 3) {
        printf("Usage: %s <num_timesteps> [initial_weights.bin]\n", argv[0]);
        return 1;
    }

    srand(time(NULL) ^ getpid());
    
    Net* net = (argc == 3) ? load_net(argv[2]) : create_net(2e-7);
    Rollout shared_rollouts[NUM_ROLLOUTS];
    volatile bool sync = false;
    double shared_mean_return = 0.0;
    
    void* collection_args[] = {net, shared_rollouts, (void*)&sync};
    void* update_args[] = {net, shared_rollouts, (void*)&sync, &shared_mean_return};
    
    pthread_t collector, updater;
    pthread_create(&collector, NULL, collection_thread, collection_args);
    pthread_create(&updater, NULL, update_thread, update_args);

    int timesteps = atoi(argv[1]);
    double best_return = -1e30;
    double theoretical_max = (1.0 - pow(GAMMA + 1e-15, MAX_STEPS))/(1.0 - (GAMMA + 1e-15));
    struct timeval start_time;
    gettimeofday(&start_time, NULL);
    
    for(int timestep = 0; timestep < timesteps; timestep++) {
        sleep(1);
        while(sync);
        
        double local_mean_return;
        memcpy(&local_mean_return, &shared_mean_return, sizeof(double));
        
        best_return = fmax(local_mean_return, best_return);

        struct timeval now;
        gettimeofday(&now, NULL);
        double elapsed = (now.tv_sec - start_time.tv_sec) + 
                        (now.tv_usec - start_time.tv_usec) / 1e6;
        
        printf("Timestep %d/%d | Return: %.2f/%.2f (%.1f%%) | Best: %.2f | Rate: %.3f %%/s\n", 
            timestep+1, timesteps,
            local_mean_return, theoretical_max, 
            (local_mean_return/theoretical_max) * 100.0, best_return,
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