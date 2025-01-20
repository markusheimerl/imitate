#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "grad/grad.h"
#include "sim/sim.h"
#include <sys/time.h>

#define DT_PHYSICS (1.0/1000.0)
#define DT_CONTROL (1.0/60.0)
#define MAX_DISTANCE 2.0
#define MAX_VELOCITY 5.0
#define MAX_ANGULAR_VELOCITY 5.0

#define INPUT_DIM 15
#define HIDDEN_DIM 64
#define ACTION_DIM 8

#define MAX_STEPS 1000
#define NUM_ROLLOUTS 128
#define GAMMA 0.999
#define ALPHA 1e-5

#define INITIAL_MAX_STD 4.0
#define FINAL_MAX_STD 1e-5
#define MIN_STD 1e-6
#define MAXIMUM_LEARNING_RATE 1e-4
#define MINIMUM_LEARNING_RATE 1e-8
#define TARGET_OFFSET 0.02

const double TARGET_POS[3] = {0.0, 1.0, 0.0};

void get_state(Quad* q, double* state, const double* target) {
    memcpy(state, q->linear_position_W, 3 * sizeof(double));
    memcpy(state + 3, q->linear_velocity_W, 3 * sizeof(double));
    memcpy(state + 6, q->angular_velocity_B, 3 * sizeof(double));
    state[9] = q->R_W_B[0];
    state[10] = q->R_W_B[4];
    state[11] = q->R_W_B[8];
    memcpy(state + 12, target, 3 * sizeof(double));
}

double compute_reward(Quad* q, const double* target) {
    double pos_error = 0.0;
    for(int i = 0; i < 3; i++) {
        pos_error += pow(q->linear_position_W[i] - target[i], 2);
    }
    pos_error = sqrt(pos_error);
    
    double vel_magnitude = 0.0;
    for(int i = 0; i < 3; i++) {
        vel_magnitude += pow(q->linear_velocity_W[i], 2);
    }
    vel_magnitude = sqrt(vel_magnitude);
    
    double ang_vel_magnitude = 0.0;
    for(int i = 0; i < 3; i++) {
        ang_vel_magnitude += pow(q->angular_velocity_B[i], 2);
    }
    ang_vel_magnitude = sqrt(ang_vel_magnitude);
    
    double orientation_error = fabs(1.0 - q->R_W_B[4]);
    
    double total_error = (pos_error * 2.0) +
                        (vel_magnitude * 1.0) +
                        (ang_vel_magnitude * 0.5) +
                        (orientation_error * 2.0);
    
    return exp(-total_error);
}

bool is_terminated(Quad* q, const double* target) {
    double dist = 0.0, vel = 0.0, ang_vel = 0.0;
    for(int i = 0; i < 3; i++) {
        dist += pow(q->linear_position_W[i] - target[i], 2);
        vel += pow(q->linear_velocity_W[i], 2);
        ang_vel += pow(q->angular_velocity_B[i], 2);
    }
    return sqrt(dist) > MAX_DISTANCE || 
           sqrt(vel) > MAX_VELOCITY || 
           sqrt(ang_vel) > MAX_ANGULAR_VELOCITY || 
           q->R_W_B[4] < 0.0;
}

int collect_rollout(Sim* sim, Net* policy, double** act, double** states, double** actions, double* rewards, double current_max_std) {
    double target[3] = {
        TARGET_POS[0] + ((double)rand()/RAND_MAX - 0.5) * TARGET_OFFSET,
        TARGET_POS[1] + ((double)rand()/RAND_MAX - 0.5) * TARGET_OFFSET,
        TARGET_POS[2] + ((double)rand()/RAND_MAX - 0.5) * TARGET_OFFSET
    };
    
    reset_quad(sim->quad, 
        target[0] + ((double)rand()/RAND_MAX - 0.5) * 0.2,
        target[1] + ((double)rand()/RAND_MAX - 0.5) * 0.2, 
        target[2] + ((double)rand()/RAND_MAX - 0.5) * 0.2
    );
    
    double t_physics = 0.0, t_control = 0.0;
    int steps = 0;
    
    while(steps < MAX_STEPS && !is_terminated(sim->quad, target)) {
        update_quad(sim->quad, DT_PHYSICS);
        t_physics += DT_PHYSICS;
        
        if(t_control <= t_physics) {
            get_state(sim->quad, states[steps], target);
            fwd(policy, states[steps], act);
            
            for(int i = 0; i < 4; i++) {
                double std = squash(act[4][i + 4], MIN_STD, current_max_std);
                double safe_margin = 4.0 * std;
                double mean_min = OMEGA_MIN + safe_margin;
                double mean_max = OMEGA_MAX - safe_margin;
                double mean = squash(act[4][i], mean_min, mean_max);
                
                double u1 = (double)rand()/RAND_MAX;
                double u2 = (double)rand()/RAND_MAX;
                double noise = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                
                actions[steps][i] = mean + std * noise;
                sim->quad->omega_next[i] = actions[steps][i];
            }
            
            rewards[steps] = compute_reward(sim->quad, target);
            steps++;
            t_control += DT_CONTROL;
        }
    }
    
    double G = 0.0;
    for(int i = steps-1; i >= 0; i--) {
        rewards[i] = G = rewards[i] + GAMMA * G;
    }
    
    return steps;
}

void update_policy(Net* policy, double** states, double** actions, double* returns, int steps, double** act, double** grad, double current_max_std) {
    for(int t = 0; t < steps; t++) {
        fwd(policy, states[t], act);
        
        for(int i = 0; i < 4; i++) {
            double std = squash(act[4][i + 4], MIN_STD, current_max_std);
            double safe_margin = 4.0 * std;
            double mean_min = OMEGA_MIN + safe_margin;
            double mean_max = OMEGA_MAX - safe_margin;
            double mean = squash(act[4][i], mean_min, mean_max);
            
            double z = (actions[t][i] - mean) / std;
            double log_prob = -0.5 * (1.8378770664093453 + 2.0 * log(std) + z * z);
            double entropy = 0.5 * (2.837877066 + 2.0 * log(std));
            
            double dmean = z / std;
            grad[4][i] = (returns[t] * log_prob + ALPHA * entropy) * dmean * 
                        dsquash(act[4][i], mean_min, mean_max);
            
            double dstd_direct = (z * z - 1.0) / std;
            double dmean_dstd = -4.0 * dsquash(act[4][i], mean_min, mean_max);
            double dstd = dstd_direct + (z / std) * dmean_dstd;
            
            grad[4][i + 4] = (returns[t] * log_prob * dstd + ALPHA * (1.0 / std)) * 
                            dsquash(act[4][i + 4], MIN_STD, current_max_std);
        }
        
        bwd(policy, act, grad);
    }
}

int main(int argc, char** argv) {
    if(argc != 2 && argc != 3) {
        printf("Usage: %s <num_iterations> [initial_weights.bin]\n", argv[0]);
        return 1;
    }

    srand(time(NULL));
    
    Net* net;
    if(argc == 3) {
        net = load_weights(argv[2], adamw);
    } else {
        int layers[] = {INPUT_DIM, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, ACTION_DIM};
        net = init_net(5, layers, adamw);
    }

    const int num_threads = omp_get_max_threads();
    Sim** sims = malloc(num_threads * sizeof(Sim*));
    double*** acts = malloc(num_threads * sizeof(double**));
    double*** grads = malloc(num_threads * sizeof(double**));
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        sims[tid] = init_sim("", false);
        acts[tid] = malloc(5 * sizeof(double*));
        grads[tid] = malloc(5 * sizeof(double*));
        for(int i = 0; i < 5; i++) {
            acts[tid][i] = malloc(net->sz[i] * sizeof(double));
            grads[tid][i] = calloc(net->sz[i], sizeof(double));
        }
    }

    int iterations = atoi(argv[1]);
    double best_return = -1e30;
    double initial_best = -1e30;
    struct timeval start_time, current_time;
    gettimeofday(&start_time, NULL);
    
    double theoretical_max = (1.0 - pow(GAMMA + 1e-15, MAX_STEPS))/(1.0 - (GAMMA + 1e-15));
    double current_lr = MINIMUM_LEARNING_RATE;
    double current_max_std = INITIAL_MAX_STD;
    
    for(int iter = 0; iter < iterations; iter++) {
        double sum_returns = 0.0;
        int total_steps = 0;
        
        #pragma omp parallel reduction(+:sum_returns,total_steps)
        {
            int tid = omp_get_thread_num();
            
            #pragma omp for schedule(dynamic)
            for(int r = 0; r < NUM_ROLLOUTS; r++) {
                double* states[MAX_STEPS];
                double* actions[MAX_STEPS];
                double rewards[MAX_STEPS];
                
                for(int i = 0; i < MAX_STEPS; i++) {
                    states[i] = malloc(INPUT_DIM * sizeof(double));
                    actions[i] = malloc(4 * sizeof(double));
                }

                int steps = collect_rollout(sims[tid], net, acts[tid], states, actions, rewards, current_max_std);
                sum_returns += rewards[0];
                total_steps += steps;

                #pragma omp critical
                {
                    update_policy(net, states, actions, rewards, steps, acts[tid], grads[tid], current_max_std);
                }

                for(int i = 0; i < MAX_STEPS; i++) {
                    free(states[i]);
                    free(actions[i]);
                }
            }
        }

        double mean_return = sum_returns / NUM_ROLLOUTS;
        if(mean_return > best_return) {
            best_return = mean_return;
            save_weights("best_policy.bin", net);
        }

        if(iter == 0) {
            initial_best = best_return;
        }

        gettimeofday(&current_time, NULL);
        double elapsed = (current_time.tv_sec - start_time.tv_sec) + 
                        (current_time.tv_usec - start_time.tv_usec) / 1000000.0;
        
        double performance_ratio = best_return/theoretical_max;
        double initial_percentage = (initial_best / theoretical_max) * 100.0;
        double current_percentage = (best_return / theoretical_max) * 100.0;
        double percentage_rate = (current_percentage - initial_percentage) / elapsed;

        current_lr = MAXIMUM_LEARNING_RATE * (1.0 - (performance_ratio + mean_return/theoretical_max)/2) + 
                    MINIMUM_LEARNING_RATE;
        current_max_std = INITIAL_MAX_STD * (1.0 - (performance_ratio + mean_return/theoretical_max)/2) + 
                         FINAL_MAX_STD;
        
        net->lr = current_lr;

        printf("\rIter %d/%d | Return: %.2f / %.2f (%.1f%%) | Best: %.2f | Rate: %.3f %%/s | lr: %.2e | std: %.2f", 
               iter+1, iterations, mean_return, theoretical_max, 
               (mean_return/theoretical_max) * 100.0, best_return, percentage_rate,
               current_lr, current_max_std);
        fflush(stdout);
    }
    printf("\n");

    char final_weights[64];
    strftime(final_weights, sizeof(final_weights), "%Y%m%d_%H%M%S_policy.bin", 
             localtime(&(time_t){time(NULL)}));
    save_weights(final_weights, net);
    printf("Final weights saved to: %s\n", final_weights);

    // Cleanup
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        for(int i = 0; i < 5; i++) {
            free(acts[tid][i]);
            free(grads[tid][i]);
        }
        free(acts[tid]);
        free(grads[tid]);
        free_sim(sims[tid]);
    }
    free(sims);
    free(acts);
    free(grads);
    free_net(net);

    return 0;
}