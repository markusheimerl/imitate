#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>
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
#define NUM_ROLLOUTS 40
#define NUM_ITERATIONS 8
#define NUM_PROCESSES 8
#define ELITE_COUNT 2

#define GAMMA 0.999
#define ALPHA 0.001
#define MAX_STD 3.0
#define MIN_STD 0.0001
#define STARTING_LEARNING_RATE 1e-4
#define MINIMUM_LEARNING_RATE 1e-8

#define SHARED_NET_SIZE (sizeof(SharedNet))

const double TARGET_POS[3] = {0.0, 1.0, 0.0};

typedef struct {
    double mean_return;
    double std_return;
} ProcessResult;

typedef struct {
    int n_layers;
    int layer_sizes[10];
    double learning_rate;
    int step;
    double weights[100000];
    double biases[1000];
    double optimizer_data[100000];
    int weight_offsets[10];
    int bias_offsets[10];
    int optim_offsets[10];
    int total_weights;
    int total_biases;
    int total_optim_data;
} SharedNet;

void get_state(Quad* q, double* state) {
    memcpy(state, q->linear_position_W, 3 * sizeof(double));
    memcpy(state + 3, q->linear_velocity_W, 3 * sizeof(double));
    memcpy(state + 6, q->angular_velocity_B, 3 * sizeof(double));
    state[9] = q->R_W_B[0];
    state[10] = q->R_W_B[4];
    state[11] = q->R_W_B[8];
}

double compute_reward(Quad* q) {
    double pos_error = 0.0;
    for(int i = 0; i < 3; i++) {
        pos_error += pow(q->linear_position_W[i] - TARGET_POS[i], 2);
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

bool is_terminated(Quad* q) {
    double dist = 0.0, vel = 0.0, ang_vel = 0.0;
    for(int i = 0; i < 3; i++) {
        dist += pow(q->linear_position_W[i] - TARGET_POS[i], 2);
        vel += pow(q->linear_velocity_W[i], 2);
        ang_vel += pow(q->angular_velocity_B[i], 2);
    }
    return sqrt(dist) > MAX_DISTANCE || 
           sqrt(vel) > MAX_VELOCITY || 
           sqrt(ang_vel) > MAX_ANGULAR_VELOCITY || 
           q->R_W_B[4] < 0.0;
}

void net_to_shared(Net* net, SharedNet* shared) {
    shared->n_layers = net->n + 1;
    if(shared->n_layers > 10) {
        fprintf(stderr, "Too many layers\n");
        exit(1);
    }
    
    memcpy(shared->layer_sizes, net->sz, shared->n_layers * sizeof(int));
    shared->learning_rate = net->lr;
    shared->step = net->step;
    
    int w_offset = 0, b_offset = 0, o_offset = 0;
    for(int i = 0; i < net->n; i++) {
        int in = net->sz[i], out = net->sz[i+1];
        
        // Add size checks
        if(w_offset + in * out > 100000 || 
           b_offset + out > 1000 || 
           o_offset + (in * out + out) * net->optimizer.aux_doubles_per_param > 100000) {
            fprintf(stderr, "Network too large for shared memory\n");
            exit(1);
        }
        
        shared->weight_offsets[i] = w_offset;
        shared->bias_offsets[i] = b_offset;
        shared->optim_offsets[i] = o_offset;
        
        memcpy(&shared->weights[w_offset], net->w[i], in * out * sizeof(double));
        memcpy(&shared->biases[b_offset], net->b[i], out * sizeof(double));
        
        int optim_size = (in * out + out) * net->optimizer.aux_doubles_per_param;
        if(optim_size > 0) {
            memcpy(&shared->optimizer_data[o_offset], net->o_data[i], 
                   optim_size * sizeof(double));
        }
        
        w_offset += in * out;
        b_offset += out;
        o_offset += optim_size;
    }
    shared->total_weights = w_offset;
    shared->total_biases = b_offset;
    shared->total_optim_data = o_offset;
}

Net* shared_to_net(SharedNet* shared) {
    if(shared->n_layers > 10) {
        fprintf(stderr, "Too many layers\n");
        return NULL;
    }
    
    Net* net = init_net(shared->n_layers, shared->layer_sizes, adamw);
    if(!net) return NULL;
    
    net->lr = shared->learning_rate;
    net->step = shared->step;
    
    for(int i = 0; i < net->n; i++) {
        int in = net->sz[i], out = net->sz[i+1];
        
        if(shared->weight_offsets[i] + in * out > 100000 || 
           shared->bias_offsets[i] + out > 1000 || 
           shared->optim_offsets[i] + (in * out + out) * net->optimizer.aux_doubles_per_param > 100000) {
            fprintf(stderr, "Network too large for shared memory\n");
            free_net(net);
            return NULL;
        }
        
        memcpy(net->w[i], &shared->weights[shared->weight_offsets[i]], 
               in * out * sizeof(double));
        memcpy(net->b[i], &shared->biases[shared->bias_offsets[i]], 
               out * sizeof(double));
        
        int optim_size = (in * out + out) * net->optimizer.aux_doubles_per_param;
        if(optim_size > 0) {
            memcpy(net->o_data[i], &shared->optimizer_data[shared->optim_offsets[i]], 
                   optim_size * sizeof(double));
        }
    }
    return net;
}

void interpolate_weights(Net* net, Net* elite, double alpha) {
    for(int i = 0; i < net->n; i++) {
        int in = net->sz[i], out = net->sz[i+1];
        int optim_size = (in * out + out) * net->optimizer.aux_doubles_per_param;
    
        for(int j = 0; j < in*out; j++) {
            net->w[i][j] = alpha * net->w[i][j] + (1.0 - alpha) * elite->w[i][j];
        }
        for(int j = 0; j < out; j++) {
            net->b[i][j] = alpha * net->b[i][j] + (1.0 - alpha) * elite->b[i][j];
        }
        
        if(optim_size > 0) {
            memset(net->o_data[i], 0, optim_size * sizeof(double));
        }
    }
    net->step = 1;
}

int collect_rollout(Sim* sim, Net* policy, double** act, double** states, double** actions, double* rewards) {
    reset_quad(sim->quad, 
        TARGET_POS[0] + ((double)rand()/RAND_MAX - 0.5) * 0.2,
        TARGET_POS[1] + ((double)rand()/RAND_MAX - 0.5) * 0.2, 
        TARGET_POS[2] + ((double)rand()/RAND_MAX - 0.5) * 0.2
    );
    
    double t_physics = 0.0, t_control = 0.0;
    int steps = 0;
    
    while(steps < MAX_STEPS && !is_terminated(sim->quad)) {
        update_quad(sim->quad, DT_PHYSICS);
        t_physics += DT_PHYSICS;
        
        if(t_control <= t_physics) {
            get_state(sim->quad, states[steps]);
            fwd(policy, states[steps], act);
            
            for(int i = 0; i < 4; i++) {
                double std = squash(act[4][i + 4], MIN_STD, MAX_STD);
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
            
            rewards[steps] = compute_reward(sim->quad);
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

void update_policy(Net* policy, double** states, double** actions, double* returns, int steps, double** act, double** grad) {
    for(int t = 0; t < steps; t++) {
        fwd(policy, states[t], act);
        
        for(int i = 0; i < 4; i++) {
            double std = squash(act[4][i + 4], MIN_STD, MAX_STD);
            double safe_margin = 4.0 * std;
            double mean_min = OMEGA_MIN + safe_margin;
            double mean_max = OMEGA_MAX - safe_margin;
            double mean = squash(act[4][i], mean_min, mean_max);
            
            double z = (actions[t][i] - mean) / std;
            double log_prob = -0.5 * (1.8378770664093453 + 2.0 * log(std) + z * z);
            double entropy = 0.5 * (2.837877066 + 2.0 * log(std));
            
            double dmean = z / std;
            grad[4][i] = (returns[t] * log_prob + ALPHA * entropy) * dmean * dsquash(act[4][i], mean_min, mean_max);
            
            double dstd_direct = (z * z - 1.0) / std;
            double dmean_dstd = -4.0 * dsquash(act[4][i], mean_min, mean_max);
            double dstd = dstd_direct + (z / std) * dmean_dstd;
            
            grad[4][i + 4] = (returns[t] * log_prob * dstd + ALPHA * (1.0 / std)) * dsquash(act[4][i + 4], MIN_STD, MAX_STD);
        }
        
        bwd(policy, act, grad);
    }
}

int compare_results(const void* a, const void* b) {
    const ProcessResult* pa = (const ProcessResult*)a;
    const ProcessResult* pb = (const ProcessResult*)b;
    if(pb->mean_return > pa->mean_return) return 1;
    if(pb->mean_return < pa->mean_return) return -1;
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
    
    SharedNet* shared_nets = mmap(NULL, (NUM_PROCESSES + 1) * SHARED_NET_SIZE,
                                PROT_READ | PROT_WRITE,
                                MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    if(shared_nets == MAP_FAILED) {
        perror("mmap failed");
        munmap(results, NUM_PROCESSES * sizeof(ProcessResult));
        return 1;
    }
    
    Net* initial_net;
    if(argc == 3) {
        initial_net = load_weights(argv[2], adamw);
    } else {
        int layers[] = {12, 64, 64, 64, 8};
        initial_net = init_net(5, layers, adamw);
    }
    
    if(!initial_net) {
        munmap(results, NUM_PROCESSES * sizeof(ProcessResult));
        munmap(shared_nets, (NUM_PROCESSES + 1) * sizeof(SharedNet));
        return 1;
    }
    
    net_to_shared(initial_net, &shared_nets[NUM_PROCESSES]);
    free_net(initial_net);
    
    double best_return = -1e30;
    double initial_best = -1e30;
    SharedNet best_net;
    
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    double theoretical_max = (1.0 - pow(GAMMA, MAX_STEPS))/(1.0 - GAMMA);

    double current_lr = STARTING_LEARNING_RATE;
    int generations = atoi(argv[1]);
    for(int gen = 0; gen < generations; gen++) {
        printf("\nGeneration %d/%d (lr: %.2e)\n", gen + 1, generations, current_lr);
        
        ProcessResult sorted_results[NUM_PROCESSES];
        memcpy(sorted_results, results, NUM_PROCESSES * sizeof(ProcessResult));
        qsort(sorted_results, NUM_PROCESSES, sizeof(ProcessResult), compare_results);
        
        int best_idx = 0;
        if(gen > 0) {
            for(int i = 1; i < NUM_PROCESSES; i++) {
                if(results[i].mean_return > results[best_idx].mean_return) {
                    best_idx = i;
                }
            }
        }
        
        for(int i = 0; i < NUM_PROCESSES; i++) {
            if(fork() == 0) {
                srand(time(NULL) ^ getpid());
                
                Net* net;
                if(gen == 0) {
                    net = shared_to_net(&shared_nets[NUM_PROCESSES]);
                } else {
                    if(results[i].mean_return >= sorted_results[1].mean_return) {
                        net = shared_to_net(&shared_nets[i]);
                    } else {
                        net = shared_to_net(&shared_nets[NUM_PROCESSES]);
                        Net* mutation = shared_to_net(&best_net);
                        interpolate_weights(net, mutation, 0.5);
                        free_net(mutation);
                    }
                }
                
                net->lr = current_lr;
                
                Sim* sim = init_sim("", false);
                double** act = malloc(5 * sizeof(double*));
                double** grad = malloc(5 * sizeof(double*));
                double*** states = malloc(NUM_ROLLOUTS * sizeof(double**));
                double*** actions = malloc(NUM_ROLLOUTS * sizeof(double**));
                double** rewards = malloc(NUM_ROLLOUTS * sizeof(double*));
                int* steps = malloc(NUM_ROLLOUTS * sizeof(int));
                
                for(int i = 0; i < 5; i++) {
                    act[i] = malloc(net->sz[i] * sizeof(double));
                    grad[i] = calloc(net->sz[i], sizeof(double));
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
                
                double sum_returns = 0.0, sum_squared = 0.0;
                for(int iter = 1; iter <= NUM_ITERATIONS; iter++) {
                    for(int r = 0; r < NUM_ROLLOUTS; r++) {
                        steps[r] = collect_rollout(sim, net, act, states[r], actions[r], rewards[r]);
                        if(iter == NUM_ITERATIONS) {
                            sum_returns += rewards[r][0];
                            sum_squared += rewards[r][0] * rewards[r][0];
                        }
                    }
                    
                    for(int r = 0; r < NUM_ROLLOUTS; r++) 
                        update_policy(net, states[r], actions[r], rewards[r], steps[r], act, grad);
                }
                
                results[i].mean_return = sum_returns / NUM_ROLLOUTS;
                results[i].std_return = sqrt(sum_squared/NUM_ROLLOUTS - 
                                          pow(sum_returns/NUM_ROLLOUTS, 2));
                net_to_shared(net, &shared_nets[i]);
                
                for(int r = 0; r < NUM_ROLLOUTS; r++) {
                    for(int i = 0; i < MAX_STEPS; i++) {
                        free(states[r][i]);
                        free(actions[r][i]);
                    }
                    free(states[r]); free(actions[r]); free(rewards[r]);
                }
                for(int i = 0; i < 5; i++) { free(act[i]); free(grad[i]); }
                free(states); free(actions); free(rewards); free(steps);
                free(act); free(grad);
                free_net(net);
                free_sim(sim);
                exit(0);
            }
        }

        for(int i = 0; i < NUM_PROCESSES; i++) {
            wait(NULL);
        }
        
        for(int i = 1; i < NUM_PROCESSES; i++) {
            if(results[i].mean_return > results[best_idx].mean_return) {
                best_idx = i;
            }
        }
        
        if(results[best_idx].mean_return > best_return) {
            best_return = results[best_idx].mean_return;
            memcpy(&best_net, &shared_nets[best_idx], sizeof(SharedNet));
        }
        
        if(gen == 0) {
            initial_best = best_return;
        }
        
        memcpy(&shared_nets[NUM_PROCESSES], &shared_nets[best_idx], sizeof(SharedNet));
        
        printf("\nGeneration Results:\n");
        double performance_ratio = best_return/theoretical_max;
        current_lr = STARTING_LEARNING_RATE * (1.0 - performance_ratio) + MINIMUM_LEARNING_RATE;
        printf("Best Ever: %.2f / %.2f (%.1f%%)\n", 
               best_return, theoretical_max, performance_ratio * 100.0);
        for(int i = 0; i < NUM_PROCESSES; i++) {
            printf("Agent %d: %.2f ± %.2f%s\n", i, 
                   results[i].mean_return, results[i].std_return,
                   i == best_idx ? " *" : "");
        }
    }
    
    gettimeofday(&end_time, NULL);
    double seconds_elapsed = (end_time.tv_sec - start_time.tv_sec) + 
                           (end_time.tv_usec - start_time.tv_usec) / 1000000.0;

    double nominal_improvement = best_return - initial_best;
    double nominal_rate = nominal_improvement / seconds_elapsed;

    double initial_percentage = (initial_best / theoretical_max) * 100.0;
    double final_percentage = (best_return / theoretical_max) * 100.0;
    double percentage_improvement = final_percentage - initial_percentage;
    double percentage_rate = percentage_improvement / seconds_elapsed;

    printf("\nTraining Summary:\n");
    printf("Total time: %.1f seconds\n", seconds_elapsed);
    printf("Improvement rate: %.3f score/second\n", nominal_rate);
    printf("Progress towards theoretical maximum: %.2f%% -> %.2f%%\n", 
           initial_percentage, final_percentage);
    printf("Rate of progress: %.3f percentage points/second\n", percentage_rate);
    
    char final_weights[64];
    strftime(final_weights, sizeof(final_weights), "%Y%m%d_%H%M%S_policy.bin", 
             localtime(&(time_t){time(NULL)}));
    
    Net* final_net = shared_to_net(&best_net);
    save_weights(final_weights, final_net);
    printf("\nFinal weights saved to: %s\n", final_weights);
    free_net(final_net);
    
    munmap(results, NUM_PROCESSES * sizeof(ProcessResult));
    munmap(shared_nets, (NUM_PROCESSES + 1) * sizeof(SharedNet));
    return 0;
}