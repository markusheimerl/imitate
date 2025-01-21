#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "grad/grad.h"
#include "sim/sim.h"

#define INPUT_DIM 15
#define ACTION_DIM 8
#define HIDDEN_DIM 64
#define DT_PHYSICS (1.0/1000.0)
#define DT_CONTROL (1.0/60.0)
#define DT_RENDER (1.0/30.0)
#define DURATION 5.0

#define MAX_STD 2.0
#define MIN_STD 1e-5
#define TARGET_OFFSET 0.2

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

int main(int argc, char** argv) {
    if(argc != 2) {
        printf("Usage: %s <policy_weights.bin>\n", argv[0]);
        return 1;
    }

    srand(time(NULL) ^ getpid());

    // Initialize policy network
    int layers[] = {INPUT_DIM, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, ACTION_DIM};
    Net* policy = load_weights(argv[1], null_opt);
    if(!policy) {
        printf("Failed to load weights from %s\n", argv[1]);
        return 1;
    }

    // Initialize simulation and visualization
    Sim* sim = init_sim("sim/", true);
    
    // Initialize network activations
    double** act = malloc(5 * sizeof(double*));
    for(int i = 0; i < 5; i++) {
        act[i] = malloc(policy->sz[i] * sizeof(double));
    }

    // Initialize state buffer
    double state[INPUT_DIM];

    // Generate random target position
    double target[3] = {
        TARGET_POS[0] + ((double)rand()/RAND_MAX - 0.5) * TARGET_OFFSET,
        TARGET_POS[1] + ((double)rand()/RAND_MAX - 0.5) * TARGET_OFFSET,
        TARGET_POS[2] + ((double)rand()/RAND_MAX - 0.5) * TARGET_OFFSET
    };

    // Reset quadcopter to slightly offset initial position
    reset_quad(sim->quad, 
        target[0] + ((double)rand()/RAND_MAX - 0.5) * 0.2,
        target[1] + ((double)rand()/RAND_MAX - 0.5) * 0.2, 
        target[2] + ((double)rand()/RAND_MAX - 0.5) * 0.2
    );

    double t_physics = 0.0, t_control = 0.0, t_render = 0.0;
    
    printf("Generating visualization...\n");
    printf("Target position: (%.2f, %.2f, %.2f)\n", 
           target[0], target[1], target[2]);
    
    while(t_physics < DURATION) {
        update_quad(sim->quad, DT_PHYSICS);
        t_physics += DT_PHYSICS;
        
        if(t_control <= t_physics) {
            // Get current state including target position
            get_state(sim->quad, state, target);
            
            // Forward pass through policy network
            fwd(policy, state, act);
            
            // Apply actions using the same squash function as in training
            // but only using the mean (no sampling during visualization)
            for(int i = 0; i < 4; i++) {
                // Get standard deviation (needed for mean bounds)
                double std = squash(act[4][i + 4], MIN_STD, MAX_STD);
                
                // Compute dynamic bounds for mean based on current std
                double safe_margin = 4.0 * std;
                double mean_min = OMEGA_MIN + safe_margin;
                double mean_max = OMEGA_MAX - safe_margin;
                
                // Get mean using squash with dynamic bounds
                double mean = squash(act[4][i], mean_min, mean_max);
                
                // Apply the mean action
                // (no sampling, i.e. use "maximum a posteriori" (MAP) policy)
                sim->quad->omega_next[i] = mean;
            }
            
            t_control += DT_CONTROL;
            
            // Print progress, current position, and distance to target
            double dist = sqrt(
                pow(sim->quad->linear_position_W[0] - target[0], 2) +
                pow(sim->quad->linear_position_W[1] - target[1], 2) +
                pow(sim->quad->linear_position_W[2] - target[2], 2)
            );
            
            printf("\rTime: %.2f/%.2f seconds | Pos: (%.2f, %.2f, %.2f) | Dist to target: %.3f", 
                   t_physics, DURATION,
                   sim->quad->linear_position_W[0],
                   sim->quad->linear_position_W[1],
                   sim->quad->linear_position_W[2],
                   dist);
            fflush(stdout);
        }
        
        if(t_render <= t_physics) {
            render_sim(sim);
            t_render += DT_RENDER;
        }
    }

    // Calculate final distance to target
    double final_dist = sqrt(
        pow(sim->quad->linear_position_W[0] - target[0], 2) +
        pow(sim->quad->linear_position_W[1] - target[1], 2) +
        pow(sim->quad->linear_position_W[2] - target[2], 2)
    );

    printf("\nVisualization complete!\n");
    printf("Final position: (%.2f, %.2f, %.2f)\n",
           sim->quad->linear_position_W[0],
           sim->quad->linear_position_W[1],
           sim->quad->linear_position_W[2]);
    printf("Target position: (%.2f, %.2f, %.2f)\n",
           target[0], target[1], target[2]);
    printf("Final distance to target: %.3f\n", final_dist);

    // Cleanup
    for(int i = 0; i < 5; i++) {
        free(act[i]);
    }
    save_sim(sim);
    free(act);
    free_net(policy);
    free_sim(sim);

    return 0;
}