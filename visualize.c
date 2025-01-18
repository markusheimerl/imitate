#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "grad/grad.h"
#include "sim/sim.h"

#define STATE_DIM 12
#define ACTION_DIM 8
#define HIDDEN_DIM 64
#define DT_PHYSICS (1.0/1000.0)
#define DT_CONTROL (1.0/60.0)
#define DT_RENDER (1.0/30.0)
#define DURATION 5.0

const double TARGET_POS[3] = {0.0, 1.0, 0.0};

void get_state(Quad* q, double* state) {
    memcpy(state, q->linear_position_W, 3 * sizeof(double));
    memcpy(state + 3, q->linear_velocity_W, 3 * sizeof(double));
    memcpy(state + 6, q->angular_velocity_B, 3 * sizeof(double));
    state[9] = q->R_W_B[0];
    state[10] = q->R_W_B[4];
    state[11] = q->R_W_B[8];
}

int main(int argc, char** argv) {
    if(argc != 2) {
        printf("Usage: %s <policy_weights.bin>\n", argv[0]);
        return 1;
    }

    // Initialize policy network
    int layers[] = {STATE_DIM, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, ACTION_DIM};
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
    double state[STATE_DIM];

    // Reset quadcopter to slightly offset initial position
    reset_quad(sim->quad, 
              TARGET_POS[0], 
              TARGET_POS[1], 
              TARGET_POS[2]);

    double t_physics = 0.0, t_control = 0.0, t_render = 0.0;
    
    printf("Generating visualization...\n");
    
    while(t_physics < DURATION) {
        update_quad(sim->quad, DT_PHYSICS);
        t_physics += DT_PHYSICS;
        
        if(t_control <= t_physics) {
            // Get current state
            get_state(sim->quad, state);
            
            // Forward pass through policy network
            fwd(policy, state, act);
            
            // Apply actions
            for(int i = 0; i < 4; i++) {
                double mean = 50.0 + 20.0 * tanh(act[4][i]);
                sim->quad->omega_next[i] = mean;
            }
            
            t_control += DT_CONTROL;
            
            // Print progress
            printf("\rTime: %.2f/%.2f seconds", t_physics, DURATION);
            fflush(stdout);
        }
        
        if(t_render <= t_physics) {
            render_sim(sim);
            t_render += DT_RENDER;
        }
    }

    printf("\nVisualization complete!\n");

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