#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "grad/grad.h"
#include "sim/sim.h"

#define DT_PHYSICS (1.0/1000.0)
#define DT_CONTROL (1.0/60.0)
#define DT_RENDER (1.0/30.0)

#define STATE_DIM 15
#define HIDDEN_DIM 64
#define ACTION_DIM 8

#define MAX_STD 3.0
#define MIN_STD 1e-5

#define TASK_RADIUS 0.1
#define MIN_HEIGHT 0.5
#define MAX_HEIGHT 1.5
#define SIMULATION_DURATION 10.0

double squash(double x, double min, double max) { 
    return ((max + min) / 2.0) + ((max - min) / 2.0) * tanh(x); 
}

void get_random_position(double pos[3], double center[3], double radius) {
    double theta = ((double)rand()/RAND_MAX) * 2.0 * M_PI;
    double phi = acos(2.0 * ((double)rand()/RAND_MAX) - 1.0);
    double r = radius * ((double)rand()/RAND_MAX);

    pos[0] = center[0] + r * sin(phi) * cos(theta);
    pos[1] = center[1] + r * sin(phi) * sin(theta);
    pos[2] = center[2] + r * cos(phi);

    pos[1] = fmax(pos[1], MIN_HEIGHT);
    pos[1] = fmin(pos[1], MAX_HEIGHT);
}

void get_state(Quad* q, double* state, double* target_pos) {
    memcpy(state, q->linear_position_W, 3 * sizeof(double));
    memcpy(state + 3, q->linear_velocity_W, 3 * sizeof(double));
    memcpy(state + 6, q->angular_velocity_B, 3 * sizeof(double));
    state[9] = q->R_W_B[0];
    state[10] = q->R_W_B[4];
    state[11] = q->R_W_B[8];
    memcpy(state + 12, target_pos, 3 * sizeof(double));
}

int main(int argc, char** argv) {
    if(argc != 2) {
        printf("Usage: %s <policy_weights.bin>\n", argv[0]);
        return 1;
    }

    srand(time(NULL) ^ getpid());

    // Initialize policy network
    Net* policy = load_net(argv[1]);
    if(!policy) {
        printf("Failed to load weights from %s\n", argv[1]);
        return 1;
    }

    // Initialize simulation
    Sim* sim = init_sim("sim/", true);

    // Generate start and target positions
    double start_pos[3], target_pos[3];
    double center[3] = {0, 1, 0};
    
    get_random_position(start_pos, center, TASK_RADIUS);
    get_random_position(target_pos, start_pos, TASK_RADIUS);
    
    // Reset quadcopter to start position
    reset_quad(sim->quad, start_pos[0], start_pos[1], start_pos[2]);

    // Initialize state buffer
    double state[STATE_DIM];

    double t_physics = 0.0, t_control = 0.0, t_render = 0.0;
    
    printf("Starting visualization...\n");
    printf("Start position: (%.2f, %.2f, %.2f)\n", start_pos[0], start_pos[1], start_pos[2]);
    printf("Target position: (%.2f, %.2f, %.2f)\n", target_pos[0], target_pos[1], target_pos[2]);
    
    while(t_physics < SIMULATION_DURATION) {
        update_quad(sim->quad, DT_PHYSICS);
        t_physics += DT_PHYSICS;
        
        if(t_control <= t_physics) {
            get_state(sim->quad, state, target_pos);
            forward(policy, state);
            
            for(int i = 0; i < 4; i++) {
                double std = squash(policy->layers[policy->n_layers-1].x[i + 4], MIN_STD, MAX_STD);
                double safe_margin = 4.0 * std;
                double mean_min = OMEGA_MIN + safe_margin;
                double mean_max = OMEGA_MAX - safe_margin;
                double mean = squash(policy->layers[policy->n_layers-1].x[i], mean_min, mean_max);
                
                sim->quad->omega_next[i] = mean;
            }
            
            t_control += DT_CONTROL;
            
            double dist = sqrt(
                pow(sim->quad->linear_position_W[0] - target_pos[0], 2) +
                pow(sim->quad->linear_position_W[1] - target_pos[1], 2) +
                pow(sim->quad->linear_position_W[2] - target_pos[2], 2)
            );
            
            printf("\rTime: %.2f/%.2f | Pos: (%.2f, %.2f, %.2f) | Dist: %.3f", 
                   t_physics, SIMULATION_DURATION,
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

    double final_dist = sqrt(
        pow(sim->quad->linear_position_W[0] - target_pos[0], 2) +
        pow(sim->quad->linear_position_W[1] - target_pos[1], 2) +
        pow(sim->quad->linear_position_W[2] - target_pos[2], 2)
    );

    printf("\n\nSimulation complete!\n");
    printf("Final position: (%.2f, %.2f, %.2f)\n",
           sim->quad->linear_position_W[0],
           sim->quad->linear_position_W[1],
           sim->quad->linear_position_W[2]);
    printf("Target position: (%.2f, %.2f, %.2f)\n",
           target_pos[0], target_pos[1], target_pos[2]);
    printf("Final distance to target: %.3f\n", final_dist);

    // Cleanup
    free_net(policy);
    save_sim(sim);
    free_sim(sim);

    return 0;
}