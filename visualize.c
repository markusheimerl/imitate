#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "grad/grad.h"
#include "sim/quad.h"
#include "sim/raytracer/scene.h"

#define DT_PHYSICS (1.0/1000.0)
#define DT_CONTROL (1.0/60.0)
#define DT_RENDER (1.0/24.0)

#define STATE_DIM 15
#define HIDDEN_DIM 64
#define ACTION_DIM 8

#define MAX_STD 3.0
#define MIN_STD 1e-5

#define TASK_RADIUS 3.0
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

    // Initialize quadcopter
    Quad* quad = init_quad(0.0, 0.0, 0.0);

    // Initialize scene
    Scene scene = create_scene(800, 600, (int)(SIMULATION_DURATION/DT_RENDER), 24, 0.9f);
    
    // Set up camera
    set_scene_camera(&scene,
        (Vec3){-3.0f, 3.0f, -3.0f},
        (Vec3){0.0f, 0.0f, 0.0f},
        (Vec3){0.0f, 1.0f, 0.0f},
        60.0f
    );
    
    // Set up light
    set_scene_light(&scene,
        (Vec3){1.0f, 1.0f, -1.0f},     // Direction
        (Vec3){1.4f, 1.4f, 1.4f}       // White light
    );
    
    // Add meshes to scene
    Mesh drone = create_mesh("sim/raytracer/drone.obj", "sim/raytracer/drone.webp");
    add_mesh_to_scene(&scene, drone);
    
    Mesh ground = create_mesh("sim/raytracer/ground.obj", "sim/raytracer/ground.webp");
    add_mesh_to_scene(&scene, ground);

    // Generate start and target positions
    double start_pos[3], target_pos[3];
    double center[3] = {0, 1, 0};
    
    get_random_position(start_pos, center, TASK_RADIUS);
    get_random_position(target_pos, start_pos, TASK_RADIUS);
    
    // Reset quadcopter to start position
    reset_quad(quad, start_pos[0], start_pos[1], start_pos[2]);

    // Initialize state buffer
    double state[STATE_DIM];

    double t_physics = 0.0, t_control = 0.0, t_render = 0.0;
    int frame = 0;
    
    printf("Starting visualization...\n");
    printf("Start position: (%.2f, %.2f, %.2f)\n", start_pos[0], start_pos[1], start_pos[2]);
    printf("Target position: (%.2f, %.2f, %.2f)\n", target_pos[0], target_pos[1], target_pos[2]);
    
    while(t_physics < SIMULATION_DURATION) {
        update_quad(quad, DT_PHYSICS);
        t_physics += DT_PHYSICS;
        
        if(t_control <= t_physics) {
            get_state(quad, state, target_pos);
            forward(policy, state);
            
            for(int i = 0; i < 4; i++) {
                double std = squash(policy->layers[policy->n_layers-1].x[i + 4], MIN_STD, MAX_STD);
                double safe_margin = 4.0 * std;
                double mean_min = OMEGA_MIN + safe_margin;
                double mean_max = OMEGA_MAX - safe_margin;
                double mean = squash(policy->layers[policy->n_layers-1].x[i], mean_min, mean_max);
                
                quad->omega_next[i] = mean;
            }
            
            t_control += DT_CONTROL;
            
            double dist = sqrt(
                pow(quad->linear_position_W[0] - target_pos[0], 2) +
                pow(quad->linear_position_W[1] - target_pos[1], 2) +
                pow(quad->linear_position_W[2] - target_pos[2], 2)
            );
            
            printf("\rTime: %.2f/%.2f | Pos: (%.2f, %.2f, %.2f) | Dist: %.3f", 
                   t_physics, SIMULATION_DURATION,
                   quad->linear_position_W[0],
                   quad->linear_position_W[1],
                   quad->linear_position_W[2],
                   dist);
            fflush(stdout);
        }
        
        if(t_render <= t_physics) {
            // Update drone position and orientation in the scene
            set_mesh_position(&scene.meshes[0], 
                (Vec3){(float)quad->linear_position_W[0], 
                       (float)quad->linear_position_W[1], 
                       (float)quad->linear_position_W[2]});
            
            // Convert rotation matrix to Euler angles for visualization
            float roll = atan2f(quad->R_W_B[7], quad->R_W_B[8]);
            float pitch = asinf(-quad->R_W_B[6]);
            float yaw = atan2f(quad->R_W_B[3], quad->R_W_B[0]);
            
            set_mesh_rotation(&scene.meshes[0], (Vec3){roll, pitch, yaw});
            
            // Render frame
            render_scene(&scene);
            next_frame(&scene);
            
            frame++;
            t_render += DT_RENDER;
        }
    }

    double final_dist = sqrt(
        pow(quad->linear_position_W[0] - target_pos[0], 2) +
        pow(quad->linear_position_W[1] - target_pos[1], 2) +
        pow(quad->linear_position_W[2] - target_pos[2], 2)
    );

    printf("\n\nSimulation complete!\n");
    printf("Final position: (%.2f, %.2f, %.2f)\n",
           quad->linear_position_W[0],
           quad->linear_position_W[1],
           quad->linear_position_W[2]);
    printf("Target position: (%.2f, %.2f, %.2f)\n",
           target_pos[0], target_pos[1], target_pos[2]);
    printf("Final distance to target: %.3f\n", final_dist);

    // Save the animation
    char filename[64];
    strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_flight.webp", 
             localtime(&(time_t){time(NULL)}));
    save_scene(&scene, filename);
    printf("Animation saved as: %s\n", filename);

    // Cleanup
    free_net(policy);
    free(quad);
    destroy_scene(&scene);

    return 0;
}