#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include "quad.h"
#include "scene.h"
#include "grad.h"

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define DT_RENDER   (1.0 / 24.0)

#define STATE_DIM 15
#define ACTION_DIM 8
#define MAX_STD 3.0
#define MIN_STD 1e-5

double squash(double x, double min, double max) { 
    return ((max + min) / 2.0) + ((max - min) / 2.0) * tanh(x); 
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

    srand(time(NULL));
    
    // Load policy network
    Net* policy = load_net(argv[1]);

    // Initialize random target position
    double target_pos[3] = {
        (double)rand() / RAND_MAX * 4.0 - 2.0,  // Range: -2 to 2
        1.5,                                     // Fixed height
        (double)rand() / RAND_MAX * 4.0 - 2.0   // Range: -2 to 2
    };
    
    printf("Target position: (%.2f, %.2f, %.2f)\n", 
           target_pos[0], target_pos[1], target_pos[2]);
    
    // Initialize quadcopter
    Quad* quad = init_quad(0.0, 0.0, 0.0);
    
    // Initialize raytracer scene
    Scene scene = create_scene(800, 600, 10000, 24, 0.9f);
    
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
    Mesh drone = create_mesh("raytracer/drone.obj", "raytracer/drone.webp");
    add_mesh_to_scene(&scene, drone);
    
    Mesh ground = create_mesh("raytracer/ground.obj", "raytracer/ground.webp");
    add_mesh_to_scene(&scene, ground);

    // Initialize timers
    double t_physics = 0.0;
    double t_control = 0.0;
    double t_render = 0.0;
    int frame = 0;
    
    // State buffer for neural network
    double state[STATE_DIM];

    // Main simulation loop
    while (frame < scene.frame_count) {
        // Physics update
        if (t_physics >= DT_PHYSICS) {
            update_quad(quad, DT_PHYSICS);
            t_physics = 0.0;
        }
        
        // Control update
        if (t_control >= DT_CONTROL) {
            // Get current state and run through policy network
            get_state(quad, state, target_pos);
            forward(policy, state);
            
            // Extract actions from network output
            for(int i = 0; i < 4; i++) {
                double std = squash(policy->layers[policy->n_layers-1].x[i + 4], MIN_STD, MAX_STD);
                double safe_margin = 4.0 * std;
                double mean_min = OMEGA_MIN + safe_margin;
                double mean_max = OMEGA_MAX - safe_margin;
                double mean = squash(policy->layers[policy->n_layers-1].x[i], mean_min, mean_max);
                
                quad->omega_next[i] = mean;
            }
            
            t_control = 0.0;
            
            // Print current status
            double dist = sqrt(
                pow(quad->linear_position_W[0] - target_pos[0], 2) +
                pow(quad->linear_position_W[1] - target_pos[1], 2) +
                pow(quad->linear_position_W[2] - target_pos[2], 2)
            );
            printf("\rTime: %.2f | Pos: (%.2f, %.2f, %.2f) | Dist: %.3f", 
                   frame * DT_RENDER,
                   quad->linear_position_W[0],
                   quad->linear_position_W[1],
                   quad->linear_position_W[2],
                   dist);
            fflush(stdout);
        }
        
        // Render update
        if (t_render >= DT_RENDER) {
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
            t_render = 0.0;
        }
        
        // Increment timers
        t_physics += DT_PHYSICS;
        t_control += DT_PHYSICS;
        t_render += DT_PHYSICS;
    }

    // Save animation
    char filename[64];
    strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_flight.webp", 
             localtime(&(time_t){time(NULL)}));
    save_scene(&scene, filename);

    // Cleanup
    destroy_scene(&scene);
    free(quad);
    free_net(policy);
    
    return 0;
}