#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include "quad.h"
#include "scene.h"
#include "net.h"

double squash(double x, double min, double max) { 
    return ((max + min) / 2.0) + ((max - min) / 2.0) * tanh(x); 
}

int main(int argc, char** argv) {
    if(argc != 2) {
        printf("Usage: %s <policy_weights.bin>\n", argv[0]);
        return 1;
    }

    srand(time(NULL));
    
    // Load policy network
    Net* policy = load_net(argv[1]);

    // Generate random start and target positions using spherical coordinates
    double start[3], target[3];
    
    // Random start position
    double r1 = MAX_DISTANCE * ((double)rand() / RAND_MAX);
    double theta1 = 2 * M_PI * ((double)rand() / RAND_MAX);
    double phi1 = acos(2 * ((double)rand() / RAND_MAX) - 1);
    start[0] = r1 * sin(phi1) * cos(theta1);
    start[1] = r1 * sin(phi1) * sin(theta1) + 1.0;
    start[2] = r1 * cos(phi1);
    
    // Random target position
    double r2 = MAX_DISTANCE * ((double)rand() / RAND_MAX);
    double theta2 = 2 * M_PI * ((double)rand() / RAND_MAX);
    double phi2 = acos(2 * ((double)rand() / RAND_MAX) - 1);
    target[0] = r2 * sin(phi2) * cos(theta2);
    target[1] = r2 * sin(phi2) * sin(theta2) + 1.0;
    target[2] = r2 * cos(phi2);
    
    printf("Start position: (%.2f, %.2f, %.2f)\n", start[0], start[1], start[2]);
    printf("Target position: (%.2f, %.2f, %.2f)\n", target[0], target[1], target[2]);
    printf("Distance: %.2f meters\n", 
           sqrt(pow(start[0]-target[0], 2) + pow(start[1]-target[1], 2) + pow(start[2]-target[2], 2)));
    
    // Initialize quadcopter at start position
    Quad quad = create_quad(start[0], start[1], start[2]);
    
    // Initialize raytracer scene
    Scene scene = create_scene(400, 300, ((int)(DT_CONTROL * MAX_STEPS * 1000)), ((int)(1.0 / DT_RENDER)), 0.4f);
    
    // Set up camera to view both start and target positions
    Vec3 center = {
        (float)(start[0] + target[0]) / 2.0f,
        (float)(start[1] + target[1]) / 2.0f,
        (float)(start[2] + target[2]) / 2.0f
    };
    
    float max_dist = (float)MAX_DISTANCE;
    set_scene_camera(&scene,
        (Vec3){center.x - max_dist*1.5f, center.y + max_dist, center.z - max_dist*1.5f},
        center,
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

    // Add target visualization (using treasure mesh)
    Mesh treasure = create_mesh("raytracer/treasure.obj", "raytracer/treasure.webp");
    add_mesh_to_scene(&scene, treasure);
    Vec3 vtarget = {(float)target[0], (float)target[1], (float)target[2]};
    set_mesh_position(&scene.meshes[2], vtarget);
    
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
            update_quad(&quad, DT_PHYSICS);
            t_physics = 0.0;
        }
        
        // Control update
        if (t_control >= DT_CONTROL) {
            // Get current state and run through policy network
            get_quad_state(quad, state);
            memcpy(state + 12, target, 3 * sizeof(double));
            forward(policy, state);
            
            // Extract actions from network output
            for(int i = 0; i < 4; i++) {
                double std = squash(policy->layers[policy->n_layers-1].x[i + 4], MIN_STD, MAX_STD);
                double safe_margin = 4.0 * std;
                double mean_min = OMEGA_MIN + safe_margin;
                double mean_max = OMEGA_MAX - safe_margin;
                double mean = squash(policy->layers[policy->n_layers-1].x[i], mean_min, mean_max);
                
                quad.omega_next[i] = mean;
            }
            
            t_control = 0.0;
            
            // Print current status
            double dist = sqrt(
                pow(quad.linear_position_W[0] - target[0], 2) +
                pow(quad.linear_position_W[1] - target[1], 2) +
                pow(quad.linear_position_W[2] - target[2], 2)
            );
            printf("\rTime: %.2f | Pos: (%.2f, %.2f, %.2f) | Dist to target: %.3f", 
                   frame * DT_RENDER,
                   quad.linear_position_W[0],
                   quad.linear_position_W[1],
                   quad.linear_position_W[2],
                   dist);
            fflush(stdout);
        }
        
        // Render update
        if (t_render >= DT_RENDER) {
            // Update drone position and orientation in the scene
            set_mesh_position(&scene.meshes[0], 
                (Vec3){(float)quad.linear_position_W[0], 
                       (float)quad.linear_position_W[1], 
                       (float)quad.linear_position_W[2]});
            
            // Convert rotation matrix to Euler angles for visualization
            float roll = atan2f(quad.R_W_B[7], quad.R_W_B[8]);
            float pitch = asinf(-quad.R_W_B[6]);
            float yaw = atan2f(quad.R_W_B[3], quad.R_W_B[0]);
            
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
    time_t current_time = time(NULL);
    strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_flight.webp", 
             localtime(&current_time));
    save_scene(&scene, filename);
    printf("\nSaved animation to %s\n", filename);

    // Cleanup
    destroy_scene(&scene);
    free_net(policy);
    
    return 0;
}