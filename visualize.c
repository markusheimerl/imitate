#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include "quad.h"
#include "scene.h"
#include "net.h"

#define DT_PHYSICS (1.0/1000.0)
#define DT_CONTROL (1.0/60.0)
#define DT_RENDER (1.0/24.0)

#define STATE_DIM 6      // 3 accel + 3 gyro
#define ACTION_DIM 8     // 4 means + 4 stds
#define MAX_STEPS 1000
#define NUM_ROLLOUTS 128

#define GAMMA 0.999
#define MAX_STD 3.0
#define MIN_STD 1e-5

#define MAX_MEAN (OMEGA_MAX - 4.0 * MAX_STD)
#define MIN_MEAN (OMEGA_MIN + 4.0 * MAX_STD)

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
    if (!policy) {
        printf("Failed to load policy network\n");
        return 1;
    }
    
    // Initialize quadcopter at hover height
    Quad quad = create_quad(0.0, 1.0, 0.0);
    
    // Initialize raytracer scene
    Scene scene = create_scene(400, 300, ((int)(DT_CONTROL * MAX_STEPS * 1000)), ((int)(1.0 / DT_RENDER)), 0.8f);

    set_scene_camera(&scene,
        (Vec3){-3.0f, 3.0f, -3.0f}, // Camera position
        (Vec3){0.0f, 1.0f, 0.0f},   // Look at center
        (Vec3){0.0f, 1.0f, 0.0f},   // Up vector
        60.0f                       // FOV
    );
    
    // Set up light
    set_scene_light(&scene,
        (Vec3){1.0f, 1.0f, -1.0f},  // Direction
        (Vec3){1.4f, 1.4f, 1.4f}    // White light
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
    
    // State buffer for neural network (only sensor readings)
    double state[STATE_DIM];  // 6D: 3 accel + 3 gyro

    // Main simulation loop
    while (frame < scene.frame_count) {
        // Physics update
        if (t_physics >= DT_PHYSICS) {
            update_quad(&quad, DT_PHYSICS);
            t_physics = 0.0;
        }
        
        // Control update
        if (t_control >= DT_CONTROL) {
            // Get current sensor readings
            memcpy(state, quad.linear_acceleration_B_s, 3 * sizeof(double));
            memcpy(state + 3, quad.angular_velocity_B_s, 3 * sizeof(double));
            
            // Forward pass through network
            forward_net(policy, state);
            
            // Extract actions from network output
            for(int i = 0; i < 4; i++) {
                double mean = squash(policy->layers[policy->num_layers-1].values[i], MIN_MEAN, MAX_MEAN);
                quad.omega_next[i] = mean;
            }
            
            t_control = 0.0;
            
            // Print current status with more detailed stability metrics
            double accel_magnitude = sqrt(
                quad.linear_acceleration_B_s[0] * quad.linear_acceleration_B_s[0] +
                quad.linear_acceleration_B_s[1] * quad.linear_acceleration_B_s[1] +
                quad.linear_acceleration_B_s[2] * quad.linear_acceleration_B_s[2]
            );
            
            double angvel_magnitude = sqrt(
                quad.angular_velocity_B_s[0] * quad.angular_velocity_B_s[0] +
                quad.angular_velocity_B_s[1] * quad.angular_velocity_B_s[1] +
                quad.angular_velocity_B_s[2] * quad.angular_velocity_B_s[2]
            );
            
            printf("\rTime: %.2f | Height: %.2f | AccelMag: %.2f | AngVelMag: %.2f | Tilt: %.2f°", 
                   frame * DT_RENDER,
                   quad.linear_position_W[1],
                   accel_magnitude,
                   angvel_magnitude,
                   acos(quad.R_W_B[4]) * 180.0 / M_PI);  // Convert tilt to degrees
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
    strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_flight.webp", localtime(&current_time));
    save_scene(&scene, filename);
    printf("\nSaved animation to %s\n", filename);

    // Cleanup
    destroy_scene(&scene);
    free_net(policy);
    
    return 0;
}