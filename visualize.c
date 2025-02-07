#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include "net.h"
#include "quad.h"
#include "scene.h"

void visualize_rollout(Net* policy, Scene* scene) {
    // Initialize environment (quadcopter) with starting state s_0
    Quad quad = create_quad(0.0, 1.0, 0.0);
    double t_physics = 0.0;
    double t_control = 0.0;
    double t_render = 0.0;
    int frame = 0;
    
    while(frame < scene->frame_count) {
        // Physics simulation steps
        if (t_physics >= DT_PHYSICS) {
            update_quad(&quad, DT_PHYSICS);
            t_physics = 0.0;
        }
        
        // Control at lower frequency
        if (t_control >= DT_CONTROL) {
            // Get state s_t (sensor readings)
            double state[STATE_DIM];
            memcpy(state, quad.linear_acceleration_B_s, 3 * sizeof(double));
            memcpy(state + 3, quad.angular_velocity_B_s, 3 * sizeof(double));

            // Forward pass through policy network
            forward_net(policy, state);
            
            // Get deterministic actions (means only)
            for(int i = 0; i < 4; i++) {
                double mu = squash(policy->h[2][i], MIN_MEAN, MAX_MEAN);
                quad.omega_next[i] = mu;
            }

            // Print status
            double accel_mag = sqrt(dotVec3f(quad.linear_acceleration_B_s, 
                                           quad.linear_acceleration_B_s));
            double angvel_mag = sqrt(dotVec3f(quad.angular_velocity_B_s, 
                                            quad.angular_velocity_B_s));
            
            printf("\rStep %d/%d | Pos: (%.2f, %.2f, %.2f) | AccelMag: %.2f | AngVelMag: %.2f", 
                frame, scene->frame_count,
                quad.linear_position_W[0], 
                quad.linear_position_W[1], 
                quad.linear_position_W[2],
                accel_mag, angvel_mag);
            fflush(stdout);
            
            t_control = 0.0;
        }
        
        // Render at animation framerate
        if (t_render >= DT_RENDER) {
            // Update drone transform in scene
            set_mesh_position(&scene->meshes[0], 
                (Vec3){(float)quad.linear_position_W[0], 
                       (float)quad.linear_position_W[1], 
                       (float)quad.linear_position_W[2]});
            
            float roll = atan2f(quad.R_W_B[7], quad.R_W_B[8]);
            float pitch = asinf(-quad.R_W_B[6]);
            float yaw = atan2f(quad.R_W_B[3], quad.R_W_B[0]);
            set_mesh_rotation(&scene->meshes[0], (Vec3){roll, pitch, yaw});
            
            render_scene(scene);
            next_frame(scene);
            frame++;
            t_render = 0.0;
        }
        
        t_physics += DT_PHYSICS;
        t_control += DT_PHYSICS;
        t_render += DT_PHYSICS;
    }
}

int main(int argc, char** argv) {
    if(argc != 2) {
        printf("Usage: %s <policy_weights.bin>\n", argv[0]);
        return 1;
    }
    
    // Load policy network
    Net* policy = (Net*)calloc(1, sizeof(Net));
    if(!load_net(policy, argv[1])) {
        printf("Failed to load policy from %s\n", argv[1]);
        free(policy);
        return 1;
    }

    // Initialize scene
    Scene scene = create_scene(400, 300, 
        ((int)(DT_CONTROL * MAX_STEPS * 1000)), 
        ((int)(1.0 / DT_RENDER)), 0.8f);

    set_scene_camera(&scene,
        (Vec3){-3.0f, 3.0f, -3.0f},  // Camera position
        (Vec3){0.0f, 1.0f, 0.0f},    // Look at point
        (Vec3){0.0f, 1.0f, 0.0f},    // Up vector
        60.0f                         // FOV
    );
    
    set_scene_light(&scene,
        (Vec3){1.0f, 1.0f, -1.0f},   // Direction
        (Vec3){1.4f, 1.4f, 1.4f}     // Color
    );
    
    // Add meshes
    add_mesh_to_scene(&scene, 
        create_mesh("raytracer/drone.obj", "raytracer/drone.webp"));
    add_mesh_to_scene(&scene,
        create_mesh("raytracer/ground.obj", "raytracer/ground.webp"));

    // Start timing
    struct timeval start_time;
    gettimeofday(&start_time, NULL);

    // Run visualization
    visualize_rollout(policy, &scene);

    // Save animation
    char filename[64];
    time_t current_time = time(NULL);
    strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_flight.webp", 
             localtime(&current_time));
    save_scene(&scene, filename);
    
    struct timeval end_time;
    gettimeofday(&end_time, NULL);
    double elapsed = (end_time.tv_sec - start_time.tv_sec) + 
                    (end_time.tv_usec - start_time.tv_usec) / 1e6;
    
    printf("\nRendered %d frames in %.1f seconds (%.1f FPS)\n", 
           scene.frame_count, elapsed, scene.frame_count / elapsed);
    printf("Saved animation to %s\n", filename);

    // Cleanup
    destroy_scene(&scene);
    free(policy);
    return 0;
}