#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "sim/quad.h"
#include "sim/raytracer/scene.h"
#include "mlp/mlp.h"

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define DT_RENDER   (1.0 / 24.0)
#define SIM_TIME    5.0  // Simulation duration in seconds

double random_range(double min, double max) {
    return min + (double)rand() / RAND_MAX * (max - min);
}

int main(int argc, char* argv[]) {
    if(argc != 2) {
        printf("Usage: %s <policy_file>\n", argv[0]);
        return 1;
    }

    // Load policy network
    Net* policy = load_model(argv[1]);

    // Print network dimensions
    printf("Loaded policy network dimensions:\n");
    printf("Input dim: %d\n", policy->input_dim);
    printf("Hidden dim: %d\n", policy->hidden_dim);
    printf("Output dim: %d\n", policy->output_dim);
    printf("Batch size: %d\n", policy->batch_size);

    srand(time(NULL));
    
    // Initialize quadcopter
    Quad* quad = create_quad(
        random_range(-2.0, 2.0),
        random_range(0.0, 2.0),    // Always at or above ground
        random_range(-2.0, 2.0)
    );
    
    // Initialize random target position and yaw
    double target[7] = {
        random_range(-2.0, 2.0),    // x
        random_range(1.0, 3.0),     // y: Always above ground
        random_range(-2.0, 2.0),    // z
        0.0, 0.0, 0.0,              // vx, vy, vz
        random_range(0.0, 2*M_PI)   // yaw
    };
    
    printf("Target position: (%.2f, %.2f, %.2f) with yaw: %.2f rad\n", 
           target[0], target[1], target[2], target[6]);
    
    // Initialize both scenes identically
    Scene scene = create_scene(400, 300, (int)(SIM_TIME * 1000), 24, 0.4f);
    Scene drone_fpv = create_scene(400, 300, (int)(SIM_TIME * 1000), 24, 0.4f);
    
    // Set up identical lighting for both scenes
    set_scene_light(&scene,
        (Vec3){1.0f, 1.0f, -1.0f},
        (Vec3){1.4f, 1.4f, 1.4f}
    );
    set_scene_light(&drone_fpv,
        (Vec3){1.0f, 1.0f, -1.0f},
        (Vec3){1.4f, 1.4f, 1.4f}
    );
    
    // Add identical meshes to both scenes
    Mesh drone = create_mesh("sim/raytracer/drone.obj", "sim/raytracer/drone.webp");
    Mesh ground = create_mesh("sim/raytracer/ground.obj", "sim/raytracer/ground.webp");
    
    add_mesh_to_scene(&scene, drone);
    add_mesh_to_scene(&scene, ground);
    
    Mesh drone_fpv_model = create_mesh("sim/raytracer/drone.obj", "sim/raytracer/drone.webp");
    Mesh ground_fpv = create_mesh("sim/raytracer/ground.obj", "sim/raytracer/ground.webp");
    
    add_mesh_to_scene(&drone_fpv, drone_fpv_model);
    add_mesh_to_scene(&drone_fpv, ground_fpv);

    // Set up initial camera for third-person view
    set_scene_camera(&scene,
        (Vec3){-3.0f, 3.0f, -3.0f},
        (Vec3){0.0f, 0.0f, 0.0f},
        (Vec3){0.0f, 1.0f, 0.0f},
        60.0f
    );

    // Initialize timers
    double t_physics = 0.0;
    double t_control = 0.0;
    double t_render = 0.0;
    clock_t start_time = clock();
    
    // Preallocate batch-sized input buffer
    float* batch_input = (float*)calloc(policy->batch_size * policy->input_dim, sizeof(float));

    // Main simulation loop
    for (int t = 0; t < (int)(SIM_TIME / DT_PHYSICS); t++) {
        // Physics update
        if (t_physics >= DT_PHYSICS) {
            update_quad(quad, DT_PHYSICS);
            t_physics = 0.0;
        }
        
        // Control update
        if (t_control >= DT_CONTROL) {
            // Current position (3)
            for(int i = 0; i < 3; i++) {
                batch_input[i] = (float)quad->linear_position_W[i];
            }
            
            // Current velocity (3)
            for(int i = 0; i < 3; i++) {
                batch_input[i+3] = (float)quad->linear_velocity_W[i];
            }
            
            // Current orientation (9)
            for(int i = 0; i < 9; i++) {
                batch_input[i+6] = (float)quad->R_W_B[i];
            }
            
            // Current angular velocity (3)
            for(int i = 0; i < 3; i++) {
                batch_input[i+15] = (float)quad->angular_velocity_B[i];
            }

            // Target position (3)
            for(int i = 0; i < 3; i++) {
                batch_input[i+18] = (float)target[i];
            }
            
            // Target yaw (1)
            batch_input[21] = (float)target[6];
            
            // Forward pass through policy network
            forward_pass(policy, batch_input);
            
            // Apply predicted motor commands
            for(int i = 0; i < 4; i++) {
                quad->omega_next[i] = (double)policy->predictions[i];
            }
            
            t_control = 0.0;
        }
        
        // Render update
        if (t_render >= DT_RENDER) {
            // Update mesh positions and rotations (identical for both scenes)
            Vec3 pos = {
                (float)quad->linear_position_W[0],
                (float)quad->linear_position_W[1],
                (float)quad->linear_position_W[2]
            };
            
            Vec3 rot = {
                atan2f(quad->R_W_B[7], quad->R_W_B[8]),
                asinf(-quad->R_W_B[6]),
                atan2f(quad->R_W_B[3], quad->R_W_B[0])
            };

            // Update meshes in both scenes
            set_mesh_position(&scene.meshes[0], pos);
            set_mesh_position(&drone_fpv.meshes[0], pos);
            set_mesh_rotation(&scene.meshes[0], rot);
            set_mesh_rotation(&drone_fpv.meshes[0], rot);

            // Calculate drone FPV camera position
            Vec3 cam_offset = {0.0f, 0.2f, 0.0f};  // Small offset upward from drone center
            
            // Transform camera offset by drone's rotation matrix
            Vec3 rotated_offset = {
                cam_offset.x * quad->R_W_B[0] + cam_offset.y * quad->R_W_B[1] + cam_offset.z * quad->R_W_B[2],
                cam_offset.x * quad->R_W_B[3] + cam_offset.y * quad->R_W_B[4] + cam_offset.z * quad->R_W_B[5],
                cam_offset.x * quad->R_W_B[6] + cam_offset.y * quad->R_W_B[7] + cam_offset.z * quad->R_W_B[8]
            };

            // Set FPV camera position relative to drone
            Vec3 cam_pos = {
                pos.x + rotated_offset.x,
                pos.y + rotated_offset.y,
                pos.z + rotated_offset.z
            };

            // Calculate look-at point (a bit forward from drone's position)
            Vec3 forward_offset = {0.0f, 0.0f, 1.0f};  // Looking forward along drone's z-axis
            Vec3 rotated_forward = {
                forward_offset.x * quad->R_W_B[0] + forward_offset.y * quad->R_W_B[1] + forward_offset.z * quad->R_W_B[2],
                forward_offset.x * quad->R_W_B[3] + forward_offset.y * quad->R_W_B[4] + forward_offset.z * quad->R_W_B[5],
                forward_offset.x * quad->R_W_B[6] + forward_offset.y * quad->R_W_B[7] + forward_offset.z * quad->R_W_B[8]
            };

            Vec3 look_at = {
                cam_pos.x + rotated_forward.x,
                cam_pos.y + rotated_forward.y,
                cam_pos.z + rotated_forward.z
            };

            // Update FPV camera
            set_scene_camera(&drone_fpv,
                cam_pos,
                look_at,
                (Vec3){0.0f, 1.0f, 0.0f},  // Keep world-up vector
                70.0f
            );
            
            // Render both views
            render_scene(&scene);
            render_scene(&drone_fpv);
            
            next_frame(&scene);
            next_frame(&drone_fpv);
            
            update_progress_bar((int)(t * DT_PHYSICS / DT_RENDER), 
                              (int)(SIM_TIME * 24), start_time);
            
            t_render = 0.0;
        }
        
        // Increment timers
        t_physics += DT_PHYSICS;
        t_control += DT_PHYSICS;
        t_render += DT_PHYSICS;
    }

    printf("\nFinal position: (%.2f, %.2f, %.2f)\n", 
           quad->linear_position_W[0], quad->linear_position_W[1], quad->linear_position_W[2]);

    // Save both animations
    char filename[64];
    char filename_fpv[64];
    strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_policy_flight.webp", 
             localtime(&(time_t){time(NULL)}));
    strftime(filename_fpv, sizeof(filename_fpv), "%Y%m%d_%H%M%S_policy_flight_fpv.webp", 
             localtime(&(time_t){time(NULL)}));
             
    save_scene(&scene, filename);
    save_scene(&drone_fpv, filename_fpv);

    // Cleanup
    free(batch_input);
    destroy_scene(&scene);
    destroy_scene(&drone_fpv);
    free_net(policy);
    free(quad);
    return 0;
}