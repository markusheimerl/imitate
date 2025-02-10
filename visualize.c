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
    
    
    // Initialize raytracer scene
    Scene scene = create_scene(800, 600, (int)(SIM_TIME * 1000), 24, 1.0f);
    
    // Set up camera and scene
    set_scene_camera(&scene,
        (Vec3){-3.0f, 3.0f, -3.0f},
        (Vec3){0.0f, 0.0f, 0.0f},
        (Vec3){0.0f, 1.0f, 0.0f},
        60.0f
    );
    
    set_scene_light(&scene,
        (Vec3){1.0f, 1.0f, -1.0f},
        (Vec3){1.4f, 1.4f, 1.4f}
    );
    
    Mesh drone = create_mesh("sim/raytracer/drone.obj", "sim/raytracer/drone.webp");
    add_mesh_to_scene(&scene, drone);
    
    Mesh ground = create_mesh("sim/raytracer/ground.obj", "sim/raytracer/ground.webp");
    add_mesh_to_scene(&scene, ground);

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

            // Target velocity (3)
            for(int i = 0; i < 3; i++) {
                batch_input[i+21] = (float)target[i+3];
            }
            
            // Target yaw (1)
            batch_input[24] = (float)target[6];
            
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
            set_mesh_position(&scene.meshes[0], 
                (Vec3){(float)quad->linear_position_W[0], 
                       (float)quad->linear_position_W[1], 
                       (float)quad->linear_position_W[2]});
            
            set_mesh_rotation(&scene.meshes[0], 
                (Vec3){
                    atan2f(quad->R_W_B[7], quad->R_W_B[8]),
                    asinf(-quad->R_W_B[6]),
                    atan2f(quad->R_W_B[3], quad->R_W_B[0])
                }
            );
            
            render_scene(&scene);
            next_frame(&scene);
            update_progress_bar((int)(t * DT_PHYSICS / DT_RENDER), (int)(SIM_TIME * 24), start_time);
            
            t_render = 0.0;
        }
        
        // Increment timers
        t_physics += DT_PHYSICS;
        t_control += DT_PHYSICS;
        t_render += DT_PHYSICS;
    }

    printf("Final position: (%.2f, %.2f, %.2f)\n", 
           quad->linear_position_W[0], quad->linear_position_W[1], quad->linear_position_W[2]);

    // Save animation
    char filename[64];
    strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_policy_flight.webp", 
             localtime(&(time_t){time(NULL)}));
    save_scene(&scene, filename);

    // Cleanup
    free(batch_input);
    destroy_scene(&scene);
    free_net(policy);
    free(quad);
    return 0;
}