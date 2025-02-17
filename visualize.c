#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "sim/quad.h"
#include "sim/raytracer/scene.h"
#include "ssm/ssm.h"

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
    SSM* policy = load_model(argv[1]);
    policy->batch_size = 1;  // Set batch size to 1 for inference

    // Print network dimensions
    printf("Loaded SSM policy dimensions:\n");
    printf("Input dim: %d\n", policy->input_dim);
    printf("State dim: %d\n", policy->state_dim);
    printf("Output dim: %d\n", policy->output_dim);
    printf("Batch size: %d\n", policy->batch_size);

    srand(time(NULL));
    
    // Initialize quadcopter with random position
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
    
    // Initialize scene
    Scene scene = create_scene(400, 300, (int)(SIM_TIME * 1000), 24, 0.4f);
    
    // Set up lighting
    set_scene_light(&scene,
        (Vec3){1.0f, 1.0f, -1.0f},
        (Vec3){1.4f, 1.4f, 1.4f}
    );
    
    // Add meshes to scene
    Mesh drone = create_mesh("sim/raytracer/drone.obj", "sim/raytracer/drone.webp");
    Mesh ground = create_mesh("sim/raytracer/ground.obj", "sim/raytracer/ground.webp");
    Mesh treasure = create_mesh("sim/raytracer/treasure.obj", "sim/raytracer/treasure.webp");
    
    add_mesh_to_scene(&scene, drone);
    add_mesh_to_scene(&scene, ground);
    add_mesh_to_scene(&scene, treasure);

    // Set treasure position
    Vec3 treasure_pos = {
        (float)target[0],
        (float)target[1],
        (float)target[2]
    };
    set_mesh_position(&scene.meshes[2], treasure_pos);

    // Set up camera
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
    
    // Preallocate input buffer
    float* input = (float*)malloc(policy->input_dim * sizeof(float));

    // Main simulation loop
    for (int t = 0; t < (int)(SIM_TIME / DT_PHYSICS); t++) {
        // Physics update
        if (t_physics >= DT_PHYSICS) {
            update_quad(quad, DT_PHYSICS);
            t_physics = 0.0;
        }
        
        // Control update
        if (t_control >= DT_CONTROL) {
            // Prepare reduced input (velocity, angular velocity, target)
            for(int i = 0; i < 3; i++) input[i] = (float)quad->linear_velocity_W[i];
            for(int i = 0; i < 3; i++) input[i+3] = (float)quad->angular_velocity_B[i];
            for(int i = 0; i < 3; i++) input[i+6] = (float)target[i];
            input[9] = (float)target[6];
            
            // Forward pass through policy network
            forward_pass(policy, input);
            
            // Apply predicted motor commands
            for (int i = 0; i < 4; i++) {
                quad->omega_next[i] = (double)policy->predictions[i];
            }
            
            t_control = 0.0;
        }
        
        // Render update
        if (t_render >= DT_RENDER) {
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

            set_mesh_position(&scene.meshes[0], pos);
            set_mesh_rotation(&scene.meshes[0], rot);
            
            render_scene(&scene);
            next_frame(&scene);
            
            update_progress_bar((int)(t * DT_PHYSICS / DT_RENDER), 
                              (int)(SIM_TIME * 24), start_time);
            
            t_render = 0.0;
        }
        
        // Increment timers
        t_physics += DT_PHYSICS;
        t_control += DT_PHYSICS;
        t_render += DT_PHYSICS;
    }

    printf("\nFinal position: (%.2f, %.2f, %.2f) with yaw: %.2f rad\n", 
           quad->linear_position_W[0], quad->linear_position_W[1], 
           quad->linear_position_W[2], 
           fmod(atan2(quad->R_W_B[3], quad->R_W_B[0]) + 2 * M_PI, 2 * M_PI));
    
    // Save animation
    char filename[64];
    time_t now = time(NULL);
    strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_policy_flight.webp", 
             localtime(&now));  
    save_scene(&scene, filename);

    // Cleanup
    free(input);
    destroy_scene(&scene);
    free_ssm(policy);
    free(quad);
    return 0;
}