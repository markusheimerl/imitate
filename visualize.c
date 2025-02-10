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
#define SIM_TIME    10.0  // Simulation duration in seconds

// Prepare state vector for policy input
void prepare_state(const Quad* quad, const double* target, float* state) {
    // Position error (3)
    for(int i = 0; i < 3; i++) {
        state[i] = (float)(target[i] - quad->linear_position_W[i]);
    }
    
    // Velocity error (3)
    for(int i = 0; i < 3; i++) {
        state[i+3] = (float)(target[i+3] - quad->linear_velocity_W[i]);
    }
    
    // Current orientation (9)
    for(int i = 0; i < 9; i++) {
        state[i+6] = (float)quad->R_W_B[i];
    }
    
    // Current angular velocity (3)
    for(int i = 0; i < 3; i++) {
        state[i+15] = (float)quad->angular_velocity_B[i];
    }
    
    // Target yaw (1)
    state[18] = (float)target[6];
}

int main(int argc, char* argv[]) {
    if(argc != 2) {
        printf("Usage: %s <policy_file>\n", argv[0]);
        return 1;
    }

    // Load policy network
    Net* policy = load_model(argv[1]);
    if(!policy) {
        printf("Failed to load policy from %s\n", argv[1]);
        return 1;
    }

    srand(time(NULL));
    
    // Initialize random target position and yaw
    double target[7] = {
        (double)rand() / RAND_MAX * 4.0 - 2.0,    // x: [-2,2]
        (double)rand() / RAND_MAX * 2.0 + 0.5,    // y: [0.5,2.5]
        (double)rand() / RAND_MAX * 4.0 - 2.0,    // z: [-2,2]
        0.0, 0.0, 0.0,                            // Zero velocity target
        (double)rand() / RAND_MAX * 2.0 * M_PI    // yaw: [0,2π]
    };
    
    printf("Target position: (%.2f, %.2f, %.2f) with yaw: %.2f rad\n", 
           target[0], target[1], target[2], target[6]);
    
    // Initialize quadcopter
    Quad* quad = create_quad(0.0, 0.0, 0.0);
    
    // Initialize raytracer scene
    Scene scene = create_scene(400, 300, (int)(SIM_TIME * 1000), 24, 0.4f);
    
    // Set up camera and scene exactly as in sim.c
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
    
    // Allocate state buffer for policy
    float state[25];

    // Main simulation loop
    for (int t = 0; t < (int)(SIM_TIME / DT_PHYSICS); t++) {
        // Physics update
        if (t_physics >= DT_PHYSICS) {
            update_quad(quad, DT_PHYSICS);
            t_physics = 0.0;
        }
        
        // Control update
        if (t_control >= DT_CONTROL) {
            // Use learned policy instead of geometric controller
            prepare_state(quad, target, state);
            forward_pass(policy, state);
            
            // Apply policy outputs as motor commands
            for(int i = 0; i < 4; i++) {
                quad->omega_next[i] = (double)policy->predictions[i];
                printf("Motor %d: %.2f\n", i, quad->omega_next[i]);
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

    // Save animation
    char filename[64];
    strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_policy_flight.webp", 
             localtime(&(time_t){time(NULL)}));
    save_scene(&scene, filename);

    // Cleanup
    destroy_scene(&scene);
    free_net(policy);
    free(quad);
    return 0;
}