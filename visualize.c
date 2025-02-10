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
#define SIM_TIME    10.0  // 10 second flight

// Prepare state vector for policy input (no normalization)
void prepare_state(const Quad* quad, const double* target, float* state) {
    // Position
    for(int i = 0; i < 3; i++) {
        state[i] = (float)quad->linear_position_W[i];
    }
    
    // Velocity
    for(int i = 0; i < 3; i++) {
        state[i+3] = (float)quad->linear_velocity_W[i];
    }
    
    // Rotation matrix (quaternion)
    for(int i = 0; i < 4; i++) {
        state[i+6] = (float)quad->R_W_B[i];
    }
    
    // Angular velocity
    for(int i = 0; i < 3; i++) {
        state[i+10] = (float)quad->angular_velocity_B[i];
    }
    
    // Target position
    for(int i = 0; i < 3; i++) {
        state[i+13] = (float)target[i];
    }
    
    // Target velocity
    for(int i = 0; i < 3; i++) {
        state[i+16] = (float)target[i+3];
    }
    
    // Target yaw
    state[19] = (float)target[6];
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

    // Initialize scene
    Scene scene = create_scene(800, 600, 240, 24, 0.4f); // 10 seconds at 24fps
    
    // Set up camera
    set_scene_camera(&scene,
        (Vec3){-3.0f, 3.0f, -3.0f},
        (Vec3){0.0f, 0.0f, 0.0f},
        (Vec3){0.0f, 1.0f, 0.0f},
        60.0f
    );
    
    // Set up light
    set_scene_light(&scene,
        (Vec3){1.0f, 1.0f, -1.0f},
        (Vec3){1.4f, 1.4f, 1.4f}
    );
    
    // Add meshes
    Mesh drone = create_mesh("sim/raytracer/drone.obj", "sim/raytracer/drone.webp");
    add_mesh_to_scene(&scene, drone);
    
    Mesh ground = create_mesh("sim/raytracer/ground.obj", "sim/raytracer/ground.webp");
    add_mesh_to_scene(&scene, ground);

    // Initialize quadcopter with random starting position
    Quad* quad = create_quad(
        (double)rand() / RAND_MAX * 2.0 - 1.0,  // x: [-1,1]
        0.5,                                     // y: start at 0.5m
        (double)rand() / RAND_MAX * 2.0 - 1.0   // z: [-1,1]
    );

    // Set random target
    double target[7] = {
        (double)rand() / RAND_MAX * 4.0 - 2.0,  // x: [-2,2]
        1.5,                                     // y: fixed at 1.5m
        (double)rand() / RAND_MAX * 4.0 - 2.0,  // z: [-2,2]
        0.0, 0.0, 0.0,                          // zero velocity target
        (double)rand() / RAND_MAX * 2.0 * M_PI  // random yaw
    };

    // Initialize timers
    double t_physics = 0.0;
    double t_control = 0.0;
    double t_render = 0.0;
    int frame = 0;
    clock_t start_time = clock();

    // Allocate state buffer for policy
    float state[20];

    // Main simulation loop
    while (frame < scene.frame_count) {
        // Physics update
        if (t_physics >= DT_PHYSICS) {
            update_quad(quad, DT_PHYSICS);
            t_physics = 0.0;
        }
        
        // Control update
        if (t_control >= DT_CONTROL) {
            // Prepare state input for policy
            prepare_state(quad, target, state);
            
            // Get action from policy
            forward_pass(policy, state);
            
            // Use policy output directly as motor commands
            for(int i = 0; i < 4; i++) {
                quad->omega_next[i] = (double)policy->predictions[i];
            }
            
            t_control = 0.0;
        }
        
        // Render update
        if (t_render >= DT_RENDER) {
            // Update drone position and orientation
            set_mesh_position(&scene.meshes[0], 
                (Vec3){(float)quad->linear_position_W[0], 
                       (float)quad->linear_position_W[1], 
                       (float)quad->linear_position_W[2]});
            
            // Convert rotation matrix to Euler angles
            float roll = atan2f(quad->R_W_B[7], quad->R_W_B[8]);
            float pitch = asinf(-quad->R_W_B[6]);
            float yaw = atan2f(quad->R_W_B[3], quad->R_W_B[0]);
            
            set_mesh_rotation(&scene.meshes[0], (Vec3){roll, pitch, yaw});
            
            // Render frame
            render_scene(&scene);
            next_frame(&scene);
            
            // Update progress
            update_progress_bar(frame, scene.frame_count, start_time);
            
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
    strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_policy_flight.webp", 
             localtime(&(time_t){time(NULL)}));
    save_scene(&scene, filename);

    // Cleanup
    destroy_scene(&scene);
    free_net(policy);
    free(quad);
    
    printf("\nVisualization completed and saved to %s\n", filename);
    return 0;
}