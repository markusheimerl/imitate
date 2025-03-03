#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "sim/quad.h"
#include "sim/raytracer/scene.h"
#include "ssm/ssm.h"  // Using CPU version

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define DT_RENDER   (1.0 / 24.0)
#define SIM_TIME    10.0  // Simulation duration in seconds

double random_range(double min, double max) {
    return min + (double)rand() / RAND_MAX * (max - min);
}

int main(int argc, char* argv[]) {
    if(argc != 4) {
        printf("Usage: %s <perception_model_file> <planning_model_file> <control_model_file>\n", argv[0]);
        return 1;
    }

    // Load all three SSM models
    SSM* perception_ssm = load_ssm(argv[1]);
    SSM* planning_ssm = load_ssm(argv[2]);
    SSM* control_ssm = load_ssm(argv[3]);

    // Print network dimensions
    printf("Loaded Perception Model dimensions:\n");
    printf("Input dim: %d\n", perception_ssm->input_dim);
    printf("State dim: %d\n", perception_ssm->state_dim);
    printf("Output dim: %d\n", perception_ssm->output_dim);
    printf("Batch size: %d\n", perception_ssm->batch_size);
    
    printf("\nLoaded Planning Model dimensions:\n");
    printf("Input dim: %d\n", planning_ssm->input_dim);
    printf("State dim: %d\n", planning_ssm->state_dim);
    printf("Output dim: %d\n", planning_ssm->output_dim);
    printf("Batch size: %d\n", planning_ssm->batch_size);
    
    printf("\nLoaded Control Model dimensions:\n");
    printf("Input dim: %d\n", control_ssm->input_dim);
    printf("State dim: %d\n", control_ssm->state_dim);
    printf("Output dim: %d\n", control_ssm->output_dim);
    printf("Batch size: %d\n", control_ssm->batch_size);
    
    // Verify model compatibility
    if (perception_ssm->output_dim != planning_ssm->input_dim) {
        printf("Error: Model dimensions don't match! Perception output: %d, Planning input: %d\n", 
               perception_ssm->output_dim, planning_ssm->input_dim);
        free_ssm(perception_ssm);
        free_ssm(planning_ssm);
        free_ssm(control_ssm);
        return 1;
    }
    
    if (planning_ssm->output_dim != control_ssm->input_dim) {
        printf("Error: Model dimensions don't match! Planning output: %d, Control input: %d\n", 
               planning_ssm->output_dim, control_ssm->input_dim);
        free_ssm(perception_ssm);
        free_ssm(planning_ssm);
        free_ssm(control_ssm);
        return 1;
    }

    srand(time(NULL));
    
    // Initialize quadcopter with random position
    Quad quad = create_quad(
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
        random_range(-M_PI, M_PI)   // yaw
    };
    
    printf("Target position: (%.2f, %.2f, %.2f) with yaw: %.2f rad\n", target[0], target[1], target[2], target[6]);
    
    // Initialize scenes
    Scene scene = create_scene(400, 300, (int)(SIM_TIME * 1000), 24, 0.4f);
    Scene fpv_scene = create_scene(20, 15, (int)(SIM_TIME * 1000), 24, 1.0f);
    
    // Set up lighting for both scenes
    set_scene_light(&scene,
        (Vec3){1.0f, 1.0f, -1.0f},
        (Vec3){1.4f, 1.4f, 1.4f}
    );
    
    set_scene_light(&fpv_scene,
        (Vec3){1.0f, 1.0f, -1.0f},
        (Vec3){1.4f, 1.4f, 1.4f}
    );
    
    // Create meshes
    Mesh drone = create_mesh("sim/raytracer/drone.obj", "sim/raytracer/drone.webp");
    Mesh ground = create_mesh("sim/raytracer/ground.obj", "sim/raytracer/ground.webp");
    Mesh treasure = create_mesh("sim/raytracer/treasure.obj", "sim/raytracer/treasure.webp");
    
    // Add meshes to scenes
    add_mesh_to_scene(&scene, drone);
    add_mesh_to_scene(&scene, ground);
    add_mesh_to_scene(&scene, treasure);
    add_mesh_to_scene(&fpv_scene, drone);
    add_mesh_to_scene(&fpv_scene, ground);
    add_mesh_to_scene(&fpv_scene, treasure);

    // Set treasure position (target) for both scenes
    set_mesh_position(&scene.meshes[2], (Vec3){(float)target[0], (float)target[1], (float)target[2]});
    set_mesh_rotation(&scene.meshes[2], (Vec3){0.0f, (float)target[6], 0.0f});
    
    set_mesh_position(&fpv_scene.meshes[2], (Vec3){(float)target[0], (float)target[1], (float)target[2]});
    set_mesh_rotation(&fpv_scene.meshes[2], (Vec3){0.0f, (float)target[6], 0.0f});

    // Set up chase camera with 60 degree FOV
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
    
    // Allocate input buffer for perception model
    float* perception_input = (float*)calloc(perception_ssm->batch_size * perception_ssm->input_dim, sizeof(float));
    
    // Allocate intermediate buffers for connecting the models
    float* planning_input = (float*)calloc(planning_ssm->batch_size * planning_ssm->input_dim, sizeof(float));
    float* control_input = (float*)calloc(control_ssm->batch_size * control_ssm->input_dim, sizeof(float));
    
    // Reset internal states of all models
    memset(perception_ssm->state, 0, perception_ssm->batch_size * perception_ssm->state_dim * sizeof(float));
    memset(planning_ssm->state, 0, planning_ssm->batch_size * planning_ssm->state_dim * sizeof(float));
    memset(control_ssm->state, 0, control_ssm->batch_size * control_ssm->state_dim * sizeof(float));

    // Main simulation loop
    for (int t = 0; t < (int)(SIM_TIME / DT_PHYSICS); t++) {
        // Physics update
        if (t_physics >= DT_PHYSICS) {
            update_quad(&quad, DT_PHYSICS);
            t_physics = 0.0;
        }
        
        // Control update
        if (t_control >= DT_CONTROL) {
            // Fill input for perception model: IMU data, position, velocity, and target
            int idx = 0;
            
            // IMU measurements (6)
            for(int i = 0; i < 3; i++) perception_input[idx++] = (float)quad.gyro_measurement[i];
            for(int i = 0; i < 3; i++) perception_input[idx++] = (float)quad.accel_measurement[i];
            
            // Position and velocity (6)
            for(int i = 0; i < 3; i++) perception_input[idx++] = (float)quad.linear_position_W[i];
            for(int i = 0; i < 3; i++) perception_input[idx++] = (float)quad.linear_velocity_W[i];
            
            // Target position and yaw (4)
            for(int i = 0; i < 3; i++) perception_input[idx++] = (float)target[i];
            perception_input[idx++] = (float)target[6];
            
            // Forward pass through perception model
            forward_pass(perception_ssm, perception_input);
            
            // Copy the output of the perception model as input to the planning model
            memcpy(planning_input, perception_ssm->predictions, perception_ssm->output_dim * sizeof(float));
            
            // Forward pass through planning model
            forward_pass(planning_ssm, planning_input);
            
            // Copy the output of the planning model as input to the control model
            memcpy(control_input, planning_ssm->predictions, planning_ssm->output_dim * sizeof(float));
            
            // Forward pass through control model
            forward_pass(control_ssm, control_input);
            
            // Apply predicted motor commands from the control model
            for (int i = 0; i < 4; i++) {
                quad.omega_next[i] = (double)control_ssm->predictions[i];
            }
            
            t_control = 0.0;
        }
        
        // Render update
        if (t_render >= DT_RENDER) {
            // Get drone position and orientation for visualization
            Vec3 pos = {
                (float)quad.linear_position_W[0],
                (float)quad.linear_position_W[1],
                (float)quad.linear_position_W[2]
            };
            
            Vec3 rot = {
                atan2f(quad.R_W_B[7], quad.R_W_B[8]),
                asinf(-quad.R_W_B[6]),
                atan2f(quad.R_W_B[3], quad.R_W_B[0])
            };

            // Update drone position in both scenes
            set_mesh_position(&scene.meshes[0], pos);
            set_mesh_rotation(&scene.meshes[0], rot);
            
            set_mesh_position(&fpv_scene.meshes[0], pos);
            set_mesh_rotation(&fpv_scene.meshes[0], rot);
            
            // Update FPV camera to match drone's position and orientation
            Vec3 forward = {
                (float)quad.R_W_B[2],  // Third column
                (float)quad.R_W_B[5],
                (float)quad.R_W_B[8]
            };
            
            Vec3 up = {
                (float)quad.R_W_B[1],  // Second column
                (float)quad.R_W_B[4],
                (float)quad.R_W_B[7]
            };
            
            // Set camera position above the drone
            Vec3 camera_offset = {
                up.x * 0.15f,
                up.y * 0.15f,
                up.z * 0.15f
            };
            
            Vec3 fpv_pos = {
                pos.x + camera_offset.x,
                pos.y + camera_offset.y,
                pos.z + camera_offset.z
            };
            
            // Calculate look-at point (position + forward)
            Vec3 look_at = {
                pos.x + forward.x,  // Look at point is in front of drone's position
                pos.y + forward.y,
                pos.z + forward.z
            };
            
            // Set FPV camera
            set_scene_camera(&fpv_scene, fpv_pos, look_at, up, 70.0f);
            
            // Render both scenes
            render_scene(&scene);
            render_scene(&fpv_scene);
            
            // Advance to next frame in both scenes
            next_frame(&scene);
            next_frame(&fpv_scene);
            
            update_progress_bar((int)(t * DT_PHYSICS / DT_RENDER), (int)(SIM_TIME * 24), start_time);
            
            t_render = 0.0;
        }
        
        // Increment timers
        t_physics += DT_PHYSICS;
        t_control += DT_PHYSICS;
        t_render += DT_PHYSICS;
    }

    printf("\nFinal position: (%.2f, %.2f, %.2f) with yaw %.2f or ±%.2f\n", 
           quad.linear_position_W[0], quad.linear_position_W[1], quad.linear_position_W[2],
           asinf(-quad.R_W_B[6]), M_PI - fabs(asinf(-quad.R_W_B[6])));
    
    // Calculate distance to target
    double dist = sqrt(pow(quad.linear_position_W[0] - target[0], 2) + 
                     pow(quad.linear_position_W[1] - target[1], 2) + 
                     pow(quad.linear_position_W[2] - target[2], 2));
    printf("Distance to target: %.2f meters\n", dist);
    
    // Save animations
    char filename[64];
    char fpv_filename[64];
    
    time_t now = time(NULL);
    strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_flight.webp", localtime(&now));
    strftime(fpv_filename, sizeof(fpv_filename), "%Y%m%d_%H%M%S_flight_fpv.webp", localtime(&now));
    
    save_scene(&scene, filename);
    save_scene(&fpv_scene, fpv_filename);
    
    printf("Third-person view saved to: %s\n", filename);
    printf("First-person view saved to: %s\n", fpv_filename);

    // Cleanup
    free(perception_input);
    free(planning_input);
    free(control_input);
    destroy_mesh(&drone);
    destroy_mesh(&ground);
    destroy_mesh(&treasure);
    destroy_scene(&scene);
    destroy_scene(&fpv_scene);
    free_ssm(perception_ssm);
    free_ssm(planning_ssm);
    free_ssm(control_ssm);
    return 0;
}