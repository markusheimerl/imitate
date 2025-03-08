#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "sim/quad.h"
#include "sim/raytracer/scene.h"
#include "ssm/gpu/ssm.h"

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define DT_RENDER   (1.0 / 24.0)
#define SIM_TIME    10.0  // Simulation duration in seconds

double random_range(double min, double max) {
    return min + (double)rand() / RAND_MAX * (max - min);
}

int main(int argc, char* argv[]) {
    if(argc != 5) {
        printf("Usage: %s <layer1_model_file> <layer2_model_file> <layer3_model_file> <layer4_model_file>\n", argv[0]);
        return 1;
    }

    // Load all four SSM models
    SSM* layer1_ssm = load_ssm(argv[1], 1);
    SSM* layer2_ssm = load_ssm(argv[2], 1);
    SSM* layer3_ssm = load_ssm(argv[3], 1);
    SSM* layer4_ssm = load_ssm(argv[4], 1);

    // Initialize random seed
    srand(time(NULL));
    
    // Initialize drone with random position and orientation
    double drone_x = random_range(-2.0, 2.0);
    double drone_y = random_range(0.5, 2.0);
    double drone_z = random_range(-2.0, 2.0);
    double drone_yaw = random_range(-M_PI, M_PI);
    
    // Create quad with random position and orientation
    Quad quad = create_quad(drone_x, drone_y, drone_z, drone_yaw);
    
    // Initialize random target position and yaw
    double target_x = random_range(-2.0, 2.0);
    double target_y = random_range(0.5, 2.5);
    double target_z = random_range(-2.0, 2.0);
    double target_yaw = random_range(-M_PI, M_PI);
    
    // Create target array (position, velocity, and desired yaw)
    double target[7] = {
        target_x, target_y, target_z,    // Target position
        0.0, 0.0, 0.0,                   // Zero velocity target
        target_yaw                       // Random target yaw
    };
    
    printf("Drone starts at (%.2f, %.2f, %.2f) with yaw %.2f\n", 
           drone_x, drone_y, drone_z, drone_yaw);
    printf("Target at (%.2f, %.2f, %.2f) with yaw %.2f\n", 
           target_x, target_y, target_z, target_yaw);
    
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
    
    // Set up camera
    set_scene_camera(&scene,
        (Vec3){-3.0f, 3.0f, -3.0f},
        (Vec3){0.0f, 0.0f, 0.0f},
        (Vec3){0.0f, 1.0f, 0.0f},
        60.0f
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
    set_mesh_position(&scene.meshes[2], (Vec3){(float)target_x, (float)target_y, (float)target_z});
    set_mesh_position(&fpv_scene.meshes[2], (Vec3){(float)target_x, (float)target_y, (float)target_z});

    // Initialize timers
    double t_physics = 0.0;
    double t_control = 0.0;
    double t_render = 0.0;
    clock_t start_time = clock();
    
    // Allocate and prepare host and device memory for inputs and outputs
    
    // Host buffers for inputs and intermediate results
    float* h_layer1_input = (float*)calloc(layer1_ssm->batch_size * layer1_ssm->input_dim, sizeof(float));
    float* h_layer4_output = (float*)calloc(layer4_ssm->batch_size * layer4_ssm->output_dim, sizeof(float));

    // GPU buffers for inputs and outputs
    float *d_layer1_input, *d_layer2_input, *d_layer3_input, *d_layer4_input;
    CHECK_CUDA(cudaMalloc(&d_layer1_input, layer1_ssm->batch_size * layer1_ssm->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_layer2_input, layer2_ssm->batch_size * layer2_ssm->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_layer3_input, layer3_ssm->batch_size * layer3_ssm->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_layer4_input, layer4_ssm->batch_size * layer4_ssm->input_dim * sizeof(float)));
    
    // Reset internal states of all models
    CHECK_CUDA(cudaMemset(layer1_ssm->d_state, 0, layer1_ssm->batch_size * layer1_ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(layer2_ssm->d_state, 0, layer2_ssm->batch_size * layer2_ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(layer3_ssm->d_state, 0, layer3_ssm->batch_size * layer3_ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(layer4_ssm->d_state, 0, layer4_ssm->batch_size * layer4_ssm->state_dim * sizeof(float)));

    // Main simulation loop
    for (int t = 0; t < (int)(SIM_TIME / DT_PHYSICS); t++) {
        // Physics update
        if (t_physics >= DT_PHYSICS) {
            update_quad(&quad, DT_PHYSICS);
            t_physics = 0.0;
        }
        
        // Control update
        if (t_control >= DT_CONTROL) {
            // Fill input for layer1 model: IMU data (including magnetometer), position, velocity, and target
            int idx = 0;
            
            // IMU measurements (9)
            for(int i = 0; i < 3; i++) h_layer1_input[idx++] = (float)quad.gyro_measurement[i];
            for(int i = 0; i < 3; i++) h_layer1_input[idx++] = (float)quad.accel_measurement[i];
            for(int i = 0; i < 3; i++) h_layer1_input[idx++] = (float)quad.mag_measurement[i];
            
            // Position and velocity (6)
            for(int i = 0; i < 3; i++) h_layer1_input[idx++] = (float)quad.linear_position_W[i];
            for(int i = 0; i < 3; i++) h_layer1_input[idx++] = (float)quad.linear_velocity_W[i];
            
            // Target position and yaw (4)
            for(int i = 0; i < 3; i++) h_layer1_input[idx++] = (float)target[i];
            h_layer1_input[idx++] = (float)target[6];
            
            // Copy input data to GPU
            CHECK_CUDA(cudaMemcpy(d_layer1_input, h_layer1_input, 
                                 layer1_ssm->batch_size * layer1_ssm->input_dim * sizeof(float), 
                                 cudaMemcpyHostToDevice));
            
            // Forward pass through all models
            forward_pass(layer1_ssm, d_layer1_input);
            
            // Copy output from layer 1 to input for layer 2
            CHECK_CUDA(cudaMemcpy(d_layer2_input, layer1_ssm->d_predictions, 
                                 layer1_ssm->batch_size * layer1_ssm->output_dim * sizeof(float), 
                                 cudaMemcpyDeviceToDevice));
            
            // Forward pass through layer 2
            forward_pass(layer2_ssm, d_layer2_input);
            
            // Copy output from layer 2 to input for layer 3
            CHECK_CUDA(cudaMemcpy(d_layer3_input, layer2_ssm->d_predictions, 
                                 layer2_ssm->batch_size * layer2_ssm->output_dim * sizeof(float), 
                                 cudaMemcpyDeviceToDevice));
            
            // Forward pass through layer 3
            forward_pass(layer3_ssm, d_layer3_input);
            
            // Copy output from layer 3 to input for layer 4
            CHECK_CUDA(cudaMemcpy(d_layer4_input, layer3_ssm->d_predictions, 
                                 layer3_ssm->batch_size * layer3_ssm->output_dim * sizeof(float), 
                                 cudaMemcpyDeviceToDevice));
            
            // Forward pass through layer 4
            forward_pass(layer4_ssm, d_layer4_input);
            
            // Copy final predictions back to host
            CHECK_CUDA(cudaMemcpy(h_layer4_output, layer4_ssm->d_predictions, 
                                 layer4_ssm->batch_size * layer4_ssm->output_dim * sizeof(float), 
                                 cudaMemcpyDeviceToHost));
            
            // Apply predicted motor commands from layer4
            for (int i = 0; i < 4; i++) {
                quad.omega_next[i] = (double)h_layer4_output[i];
            }
            
            t_control = 0.0;
        }
        
        // Render update
        if (t_render >= DT_RENDER) {
            // Update drone position and rotation in the scene
            set_mesh_position(&scene.meshes[0], 
                (Vec3){(float)quad.linear_position_W[0], 
                       (float)quad.linear_position_W[1], 
                       (float)quad.linear_position_W[2]});
            
            set_mesh_rotation(&scene.meshes[0], 
                (Vec3){
                    atan2f(quad.R_W_B[7], quad.R_W_B[8]),
                    asinf(-quad.R_W_B[6]),
                    atan2f(quad.R_W_B[3], quad.R_W_B[0])
                }
            );
            
            // Update FPV scene
            set_mesh_position(&fpv_scene.meshes[0], 
                (Vec3){(float)quad.linear_position_W[0], 
                       (float)quad.linear_position_W[1], 
                       (float)quad.linear_position_W[2]});
            
            set_mesh_rotation(&fpv_scene.meshes[0], 
                (Vec3){
                    atan2f(quad.R_W_B[7], quad.R_W_B[8]),
                    asinf(-quad.R_W_B[6]),
                    atan2f(quad.R_W_B[3], quad.R_W_B[0])
                }
            );
            
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
                (float)quad.linear_position_W[0] + camera_offset.x,
                (float)quad.linear_position_W[1] + camera_offset.y,
                (float)quad.linear_position_W[2] + camera_offset.z
            };
            
            // Calculate look-at point (position + forward)
            Vec3 look_at = {
                fpv_pos.x + forward.x,  // Look at point is in front of drone's position
                fpv_pos.y + forward.y,
                fpv_pos.z + forward.z
            };
            
            // Set FPV camera
            set_scene_camera(&fpv_scene, fpv_pos, look_at, up, 70.0f);
            
            // Render scenes
            render_scene(&scene);
            render_scene(&fpv_scene);
            
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

    // Display final results
    printf("\nFinal position: (%.2f, %.2f, %.2f) with yaw %.2f or ±%.2f\n", 
           quad.linear_position_W[0], quad.linear_position_W[1], quad.linear_position_W[2],
           asinf(-quad.R_W_B[6]), M_PI - fabs(asinf(-quad.R_W_B[6])));
    
    // Calculate distance to target
    double dist = sqrt(pow(quad.linear_position_W[0] - target_x, 2) + 
                     pow(quad.linear_position_W[1] - target_y, 2) + 
                     pow(quad.linear_position_W[2] - target_z, 2));
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
    free(h_layer1_input);
    free(h_layer4_output);
    cudaFree(d_layer1_input);
    cudaFree(d_layer2_input);
    cudaFree(d_layer3_input);
    cudaFree(d_layer4_input);
    
    destroy_mesh(&drone);
    destroy_mesh(&ground);
    destroy_mesh(&treasure);
    destroy_scene(&scene);
    destroy_scene(&fpv_scene);
    free_ssm(layer1_ssm);
    free_ssm(layer2_ssm);
    free_ssm(layer3_ssm);
    free_ssm(layer4_ssm);
    return 0;
}