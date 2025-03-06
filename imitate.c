#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include "sim/quad.h"
#include "sim/raytracer/scene.h"
#include "ssm/gpu/ssm.h"

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define DT_RENDER   (1.0 / 24.0)
#define SIM_TIME    10.0  // 10 seconds per episode

// Helper function to get random value in range [min, max]
double random_range(double min, double max) {
    return min + (double)rand() / RAND_MAX * (max - min);
}

// Helper function to reorganize data for batch processing
void reorganize_data(float* input, float* output, int num_episodes, int seq_length, int feature_dim) {
    for (int episode = 0; episode < num_episodes; episode++) {
        for (int step = 0; step < seq_length; step++) {
            int src_idx = episode * seq_length + step;
            int dst_idx = step * num_episodes + episode;
            
            // Check bounds to prevent out-of-range access
            if (src_idx < num_episodes * seq_length && dst_idx < num_episodes * seq_length) {
                for (int j = 0; j < feature_dim; j++) {
                    output[dst_idx * feature_dim + j] = input[src_idx * feature_dim + j];
                }
            }
        }
    }
}

// Function to convert raw FPV RGB image data to grayscale values
void convert_to_grayscale(unsigned char* fpv_frame, float* grayscale_pixels, int width, int height, int channels) {
    int total_pixels = width * height;
    
    if (fpv_frame == NULL) {
        // If frame is NULL, set all pixels to black (0.0)
        for (int i = 0; i < total_pixels; i++) {
            grayscale_pixels[i] = 0.0f;
        }
        return;
    }
    
    for (int i = 0; i < total_pixels; i++) {
        // Get RGB values
        float r = fpv_frame[i * channels] / 255.0f;
        float g = fpv_frame[i * channels + 1] / 255.0f;
        float b = fpv_frame[i * channels + 2] / 255.0f;
        
        // Convert to grayscale using standard luminance formula
        grayscale_pixels[i] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

// Generate training data for the SSM
void generate_data(const char* data_file, int num_episodes) {
    FILE* f_data = fopen(data_file, "w");
    if (!f_data) {
        printf("Error opening file: %s\n", data_file);
        return;
    }
    
    // Define constants for the FPV rendering
    const int fpv_width = 8;
    const int fpv_height = 4;
    const int fpv_channels = 3;
    const int fpv_pixels = fpv_width * fpv_height;
    
    // Write header: Visual grayscale pixels, IMU measurements, position, velocity, target position+yaw, motor commands
    fprintf(f_data, "pix1");
    for (int i = 2; i <= fpv_pixels; i++) {
        fprintf(f_data, ",pix%d", i);
    }
    fprintf(f_data, ",gx,gy,gz,ax,ay,az,"); // IMU measurements (6)
    fprintf(f_data, "px,py,pz,vx,vy,vz,"); // Position and velocity (6)
    fprintf(f_data, "tx,ty,tz,tyaw,"); // Target (4)
    fprintf(f_data, "m1,m2,m3,m4"); // Output motor commands (4)
    
    // Set up rendering scene for FPV
    Scene fpv_scene = create_scene(fpv_width, fpv_height, (int)(SIM_TIME * 1000), 24, 1.0f);
    
    // Set up lighting for scene
    set_scene_light(&fpv_scene,
        (Vec3){1.0f, 1.0f, -1.0f},
        (Vec3){1.4f, 1.4f, 1.4f}
    );
    
    // Create meshes
    Mesh drone = create_mesh("sim/raytracer/drone.obj", "sim/raytracer/drone.webp");
    Mesh ground = create_mesh("sim/raytracer/ground.obj", "sim/raytracer/ground.webp");
    Mesh treasure = create_mesh("sim/raytracer/treasure.obj", "sim/raytracer/treasure.webp");
    
    // Add meshes to scene
    add_mesh_to_scene(&fpv_scene, drone);
    add_mesh_to_scene(&fpv_scene, ground);
    add_mesh_to_scene(&fpv_scene, treasure);
    
    // Buffer for grayscale pixels
    float* grayscale_pixels = (float*)calloc(fpv_pixels, sizeof(float));
    
    for (int episode = 0; episode < num_episodes; episode++) {
        // Random initial state
        Quad quad = create_quad(
            random_range(-2.0, 2.0),
            random_range(0.0, 2.0),    // Always at or above ground
            random_range(-2.0, 2.0)
        );

        // Initialize state estimator
        StateEstimator estimator = {
            .R = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0},
            .angular_velocity = {0.0, 0.0, 0.0},
            .gyro_bias = {0.0, 0.0, 0.0}
        };
        
        // Random target
        double target[7] = {
            random_range(-2.0, 2.0),    // x
            random_range(1.0, 3.0),     // y: Always above ground
            random_range(-2.0, 2.0),    // z
            0.0, 0.0, 0.0,              // vx, vy, vz
            random_range(-M_PI, M_PI)   // yaw
        };
        
        // Set treasure position for target
        set_mesh_position(&fpv_scene.meshes[2], (Vec3){(float)target[0], (float)target[1], (float)target[2]});
        set_mesh_rotation(&fpv_scene.meshes[2], (Vec3){0.0f, (float)target[6], 0.0f});
        
        double t_physics = 0.0;
        double t_control = 0.0;
        double t_render = 0.0;
        
        for (int i = 0; i < (int)(SIM_TIME / DT_PHYSICS); i++) {
            if (t_physics >= DT_PHYSICS) {
                update_quad(&quad, DT_PHYSICS);
                t_physics = 0.0;
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
                
                // Render scene
                render_scene(&fpv_scene);
                
                // Advance to next frame
                next_frame(&fpv_scene);
                
                t_render = 0.0;
            }
            
            if (t_control >= DT_CONTROL) {
                // Check if we have a valid frame index
                unsigned char* frame_data = NULL;
                if (fpv_scene.current_frame > 0 && fpv_scene.current_frame <= fpv_scene.frame_count) {
                    frame_data = fpv_scene.frames[fpv_scene.current_frame - 1];
                }
                
                // Convert RGB to grayscale (handles NULL gracefully)
                convert_to_grayscale(frame_data, grayscale_pixels, fpv_width, fpv_height, fpv_channels);
                
                // Update state estimator
                update_estimator(
                    quad.gyro_measurement,
                    quad.accel_measurement,
                    DT_CONTROL,
                    &estimator
                );
                
                // Get motor commands from geometric controller
                double new_omega[4];
                control_quad_commands(
                    quad.linear_position_W,
                    quad.linear_velocity_W,
                    estimator.R,
                    estimator.angular_velocity,
                    quad.inertia,
                    target,
                    new_omega
                );
                memcpy(quad.omega_next, new_omega, 4 * sizeof(double));
                
                // Write training sample: 
                // Grayscale pixels, IMU, position, velocity, target, and motor commands
                
                // First write grayscale pixels
                fprintf(f_data, "\n%.6f", grayscale_pixels[0]);
                for (int j = 1; j < fpv_pixels; j++) {
                    fprintf(f_data, ",%.6f", grayscale_pixels[j]);
                }
                
                fprintf(f_data, ",%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,", // IMU
                       quad.gyro_measurement[0], quad.gyro_measurement[1], quad.gyro_measurement[2],
                       quad.accel_measurement[0], quad.accel_measurement[1], quad.accel_measurement[2]);
                       
                fprintf(f_data, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,", // Position and velocity
                       quad.linear_position_W[0], quad.linear_position_W[1], quad.linear_position_W[2],
                       quad.linear_velocity_W[0], quad.linear_velocity_W[1], quad.linear_velocity_W[2]);
                
                fprintf(f_data, "%.6f,%.6f,%.6f,%.6f,", // Target
                       target[0], target[1], target[2], target[6]);
                       
                fprintf(f_data, "%.6f,%.6f,%.6f,%.6f", // Motor commands
                       quad.omega_next[0],
                       quad.omega_next[1],
                       quad.omega_next[2],
                       quad.omega_next[3]);
                       
                t_control = 0.0;
            }
            
            t_physics += DT_PHYSICS;
            t_control += DT_PHYSICS;
            t_render += DT_PHYSICS;
        }
        
        if ((episode + 1) % 100 == 0) {
            printf("Generated %d episodes\n", episode + 1);
        }
    }
    
    fclose(f_data);
    free(grayscale_pixels);
    
    // Clean up raytracer resources
    destroy_mesh(&drone);
    destroy_mesh(&ground);
    destroy_mesh(&treasure);
    destroy_scene(&fpv_scene);
}

// Custom function to propagate gradients between models
void backward_between_models(SSM* first_model, SSM* second_model, float* d_first_model_input) {
    // Zero gradients for first model
    zero_gradients(first_model);
    
    // The error to propagate back is already in second_model->d_error
    // Need to convert it to gradient w.r.t first model output
    
    const float alpha = 1.0f, beta = 0.0f;
    
    // Compute gradient from state path: d_input_grad = B^T * state_error
    CHECK_CUBLAS(cublasSgemm(second_model->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                           first_model->output_dim, first_model->batch_size, second_model->state_dim,
                           &alpha,
                           second_model->d_B, second_model->state_dim,
                           second_model->d_state_error, second_model->state_dim,
                           &beta,
                           first_model->d_error, first_model->output_dim));
    
    // Add gradient from direct path: d_input_grad += D^T * error
    CHECK_CUBLAS(cublasSgemm(second_model->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                           first_model->output_dim, first_model->batch_size, second_model->output_dim,
                           &alpha,
                           second_model->d_D, second_model->output_dim,
                           second_model->d_error, second_model->output_dim,
                           &alpha, // Add to existing gradient
                           first_model->d_error, first_model->output_dim));
    
    // Now do the backward pass for the first model
    backward_pass(first_model, d_first_model_input);
}

// Train four SSM models in sequence:
// - Layer 1 processes visual + sensor input data
// - Layer 2 processes layer 1 output for further feature extraction
// - Layer 3 processes layer 2 output and produces a navigation plan
// - Layer 4 takes layer 3 output and produces motor commands
void train_stacked_models(const char* data_file, const char* layer1_file, 
                          const char* layer2_file, const char* layer3_file,
                          const char* layer4_file, int num_episodes) {
    printf("Loading training data from %s...\n", data_file);
    
    // Count lines in CSV to determine number of samples
    FILE* f = fopen(data_file, "r");
    if (!f) {
        printf("Error opening file: %s\n", data_file);
        return;
    }
    
    char line[8192];  // Increased buffer size for additional visual features
    int total_samples = 0;
    // Skip header
    fgets(line, sizeof(line), f);
    while (fgets(line, sizeof(line), f)) {
        total_samples++;
    }
    fclose(f);
    
    // Calculate steps per episode
    const int seq_length = total_samples / num_episodes;
    
    printf("Found %d total samples across %d episodes, %d steps per episode\n", 
           total_samples, num_episodes, seq_length);
    
    // Parameters for the updated architecture
    const int fpv_width = 8;
    const int fpv_height = 4;
    const int fpv_pixels = fpv_width * fpv_height;
    const int sensor_dim = 16;        // IMU + position + velocity + target
    const int input_dim = fpv_pixels + sensor_dim;  // Combined input dimension
    const int layer1_dim = 192;      // Output dimension for layer 1
    const int layer2_dim = 128;      // Output dimension for layer 2
    const int layer3_dim = 64;       // Output dimension for layer 3
    const int output_dim = 4;        // Motor commands (layer 4 output)
    
    const int layer1_state_dim = 384;  // State dimension for layer 1
    const int layer2_state_dim = 256;  // State dimension for layer 2
    const int layer3_state_dim = 128;  // State dimension for layer 3
    const int layer4_state_dim = 64;   // State dimension for layer 4
    
    const int batch_size = num_episodes;   // Process all episodes in parallel
    
    // Allocate memory for data
    float* h_X = (float*)malloc(total_samples * input_dim * sizeof(float));
    float* h_y = (float*)malloc(total_samples * output_dim * sizeof(float));
    
    // Load data from CSV
    f = fopen(data_file, "r");
    if (!f) {
        printf("Error reopening file: %s\n", data_file);
        free(h_X);
        free(h_y);
        return;
    }
    
    // Skip header
    fgets(line, sizeof(line), f);
    
    for (int i = 0; i < total_samples; i++) {
        if (!fgets(line, sizeof(line), f)) break;
        
        char* token = strtok(line, ",");
        
        // Read all input data (raw pixels + sensor data)
        for (int j = 0; j < input_dim; j++) {
            if (token) {
                h_X[i * input_dim + j] = atof(token);
                token = strtok(NULL, ",");
            }
        }
        
        // Read motor commands (4 values)
        for (int j = 0; j < output_dim; j++) {
            if (token) {
                h_y[i * output_dim + j] = atof(token);
                token = strtok(NULL, ",");
            }
        }
    }
    fclose(f);
    
    // Reorganize data for batch processing
    float* h_X_episodes = (float*)malloc(total_samples * input_dim * sizeof(float));
    float* h_y_episodes = (float*)malloc(total_samples * output_dim * sizeof(float));
    
    reorganize_data(h_X, h_X_episodes, num_episodes, seq_length, input_dim);
    reorganize_data(h_y, h_y_episodes, num_episodes, seq_length, output_dim);
    
    free(h_X);
    free(h_y);
    
    // Transfer reorganized data to GPU
    float *d_X, *d_y;
    CHECK_CUDA(cudaMalloc(&d_X, total_samples * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, total_samples * output_dim * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(d_X, h_X_episodes, total_samples * input_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y_episodes, total_samples * output_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    free(h_X_episodes);
    free(h_y_episodes);
    
    // Initialize the four SSM models
    SSM* layer1_ssm = init_ssm(input_dim, layer1_state_dim, layer1_dim, batch_size);
    SSM* layer2_ssm = init_ssm(layer1_dim, layer2_state_dim, layer2_dim, batch_size);
    SSM* layer3_ssm = init_ssm(layer2_dim, layer3_state_dim, layer3_dim, batch_size);
    SSM* layer4_ssm = init_ssm(layer3_dim, layer4_state_dim, output_dim, batch_size);
    
    // Allocate memory for intermediate outputs
    float *d_layer1_output, *d_layer2_output, *d_layer3_output;
    CHECK_CUDA(cudaMalloc(&d_layer1_output, batch_size * layer1_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_layer2_output, batch_size * layer2_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_layer3_output, batch_size * layer3_dim * sizeof(float)));
    
    // Training parameters
    const int num_epochs = 2000;
    const float learning_rate = 0.00001f;
    
    printf("Starting four-stage model training for %d epochs...\n", num_epochs);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        int num_steps = 0;
        
        // Reset states at the beginning of each epoch
        CHECK_CUDA(cudaMemset(layer1_ssm->d_state, 0, layer1_ssm->batch_size * layer1_ssm->state_dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(layer2_ssm->d_state, 0, layer2_ssm->batch_size * layer2_ssm->state_dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(layer3_ssm->d_state, 0, layer3_ssm->batch_size * layer3_ssm->state_dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(layer4_ssm->d_state, 0, layer4_ssm->batch_size * layer4_ssm->state_dim * sizeof(float)));
        
        // Process sequence in batches
        for (int step = 0; step < seq_length; step++) {
            // Get current batch data
            float* d_current_X = d_X + step * batch_size * input_dim;
            float* d_current_y = d_y + step * batch_size * output_dim;
            
            // Forward pass through layer 1
            forward_pass(layer1_ssm, d_current_X);
            
            // Copy output from layer 1 to input for layer 2
            CHECK_CUDA(cudaMemcpy(d_layer1_output, layer1_ssm->d_predictions, 
                               batch_size * layer1_dim * sizeof(float), 
                               cudaMemcpyDeviceToDevice));
            
            // Forward pass through layer 2
            forward_pass(layer2_ssm, d_layer1_output);
            
            // Copy output from layer 2 to input for layer 3
            CHECK_CUDA(cudaMemcpy(d_layer2_output, layer2_ssm->d_predictions, 
                               batch_size * layer2_dim * sizeof(float), 
                               cudaMemcpyDeviceToDevice));
            
            // Forward pass through layer 3
            forward_pass(layer3_ssm, d_layer2_output);
            
            // Copy output from layer 3 to input for layer 4
            CHECK_CUDA(cudaMemcpy(d_layer3_output, layer3_ssm->d_predictions, 
                               batch_size * layer3_dim * sizeof(float), 
                               cudaMemcpyDeviceToDevice));
            
            // Forward pass through layer 4
            forward_pass(layer4_ssm, d_layer3_output);
            
            // Calculate loss
            float loss = calculate_loss(layer4_ssm, d_current_y);
            epoch_loss += loss;
            num_steps++;
            
            // Backward pass through layer 4
            zero_gradients(layer4_ssm);
            backward_pass(layer4_ssm, d_layer3_output);
            
            // Backpropagate through layer 3
            backward_between_models(layer3_ssm, layer4_ssm, d_layer2_output);
            
            // Backpropagate through layer 2
            backward_between_models(layer2_ssm, layer3_ssm, d_layer1_output);
            
            // Backpropagate through layer 1
            backward_between_models(layer1_ssm, layer2_ssm, d_current_X);
            
            // Update weights
            update_weights(layer1_ssm, learning_rate);
            update_weights(layer2_ssm, learning_rate);
            update_weights(layer3_ssm, learning_rate);
            update_weights(layer4_ssm, learning_rate);
        }
        
        // Print progress
        if (epoch == 0 || (epoch + 1) % 10 == 0) {
            printf("Epoch [%d/%d], Average Loss: %.8f\n", epoch + 1, num_epochs, epoch_loss / num_steps);
        }
    }
    
    save_ssm(layer1_ssm, layer1_file);
    save_ssm(layer2_ssm, layer2_file);
    save_ssm(layer3_ssm, layer3_file);
    save_ssm(layer4_ssm, layer4_file);
    
    // Cleanup
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_layer1_output);
    cudaFree(d_layer2_output);
    cudaFree(d_layer3_output);
    free_ssm(layer1_ssm);
    free_ssm(layer2_ssm);
    free_ssm(layer3_ssm);
    free_ssm(layer4_ssm);
}

int main() {
    srand(time(NULL) ^ getpid());
    
    // Generate timestamped filenames
    char data_fname[64], layer1_fname[64], layer2_fname[64], layer3_fname[64], layer4_fname[64];
    time_t now = time(NULL);
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", localtime(&now));
    strftime(layer1_fname, sizeof(layer1_fname), "%Y%m%d_%H%M%S_layer1_model.bin", localtime(&now));
    strftime(layer2_fname, sizeof(layer2_fname), "%Y%m%d_%H%M%S_layer2_model.bin", localtime(&now));
    strftime(layer3_fname, sizeof(layer3_fname), "%Y%m%d_%H%M%S_layer3_model.bin", localtime(&now));
    strftime(layer4_fname, sizeof(layer4_fname), "%Y%m%d_%H%M%S_layer4_model.bin", localtime(&now));
    
    // Number of episodes for training
    int num_episodes = 2000;
    
    printf("Phase 1: Generating training data with FPV rendering...\n");
    generate_data(data_fname, num_episodes);
    
    printf("Phase 2: Training four-stage SSM model with raw pixel input...\n");
    train_stacked_models(data_fname, layer1_fname, layer2_fname, layer3_fname, layer4_fname, num_episodes);
    
    printf("Training complete!\n");
    printf("Data saved to: %s\n", data_fname);
    printf("Layer 1 model saved to: %s\n", layer1_fname);
    printf("Layer 2 model saved to: %s\n", layer2_fname);
    printf("Layer 3 model saved to: %s\n", layer3_fname);
    printf("Layer 4 model saved to: %s\n", layer4_fname);
    
    return 0;
}