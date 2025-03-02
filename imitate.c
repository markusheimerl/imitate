#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include "sim/quad.h"
#include "ssm/gpu/ssm.h"

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define SIM_TIME    10.0  // 10 seconds per episode

// Helper function to get random value in range [min, max]
double random_range(double min, double max) {
    return min + (double)rand() / RAND_MAX * (max - min);
}

// Generate training data for the SSM
void generate_data(const char* data_file, int num_episodes) {
    FILE* f_data = fopen(data_file, "w");
    if (!f_data) {
        printf("Error opening file: %s\n", data_file);
        return;
    }
    
    // Write header: IMU measurements, position, velocity, target position+yaw, motor commands
    fprintf(f_data, "gx,gy,gz,ax,ay,az,"); // IMU measurements (6)
    fprintf(f_data, "px,py,pz,vx,vy,vz,"); // Position and velocity (6)
    fprintf(f_data, "tx,ty,tz,tyaw,"); // Target (4)
    fprintf(f_data, "m1,m2,m3,m4"); // Output motor commands (4)
    
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
        
        double t_physics = 0.0;
        double t_control = 0.0;
        
        for (int i = 0; i < (int)(SIM_TIME / DT_PHYSICS); i++) {
            if (t_physics >= DT_PHYSICS) {
                update_quad(&quad, DT_PHYSICS);
                t_physics = 0.0;
            }
            
            if (t_control >= DT_CONTROL) {
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
                
                // Write training sample: IMU, position, velocity, target, and motor commands
                fprintf(f_data, "\n%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,", // IMU
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
        }
        
        if ((episode + 1) % 1000 == 0) {
            printf("Generated %d episodes\n", episode + 1);
        }
    }
    
    fclose(f_data);
}

// Custom function to propagate gradients between the two models
void backward_between_models(SSM* upstream_model, SSM* downstream_model, float* d_upstream_input) {
    // Zero gradients for upstream model
    zero_gradients(upstream_model);
    
    // Compute gradient of loss w.r.t upstream model's output (downstream model's input)
    const float alpha = 1.0f, beta = 0.0f;
    
    // Create a temporary buffer for gradient computation
    float* d_temp_grad;
    CHECK_CUDA(cudaMalloc(&d_temp_grad, upstream_model->batch_size * upstream_model->output_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_temp_grad, 0, upstream_model->batch_size * upstream_model->output_dim * sizeof(float)));
    
    // Gradient contribution from state path: dL/dOutput = C^T * error
    CHECK_CUBLAS(cublasSgemm(downstream_model->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                          upstream_model->output_dim, upstream_model->batch_size, downstream_model->output_dim,
                          &alpha,
                          downstream_model->d_C, downstream_model->output_dim,
                          downstream_model->d_error, downstream_model->output_dim,
                          &beta,
                          d_temp_grad, upstream_model->output_dim));
    
    // Gradient contribution from direct path: dL/dOutput += D^T * error
    CHECK_CUBLAS(cublasSgemm(downstream_model->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                          upstream_model->output_dim, upstream_model->batch_size, downstream_model->output_dim,
                          &alpha,
                          downstream_model->d_D, downstream_model->output_dim,
                          downstream_model->d_error, downstream_model->output_dim,
                          &alpha, // Add to existing gradient
                          d_temp_grad, upstream_model->output_dim));
    
    // Copy the computed gradient to upstream model's error buffer
    CHECK_CUDA(cudaMemcpy(upstream_model->d_error, d_temp_grad, 
                       upstream_model->batch_size * upstream_model->output_dim * sizeof(float), 
                       cudaMemcpyDeviceToDevice));
    
    // Free temporary buffer
    cudaFree(d_temp_grad);
    
    // Now do the backward pass for the upstream model
    backward_pass(upstream_model, d_upstream_input);
}

// Train two SSM models end-to-end
void train_dual_models(const char* data_file, const char* model1_file, const char* model2_file, int num_episodes) {
    printf("Loading training data from %s...\n", data_file);
    
    // Count lines in CSV to determine number of samples
    FILE* f = fopen(data_file, "r");
    if (!f) {
        printf("Error opening file: %s\n", data_file);
        return;
    }
    
    char line[1024];
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
    
    // Parameters
    const int input_dim = 16;    // IMU (6) + position (3) + velocity (3) + target (4)
    const int mid_dim = 32;      // Intermediate dimension connecting the two models
    const int state_dim1 = 128;  // Internal state dimension for first model
    const int state_dim2 = 256;  // Internal state dimension for second model
    const int output_dim = 4;    // Motor commands (4)
    const int batch_size = num_episodes;  // Process all episodes in parallel
    
    // Allocate memory for data, reorganized by episode
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
        for (int j = 0; j < input_dim; j++) {
            if (token) {
                h_X[i * input_dim + j] = atof(token);
                token = strtok(NULL, ",");
            }
        }
        
        for (int j = 0; j < output_dim; j++) {
            if (token) {
                h_y[i * output_dim + j] = atof(token);
                token = strtok(NULL, ",");
            }
        }
    }
    fclose(f);
    
    // Reorganize data for batch processing by episodes
    float* h_X_episodes = (float*)malloc(total_samples * input_dim * sizeof(float));
    float* h_y_episodes = (float*)malloc(total_samples * output_dim * sizeof(float));
    
    // Reorder from [sample0, sample1, ...] to [episode0_sample0, episode1_sample0, ..., episode0_sample1, ...]
    for (int episode = 0; episode < num_episodes; episode++) {
        for (int step = 0; step < seq_length; step++) {
            int src_idx = episode * seq_length + step;
            int dst_idx = step * num_episodes + episode;
            
            // Check bounds to prevent out-of-range access
            if (src_idx < total_samples && dst_idx < total_samples) {
                for (int j = 0; j < input_dim; j++) {
                    h_X_episodes[dst_idx * input_dim + j] = h_X[src_idx * input_dim + j];
                }
                
                for (int j = 0; j < output_dim; j++) {
                    h_y_episodes[dst_idx * output_dim + j] = h_y[src_idx * output_dim + j];
                }
            }
        }
    }
    
    // Free original data
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
    
    // Free host episode data after transfer
    free(h_X_episodes);
    free(h_y_episodes);
    
    // Allocate memory for intermediate outputs between the two models
    float *d_mid_output;
    CHECK_CUDA(cudaMalloc(&d_mid_output, batch_size * mid_dim * sizeof(float)));
    
    // Initialize two state space models for end-to-end training
    SSM* ssm1 = init_ssm(input_dim, state_dim1, mid_dim, batch_size);
    SSM* ssm2 = init_ssm(mid_dim, state_dim2, output_dim, batch_size);
    
    // Training parameters
    const int num_epochs = 1000;
    const float learning_rate = 0.0001f;
    
    printf("Starting end-to-end SSM training for %d epochs...\n", num_epochs);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        int num_batches = 0;
        
        // Reset states at the beginning of each epoch
        CHECK_CUDA(cudaMemset(ssm1->d_state, 0, ssm1->batch_size * ssm1->state_dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(ssm2->d_state, 0, ssm2->batch_size * ssm2->state_dim * sizeof(float)));
        
        // Process full sequence of all episodes
        for (int step = 0; step < seq_length; step++) {
            // Get batch data (one time step across all episodes)
            float *d_batch_X = &d_X[step * batch_size * input_dim];
            float *d_batch_y = &d_y[step * batch_size * output_dim];
            
            // Forward pass through first model
            forward_pass(ssm1, d_batch_X);
            
            // Copy predictions from first model to intermediate buffer
            CHECK_CUDA(cudaMemcpy(d_mid_output, ssm1->d_predictions, 
                                 batch_size * mid_dim * sizeof(float), 
                                 cudaMemcpyDeviceToDevice));
            
            // Forward pass through second model
            forward_pass(ssm2, d_mid_output);
            
            // Calculate loss for the combined model output
            float loss = calculate_loss(ssm2, d_batch_y);
            epoch_loss += loss;
            num_batches++;
            
            // Backward pass through second model first
            zero_gradients(ssm2);
            backward_pass(ssm2, d_mid_output);
            
            // Propagate gradients back to the first model using our custom function
            backward_between_models(ssm1, ssm2, d_batch_X);
            
            // Update weights for both models
            update_weights(ssm1, learning_rate);
            update_weights(ssm2, learning_rate);
        }
        
        // Print progress
        if (epoch == 0 || (epoch + 1) % 10 == 0) {
            printf("Epoch [%d/%d], Average Loss: %.8f\n", 
                   epoch + 1, num_epochs, epoch_loss / num_batches);
        }
    }
    
    // Save models with batch_size=1 for inference
    ssm1->batch_size = 1;
    ssm2->batch_size = 1;
    save_ssm(ssm1, model1_file);
    save_ssm(ssm2, model2_file);
    
    // Cleanup
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_mid_output);
    free_ssm(ssm1);
    free_ssm(ssm2);
}

int main() {
    srand(time(NULL) ^ getpid());
    
    // Generate timestamped filenames
    char data_fname[64], model1_fname[64], model2_fname[64];
    time_t now = time(NULL);
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", localtime(&now));
    strftime(model1_fname, sizeof(model1_fname), "%Y%m%d_%H%M%S_model1.bin", localtime(&now));
    strftime(model2_fname, sizeof(model2_fname), "%Y%m%d_%H%M%S_model2.bin", localtime(&now));
    
    // Number of episodes for training
    int num_episodes = 10000;
    
    printf("Phase 1: Generating training data...\n");
    generate_data(data_fname, num_episodes);
    
    printf("Phase 2: Training dual SSM models end-to-end...\n");
    train_dual_models(data_fname, model1_fname, model2_fname, num_episodes);
    
    printf("Training complete!\n");
    printf("Data saved to: %s\n", data_fname);
    printf("First model saved to: %s\n", model1_fname);
    printf("Second model saved to: %s\n", model2_fname);
    
    return 0;
}