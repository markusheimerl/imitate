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

// Train three SSM models in sequence:
// - Perception model processes raw input data
// - Planning model processes perception output and produces navigation plan
// - Control model takes planning output and produces motor commands
void train_stacked_models(const char* data_file, const char* perception_file, 
                          const char* planning_file, const char* control_file, int num_episodes) {
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
    
    // Parameters for the new architecture
    const int input_dim = 16;         // All input data (IMU + position + velocity + target)
    const int perception_dim = 48;    // Output dimension for perception model
    const int planning_dim = 32;      // Output dimension for planning model
    const int output_dim = 4;         // Motor commands
    
    const int perception_state_dim = 128;  // Internal state dimension for perception model
    const int planning_state_dim = 128;    // Internal state dimension for planning model
    const int control_state_dim = 64;      // Internal state dimension for control model
    
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
        
        // Read all input data (16 values)
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
    
    // Initialize the three SSM models
    SSM* perception_ssm = init_ssm(input_dim, perception_state_dim, perception_dim, batch_size);
    SSM* planning_ssm = init_ssm(perception_dim, planning_state_dim, planning_dim, batch_size);
    SSM* control_ssm = init_ssm(planning_dim, control_state_dim, output_dim, batch_size);
    
    // Allocate memory for intermediate outputs
    float *d_perception_output, *d_planning_output;
    CHECK_CUDA(cudaMalloc(&d_perception_output, batch_size * perception_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_planning_output, batch_size * planning_dim * sizeof(float)));
    
    // Training parameters
    const int num_epochs = 1000;
    const float learning_rate = 0.0001f;
    
    printf("Starting three-stage model training for %d epochs...\n", num_epochs);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        int num_steps = 0;
        
        // Reset states at the beginning of each epoch
        CHECK_CUDA(cudaMemset(perception_ssm->d_state, 0, perception_ssm->batch_size * perception_ssm->state_dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(planning_ssm->d_state, 0, planning_ssm->batch_size * planning_ssm->state_dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(control_ssm->d_state, 0, control_ssm->batch_size * control_ssm->state_dim * sizeof(float)));
        
        // Process sequence in batches
        for (int step = 0; step < seq_length; step++) {
            // Get current batch data
            float* d_current_X = d_X + step * batch_size * input_dim;
            float* d_current_y = d_y + step * batch_size * output_dim;
            
            // Forward pass through perception model
            forward_pass(perception_ssm, d_current_X);
            
            // Copy output from perception model to input for planning model
            CHECK_CUDA(cudaMemcpy(d_perception_output, perception_ssm->d_predictions, 
                               batch_size * perception_dim * sizeof(float), 
                               cudaMemcpyDeviceToDevice));
            
            // Forward pass through planning model
            forward_pass(planning_ssm, d_perception_output);
            
            // Copy output from planning model to input for control model
            CHECK_CUDA(cudaMemcpy(d_planning_output, planning_ssm->d_predictions, 
                               batch_size * planning_dim * sizeof(float), 
                               cudaMemcpyDeviceToDevice));
            
            // Forward pass through control model
            forward_pass(control_ssm, d_planning_output);
            
            // Calculate loss
            float loss = calculate_loss(control_ssm, d_current_y);
            epoch_loss += loss;
            num_steps++;
            
            // Backward pass through control model
            zero_gradients(control_ssm);
            backward_pass(control_ssm, d_planning_output);
            
            // Backpropagate through planning model
            backward_between_models(planning_ssm, control_ssm, d_perception_output);
            
            // Backpropagate through perception model
            backward_between_models(perception_ssm, planning_ssm, d_current_X);
            
            // Update weights
            update_weights(perception_ssm, learning_rate);
            update_weights(planning_ssm, learning_rate);
            update_weights(control_ssm, learning_rate);
        }
        
        // Print progress
        if (epoch == 0 || (epoch + 1) % 10 == 0) {
            printf("Epoch [%d/%d], Average Loss: %.8f\n", epoch + 1, num_epochs, epoch_loss / num_steps);
        }
    }
    
    // Save models with batch_size=1 for inference
    perception_ssm->batch_size = 1;
    planning_ssm->batch_size = 1;
    control_ssm->batch_size = 1;
    
    save_ssm(perception_ssm, perception_file);
    save_ssm(planning_ssm, planning_file);
    save_ssm(control_ssm, control_file);
    
    // Cleanup
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_perception_output);
    cudaFree(d_planning_output);
    free_ssm(perception_ssm);
    free_ssm(planning_ssm);
    free_ssm(control_ssm);
}

int main() {
    srand(time(NULL) ^ getpid());
    
    // Generate timestamped filenames
    char data_fname[64], perception_fname[64], planning_fname[64], control_fname[64];
    time_t now = time(NULL);
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", localtime(&now));
    strftime(perception_fname, sizeof(perception_fname), "%Y%m%d_%H%M%S_perception_model.bin", localtime(&now));
    strftime(planning_fname, sizeof(planning_fname), "%Y%m%d_%H%M%S_planning_model.bin", localtime(&now));
    strftime(control_fname, sizeof(control_fname), "%Y%m%d_%H%M%S_control_model.bin", localtime(&now));
    
    // Number of episodes for training
    int num_episodes = 10000;
    
    printf("Phase 1: Generating training data...\n");
    generate_data(data_fname, num_episodes);
    
    printf("Phase 2: Training three-stage SSM model...\n");
    train_stacked_models(data_fname, perception_fname, planning_fname, control_fname, num_episodes);
    
    printf("Training complete!\n");
    printf("Data saved to: %s\n", data_fname);
    printf("Perception model saved to: %s\n", perception_fname);
    printf("Planning model saved to: %s\n", planning_fname);
    printf("Control model saved to: %s\n", control_fname);
    
    return 0;
}