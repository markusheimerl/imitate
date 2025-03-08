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
        // Initialize drone with random position and orientation
        double drone_x = random_range(-2.0, 2.0);
        double drone_y = random_range(0.5, 2.0);
        double drone_z = random_range(-2.0, 2.0);
        double drone_yaw = 0.0; // random_range(-M_PI, M_PI);
        
        // Create quad with random position and orientation
        Quad quad = create_quad(drone_x, drone_y, drone_z, drone_yaw);
        
        // Place target completely randomly
        double target_x = random_range(-2.0, 2.0);
        double target_y = random_range(0.5, 2.5);
        double target_z = random_range(-2.0, 2.0);
        double target_yaw = 0.0; // random_range(-M_PI, M_PI);
        
        // Create target array (position, velocity, and desired yaw)
        double target[7] = {
            target_x, target_y, target_z,    // Target position
            0.0, 0.0, 0.0,                   // Zero velocity target
            target_yaw                       // Random target yaw
        };

        // Initialize state estimator
        StateEstimator estimator = {
            .angular_velocity = {0.0, 0.0, 0.0},
            .gyro_bias = {0.0, 0.0, 0.0}
        };
        memcpy(estimator.R, quad.R_W_B, 9 * sizeof(double));
        
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
void backward_between_models(SSM* prev_model, SSM* next_model, float* d_prev_model_input) {
    // Zero gradients for previous model
    zero_gradients(prev_model);
    
    // The error to propagate back is already in next_model->d_error
    // Need to convert it to gradient w.r.t previous model output
    
    const float alpha = 1.0f, beta = 0.0f;
    
    // Compute gradient from state path: d_input_grad = B^T * state_error
    CHECK_CUBLAS(cublasSgemm(next_model->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                           prev_model->output_dim, prev_model->batch_size, next_model->state_dim,
                           &alpha,
                           next_model->d_B, next_model->state_dim,
                           next_model->d_state_error, next_model->state_dim,
                           &beta,
                           prev_model->d_error, prev_model->output_dim));
    
    // Add gradient from direct path: d_input_grad += D^T * error
    CHECK_CUBLAS(cublasSgemm(next_model->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                           prev_model->output_dim, prev_model->batch_size, next_model->output_dim,
                           &alpha,
                           next_model->d_D, next_model->output_dim,
                           next_model->d_error, next_model->output_dim,
                           &alpha, // Add to existing gradient
                           prev_model->d_error, prev_model->output_dim));
    
    // Now do the backward pass for the previous model
    backward_pass(prev_model, d_prev_model_input);
}

// Train four SSM models in a stack
void train_stacked_models(const char* data_file, const char* model1_file, const char* model2_file, 
                         const char* model3_file, const char* model4_file, int num_episodes) {
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
    const int input_dim = 16;       // All input data (IMU + position + velocity + target)
    const int hidden_dim1 = 96;     // Output dimension for first model
    const int hidden_dim2 = 64;     // Output dimension for second model 
    const int hidden_dim3 = 32;     // Output dimension for third model
    const int output_dim = 4;       // Motor commands
    
    const int state_dim1 = 128;     // Internal state dimension for first model
    const int state_dim2 = 128;     // Internal state dimension for second model
    const int state_dim3 = 128;     // Internal state dimension for third model
    const int state_dim4 = 128;     // Internal state dimension for fourth model
    
    const int batch_size = num_episodes;  // Process all episodes in parallel
    
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
    
    // Initialize the SSM models
    SSM* layer1_ssm = init_ssm(input_dim, state_dim1, hidden_dim1, batch_size);
    SSM* layer2_ssm = init_ssm(hidden_dim1, state_dim2, hidden_dim2, batch_size);
    SSM* layer3_ssm = init_ssm(hidden_dim2, state_dim3, hidden_dim3, batch_size);
    SSM* layer4_ssm = init_ssm(hidden_dim3, state_dim4, output_dim, batch_size);
    
    // Allocate memory for intermediate outputs
    float *d_hidden_output1, *d_hidden_output2, *d_hidden_output3;
    CHECK_CUDA(cudaMalloc(&d_hidden_output1, batch_size * hidden_dim1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_hidden_output2, batch_size * hidden_dim2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_hidden_output3, batch_size * hidden_dim3 * sizeof(float)));
    
    // Training parameters
    const int num_epochs = 200;
    const float learning_rate = 0.0001f;
    
    printf("Starting stacked model training for %d epochs...\n", num_epochs);
    
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
            CHECK_CUDA(cudaMemcpy(d_hidden_output1, layer1_ssm->d_predictions, 
                               batch_size * hidden_dim1 * sizeof(float), 
                               cudaMemcpyDeviceToDevice));
            
            // Forward pass through layer 2
            forward_pass(layer2_ssm, d_hidden_output1);
            
            // Copy output from layer 2 to input for layer 3
            CHECK_CUDA(cudaMemcpy(d_hidden_output2, layer2_ssm->d_predictions, 
                               batch_size * hidden_dim2 * sizeof(float), 
                               cudaMemcpyDeviceToDevice));
            
            // Forward pass through layer 3
            forward_pass(layer3_ssm, d_hidden_output2);
            
            // Copy output from layer 3 to input for layer 4
            CHECK_CUDA(cudaMemcpy(d_hidden_output3, layer3_ssm->d_predictions, 
                               batch_size * hidden_dim3 * sizeof(float), 
                               cudaMemcpyDeviceToDevice));
            
            // Forward pass through layer 4
            forward_pass(layer4_ssm, d_hidden_output3);
            
            // Calculate loss
            float loss = calculate_loss(layer4_ssm, d_current_y);
            epoch_loss += loss;
            num_steps++;
            
            // Backward pass through all layers
            zero_gradients(layer4_ssm);
            backward_pass(layer4_ssm, d_hidden_output3);
            
            backward_between_models(layer3_ssm, layer4_ssm, d_hidden_output2);
            backward_between_models(layer2_ssm, layer3_ssm, d_hidden_output1);
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

    save_ssm(layer1_ssm, model1_file);
    save_ssm(layer2_ssm, model2_file);
    save_ssm(layer3_ssm, model3_file);
    save_ssm(layer4_ssm, model4_file);
    
    // Cleanup
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_hidden_output1);
    cudaFree(d_hidden_output2);
    cudaFree(d_hidden_output3);
    free_ssm(layer1_ssm);
    free_ssm(layer2_ssm);
    free_ssm(layer3_ssm);
    free_ssm(layer4_ssm);
}

int main() {
    srand(time(NULL) ^ getpid());
    
    // Generate timestamped filenames
    char data_fname[64], model1_fname[64], model2_fname[64], model3_fname[64], model4_fname[64];
    time_t now = time(NULL);
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", localtime(&now));
    strftime(model1_fname, sizeof(model1_fname), "%Y%m%d_%H%M%S_layer1_model.bin", localtime(&now));
    strftime(model2_fname, sizeof(model2_fname), "%Y%m%d_%H%M%S_layer2_model.bin", localtime(&now));
    strftime(model3_fname, sizeof(model3_fname), "%Y%m%d_%H%M%S_layer3_model.bin", localtime(&now));
    strftime(model4_fname, sizeof(model4_fname), "%Y%m%d_%H%M%S_layer4_model.bin", localtime(&now));
    
    // Number of episodes for training
    int num_episodes = 10000;
    
    printf("Phase 1: Generating training data...\n");
    generate_data(data_fname, num_episodes);
    
    printf("Phase 2: Training stacked SSM models...\n");
    train_stacked_models(data_fname, model1_fname, model2_fname, model3_fname, model4_fname, num_episodes);
    
    printf("Training complete!\n");
    printf("Data saved to: %s\n", data_fname);
    printf("Layer 1 model saved to: %s\n", model1_fname);
    printf("Layer 2 model saved to: %s\n", model2_fname);
    printf("Layer 3 model saved to: %s\n", model3_fname);
    printf("Layer 4 model saved to: %s\n", model4_fname);
    
    return 0;
}