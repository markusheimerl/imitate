#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "sim/quad.h"
#include "mlp/gpu/mlp.h"
#include "mlp/data.h"
#include "ssm/gpu/ssm.h"

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define SIM_TIME    10.0  // 10 seconds per episode

// Helper function to get random value in range [min, max]
double random_range(double min, double max) {
    return min + (double)rand() / RAND_MAX * (max - min);
}

// Generate training data for both policy network and state estimator
void generate_training_data(const char* policy_file, const char* estimator_file, int num_episodes) {
    FILE* f_policy = fopen(policy_file, "w");
    if (!f_policy) {
        printf("Error opening file: %s\n", policy_file);
        return;
    }
    
    FILE* f_estimator = fopen(estimator_file, "w");
    if (!f_estimator) {
        printf("Error opening file: %s\n", estimator_file);
        fclose(f_policy);
        return;
    }
    
    // Write policy data header
    fprintf(f_policy, "px,py,pz,vx,vy,vz,"); // Position and velocity (6)
    fprintf(f_policy, "r11,r12,r13,r21,r22,r23,r31,r32,r33,"); // Rotation matrix (9)
    fprintf(f_policy, "wx,wy,wz,"); // Angular velocity (3)
    fprintf(f_policy, "tx,ty,tz,tyaw,"); // Target (7)
    fprintf(f_policy, "m1,m2,m3,m4"); // Actions (4)
    
    // Write estimator data header
    fprintf(f_estimator, "gx,gy,gz,ax,ay,az,"); // IMU measurements (6)
    fprintf(f_estimator, "r11,r12,r13,r21,r22,r23,r31,r32,r33,"); // Rotation matrix (9) 
    fprintf(f_estimator, "wx,wy,wz"); // Angular velocity (3)
    
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
            random_range(0.0, 2*M_PI)   // yaw
        };
        
        double t_physics = 0.0;
        double t_control = 0.0;
        
        for (int i = 0; i < (int)(SIM_TIME / DT_PHYSICS); i++) {
            if (t_physics >= DT_PHYSICS) {
                update_quad(&quad, DT_PHYSICS);
                t_physics = 0.0;
            }
            
            if (t_control >= DT_CONTROL) {
                // Input: gyro and accel measurements
                fprintf(f_estimator, "\n%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,",
                       quad.gyro_measurement[0], quad.gyro_measurement[1], quad.gyro_measurement[2],
                       quad.accel_measurement[0], quad.accel_measurement[1], quad.accel_measurement[2]);
                
                // Update state estimator
                update_estimator(
                    quad.gyro_measurement,
                    quad.accel_measurement,
                    DT_CONTROL,
                    &estimator
                );
                
                // Output: rotation matrix and angular velocity
                fprintf(f_estimator, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,",
                       estimator.R[0], estimator.R[1], estimator.R[2],
                       estimator.R[3], estimator.R[4], estimator.R[5],
                       estimator.R[6], estimator.R[7], estimator.R[8]);
                fprintf(f_estimator, "%.6f,%.6f,%.6f",
                       estimator.angular_velocity[0],
                       estimator.angular_velocity[1],
                       estimator.angular_velocity[2]);

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
                
                // Write state, target, and action to file for policy network
                fprintf(f_policy, "\n%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,", // Position and velocity
                       quad.linear_position_W[0], quad.linear_position_W[1], quad.linear_position_W[2],
                       quad.linear_velocity_W[0], quad.linear_velocity_W[1], quad.linear_velocity_W[2]);
                       
                fprintf(f_policy, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,", // Estimated rotation matrix
                       estimator.R[0], estimator.R[1], estimator.R[2],
                       estimator.R[3], estimator.R[4], estimator.R[5],
                       estimator.R[6], estimator.R[7], estimator.R[8]);
                       
                fprintf(f_policy, "%.6f,%.6f,%.6f,", // Estimated angular velocity
                       estimator.angular_velocity[0],
                       estimator.angular_velocity[1],
                       estimator.angular_velocity[2]);
                       
                fprintf(f_policy, "%.6f,%.6f,%.6f,%.6f,", // Target
                       target[0], target[1], target[2], target[6]);
                       
                fprintf(f_policy, "%.6f,%.6f,%.6f,%.6f", // Motor commands
                       quad.omega_next[0],
                       quad.omega_next[1],
                       quad.omega_next[2],
                       quad.omega_next[3]);
                       
                t_control = 0.0;
            }
            
            t_physics += DT_PHYSICS;
            t_control += DT_PHYSICS;
        }
        
        if ((episode + 1) % 10 == 0) {
            printf("Generated %d episodes\n", episode + 1);
        }
    }
    
    fclose(f_policy);
    fclose(f_estimator);
}

// Train MLP policy
void train_policy(const char* data_file, const char* model_file) {
    printf("Loading policy training data from %s...\n", data_file);
    
    float *X, *y;
    int num_samples;
    load_csv(data_file, &X, &y, &num_samples, 22, 4);
    
    printf("Policy training data loaded: %d samples\n", num_samples);
    
    // Initialize MLP
    const int input_dim = 22;   // 18 state + 4 target
    const int hidden_dim = 512;
    const int output_dim = 4;   // 4 motor commands
    const int batch_size = num_samples;
    
    Net* net = init_net(input_dim, hidden_dim, output_dim, batch_size);
    
    // Training parameters
    const int num_epochs = 25000;
    const float learning_rate = 0.001f;
    
    printf("Starting policy training for %d epochs...\n", num_epochs);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        forward_pass(net, X);
        float loss = calculate_loss(net, y);
        zero_gradients(net);
        backward_pass(net, X);
        update_weights(net, learning_rate);
        
        if (epoch == 0 || (epoch + 1) % 100 == 0) {
            printf("Policy Epoch [%d/%d], Loss: %.8f\n", 
                   epoch + 1, num_epochs, loss);
        }
    }
    
    // Save trained model
    net->batch_size = 1;
    save_model(net, model_file);
    
    // Cleanup
    free(X);
    free(y);
    free_net(net);
}

// Train SSM state estimator
void train_estimator(const char* data_file, const char* model_file) {
    printf("Loading estimator training data from %s...\n", data_file);
    
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
    
    // We know we generated 500 episodes, so calculate the steps per episode
    const int num_episodes = 500;
    const int seq_length = total_samples / num_episodes;
    
    printf("Found %d total samples across %d episodes, %d steps per episode\n", 
           total_samples, num_episodes, seq_length);
    
    // Parameters
    const int input_dim = 6;    // gyro (3) + accel (3)
    const int state_dim = 32;  // Internal state dimension
    const int output_dim = 12;  // rotation matrix (9) + angular velocity (3)
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
    
    // Initialize state space model with the correct batch size
    SSM* ssm = init_ssm(input_dim, state_dim, output_dim, batch_size);
    
    // Training parameters
    const int num_epochs = 2000;
    const float learning_rate = 0.00001f;
    
    printf("Starting estimator training for %d epochs...\n", num_epochs);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        int num_batches = 0;
        
        // Reset state at the beginning of each epoch
        CHECK_CUDA(cudaMemset(ssm->d_state, 0, 
                             ssm->batch_size * ssm->state_dim * sizeof(float)));
        
        // Process full sequence of all episodes
        for (int step = 0; step < seq_length; step++) {
            // Get batch data (one time step across all episodes)
            float *d_batch_X = &d_X[step * batch_size * input_dim];
            float *d_batch_y = &d_y[step * batch_size * output_dim];
            
            // Forward pass
            forward_pass(ssm, d_batch_X);
            
            // Calculate loss
            float loss = calculate_loss(ssm, d_batch_y);
            epoch_loss += loss;
            num_batches++;
            
            // Backward pass
            zero_gradients(ssm);
            backward_pass(ssm, d_batch_X);
            
            // Update weights
            update_weights(ssm, learning_rate);
        }
        
        // Print progress
        if (epoch == 0 || (epoch + 1) % 10 == 0) {
            printf("Estimator Epoch [%d/%d], Average Loss: %.8f\n", 
                   epoch + 1, num_epochs, epoch_loss / num_batches);
        }
    }
    
    // Save model
    ssm->batch_size = 1;
    save_ssm(ssm, model_file);
    
    // Cleanup
    cudaFree(d_X);
    cudaFree(d_y);
    free_ssm(ssm);
}

int main() {
    srand(time(NULL) ^ getpid());
    
    // Generate timestamped filenames
    char policy_data_fname[64], policy_model_fname[64];
    char estimator_data_fname[64], estimator_model_fname[64];
    time_t now = time(NULL);
    strftime(policy_data_fname, sizeof(policy_data_fname), "%Y%m%d_%H%M%S_policy_data.csv", localtime(&now));
    strftime(policy_model_fname, sizeof(policy_model_fname), "%Y%m%d_%H%M%S_policy.bin", localtime(&now));
    strftime(estimator_data_fname, sizeof(estimator_data_fname), "%Y%m%d_%H%M%S_estimator_data.csv", localtime(&now));
    strftime(estimator_model_fname, sizeof(estimator_model_fname), "%Y%m%d_%H%M%S_estimator.bin", localtime(&now));
    
    printf("Phase 1: Generating training data...\n");
    generate_training_data(policy_data_fname, estimator_data_fname, 500);
    
    printf("Phase 2: Training policy network...\n");
    train_policy(policy_data_fname, policy_model_fname);
    
    printf("Phase 3: Training state estimator...\n");
    train_estimator(estimator_data_fname, estimator_model_fname);
    
    printf("Training complete!\n");
    printf("Policy data saved to: %s\n", policy_data_fname);
    printf("Policy model saved to: %s\n", policy_model_fname);
    printf("Estimator data saved to: %s\n", estimator_data_fname);
    printf("Estimator model saved to: %s\n", estimator_model_fname);
    
    return 0;
}