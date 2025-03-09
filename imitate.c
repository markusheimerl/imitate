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

// Train the SSM
void train_model(const char* data_file, const char* model_file, int num_episodes) {
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
    const int state_dim = 512;   // Internal state dimension
    const int output_dim = 4;    // Motor commands (4)
    
    // Split data into training and validation sets (80% train, 20% validation)
    const int train_episodes = num_episodes * 0.8;
    const int val_episodes = num_episodes - train_episodes;
    const int train_batch_size = train_episodes;
    const int val_batch_size = val_episodes;
    
    printf("Split: %d training episodes, %d validation episodes\n", 
           train_episodes, val_episodes);
    
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
    
    // Allocate host memory for train and validation data
    float* h_X_train = (float*)malloc(train_episodes * seq_length * input_dim * sizeof(float));
    float* h_y_train = (float*)malloc(train_episodes * seq_length * output_dim * sizeof(float));
    float* h_X_val = (float*)malloc(val_episodes * seq_length * input_dim * sizeof(float));
    float* h_y_val = (float*)malloc(val_episodes * seq_length * output_dim * sizeof(float));
    
    // Reorganize training data for batch processing by episodes
    for (int episode = 0; episode < train_episodes; episode++) {
        for (int step = 0; step < seq_length; step++) {
            int src_idx = episode * seq_length + step;
            int dst_idx = step * train_episodes + episode;
            
            // Check bounds to prevent out-of-range access
            if (src_idx < total_samples && dst_idx < train_episodes * seq_length) {
                for (int j = 0; j < input_dim; j++) {
                    h_X_train[dst_idx * input_dim + j] = h_X[src_idx * input_dim + j];
                }
                
                for (int j = 0; j < output_dim; j++) {
                    h_y_train[dst_idx * output_dim + j] = h_y[src_idx * output_dim + j];
                }
            }
        }
    }
    
    // Reorganize validation data for batch processing by episodes
    for (int episode = 0; episode < val_episodes; episode++) {
        for (int step = 0; step < seq_length; step++) {
            int src_idx = (episode + train_episodes) * seq_length + step;
            int dst_idx = step * val_episodes + episode;
            
            // Check bounds to prevent out-of-range access
            if (src_idx < total_samples && dst_idx < val_episodes * seq_length) {
                for (int j = 0; j < input_dim; j++) {
                    h_X_val[dst_idx * input_dim + j] = h_X[src_idx * input_dim + j];
                }
                
                for (int j = 0; j < output_dim; j++) {
                    h_y_val[dst_idx * output_dim + j] = h_y[src_idx * output_dim + j];
                }
            }
        }
    }
    
    // Free original data
    free(h_X);
    free(h_y);
    
    // Transfer training data to GPU
    float *d_X_train, *d_y_train;
    CHECK_CUDA(cudaMalloc(&d_X_train, train_episodes * seq_length * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y_train, train_episodes * seq_length * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_X_train, h_X_train, train_episodes * seq_length * input_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y_train, h_y_train, train_episodes * seq_length * output_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    // Transfer validation data to GPU
    float *d_X_val, *d_y_val;
    CHECK_CUDA(cudaMalloc(&d_X_val, val_episodes * seq_length * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y_val, val_episodes * seq_length * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_X_val, h_X_val, val_episodes * seq_length * input_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y_val, h_y_val, val_episodes * seq_length * output_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    // Free host data after transfer
    free(h_X_train);
    free(h_y_train);
    free(h_X_val);
    free(h_y_val);
    
    // Initialize state space models for training and validation
    SSM* train_ssm = init_ssm(input_dim, state_dim, output_dim, train_batch_size);
    SSM* val_ssm = init_ssm(input_dim, state_dim, output_dim, val_batch_size);
    
    // Training parameters
    const int num_epochs = 1000;
    const float learning_rate = 0.0001f;
    
    printf("Starting SSM training for %d epochs...\n", num_epochs);
    
    // For tracking best model
    float best_val_loss = INFINITY;
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float train_loss = 0.0f;
        int train_batches = 0;
        
        // Reset state at the beginning of each epoch
        CHECK_CUDA(cudaMemset(train_ssm->d_state, 0, 
                             train_ssm->batch_size * train_ssm->state_dim * sizeof(float)));
        
        // Training pass
        for (int step = 0; step < seq_length; step++) {
            // Get batch data (one time step across all training episodes)
            float *d_batch_X = &d_X_train[step * train_batch_size * input_dim];
            float *d_batch_y = &d_y_train[step * train_batch_size * output_dim];
            
            // Forward pass
            forward_pass(train_ssm, d_batch_X);
            
            // Calculate loss
            float loss = calculate_loss(train_ssm, d_batch_y);
            train_loss += loss;
            train_batches++;
            
            // Backward pass
            zero_gradients(train_ssm);
            backward_pass(train_ssm, d_batch_X);
            
            // Update weights
            update_weights(train_ssm, learning_rate);
        }
        
        // Calculate average training loss
        float avg_train_loss = train_loss / train_batches;
        
        // Copy updated weights to validation model
        CHECK_CUDA(cudaMemcpy(val_ssm->d_A, train_ssm->d_A, 
                             state_dim * state_dim * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(val_ssm->d_B, train_ssm->d_B, 
                             state_dim * input_dim * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(val_ssm->d_C, train_ssm->d_C, 
                             output_dim * state_dim * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(val_ssm->d_D, train_ssm->d_D, 
                             output_dim * input_dim * sizeof(float), cudaMemcpyDeviceToDevice));
        
        // Validation pass (no weight updates)
        float val_loss = 0.0f;
        int val_batches = 0;
        
        // Reset validation state
        CHECK_CUDA(cudaMemset(val_ssm->d_state, 0, 
                             val_ssm->batch_size * val_ssm->state_dim * sizeof(float)));
        
        for (int step = 0; step < seq_length; step++) {
            // Get validation batch data
            float *d_batch_X = &d_X_val[step * val_batch_size * input_dim];
            float *d_batch_y = &d_y_val[step * val_batch_size * output_dim];
            
            // Forward pass only (no backprop or weight updates)
            forward_pass(val_ssm, d_batch_X);
            
            // Calculate validation loss
            float loss = calculate_loss(val_ssm, d_batch_y);
            val_loss += loss;
            val_batches++;
        }
        
        // Calculate average validation loss
        float avg_val_loss = val_loss / val_batches;
        
        // Check if this is the best model so far
        if (avg_val_loss < best_val_loss) {
            best_val_loss = avg_val_loss;
            
            // Save the best model
            if (epoch > 0 && (epoch + 1) % 100 == 0) {
                char best_model_fname[128];
                sprintf(best_model_fname, "%s.best", model_file);
                save_ssm(train_ssm, best_model_fname);
                printf("Saved best model with validation loss: %.8f\n", best_val_loss);
            }
        }
        
        // Print progress
        if (epoch == 0 || (epoch + 1) % 10 == 0) {
            printf("Epoch [%d/%d], Train Loss: %.8f, Val Loss: %.8f\n", 
                   epoch + 1, num_epochs, avg_train_loss, avg_val_loss);
        }
    }
    
    // Save final model
    save_ssm(train_ssm, model_file);
    
    printf("Training complete. Best validation loss: %.8f\n", best_val_loss);
    
    // Cleanup
    cudaFree(d_X_train);
    cudaFree(d_y_train);
    cudaFree(d_X_val);
    cudaFree(d_y_val);
    free_ssm(train_ssm);
    free_ssm(val_ssm);
}

int main(int argc, char* argv[]) {
    if(argc != 3) {
        printf("Usage: %s <num_episodes> <data_file>\n", argv[0]);
        return 1;
    }
    srand(time(NULL) ^ getpid());
    
    // Generate timestamped filenames
    char model_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_model.bin", localtime(&now));
    
    // Train SSM model
    printf("Phase 2: Training SSM...\n");
    train_model(argv[2], model_fname, atoi(argv[1]));
    
    return 0;
}