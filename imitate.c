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

// Custom function to propagate gradients between models
void backward_between_models(SSM* prev_model, SSM* next_model) {
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
}

int main(int argc, char* argv[]) {
    if(argc != 3) {
        printf("Usage: %s <num_episodes> <data_file>\n", argv[0]);
        return 1;
    }
    
    srand(time(NULL) ^ getpid());
    
    // Generate timestamped filenames
    char model_prefix[64];
    time_t now = time(NULL);
    strftime(model_prefix, sizeof(model_prefix), "%Y%m%d_%H%M%S_model", localtime(&now));
    
    const char* data_file = argv[2];
    int num_episodes = atoi(argv[1]);
    
    printf("Loading training data from %s...\n", data_file);
    
    // Count lines in CSV to determine number of samples
    FILE* f = fopen(data_file, "r");
    if (!f) {
        printf("Error opening file: %s\n", data_file);
        return 1;
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
    
    // Parameters for both layers
    const int input_dim = 16;      // IMU (6) + position (3) + velocity (3) + target (4)
    const int hidden_dim = 16;     // Hidden layer output dimension
    const int state_dim1 = 256;    // First layer state dimension
    const int state_dim2 = 256;    // Second layer state dimension
    const int output_dim = 4;      // Motor commands (4)
    const int batch_size = num_episodes;
    
    // Allocate memory for data, reorganized by episode
    float* h_X = (float*)malloc(total_samples * input_dim * sizeof(float));
    float* h_y = (float*)malloc(total_samples * output_dim * sizeof(float));
    
    // Load data from CSV
    f = fopen(data_file, "r");
    if (!f) {
        printf("Error reopening file: %s\n", data_file);
        free(h_X);
        free(h_y);
        return 1;
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
    
    // Allocate host memory for batch data
    float* h_X_batch = (float*)malloc(batch_size * seq_length * input_dim * sizeof(float));
    float* h_y_batch = (float*)malloc(batch_size * seq_length * output_dim * sizeof(float));
    
    // Reorganize data for batch processing by episodes
    for (int episode = 0; episode < batch_size; episode++) {
        for (int step = 0; step < seq_length; step++) {
            int src_idx = episode * seq_length + step;
            int dst_idx = step * batch_size + episode;
            
            // Check bounds to prevent out-of-range access
            if (src_idx < total_samples && dst_idx < batch_size * seq_length) {
                for (int j = 0; j < input_dim; j++) {
                    h_X_batch[dst_idx * input_dim + j] = h_X[src_idx * input_dim + j];
                }
                
                for (int j = 0; j < output_dim; j++) {
                    h_y_batch[dst_idx * output_dim + j] = h_y[src_idx * output_dim + j];
                }
            }
        }
    }
    
    // Free original data
    free(h_X);
    free(h_y);
    
    // Transfer batch data to GPU
    float *d_X_batch, *d_y_batch;
    CHECK_CUDA(cudaMalloc(&d_X_batch, batch_size * seq_length * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y_batch, batch_size * seq_length * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_X_batch, h_X_batch, batch_size * seq_length * input_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y_batch, h_y_batch, batch_size * seq_length * output_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    // Free host data after transfer
    free(h_X_batch);
    free(h_y_batch);
    
    // Initialize new models
    SSM* layer1 = init_ssm(input_dim, state_dim1, hidden_dim, batch_size);
    SSM* layer2 = init_ssm(hidden_dim, state_dim2, output_dim, batch_size);
    
    // Training parameters
    const int num_epochs = 1000;
    float learning_rate = 0.0001f;
    
    printf("Starting SSM training for %d epochs...\n", num_epochs);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float total_loss = 0.0f;
        int num_batches = 0;
        
        // Reset states at the beginning of each epoch
        CHECK_CUDA(cudaMemset(layer1->d_state, 0, 
                            layer1->batch_size * layer1->state_dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(layer2->d_state, 0, 
                            layer2->batch_size * layer2->state_dim * sizeof(float)));
        
        // Training pass
        for (int step = 0; step < seq_length; step++) {
            // Get batch data (one time step across all episodes)
            float *d_batch_X = &d_X_batch[step * batch_size * input_dim];
            float *d_batch_y = &d_y_batch[step * batch_size * output_dim];
            
            // Forward pass through first layer
            forward_pass_ssm(layer1, d_batch_X);
            
            // Forward pass through second layer
            forward_pass_ssm(layer2, layer1->d_predictions);
            
            // Calculate loss
            float loss = calculate_loss_ssm(layer2, d_batch_y);
            total_loss += loss;
            num_batches++;
            
            // Zero gradients
            zero_gradients_ssm(layer1);
            zero_gradients_ssm(layer2);
            
            // Backward pass
            backward_pass_ssm(layer2, layer1->d_predictions);
            backward_between_models(layer1, layer2);
            backward_pass_ssm(layer1, d_batch_X);
            
            // Update weights
            update_weights_ssm(layer1, learning_rate);
            update_weights_ssm(layer2, learning_rate);
        }
        
        // Calculate average loss
        float avg_loss = total_loss / num_batches;
        
        // Print progress
        if (epoch == 0 || (epoch + 1) % 10 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", 
                   epoch + 1, num_epochs, avg_loss);
        }
    }
    
    // Save final models
    char layer1_path[128], layer2_path[128];
    sprintf(layer1_path, "%s_layer1.bin", model_prefix);
    sprintf(layer2_path, "%s_layer2.bin", model_prefix);
    save_ssm(layer1, layer1_path);
    save_ssm(layer2, layer2_path);
    printf("Final models saved to %s and %s\n", layer1_path, layer2_path);
    
    printf("Training complete.\n");
    
    // Cleanup
    cudaFree(d_X_batch);
    cudaFree(d_y_batch);
    free_ssm(layer1);
    free_ssm(layer2);
    
    return 0;
}