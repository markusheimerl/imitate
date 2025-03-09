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

// Train the SSM
void train_model(const char* data_file, const char* model_file, int num_episodes, const char* continue_model) {
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
    
    // Parameters for both layers
    const int input_dim = 16;      // IMU (6) + position (3) + velocity (3) + target (4)
    const int hidden_dim = 16;     // Hidden layer output dimension
    const int state_dim1 = 256;    // First layer state dimension
    const int state_dim2 = 256;    // Second layer state dimension
    const int output_dim = 4;      // Motor commands (4)
    
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
    
    // Allocate buffer for intermediate layer outputs
    float *d_hidden_train, *d_hidden_val;
    CHECK_CUDA(cudaMalloc(&d_hidden_train, train_episodes * seq_length * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_hidden_val, val_episodes * seq_length * hidden_dim * sizeof(float)));
    
    // Initialize state space models (2 layers for training and 2 for validation)
    SSM *train_ssm1, *train_ssm2, *val_ssm1, *val_ssm2;
    
    // Either load existing models or initialize new ones
    if (continue_model != NULL) {
        char model1_path[128], model2_path[128];
        sprintf(model1_path, "%s.layer1", continue_model);
        sprintf(model2_path, "%s.layer2", continue_model);
        
        printf("Continuing training from models:\n%s\n%s\n", model1_path, model2_path);
        
        // Load both layers
        train_ssm1 = load_ssm(model1_path, train_batch_size);
        train_ssm2 = load_ssm(model2_path, train_batch_size);
        
        // Initialize validation models with same dimensions
        val_ssm1 = init_ssm(train_ssm1->input_dim, train_ssm1->state_dim, 
                          train_ssm1->output_dim, val_batch_size);
        val_ssm2 = init_ssm(train_ssm2->input_dim, train_ssm2->state_dim, 
                          train_ssm2->output_dim, val_batch_size);
        
        // Copy weights from training models to validation models
        CHECK_CUDA(cudaMemcpy(val_ssm1->d_A, train_ssm1->d_A, 
                            train_ssm1->state_dim * train_ssm1->state_dim * sizeof(float), 
                            cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(val_ssm1->d_B, train_ssm1->d_B, 
                            train_ssm1->state_dim * train_ssm1->input_dim * sizeof(float), 
                            cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(val_ssm1->d_C, train_ssm1->d_C, 
                            train_ssm1->output_dim * train_ssm1->state_dim * sizeof(float), 
                            cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(val_ssm1->d_D, train_ssm1->d_D, 
                            train_ssm1->output_dim * train_ssm1->input_dim * sizeof(float), 
                            cudaMemcpyDeviceToDevice));
                            
        CHECK_CUDA(cudaMemcpy(val_ssm2->d_A, train_ssm2->d_A, 
                            train_ssm2->state_dim * train_ssm2->state_dim * sizeof(float), 
                            cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(val_ssm2->d_B, train_ssm2->d_B, 
                            train_ssm2->state_dim * train_ssm2->input_dim * sizeof(float), 
                            cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(val_ssm2->d_C, train_ssm2->d_C, 
                            train_ssm2->output_dim * train_ssm2->state_dim * sizeof(float), 
                            cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(val_ssm2->d_D, train_ssm2->d_D, 
                            train_ssm2->output_dim * train_ssm2->input_dim * sizeof(float), 
                            cudaMemcpyDeviceToDevice));
    } else {
        // Initialize new models
        train_ssm1 = init_ssm(input_dim, state_dim1, hidden_dim, train_batch_size);
        train_ssm2 = init_ssm(hidden_dim, state_dim2, output_dim, train_batch_size);
        val_ssm1 = init_ssm(input_dim, state_dim1, hidden_dim, val_batch_size);
        val_ssm2 = init_ssm(hidden_dim, state_dim2, output_dim, val_batch_size);
    }
    
    // Training parameters
    const int num_epochs = 1000;
    float learning_rate = 0.0001f;
    
    // If continuing, use a lower learning rate
    if (continue_model != NULL) {
        learning_rate *= 0.5f;
        printf("Reduced learning rate to %.6f for continued training\n", learning_rate);
    }
    
    printf("Starting SSM training for %d epochs...\n", num_epochs);
    
    // For tracking best model
    float best_val_loss = INFINITY;
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float train_loss = 0.0f;
        int train_batches = 0;
        
        // Reset states at the beginning of each epoch
        CHECK_CUDA(cudaMemset(train_ssm1->d_state, 0, 
                            train_ssm1->batch_size * train_ssm1->state_dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(train_ssm2->d_state, 0, 
                            train_ssm2->batch_size * train_ssm2->state_dim * sizeof(float)));
        
        // Training pass
        for (int step = 0; step < seq_length; step++) {
            // Get batch data (one time step across all training episodes)
            float *d_batch_X = &d_X_train[step * train_batch_size * input_dim];
            float *d_batch_y = &d_y_train[step * train_batch_size * output_dim];
            
            // Forward pass through first layer
            forward_pass(train_ssm1, d_batch_X);
            
            // Store outputs of first layer to use as inputs to second layer
            // and for backpropagation
            CHECK_CUDA(cudaMemcpy(&d_hidden_train[step * train_batch_size * hidden_dim], 
                                train_ssm1->d_predictions, 
                                train_batch_size * hidden_dim * sizeof(float), 
                                cudaMemcpyDeviceToDevice));
            
            // Forward pass through second layer
            forward_pass(train_ssm2, train_ssm1->d_predictions);
            
            // Calculate loss
            float loss = calculate_loss(train_ssm2, d_batch_y);
            train_loss += loss;
            train_batches++;
            
            // Backward pass through second layer
            zero_gradients(train_ssm2);
            backward_pass(train_ssm2, train_ssm1->d_predictions);
            
            // Backpropagate through the connection between layers
            backward_between_models(train_ssm1, train_ssm2, d_batch_X);
            
            // Update weights for both layers
            update_weights(train_ssm1, learning_rate);
            update_weights(train_ssm2, learning_rate);
        }
        
        // Calculate average training loss
        float avg_train_loss = train_loss / train_batches;
        
        // Copy updated weights to validation models
        CHECK_CUDA(cudaMemcpy(val_ssm1->d_A, train_ssm1->d_A, 
                            train_ssm1->state_dim * train_ssm1->state_dim * sizeof(float), 
                            cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(val_ssm1->d_B, train_ssm1->d_B, 
                            train_ssm1->state_dim * train_ssm1->input_dim * sizeof(float), 
                            cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(val_ssm1->d_C, train_ssm1->d_C, 
                            train_ssm1->output_dim * train_ssm1->state_dim * sizeof(float), 
                            cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(val_ssm1->d_D, train_ssm1->d_D, 
                            train_ssm1->output_dim * train_ssm1->input_dim * sizeof(float), 
                            cudaMemcpyDeviceToDevice));
                            
        CHECK_CUDA(cudaMemcpy(val_ssm2->d_A, train_ssm2->d_A, 
                            train_ssm2->state_dim * train_ssm2->state_dim * sizeof(float), 
                            cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(val_ssm2->d_B, train_ssm2->d_B, 
                            train_ssm2->state_dim * train_ssm2->input_dim * sizeof(float), 
                            cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(val_ssm2->d_C, train_ssm2->d_C, 
                            train_ssm2->output_dim * train_ssm2->state_dim * sizeof(float), 
                            cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(val_ssm2->d_D, train_ssm2->d_D, 
                            train_ssm2->output_dim * train_ssm2->input_dim * sizeof(float), 
                            cudaMemcpyDeviceToDevice));
        
        // Validation pass (no weight updates)
        float val_loss = 0.0f;
        int val_batches = 0;
        
        // Reset validation states
        CHECK_CUDA(cudaMemset(val_ssm1->d_state, 0, 
                            val_ssm1->batch_size * val_ssm1->state_dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(val_ssm2->d_state, 0, 
                            val_ssm2->batch_size * val_ssm2->state_dim * sizeof(float)));
        
        for (int step = 0; step < seq_length; step++) {
            // Get validation batch data
            float *d_batch_X = &d_X_val[step * val_batch_size * input_dim];
            float *d_batch_y = &d_y_val[step * val_batch_size * output_dim];
            
            // Forward pass through first layer
            forward_pass(val_ssm1, d_batch_X);
            
            // Store outputs of first layer
            CHECK_CUDA(cudaMemcpy(&d_hidden_val[step * val_batch_size * hidden_dim], 
                                val_ssm1->d_predictions, 
                                val_batch_size * hidden_dim * sizeof(float), 
                                cudaMemcpyDeviceToDevice));
            
            // Forward pass through second layer
            forward_pass(val_ssm2, val_ssm1->d_predictions);
            
            // Calculate validation loss
            float loss = calculate_loss(val_ssm2, d_batch_y);
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
                char best_model_fname1[128], best_model_fname2[128];
                sprintf(best_model_fname1, "%s.layer1.best", model_file);
                sprintf(best_model_fname2, "%s.layer2.best", model_file);
                save_ssm(train_ssm1, best_model_fname1);
                save_ssm(train_ssm2, best_model_fname2);
                printf("Saved best model with validation loss: %.8f\n", best_val_loss);
            }
        }
        
        // Print progress
        if (epoch == 0 || (epoch + 1) % 10 == 0) {
            printf("Epoch [%d/%d], Train Loss: %.8f, Val Loss: %.8f\n", 
                   epoch + 1, num_epochs, avg_train_loss, avg_val_loss);
        }
    }
    
    // Save final models
    char final_model_fname1[128], final_model_fname2[128];
    sprintf(final_model_fname1, "%s.layer1", model_file);
    sprintf(final_model_fname2, "%s.layer2", model_file);
    save_ssm(train_ssm1, final_model_fname1);
    save_ssm(train_ssm2, final_model_fname2);
    printf("Final models saved to %s.layer1 and %s.layer2\n", model_file, model_file);
    
    printf("Training complete. Best validation loss: %.8f\n", best_val_loss);
    
    // Cleanup
    cudaFree(d_X_train);
    cudaFree(d_y_train);
    cudaFree(d_X_val);
    cudaFree(d_y_val);
    cudaFree(d_hidden_train);
    cudaFree(d_hidden_val);
    free_ssm(train_ssm1);
    free_ssm(train_ssm2);
    free_ssm(val_ssm1);
    free_ssm(val_ssm2);
}

int main(int argc, char* argv[]) {
    if(argc != 3 && argc != 4) {
        printf("Usage: %s <num_episodes> <data_file> [continue_model]\n", argv[0]);
        return 1;
    }
    srand(time(NULL) ^ getpid());
    
    // Generate timestamped filenames
    char model_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_model.bin", localtime(&now));
    
    // Determine if we're continuing from an existing model
    const char* continue_model = (argc == 4) ? argv[3] : NULL;
    
    // Train SSM model
    printf("Training SSM with two layers...\n");
    train_model(argv[2], model_fname, atoi(argv[1]), continue_model);
    
    return 0;
}