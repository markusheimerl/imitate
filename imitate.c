#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include "ssm/gpu/ssm.h"

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

// Custom function to propagate gradients between models
void backward_between_models(SSM* first_model, SSM* second_model, float* d_first_model_input) {
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

// Train four SSM models in sequence
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
    
    char line[16384];  // Increased buffer size for the larger resolution
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
    const int fpv_width = 32;
    const int fpv_height = 16;
    const int fpv_pixels = fpv_width * fpv_height;
    const int sensor_dim = 6;         // IMU data (6)
    const int input_dim = fpv_pixels + sensor_dim;  // Combined input dimension
    const int layer1_dim = 256;       // Output dimension for layer 1
    const int layer2_dim = 192;       // Output dimension for layer 2
    const int layer3_dim = 96;        // Output dimension for layer 3
    const int output_dim = 4;         // Motor commands (layer 4 output)
    
    const int layer1_state_dim = 512;  // State dimension for layer 1
    const int layer2_state_dim = 384;  // State dimension for layer 2
    const int layer3_state_dim = 192;  // State dimension for layer 3
    const int layer4_state_dim = 96;   // State dimension for layer 4
    
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
        
        // Read all input data (raw pixels + IMU data)
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
    SSM* layer1_ssm = NULL;
    SSM* layer2_ssm = NULL;
    SSM* layer3_ssm = NULL;
    SSM* layer4_ssm = NULL;
    
    // Training parameters
    const int num_epochs = 1000;
    const float learning_rate = 0.0001f;
    const int grad_accum_steps = 8;  // Number of steps to accumulate gradients
    
    // Check if we're continuing training from existing models
    if (access(layer1_file, F_OK) != -1 && 
        access(layer2_file, F_OK) != -1 && 
        access(layer3_file, F_OK) != -1 && 
        access(layer4_file, F_OK) != -1) {
        printf("Loading existing models for continued training...\n");
        layer1_ssm = load_ssm(layer1_file, batch_size);
        layer2_ssm = load_ssm(layer2_file, batch_size);
        layer3_ssm = load_ssm(layer3_file, batch_size);
        layer4_ssm = load_ssm(layer4_file, batch_size);
    } else {
        printf("Initializing new models for training...\n");
        layer1_ssm = init_ssm(input_dim, layer1_state_dim, layer1_dim, batch_size);
        layer2_ssm = init_ssm(layer1_dim, layer2_state_dim, layer2_dim, batch_size);
        layer3_ssm = init_ssm(layer2_dim, layer3_state_dim, layer3_dim, batch_size);
        layer4_ssm = init_ssm(layer3_dim, layer4_state_dim, output_dim, batch_size);
    }
    
    // Allocate memory for intermediate outputs
    float *d_layer1_output, *d_layer2_output, *d_layer3_output;
    CHECK_CUDA(cudaMalloc(&d_layer1_output, batch_size * layer1_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_layer2_output, batch_size * layer2_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_layer3_output, batch_size * layer3_dim * sizeof(float)));
    
    printf("Starting four-stage model training for %d epochs...\n", num_epochs);
    printf("Using gradient accumulation with %d steps\n", grad_accum_steps);
    printf("Learning rate: %.6f\n", learning_rate);
    
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
            backward_pass(layer4_ssm, d_layer3_output);
            
            // Backpropagate through layer 3
            backward_between_models(layer3_ssm, layer4_ssm, d_layer2_output);
            
            // Backpropagate through layer 2
            backward_between_models(layer2_ssm, layer3_ssm, d_layer1_output);
            
            // Backpropagate through layer 1
            backward_between_models(layer1_ssm, layer2_ssm, d_current_X);
            
            // Update weights only after accumulating gradients for grad_accum_steps or at end of sequence
            if ((step + 1) % grad_accum_steps == 0 || step == seq_length - 1) {
                // Update weights using accumulated gradients
                update_weights(layer1_ssm, learning_rate);
                update_weights(layer2_ssm, learning_rate);
                update_weights(layer3_ssm, learning_rate);
                update_weights(layer4_ssm, learning_rate);
                
                // Zero gradients after update
                zero_gradients(layer1_ssm);
                zero_gradients(layer2_ssm);
                zero_gradients(layer3_ssm);
                zero_gradients(layer4_ssm);
            }
        }
        
        // Print progress
        if (epoch == 0 || (epoch + 1) % 10 == 0) {
            printf("Epoch [%d/%d], Average Loss: %.8f\n", epoch + 1, num_epochs, epoch_loss / num_steps);
        }
    }
    
    // Generate timestamped filenames for output
    char time_prefix[64];
    time_t now = time(NULL);
    strftime(time_prefix, sizeof(time_prefix), "%Y%m%d_%H%M%S", localtime(&now));
    
    char final_layer1_file[128];
    char final_layer2_file[128];
    char final_layer3_file[128];
    char final_layer4_file[128];

    sprintf(final_layer1_file, "%s_layer1_model.bin", time_prefix);
    sprintf(final_layer2_file, "%s_layer2_model.bin", time_prefix);
    sprintf(final_layer3_file, "%s_layer3_model.bin", time_prefix);
    sprintf(final_layer4_file, "%s_layer4_model.bin", time_prefix);

    save_ssm(layer1_ssm, final_layer1_file);
    save_ssm(layer2_ssm, final_layer2_file);
    save_ssm(layer3_ssm, final_layer3_file);
    save_ssm(layer4_ssm, final_layer4_file);
    
    printf("Models saved to:\n");
    printf("  %s\n", final_layer1_file);
    printf("  %s\n", final_layer2_file);
    printf("  %s\n", final_layer3_file);
    printf("  %s\n", final_layer4_file);
    
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

int main(int argc, char** argv) {
    srand(time(NULL) ^ getpid());
    
    // Default filenames and parameters
    char data_fname[128];
    char layer1_fname[128], layer2_fname[128], layer3_fname[128], layer4_fname[128];
    int num_episodes = 2000;
    
    // Generate timestamped filenames for models
    char time_prefix[64];
    time_t now = time(NULL);
    strftime(time_prefix, sizeof(time_prefix), "%Y%m%d_%H%M%S", localtime(&now));
    
    sprintf(layer1_fname, "%s_layer1_model.bin", time_prefix);
    sprintf(layer2_fname, "%s_layer2_model.bin", time_prefix);
    sprintf(layer3_fname, "%s_layer3_model.bin", time_prefix);
    sprintf(layer4_fname, "%s_layer4_model.bin", time_prefix);
    
    // Check for command line arguments
    if (argc >= 6) {
        // First argument is data file, next four are model files
        strcpy(data_fname, argv[1]);
        strcpy(layer1_fname, argv[2]);
        strcpy(layer2_fname, argv[3]);
        strcpy(layer3_fname, argv[4]);
        strcpy(layer4_fname, argv[5]);
        
        printf("Using data file: %s\n", data_fname);
        printf("Continuing training with existing models:\n");
        printf("Layer 1: %s\n", layer1_fname);
        printf("Layer 2: %s\n", layer2_fname);
        printf("Layer 3: %s\n", layer3_fname);
        printf("Layer 4: %s\n", layer4_fname);
    } else if (argc >= 2) {
        // First argument is data file, use new model files
        strcpy(data_fname, argv[1]);
        printf("Using data file: %s\n", data_fname);
        printf("Starting new training run with timestamped model files\n");
    } else {
        printf("Error: Supply data file.\n");
        return 1;
    }
    
    printf("Training four-stage SSM model with raw pixel input...\n");
    train_stacked_models(data_fname, layer1_fname, layer2_fname, layer3_fname, layer4_fname, num_episodes);
    
    printf("Training complete!\n");
    
    return 0;
}