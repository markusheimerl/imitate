#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/stat.h>
#include "sim/quad.h"
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

// Helper function for cosine learning rate schedule with a final constant period
float cosine_learning_rate(float initial_lr, float min_lr, int current_epoch, int total_epochs) {
    // 80% cosine decay, 20% constant minimum rate
    int cosine_epochs = 0.9 * total_epochs;
    
    if (current_epoch >= cosine_epochs) {
        // In the constant period
        return min_lr;
    } else {
        // In the cosine decay period
        float cosine_decay = 0.5f * (1.0f + cosf(M_PI * current_epoch / cosine_epochs));
        return min_lr + (initial_lr - min_lr) * cosine_decay;
    }
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

// Train a stacked model of four SSMs
void train_model(const char* data_file, const char* model1_out_file, const char* model2_out_file, 
                 const char* model3_out_file, const char* model4_out_file, int num_episodes,
                 const char* model1_in_file, const char* model2_in_file, 
                 const char* model3_in_file, const char* model4_in_file) {
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
    const int input_dim = 19;       // All input data (IMU + mag + position + velocity + target)
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
        
        // Read all input data (19 values)
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
    SSM* layer1_ssm;
    SSM* layer2_ssm;
    SSM* layer3_ssm;
    SSM* layer4_ssm;
    
    // Check if we're continuing training from existing models
    if (model1_in_file && model2_in_file && model3_in_file && model4_in_file) {
        printf("Continuing training from existing models...\n");
        layer1_ssm = load_ssm(model1_in_file, batch_size);
        layer2_ssm = load_ssm(model2_in_file, batch_size);
        layer3_ssm = load_ssm(model3_in_file, batch_size);
        layer4_ssm = load_ssm(model4_in_file, batch_size);
    } else {
        printf("Starting training from scratch...\n");
        layer1_ssm = init_ssm(input_dim, state_dim1, hidden_dim1, batch_size);
        layer2_ssm = init_ssm(hidden_dim1, state_dim2, hidden_dim2, batch_size);
        layer3_ssm = init_ssm(hidden_dim2, state_dim3, hidden_dim3, batch_size);
        layer4_ssm = init_ssm(hidden_dim3, state_dim4, output_dim, batch_size);
    }
    
    // Allocate memory for intermediate outputs
    float *d_hidden_output1, *d_hidden_output2, *d_hidden_output3;
    CHECK_CUDA(cudaMalloc(&d_hidden_output1, batch_size * hidden_dim1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_hidden_output2, batch_size * hidden_dim2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_hidden_output3, batch_size * hidden_dim3 * sizeof(float)));
    
    // Training parameters
    const int num_epochs = 500;
    const float initial_lr = 0.0001f;
    const float min_lr = 0.00001f;
    
    printf("Starting stacked model training for %d epochs...\n", num_epochs);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Calculate learning rate for current epoch
        float learning_rate = cosine_learning_rate(initial_lr, min_lr, epoch, num_epochs);
        
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
            
            // Update weights with current learning rate
            update_weights(layer1_ssm, learning_rate);
            update_weights(layer2_ssm, learning_rate);
            update_weights(layer3_ssm, learning_rate);
            update_weights(layer4_ssm, learning_rate);
        }
        
        // Print progress
        if (epoch == 0 || (epoch + 1) % 10 == 0) {
            printf("Epoch [%d/%d], Learning Rate: %.8f, Average Loss: %.8f\n", 
                   epoch + 1, num_epochs, learning_rate, epoch_loss / num_steps);
        }
    }

    save_ssm(layer1_ssm, model1_out_file);
    save_ssm(layer2_ssm, model2_out_file);
    save_ssm(layer3_ssm, model3_out_file);
    save_ssm(layer4_ssm, model4_out_file);
    
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

int main(int argc, char* argv[]) {
    // Check for minimum required arguments
    if (argc < 3) {
        printf("Usage: %s <data_file> <num_episodes> [model1_in model2_in model3_in model4_in]\n", argv[0]);
        return 1;
    }
    
    // Generate timestamped filenames for output models
    char model1_out_fname[64], model2_out_fname[64], model3_out_fname[64], model4_out_fname[64];
    time_t now = time(NULL);
    strftime(model1_out_fname, sizeof(model1_out_fname), "%Y%m%d_%H%M%S_layer1_model.bin", localtime(&now));
    strftime(model2_out_fname, sizeof(model2_out_fname), "%Y%m%d_%H%M%S_layer2_model.bin", localtime(&now));
    strftime(model3_out_fname, sizeof(model3_out_fname), "%Y%m%d_%H%M%S_layer3_model.bin", localtime(&now));
    strftime(model4_out_fname, sizeof(model4_out_fname), "%Y%m%d_%H%M%S_layer4_model.bin", localtime(&now));
    
    // Get data file and number of episodes from command line
    const char* data_file = argv[1];
    int num_episodes = atoi(argv[2]);
    
    // Check if we're continuing training from existing models
    const char* model1_in_fname = NULL;
    const char* model2_in_fname = NULL;
    const char* model3_in_fname = NULL;
    const char* model4_in_fname = NULL;
    
    if (argc >= 7) {
        model1_in_fname = argv[3];
        model2_in_fname = argv[4];
        model3_in_fname = argv[5];
        model4_in_fname = argv[6];
        printf("Continuing training from existing models\n");
    }
    
    printf("Training with data from: %s\n", data_file);
    printf("Number of episodes: %d\n", num_episodes);
    
    train_model(data_file, model1_out_fname, model2_out_fname, model3_out_fname, model4_out_fname, 
                num_episodes, model1_in_fname, model2_in_fname, model3_in_fname, model4_in_fname);
    
    return 0;
}