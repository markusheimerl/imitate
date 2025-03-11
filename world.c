#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mlp/gpu/mlp.h"

#define MAX_LINE_LENGTH 4096

// Helper function to count lines in file
int count_lines(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file: %s\n", filename);
        return -1;
    }
    
    int count = 0;
    char c;
    while ((c = fgetc(file)) != EOF) {
        if (c == '\n') count++;
    }
    
    fclose(file);
    return count;
}

// Helper function to count columns in CSV header
int count_columns(const char* header) {
    int count = 1; // At least one column
    for (int i = 0; header[i]; i++) {
        if (header[i] == ',') count++;
    }
    return count;
}

// Load dynamics data from CSV
int load_dynamics_data(const char* filename, float** X_data, float** y_data, int* num_samples, int* num_inputs, int* num_outputs) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file: %s\n", filename);
        return 0;
    }
    
    // Read header to determine structure
    char header[MAX_LINE_LENGTH];
    if (!fgets(header, MAX_LINE_LENGTH, file)) {
        printf("Error reading header from file: %s\n", filename);
        fclose(file);
        return 0;
    }
    
    // Count total lines (samples + header)
    int line_count = count_lines(filename);
    if (line_count <= 1) {
        printf("Not enough data in file: %s\n", filename);
        fclose(file);
        return 0;
    }
    
    // Determine number of samples (excluding header)
    *num_samples = line_count - 1;
    
    // Based on the dynamics data structure from data.c:
    // First 39 columns are inputs
    // Next 31 columns are outputs
    *num_inputs = 39;
    *num_outputs = 31;
    
    printf("Data file: %s\n", filename);
    printf("Samples: %d\n", *num_samples);
    printf("Input features: %d\n", *num_inputs);
    printf("Output features: %d\n", *num_outputs);
    
    // Allocate memory for data
    *X_data = (float*)malloc(*num_samples * (*num_inputs) * sizeof(float));
    *y_data = (float*)malloc(*num_samples * (*num_outputs) * sizeof(float));
    
    if (!*X_data || !*y_data) {
        printf("Memory allocation failed\n");
        free(*X_data);
        free(*y_data);
        fclose(file);
        return 0;
    }
    
    // Rewind to skip header and prepare to read data
    rewind(file);
    fgets(header, MAX_LINE_LENGTH, file); // Skip header
    
    // Read all data samples
    char line[MAX_LINE_LENGTH];
    int row = 0;
    
    while (fgets(line, MAX_LINE_LENGTH, file) && row < *num_samples) {
        char* token = strtok(line, ",");
        int col = 0;
        
        while (token && col < *num_inputs + *num_outputs) {
            float value = atof(token);
            
            if (col < *num_inputs) {
                // Input feature
                (*X_data)[row * (*num_inputs) + col] = value;
            } else {
                // Output feature
                (*y_data)[row * (*num_outputs) + (col - *num_inputs)] = value;
            }
            
            token = strtok(NULL, ",");
            col++;
        }
        
        row++;

    }
    
    fclose(file);
    printf("Successfully loaded %d data samples\n", row);
    return 1;
}

int main(int argc, char *argv[]) {
    srand(time(NULL));
    
    // Check command-line arguments
    if (argc < 2) {
        printf("Usage: %s <dynamics_data.csv> [num_epochs]\n", argv[0]);
        return 1;
    }
    
    const char* data_file = argv[1];
    int num_epochs = (argc > 2) ? atoi(argv[2]) : 1000;
    
    float* X_data = NULL;
    float* y_data = NULL;
    int num_samples, input_dim, output_dim;
    
    // Load data
    if (!load_dynamics_data(data_file, &X_data, &y_data, &num_samples, &input_dim, &output_dim)) {
        printf("Failed to load data\n");
        return 1;
    }
    
    // Use batch size equal to number of samples as instructed
    int batch_size = num_samples;
    int hidden_dim = 1024;
    
    printf("Starting model training:\n");
    printf("- Epochs: %d\n", num_epochs);
    printf("- Batch size: %d (full dataset)\n", batch_size);
    printf("- Hidden dim: %d\n", hidden_dim);
    
    // Initialize network
    MLP* mlp = init_mlp(input_dim, hidden_dim, output_dim, batch_size);
    
    // Training parameters
    const float learning_rate = 0.001f;
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Forward pass with full batch
        forward_pass_mlp(mlp, X_data);
        
        // Calculate loss
        float loss = calculate_loss_mlp(mlp, y_data);
        
        // Backward pass
        zero_gradients_mlp(mlp);
        backward_pass_mlp(mlp, X_data);
        
        // Update weights
        update_weights_mlp(mlp, learning_rate);
        
        // Print progress
        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch + 1, num_epochs, loss);
        }
    }

    // Get timestamp for filename
    char model_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_world_model.bin", 
             localtime(&now));
    
    // Save model
    save_mlp(mlp, model_fname);
    printf("Dynamics model saved to: %s\n", model_fname);
    
    // Cleanup
    free(X_data);
    free(y_data);
    free_mlp(mlp);
    
    return 0;
}