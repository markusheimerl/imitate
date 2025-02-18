#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "sim/quad.h"
#include "ssm/gpu/ssm.h"
#include "ssm/data.h"

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define SIM_TIME    0.1  // 5 seconds per episode

// Helper function to get random value in range [min, max]
double random_range(double min, double max) {
    return min + (double)rand() / RAND_MAX * (max - min);
}

// Generate training data
void generate_training_data(const char* filename, int num_episodes) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        printf("Error opening file: %s\n", filename);
        return;
    }
    
    // Write header
    fprintf(f, "vx,vy,vz,"); // Velocity (3)
    fprintf(f, "wx,wy,wz,"); // Angular velocity (3)
    fprintf(f, "tx,ty,tz,tyaw,"); // Target (4)
    fprintf(f, "m1,m2,m3,m4\n"); // Actions (4)
    
    for (int episode = 0; episode < num_episodes; episode++) {
        // Random initial state
        Quad* quad = create_quad(
            random_range(-2.0, 2.0),
            random_range(0.0, 2.0),    // Always at or above ground
            random_range(-2.0, 2.0)
        );
        
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
                update_quad(quad, DT_PHYSICS);
                t_physics = 0.0;
            }
            
            if (t_control >= DT_CONTROL) {
                // Get motor commands from geometric controller
                control_quad(quad, target);
                
                // Write state, target, and action to file
                fprintf(f, "%.6f,%.6f,%.6f,", // Velocity
                       quad->linear_velocity_W[0],
                       quad->linear_velocity_W[1],
                       quad->linear_velocity_W[2]);
                       
                fprintf(f, "%.6f,%.6f,%.6f,", // Angular velocity
                       quad->angular_velocity_B[0],
                       quad->angular_velocity_B[1],
                       quad->angular_velocity_B[2]);
                       
                fprintf(f, "%.6f,%.6f,%.6f,%.6f,", // Target
                       target[0], target[1], target[2], target[6]);
                       
                fprintf(f, "%.6f,%.6f,%.6f,%.6f\n", // Motor commands
                       quad->omega_next[0],
                       quad->omega_next[1],
                       quad->omega_next[2],
                       quad->omega_next[3]);
                       
                t_control = 0.0;
            }
            
            t_physics += DT_PHYSICS;
            t_control += DT_PHYSICS;
        }
        
        free(quad);
        
        if ((episode + 1) % 10 == 0) {
            printf("Generated %d episodes\n", episode + 1);
        }
    }
    
    fclose(f);
}

// Train SSM
void train_policy(const char* data_file, const char* model_file, int num_episodes) {
    printf("Loading training data from %s...\n", data_file);
    
    float *X, *y;
    int num_samples;
    load_csv(data_file, &X, &y, &num_samples, 10, 4); // Changed input dim to 10
    
    printf("Training data loaded: %d samples\n", num_samples);
    
    // Initialize SSM
    const int input_dim = 10;   // 6 state + 4 target
    const int state_dim = 512;  // Hidden state dimension
    const int output_dim = 4;   // 4 motor commands
    const int batch_size = num_episodes;  // Process in batches for SSM
    const int seq_length = (int)(SIM_TIME / DT_CONTROL - 0.5);  // Sequence length for temporal dependencies
    printf("Sequence length times batch size: %d (must equal total training samples)\n", seq_length * batch_size);
    
    // Transfer data to GPU
    float *d_X, *d_y;
    CHECK_CUDA(cudaMalloc(&d_X, num_samples * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, num_samples * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_X, X, num_samples * input_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, y, num_samples * output_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    // Initialize SSM
    SSM* ssm = init_ssm(input_dim, state_dim, output_dim, batch_size);
    
    // Training parameters
    const int num_epochs = 50000;
    const float learning_rate = 0.0001;
    
    // Allocate memory for batch data
    float *d_batch_X, *d_batch_y;
    CHECK_CUDA(cudaMalloc(&d_batch_X, batch_size * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_batch_y, batch_size * output_dim * sizeof(float)));
    
    printf("Starting training for %d epochs...\n", num_epochs);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        int num_batches = 0;
        
        // Process data in sequences
        for (int seq_start = 0; seq_start <= num_samples - seq_length; seq_start += seq_length) {
            // Reset state at the beginning of each sequence
            CHECK_CUDA(cudaMemset(ssm->d_state, 0, 
                                 ssm->batch_size * ssm->state_dim * sizeof(float)));
            
            // Process sequence in batches
            for (int step = 0; step < seq_length; step += batch_size) {
                int current_batch_size = batch_size;
                if (seq_start + step + batch_size > num_samples) {
                    current_batch_size = num_samples - (seq_start + step);
                }
                
                // Prepare batch data
                CHECK_CUDA(cudaMemcpy(d_batch_X, 
                                    &d_X[(seq_start + step) * input_dim],
                                    current_batch_size * input_dim * sizeof(float),
                                    cudaMemcpyDeviceToDevice));
                CHECK_CUDA(cudaMemcpy(d_batch_y,
                                    &d_y[(seq_start + step) * output_dim],
                                    current_batch_size * output_dim * sizeof(float),
                                    cudaMemcpyDeviceToDevice));
                
                // Forward pass
                forward_pass(ssm, d_batch_X);
                float loss = calculate_loss(ssm, d_batch_y);
                epoch_loss += loss;
                num_batches++;
                
                // Backward pass and update
                zero_gradients(ssm);
                backward_pass(ssm, d_batch_X);
                update_weights(ssm, learning_rate);
            }
        }
        
        if ((epoch + 1) % 100 == 0) {
            printf("Epoch [%d/%d], Average Loss: %.8f\n", 
                   epoch + 1, num_epochs, epoch_loss / num_batches);
        }
    }
    
    // Save trained model
    save_model(ssm, model_file);
    
    // Cleanup
    free(X);
    free(y);
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_batch_X);
    cudaFree(d_batch_y);
    free_ssm(ssm);
}

int main() {
    srand(time(NULL) ^ getpid());
    
    // Generate timestamped filenames
    char data_fname[64], model_fname[64];
    time_t now = time(NULL);
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_flight.csv", localtime(&now));
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_policy.bin", localtime(&now));
    
    printf("Phase 1: Generating training data...\n");
    generate_training_data(data_fname, 8);
    
    printf("Phase 2: Training policy network...\n");
    train_policy(data_fname, model_fname, 8);
    
    printf("Training complete!\n");
    printf("Data saved to: %s\n", data_fname);
    printf("Model saved to: %s\n", model_fname);
    
    return 0;
}