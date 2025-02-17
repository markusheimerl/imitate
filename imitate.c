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
#define SIM_TIME    3000.0

// Helper function to get random value in range [min, max]
double random_range(double min, double max) {
    return min + (double)rand() / RAND_MAX * (max - min);
}

// Check if quadcopter has reached target
int target_reached(Quad* quad, double* target) {
    double dx = quad->linear_position_W[0] - target[0];
    double dy = quad->linear_position_W[1] - target[1];
    double dz = quad->linear_position_W[2] - target[2];
    double distance = sqrt(dx*dx + dy*dy + dz*dz);
    double velocity_mag = sqrt(
        quad->linear_velocity_W[0] * quad->linear_velocity_W[0] +
        quad->linear_velocity_W[1] * quad->linear_velocity_W[1] +
        quad->linear_velocity_W[2] * quad->linear_velocity_W[2]
    );
    return (distance < 0.05 && velocity_mag < 0.02);
}

// Generate new random target
void generate_target(double* target) {
    target[0] = random_range(-2.0, 2.0);     // x
    target[1] = random_range(1.0, 3.0);      // y: Always above ground
    target[2] = random_range(-2.0, 2.0);     // z
    target[3] = target[4] = target[5] = 0.0; // vx, vy, vz
    target[6] = random_range(0.0, 2*M_PI);   // yaw
}

// Generate training data
void generate_training_data(const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        printf("Error opening file: %s\n", filename);
        return;
    }
    
    // Write header
    fprintf(f, "px,py,pz,vx,vy,vz,"); // Position and velocity (6)
    fprintf(f, "wx,wy,wz,"); // Angular velocity (3)
    fprintf(f, "tx,ty,tz,tyaw,"); // Target (4)
    fprintf(f, "m1,m2,m3,m4\n"); // Actions (4)
    
    // Initialize quadcopter randomly
    Quad* quad = create_quad(
        random_range(-2.0, 2.0),
        random_range(1.0, 3.0),
        random_range(-2.0, 2.0)
    );
    
    // Initialize first target
    double target[7];
    generate_target(target);
    
    int target_count = 0;
    
    double t_physics = 0.0;
    double t_control = 0.0;
    double total_time = 0.0;
    
    while (total_time < SIM_TIME) {
        if (t_physics >= DT_PHYSICS) {
            update_quad(quad, DT_PHYSICS);
            t_physics = 0.0;
            total_time += DT_PHYSICS;
            
            // Check if current target is reached
            if (target_reached(quad, target)) {
                target_count++;
                generate_target(target);
            }
        }
        
        if (t_control >= DT_CONTROL) {
            // Get motor commands from geometric controller
            control_quad(quad, target);
            
            // Write state, target, and action to file
            fprintf(f, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,", // Position and velocity
                   quad->linear_position_W[0], quad->linear_position_W[1], quad->linear_position_W[2],
                   quad->linear_velocity_W[0], quad->linear_velocity_W[1], quad->linear_velocity_W[2]);
                   
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
    fclose(f);
    
    printf("Generated data for %.1f seconds with %d targets reached\n", total_time, target_count);
}

// Train SSM
void train_policy(const char* data_file, const char* model_file) {
    printf("Loading training data from %s...\n", data_file);
    
    float *h_X, *h_y;  // Host data
    int num_samples;
    load_csv(data_file, &h_X, &h_y, &num_samples, 13, 4);
    
    printf("Training data loaded: %d samples\n", num_samples);
    
    // Initialize SSM
    const int input_dim = 13;   // 6 pos/vel + 3 angular vel + 4 target
    const int hidden_dim = 512;
    const int output_dim = 4;   // 4 motor commands
    const int batch_size = num_samples;
    
    // Allocate device memory for training data
    float *d_X, *d_y;
    size_t X_size = num_samples * input_dim * sizeof(float);
    size_t y_size = num_samples * output_dim * sizeof(float);
    
    CHECK_CUDA(cudaMalloc(&d_X, X_size));
    CHECK_CUDA(cudaMalloc(&d_y, y_size));
    
    // Copy training data to device
    CHECK_CUDA(cudaMemcpy(d_X, h_X, X_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, y_size, cudaMemcpyHostToDevice));
    
    // Initialize SSM
    SSM* ssm = init_ssm(input_dim, hidden_dim, output_dim, batch_size);
    
    // Training parameters
    const int num_epochs = 100000;
    const float learning_rate = 0.001f;
    
    printf("Starting training for %d epochs...\n", num_epochs);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        CHECK_CUDA(cudaMemset(ssm->d_state, 0, batch_size * hidden_dim * sizeof(float)));
        
        forward_pass(ssm, d_X);
        float loss = calculate_loss(ssm, d_y);
        zero_gradients(ssm);
        backward_pass(ssm, d_X);
        update_weights(ssm, learning_rate);
        
        if (epoch == 0 || (epoch + 1) % 100 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", 
                   epoch + 1, num_epochs, loss);
        }
    }
    
    // Save trained model
    ssm->batch_size = 1;
    save_model(ssm, model_file);
    
    // Cleanup
    free(h_X);
    free(h_y);
    cudaFree(d_X);
    cudaFree(d_y);
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
    generate_training_data(data_fname);
    
    printf("Phase 2: Training policy network...\n");
    train_policy(data_fname, model_fname);
    
    printf("Training complete!\n");
    printf("Data saved to: %s\n", data_fname);
    printf("Model saved to: %s\n", model_fname);
    
    return 0;
}