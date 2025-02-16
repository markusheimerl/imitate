#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "sim/quad.h"
#include "mlp/gpu/mlp.h"
#include "mlp/data.h"

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define SIM_TIME    5.0  // 5 seconds per episode

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
    fprintf(f, "px,py,pz,vx,vy,vz,"); // Position and velocity (6)
    fprintf(f, "r11,r12,r13,r21,r22,r23,r31,r32,r33,"); // Rotation matrix (9)
    fprintf(f, "wx,wy,wz,"); // Angular velocity (3)
    fprintf(f, "tx,ty,tz,"); // Target (3)
    fprintf(f, "m1,m2,m3,m4\n"); // Actions (4)
    
    for (int episode = 0; episode < num_episodes; episode++) {
        // Random initial state (y always positive)
        Quad* quad = create_quad(
            random_range(-2.0, 2.0),
            random_range(0.0, 2.0),    // Always at or above ground
            random_range(-2.0, 2.0)
        );

        double random_yaw = random_range(0.0, 2 * M_PI);
        double R_yaw[9] = {cos(random_yaw), 0.0, sin(random_yaw), 0.0, 1.0, 0.0, -sin(random_yaw), 0.0, cos(random_yaw)};
        memcpy(quad->R_W_B, R_yaw, 9 * sizeof(double));
        
        // Random target
        double target[7] = {
            random_range(-2.0, 2.0),   // x
            random_range(1.0, 3.0),    // y: Always above ground
            random_range(-2.0, 2.0),   // z
            0.0, 0.0, 0.0,             // vx, vy, vz
            random_yaw                 // yaw
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
                fprintf(f, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,", // Position and velocity
                       quad->linear_position_W[0], quad->linear_position_W[1], quad->linear_position_W[2],
                       quad->linear_velocity_W[0], quad->linear_velocity_W[1], quad->linear_velocity_W[2]);
                       
                fprintf(f, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,", // Rotation matrix
                       quad->R_W_B[0], quad->R_W_B[1], quad->R_W_B[2],
                       quad->R_W_B[3], quad->R_W_B[4], quad->R_W_B[5],
                       quad->R_W_B[6], quad->R_W_B[7], quad->R_W_B[8]);
                       
                fprintf(f, "%.6f,%.6f,%.6f,", // Angular velocity
                       quad->angular_velocity_B[0],
                       quad->angular_velocity_B[1],
                       quad->angular_velocity_B[2]);
                       
                fprintf(f, "%.6f,%.6f,%.6f,", // Target position
                       target[0], target[1], target[2]);
                       
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

// Train MLP
void train_policy(const char* data_file, const char* model_file) {
    printf("Loading training data from %s...\n", data_file);
    
    float *X, *y;
    int num_samples;
    load_csv(data_file, &X, &y, &num_samples, 21, 4);
    
    printf("Training data loaded: %d samples\n", num_samples);
    
    // Initialize MLP
    const int input_dim = 21;   // 18 state + 3 target
    const int hidden_dim = 512;
    const int output_dim = 4;   // 4 motor commands
    const int batch_size = num_samples;
    
    Net* net = init_net(input_dim, hidden_dim, output_dim, batch_size);
    
    // Training parameters
    const int num_epochs = 25000;
    const float learning_rate = 0.001f;
    
    printf("Starting training for %d epochs...\n", num_epochs);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        forward_pass(net, X);
        float loss = calculate_loss(net, y);
        zero_gradients(net);
        backward_pass(net, X);
        update_weights(net, learning_rate);
        
        if (epoch == 0 || (epoch + 1) % 100 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", 
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

int main() {
    srand(time(NULL) ^ getpid());
    
    // Generate timestamped filenames
    char data_fname[64], model_fname[64];
    time_t now = time(NULL);
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_flight.csv", localtime(&now));
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_policy.bin", localtime(&now));
    
    printf("Phase 1: Generating training data...\n");
    generate_training_data(data_fname, 500);
    
    printf("Phase 2: Training policy network...\n");
    train_policy(data_fname, model_fname);
    
    printf("Training complete!\n");
    printf("Data saved to: %s\n", data_fname);
    printf("Model saved to: %s\n", model_fname);
    
    return 0;
}