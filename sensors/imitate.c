#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "../sim/quad.h"
#include "../mlp/gpu/mlp.h"
#include "../mlp/data.h"

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define SIM_TIME    5.0  // 5 seconds per episode

#define HISTORY_LENGTH 10  // Number of historical readings to keep
#define SENSOR_DIMS 6     // 3 gyro + 3 accel readings

// Helper function to get random value in range [min, max]
double random_range(double min, double max) {
    return min + (double)rand() / RAND_MAX * (max - min);
}

// Simple helper to add new reading to history buffer
void update_sensor_history(double* history, const double* gyro, const double* accel) {
    // Shift old readings back
    memmove(history + SENSOR_DIMS, history, SENSOR_DIMS * (HISTORY_LENGTH - 1) * sizeof(double));
    
    // Add new reading at front
    for(int i = 0; i < 3; i++) {
        history[i] = gyro[i];
        history[i + 3] = accel[i];
    }
}

// Calculate linear acceleration in world frame from quad state
void calculate_linear_acceleration(const Quad* q, double* linear_acceleration_W) {
    // Calculate thrust from rotor speeds
    double thrust = 0;
    for(int i = 0; i < 4; i++) {
        double omega_sq = q->omega[i] * fabs(q->omega[i]);
        thrust += K_F * omega_sq;
    }

    // Calculate linear acceleration in world frame
    double f_B_thrust[3] = {0, thrust, 0};
    double f_thrust_W[3];
    multMatVec3f(q->R_W_B, f_B_thrust, f_thrust_W);

    // Convert force to acceleration
    for(int i = 0; i < 3; i++) {
        linear_acceleration_W[i] = f_thrust_W[i] / MASS;
    }
    linear_acceleration_W[1] -= GRAVITY;
}

// Simulate ideal gyroscope reading (just returns angular velocity in body frame)
void simulate_gyro(const Quad* q, double* gyro_reading) {
    for(int i = 0; i < 3; i++) {
        gyro_reading[i] = q->angular_velocity_B[i];
    }
}

// Simulate ideal accelerometer reading
void simulate_accelerometer(const Quad* q, const double* linear_acceleration_W, double* accel_reading) {
    // Convert world acceleration to body frame
    double R_B_W[9];
    transpMat3f(q->R_W_B, R_B_W);
    
    // Transform world acceleration to body frame
    double accel_B[3];
    multMatVec3f(R_B_W, linear_acceleration_W, accel_B);
    
    // Copy result
    for(int i = 0; i < 3; i++) {
        accel_reading[i] = accel_B[i];
    }
}

// Generate training data
void generate_training_data(const char* filename, int num_episodes) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        printf("Error opening file: %s\n", filename);
        return;
    }
    
    // Write header
    fprintf(f, "px,py,pz,"); // Position (3)
    for(int t = 0; t < HISTORY_LENGTH; t++) {
        fprintf(f, "gx%d,gy%d,gz%d,ax%d,ay%d,az%d,", t, t, t, t, t, t);
    }
    fprintf(f, "tx,ty,tz,tvx,tvy,tvz,tyaw,"); // Target (7)
    fprintf(f, "m1,m2,m3,m4\n"); // Actions (4)
    
    // Preallocate sensor history buffer
    double sensor_history[HISTORY_LENGTH * SENSOR_DIMS] = {0};
    
    for (int episode = 0; episode < num_episodes; episode++) {
        // Random initial state (y always positive)
        Quad* quad = create_quad(
            random_range(-2.0, 2.0),
            random_range(0.0, 2.0),    // Always at or above ground
            random_range(-2.0, 2.0)
        );
        
        // Random target (y always positive)
        double target[7] = {
            random_range(-2.0, 2.0),    // x
            random_range(1.0, 3.0),     // y: Always above ground
            random_range(-2.0, 2.0),    // z
            0.0, 0.0, 0.0,              // vx, vy, vz
            random_range(0.0, 2*M_PI)   // yaw
        };
        
        // Clear history at start of episode
        memset(sensor_history, 0, sizeof(sensor_history));
        
        double t_physics = 0.0;
        double t_control = 0.0;
        
        for(int i = 0; i < (int)(SIM_TIME / DT_PHYSICS); i++) {
            if (t_physics >= DT_PHYSICS) {
                update_quad(quad, DT_PHYSICS);
                t_physics = 0.0;
            }
            
            if (t_control >= DT_CONTROL) {
                // Get motor commands from geometric controller
                control_quad(quad, target);
                
                // Calculate accelerations and get sensor readings
                double linear_acceleration_W[3];
                calculate_linear_acceleration(quad, linear_acceleration_W);

                double gyro_reading[3];
                simulate_gyro(quad, gyro_reading);

                double accel_reading[3];
                simulate_accelerometer(quad, linear_acceleration_W, accel_reading);
                
                // Update history
                update_sensor_history(sensor_history, gyro_reading, accel_reading);
                
                // Write state to file
                fprintf(f, "%.6f,%.6f,%.6f,", // Position
                       quad->linear_position_W[0],
                       quad->linear_position_W[1],
                       quad->linear_position_W[2]);
                       
                // Write full sensor history
                for(int t = 0; t < HISTORY_LENGTH * SENSOR_DIMS; t++) {
                    fprintf(f, "%.6f,", sensor_history[t]);
                }
                       
                fprintf(f, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,", // Target
                       target[0], target[1], target[2],
                       target[3], target[4], target[5], target[6]);
                       
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
    const int input_dim = 3 + (HISTORY_LENGTH * SENSOR_DIMS) + 7;  // 3 pos + (history * sensors) + 7 target
    load_csv(data_file, &X, &y, &num_samples, input_dim, 4);
    
    printf("Training data loaded: %d samples\n", num_samples);
    
    // Initialize MLP
    const int hidden_dim = 1024;
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
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_flight.csv", 
             localtime(&now));
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_policy.bin", 
             localtime(&now));
    
    printf("Phase 1: Generating training data...\n");
    generate_training_data(data_fname, 500);
    
    printf("Phase 2: Training policy network...\n");
    train_policy(data_fname, model_fname);
    
    printf("Training complete!\n");
    printf("Data saved to: %s\n", data_fname);
    printf("Model saved to: %s\n", model_fname);
    
    return 0;
}