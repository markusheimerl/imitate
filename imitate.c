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
#define SIM_TIME    3000.0
#define VELOCITY_HISTORY 10  // Number of velocity samples to keep

typedef struct {
    double velocities[VELOCITY_HISTORY][3];
    int current_idx;
    int filled;
} VelocityEstimator;

// Check if a value is NaN
int is_nan(double x) {
    return x != x;
}

// Check if any value in array is NaN
int array_has_nan(double* arr, int size) {
    for(int i = 0; i < size; i++) {
        if(is_nan(arr[i])) return 1;
    }
    return 0;
}

// Initialize velocity estimator
void init_estimator(VelocityEstimator* est) {
    memset(est->velocities, 0, sizeof(est->velocities));
    est->current_idx = 0;
    est->filled = 0;
}

// Update velocity history
void update_estimator(VelocityEstimator* est, double* velocity) {
    // Check for NaN in input velocity
    if(array_has_nan(velocity, 3)) {
        // If NaN detected, don't update
        return;
    }
    
    memcpy(est->velocities[est->current_idx], velocity, 3 * sizeof(double));
    est->current_idx = (est->current_idx + 1) % VELOCITY_HISTORY;
    if (est->current_idx == 0) {
        est->filled = 1;
    }
}

// Normalize vector safely
int safe_normalize(double* vec, int size) {
    double sum_sq = 0.0;
    for(int i = 0; i < size; i++) {
        if(is_nan(vec[i])) return 0;
        sum_sq += vec[i] * vec[i];
    }
    
    if(sum_sq < 1e-10) return 0;
    
    double mag = sqrt(sum_sq);
    for(int i = 0; i < size; i++) {
        vec[i] /= mag;
    }
    return 1;
}

// Estimate rotation matrix from velocity history
void estimate_rotation(VelocityEstimator* est, double* R_est) {
    // Initialize to identity matrix
    for(int i = 0; i < 9; i++) {
        R_est[i] = (i % 4 == 0) ? 1.0 : 0.0;
    }
    
    // Calculate primary direction (z-axis) from average velocity
    double avg_vel[3] = {0, 0, 0};
    int count = est->filled ? VELOCITY_HISTORY : est->current_idx;
    if (count == 0) return;
    
    int valid_samples = 0;
    for(int i = 0; i < count; i++) {
        if(!array_has_nan(est->velocities[i], 3)) {
            for(int j = 0; j < 3; j++) {
                avg_vel[j] += est->velocities[i][j];
            }
            valid_samples++;
        }
    }
    
    if(valid_samples == 0) return;
    
    for(int j = 0; j < 3; j++) {
        avg_vel[j] /= valid_samples;
    }
    
    // Normalize average velocity to get z-axis
    double z_axis[3];
    memcpy(z_axis, avg_vel, 3 * sizeof(double));
    if(!safe_normalize(z_axis, 3)) return;
    
    // Create arbitrary x-axis perpendicular to z-axis
    double x_axis[3];
    if (fabs(z_axis[1]) < 0.9) {
        x_axis[0] = -z_axis[1];
        x_axis[1] = z_axis[0];
        x_axis[2] = 0;
    } else {
        x_axis[0] = -z_axis[2];
        x_axis[1] = 0;
        x_axis[2] = z_axis[0];
    }
    
    if(!safe_normalize(x_axis, 3)) return;
    
    // Calculate y-axis as cross product of z and x
    double y_axis[3] = {
        z_axis[1]*x_axis[2] - z_axis[2]*x_axis[1],
        z_axis[2]*x_axis[0] - z_axis[0]*x_axis[2],
        z_axis[0]*x_axis[1] - z_axis[1]*x_axis[0]
    };
    
    if(array_has_nan(y_axis, 3)) return;
    if(!safe_normalize(y_axis, 3)) return;
    
    // Final NaN check before forming rotation matrix
    if(array_has_nan(x_axis, 3) || array_has_nan(y_axis, 3) || array_has_nan(z_axis, 3)) {
        return;
    }
    
    // Form rotation matrix
    R_est[0] = x_axis[0]; R_est[1] = y_axis[0]; R_est[2] = z_axis[0];
    R_est[3] = x_axis[1]; R_est[4] = y_axis[1]; R_est[5] = z_axis[1];
    R_est[6] = x_axis[2]; R_est[7] = y_axis[2]; R_est[8] = z_axis[2];
}

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
    target[1] = random_range(1.0, 3.0);      // y
    target[2] = random_range(-2.0, 2.0);     // z
    target[3] = target[4] = target[5] = 0.0; // vx, vy, vz
    target[6] = random_range(0.0, 2*M_PI);   // yaw
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
    fprintf(f, "r11,r12,r13,r21,r22,r23,r31,r32,r33,"); // Estimated rotation matrix (9)
    fprintf(f, "wx,wy,wz,"); // Angular velocity (3)
    fprintf(f, "tx,ty,tz,tyaw,"); // Target (7)
    fprintf(f, "m1,m2,m3,m4\n"); // Actions (4)
    
    Quad* quad = create_quad(
        random_range(-2.0, 2.0),
        random_range(1.0, 3.0),
        random_range(-2.0, 2.0)
    );
    
    // Initialize estimator
    VelocityEstimator estimator;
    init_estimator(&estimator);
    
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
            
            // Update velocity estimator
            update_estimator(&estimator, quad->linear_velocity_W);
            
            if (target_reached(quad, target)) {
                target_count++;
                generate_target(target);
            }
        }
        
        if (t_control >= DT_CONTROL) {
            control_quad(quad, target);
            
            // Estimate rotation matrix
            double R_est[9];
            estimate_rotation(&estimator, R_est);
            
            // Check for NaN in all values before writing
            if(!array_has_nan(quad->linear_position_W, 3) && 
               !array_has_nan(quad->linear_velocity_W, 3) && 
               !array_has_nan(R_est, 9) &&
               !array_has_nan(quad->angular_velocity_B, 3) &&
               !array_has_nan(target, 4) &&
               !array_has_nan(quad->omega_next, 4)) {
                
                fprintf(f, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,", 
                       quad->linear_position_W[0], quad->linear_position_W[1], quad->linear_position_W[2],
                       quad->linear_velocity_W[0], quad->linear_velocity_W[1], quad->linear_velocity_W[2]);
                       
                fprintf(f, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,",
                       R_est[0], R_est[1], R_est[2],
                       R_est[3], R_est[4], R_est[5],
                       R_est[6], R_est[7], R_est[8]);
                       
                fprintf(f, "%.6f,%.6f,%.6f,",
                       quad->angular_velocity_B[0],
                       quad->angular_velocity_B[1],
                       quad->angular_velocity_B[2]);
                       
                fprintf(f, "%.6f,%.6f,%.6f,%.6f,",
                       target[0], target[1], target[2], target[6]);
                       
                fprintf(f, "%.6f,%.6f,%.6f,%.6f\n",
                       quad->omega_next[0],
                       quad->omega_next[1],
                       quad->omega_next[2],
                       quad->omega_next[3]);
            }
            
            t_control = 0.0;
        }
        
        t_physics += DT_PHYSICS;
        t_control += DT_PHYSICS;
    }
    
    free(quad);
    fclose(f);
}

// Train MLP
void train_policy(const char* data_file, const char* model_file) {
    printf("Loading training data from %s...\n", data_file);
    
    float *X, *y;
    int num_samples;
    load_csv(data_file, &X, &y, &num_samples, 22, 4);
    
    printf("Training data loaded: %d samples\n", num_samples);
    
    const int input_dim = 22;   // State + target
    const int hidden_dim = 512;
    const int output_dim = 4;   // Motor commands
    const int batch_size = num_samples;
    
    Net* net = init_net(input_dim, hidden_dim, output_dim, batch_size);
    
    const int num_epochs = 35000;
    const float learning_rate = 0.001f;
    
    printf("Starting training for %d epochs...\n", num_epochs);
    
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
    
    net->batch_size = 1;
    save_model(net, model_file);
    
    free(X);
    free(y);
    free_net(net);
}

int main() {
    srand(time(NULL) ^ getpid());
    
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