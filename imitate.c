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
#define SIM_TIME    10.0  // 10 seconds per episode

void get_linear_acceleration_B(const Quad* q, double* a_B) {
    // 1. Calculate total thrust force in body frame (aligned with y-axis)
    double total_thrust = 0;
    for(int i = 0; i < 4; i++) {
        double omega_sq = q->omega[i] * fabs(q->omega[i]);
        total_thrust += K_F * omega_sq;
    }
    
    // 2. Create thrust vector in body frame [0, T/m, 0]
    double thrust_B[3] = {0, total_thrust/MASS, 0};
    
    // 3. Transform gravity from world to body frame
    double gravity_W[3] = {0, -GRAVITY, 0};
    double R_W_B_T[9];
    transpMat3f(q->R_W_B, R_W_B_T);
    double gravity_B[3];
    multMatVec3f(R_W_B_T, gravity_W, gravity_B);
    
    // 4. Sum accelerations in body frame
    for(int i = 0; i < 3; i++) {
        a_B[i] = thrust_B[i] + gravity_B[i];
    }
}

// State estimator structure
typedef struct {
    double estimated_pos[3];
    double estimated_vel[3];
    double estimated_R[9];
} StateEstimator;

// Initialize state estimator
StateEstimator* create_estimator(double initial_x, double initial_y, double initial_z, 
                                const double* initial_R) {
    StateEstimator* est = (StateEstimator*)malloc(sizeof(StateEstimator));
    est->estimated_pos[0] = initial_x;
    est->estimated_pos[1] = initial_y;
    est->estimated_pos[2] = initial_z;
    est->estimated_vel[0] = 0.0;
    est->estimated_vel[1] = 0.0;
    est->estimated_vel[2] = 0.0;
    memcpy(est->estimated_R, initial_R, 9 * sizeof(double));
    return est;
}

// Update state estimate using body acceleration
void update_state_estimate(StateEstimator* est, const Quad* quad, double dt) {
    // 1. Get acceleration in body frame
    double a_B[3];
    get_linear_acceleration_B(quad, a_B);
    
    // 2. Transform acceleration to world frame
    double a_W[3];
    multMatVec3f(est->estimated_R, a_B, a_W);
    
    // 3. Update velocity estimate using acceleration
    for(int i = 0; i < 3; i++) {
        est->estimated_vel[i] += a_W[i] * dt;
    }
    
    // 4. Update position estimate using updated velocity
    for(int i = 0; i < 3; i++) {
        est->estimated_pos[i] += est->estimated_vel[i] * dt;
    }
    
    // 5. Update rotation estimate using angular velocity
    double w_hat[9];
    so3hat(quad->angular_velocity_B, w_hat);
    
    double R_dot[9];
    multMat3f(est->estimated_R, w_hat, R_dot);
    
    double R_dot_scaled[9];
    multScalMat3f(dt, R_dot, R_dot_scaled);
    addMat3f(est->estimated_R, R_dot_scaled, est->estimated_R);
    
    // 6. Orthonormalize rotation matrix
    orthonormalize_rotation_matrix(est->estimated_R);
    
    // 7. Ground collision check
    if(est->estimated_pos[1] < 0.0) {
        est->estimated_pos[1] = 0.0;
        est->estimated_vel[1] = 0.0;
    }
}

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
    fprintf(f, "px,py,pz,vx,vy,vz,"); // Estimated position and velocity (6)
    fprintf(f, "r11,r12,r13,r21,r22,r23,r31,r32,r33,"); // Estimated rotation matrix (9)
    fprintf(f, "wx,wy,wz,"); // Angular velocity (3)
    fprintf(f, "tx,ty,tz,tyaw,"); // Target (7)
    fprintf(f, "m1,m2,m3,m4\n"); // Actions (4)
    
    for (int episode = 0; episode < num_episodes; episode++) {
        // Random initial state
        Quad* quad = create_quad(
            random_range(-2.0, 2.0),
            random_range(0.0, 2.0),    // Always at or above ground
            random_range(-2.0, 2.0)
        );
        
        // Create state estimator with same initial state
        StateEstimator* estimator = create_estimator(
            quad->linear_position_W[0],
            quad->linear_position_W[1],
            quad->linear_position_W[2],
            quad->R_W_B
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

                update_state_estimate(estimator, quad, DT_CONTROL);
                
                // Write state (using estimated states), target, and action to file
                fprintf(f, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,", // Estimated position and velocity
                       estimator->estimated_pos[0], 
                       estimator->estimated_pos[1], 
                       estimator->estimated_pos[2],
                       estimator->estimated_vel[0], 
                       estimator->estimated_vel[1], 
                       estimator->estimated_vel[2]);
                       
                fprintf(f, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,", // Estimated rotation
                       estimator->estimated_R[0], estimator->estimated_R[1], estimator->estimated_R[2],
                       estimator->estimated_R[3], estimator->estimated_R[4], estimator->estimated_R[5],
                       estimator->estimated_R[6], estimator->estimated_R[7], estimator->estimated_R[8]);
                       
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
        free(estimator);
        
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
    load_csv(data_file, &X, &y, &num_samples, 22, 4);
    
    printf("Training data loaded: %d samples\n", num_samples);
    
    // Initialize MLP
    const int input_dim = 22;   // 18 state + 4 target
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