#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "sim/quad.h"

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define SIM_TIME    10.0  // 10 seconds per episode
#define PROGRESS_BAR_WIDTH 50

// Helper function to get random value in range [min, max]
double random_range(double min, double max) {
    return min + (double)rand() / RAND_MAX * (max - min);
}  

// Print a progress bar
void print_progress(int current, int total) {
    double percentage = (double)current / (double)total;
    percentage = percentage > 1.0 ? 1.0 : percentage;
    
    int filled_width = (int)(PROGRESS_BAR_WIDTH * percentage);
    
    // Calculate ETA
    static time_t start_time = 0;
    if (start_time == 0) start_time = time(NULL);
    
    time_t current_time = time(NULL);
    int elapsed_seconds = (int)difftime(current_time, start_time);
    
    int eta_seconds = (elapsed_seconds / percentage) * (1.0 - percentage);
    
    // Format time as mm:ss
    int eta_mins = eta_seconds / 60;
    int eta_secs = eta_seconds % 60;
    
    printf("\rGenerating: [");
    for (int i = 0; i < PROGRESS_BAR_WIDTH; ++i) {
        printf(i < filled_width ? "=" : " ");
    }
    printf("] %3.1f%% (%d/%d) ETA: %02d:%02d", 
           percentage * 100, current, total,
           eta_mins, eta_secs);
    
    fflush(stdout);
}

// Generate training data for the SSM
void generate_data(const char* data_file, const char* dynamics_file, int num_episodes) {
    FILE* f_data = fopen(data_file, "w");
    if (!f_data) {
        printf("Error opening file: %s\n", data_file);
        return;
    }
    
    FILE* f_dyn = fopen(dynamics_file, "w");
    if (!f_dyn) {
        printf("Error opening file: %s\n", dynamics_file);
        fclose(f_data);
        return;
    }
    
    // Write header for controller data: IMU measurements, position, velocity, target position+yaw, motor commands
    fprintf(f_data, "gx,gy,gz,ax,ay,az,"); // IMU measurements (6)
    fprintf(f_data, "px,py,pz,vx,vy,vz,"); // Position and velocity (6)
    fprintf(f_data, "tx,ty,tz,tyaw,"); // Target (4)
    fprintf(f_data, "m1,m2,m3,m4"); // Output motor commands (4)
    
    // Write header for dynamics data - ALL inputs and outputs of update_quad_states function
    // INPUTS
    fprintf(f_dyn, "omega1,omega2,omega3,omega4,"); // Current rotor speeds (4)
    fprintf(f_dyn, "pos_x,pos_y,pos_z,"); // Current position (3)
    fprintf(f_dyn, "vel_x,vel_y,vel_z,"); // Current velocity (3)
    fprintf(f_dyn, "ang_vel_x,ang_vel_y,ang_vel_z,"); // Current angular velocity (3)
    
    // Rotation matrix (9)
    fprintf(f_dyn, "r11,r12,r13,r21,r22,r23,r31,r32,r33,");
    
    // Inertia (3)
    fprintf(f_dyn, "inertia_x,inertia_y,inertia_z,");
    
    // Sensor biases and scales (12)
    fprintf(f_dyn, "accel_bias_x,accel_bias_y,accel_bias_z,");
    fprintf(f_dyn, "gyro_bias_x,gyro_bias_y,gyro_bias_z,");
    fprintf(f_dyn, "accel_scale_x,accel_scale_y,accel_scale_z,");
    fprintf(f_dyn, "gyro_scale_x,gyro_scale_y,gyro_scale_z,");
    
    // Motor commands and time step (5)
    fprintf(f_dyn, "omega_next1,omega_next2,omega_next3,omega_next4,");
    fprintf(f_dyn, "dt,");
    
    // Random values (4)
    fprintf(f_dyn, "rand1,rand2,rand3,rand4,");
    
    // OUTPUTS
    fprintf(f_dyn, "new_pos_x,new_pos_y,new_pos_z,"); // New position (3)
    fprintf(f_dyn, "new_vel_x,new_vel_y,new_vel_z,"); // New velocity (3)
    fprintf(f_dyn, "new_ang_vel_x,new_ang_vel_y,new_ang_vel_z,"); // New angular velocity (3)
    
    // New rotation matrix (9)
    fprintf(f_dyn, "new_r11,new_r12,new_r13,new_r21,new_r22,new_r23,new_r31,new_r32,new_r33,");
    
    // Sensor measurements (6)
    fprintf(f_dyn, "accel_x,accel_y,accel_z,");
    fprintf(f_dyn, "gyro_x,gyro_y,gyro_z,");
    
    // New biases (6)
    fprintf(f_dyn, "new_accel_bias_x,new_accel_bias_y,new_accel_bias_z,");
    fprintf(f_dyn, "new_gyro_bias_x,new_gyro_bias_y,new_gyro_bias_z,");
    
    // New rotor speeds (4)
    fprintf(f_dyn, "new_omega1,new_omega2,new_omega3,new_omega4");
    
    // Initialize progress bar
    printf("Starting data generation for %d episodes...\n", num_episodes);
    
    for (int episode = 0; episode < num_episodes; episode++) {
        // Print progress every 10 episodes
        if (episode % 10 == 0 || episode == num_episodes - 1) {
            print_progress(episode + 1, num_episodes);
        }
        
        // Initialize drone with random position and orientation
        double drone_x = random_range(-2.0, 2.0);
        double drone_y = random_range(0.0, 2.0);
        double drone_z = random_range(-2.0, 2.0);
        double drone_yaw = 0.0; // random_range(-M_PI, M_PI);
        
        // Create quad with random position and orientation
        Quad quad = create_quad(drone_x, drone_y, drone_z, drone_yaw);
        
        // Place target completely randomly
        double target_x = random_range(-2.0, 2.0);
        double target_y = random_range(1.0, 3.0);
        double target_z = random_range(-2.0, 2.0);
        double target_yaw = random_range(-M_PI, M_PI);
        
        // Create target array (position, velocity, and desired yaw)
        double target[7] = {
            target_x, target_y, target_z,    // Target position
            0.0, 0.0, 0.0,                   // Zero velocity target
            target_yaw                       // Random target yaw
        };

        // Initialize state estimator
        StateEstimator estimator = {
            .R = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0},
            .angular_velocity = {0.0, 0.0, 0.0},
            .gyro_bias = {0.0, 0.0, 0.0}
        };
        
        double t_physics = 0.0;
        double t_control = 0.0;
        
        for (int i = 0; i < (int)(SIM_TIME / DT_PHYSICS); i++) {
            if (t_physics >= DT_PHYSICS) {
                double new_linear_position_W[3];
                double new_linear_velocity_W[3];
                double new_angular_velocity_B[3];
                double new_R_W_B[9];
                double accel_measurement[3];
                double gyro_measurement[3];
                double new_accel_bias[3];
                double new_gyro_bias[3];
                double new_omega[4];
                
                // Generate 4 random values
                double rand1 = (double)rand() / RAND_MAX;
                double rand2 = (double)rand() / RAND_MAX;
                double rand3 = (double)rand() / RAND_MAX;
                double rand4 = (double)rand() / RAND_MAX;
                
                // Only record dynamics data every DT_CONTROL seconds
                // to limit the size of the file
                if (t_control >= DT_CONTROL) {
                    // Record ALL inputs to update_quad_states
                    // Start a new line for each sample
                    fprintf(f_dyn, "\n");
                    
                    // Rotor speeds (4)
                    fprintf(f_dyn, "%.6f,%.6f,%.6f,%.6f,", 
                        quad.omega[0], quad.omega[1], quad.omega[2], quad.omega[3]);
                    
                    // Position and velocity (6)
                    fprintf(f_dyn, "%.6f,%.6f,%.6f,", 
                        quad.linear_position_W[0], quad.linear_position_W[1], quad.linear_position_W[2]);
                    fprintf(f_dyn, "%.6f,%.6f,%.6f,", 
                        quad.linear_velocity_W[0], quad.linear_velocity_W[1], quad.linear_velocity_W[2]);
                    
                    // Angular velocity (3)
                    fprintf(f_dyn, "%.6f,%.6f,%.6f,", 
                        quad.angular_velocity_B[0], quad.angular_velocity_B[1], quad.angular_velocity_B[2]);
                    
                    // Rotation matrix (9)
                    for(int j = 0; j < 9; j++) {
                        fprintf(f_dyn, "%.6f,", quad.R_W_B[j]);
                    }
                    
                    // Inertia (3)
                    fprintf(f_dyn, "%.6f,%.6f,%.6f,", 
                        quad.inertia[0], quad.inertia[1], quad.inertia[2]);
                    
                    // Sensor biases (6)
                    fprintf(f_dyn, "%.6f,%.6f,%.6f,", 
                        quad.accel_bias[0], quad.accel_bias[1], quad.accel_bias[2]);
                    fprintf(f_dyn, "%.6f,%.6f,%.6f,", 
                        quad.gyro_bias[0], quad.gyro_bias[1], quad.gyro_bias[2]);
                    
                    // Sensor scales (6)
                    fprintf(f_dyn, "%.6f,%.6f,%.6f,", 
                        quad.accel_scale[0], quad.accel_scale[1], quad.accel_scale[2]);
                    fprintf(f_dyn, "%.6f,%.6f,%.6f,", 
                        quad.gyro_scale[0], quad.gyro_scale[1], quad.gyro_scale[2]);
                    
                    // Target rotor speeds (4)
                    fprintf(f_dyn, "%.6f,%.6f,%.6f,%.6f,", 
                        quad.omega_next[0], quad.omega_next[1], quad.omega_next[2], quad.omega_next[3]);
                    
                    // Time step (1)
                    fprintf(f_dyn, "%.6f,", DT_PHYSICS);
                    
                    // Random values (4)
                    fprintf(f_dyn, "%.6f,%.6f,%.6f,%.6f,", rand1, rand2, rand3, rand4);
                }
                
                // Update the quad state
                update_quad_states(
                    quad.omega,                 // Current rotor speeds
                    quad.linear_position_W,     // Current position
                    quad.linear_velocity_W,     // Current velocity
                    quad.angular_velocity_B,    // Current angular velocity
                    quad.R_W_B,                 // Current rotation matrix
                    quad.inertia,               // Inertia matrix
                    quad.accel_bias,            // Current accel bias
                    quad.gyro_bias,             // Current gyro bias
                    quad.accel_scale,           // Accel scale factors
                    quad.gyro_scale,            // Gyro scale factors
                    quad.omega_next,            // Target rotor speeds
                    DT_PHYSICS,                 // Time step
                    rand1, rand2, rand3, rand4, // Random values
                    // Outputs
                    new_linear_position_W,      // New position
                    new_linear_velocity_W,      // New velocity
                    new_angular_velocity_B,     // New angular velocity
                    new_R_W_B,                  // New rotation matrix
                    accel_measurement,          // Accelerometer readings
                    gyro_measurement,           // Gyroscope readings
                    new_accel_bias,             // Updated accel bias
                    new_gyro_bias,              // Updated gyro bias
                    new_omega                   // New rotor speeds
                );
                
                if(t_control >= DT_CONTROL) {
                    // Record ALL outputs from update_quad_states
                    
                    // New position (3)
                    fprintf(f_dyn, "%.6f,%.6f,%.6f,", 
                        new_linear_position_W[0], new_linear_position_W[1], new_linear_position_W[2]);
                    
                    // New velocity (3)
                    fprintf(f_dyn, "%.6f,%.6f,%.6f,", 
                        new_linear_velocity_W[0], new_linear_velocity_W[1], new_linear_velocity_W[2]);
                    
                    // New angular velocity (3)
                    fprintf(f_dyn, "%.6f,%.6f,%.6f,", 
                        new_angular_velocity_B[0], new_angular_velocity_B[1], new_angular_velocity_B[2]);
                    
                    // New rotation matrix (9)
                    for(int j = 0; j < 8; j++) {
                        fprintf(f_dyn, "%.6f,", new_R_W_B[j]);
                    }
                    fprintf(f_dyn, "%.6f,", new_R_W_B[8]);
                    
                    // Sensor measurements (6)
                    fprintf(f_dyn, "%.6f,%.6f,%.6f,", 
                        accel_measurement[0], accel_measurement[1], accel_measurement[2]);
                    fprintf(f_dyn, "%.6f,%.6f,%.6f,", 
                        gyro_measurement[0], gyro_measurement[1], gyro_measurement[2]);
                    
                    // New biases (6)
                    fprintf(f_dyn, "%.6f,%.6f,%.6f,", 
                        new_accel_bias[0], new_accel_bias[1], new_accel_bias[2]);
                    fprintf(f_dyn, "%.6f,%.6f,%.6f,", 
                        new_gyro_bias[0], new_gyro_bias[1], new_gyro_bias[2]);
                    
                    // New rotor speeds (4)
                    fprintf(f_dyn, "%.6f,%.6f,%.6f,%.6f", 
                        new_omega[0], new_omega[1], new_omega[2], new_omega[3]);
                }
                
                // Update the quad state
                memcpy(quad.linear_position_W, new_linear_position_W, 3 * sizeof(double));
                memcpy(quad.linear_velocity_W, new_linear_velocity_W, 3 * sizeof(double));
                memcpy(quad.angular_velocity_B, new_angular_velocity_B, 3 * sizeof(double));
                memcpy(quad.R_W_B, new_R_W_B, 9 * sizeof(double));
                memcpy(quad.accel_measurement, accel_measurement, 3 * sizeof(double));
                memcpy(quad.gyro_measurement, gyro_measurement, 3 * sizeof(double));
                memcpy(quad.accel_bias, new_accel_bias, 3 * sizeof(double));
                memcpy(quad.gyro_bias, new_gyro_bias, 3 * sizeof(double));
                memcpy(quad.omega, new_omega, 4 * sizeof(double));
                
                t_physics = 0.0;
            }
            
            if (t_control >= DT_CONTROL) {
                // Update state estimator
                update_estimator(
                    quad.gyro_measurement,
                    quad.accel_measurement,
                    DT_CONTROL,
                    &estimator
                );
                
                // Get motor commands from geometric controller
                double new_omega[4];
                control_quad_commands(
                    quad.linear_position_W,
                    quad.linear_velocity_W,
                    estimator.R,
                    estimator.angular_velocity,
                    quad.inertia,
                    target,
                    new_omega
                );
                memcpy(quad.omega_next, new_omega, 4 * sizeof(double));
                
                // Write training sample: IMU, position, velocity, target, and motor commands
                fprintf(f_data, "\n%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,", // IMU
                       quad.gyro_measurement[0], quad.gyro_measurement[1], quad.gyro_measurement[2],
                       quad.accel_measurement[0], quad.accel_measurement[1], quad.accel_measurement[2]);
                       
                fprintf(f_data, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,", // Position and velocity
                       quad.linear_position_W[0], quad.linear_position_W[1], quad.linear_position_W[2],
                       quad.linear_velocity_W[0], quad.linear_velocity_W[1], quad.linear_velocity_W[2]);
                
                fprintf(f_data, "%.6f,%.6f,%.6f,%.6f,", // Target
                       target[0], target[1], target[2], target[6]);
                       
                fprintf(f_data, "%.6f,%.6f,%.6f,%.6f", // Motor commands
                       quad.omega_next[0],
                       quad.omega_next[1],
                       quad.omega_next[2],
                       quad.omega_next[3]);
                
                t_control = 0.0;
            }
            
            t_physics += DT_PHYSICS;
            t_control += DT_PHYSICS;
        }
    }
    
    fclose(f_data);
    fclose(f_dyn);
}

int main(int argc, char *argv[]) {
    srand(time(NULL) ^ getpid());
    
    // Default number of episodes
    int num_episodes = 10000;
    
    // Parse command line arguments
    if (argc > 1) {
        num_episodes = atoi(argv[1]);
        if (num_episodes <= 0) {
            printf("Invalid number of episodes. Using default: 10000\n");
            num_episodes = 10000;
        }
    }
    
    // Generate timestamped filenames
    char data_fname[64], dynamics_fname[64];
    time_t now = time(NULL);
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", localtime(&now));
    strftime(dynamics_fname, sizeof(dynamics_fname), "%Y%m%d_%H%M%S_dynamics.csv", localtime(&now));
    
    generate_data(data_fname, dynamics_fname, num_episodes);
    
    printf("\nController data saved to: %s", data_fname);
    printf("\nDynamics data saved to: %s\n", dynamics_fname);
    
    return 0;
}