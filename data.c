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
    
    // Write header for dynamics data
    // Input: current state (26) + motor commands (4)
    fprintf(f_dyn, "w1,w2,w3,w4,"); // Motor speeds (4)
    fprintf(f_dyn, "px,py,pz,"); // Position (3)
    fprintf(f_dyn, "vx,vy,vz,"); // Velocity (3)
    fprintf(f_dyn, "wx,wy,wz,"); // Angular velocity (3)
    fprintf(f_dyn, "r11,r12,r13,r21,r22,r23,r31,r32,r33,"); // Rotation matrix (9)
    fprintf(f_dyn, "ax_bias,ay_bias,az_bias,"); // Accel bias (3)
    fprintf(f_dyn, "gx_bias,gy_bias,gz_bias,"); // Gyro bias (3)
    fprintf(f_dyn, "w_next1,w_next2,w_next3,w_next4,"); // Motor commands (4)
    fprintf(f_dyn, "ab_noise1,ab_noise2,ab_noise3,"); // Accel bias noise (3)
    fprintf(f_dyn, "gb_noise1,gb_noise2,gb_noise3,"); // Gyro bias noise (3)
    fprintf(f_dyn, "am_noise1,am_noise2,am_noise3,"); // Accel measurement noise (3)
    fprintf(f_dyn, "gm_noise1,gm_noise2,gm_noise3,"); // Gyro measurement noise (3)
    // Output: next state (20) - we don't predict biases/scales
    fprintf(f_dyn, "next_px,next_py,next_pz,"); // Next position (3)
    fprintf(f_dyn, "next_vx,next_vy,next_vz,"); // Next velocity (3)
    fprintf(f_dyn, "next_wx,next_wy,next_wz,"); // Next angular velocity (3)
    fprintf(f_dyn, "next_r11,next_r12,next_r13,next_r21,next_r22,next_r23,next_r31,next_r32,next_r33,"); // Next rotation (9)
    fprintf(f_dyn, "next_ax,next_ay,next_az,"); // Accel measurements (3)
    fprintf(f_dyn, "next_gx,next_gy,next_gz"); // Gyro measurements (3)
    
    // Initialize progress bar
    printf("Starting data generation for %d episodes...\n", num_episodes);
    
    time_t start_time = time(NULL);
    
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
        
        // Initialize sensor scale factors
        double accel_scale[3], gyro_scale[3];
        for(int i = 0; i < 3; i++) {
            accel_scale[i] = random_range(-0.01, 0.01);
            gyro_scale[i] = random_range(-0.01, 0.01);
        }
        
        // Create quad with random position and orientation
        Quad quad = create_quad(drone_x, drone_y, drone_z, drone_yaw, accel_scale, gyro_scale);
        
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
                
                // Generate noise terms
                double accel_bias_noise[3], gyro_bias_noise[3], accel_meas_noise[3], gyro_meas_noise[3];
                for(int j = 0; j < 3; j++) {
                    accel_bias_noise[j] = random_range(-0.0001, 0.0001);
                    gyro_bias_noise[j] = random_range(-0.0001, 0.0001);
                    accel_meas_noise[j] = random_range(-0.01, 0.01);
                    gyro_meas_noise[j] = random_range(-0.01, 0.01);
                }
                
                // Record dynamics data (input to update_quad_states)
                fprintf(f_dyn, "\n%.6f,%.6f,%.6f,%.6f,", // Motor speeds
                       quad.omega[0], quad.omega[1], quad.omega[2], quad.omega[3]);
                fprintf(f_dyn, "%.6f,%.6f,%.6f,", // Position
                       quad.linear_position_W[0], quad.linear_position_W[1], quad.linear_position_W[2]);
                fprintf(f_dyn, "%.6f,%.6f,%.6f,", // Velocity
                       quad.linear_velocity_W[0], quad.linear_velocity_W[1], quad.linear_velocity_W[2]);
                fprintf(f_dyn, "%.6f,%.6f,%.6f,", // Angular velocity
                       quad.angular_velocity_B[0], quad.angular_velocity_B[1], quad.angular_velocity_B[2]);
                
                // Rotation matrix
                for(int j = 0; j < 9; j++) {
                    fprintf(f_dyn, "%.6f,", quad.R_W_B[j]);
                }
                
                // Biases
                fprintf(f_dyn, "%.6f,%.6f,%.6f,", 
                       quad.accel_bias[0], quad.accel_bias[1], quad.accel_bias[2]);
                fprintf(f_dyn, "%.6f,%.6f,%.6f,", 
                       quad.gyro_bias[0], quad.gyro_bias[1], quad.gyro_bias[2]);
                
                // Motor commands
                fprintf(f_dyn, "%.6f,%.6f,%.6f,%.6f,", 
                       quad.omega_next[0], quad.omega_next[1], quad.omega_next[2], quad.omega_next[3]);
                
                // Noise terms
                for(int j = 0; j < 3; j++) fprintf(f_dyn, "%.6f,", accel_bias_noise[j]);
                for(int j = 0; j < 3; j++) fprintf(f_dyn, "%.6f,", gyro_bias_noise[j]);
                for(int j = 0; j < 3; j++) fprintf(f_dyn, "%.6f,", accel_meas_noise[j]);
                for(int j = 0; j < 2; j++) fprintf(f_dyn, "%.6f,", gyro_meas_noise[j]);
                fprintf(f_dyn, "%.6f,", gyro_meas_noise[2]);
                
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
                    // Noise inputs
                    accel_bias_noise,           // Accelerometer bias noise
                    gyro_bias_noise,            // Gyroscope bias noise
                    accel_meas_noise,           // Accelerometer measurement noise
                    gyro_meas_noise,            // Gyroscope measurement noise
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
                
                // Record dynamics data (output of update_quad_states)
                fprintf(f_dyn, "%.6f,%.6f,%.6f,", // Next position
                       new_linear_position_W[0], new_linear_position_W[1], new_linear_position_W[2]);
                fprintf(f_dyn, "%.6f,%.6f,%.6f,", // Next velocity
                       new_linear_velocity_W[0], new_linear_velocity_W[1], new_linear_velocity_W[2]);
                fprintf(f_dyn, "%.6f,%.6f,%.6f,", // Next angular velocity
                       new_angular_velocity_B[0], new_angular_velocity_B[1], new_angular_velocity_B[2]);
                
                // Next rotation matrix
                for(int j = 0; j < 8; j++) {
                    fprintf(f_dyn, "%.6f,", new_R_W_B[j]);
                }
                fprintf(f_dyn, "%.6f,", new_R_W_B[8]);
                
                // Sensor measurements
                fprintf(f_dyn, "%.6f,%.6f,%.6f,", 
                       accel_measurement[0], accel_measurement[1], accel_measurement[2]);
                fprintf(f_dyn, "%.6f,%.6f,%.6f", 
                       gyro_measurement[0], gyro_measurement[1], gyro_measurement[2]);
                
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
    
    time_t end_time = time(NULL);
    int total_seconds = (int)difftime(end_time, start_time);
    int mins = total_seconds / 60;
    int secs = total_seconds % 60;
    
    printf("\nGeneration complete!\n");
    printf("Generated %d episodes in %02d:%02d\n", num_episodes, mins, secs);
    printf("Wrote %d control samples (%.2f MB)\n", 
           num_episodes * (int)(SIM_TIME / DT_CONTROL),
           (float)ftell(f_data) / (1024 * 1024));
    printf("Wrote %d dynamics samples (%.2f MB)\n", 
           num_episodes * (int)(SIM_TIME / DT_PHYSICS),
           (float)ftell(f_dyn) / (1024 * 1024));
    
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
    
    printf("Controller data saved to: %s\n", data_fname);
    printf("Dynamics data saved to: %s\n", dynamics_fname);
    
    return 0;
}