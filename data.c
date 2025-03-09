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
void generate_data(const char* data_file, int num_episodes) {
    FILE* f_data = fopen(data_file, "w");
    if (!f_data) {
        printf("Error opening file: %s\n", data_file);
        return;
    }
    
    // Write header: IMU measurements, position, velocity, target position+yaw, motor commands
    fprintf(f_data, "gx,gy,gz,ax,ay,az,"); // IMU measurements (6)
    fprintf(f_data, "px,py,pz,vx,vy,vz,"); // Position and velocity (6)
    fprintf(f_data, "tx,ty,tz,tyaw,"); // Target (4)
    fprintf(f_data, "m1,m2,m3,m4"); // Output motor commands (4)
    
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
        double drone_y = random_range(0.5, 2.0);
        double drone_z = random_range(-2.0, 2.0);
        double drone_yaw = 0.0; // random_range(-M_PI, M_PI);
        
        // Create quad with random position and orientation
        Quad quad = create_quad(drone_x, drone_y, drone_z, drone_yaw);
        
        // Place target completely randomly
        double target_x = random_range(-2.0, 2.0);
        double target_y = random_range(0.5, 2.5);
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
                update_quad(&quad, DT_PHYSICS);
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
    
    fclose(f_data);
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
    
    // Generate timestamped filename
    char data_fname[64];
    time_t now = time(NULL);
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", localtime(&now));
    
    generate_data(data_fname, num_episodes);
    
    printf("Data saved to: %s\n", data_fname);
    
    return 0;
}