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

// Helper function to get random value in range [min, max]
double random_range(double min, double max) {
    return min + (double)rand() / RAND_MAX * (max - min);
}

// Generate training data for the SSM
void generate_data(const char* data_file, int num_episodes) {
    FILE* f_data = fopen(data_file, "w");
    if (!f_data) {
        printf("Error opening file: %s\n", data_file);
        return;
    }
    
    // Write header: IMU measurements, position, velocity, target position+yaw, motor commands
    fprintf(f_data, "gx,gy,gz,ax,ay,az,mx,my,mz,"); // IMU measurements (9) - including magnetometer
    fprintf(f_data, "px,py,pz,vx,vy,vz,"); // Position and velocity (6)
    fprintf(f_data, "tx,ty,tz,tyaw,"); // Target (4)
    fprintf(f_data, "m1,m2,m3,m4"); // Output motor commands (4)
    
    for (int episode = 0; episode < num_episodes; episode++) {
        // Initialize drone with random position and orientation
        double drone_x = random_range(-2.0, 2.0);
        double drone_y = random_range(0.5, 2.0);
        double drone_z = random_range(-2.0, 2.0);
        double drone_yaw = random_range(-M_PI, M_PI); // Random yaw
        
        // Create quad with random position and orientation
        Quad quad = create_quad(drone_x, drone_y, drone_z, drone_yaw);
        
        // Place target completely randomly
        double target_x = random_range(-2.0, 2.0);
        double target_y = random_range(0.5, 2.5);
        double target_z = random_range(-2.0, 2.0);
        double target_yaw = random_range(-M_PI, M_PI); // Random target yaw
        
        // Create target array (position, velocity, and desired yaw)
        double target[7] = {
            target_x, target_y, target_z,    // Target position
            0.0, 0.0, 0.0,                   // Zero velocity target
            target_yaw                       // Random target yaw
        };
        
        double t_physics = 0.0;
        double t_control = 0.0;
        
        for (int i = 0; i < (int)(SIM_TIME / DT_PHYSICS); i++) {
            if (t_physics >= DT_PHYSICS) {
                update_quad(&quad, DT_PHYSICS);
                t_physics = 0.0;
            }
            
            if (t_control >= DT_CONTROL) {
                // Get motor commands from geometric controller
                double new_omega[4];
                control_quad_commands(
                    quad.linear_position_W,
                    quad.linear_velocity_W,
                    quad.R_W_B,
                    quad.angular_velocity_B,
                    quad.inertia,
                    target,
                    new_omega
                );
                memcpy(quad.omega_next, new_omega, 4 * sizeof(double));
                
                // Write training sample: IMU, position, velocity, target, and motor commands
                fprintf(f_data, "\n%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,", // IMU with magnetometer
                       quad.gyro_measurement[0], quad.gyro_measurement[1], quad.gyro_measurement[2],
                       quad.accel_measurement[0], quad.accel_measurement[1], quad.accel_measurement[2],
                       quad.mag_measurement[0], quad.mag_measurement[1], quad.mag_measurement[2]);
                       
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
        
        if ((episode + 1) % 1000 == 0) {
            printf("Generated %d episodes\n", episode + 1);
        }
    }
    
    fclose(f_data);
}

int main(int argc, char* argv[]) {
    srand(time(NULL) ^ getpid());
    
    // Generate timestamped filename
    char data_fname[64];
    time_t now = time(NULL);
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", localtime(&now));
    
    // Number of episodes for training
    int num_episodes = 10000;
    
    // Command line can specify number of episodes
    if (argc > 1) {
        num_episodes = atoi(argv[1]);
    }
    
    printf("Generating training data...\n");
    printf("Output file: %s\n", data_fname);
    printf("Episodes: %d\n", num_episodes);
    
    generate_data(data_fname, num_episodes);
    
    printf("Data generation complete!\n");
    printf("Data saved to: %s\n", data_fname);
    
    return 0;
}