#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include "sim/quad.h"
#include "sim/raytracer/scene.h"

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define DT_RENDER   (1.0 / 24.0)
#define SIM_TIME    10.0  // 10 seconds per episode
#define PROGRESS_BAR_WIDTH 50

// Helper function to get random value in range [min, max]
double random_range(double min, double max) {
    return min + (double)rand() / RAND_MAX * (max - min);
}

// Helper function to calculate the angle between two points in the XZ plane
double calculate_yaw_to_target(double x1, double z1, double x2, double z2) {
    // Calculate direction vector from (x1,z1) to (x2,z2)
    double dx = x2 - x1;
    double dz = z2 - z1;
    
    // Compute angle (atan2 returns angle in range [-π, π])
    return atan2(dx, dz);
}

// Function to convert raw FPV RGB image data to grayscale values
void convert_to_grayscale(unsigned char* fpv_frame, float* grayscale_pixels, int width, int height, int channels) {
    int total_pixels = width * height;
    
    if (fpv_frame == NULL) {
        // If frame is NULL, set all pixels to black (0.0)
        for (int i = 0; i < total_pixels; i++) {
            grayscale_pixels[i] = 0.0f;
        }
        return;
    }
    
    for (int i = 0; i < total_pixels; i++) {
        // Get RGB values
        float r = fpv_frame[i * channels] / 255.0f;
        float g = fpv_frame[i * channels + 1] / 255.0f;
        float b = fpv_frame[i * channels + 2] / 255.0f;
        
        // Convert to grayscale using standard luminance formula
        grayscale_pixels[i] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

// Function to print a progress bar
void print_progress_bar(int current, int total, const char* phase) {
    double percentage = (double)current / (double)total;
    if (percentage > 1.0) percentage = 1.0; // Cap at 100%
    
    int filled_width = (int)(PROGRESS_BAR_WIDTH * percentage);
    
    printf("\r%s: [", phase);
    for (int i = 0; i < PROGRESS_BAR_WIDTH; ++i) {
        if (i < filled_width) printf("=");
        else printf(" ");
    }
    printf("] %.1f%% (%d/%d)", percentage * 100, current, total);
    fflush(stdout);
}

int main() {
    srand(time(NULL) ^ getpid());
    
    // Generate timestamped filename
    char data_fname[64];
    time_t now = time(NULL);
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", localtime(&now));
    
    // Number of episodes for training
    int num_episodes = 2000;
    
    FILE* f_data = fopen(data_fname, "w");
    if (!f_data) {
        printf("Error opening file: %s\n", data_fname);
        return 1;
    }
    
    // Define constants for the FPV rendering
    const int fpv_width = 32;
    const int fpv_height = 16;
    const int fpv_channels = 3;
    const int fpv_pixels = fpv_width * fpv_height;
    
    // Write header: Visual grayscale pixels, IMU measurements, motor commands
    fprintf(f_data, "pix1");
    for (int i = 2; i <= fpv_pixels; i++) {
        fprintf(f_data, ",pix%d", i);
    }
    fprintf(f_data, ",gx,gy,gz,ax,ay,az,"); // IMU measurements (6)
    fprintf(f_data, "m1,m2,m3,m4"); // Output motor commands (4)
    
    printf("Generating drone flight data with %d episodes...\n", num_episodes);
    
    // Set up rendering scene for FPV
    Scene fpv_scene = create_scene(fpv_width, fpv_height, (int)(SIM_TIME * 1000), 24, 1.0f);
    
    // Set up lighting for scene
    set_scene_light(&fpv_scene,
        (Vec3){1.0f, 1.0f, -1.0f},
        (Vec3){1.4f, 1.4f, 1.4f}
    );
    
    // Create meshes - no drone mesh for FPV
    Mesh ground = create_mesh("sim/raytracer/ground.obj", "sim/raytracer/ground.webp");
    Mesh treasure = create_mesh("sim/raytracer/treasure.obj", "sim/raytracer/treasure.webp");
    
    // Add meshes to scene (only ground and treasure)
    add_mesh_to_scene(&fpv_scene, ground);
    add_mesh_to_scene(&fpv_scene, treasure);
    
    // Buffer for grayscale pixels
    float* grayscale_pixels = (float*)calloc(fpv_pixels, sizeof(float));
    
    for (int episode = 0; episode < num_episodes; episode++) {
        // Display progress
        print_progress_bar(episode, num_episodes, "Generating episodes");
        
        // Initialize random drone position
        double drone_x = random_range(-2.0, 2.0);
        double drone_y = random_range(0.5, 2.0);
        double drone_z = random_range(-2.0, 2.0);
        
        // Initialize random drone yaw
        double drone_yaw = random_range(-M_PI, M_PI);
        
        // Calculate a random distance (between 1 and 4 units) in front of the drone
        double distance = random_range(1.0, 4.0);
        
        // Add some random deviation to make it more natural (±30° from the center of view)
        double angle_deviation = random_range(-M_PI/6, M_PI/6);  // ±30 degrees
        double adjusted_yaw = drone_yaw + angle_deviation;
        
        // Calculate the target position based on the drone's position, adjusted yaw, and distance
        double target_x = drone_x + sin(adjusted_yaw) * distance;
        double target_z = drone_z + cos(adjusted_yaw) * distance;
        
        // Keep the target within boundaries
        target_x = fmax(-2.0, fmin(2.0, target_x));
        target_z = fmax(-2.0, fmin(2.0, target_z));
        
        // Set a random target height
        double target_y = random_range(0.5, 2.5);
        
        // Calculate initial desired drone yaw to face the target
        double desired_yaw = calculate_yaw_to_target(
            drone_x,
            drone_z,
            target_x,
            target_z
        );
        
        // Create combined target array with the target position and desired drone yaw
        double target[7] = {
            target_x, target_y, target_z,    // Target position
            0.0, 0.0, 0.0,                  // Zero velocity target
            desired_yaw                     // Target yaw for the drone
        };
        
        // Initialize quadcopter with random position and yaw
        Quad quad = create_quad(drone_x, drone_y, drone_z, drone_yaw);
        
        // Initialize state estimator
        StateEstimator estimator = {
            .angular_velocity = {0.0, 0.0, 0.0},
            .gyro_bias = {0.0, 0.0, 0.0}
        };
        // Copy the quad's rotation matrix to the estimator
        memcpy(estimator.R, quad.R_W_B, 9 * sizeof(double));
        
        // Set treasure position for target with fixed yaw (0.0)
        set_mesh_position(&fpv_scene.meshes[1], (Vec3){(float)target[0], (float)target[1], (float)target[2]});
        set_mesh_rotation(&fpv_scene.meshes[1], (Vec3){0.0f, 0.0f, 0.0f});  // Fixed yaw at 0.0
        
        double t_physics = 0.0;
        double t_control = 0.0;
        double t_render = 0.0;
        
        for (int i = 0; i < (int)(SIM_TIME / DT_PHYSICS); i++) {
            if (t_physics >= DT_PHYSICS) {
                update_quad(&quad, DT_PHYSICS);
                t_physics = 0.0;
            }
            
            // Render update
            if (t_render >= DT_RENDER) {
                // Update FPV camera to match drone's position and orientation
                Vec3 pos = {
                    (float)quad.linear_position_W[0],
                    (float)quad.linear_position_W[1],
                    (float)quad.linear_position_W[2]
                };
                
                Vec3 forward = {
                    (float)quad.R_W_B[2],  // Third column
                    (float)quad.R_W_B[5],
                    (float)quad.R_W_B[8]
                };
                
                Vec3 up = {
                    (float)quad.R_W_B[1],  // Second column
                    (float)quad.R_W_B[4],
                    (float)quad.R_W_B[7]
                };
                
                // Set camera position slightly above the drone
                Vec3 camera_offset = {
                    up.x * 0.15f,
                    up.y * 0.15f,
                    up.z * 0.15f
                };
                
                Vec3 fpv_pos = {
                    pos.x + camera_offset.x,
                    pos.y + camera_offset.y,
                    pos.z + camera_offset.z
                };
                
                // Calculate look-at point (position + forward)
                Vec3 look_at = {
                    pos.x + forward.x,  // Look at point is in front of drone's position
                    pos.y + forward.y,
                    pos.z + forward.z
                };
                
                // Set FPV camera
                set_scene_camera(&fpv_scene, fpv_pos, look_at, up, 70.0f);
                
                // Render scene
                render_scene(&fpv_scene);
                
                // Advance to next frame
                next_frame(&fpv_scene);
                
                t_render = 0.0;
            }
            
            if (t_control >= DT_CONTROL) {
                // Check if we have a valid frame index
                unsigned char* frame_data = NULL;
                if (fpv_scene.current_frame > 0 && fpv_scene.current_frame <= fpv_scene.frame_count) {
                    frame_data = fpv_scene.frames[fpv_scene.current_frame - 1];
                }
                
                // Convert RGB to grayscale (handles NULL gracefully)
                convert_to_grayscale(frame_data, grayscale_pixels, fpv_width, fpv_height, fpv_channels);
                
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
                
                // Write training sample: Grayscale pixels, IMU, and motor commands
                
                // First write grayscale pixels
                fprintf(f_data, "\n%.6f", grayscale_pixels[0]);
                for (int j = 1; j < fpv_pixels; j++) {
                    fprintf(f_data, ",%.6f", grayscale_pixels[j]);
                }
                
                fprintf(f_data, ",%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,", // IMU
                       quad.gyro_measurement[0], quad.gyro_measurement[1], quad.gyro_measurement[2],
                       quad.accel_measurement[0], quad.accel_measurement[1], quad.accel_measurement[2]);
                       
                fprintf(f_data, "%.6f,%.6f,%.6f,%.6f", // Motor commands
                       quad.omega_next[0],
                       quad.omega_next[1],
                       quad.omega_next[2],
                       quad.omega_next[3]);
                       
                t_control = 0.0;
            }
            
            t_physics += DT_PHYSICS;
            t_control += DT_PHYSICS;
            t_render += DT_PHYSICS;
        }
    }
    
    printf("\nData generation complete!\n");
    printf("Generated %d episodes\n", num_episodes);
    printf("Data saved to: %s\n", data_fname);
    
    fclose(f_data);
    free(grayscale_pixels);
    
    // Clean up raytracer resources
    destroy_mesh(&ground);
    destroy_mesh(&treasure);
    destroy_scene(&fpv_scene);
    
    return 0;
}