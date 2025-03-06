#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "sim/quad.h"
#include "sim/raytracer/scene.h"
#include "ssm/ssm.h"  // Using CPU version

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define DT_RENDER   (1.0 / 24.0)
#define SIM_TIME    10.0  // Simulation duration in seconds

// Helper function to get random value in range [min, max]
double random_range(double min, double max) {
    return min + (double)rand() / RAND_MAX * (max - min);
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

int main(int argc, char* argv[]) {
    if(argc != 5) {
        printf("Usage: %s <layer1_model_file> <layer2_model_file> <layer3_model_file> <layer4_model_file>\n", argv[0]);
        return 1;
    }

    // Load all four SSM models
    SSM* layer1_ssm = load_ssm(argv[1], 1);
    SSM* layer2_ssm = load_ssm(argv[2], 1);
    SSM* layer3_ssm = load_ssm(argv[3], 1);
    SSM* layer4_ssm = load_ssm(argv[4], 1);

    // Print network dimensions
    printf("Loaded Layer 1 Model dimensions:\n");
    printf("Input dim: %d\n", layer1_ssm->input_dim);
    printf("State dim: %d\n", layer1_ssm->state_dim);
    printf("Output dim: %d\n", layer1_ssm->output_dim);
    printf("Batch size: %d\n", layer1_ssm->batch_size);
    
    printf("\nLoaded Layer 2 Model dimensions:\n");
    printf("Input dim: %d\n", layer2_ssm->input_dim);
    printf("State dim: %d\n", layer2_ssm->state_dim);
    printf("Output dim: %d\n", layer2_ssm->output_dim);
    printf("Batch size: %d\n", layer2_ssm->batch_size);
    
    printf("\nLoaded Layer 3 Model dimensions:\n");
    printf("Input dim: %d\n", layer3_ssm->input_dim);
    printf("State dim: %d\n", layer3_ssm->state_dim);
    printf("Output dim: %d\n", layer3_ssm->output_dim);
    printf("Batch size: %d\n", layer3_ssm->batch_size);
    
    printf("\nLoaded Layer 4 Model dimensions:\n");
    printf("Input dim: %d\n", layer4_ssm->input_dim);
    printf("State dim: %d\n", layer4_ssm->state_dim);
    printf("Output dim: %d\n", layer4_ssm->output_dim);
    printf("Batch size: %d\n", layer4_ssm->batch_size);

    srand(time(NULL));
    
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
    
    // Create target array with the target position
    double target[3] = {
        target_x, target_y, target_z    // Target position
    };
    
    printf("Initial drone position: (%.2f, %.2f, %.2f) with yaw: %.2f rad\n", 
           drone_x, drone_y, drone_z, drone_yaw);
    printf("Target position: (%.2f, %.2f, %.2f)\n", 
           target[0], target[1], target[2]);
    printf("Target is %.2f meters away in the drone's field of view (%.2f° from center)\n", 
           distance, angle_deviation * 180.0 / M_PI);
    
    // Initialize quadcopter with random position and yaw
    Quad quad = create_quad(drone_x, drone_y, drone_z, drone_yaw);
    
    // Define FPV rendering constants - increased resolution
    const int fpv_width = 32;
    const int fpv_height = 16;
    const int fpv_channels = 3;
    const int fpv_pixels = fpv_width * fpv_height;
    
    // Initialize scenes
    Scene scene = create_scene(400, 300, (int)(SIM_TIME * 1000), 24, 0.4f);
    Scene fpv_scene = create_scene(fpv_width, fpv_height, (int)(SIM_TIME * 1000), 24, 1.0f);
    
    // Set up lighting for both scenes
    set_scene_light(&scene,
        (Vec3){1.0f, 1.0f, -1.0f},
        (Vec3){1.4f, 1.4f, 1.4f}
    );
    
    set_scene_light(&fpv_scene,
        (Vec3){1.0f, 1.0f, -1.0f},
        (Vec3){1.4f, 1.4f, 1.4f}
    );
    
    // Create meshes to be shared between scenes
    Mesh drone = create_mesh("sim/raytracer/drone.obj", "sim/raytracer/drone.webp");
    Mesh ground = create_mesh("sim/raytracer/ground.obj", "sim/raytracer/ground.webp");
    Mesh treasure = create_mesh("sim/raytracer/treasure.obj", "sim/raytracer/treasure.webp");
    
    // Add meshes to third-person scene
    add_mesh_to_scene(&scene, drone);
    add_mesh_to_scene(&scene, ground);
    add_mesh_to_scene(&scene, treasure);
    
    // Add meshes to FPV scene (only ground and treasure, no drone in FPV)
    add_mesh_to_scene(&fpv_scene, ground);
    add_mesh_to_scene(&fpv_scene, treasure);

    // Set treasure position (target) for both scenes with fixed yaw (0.0)
    set_mesh_position(&scene.meshes[2], (Vec3){(float)target[0], (float)target[1], (float)target[2]});
    set_mesh_rotation(&scene.meshes[2], (Vec3){0.0f, 0.0f, 0.0f});
    
    set_mesh_position(&fpv_scene.meshes[1], (Vec3){(float)target[0], (float)target[1], (float)target[2]});
    set_mesh_rotation(&fpv_scene.meshes[1], (Vec3){0.0f, 0.0f, 0.0f});

    // Set up chase camera with 60 degree FOV
    set_scene_camera(&scene,
        (Vec3){-3.0f, 3.0f, -3.0f},
        (Vec3){0.0f, 0.0f, 0.0f},
        (Vec3){0.0f, 1.0f, 0.0f},
        60.0f
    );

    // Initialize timers
    double t_physics = 0.0;
    double t_control = 0.0;
    double t_render = 0.0;
    clock_t start_time = clock();
    
    // Allocate input buffer for layer1 model
    float* layer1_input = (float*)calloc(layer1_ssm->input_dim, sizeof(float));
    
    // Allocate intermediate buffers for connecting the models
    float* layer2_input = (float*)calloc(layer2_ssm->input_dim, sizeof(float));
    float* layer3_input = (float*)calloc(layer3_ssm->input_dim, sizeof(float));
    float* layer4_input = (float*)calloc(layer4_ssm->input_dim, sizeof(float));
    
    // Grayscale pixels buffer
    float* grayscale_pixels = (float*)calloc(fpv_pixels, sizeof(float));
    
    // Reset internal states of all models
    memset(layer1_ssm->state, 0, layer1_ssm->state_dim * sizeof(float));
    memset(layer2_ssm->state, 0, layer2_ssm->state_dim * sizeof(float));
    memset(layer3_ssm->state, 0, layer3_ssm->state_dim * sizeof(float));
    memset(layer4_ssm->state, 0, layer4_ssm->state_dim * sizeof(float));

    // Main simulation loop
    for (int t = 0; t < (int)(SIM_TIME / DT_PHYSICS); t++) {
        // Physics update
        if (t_physics >= DT_PHYSICS) {
            update_quad(&quad, DT_PHYSICS);
            t_physics = 0.0;
        }
        
        // Control update
        if (t_control >= DT_CONTROL) {
            // Check if we have a valid frame index
            unsigned char* frame_data = NULL;
            if (fpv_scene.current_frame > 0 && fpv_scene.current_frame <= fpv_scene.frame_count) {
                frame_data = fpv_scene.frames[fpv_scene.current_frame - 1];
            }
            
            // Convert RGB to grayscale
            convert_to_grayscale(frame_data, grayscale_pixels, fpv_width, fpv_height, fpv_channels);
            
            // Fill input for layer 1 model: raw grayscale pixels and IMU data only
            int idx = 0;
            
            // Grayscale pixels
            for (int i = 0; i < fpv_pixels; i++) {
                layer1_input[idx++] = grayscale_pixels[i];
            }
            
            // IMU measurements (6 values: gyro x,y,z and accel x,y,z)
            for(int i = 0; i < 3; i++) layer1_input[idx++] = (float)quad.gyro_measurement[i];
            for(int i = 0; i < 3; i++) layer1_input[idx++] = (float)quad.accel_measurement[i];
            
            // Forward pass through the four models
            forward_pass(layer1_ssm, layer1_input);
            memcpy(layer2_input, layer1_ssm->predictions, layer1_ssm->output_dim * sizeof(float));
            
            forward_pass(layer2_ssm, layer2_input);
            memcpy(layer3_input, layer2_ssm->predictions, layer2_ssm->output_dim * sizeof(float));
            
            forward_pass(layer3_ssm, layer3_input);
            memcpy(layer4_input, layer3_ssm->predictions, layer3_ssm->output_dim * sizeof(float));
            
            forward_pass(layer4_ssm, layer4_input);
            
            // Apply predicted motor commands from the layer 4 model
            for (int i = 0; i < 4; i++) {
                quad.omega_next[i] = (double)layer4_ssm->predictions[i];
            }
            
            t_control = 0.0;
        }
        
        // Render update
        if (t_render >= DT_RENDER) {
            // Get drone position and orientation for visualization
            Vec3 pos = {
                (float)quad.linear_position_W[0],
                (float)quad.linear_position_W[1],
                (float)quad.linear_position_W[2]
            };
            
            Vec3 rot = {
                atan2f(quad.R_W_B[7], quad.R_W_B[8]),
                asinf(-quad.R_W_B[6]),
                atan2f(quad.R_W_B[3], quad.R_W_B[0])
            };

            // Update drone position in third-person scene only
            set_mesh_position(&scene.meshes[0], pos);
            set_mesh_rotation(&scene.meshes[0], rot);
            
            // Update FPV camera to match drone's position and orientation
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
            
            // Render both scenes
            render_scene(&scene);
            render_scene(&fpv_scene);
            
            // Advance to next frame in both scenes
            next_frame(&scene);
            next_frame(&fpv_scene);
            
            update_progress_bar((int)(t * DT_PHYSICS / DT_RENDER), (int)(SIM_TIME * 24), start_time);
            
            t_render = 0.0;
        }
        
        // Increment timers
        t_physics += DT_PHYSICS;
        t_control += DT_PHYSICS;
        t_render += DT_PHYSICS;
    }

    printf("\nFinal position: (%.2f, %.2f, %.2f) with yaw %.2f or ±%.2f\n", 
           quad.linear_position_W[0], quad.linear_position_W[1], quad.linear_position_W[2],
           asinf(-quad.R_W_B[6]), M_PI - fabs(asinf(-quad.R_W_B[6])));
    
    // Calculate distance to target
    double dist = sqrt(pow(quad.linear_position_W[0] - target[0], 2) + 
                     pow(quad.linear_position_W[1] - target[1], 2) + 
                     pow(quad.linear_position_W[2] - target[2], 2));
    printf("Distance to target: %.2f meters\n", dist);
    
    // Save animations
    char filename[64];
    char fpv_filename[64];
    
    time_t now = time(NULL);
    strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_flight.webp", localtime(&now));
    strftime(fpv_filename, sizeof(fpv_filename), "%Y%m%d_%H%M%S_flight_fpv.webp", localtime(&now));
    
    save_scene(&scene, filename);
    save_scene(&fpv_scene, fpv_filename);
    
    printf("Third-person view saved to: %s\n", filename);
    printf("First-person view saved to: %s\n", fpv_filename);

    // Cleanup
    free(layer1_input);
    free(layer2_input);
    free(layer3_input);
    free(layer4_input);
    free(grayscale_pixels);
    
    destroy_mesh(&drone);
    destroy_mesh(&ground);
    destroy_mesh(&treasure);
    destroy_scene(&scene);
    destroy_scene(&fpv_scene);
    
    free_ssm(layer1_ssm);
    free_ssm(layer2_ssm);
    free_ssm(layer3_ssm);
    free_ssm(layer4_ssm);
    
    return 0;
}