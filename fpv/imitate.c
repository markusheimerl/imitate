#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "../sim/quad.h"
#include "../mlp/gpu/mlp.h"
#include "../mlp/data.h"
#include "../sim/raytracer/scene.h"

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define SIM_TIME    5.0  // 5 seconds per episode
#define IMAGE_WIDTH 8   // Small image for input
#define IMAGE_HEIGHT 6
#define IMAGE_PIXELS (IMAGE_WIDTH * IMAGE_HEIGHT * 3)

double random_range(double min, double max) {
    return min + (double)rand() / RAND_MAX * (max - min);
}

double calculate_target_yaw(double current_x, double current_z, double target_x, double target_z) {
    return atan2(target_z - current_z, target_x - current_x) + M_PI;
}

void generate_training_data(const char* filename, int num_episodes) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        printf("Error opening file: %s\n", filename);
        return;
    }
    
    // Write header
    for(int i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; i++) {
        fprintf(f, "r%d,g%d,b%d,", i, i, i);
    }
    fprintf(f, "px,py,pz,vx,vy,vz,"); // Position and velocity (6)
    fprintf(f, "r11,r12,r13,r21,r22,r23,r31,r32,r33,"); // Rotation matrix (9)
    fprintf(f, "wx,wy,wz,"); // Angular velocity (3)
    fprintf(f, "m1,m2,m3,m4\n"); // Actions (4)
    
    // Initialize scene
    Scene scene = create_scene(IMAGE_WIDTH, IMAGE_HEIGHT, 1, 24, 1.0f);
    set_scene_light(&scene, (Vec3){1.0f, 1.0f, -1.0f}, (Vec3){1.4f, 1.4f, 1.4f});
    
    // Add meshes
    Mesh drone = create_mesh("../sim/raytracer/drone.obj", "../sim/raytracer/drone.webp");
    Mesh ground = create_mesh("../sim/raytracer/ground.obj", "../sim/raytracer/ground.webp");
    Mesh treasure = create_mesh("../sim/raytracer/treasure.obj", "../sim/raytracer/treasure.webp");
    
    add_mesh_to_scene(&scene, drone);
    add_mesh_to_scene(&scene, ground);
    add_mesh_to_scene(&scene, treasure);
    
    for (int episode = 0; episode < num_episodes; episode++) {
        Quad* quad = create_quad(
            random_range(-2.0, 2.0),
            random_range(0.0, 2.0),
            random_range(-2.0, 2.0)
        );
        
        // Random target (y always positive)
        double target[7] = {
            random_range(-2.0, 2.0),    // x
            random_range(1.0, 3.0),     // y: Always above ground
            random_range(-2.0, 2.0),    // z
            0.0, 0.0, 0.0,              // vx, vy, vz
            0.0                         // yaw (will be calculated)
        };

        // Set treasure position
        Vec3 treasure_pos = {
            (float)target[0], 
            (float)target[1], 
            (float)target[2]
        };
        set_mesh_position(&scene.meshes[2], treasure_pos);
        
        double t_physics = 0.0;
        double t_control = 0.0;
        
        for(int i = 0; i < (int)(SIM_TIME / DT_PHYSICS); i++) {
            if (t_physics >= DT_PHYSICS) {
                update_quad(quad, DT_PHYSICS);
                t_physics = 0.0;
            }
            
            if (t_control >= DT_CONTROL) {
                // Calculate yaw to face target
                target[6] = calculate_target_yaw(
                    quad->linear_position_W[0],
                    quad->linear_position_W[2],
                    target[0],
                    target[2]
                );
                
                // Update drone in scene
                Vec3 pos = {
                    (float)quad->linear_position_W[0],
                    (float)quad->linear_position_W[1],
                    (float)quad->linear_position_W[2]
                };
                Vec3 rot = {
                    atan2f(quad->R_W_B[7], quad->R_W_B[8]),
                    asinf(-quad->R_W_B[6]),
                    atan2f(quad->R_W_B[3], quad->R_W_B[0])
                };
                set_mesh_position(&scene.meshes[0], pos);
                set_mesh_rotation(&scene.meshes[0], rot);

                // Update camera for FPV
                Vec3 cam_offset = {0.0f, 0.2f, 0.0f};
                Vec3 rotated_offset = {
                    cam_offset.x * (float)quad->R_W_B[0] + cam_offset.y * (float)quad->R_W_B[1] + cam_offset.z * (float)quad->R_W_B[2],
                    cam_offset.x * (float)quad->R_W_B[3] + cam_offset.y * (float)quad->R_W_B[4] + cam_offset.z * (float)quad->R_W_B[5],
                    cam_offset.x * (float)quad->R_W_B[6] + cam_offset.y * (float)quad->R_W_B[7] + cam_offset.z * (float)quad->R_W_B[8]
                };

                Vec3 cam_pos = {
                    pos.x + rotated_offset.x,
                    pos.y + rotated_offset.y,
                    pos.z + rotated_offset.z
                };

                Vec3 forward = {0.0f, 0.0f, 1.0f};
                Vec3 rotated_forward = {
                    forward.x * (float)quad->R_W_B[0] + forward.y * (float)quad->R_W_B[1] + forward.z * (float)quad->R_W_B[2],
                    forward.x * (float)quad->R_W_B[3] + forward.y * (float)quad->R_W_B[4] + forward.z * (float)quad->R_W_B[5],
                    forward.x * (float)quad->R_W_B[6] + forward.y * (float)quad->R_W_B[7] + forward.z * (float)quad->R_W_B[8]
                };

                Vec3 look_at = {
                    cam_pos.x + rotated_forward.x,
                    cam_pos.y + rotated_forward.y,
                    cam_pos.z + rotated_forward.z
                };

                Vec3 up = {0.0f, 1.0f, 0.0f};
                Vec3 rotated_up = {
                    up.x * (float)quad->R_W_B[0] + up.y * (float)quad->R_W_B[1] + up.z * (float)quad->R_W_B[2],
                    up.x * (float)quad->R_W_B[3] + up.y * (float)quad->R_W_B[4] + up.z * (float)quad->R_W_B[5],
                    up.x * (float)quad->R_W_B[6] + up.y * (float)quad->R_W_B[7] + up.z * (float)quad->R_W_B[8]
                };

                set_scene_camera(&scene, cam_pos, look_at, rotated_up, 70.0f);

                // Render scene
                render_scene(&scene);

                // Get motor commands from geometric controller
                control_quad(quad, target);

                // Write normalized image pixels
                unsigned char* frame = scene.frames[scene.current_frame];
                for(int p = 0; p < IMAGE_PIXELS; p++) {
                    fprintf(f, "%.6f,", frame[p] / 255.0f);
                }
                
                // Write state data
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
    
    destroy_scene(&scene);
    fclose(f);
}

void train_policy(const char* data_file, const char* model_file) {
    printf("Loading training data from %s...\n", data_file);
    
    float *X = NULL, *y = NULL;
    int num_samples;
    load_csv(data_file, &X, &y, &num_samples, IMAGE_PIXELS + 18, 4);
    
    if (!X || !y) {
        printf("Failed to load training data\n");
        return;
    }
    
    printf("Training data loaded: %d samples\n", num_samples);
    
    // Initialize MLP with smaller hidden layer
    const int input_dim = IMAGE_PIXELS + 18;  // Image pixels + state dimensions
    const int hidden_dim = 128;  // Reduced from 512
    const int output_dim = 4;   // 4 motor commands
    const int batch_size = 256; // Using smaller batches
    
    Net* net = init_net(input_dim, hidden_dim, output_dim, batch_size);
    if (!net) {
        printf("Failed to initialize network\n");
        free(X);
        free(y);
        return;
    }
    
    printf("Network initialized with dimensions:\n");
    printf("Input: %d\n", input_dim);
    printf("Hidden: %d\n", hidden_dim);
    printf("Output: %d\n", output_dim);
    printf("Batch size: %d\n", batch_size);
    
    // Training parameters
    const int num_epochs = 1000;  // Reduced from 25000
    const float learning_rate = 0.001f;
    
    printf("Starting training for %d epochs...\n", num_epochs);
    
    // Training loop with batch processing
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        int num_batches = (num_samples + batch_size - 1) / batch_size;
        
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * batch_size;
            int current_batch_size = (batch == num_batches - 1) ? 
                (num_samples - start_idx) : batch_size;
            
            forward_pass(net, &X[start_idx * input_dim]);
            float loss = calculate_loss(net, &y[start_idx * output_dim]);
            epoch_loss += loss;
            
            zero_gradients(net);
            backward_pass(net, &X[start_idx * input_dim]);
            update_weights(net, learning_rate);
        }
        
        epoch_loss /= num_batches;
        
        if (epoch == 0 || (epoch + 1) % 100 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", 
                   epoch + 1, num_epochs, epoch_loss);
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
    generate_training_data(data_fname, 200);
    
    printf("Phase 2: Training policy network...\n");
    train_policy(data_fname, model_fname);
    
    printf("Training complete!\n");
    printf("Data saved to: %s\n", data_fname);
    printf("Model saved to: %s\n", model_fname);
    
    return 0;
}