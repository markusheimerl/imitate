#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "sim/quad.h"
#include "sim/raytracer/scene.h"
#include "mlp/mlp.h"

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define DT_RENDER   (1.0 / 24.0)
#define SIM_TIME    5.0  // Simulation duration in seconds

double random_range(double min, double max) {
    return min + (double)rand() / RAND_MAX * (max - min);
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

    // Add gravity (in body frame)
    double gravity_W[3] = {0, -GRAVITY, 0};
    double gravity_B[3];
    multMatVec3f(R_B_W, gravity_W, gravity_B);

    // Accelerometer measures proper acceleration (reaction force)
    for(int i = 0; i < 3; i++) {
        accel_reading[i] = -accel_B[i] - gravity_B[i];
    }
}

int main(int argc, char* argv[]) {
    if(argc != 2) {
        printf("Usage: %s <policy_file>\n", argv[0]);
        return 1;
    }

    // Load policy network
    Net* policy = load_model(argv[1]);

    // Print network dimensions
    printf("Loaded policy network dimensions:\n");
    printf("Input dim: %d\n", policy->input_dim);
    printf("Hidden dim: %d\n", policy->hidden_dim);
    printf("Output dim: %d\n", policy->output_dim);
    printf("Batch size: %d\n", policy->batch_size);

    srand(time(NULL));
    
    // Initialize quadcopter
    Quad* quad = create_quad(
        random_range(-2.0, 2.0),
        random_range(0.0, 2.0),    // Always at or above ground
        random_range(-2.0, 2.0)
    );
    
    // Initialize random target position and yaw
    double target[7] = {
        random_range(-2.0, 2.0),    // x
        random_range(1.0, 3.0),     // y: Always above ground
        random_range(-2.0, 2.0),    // z
        0.0, 0.0, 0.0,              // vx, vy, vz
        random_range(0.0, 2*M_PI)   // yaw
    };
    
    printf("Target position: (%.2f, %.2f, %.2f) with yaw: %.2f rad\n", 
           target[0], target[1], target[2], target[6]);
    
    // Initialize raytracer scene
    Scene scene = create_scene(400, 300, (int)(SIM_TIME * 1000), 24, 0.4f);
    
    // Set up camera and scene
    set_scene_camera(&scene,
        (Vec3){-3.0f, 3.0f, -3.0f},
        (Vec3){0.0f, 0.0f, 0.0f},
        (Vec3){0.0f, 1.0f, 0.0f},
        60.0f
    );
    
    set_scene_light(&scene,
        (Vec3){1.0f, 1.0f, -1.0f},
        (Vec3){1.4f, 1.4f, 1.4f}
    );
    
    Mesh drone = create_mesh("sim/raytracer/drone.obj", "sim/raytracer/drone.webp");
    add_mesh_to_scene(&scene, drone);
    
    Mesh ground = create_mesh("sim/raytracer/ground.obj", "sim/raytracer/ground.webp");
    add_mesh_to_scene(&scene, ground);

    // Initialize timers
    double t_physics = 0.0;
    double t_control = 0.0;
    double t_render = 0.0;
    clock_t start_time = clock();
    
    // Preallocate batch-sized input buffer
    float* batch_input = (float*)calloc(policy->batch_size * policy->input_dim, sizeof(float));

    // Main simulation loop
    for (int t = 0; t < (int)(SIM_TIME / DT_PHYSICS); t++) {
        // Physics update
        if (t_physics >= DT_PHYSICS) {
            update_quad(quad, DT_PHYSICS);
            t_physics = 0.0;
        }
        
        // Control update
        if (t_control >= DT_CONTROL) {
            // Calculate accelerations and get sensor readings
            double linear_acceleration_W[3];
            calculate_linear_acceleration(quad, linear_acceleration_W);

            double gyro_reading[3];
            simulate_gyro(quad, gyro_reading);

            double accel_reading[3];
            simulate_accelerometer(quad, linear_acceleration_W, accel_reading);
            
            // Current position (3)
            for(int i = 0; i < 3; i++) {
                batch_input[i] = (float)quad->linear_position_W[i];
            }
            
            // Gyroscope readings (3)
            for(int i = 0; i < 3; i++) {
                batch_input[i+3] = (float)gyro_reading[i];
            }
            
            // Accelerometer readings (3)
            for(int i = 0; i < 3; i++) {
                batch_input[i+6] = (float)accel_reading[i];
            }

            // Target position and velocity (6)
            for(int i = 0; i < 6; i++) {
                batch_input[i+9] = (float)target[i];
            }
            
            // Target yaw (1)
            batch_input[15] = (float)target[6];
            
            // Forward pass through policy network
            forward_pass(policy, batch_input);
            
            // Apply predicted motor commands
            for(int i = 0; i < 4; i++) {
                quad->omega_next[i] = (double)policy->predictions[i];
            }
            
            t_control = 0.0;
        }
        
        // Render update
        if (t_render >= DT_RENDER) {
            set_mesh_position(&scene.meshes[0], 
                (Vec3){(float)quad->linear_position_W[0], 
                       (float)quad->linear_position_W[1], 
                       (float)quad->linear_position_W[2]});
            
            set_mesh_rotation(&scene.meshes[0], 
                (Vec3){
                    atan2f(quad->R_W_B[7], quad->R_W_B[8]),
                    asinf(-quad->R_W_B[6]),
                    atan2f(quad->R_W_B[3], quad->R_W_B[0])
                }
            );
            
            render_scene(&scene);
            next_frame(&scene);
            update_progress_bar((int)(t * DT_PHYSICS / DT_RENDER), (int)(SIM_TIME * 24), start_time);
            
            t_render = 0.0;
        }
        
        // Increment timers
        t_physics += DT_PHYSICS;
        t_control += DT_PHYSICS;
        t_render += DT_PHYSICS;
    }

    printf("Final position: (%.2f, %.2f, %.2f)\n", 
           quad->linear_position_W[0], quad->linear_position_W[1], quad->linear_position_W[2]);

    // Save animation
    char filename[64];
    strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_policy_flight.webp", 
             localtime(&(time_t){time(NULL)}));
    save_scene(&scene, filename);

    // Cleanup
    free(batch_input);
    destroy_scene(&scene);
    free_net(policy);
    free(quad);
    return 0;
}