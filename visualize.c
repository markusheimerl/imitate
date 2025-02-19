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
#define SIM_TIME    20.0  // Visualization duration in seconds

// Position estimator structure
typedef struct {
    double estimated_pos[3];
} PosEstimator;

// Initialize position estimator
PosEstimator* create_estimator(double initial_x, double initial_y, double initial_z) {
    PosEstimator* est = (PosEstimator*)malloc(sizeof(PosEstimator));
    est->estimated_pos[0] = initial_x;
    est->estimated_pos[1] = initial_y;
    est->estimated_pos[2] = initial_z;
    return est;
}

// Update position estimate using velocity
void update_position_estimate(PosEstimator* est, const double* velocity, double dt) {
    // Integrate velocity
    for (int i = 0; i < 3; i++) {
        est->estimated_pos[i] = est->estimated_pos[i] + velocity[i] * dt;
    }
}

double random_range(double min, double max) {
    return min + (double)rand() / RAND_MAX * (max - min);
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
    
    // Initialize quadcopter at center
    Quad* quad = create_quad(0.0, 1.0, 0.0);
    
    // Initialize position estimator
    PosEstimator* estimator = create_estimator(0.0, 1.0, 0.0);
    
    // Initialize random target position and yaw
    double target[7] = {
        random_range(-2.0, 2.0),    // x
        random_range(1.0, 3.0),     // y: Always above ground
        random_range(-2.0, 2.0),    // z
        0.0, 0.0, 0.0,              // vx, vy, vz
        random_range(0.0, 2*M_PI)   // yaw
    };
    
    printf("Target position: (%.2f, %.2f, %.2f) with yaw: %.2f rad\n", target[0], target[1], target[2], target[6]);
    
    // Initialize scene
    Scene scene = create_scene(400, 300, (int)(SIM_TIME * 1000), 24, 0.4f);
    
    // Set up lighting
    set_scene_light(&scene,
        (Vec3){1.0f, 1.0f, -1.0f},
        (Vec3){1.4f, 1.4f, 1.4f}
    );
    
    // Add meshes to scene
    Mesh drone = create_mesh("sim/raytracer/drone.obj", "sim/raytracer/drone.webp");
    Mesh ground = create_mesh("sim/raytracer/ground.obj", "sim/raytracer/ground.webp");
    Mesh treasure = create_mesh("sim/raytracer/treasure.obj", "sim/raytracer/treasure.webp");
    
    add_mesh_to_scene(&scene, drone);
    add_mesh_to_scene(&scene, ground);
    add_mesh_to_scene(&scene, treasure);

    // Set treasure position
    Vec3 treasure_pos = {
        (float)target[0],
        (float)target[1],
        (float)target[2]
    };
    set_mesh_position(&scene.meshes[2], treasure_pos);

    // Set up camera
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
            update_position_estimate(estimator, quad->linear_velocity_W, DT_CONTROL);
            // Use estimated position instead of true position
            for(int i = 0; i < 3; i++) batch_input[i] = (float)estimator->estimated_pos[i];
            for(int i = 0; i < 3; i++) batch_input[i+3] = (float)quad->linear_velocity_W[i];
            for(int i = 0; i < 9; i++) batch_input[i+6] = (float)quad->R_W_B[i];
            for(int i = 0; i < 3; i++) batch_input[i+15] = (float)quad->angular_velocity_B[i];
            for(int i = 0; i < 3; i++) batch_input[i+18] = (float)target[i];
            batch_input[21] = (float)target[6];
            
            // Forward pass through policy network
            forward_pass(policy, batch_input);
            
            // Apply predicted motor commands
            for (int i = 0; i < 4; i++) {
                quad->omega_next[i] = (double)policy->predictions[i];
            }
            
            t_control = 0.0;
        }
        
        // Render update
        if (t_render >= DT_RENDER) {
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

    printf("\nTotal targets reached: %d\n", target_count);
    printf("Final estimated position: (%.2f, %.2f, %.2f)\n",
           estimator->estimated_pos[0], estimator->estimated_pos[1], estimator->estimated_pos[2]);
    printf("Final true position: (%.2f, %.2f, %.2f)\n",
           quad->linear_position_W[0], quad->linear_position_W[1], quad->linear_position_W[2]);
    printf("Final yaw: %.2f rad\n", 
           fmod(atan2(quad->R_W_B[3], quad->R_W_B[0]) + 2 * M_PI, 2 * M_PI));
    
    // Save animation
    char filename[64];
    time_t now = time(NULL);
    strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_policy_flight.webp", localtime(&now));  
    save_scene(&scene, filename);

    // Cleanup
    free(batch_input);
    free(estimator);
    destroy_scene(&scene);
    free_net(policy);
    free(quad);
    return 0;
}