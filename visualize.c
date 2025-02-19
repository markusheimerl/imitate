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
#define SIM_TIME    10.0  // Simulation duration in seconds

// State estimator structure
typedef struct {
    double estimated_pos[3];
    double estimated_R[9];
} StateEstimator;

// Initialize state estimator
StateEstimator* create_estimator(double initial_x, double initial_y, double initial_z, const double* initial_R) {
    StateEstimator* est = (StateEstimator*)malloc(sizeof(StateEstimator));
    est->estimated_pos[0] = initial_x;
    est->estimated_pos[1] = initial_y;
    est->estimated_pos[2] = initial_z;
    memcpy(est->estimated_R, initial_R, 9 * sizeof(double));
    return est;
}

// Update state estimate using velocities
void update_state_estimate(StateEstimator* est, const double* linear_velocity, 
                         const double* angular_velocity, double dt) {
    // Integrate linear velocity for position
    for (int i = 0; i < 3; i++) {
        est->estimated_pos[i] = est->estimated_pos[i] + linear_velocity[i] * dt;
    }
    
    // Integrate angular velocity for rotation matrix
    // Ṙ = R[ω]ₓ
    double w_hat[9];
    so3hat(angular_velocity, w_hat);
    
    // Calculate R_dot = R[ω]ₓ
    double R_dot[9];
    multMat3f(est->estimated_R, w_hat, R_dot);
    
    // Euler integration: R(t+dt) = R(t) + dt * R_dot
    double R_dot_scaled[9];
    multScalMat3f(dt, R_dot, R_dot_scaled);
    addMat3f(est->estimated_R, R_dot_scaled, est->estimated_R);
    
    // Orthonormalize to prevent drift
    orthonormalize_rotation_matrix(est->estimated_R);
}

double random_range(double min, double max) {
    return min + (double)rand() / RAND_MAX * (max - min);
}

int main(int argc, char* argv[]) {
    if(argc != 2) {
        printf("Usage: %s <policy_file>\n", argv[0]);
        return 1;
    }

    Net* policy = load_model(argv[1]);

    printf("Loaded policy network dimensions:\n");
    printf("Input dim: %d\n", policy->input_dim);
    printf("Hidden dim: %d\n", policy->hidden_dim);
    printf("Output dim: %d\n", policy->output_dim);
    printf("Batch size: %d\n", policy->batch_size);

    srand(time(NULL));
    
    Quad* quad = create_quad(
        random_range(-2.0, 2.0),
        random_range(0.0, 2.0),
        random_range(-2.0, 2.0)
    );
    
    StateEstimator* estimator = create_estimator(
        quad->linear_position_W[0],
        quad->linear_position_W[1],
        quad->linear_position_W[2],
        quad->R_W_B
    );
    
    double target[7] = {
        random_range(-2.0, 2.0),
        random_range(1.0, 3.0),
        random_range(-2.0, 2.0),
        0.0, 0.0, 0.0,
        random_range(0.0, 2*M_PI)
    };
    
    printf("Target position: (%.2f, %.2f, %.2f) with yaw: %.2f rad\n", 
           target[0], target[1], target[2], target[6]);
    
    Scene scene = create_scene(400, 300, (int)(SIM_TIME * 1000), 24, 0.4f);
    
    set_scene_light(&scene,
        (Vec3){1.0f, 1.0f, -1.0f},
        (Vec3){1.4f, 1.4f, 1.4f}
    );
    
    Mesh drone = create_mesh("sim/raytracer/drone.obj", "sim/raytracer/drone.webp");
    Mesh ground = create_mesh("sim/raytracer/ground.obj", "sim/raytracer/ground.webp");
    Mesh treasure = create_mesh("sim/raytracer/treasure.obj", "sim/raytracer/treasure.webp");
    
    add_mesh_to_scene(&scene, drone);
    add_mesh_to_scene(&scene, ground);
    add_mesh_to_scene(&scene, treasure);

    Vec3 treasure_pos = {
        (float)target[0],
        (float)target[1],
        (float)target[2]
    };
    set_mesh_position(&scene.meshes[2], treasure_pos);

    set_scene_camera(&scene,
        (Vec3){-3.0f, 3.0f, -3.0f},
        (Vec3){0.0f, 0.0f, 0.0f},
        (Vec3){0.0f, 1.0f, 0.0f},
        60.0f
    );

    double t_physics = 0.0;
    double t_control = 0.0;
    double t_render = 0.0;
    clock_t start_time = clock();
    
    float* batch_input = (float*)calloc(policy->batch_size * policy->input_dim, sizeof(float));

    for (int t = 0; t < (int)(SIM_TIME / DT_PHYSICS); t++) {
        if (t_physics >= DT_PHYSICS) {
            update_quad(quad, DT_PHYSICS);
            t_physics = 0.0;
        }
        
        if (t_control >= DT_CONTROL) {
            update_state_estimate(estimator, quad->linear_velocity_W, 
                                quad->angular_velocity_B, DT_CONTROL);

            // Use estimated states instead of actual states
            for(int i = 0; i < 3; i++) batch_input[i] = (float)estimator->estimated_pos[i];
            for(int i = 0; i < 3; i++) batch_input[i+3] = (float)quad->linear_velocity_W[i];
            for(int i = 0; i < 9; i++) batch_input[i+6] = (float)estimator->estimated_R[i];
            for(int i = 0; i < 3; i++) batch_input[i+15] = (float)quad->angular_velocity_B[i];
            for(int i = 0; i < 3; i++) batch_input[i+18] = (float)target[i];
            batch_input[21] = (float)target[6];
            
            forward_pass(policy, batch_input);
            
            for (int i = 0; i < 4; i++) {
                quad->omega_next[i] = (double)policy->predictions[i];
            }
            
            t_control = 0.0;
        }
        
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
        
        t_physics += DT_PHYSICS;
        t_control += DT_PHYSICS;
        t_render += DT_PHYSICS;
    }

    printf("\nFinal position: (%.2f, %.2f, %.2f) with yaw: %.2f rad\n", 
           quad->linear_position_W[0], quad->linear_position_W[1], quad->linear_position_W[2], 
           fmod(atan2(quad->R_W_B[3], quad->R_W_B[0]) + 2 * M_PI, 2 * M_PI));
    
    printf("Final estimated position: (%.2f, %.2f, %.2f)\n",
           estimator->estimated_pos[0], estimator->estimated_pos[1], estimator->estimated_pos[2]);
    
    printf("Final estimated yaw: %.2f rad\n",
           fmod(atan2(estimator->estimated_R[3], estimator->estimated_R[0]) + 2 * M_PI, 2 * M_PI));

    char filename[64];
    time_t now = time(NULL);
    strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_policy_flight.webp", localtime(&now));  
    save_scene(&scene, filename);

    free(batch_input);
    free(estimator);
    destroy_scene(&scene);
    free_net(policy);
    free(quad);
    return 0;
}