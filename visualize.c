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

void get_linear_acceleration_B(const Quad* q, double* a_B) {
    // 1. Calculate total thrust force in body frame (aligned with y-axis)
    double total_thrust = 0;
    for(int i = 0; i < 4; i++) {
        double omega_sq = q->omega[i] * fabs(q->omega[i]);
        total_thrust += K_F * omega_sq;
    }
    
    // 2. Create thrust vector in body frame [0, T/m, 0]
    double thrust_B[3] = {0, total_thrust/MASS, 0};
    
    // 3. Transform gravity from world to body frame
    double gravity_W[3] = {0, -GRAVITY, 0};
    double R_W_B_T[9];
    transpMat3f(q->R_W_B, R_W_B_T);
    double gravity_B[3];
    multMatVec3f(R_W_B_T, gravity_W, gravity_B);
    
    // 4. Sum accelerations in body frame
    for(int i = 0; i < 3; i++) {
        a_B[i] = thrust_B[i] + gravity_B[i];
    }
}

// State estimator structure
typedef struct {
    double estimated_pos[3];
    double estimated_vel[3];
    double estimated_R[9];
} StateEstimator;

// Initialize state estimator
StateEstimator* create_estimator(double initial_x, double initial_y, double initial_z, const double* initial_R) {
    StateEstimator* est = (StateEstimator*)malloc(sizeof(StateEstimator));
    est->estimated_pos[0] = initial_x;
    est->estimated_pos[1] = initial_y;
    est->estimated_pos[2] = initial_z;
    est->estimated_vel[0] = 0.0;
    est->estimated_vel[1] = 0.0;
    est->estimated_vel[2] = 0.0;
    memcpy(est->estimated_R, initial_R, 9 * sizeof(double));
    return est;
}

// Update state estimate using body acceleration
void update_state_estimate(StateEstimator* est, const Quad* quad, double dt) {
    // 1. Get acceleration in body frame
    double a_B[3];
    get_linear_acceleration_B(quad, a_B);
    
    // 2. Transform acceleration to world frame
    double a_W[3];
    multMatVec3f(est->estimated_R, a_B, a_W);
    
    // 3. Update velocity estimate using acceleration
    for(int i = 0; i < 3; i++) {
        est->estimated_vel[i] += a_W[i] * dt;
    }
    
    // 4. Update position estimate using updated velocity
    for(int i = 0; i < 3; i++) {
        est->estimated_pos[i] += est->estimated_vel[i] * dt;
    }
    
    // 5. Update rotation estimate using angular velocity
    double w_hat[9];
    so3hat(quad->angular_velocity_B, w_hat);
    
    double R_dot[9];
    multMat3f(est->estimated_R, w_hat, R_dot);
    
    double R_dot_scaled[9];
    multScalMat3f(dt, R_dot, R_dot_scaled);
    addMat3f(est->estimated_R, R_dot_scaled, est->estimated_R);
    
    // 6. Orthonormalize rotation matrix
    orthonormalize_rotation_matrix(est->estimated_R);
    
    // 7. Ground collision check
    if(est->estimated_pos[1] < 0.0) {
        est->estimated_pos[1] = 0.0;
        est->estimated_vel[1] = 0.0;
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
            update_state_estimate(estimator, quad, DT_CONTROL);

            // Prepare network input using estimated states
            for(int i = 0; i < 3; i++) batch_input[i] = (float)estimator->estimated_pos[i];
            for(int i = 0; i < 3; i++) batch_input[i+3] = (float)estimator->estimated_vel[i];
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
            // Use estimated position for visualization
            Vec3 pos = {
                (float)estimator->estimated_pos[0],
                (float)estimator->estimated_pos[1],
                (float)estimator->estimated_pos[2]
            };
            
            // Calculate Euler angles from estimated rotation matrix
            Vec3 rot = {
                atan2f(estimator->estimated_R[7], estimator->estimated_R[8]),
                asinf(-estimator->estimated_R[6]),
                atan2f(estimator->estimated_R[3], estimator->estimated_R[0])
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

    // Print final states (both true and estimated)
    printf("\nTrue final position: (%.2f, %.2f, %.2f)\n", 
           quad->linear_position_W[0], quad->linear_position_W[1], quad->linear_position_W[2]);
           
    printf("Estimated final position: (%.2f, %.2f, %.2f)\n",
           estimator->estimated_pos[0], estimator->estimated_pos[1], estimator->estimated_pos[2]);
           
    printf("Position estimation error: (%.2f, %.2f, %.2f)\n",
           fabs(quad->linear_position_W[0] - estimator->estimated_pos[0]),
           fabs(quad->linear_position_W[1] - estimator->estimated_pos[1]),
           fabs(quad->linear_position_W[2] - estimator->estimated_pos[2]));
           
    printf("True final velocity: (%.2f, %.2f, %.2f)\n",
           quad->linear_velocity_W[0], quad->linear_velocity_W[1], quad->linear_velocity_W[2]);
           
    printf("Estimated final velocity: (%.2f, %.2f, %.2f)\n",
           estimator->estimated_vel[0], estimator->estimated_vel[1], estimator->estimated_vel[2]);

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