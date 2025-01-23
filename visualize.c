#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "grad/grad.h"
#include "sim/sim.h"

#define DT_PHYSICS (1.0/1000.0)
#define DT_CONTROL (1.0/60.0)
#define DT_RENDER (1.0/30.0)

#define STATE_DIM 10  // 3 accel + 3 gyro + 3 desired vel + 1 desired yaw rate
#define HIDDEN_DIM 64
#define ACTION_DIM 8

#define MAX_STD 3.0
#define MIN_STD 1e-5

#define TASK_RADIUS 5.0
#define MIN_HEIGHT 0.5
#define MAX_HEIGHT 1.5
#define SIMULATION_DURATION 10.0

double squash(double x, double min, double max) { 
    return ((max + min) / 2.0) + ((max - min) / 2.0) * tanh(x); 
}

void get_random_position(double pos[3], double center[3], double radius) {
    // Define bounds that keep positions in front of camera at (-2, 2, -2)
    // Camera looks towards (0,0,0), so keep positions in positive x and z quadrant
    double x_min = -1.0;  // Allow some negative x for maneuverability
    double x_max = 3.0;   // But bias towards positive x
    double y_min = MIN_HEIGHT;
    double y_max = MAX_HEIGHT;
    double z_min = -1.0;  // Allow some negative z for maneuverability
    double z_max = 3.0;   // But bias towards positive z

    // Generate random position within these bounds
    pos[0] = x_min + (x_max - x_min) * ((double)rand()/RAND_MAX);
    pos[1] = y_min + (y_max - y_min) * ((double)rand()/RAND_MAX);
    pos[2] = z_min + (z_max - z_min) * ((double)rand()/RAND_MAX);

    // Ensure position is within specified radius of center
    double dx = pos[0] - center[0];
    double dy = pos[1] - center[1];
    double dz = pos[2] - center[2];
    double distance = sqrt(dx*dx + dy*dy + dz*dz);
    
    if (distance > radius) {
        // Scale back to be within radius while maintaining direction
        double scale = radius / distance;
        pos[0] = center[0] + dx * scale;
        pos[1] = center[1] + dy * scale;
        pos[2] = center[2] + dz * scale;
    }

    // Ensure height bounds are respected
    pos[1] = fmax(MIN_HEIGHT, fmin(MAX_HEIGHT, pos[1]));
}

void velocity_controller(Quad* q, double* target_pos, double* desired_vel_B, double* desired_yaw_rate) {
    // 1. Calculate position error in world frame
    double error_p[3];
    subVec3f(q->linear_position_W, target_pos, error_p);
    
    // 2. Calculate desired velocity in world frame
    double desired_vel_W[3];
    multScalVec3f(-K_P, error_p, desired_vel_W);
    
    // 3. Calculate desired heading (yaw) based on velocity direction
    double desired_heading = atan2(desired_vel_W[2], desired_vel_W[0]);
    
    // Get current heading from rotation matrix
    double current_heading = atan2(q->R_W_B[2], q->R_W_B[0]);
    
    // Calculate heading error (considering circular nature of angles)
    double heading_error = desired_heading - current_heading;
    if (heading_error > M_PI) heading_error -= 2*M_PI;
    if (heading_error < -M_PI) heading_error += 2*M_PI;
    
    // Calculate desired yaw rate (proportional control)
    *desired_yaw_rate = 2.0 * heading_error;  // You can tune this gain
    
    // 4. Transform desired velocity to body frame
    double R_B_W[9];
    transpMat3f(q->R_W_B, R_B_W);
    multMatVec3f(R_B_W, desired_vel_W, desired_vel_B);
}

void get_state(Quad* q, double* state, double* desired_vel_B, double desired_yaw_rate) {
    // Accelerometer readings (body frame)
    memcpy(state, q->linear_acceleration_B_s, 3 * sizeof(double));
    
    // Gyroscope readings (body frame)
    memcpy(state + 3, q->angular_velocity_B_s, 3 * sizeof(double));
    
    // Desired velocities (body frame)
    memcpy(state + 6, desired_vel_B, 3 * sizeof(double));
    
    // Desired yaw rate
    state[9] = desired_yaw_rate;
}

int main(int argc, char** argv) {
    if(argc != 2) {
        printf("Usage: %s <policy_weights.bin>\n", argv[0]);
        return 1;
    }

    srand(time(NULL) ^ getpid());

    // Initialize policy network
    Net* policy = load_net(argv[1]);
    if(!policy) {
        printf("Failed to load weights from %s\n", argv[1]);
        return 1;
    }

    // Initialize simulation
    Sim* sim = init_sim("sim/", true);

    // Generate start and target positions
    double start_pos[3], target_pos[3];
    double center[3] = {0, 1, 0};
    
    get_random_position(start_pos, center, TASK_RADIUS);
    get_random_position(target_pos, start_pos, TASK_RADIUS);
    
    // Reset quadcopter to start position
    reset_quad(sim->quad, start_pos[0], start_pos[1], start_pos[2]);

    // Initialize state buffer and desired velocities
    double state[STATE_DIM];
    double desired_vel_B[3] = {0};
    double desired_yaw_rate = 0.0;

    double t_physics = 0.0, t_control = 0.0, t_render = 0.0;
    
    printf("Starting visualization...\n");
    printf("Start position: (%.2f, %.2f, %.2f)\n", start_pos[0], start_pos[1], start_pos[2]);
    printf("Target position: (%.2f, %.2f, %.2f)\n", target_pos[0], target_pos[1], target_pos[2]);
    
    while(t_physics < SIMULATION_DURATION) {
        update_quad(sim->quad, DT_PHYSICS);
        t_physics += DT_PHYSICS;
        
        if(t_control <= t_physics) {
            // Get desired velocities and yaw rate from high-level controller
            velocity_controller(sim->quad, target_pos, desired_vel_B, &desired_yaw_rate);
            
            // Get state using sensor data
            get_state(sim->quad, state, desired_vel_B, desired_yaw_rate);
            
            // Run policy network
            forward(policy, state);
            
            // Generate motor commands
            for(int i = 0; i < 4; i++) {
                double std = squash(policy->layers[policy->n_layers-1].x[i + 4], MIN_STD, MAX_STD);
                double safe_margin = 4.0 * std;
                double mean_min = OMEGA_MIN + safe_margin;
                double mean_max = OMEGA_MAX - safe_margin;
                double mean = squash(policy->layers[policy->n_layers-1].x[i], mean_min, mean_max);
                
                sim->quad->omega_next[i] = mean;
            }
            
            t_control += DT_CONTROL;
            
            double dist = sqrt(
                pow(sim->quad->linear_position_W[0] - target_pos[0], 2) +
                pow(sim->quad->linear_position_W[1] - target_pos[1], 2) +
                pow(sim->quad->linear_position_W[2] - target_pos[2], 2)
            );
            
            printf("\rTime: %.2f/%.2f | Pos: (%.2f, %.2f, %.2f) | Vel_d: (%.2f, %.2f, %.2f) | Yaw_d: %.2f | Dist: %.3f", 
                   t_physics, SIMULATION_DURATION,
                   sim->quad->linear_position_W[0],
                   sim->quad->linear_position_W[1],
                   sim->quad->linear_position_W[2],
                   desired_vel_B[0],
                   desired_vel_B[1],
                   desired_vel_B[2],
                   desired_yaw_rate,
                   dist);
            fflush(stdout);
        }
        
        if(t_render <= t_physics) {
            render_sim(sim);
            t_render += DT_RENDER;
        }
    }

    double final_dist = sqrt(
        pow(sim->quad->linear_position_W[0] - target_pos[0], 2) +
        pow(sim->quad->linear_position_W[1] - target_pos[1], 2) +
        pow(sim->quad->linear_position_W[2] - target_pos[2], 2)
    );

    printf("\n\nSimulation complete!\n");
    printf("Final position: (%.2f, %.2f, %.2f)\n",
           sim->quad->linear_position_W[0],
           sim->quad->linear_position_W[1],
           sim->quad->linear_position_W[2]);
    printf("Target position: (%.2f, %.2f, %.2f)\n",
           target_pos[0], target_pos[1], target_pos[2]);
    printf("Final distance to target: %.3f\n", final_dist);

    // Cleanup
    free_net(policy);
    save_sim(sim);
    free_sim(sim);

    return 0;
}