#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <omp.h>
#include "gif.h"
#include "rasterizer.h"
#include "quad.h"

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define DT_RENDER   (1.0 / 30.0)
#define VEC3_MAG2(v) ((v)[0]*(v)[0] + (v)[1]*(v)[1] + (v)[2]*(v)[2])

#define INPUT_FEATURES 10
#define HIDDEN_SIZE 32
#define N_HIDDEN 2
#define OUTPUT_FEATURES 4

// Forward declarations of neural network functions
double relu(double x) { return x > 0 ? x : 0; }

void forward_pass(const double* input, double** activations, double** weights, double** biases) {
    #pragma omp parallel for
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        double sum = biases[0][i];
        for (int j = 0; j < INPUT_FEATURES; j++) sum += input[j] * weights[0][i * INPUT_FEATURES + j];
        activations[0][i] = relu(sum);
    }
    
    for (int layer = 1; layer < N_HIDDEN; layer++) {
        #pragma omp parallel for
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            double sum = biases[layer][i];
            for (int j = 0; j < HIDDEN_SIZE; j++) sum += activations[layer-1][j] * weights[layer][i * HIDDEN_SIZE + j];
            activations[layer][i] = relu(sum);
        }
    }
    
    #pragma omp parallel for
    for (int i = 0; i < OUTPUT_FEATURES; i++) {
        double sum = biases[N_HIDDEN][i];
        for (int j = 0; j < HIDDEN_SIZE; j++) sum += activations[N_HIDDEN-1][j] * weights[N_HIDDEN][i * HIDDEN_SIZE + j];
        activations[N_HIDDEN][i] = sum;
    }
}

int load_weights(const char* filename, double** weights, double** biases) {
    FILE* f = fopen(filename, "rb");
    if (!f) return 0;
    
    size_t read = fread(weights[0], sizeof(double), INPUT_FEATURES * HIDDEN_SIZE, f);
    read += fread(biases[0], sizeof(double), HIDDEN_SIZE, f);
    
    for (int i = 1; i < N_HIDDEN; i++) {
        read += fread(weights[i], sizeof(double), HIDDEN_SIZE * HIDDEN_SIZE, f);
        read += fread(biases[i], sizeof(double), HIDDEN_SIZE, f);
    }
    
    read += fread(weights[N_HIDDEN], sizeof(double), HIDDEN_SIZE * OUTPUT_FEATURES, f);
    read += fread(biases[N_HIDDEN], sizeof(double), OUTPUT_FEATURES, f);
    
    fclose(f);
    return read == (INPUT_FEATURES * HIDDEN_SIZE + HIDDEN_SIZE + (N_HIDDEN - 1) * (HIDDEN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE) + HIDDEN_SIZE * OUTPUT_FEATURES + OUTPUT_FEATURES);
}

int main(int argc, char *argv[]) {
    if (argc != 2) { printf("Usage: %s <weights_file>\n", argv[0]); return 1; }

    // Initialize time and files
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    char filename[100], csv_filename[100];

    sprintf(csv_filename, "%d-%d-%d_%d-%d-%d_flight_data.csv", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    FILE *csv_file = fopen(csv_filename, "w");
    fprintf(csv_file, "pos_d[0],pos_d[1],pos_d[2],yaw_d,ang_vel[0],ang_vel[1],ang_vel[2],acc[0],acc[1],acc[2],omega[0],omega[1],omega[2],omega[3]\n");

    // Initialize visualization
    sprintf(filename, "%d-%d-%d_%d-%d-%d_flight.gif", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    Mesh* meshes[] = {create_mesh("sim/rasterizer/drone.obj", "sim/rasterizer/drone.bmp"), 
                      create_mesh("sim/rasterizer/ground.obj", "sim/rasterizer/ground.bmp")};
    uint8_t *frame_buffer = calloc(WIDTH * HEIGHT * 3, sizeof(uint8_t));
    ge_GIF *gif = ge_new_gif(filename, WIDTH, HEIGHT, 4, -1, 0);
    transform_mesh(meshes[1], (double[3]){0.0, -0.2, 0.0}, 1.0, (double[9]){1,0,0, 0,1,0, 0,0,1});
    
    // Initialize neural network
    double** weights = malloc((N_HIDDEN + 1) * sizeof(double*));
    double** biases = malloc((N_HIDDEN + 1) * sizeof(double*));
    double** activations = malloc((N_HIDDEN + 1) * sizeof(double*));
    
    weights[0] = malloc(INPUT_FEATURES * HIDDEN_SIZE * sizeof(double));
    biases[0] = malloc(HIDDEN_SIZE * sizeof(double));
    activations[0] = malloc(HIDDEN_SIZE * sizeof(double));
    
    for (int i = 1; i < N_HIDDEN; i++) {
        weights[i] = malloc(HIDDEN_SIZE * HIDDEN_SIZE * sizeof(double));
        biases[i] = malloc(HIDDEN_SIZE * sizeof(double));
        activations[i] = malloc(HIDDEN_SIZE * sizeof(double));
    }
    
    weights[N_HIDDEN] = malloc(HIDDEN_SIZE * OUTPUT_FEATURES * sizeof(double));
    biases[N_HIDDEN] = malloc(OUTPUT_FEATURES * sizeof(double));
    activations[N_HIDDEN] = malloc(OUTPUT_FEATURES * sizeof(double));

    if (!load_weights(argv[1], weights, biases)) { 
        printf("Failed to load weights\n"); 
        return 1; 
    }

    // Simulation variables
    double t_physics = 0.0, t_control = 0.0, t_render = 0.0, t_status = 0.0;
    int max_steps = 5;
    double input_features[INPUT_FEATURES];

    srand(time(NULL));
    for(int i = 0; i < 3; i++) {
        accel_bias[i] = (2.0 * ((double)rand() / RAND_MAX) - 1.0) * ACCEL_BIAS;
        gyro_bias[i] = (2.0 * ((double)rand() / RAND_MAX) - 1.0) * GYRO_BIAS;
    }

    // Main simulation loop
    for (int meta_step = 0; meta_step < max_steps; meta_step++) {
        for (int i = 0; i < 3; i++) linear_position_d_W[i] = (double)rand() / RAND_MAX * 10 - (i != 1 ? 5 : 0);
        yaw_d = (double)rand() / RAND_MAX * 2 * M_PI;

        printf("\n=== New Target %d ===\nDesired: P: [%.3f, %.3f, %.3f], yaw: %.3f\n", 
               meta_step, linear_position_d_W[0], linear_position_d_W[1], linear_position_d_W[2], yaw_d);

        bool position_achieved = false;
        bool stability_achieved = false;
        double min_time = t_physics + 0.5;

        while (!position_achieved || !stability_achieved || t_physics < min_time) {
            if (VEC3_MAG2(linear_position_W) > 100.0*100.0 || 
                VEC3_MAG2(linear_velocity_W) > 10.0*10.0 || 
                VEC3_MAG2(angular_velocity_B) > 10.0*10.0) {
                printf("\nSimulation diverged.\n");
                fclose(csv_file);
                return 1;
            }

            update_drone_physics(DT_PHYSICS);
            t_physics += DT_PHYSICS;
            
            if (t_control <= t_physics) {
                // Prepare input features for neural network
                memcpy(input_features, linear_position_d_W, 3 * sizeof(double));
                input_features[3] = yaw_d;
                memcpy(input_features + 4, angular_velocity_B_s, 3 * sizeof(double));
                memcpy(input_features + 7, linear_acceleration_B_s, 3 * sizeof(double));

                // Run neural network
                forward_pass(input_features, activations, weights, biases);
                memcpy(omega_next, activations[N_HIDDEN], 4 * sizeof(double));

                // Log data
                fprintf(csv_file, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
                        linear_position_d_W[0], linear_position_d_W[1], linear_position_d_W[2], yaw_d,
                        angular_velocity_B_s[0], angular_velocity_B_s[1], angular_velocity_B_s[2],
                        linear_acceleration_B_s[0], linear_acceleration_B_s[1], linear_acceleration_B_s[2],
                        omega_next[0], omega_next[1], omega_next[2], omega_next[3]);

                update_rotor_speeds();
                t_control += DT_CONTROL;

                position_achieved = true;
                stability_achieved = true;
                for (int i = 0; i < 3; i++) {
                    if (fabs(linear_position_W[i] - linear_position_d_W[i]) > 0.1) position_achieved = false;
                    if (fabs(angular_velocity_B[i]) > 0.05) stability_achieved = false;
                }

                if (t_physics >= t_status) {
                    printf("\rP: [%.3f, %.3f, %.3f] yaw: %.3f A_V_B: [%.3f, %.3f, %.3f] R: [%.3f, %.3f, %.3f, %.3f]",
                           linear_position_W[0], linear_position_W[1], linear_position_W[2],
                           atan2(R_W_B[2], R_W_B[8]) < 0 ? atan2(R_W_B[2], R_W_B[8]) + 2 * M_PI : atan2(R_W_B[2], R_W_B[8]),
                           angular_velocity_B[0], angular_velocity_B[1], angular_velocity_B[2],
                           omega[0], omega[1], omega[2], omega[3]);
                    fflush(stdout);
                    t_status = t_physics + 0.1;
                }
            }

            if (t_render <= t_physics) {
                transform_mesh(meshes[0], linear_position_W, 0.5, R_W_B);
                memset(frame_buffer, 0, WIDTH * HEIGHT * 3);
                vertex_shader(meshes, 2, (double[3]){-2.0, 2.0, -2.0}, (double[3]){0.0, 0.0, 0.0});
                rasterize(frame_buffer, meshes, 2);
                ge_add_frame(gif, frame_buffer, 6);
                t_render += DT_RENDER;
            }
        }
        printf("\nTarget achieved!\n");
    }

    // Cleanup
    fclose(csv_file);
    free(frame_buffer); 
    free_meshes(meshes, 2); 
    ge_close_gif(gif);
    
    for (int i = 0; i <= N_HIDDEN; i++) {
        free(weights[i]);
        free(biases[i]);
        free(activations[i]);
    }
    free(weights);
    free(biases);
    free(activations);
    
    return 0;
}