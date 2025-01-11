#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#ifdef RENDER
#include "gif.h"
#include "rasterizer.h"
#endif
#include "quad.h"

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define DT_RENDER   (1.0 / 30.0)
#define VEC3_MAG2(v) ((v)[0]*(v)[0] + (v)[1]*(v)[1] + (v)[2]*(v)[2])

#define D1 2048  // First hidden layer
#define D2 1024  // Second hidden layer
#define D3 512   // Third hidden layer
#define M_IN 6   // Input dimension (3 acc + 3 gyro)
#define M_OUT 4  // Output dimension (4 rotor speeds)

typedef struct {
    double *W1, *b1, *W2, *b2, *W3, *b3, *W4, *b4;
} Network;

bool load_weights(const char* filename, Network* net) {
    FILE* f = fopen(filename, "rb");
    if (!f) return false;
    size_t items_read = 0;
    items_read += fread(net->W1, sizeof(double), D1 * M_IN, f);
    items_read += fread(net->b1, sizeof(double), D1, f);
    items_read += fread(net->W2, sizeof(double), D2 * D1, f);
    items_read += fread(net->b2, sizeof(double), D2, f);
    items_read += fread(net->W3, sizeof(double), D3 * D2, f);
    items_read += fread(net->b3, sizeof(double), D3, f);
    items_read += fread(net->W4, sizeof(double), M_OUT * D3, f);
    items_read += fread(net->b4, sizeof(double), M_OUT, f);
    fclose(f);
    return items_read == (D1*M_IN + D1 + D2*D1 + D2 + D3*D2 + D3 + M_OUT*D3 + M_OUT);
}

void forward(Network* net, double* input, double* h1, double* h2, double* h3, double* output) {
    // First layer
    for(int i = 0; i < D1; i++) {
        double sum = net->b1[i];
        for(int j = 0; j < M_IN; j++) sum += net->W1[i * M_IN + j] * input[j];
        h1[i] = sum > 0 ? sum : sum * 0.1; // LeakyReLU
    }
    
    // Second layer
    for(int i = 0; i < D2; i++) {
        double sum = net->b2[i];
        for(int j = 0; j < D1; j++) sum += net->W2[i * D1 + j] * h1[j];
        h2[i] = sum > 0 ? sum : sum * 0.1;
    }
    
    // Third layer
    for(int i = 0; i < D3; i++) {
        double sum = net->b3[i];
        for(int j = 0; j < D2; j++) sum += net->W3[i * D2 + j] * h2[j];
        h3[i] = sum > 0 ? sum : sum * 0.1;
    }
    
    // Output layer
    for(int i = 0; i < M_OUT; i++) {
        double sum = net->b4[i];
        for(int j = 0; j < D3; j++) sum += net->W4[i * D3 + j] * h3[j];
        output[i] = OMEGA_MIN + (OMEGA_MAX - OMEGA_MIN) * (1.0 / (1.0 + exp(-sum))); // Sigmoid scaled to [OMEGA_MIN, OMEGA_MAX]
    }
}

int main(int argc, char *argv[]) {
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    char filename[100];
    
    Network net = {
        .W1 = malloc(D1 * M_IN * sizeof(double)), .b1 = calloc(D1, sizeof(double)),
        .W2 = malloc(D2 * D1 * sizeof(double)), .b2 = calloc(D2, sizeof(double)),
        .W3 = malloc(D3 * D2 * sizeof(double)), .b3 = calloc(D3, sizeof(double)),
        .W4 = malloc(M_OUT * D3 * sizeof(double)), .b4 = calloc(M_OUT, sizeof(double))
    };

    if (argc > 1) {
        if (!load_weights(argv[1], &net)) {
            printf("Failed to load weights from %s\n", argv[1]);
            return 1;
        }
    } else {
        // Xavier initialization
        for(int i = 0; i < D1 * M_IN; i++) net.W1[i] = ((double)rand()/RAND_MAX - 0.5) * sqrt(2.0/M_IN);
        for(int i = 0; i < D2 * D1; i++) net.W2[i] = ((double)rand()/RAND_MAX - 0.5) * sqrt(2.0/D1);
        for(int i = 0; i < D3 * D2; i++) net.W3[i] = ((double)rand()/RAND_MAX - 0.5) * sqrt(2.0/D2);
        for(int i = 0; i < M_OUT * D3; i++) net.W4[i] = ((double)rand()/RAND_MAX - 0.5) * sqrt(2.0/D3);
    }

    #ifdef RENDER
    sprintf(filename, "%d-%d-%d_%d-%d-%d_flight.gif", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    Mesh* meshes[] = {create_mesh("rasterizer/drone.obj", "rasterizer/drone.bmp"), create_mesh("rasterizer/ground.obj", "rasterizer/ground.bmp")};
    uint8_t *frame_buffer = calloc(WIDTH * HEIGHT * 3, sizeof(uint8_t));
    ge_GIF *gif = ge_new_gif(filename, WIDTH, HEIGHT, 4, -1, 0);
    transform_mesh(meshes[1], (double[3]){0.0, -0.2, 0.0}, 1.0, (double[9]){1,0,0, 0,1,0, 0,0,1});
    double t_render = 0.0;
    #endif

    #ifdef LOG
    sprintf(filename, "%d-%d-%d_%d-%d-%d_trajectory.csv", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    FILE *csv_file = fopen(filename, "w");
    fprintf(csv_file, "pos[0],pos[1],pos[2],vel[0],vel[1],vel[2],ang_vel[0],ang_vel[1],ang_vel[2],R[0],R[1],R[2],R[3],R[4],R[5],R[6],R[7],R[8],acc_s[0],acc_s[1],acc_s[2],gyro_s[0],gyro_s[1],gyro_s[2],omega[0],omega[1],omega[2],omega[3],reward,discounted_return\n");
    #endif

    double *h1 = malloc(D1 * sizeof(double));
    double *h2 = malloc(D2 * sizeof(double));
    double *h3 = malloc(D3 * sizeof(double));
    double input[M_IN], output[M_OUT];

    srand(time(NULL));
    for(int i = 0; i < 3; i++) {
        accel_bias[i] = (2.0 * ((double)rand() / RAND_MAX) - 1.0) * ACCEL_BIAS;
        gyro_bias[i] = (2.0 * ((double)rand() / RAND_MAX) - 1.0) * GYRO_BIAS;
    }

    double t_physics = 0.0, t_control = 0.0;
    
    while (t_physics < 10.0) { // Run for 10 seconds
        if (VEC3_MAG2(linear_position_W) > 100.0*100.0 || VEC3_MAG2(linear_velocity_W) > 10.0*10.0 || VEC3_MAG2(angular_velocity_B) > 10.0*10.0) {
            printf("\nSimulation diverged.\n");
            #ifdef LOG
            fclose(csv_file); remove(filename);
            #endif
            return 1;
        }

        update_drone_physics(DT_PHYSICS);
        t_physics += DT_PHYSICS;
        
        if (t_control <= t_physics) {
            // Prepare network input (sensor readings)
            for(int i = 0; i < 3; i++) {
                input[i] = linear_acceleration_B_s[i];
                input[i+3] = angular_velocity_B_s[i];
            }
            
            // Run policy network
            forward(&net, input, h1, h2, h3, output);
            memcpy(omega, output, M_OUT * sizeof(double));

            #ifdef LOG
            double pos_error = sqrt(pow(linear_position_W[0], 2) + pow(linear_position_W[1] - 1.0, 2) + pow(linear_position_W[2], 2));
            double reward = exp(-(pos_error * 2.0 + sqrt(VEC3_MAG2(angular_velocity_B)) * 0.5));
            fprintf(csv_file, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
                    linear_position_W[0], linear_position_W[1], linear_position_W[2],
                    linear_velocity_W[0], linear_velocity_W[1], linear_velocity_W[2],
                    angular_velocity_B[0], angular_velocity_B[1], angular_velocity_B[2],
                    R_W_B[0], R_W_B[1], R_W_B[2], R_W_B[3], R_W_B[4], R_W_B[5], R_W_B[6], R_W_B[7], R_W_B[8],
                    linear_acceleration_B_s[0], linear_acceleration_B_s[1], linear_acceleration_B_s[2],
                    angular_velocity_B_s[0], angular_velocity_B_s[1], angular_velocity_B_s[2],
                    omega[0], omega[1], omega[2], omega[3],
                    reward, 0.0); // Discounted return will be computed later
            #endif
            
            t_control += DT_CONTROL;
        }

        #ifdef RENDER
        if (t_render <= t_physics) {
            transform_mesh(meshes[0], linear_position_W, 0.5, R_W_B);
            memset(frame_buffer, 0, WIDTH * HEIGHT * 3);
            vertex_shader(meshes, 2, (double[3]){-2.0, 2.0, -2.0}, (double[3]){0.0, 0.0, 0.0});
            rasterize(frame_buffer, meshes, 2);
            ge_add_frame(gif, frame_buffer, 6);
            t_render += DT_RENDER;
        }
        #endif
    }

        #ifdef LOG
    fclose(csv_file);
    
    // Compute discounted returns
    FILE *csv_input = fopen(filename, "r");
    char temp_filename[100];
    sprintf(temp_filename, "%s.tmp", filename);
    FILE *csv_output = fopen(temp_filename, "w");
    
    char line[1024], header[1024];
    fgets(header, sizeof(header), csv_input);
    fprintf(csv_output, "%s", header);

    int line_count = 0;
    double *rewards = NULL;
    while (fgets(line, sizeof(line), csv_input)) {
        rewards = realloc(rewards, (line_count + 1) * sizeof(double));
        char *token = strtok(line, ",");
        for(int i = 0; i < 28; i++) token = strtok(NULL, ",");
        rewards[line_count++] = atof(token);
    }

    rewind(csv_input);
    fgets(line, sizeof(line), csv_input);

    double gamma = 0.99;
    int current_line = 0;
    while (fgets(line, sizeof(line), csv_input)) {
        double discounted_return = 0.0;
        double discount = 1.0;
        for(int i = current_line; i < line_count; i++) {
            discounted_return += discount * rewards[i];
            discount *= gamma;
        }
        
        line[strlen(line)-1] = '\0';
        char *last_comma = strrchr(line, ',');
        *last_comma = '\0';
        fprintf(csv_output, "%s,%f\n", line, discounted_return);
        current_line++;
    }

    free(rewards);
    fclose(csv_input);
    fclose(csv_output);
    remove(filename);
    rename(temp_filename, filename);
    #endif

    #ifdef RENDER
    free(frame_buffer);
    free_meshes(meshes, 2);
    ge_close_gif(gif);
    #endif

    free(net.W1); free(net.b1); free(net.W2); free(net.b2);
    free(net.W3); free(net.b3); free(net.W4); free(net.b4);
    free(h1); free(h2); free(h3);
    return 0;
}