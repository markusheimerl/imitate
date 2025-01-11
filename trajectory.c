#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef RENDER
#include "gif.h"
#include "rasterizer.h"
#endif
#include "quad.h"
#include <stdbool.h>

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define DT_RENDER   (1.0 / 30.0)
#define VEC3_MAG2(v) ((v)[0]*(v)[0] + (v)[1]*(v)[1] + (v)[2]*(v)[2])

#define D1 2048
#define D2 1024
#define D3 512
#define M_IN 6
#define M_OUT 4

bool load_weights(const char* filename, double *W1, double *b1, double *W2, double *b2, 
                 double *W3, double *b3, double *W4, double *b4) {
    FILE* f = fopen(filename, "rb");
    if (!f) return false;
    size_t items_read = fread(W1, sizeof(double), D1*M_IN, f) + fread(b1, sizeof(double), D1, f) +
                        fread(W2, sizeof(double), D2*D1, f) + fread(b2, sizeof(double), D2, f) +
                        fread(W3, sizeof(double), D3*D2, f) + fread(b3, sizeof(double), D3, f) +
                        fread(W4, sizeof(double), M_OUT*D3, f) + fread(b4, sizeof(double), M_OUT, f);
    fclose(f);
    return items_read == (D1*M_IN + D1 + D2*D1 + D2 + D3*D2 + D3 + M_OUT*D3 + M_OUT);
}

void save_weights(const char* filename, double *W1, double *b1, double *W2, double *b2,
                 double *W3, double *b3, double *W4, double *b4) {
    FILE* f = fopen(filename, "wb");
    fwrite(W1, sizeof(double), D1*M_IN, f); fwrite(b1, sizeof(double), D1, f);
    fwrite(W2, sizeof(double), D2*D1, f); fwrite(b2, sizeof(double), D2, f);
    fwrite(W3, sizeof(double), D3*D2, f); fwrite(b3, sizeof(double), D3, f);
    fwrite(W4, sizeof(double), M_OUT*D3, f); fwrite(b4, sizeof(double), M_OUT, f);
    fclose(f);
}

void forward(double *W1, double *b1, double *W2, double *b2, double *W3, double *b3, 
            double *W4, double *b4, double *input, double *h1, double *h2, double *h3, double *output) {
    for(int i = 0; i < D1; i++) {
        double sum = b1[i];
        for(int j = 0; j < M_IN; j++) sum += W1[i*M_IN + j] * input[j];
        h1[i] = sum > 0 ? sum : sum * 0.1;
    }
    for(int i = 0; i < D2; i++) {
        double sum = b2[i];
        for(int j = 0; j < D1; j++) sum += W2[i*D1 + j] * h1[j];
        h2[i] = sum > 0 ? sum : sum * 0.1;
    }
    for(int i = 0; i < D3; i++) {
        double sum = b3[i];
        for(int j = 0; j < D2; j++) sum += W3[i*D2 + j] * h2[j];
        h3[i] = sum > 0 ? sum : sum * 0.1;
    }
    for(int i = 0; i < M_OUT; i++) {
        double sum = b4[i];
        for(int j = 0; j < D3; j++) sum += W4[i*D3 + j] * h3[j];
        output[i] = OMEGA_MIN + (OMEGA_MAX - OMEGA_MIN) / (1.0 + exp(-sum));
    }
}

int main(int argc, char *argv[]) {
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    char filename[100];

    double *W1 = malloc(D1*M_IN*sizeof(double)), *b1 = calloc(D1, sizeof(double));
    double *W2 = malloc(D2*D1*sizeof(double)), *b2 = calloc(D2, sizeof(double));
    double *W3 = malloc(D3*D2*sizeof(double)), *b3 = calloc(D3, sizeof(double));
    double *W4 = malloc(M_OUT*D3*sizeof(double)), *b4 = calloc(M_OUT, sizeof(double));
    double *h1 = malloc(D1*sizeof(double)), *h2 = malloc(D2*sizeof(double));
    double *h3 = malloc(D3*sizeof(double)), input[M_IN], output[M_OUT];

    if (argc > 1) {
        if (!load_weights(argv[1], W1, b1, W2, b2, W3, b3, W4, b4)) {
            printf("Failed to load weights\n");
            return 1;
        }
    } else {
        srand(time(NULL));
        for(int i = 0; i < D1*M_IN; i++) W1[i] = ((double)rand()/RAND_MAX - 0.5) * sqrt(2.0/M_IN);
        for(int i = 0; i < D2*D1; i++) W2[i] = ((double)rand()/RAND_MAX - 0.5) * sqrt(2.0/D1);
        for(int i = 0; i < D3*D2; i++) W3[i] = ((double)rand()/RAND_MAX - 0.5) * sqrt(2.0/D2);
        for(int i = 0; i < M_OUT*D3; i++) W4[i] = ((double)rand()/RAND_MAX - 0.5) * sqrt(2.0/D3);
    }

    #ifdef RENDER
    sprintf(filename, "%d-%d-%d_%d-%d-%d_flight.gif", tm.tm_year + 1900, tm.tm_mon + 1, 
            tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    Mesh* meshes[] = {create_mesh("sim/rasterizer/drone.obj", "sim/rasterizer/drone.bmp"),
                      create_mesh("sim/rasterizer/ground.obj", "sim/rasterizer/ground.bmp")};
    uint8_t *frame_buffer = calloc(WIDTH * HEIGHT * 3, sizeof(uint8_t));
    ge_GIF *gif = ge_new_gif(filename, WIDTH, HEIGHT, 4, -1, 0);
    transform_mesh(meshes[1], (double[3]){0.0, -0.2, 0.0}, 1.0, (double[9]){1,0,0, 0,1,0, 0,0,1});
    double t_render = 0.0;
    #endif

    #ifdef LOG
    sprintf(filename, "%d-%d-%d_%d-%d-%d_trajectory.csv", tm.tm_year + 1900, tm.tm_mon + 1,
            tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    FILE *csv_file = fopen(filename, "w");
    fprintf(csv_file, "pos[0],pos[1],pos[2],vel[0],vel[1],vel[2],ang_vel[0],ang_vel[1],ang_vel[2],"
            "R[0],R[1],R[2],R[3],R[4],R[5],R[6],R[7],R[8],acc_s[0],acc_s[1],acc_s[2],"
            "gyro_s[0],gyro_s[1],gyro_s[2],omega[0],omega[1],omega[2],omega[3],reward,discounted_return\n");
    #endif

    for(int i = 0; i < 3; i++) {
        accel_bias[i] = (2.0*((double)rand()/RAND_MAX) - 1.0) * ACCEL_BIAS;
        gyro_bias[i] = (2.0*((double)rand()/RAND_MAX) - 1.0) * GYRO_BIAS;
    }

    double t_physics = 0.0, t_control = 0.0;
    
    while (t_physics < 10.0) {
        if (VEC3_MAG2(linear_position_W) > 100.0*100.0 || 
            VEC3_MAG2(linear_velocity_W) > 10.0*10.0 || 
            VEC3_MAG2(angular_velocity_B) > 10.0*10.0) break;

        update_drone_physics(DT_PHYSICS);
        t_physics += DT_PHYSICS;
        
        if (t_control <= t_physics) {
            for(int i = 0; i < 3; i++) {
                input[i] = linear_acceleration_B_s[i];
                input[i+3] = angular_velocity_B_s[i];
            }
            forward(W1, b1, W2, b2, W3, b3, W4, b4, input, h1, h2, h3, output);
            memcpy(omega, output, M_OUT * sizeof(double));

            #ifdef LOG
            double pos_error = sqrt(pow(linear_position_W[0], 2) + 
                                  pow(linear_position_W[1] - 1.0, 2) + 
                                  pow(linear_position_W[2], 2));
            double reward = exp(-(pos_error * 2.0 + sqrt(VEC3_MAG2(angular_velocity_B)) * 0.5));
            fprintf(csv_file, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,"
                    "%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
                    linear_position_W[0], linear_position_W[1], linear_position_W[2],
                    linear_velocity_W[0], linear_velocity_W[1], linear_velocity_W[2],
                    angular_velocity_B[0], angular_velocity_B[1], angular_velocity_B[2],
                    R_W_B[0], R_W_B[1], R_W_B[2], R_W_B[3], R_W_B[4], R_W_B[5], 
                    R_W_B[6], R_W_B[7], R_W_B[8],
                    linear_acceleration_B_s[0], linear_acceleration_B_s[1], linear_acceleration_B_s[2],
                    angular_velocity_B_s[0], angular_velocity_B_s[1], angular_velocity_B_s[2],
                    omega[0], omega[1], omega[2], omega[3],
                    reward, 0.0);
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
    for(int i = 0; i < line_count; i++) {
        double discounted_return = 0.0;
        double discount = 1.0;
        for(int j = i; j < line_count; j++) {
            discounted_return += discount * rewards[j];
            discount *= gamma;
        }
        
        fgets(line, sizeof(line), csv_input);
        line[strlen(line)-1] = '\0';
        char *last_comma = strrchr(line, ',');
        *last_comma = '\0';
        fprintf(csv_output, "%s,%f\n", line, discounted_return);
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

    if (argc <= 1) {
        sprintf(filename, "%d-%d-%d_%d-%d-%d_policy_weights.bin", tm.tm_year + 1900, tm.tm_mon + 1,
                tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
        save_weights(filename, W1, b1, W2, b2, W3, b3, W4, b4);
    }

    free(W1); free(b1); free(W2); free(b2);
    free(W3); free(b3); free(W4); free(b4);
    free(h1); free(h2); free(h3);
    return 0;
}