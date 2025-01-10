#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include "gif.h"
#include "rasterizer.h"
#include "quad.h"

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define DT_RENDER   (1.0 / 30.0)
#define VEC3_MAG2(v) ((v)[0]*(v)[0] + (v)[1]*(v)[1] + (v)[2]*(v)[2])
#define WAIT_TIME 1.0

#define S 32    // Sequence length
#define D 256   // Hidden dimension
#define MOTORS 4     // Input/Output dimension

bool load_weights(const char* filename, double *W_in, double *b_in, double *W_q, double *W_k, double *W_v, double *W_out, double *b_out) {
    FILE* f = fopen(filename, "rb");
    if (!f) return false;
    size_t items_read = 0;
    items_read += fread(W_in, sizeof(double), D * MOTORS, f);
    items_read += fread(b_in, sizeof(double), D, f);
    items_read += fread(W_q, sizeof(double), D * D, f);
    items_read += fread(W_k, sizeof(double), D * D, f);
    items_read += fread(W_v, sizeof(double), D * D, f);
    items_read += fread(W_out, sizeof(double), MOTORS * D, f);
    items_read += fread(b_out, sizeof(double), MOTORS, f);
    fclose(f);
    return items_read == (D*MOTORS + D + D*D*3 + MOTORS*D + MOTORS);
}

void forward(double *W_in, double *b_in, double *W_q, double *W_k, double *W_v, double *W_out, double *b_out, double *input, double *output) {
    double hidden[S * D], q[S * D], k[S * D], v[S * D], attn_scores[S * S], attn_probs[S * S], context[S * D];

    for(int s = 0; s < S; s++) {
        for(int d = 0; d < D; d++) {
            double sum = b_in[d];
            for(int m = 0; m < MOTORS; m++) sum += W_in[d * MOTORS + m] * input[m];
            hidden[s * D + d] = fmax(0.0, sum);
        }
    }

    for(int s = 0; s < S; s++) {
        for(int d = 0; d < D; d++) {
            q[s * D + d] = k[s * D + d] = v[s * D + d] = 0;
            for(int j = 0; j < D; j++) {
                q[s * D + d] += W_q[d * D + j] * hidden[s * D + j];
                k[s * D + d] += W_k[d * D + j] * hidden[s * D + j];
                v[s * D + d] += W_v[d * D + j] * hidden[s * D + j];
            }
        }
    }

    for(int i = 0; i < MOTORS; i++) {
        double sum = b_out[i];
        for(int d = 0; d < D; d++) sum += W_out[i * D + d] * context[(S-1) * D + d];
        output[i] = sum;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) { printf("Usage: %s <weights_file>\n", argv[0]); return 1; }

    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    char filename[100];
    sprintf(filename, "%d-%d-%d_%d-%d-%d_flight.gif", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);

    // Initialize visualization
    Mesh* meshes[] = {create_mesh("sim/rasterizer/drone.obj", "sim/rasterizer/drone.bmp"), create_mesh("sim/rasterizer/ground.obj", "sim/rasterizer/ground.bmp")};
    uint8_t *frame_buffer = calloc(WIDTH * HEIGHT * 3, sizeof(uint8_t));
    ge_GIF *gif = ge_new_gif(filename, WIDTH, HEIGHT, 4, -1, 0);
    transform_mesh(meshes[1], (double[3]){0.0, -0.2, 0.0}, 1.0, (double[9]){1,0,0, 0,1,0, 0,0,1});

    // Initialize neural network
    double *W_in = malloc(D * MOTORS * sizeof(double));
    double *b_in = malloc(D * sizeof(double));
    double *W_q = malloc(D * D * sizeof(double));
    double *W_k = malloc(D * D * sizeof(double));
    double *W_v = malloc(D * D * sizeof(double));
    double *W_out = malloc(MOTORS * D * sizeof(double));
    double *b_out = malloc(MOTORS * sizeof(double));

    if (!load_weights(argv[1], W_in, b_in, W_q, W_k, W_v, W_out, b_out)) {
        printf("Failed to load weights\n");
        return 1;
    }

    double t_physics = 0.0, t_control = 0.0, t_render = 0.0, t_status = 0.0, wait_start = 0.0;
    bool is_waiting = true, at_ground = true;
    double input[10], output[4];

    srand(time(NULL));
    for(int i = 0; i < 3; i++) {
        accel_bias[i] = (2.0 * ((double)rand() / RAND_MAX) - 1.0) * ACCEL_BIAS;
        gyro_bias[i] = (2.0 * ((double)rand() / RAND_MAX) - 1.0) * GYRO_BIAS;
    }

    for (int meta_step = 0; meta_step < 4; meta_step++) {
        if (is_waiting && at_ground) {
            for(int i = 0; i < 3; i++) linear_position_d_W[i] = linear_position_W[i];
            wait_start = t_physics;
            printf("\n=== Waiting at ground ===\n");
        } else {
            linear_position_d_W[0] = linear_position_d_W[2] = yaw_d = 0.0;
            linear_position_d_W[1] = at_ground ? 1.0 : 0.0;
            printf("\n=== Moving to [0.000, %.3f, 0.000] ===\n", linear_position_d_W[1]);
        }

        bool position_achieved = false, stability_achieved = false;
        double min_time = t_physics + 0.5;

        while (!position_achieved || !stability_achieved || t_physics < min_time) {
            if (VEC3_MAG2(linear_position_W) > 100.0*100.0 || VEC3_MAG2(linear_velocity_W) > 10.0*10.0 || VEC3_MAG2(angular_velocity_B) > 10.0*10.0) {
                printf("\nSimulation diverged.\n");
                return 1;
            }

            update_drone_physics(DT_PHYSICS);
            t_physics += DT_PHYSICS;
            
            if (t_control <= t_physics) {
                memcpy(input, linear_position_d_W, 3 * sizeof(double));
                input[3] = yaw_d;
                memcpy(input + 4, angular_velocity_B_s, 3 * sizeof(double));
                memcpy(input + 7, linear_acceleration_B_s, 3 * sizeof(double));

                forward(W_in, b_in, W_q, W_k, W_v, W_out, b_out, input, output);
                memcpy(omega_next, output, 4 * sizeof(double));
                update_rotor_speeds();
                t_control += DT_CONTROL;

                position_achieved = stability_achieved = true;
                for (int i = 0; i < 3; i++) {
                    if (fabs(linear_position_W[i] - linear_position_d_W[i]) > 0.1) position_achieved = false;
                    if (fabs(angular_velocity_B[i]) > 0.05) stability_achieved = false;
                }
                if (is_waiting && (t_physics - wait_start >= WAIT_TIME)) position_achieved = stability_achieved = true;

                if (t_physics >= t_status) {
                    printf("\rP: [% 6.3f, % 6.3f, % 6.3f] yaw: % 6.3f A_V_B: [% 6.3f, % 6.3f, % 6.3f] R: [% 6.3f, % 6.3f, % 6.3f, % 6.3f]", linear_position_W[0], linear_position_W[1], linear_position_W[2], atan2(R_W_B[2], R_W_B[8]) < 0 ? atan2(R_W_B[2], R_W_B[8]) + 2*M_PI : atan2(R_W_B[2], R_W_B[8]), angular_velocity_B[0], angular_velocity_B[1], angular_velocity_B[2], omega[0], omega[1], omega[2], omega[3]);
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
        
        if (is_waiting) {
            is_waiting = false;
        } else {
            at_ground = !at_ground;
            is_waiting = at_ground;
        }
    }

    free(frame_buffer);
    free_meshes(meshes, 2);
    ge_close_gif(gif);
    free(W_in); free(b_in); free(W_q); free(W_k); free(W_v); free(W_out); free(b_out);
    
    return 0;
}