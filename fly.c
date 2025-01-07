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
#include "transformer.h"

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define DT_RENDER   (1.0 / 30.0)
#define VEC3_MAG2(v) ((v)[0]*(v)[0] + (v)[1]*(v)[1] + (v)[2]*(v)[2])

int main(int argc, char *argv[]) {
    if (argc != 2) { printf("Usage: %s <weights_file>\n", argv[0]); return 1; }
    
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    char filename[100];
    
    sprintf(filename, "%d-%d-%d_%d-%d-%d_flight.gif", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    Mesh* meshes[] = {create_mesh("sim/rasterizer/drone.obj", "sim/rasterizer/drone.bmp"), create_mesh("sim/rasterizer/ground.obj", "sim/rasterizer/ground.bmp")};
    uint8_t *frame_buffer = calloc(WIDTH * HEIGHT * 3, sizeof(uint8_t));
    ge_GIF *gif = ge_new_gif(filename, WIDTH, HEIGHT, 4, -1, 0);
    transform_mesh(meshes[1], (double[3]){0.0, -0.5, 0.0}, 1.0, (double[9]){1,0,0, 0,1,0, 0,0,1});
    double t_render = 0.0, t_status = 0.0;
    int max_steps = 2;

    double *W_seq = malloc(SEQUENCE_FEATURES * D_MODEL * sizeof(double));
    double *W_cond = malloc(CONDITION_FEATURES * D_MODEL * sizeof(double));
    double *W_out = malloc(D_MODEL * SEQUENCE_FEATURES * sizeof(double));
    double *W_q = malloc(N_LAYERS * D_MODEL * D_MODEL * sizeof(double));
    double *W_k = malloc(N_LAYERS * D_MODEL * D_MODEL * sizeof(double));
    double *W_v = malloc(N_LAYERS * D_MODEL * D_MODEL * sizeof(double));
    double *W_o = malloc(N_LAYERS * D_MODEL * D_MODEL * sizeof(double));
    double *W_ff1 = malloc(N_LAYERS * D_MODEL * (D_MODEL * 4) * sizeof(double));
    double *W_ff2 = malloc(N_LAYERS * (D_MODEL * 4) * D_MODEL * sizeof(double));
    if (!load_weights(argv[1], W_seq, W_cond, W_q, W_k, W_v, W_o, W_ff1, W_ff2, W_out)) { printf("Failed to load weights\n"); return 1; }

    double *transformer_input = malloc(SEQ_LENGTH * (CONDITION_FEATURES + SEQUENCE_FEATURES) * sizeof(double));
    double *hidden = malloc(SEQ_LENGTH * D_MODEL * sizeof(double));
    double *temp = malloc(SEQ_LENGTH * D_MODEL * sizeof(double));
    double *output = malloc(SEQ_LENGTH * SEQUENCE_FEATURES * sizeof(double));
    double *q_buf = malloc(SEQ_LENGTH * D_MODEL * sizeof(double));
    double *k_buf = malloc(SEQ_LENGTH * D_MODEL * sizeof(double));
    double *v_buf = malloc(SEQ_LENGTH * D_MODEL * sizeof(double));
    double *s_buf = malloc(N_HEAD * SEQ_LENGTH * SEQ_LENGTH * sizeof(double));
    double *mid_buf = malloc(SEQ_LENGTH * (D_MODEL * 4) * sizeof(double));
    double *history = calloc(SEQ_LENGTH * (CONDITION_FEATURES + SEQUENCE_FEATURES), sizeof(double));
    int history_pos = 0;

    srand(time(NULL));
    double t_physics = 0.0, t_control = 0.0;

    for (int meta_step = 0; meta_step < max_steps; meta_step++) {
        for (int i = 0; i < 3; i++) linear_velocity_d_B[i] = 0.0;
        if (rand() % 4 != 0) { linear_velocity_d_B[rand() % 3] = (rand() % 2 ? 0.3 : -0.3) * (rand() % 3 == 1 ? 0.4 : 1.0);}
        printf("\n=== New Target %d ===\nDesired velocity (body): [%.3f, %.3f, %.3f]\n", meta_step, linear_velocity_d_B[0], linear_velocity_d_B[1], linear_velocity_d_B[2]);

        double min_time = t_physics + 0.5;
        bool velocity_achieved = false;

        while (!velocity_achieved || t_physics < min_time) {
            if (VEC3_MAG2(linear_position_W) > 100.0*100.0 || VEC3_MAG2(linear_velocity_W) > 10.0*10.0 || VEC3_MAG2(angular_velocity_B) > 10.0*10.0) {
                printf("\nSimulation diverged.\n");
                return 1;
            }

            update_drone_physics(DT_PHYSICS);
            t_physics += DT_PHYSICS;
            
            if (t_control <= t_physics) {
                memmove(history, history + (CONDITION_FEATURES + SEQUENCE_FEATURES), (SEQ_LENGTH - 1) * (CONDITION_FEATURES + SEQUENCE_FEATURES) * sizeof(double));
                double* current = history + (SEQ_LENGTH - 1) * (CONDITION_FEATURES + SEQUENCE_FEATURES);
                memcpy(current, linear_velocity_d_B, CONDITION_FEATURES * sizeof(double));
                memcpy(current + CONDITION_FEATURES, angular_velocity_B, 3 * sizeof(double));
                memcpy(current + CONDITION_FEATURES + 3, linear_acceleration_B, 3 * sizeof(double));
                memcpy(current + CONDITION_FEATURES + 6, omega, 4 * sizeof(double));
                memcpy(transformer_input, history, SEQ_LENGTH * (CONDITION_FEATURES + SEQUENCE_FEATURES) * sizeof(double));

                if (t_physics >= SEQ_LENGTH * DT_CONTROL) {
                    forward_pass(transformer_input, output, hidden, temp, W_seq, W_cond, W_q, W_k, W_v, W_o, W_ff1, W_ff2, W_out, q_buf, k_buf, v_buf, s_buf, mid_buf);
                    const double* pred = &output[(SEQ_LENGTH-1) * SEQUENCE_FEATURES];
                    memcpy(omega_next, pred + 6, 4 * sizeof(double));
                    printf("\nPredicted rotor speeds:\n");
                    printf("Omega: [%5.2f, %5.2f, %5.2f, %5.2f]\n", omega_next[0], omega_next[1], omega_next[2], omega_next[3]);
                }

                update_rotor_speeds();
                t_control += DT_CONTROL;

                velocity_achieved = true;
                for (int i = 0; i < 3; i++) {
                    if (fabs(linear_velocity_B[i] - linear_velocity_d_B[i]) > 0.01 || fabs(angular_velocity_B[i]) > 0.05) {
                        velocity_achieved = false;
                        break;
                    }
                }

                if (t_physics >= t_status) {
                    printf("\rP: [%5.2f, %5.2f, %5.2f] L_V_B: [%5.2f, %5.2f, %5.2f] A_V_B: [%5.2f, %5.2f, %5.2f] R: [%5.2f, %5.2f, %5.2f, %5.2f]", linear_position_W[0], linear_position_W[1], linear_position_W[2], linear_velocity_B[0], linear_velocity_B[1], linear_velocity_B[2], angular_velocity_B[0], angular_velocity_B[1], angular_velocity_B[2], omega[0], omega[1], omega[2], omega[3]);
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

    free(frame_buffer); free_meshes(meshes, 2); ge_close_gif(gif);
    free(transformer_input); free(hidden); free(temp); free(output);
    free(q_buf); free(k_buf); free(v_buf); free(s_buf); free(mid_buf); free(history);
    free(W_seq); free(W_cond); free(W_out); free(W_q); free(W_k); free(W_v); free(W_o); free(W_ff1); free(W_ff2);
    
    return 0;
}