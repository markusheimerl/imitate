#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "transformer.h"
#include "quad.h"
#include "gif.h"
#include "rasterizer.h"

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define DT_RENDER   (1.0 / 30.0)

static double history[SEQ_LENGTH][INPUT_FEATURES];
static int history_len = 0;

static bool is_stable(void) {
    for (int i = 0; i < 3; i++) if (fabs(angular_velocity_B[i]) > 0.005) return false;
    return true;
}

static bool is_at_target_position(void) {
    for (int i = 0; i < 3; i++) if (fabs(linear_position_W[i] - linear_position_d_W[i]) > 0.1) return false;
    return true;
}

static bool check_divergence(void) {
    for (int i = 0; i < 3; i++) {
        if (fabs(linear_position_W[i]) > 1000.0 || fabs(linear_velocity_W[i]) > 100.0 || 
            fabs(angular_velocity_B[i]) > 100.0) return true;
    }
    for (int i = 0; i < 4; i++) if (omega_next[i] < 0 || omega_next[i] > 1000) return true;
    return false;
}

static void update_history(void) {
    if (history_len == SEQ_LENGTH) {
        for (int i = 0; i < SEQ_LENGTH - 1; i++)
            for (int j = 0; j < INPUT_FEATURES; j++)
                history[i][j] = history[i + 1][j];
    } else history_len++;
    
    int idx = 0;
    for (int i = 0; i < 3; i++) history[history_len-1][idx++] = linear_position_d_W[i];
    history[history_len-1][idx++] = yaw_d;
    for (int i = 0; i < 3; i++) history[history_len-1][idx++] = angular_velocity_B[i];
    for (int i = 0; i < 3; i++) history[history_len-1][idx++] = linear_acceleration_B[i];
    for (int i = 0; i < 4; i++) history[history_len-1][idx++] = omega[i];
}

static void prepare_transformer_input(double* input) {
    for (int s = 0; s < SEQ_LENGTH; s++) {
        if (s < history_len) {
            for (int f = 0; f < INPUT_FEATURES; f++)
                input[s * INPUT_FEATURES + f] = history[s][f];
        } else {
            for (int f = 0; f < INPUT_FEATURES; f++)
                input[s * INPUT_FEATURES + f] = 0.0;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) { printf("Usage: %s <weights_file>\n", argv[0]); return 1; }

    Tensor W_seq = {malloc(SEQUENCE_FEATURES * D_MODEL * sizeof(double)), NULL, NULL, SEQUENCE_FEATURES * D_MODEL};
    Tensor W_cond = {malloc(CONDITION_FEATURES * D_MODEL * sizeof(double)), NULL, NULL, CONDITION_FEATURES * D_MODEL};
    Tensor W_out = {malloc(D_MODEL * SEQUENCE_FEATURES * sizeof(double)), NULL, NULL, D_MODEL * SEQUENCE_FEATURES};
    
    Tensor W_q[N_LAYERS], W_k[N_LAYERS], W_v[N_LAYERS], W_o[N_LAYERS], W_ff1[N_LAYERS], W_ff2[N_LAYERS];
    for (int l = 0; l < N_LAYERS; l++) {
        const int attn_size = D_MODEL * D_MODEL;
        const int ff_size1 = D_MODEL * (D_MODEL * 4);
        const int ff_size2 = (D_MODEL * 4) * D_MODEL;
        W_q[l] = (Tensor){malloc(attn_size * sizeof(double)), NULL, NULL, attn_size};
        W_k[l] = (Tensor){malloc(attn_size * sizeof(double)), NULL, NULL, attn_size};
        W_v[l] = (Tensor){malloc(attn_size * sizeof(double)), NULL, NULL, attn_size};
        W_o[l] = (Tensor){malloc(attn_size * sizeof(double)), NULL, NULL, attn_size};
        W_ff1[l] = (Tensor){malloc(ff_size1 * sizeof(double)), NULL, NULL, ff_size1};
        W_ff2[l] = (Tensor){malloc(ff_size2 * sizeof(double)), NULL, NULL, ff_size2};
    }

    if (!load_weights(argv[1], &W_seq, &W_cond, W_q, W_k, W_v, W_o, W_ff1, W_ff2, &W_out)) {
        printf("Failed to load weights from %s\n", argv[1]); return 1;
    }

    double *transformer_input = malloc(BATCH_SIZE * (SEQ_LENGTH + 1) * INPUT_FEATURES * sizeof(double));
    Tensor hidden = {malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double)), NULL, NULL, BATCH_SIZE * SEQ_LENGTH * D_MODEL};
    Tensor temp = {malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double)), NULL, NULL, BATCH_SIZE * SEQ_LENGTH * D_MODEL};
    Tensor output = {malloc(BATCH_SIZE * SEQ_LENGTH * SEQUENCE_FEATURES * sizeof(double)), NULL, NULL, BATCH_SIZE * SEQ_LENGTH * SEQUENCE_FEATURES};
    double *q_buf = malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double *k_buf = malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double *v_buf = malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double *s_buf = malloc(BATCH_SIZE * N_HEAD * SEQ_LENGTH * SEQ_LENGTH * sizeof(double));
    double *mid_buf = malloc(BATCH_SIZE * SEQ_LENGTH * (D_MODEL * 4) * sizeof(double));

    Mesh* meshes[] = {create_mesh("sim/rasterizer/drone.obj", "sim/rasterizer/drone.bmp"), 
                      create_mesh("sim/rasterizer/ground.obj", "sim/rasterizer/ground.bmp")};
    uint8_t *frame_buffer = calloc(WIDTH * HEIGHT * 3, sizeof(uint8_t));
    ge_GIF *gif = ge_new_gif("transformer_flight.gif", WIDTH, HEIGHT, 4, -1, 0);
    transform_mesh(meshes[1], (double[3]){0.0, -0.5, 0.0}, 1.0, (double[9]){1,0,0, 0,1,0, 0,0,1});

    double t_physics = 0.0, t_control = 0.0, t_render = 0.0;
    linear_position_d_W[0] = 2.0; linear_position_d_W[1] = 2.0; linear_position_d_W[2] = 2.0; yaw_d = M_PI / 4;

    while (!is_stable() || !is_at_target_position()) {
        if (check_divergence()) { printf("Simulation diverged.\n"); return 1; }

        update_drone_physics(DT_PHYSICS);
        t_physics += DT_PHYSICS;
        
        if (t_control <= t_physics) {
            update_history();
            if (history_len == SEQ_LENGTH) {
                prepare_transformer_input(transformer_input);
                forward_pass(transformer_input, &output, &hidden, &temp,
                           &W_seq, &W_cond, W_q, W_k, W_v, W_o, W_ff1, W_ff2, &W_out,
                           q_buf, k_buf, v_buf, s_buf, mid_buf);
                int last_seq = (SEQ_LENGTH - 1) * SEQUENCE_FEATURES;
                for (int i = 0; i < 4; i++) omega_next[i] = output.data[last_seq + i + 6];
            }
            update_rotor_speeds();
            t_control += DT_CONTROL;
        }

        if (t_render <= t_physics) {
            transform_mesh(meshes[0], linear_position_W, 0.5, R_W_B);
            memset(frame_buffer, 0, WIDTH * HEIGHT * 3);
            vertex_shader(meshes, 2, (double[3]){-2.0, 2.0, -2.0}, (double[3]){0.0, 0.0, 0.0});
            rasterize(frame_buffer, meshes, 2);
            ge_add_frame(gif, frame_buffer, 6);
            t_render += DT_RENDER;
        }
        printf("Position: [%.3f, %.3f, %.3f]\nTarget: [%.3f, %.3f, %.3f]\nRotor speeds: [%.1f, %.1f, %.1f, %.1f]\n---\n",
               linear_position_W[0], linear_position_W[1], linear_position_W[2],
               linear_position_d_W[0], linear_position_d_W[1], linear_position_d_W[2],
               omega[0], omega[1], omega[2], omega[3]);
    }

    free(frame_buffer); free_meshes(meshes, 2); ge_close_gif(gif);
    free(transformer_input); free(hidden.data); free(temp.data); free(output.data);
    free(q_buf); free(k_buf); free(v_buf); free(s_buf); free(mid_buf);
    free(W_seq.data); free(W_cond.data); free(W_out.data);
    for (int l = 0; l < N_LAYERS; l++) {
        free(W_q[l].data); free(W_k[l].data); free(W_v[l].data); free(W_o[l].data);
        free(W_ff1[l].data); free(W_ff2[l].data);
    }
    return 0;
}