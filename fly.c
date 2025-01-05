#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "transformer.h"
#include "sim/quad.h"
#include "sim/gif.h"
#include "sim/rasterizer.h"

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define DT_RENDER   (1.0 / 30.0)

// Structure to hold state history
typedef struct {
    double pos_d[3];
    double yaw_d;
    double ang_vel[3];
    double acc[3];
    double omega[4];
} StateVector;

static StateVector history[SEQ_LENGTH];
static int history_len = 0;
static Dataset normalization_data;  // Keep normalization parameters

static bool is_stable(void) {
    for (int i = 0; i < 3; i++)
        if (fabs(angular_velocity_B[i]) > 0.005) return false;
    return true;
}

static bool is_at_target_position(void) {
    for (int i = 0; i < 3; i++)
        if (fabs(linear_position_W[i] - linear_position_d_W[i]) > 0.1) return false;
    return true;
}

static bool check_divergence(void) {
    for (int i = 0; i < 3; i++) {
        if (fabs(linear_position_W[i]) > 1000.0 || 
            fabs(linear_velocity_W[i]) > 100.0 || 
            fabs(angular_velocity_B[i]) > 100.0) return true;
    }
    for (int i = 0; i < 4; i++)
        if (omega_next[i] < 0 || omega_next[i] > 1000) return true;
    return false;
}

// Update history with current state
static void update_history(void) {
    // Shift history
    if (history_len == SEQ_LENGTH) {
        for (int i = 0; i < SEQ_LENGTH - 1; i++) {
            history[i] = history[i + 1];
        }
    } else {
        history_len++;
    }
    
    // Add new state
    StateVector *current = &history[history_len - 1];
    for (int i = 0; i < 3; i++) {
        current->pos_d[i] = linear_position_d_W[i];
        current->ang_vel[i] = angular_velocity_B[i];
        current->acc[i] = linear_acceleration_B[i];
    }
    current->yaw_d = yaw_d;
    for (int i = 0; i < 4; i++) {
        current->omega[i] = omega[i];
    }
}

// Prepare input for transformer
static void prepare_transformer_input(double* input) {
    // Current conditions (same for all sequence steps)
    double conditions[CONDITION_FEATURES] = {
        linear_position_d_W[0], linear_position_d_W[1], 
        linear_position_d_W[2], yaw_d
    };
    
    // Normalize conditions
    for (int f = 0; f < CONDITION_FEATURES; f++) {
        conditions[f] = normalize(conditions[f], 
                                normalization_data.mins[f], 
                                normalization_data.maxs[f]);
    }
    
    // Fill sequence data
    for (int s = 0; s < SEQ_LENGTH; s++) {
        // Copy conditions
        for (int f = 0; f < CONDITION_FEATURES; f++) {
            input[s * INPUT_FEATURES + f] = conditions[f];
        }
        
        // Add sequence features if we have them
        if (s < history_len) {
            int idx = CONDITION_FEATURES;
            // Angular velocities
            for (int i = 0; i < 3; i++) {
                input[s * INPUT_FEATURES + idx] = normalize(
                    history[s].ang_vel[i],
                    normalization_data.mins[idx],
                    normalization_data.maxs[idx]
                );
                idx++;
            }
            // Accelerations
            for (int i = 0; i < 3; i++) {
                input[s * INPUT_FEATURES + idx] = normalize(
                    history[s].acc[i],
                    normalization_data.mins[idx],
                    normalization_data.maxs[idx]
                );
                idx++;
            }
            // Rotor speeds
            for (int i = 0; i < 4; i++) {
                input[s * INPUT_FEATURES + idx] = normalize(
                    history[s].omega[i],
                    normalization_data.mins[idx],
                    normalization_data.maxs[idx]
                );
                idx++;
            }
        } else {
            // Zero padding for missing history
            for (int f = CONDITION_FEATURES; f < INPUT_FEATURES; f++) {
                input[s * INPUT_FEATURES + f] = 0.0;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <weights_file>\n", argv[0]);
        return 1;
    }

    // Load normalization parameters
    normalization_data = load_csv("2024-12-29_6-25-1_control_data.csv");

    // Initialize transformer weights
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
        printf("Failed to load weights from %s\n", argv[1]);
        return 1;
    }

    // Allocate transformer buffers
    double* transformer_input = malloc(BATCH_SIZE * (SEQ_LENGTH + 1) * INPUT_FEATURES * sizeof(double));
    Tensor hidden = {malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double)), NULL, NULL, BATCH_SIZE * SEQ_LENGTH * D_MODEL};
    Tensor temp = {malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double)), NULL, NULL, BATCH_SIZE * SEQ_LENGTH * D_MODEL};
    Tensor output = {malloc(BATCH_SIZE * SEQ_LENGTH * SEQUENCE_FEATURES * sizeof(double)), NULL, NULL, BATCH_SIZE * SEQ_LENGTH * SEQUENCE_FEATURES};
    double* q_buf = malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double* k_buf = malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double* v_buf = malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double* s_buf = malloc(BATCH_SIZE * N_HEAD * SEQ_LENGTH * SEQ_LENGTH * sizeof(double));
    double* mid_buf = malloc(BATCH_SIZE * SEQ_LENGTH * (D_MODEL * 4) * sizeof(double));

    // Initialize rendering
    Mesh* meshes[] = {create_mesh("sim/rasterizer/drone.obj", "sim/rasterizer/drone.bmp"), 
                      create_mesh("sim/rasterizer/ground.obj", "sim/rasterizer/ground.bmp")};
    uint8_t *frame_buffer = calloc(WIDTH * HEIGHT * 3, sizeof(uint8_t));
    ge_GIF *gif = ge_new_gif("transformer_flight.gif", WIDTH, HEIGHT, 4, -1, 0);
    transform_mesh(meshes[1], (double[3]){0.0, -0.5, 0.0}, 1.0, (double[9]){1,0,0, 0,1,0, 0,0,1});

    double t_physics = 0.0, t_control = 0.0, t_render = 0.0;

    // Set target position
    linear_position_d_W[0] = 2.0;
    linear_position_d_W[1] = 2.0;
    linear_position_d_W[2] = 2.0;
    yaw_d = M_PI / 4;

    while (!is_stable() || !is_at_target_position()) {
        if (check_divergence()) {
            printf("Simulation diverged.\n");
            return 1;
        }

        update_drone_physics(DT_PHYSICS);
        t_physics += DT_PHYSICS;
        
        if (t_control <= t_physics) {
            update_history();
            
            if (history_len == SEQ_LENGTH) {
                prepare_transformer_input(transformer_input);
                
                forward_pass(transformer_input, &output, &hidden, &temp,
                           &W_seq, &W_cond, W_q, W_k, W_v, W_o, W_ff1, W_ff2, &W_out,
                           q_buf, k_buf, v_buf, s_buf, mid_buf);

                // Use last sequence prediction
                int last_seq = (SEQ_LENGTH - 1) * SEQUENCE_FEATURES;
                for (int i = 0; i < 4; i++) {
                    omega_next[i] = denormalize(
                        output.data[last_seq + i + 6],  // Skip ang_vel and acc
                        normalization_data.mins[i + CONDITION_FEATURES + 6],
                        normalization_data.maxs[i + CONDITION_FEATURES + 6]
                    );
                }
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

    // Cleanup
    free(frame_buffer);
    free_meshes(meshes, 2);
    ge_close_gif(gif);

    // Free transformer resources
    free(transformer_input);
    free(hidden.data);
    free(temp.data);
    free(output.data);
    free(q_buf);
    free(k_buf);
    free(v_buf);
    free(s_buf);
    free(mid_buf);
    
    free(W_seq.data);
    free(W_cond.data);
    free(W_out.data);
    
    for (int l = 0; l < N_LAYERS; l++) {
        free(W_q[l].data);
        free(W_k[l].data);
        free(W_v[l].data);
        free(W_o[l].data);
        free(W_ff1[l].data);
        free(W_ff2[l].data);
    }

    free(normalization_data.data);
    free(normalization_data.mins);
    free(normalization_data.maxs);
    
    return 0;
}