#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include "sim.h"
#include "util.h"
#include "grad.h"

#define D1 64
#define D2 32
#define D3 16
#define M_IN 6
#define M_OUT 8  // 4 means, 4 variances

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define DT_RENDER   (1.0 / 30.0)

void forward(double *W1, double *b1, double *W2, double *b2, double *W3, double *b3, double *W4, double *b4, double *input, double *h1, double *h2, double *h3, double *output) {
    for (int i = 0; i < D1; i++) h1[i] = l_relu(b1[i] + dot(&W1[i * M_IN], input, M_IN));
    for (int i = 0; i < D2; i++) h2[i] = l_relu(b2[i] + dot(&W2[i * D1], h1, D1));
    for (int i = 0; i < D3; i++) h3[i] = l_relu(b3[i] + dot(&W3[i * D2], h2, D2));
    for (int i = 0; i < M_OUT / 2; i++) output[i] = b4[i] + dot(&W4[i * D3], h3, D3);
    for (int i = M_OUT / 2; i < M_OUT; i++) output[i] = exp(b4[i] + dot(&W4[i * D3], h3, D3));
}

int main(int argc, char *argv[]) {
    srand(time(NULL));

    double *W1, *b1, *W2, *b2, *W3, *b3, *W4, *b4;
    double *h1, *h2, *h3, input[M_IN], output[M_OUT];
    init_linear(&W1, &b1, M_IN, D1);
    init_linear(&W2, &b2, D1, D2);
    init_linear(&W3, &b3, D2, D3);
    init_linear(&W4, &b4, D3, M_OUT);
    h1 = malloc(D1 * sizeof(double));
    h2 = malloc(D2 * sizeof(double));
    h3 = malloc(D3 * sizeof(double));

    if(argc > 1) load_weights(argv[1], (double*[]){W1, b1, W2, b2, W3, b3, W4, b4}, (int[]){M_IN * D1, D1, D1 * D2, D2, D2 * D3, D3, D3 * M_OUT, M_OUT}, 8);

    Sim* sim = init_sim();

    #ifdef RENDER
    double t_render = 0.0;
    int max_rollouts = 1;
    #endif

    #ifdef LOG
    double *rewards = NULL;
    char **trajectory_lines = NULL;
    int reward_count = 0;
    int max_rollouts = 1000;

    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    char filename[100];
    sprintf(filename, "%d-%d-%d_%d-%d-%d_trajectory.csv", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    FILE* csv_file = fopen(filename, "w");
    fprintf(csv_file, "pos[0],pos[1],pos[2],vel[0],vel[1],vel[2],ang_vel[0],ang_vel[1],ang_vel[2],acc_s[0],acc_s[1],acc_s[2],gyro_s[0],gyro_s[1],gyro_s[2],mean[0],mean[1],mean[2],mean[3],var[0],var[1],var[2],var[3],omega[0],omega[1],omega[2],omega[3],reward,discounted_return\n");
    #endif

    for(int rollout = 0; rollout < max_rollouts; rollout++) {
        printf("\rRollout %d/%d ", rollout + 1, max_rollouts);
        fflush(stdout);

        reset_quad(sim->quad, 0.0, 1.0, 0.0);

        double t_physics = 0.0, t_control = 0.0;
        while (t_physics < 3000.0 && sim->quad->linear_position_W[1] > 0.2 && fabs(sim->quad->linear_position_W[0]) < 2.0 && fabs(sim->quad->linear_position_W[1]) < 2.0 && fabs(sim->quad->linear_position_W[2]) < 2.0) {
            
            update_quad(sim->quad, DT_PHYSICS);
            t_physics += DT_PHYSICS;

            if (t_control <= t_physics) {
                for(int i = 0; i < 3; i++) {
                    input[i] = sim->quad->linear_acceleration_B_s[i];
                    input[i+3] = sim->quad->angular_velocity_B_s[i];
                }
                forward(W1, b1, W2, b2, W3, b3, W4, b4, input, h1, h2, h3, output);

                // Sample actions from Gaussian distributions
                for(int i = 0; i < 4; i++) {
                    double mean = output[i];
                    double std = sqrt(output[i + 4]);
                    double u1 = (double)rand() / RAND_MAX;
                    double u2 = (double)rand() / RAND_MAX;
                    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                    sim->quad->omega_next[i] = mean + std * z;
                    if (rollout == 0 && t_physics < 0.1) printf("Motor %d: mean=%.3f, std=%.3f, action=%.3f\n", i, mean, std, sim->quad->omega_next[i]);
                }

                #ifdef LOG
                double height_error = fabs(sim->quad->linear_position_W[1] - 1.0);
                double horizontal_error = fabs(sim->quad->linear_position_W[0]) + fabs(sim->quad->linear_position_W[2]);
                double velocity_penalty = (fabs(sim->quad->linear_velocity_W[0]) + fabs(sim->quad->linear_velocity_W[1]) + fabs(sim->quad->linear_velocity_W[2])) * 0.1;
                double reward = 1.0 + (1.0 / (1.0 + height_error) - horizontal_error - velocity_penalty);

                rewards = realloc(rewards, (reward_count + 1) * sizeof(double));
                rewards[reward_count] = reward;
                trajectory_lines = realloc(trajectory_lines, (reward_count + 1) * sizeof(char*));
                trajectory_lines[reward_count] = malloc(1024);
                snprintf(trajectory_lines[reward_count], 1024, 
                        "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,0.0\n",
                        sim->quad->linear_position_W[0], sim->quad->linear_position_W[1], sim->quad->linear_position_W[2],
                        sim->quad->linear_velocity_W[0], sim->quad->linear_velocity_W[1], sim->quad->linear_velocity_W[2],
                        sim->quad->angular_velocity_B[0], sim->quad->angular_velocity_B[1], sim->quad->angular_velocity_B[2],
                        sim->quad->linear_acceleration_B_s[0], sim->quad->linear_acceleration_B_s[1], sim->quad->linear_acceleration_B_s[2],
                        sim->quad->angular_velocity_B_s[0], sim->quad->angular_velocity_B_s[1], sim->quad->angular_velocity_B_s[2],
                        output[0], output[1], output[2], output[3],  // means
                        output[4], output[5], output[6], output[7],  // variances
                        sim->quad->omega_next[0], sim->quad->omega_next[1], sim->quad->omega_next[2], sim->quad->omega_next[3],
                        reward);
                reward_count++;
                #endif

                #ifdef RENDER
                print_quad(sim->quad);
                fflush(stdout);
                #endif

                t_control += DT_CONTROL;
            }

            #ifdef RENDER
            if (t_render <= t_physics) {
                render_sim(sim);
                t_render += DT_RENDER;
            }
            #endif
        }

        #ifdef LOG
        // Calculate and write trajectory with discounted returns
        if (reward_count > 0) {
            for(int i = 0; i < reward_count; i++) {
                double discounted_return = 0.0;
                double gamma = 0.99;
                double discount = 1.0;
                for(int j = i; j < reward_count; j++) {
                    discounted_return += discount * rewards[j];
                    discount *= gamma;
                }
                char *line = trajectory_lines[i];
                char *last_comma = strrchr(line, ',');
                sprintf(last_comma + 1, "%f\n", discounted_return);
                fprintf(csv_file, "%s", line);
                free(line);
            }
            
            free(trajectory_lines);
            free(rewards);
            rewards = NULL;
            trajectory_lines = NULL;
            reward_count = 0;
        }
        #endif
    }

    printf("\nSimulation complete\n");

    if (argc > 1) {
        save_weights(argv[1], (double*[]){W1, b1, W2, b2, W3, b3, W4, b4}, (int[]){M_IN * D1, D1, D1 * D2, D2, D2 * D3, D3, D3 * M_OUT, M_OUT}, 8);
    }
    else {
        #ifndef LOG
        time_t t = time(NULL);
        struct tm tm = *localtime(&t);
        char filename[100];
        #endif
        sprintf(filename, "%d-%d-%d_%d-%d-%d_policy_weights.bin", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
        save_weights(filename, (double*[]){W1, b1, W2, b2, W3, b3, W4, b4}, (int[]){M_IN * D1, D1, D1 * D2, D2, D2 * D3, D3, D3 * M_OUT, M_OUT}, 8);
    }

    free(W1); free(b1); free(W2); free(b2); free(W3); free(b3); free(W4); free(b4);
    free(h1); free(h2); free(h3);
    
    #ifdef RENDER
    free_sim(sim);
    #endif

    #ifdef LOG
    fclose(csv_file);
    #endif

    return 0;
}