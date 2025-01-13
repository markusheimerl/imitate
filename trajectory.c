#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include "sim.h"
#include "util.h"
#include "grad.h"
#include "log.h"

#define D1 64
#define D2 32
#define D3 16
#define M_IN 6
#define M_OUT 8  // 4 means, 4 variances

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define DT_RENDER   (1.0 / 30.0)

int calculate_reward(Sim* sim) {
    double height_error = fabs(sim->quad->linear_position_W[1] - 1.0);
    double horizontal_error = fabs(sim->quad->linear_position_W[0]) + fabs(sim->quad->linear_position_W[2]);
    double velocity_penalty = (fabs(sim->quad->linear_velocity_W[0]) + fabs(sim->quad->linear_velocity_W[1]) + fabs(sim->quad->linear_velocity_W[2])) * 0.1;
    return 1.0 + (1.0 / (1.0 + height_error) - horizontal_error - velocity_penalty);
}

void sample_action(double *output, double *omega_next) {
    for(int i = 0; i < 4; i++) {
        double mean = output[i];
        double std = sqrt(output[i + 4]);
        omega_next[i] = mean + std * sqrt(-2.0 * log((double)rand() / RAND_MAX)) * cos(2.0 * M_PI * ((double)rand() / RAND_MAX));
    }
}

void forward_policy(double *W1, double *b1, double *W2, double *b2, double *W3, double *b3, double *W4, double *b4, double *input, double *h1, double *h2, double *h3, double *output) {
    for (int i = 0; i < D1; i++) h1[i] = l_relu(b1[i] + dot(&W1[i * M_IN], input, M_IN));
    for (int i = 0; i < D2; i++) h2[i] = l_relu(b2[i] + dot(&W2[i * D1], h1, D1));
    for (int i = 0; i < D3; i++) h3[i] = l_relu(b3[i] + dot(&W3[i * D2], h2, D2));
    for (int i = 0; i < M_OUT / 2; i++) output[i] = b4[i] + dot(&W4[i * D3], h3, D3);
    for (int i = M_OUT / 2; i < M_OUT; i++) output[i] = exp(b4[i] + dot(&W4[i * D3], h3, D3));
}

int main(int argc, char *argv[]) {

    #if defined(RENDER)
        Sim* sim = init_sim(true);
        int max_rollouts = 1;
        double t_render = 0.0;
    #elif defined(LOG)
        Sim* sim = init_sim(false);
        Logger* logger = init_logger();
        int max_rollouts = 1000;
    #else
        return 1;
    #endif

    srand(time(NULL));

    double *W1, *b1, *W2, *b2, *W3, *b3, *W4, *b4;
    double* h1 = malloc(D1 * sizeof(double));
    double* h2 = malloc(D2 * sizeof(double));
    double* h3 = malloc(D3 * sizeof(double));

    if(argc > 1){
        load_weights(argv[1], (double*[]){W1, b1, W2, b2, W3, b3, W4, b4}, (int[]){M_IN * D1, D1, D1 * D2, D2, D2 * D3, D3, D3 * M_OUT, M_OUT}, 8);
    }else{
        init_linear(&W1, &b1, M_IN, D1);
        init_linear(&W2, &b2, D1, D2);
        init_linear(&W3, &b3, D2, D3);
        init_linear(&W4, &b4, D3, M_OUT);
    }

    for(int rollout = 0; rollout < max_rollouts; rollout++) {
        #if defined(LOG)
            printf("\rRollout %d/%d ", rollout + 1, max_rollouts);
            fflush(stdout);
        #endif

        reset_quad(sim->quad, 0.0, 1.0, 0.0);

        double t_physics = 0.0, t_control = 0.0;
        while (t_physics < 3000.0 && sim->quad->linear_position_W[1] > 0.2 && fabs(sim->quad->linear_position_W[0]) < 2.0 && fabs(sim->quad->linear_position_W[1]) < 2.0 && fabs(sim->quad->linear_position_W[2]) < 2.0) {
            
            update_quad(sim->quad, DT_PHYSICS);
            t_physics += DT_PHYSICS;

            #ifdef RENDER
            if (t_render <= t_physics) {
                render_sim(sim);
                t_render += DT_RENDER;
            }
            #endif

            if (t_control <= t_physics) {
                double input[M_IN] = {sim->quad->linear_acceleration_B_s[0], sim->quad->linear_acceleration_B_s[1], sim->quad->linear_acceleration_B_s[2], sim->quad->angular_velocity_B_s[0], sim->quad->angular_velocity_B_s[1], sim->quad->angular_velocity_B_s[2]};
                double output[M_OUT] = {0.0};
                forward_policy(W1, b1, W2, b2, W3, b3, W4, b4, input, h1, h2, h3, output);
                sample_action(output, sim->quad->omega_next);

                #if defined(RENDER)
                    print_quad(sim->quad);
                    fflush(stdout);
                #elif defined(LOG)               
                    log_trajectory(logger, sim, output, calculate_reward(sim));
                #endif

                t_control += DT_CONTROL;
            }
        }

        #ifdef LOG
            save_logger(logger);
        #endif
    }

    printf("\n");

    if (argc > 1) {
        save_weights(argv[1], (double*[]){W1, b1, W2, b2, W3, b3, W4, b4}, (int[]){M_IN * D1, D1, D1 * D2, D2, D2 * D3, D3, D3 * M_OUT, M_OUT}, 8);
    }
    else {
        char filename[100];
        strftime(filename, 100, "%Y-%m-%d_%H-%M-%S_policy_weights.bin", localtime(&(time_t){time(NULL)}));
        save_weights(filename, (double*[]){W1, b1, W2, b2, W3, b3, W4, b4}, (int[]){M_IN * D1, D1, D1 * D2, D2, D2 * D3, D3, D3 * M_OUT, M_OUT}, 8);
    }

    free(W1); free(b1); free(W2); free(b2); free(W3); free(b3); free(W4); free(b4); free(h1); free(h2); free(h3);
    free_sim(sim);

    #ifdef LOG
        free_logger(logger);
    #endif

    return 0;
}