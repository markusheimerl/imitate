#ifndef CONFIG_H
#define CONFIG_H

#include "quad.h"

#define DT_PHYSICS (1.0/1000.0)
#define DT_CONTROL (1.0/60.0)
#define DT_RENDER (1.0/24.0)

#define STATE_DIM 6  // 3 accel + 3 gyro
#define ACTION_DIM 8
#define MAX_STEPS 1000
#define NUM_ROLLOUTS 128

#define GAMMA 0.999
#define MAX_STD 3.0
#define MIN_STD 1e-5

#define MAX_MEAN (OMEGA_MAX - 4.0 * MAX_STD)
#define MIN_MEAN (OMEGA_MIN + 4.0 * MAX_STD)

typedef struct {
    double** states;    // [MAX_STEPS][STATE_DIM]
    double** actions;   // [MAX_STEPS][NUM_ROTORS]
    double* rewards;    // [MAX_STEPS]
    int length;
} Rollout;

Rollout* create_rollout() {
    Rollout* r = malloc(sizeof(Rollout));
    r->states = malloc(MAX_STEPS * sizeof(double*));
    r->actions = malloc(MAX_STEPS * sizeof(double*));
    r->rewards = malloc(MAX_STEPS * sizeof(double));
    
    for(int i = 0; i < MAX_STEPS; i++) {
        r->states[i] = malloc(STATE_DIM * sizeof(double));
        r->actions[i] = malloc(4 * sizeof(double));
    }
    return r;
}

void free_rollout(Rollout* r) {
    for(int i = 0; i < MAX_STEPS; i++) {
        free(r->states[i]);
        free(r->actions[i]);
    }
    free(r->states);
    free(r->actions);
    free(r->rewards);
    free(r);
}

#endif