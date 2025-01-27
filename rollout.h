#ifndef ROLLOUT_H
#define ROLLOUT_H

#include "config.h"

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