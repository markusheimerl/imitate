#ifndef CONFIG_H
#define CONFIG_H

#include "quad.h"

#define DT_PHYSICS (1.0/1000.0)
#define DT_CONTROL (1.0/60.0)
#define DT_RENDER (1.0/24.0)

#define HISTORY_LENGTH 8  // Remember last 8 timesteps
#define SENSOR_DIM 6      // 3 accel + 3 gyro
#define ACTION_DIM 8      // 4 means + 4 stds
#define STATE_DIM ((SENSOR_DIM + ACTION_DIM) * HISTORY_LENGTH)  // Both sensors and past actions
#define MAX_STEPS 1000
#define NUM_ROLLOUTS 128

#define GAMMA 0.999
#define MAX_STD 3.0
#define MIN_STD 1e-5

#define MAX_MEAN (OMEGA_MAX - 4.0 * MAX_STD)
#define MIN_MEAN (OMEGA_MIN + 4.0 * MAX_STD)

typedef struct {
    double* buffer;     // Circular buffer for state history
    int head;          // Current position in buffer
    int filled;        // Number of valid entries
} History;

History* create_history() {
    History* h = malloc(sizeof(History));
    h->buffer = calloc(STATE_DIM, sizeof(double));
    h->head = 0;
    h->filled = 0;
    return h;
}

void update_history(History* h, double* sensors, double* actions) {
    // Calculate where this new entry should go
    int base = h->head * (SENSOR_DIM + ACTION_DIM);
    
    // Copy new sensor readings and actions
    memcpy(h->buffer + base, sensors, SENSOR_DIM * sizeof(double));
    memcpy(h->buffer + base + SENSOR_DIM, actions, ACTION_DIM * sizeof(double));
    
    // Update head and filled count
    h->head = (h->head + 1) % HISTORY_LENGTH;
    h->filled = fmin(h->filled + 1, HISTORY_LENGTH);
}

void get_state_vector(History* h, double* state) {
    memset(state, 0, STATE_DIM * sizeof(double));
    
    // Copy from newest to oldest
    for(int i = 0; i < h->filled; i++) {
        int src_idx = ((h->head - i - 1 + HISTORY_LENGTH) % HISTORY_LENGTH) * (SENSOR_DIM + ACTION_DIM);
        int dst_idx = i * (SENSOR_DIM + ACTION_DIM);
        memcpy(state + dst_idx, 
               h->buffer + src_idx, 
               (SENSOR_DIM + ACTION_DIM) * sizeof(double));
    }
}

void free_history(History* h) {
    free(h->buffer);
    free(h);
}

typedef struct {
    double** states;    // [MAX_STEPS][STATE_DIM]
    double** actions;   // [MAX_STEPS][ACTION_DIM]
    double* rewards;    // [MAX_STEPS]
    History* history;   // Temporal history
    int length;
} Rollout;

Rollout* create_rollout() {
    Rollout* r = malloc(sizeof(Rollout));
    r->states = malloc(MAX_STEPS * sizeof(double*));
    r->actions = malloc(MAX_STEPS * sizeof(double*));
    r->rewards = malloc(MAX_STEPS * sizeof(double));
    r->history = create_history();
    
    for(int i = 0; i < MAX_STEPS; i++) {
        r->states[i] = malloc(STATE_DIM * sizeof(double));
        r->actions[i] = malloc(ACTION_DIM * sizeof(double));
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
    free_history(r->history);
    free(r);
}

#endif