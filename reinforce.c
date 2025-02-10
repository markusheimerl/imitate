#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include "sim/quad.h"

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define SIM_TIME    20.0

int main() {
    srand(time(NULL));
    
    double target[7] = {
        (double)rand() / RAND_MAX * 4.0 - 2.0,   // x
        (double)rand() / RAND_MAX * 4.0,         // y    
        (double)rand() / RAND_MAX * 4.0 - 2.0,   // z
        0.0, 0.0, 0.0,                           // vx, vy, vz
        (double)rand() / RAND_MAX * 2.0 * M_PI}; // yaw
    
    printf("Target position: (%.2f, %.2f, %.2f) with yaw: %.2f rad\n", target[0], target[1], target[2], target[6]);
    
    Quad* quad = create_quad(0.0, 0.0, 0.0);
    double t_physics = 0.0;
    double t_control = 0.0;

    for(int i = 0; i < (int)(SIM_TIME / DT_PHYSICS); i++) {

        if (t_physics >= DT_PHYSICS) {
            update_quad(quad, DT_PHYSICS);
            t_physics = 0.0;
        }
        
        if (t_control >= DT_CONTROL) {
            control_quad(quad, target);
            t_control = 0.0;
        }
        
        t_physics += DT_PHYSICS;
        t_control += DT_PHYSICS;
    }

    printf("Final position: (%.2f, %.2f, %.2f) with yaw: %.2f rad\n", quad->linear_position_W[0], quad->linear_position_W[1], quad->linear_position_W[2], atan2(quad->R_W_B[3], quad->R_W_B[0]));

    free(quad);
    return 0;
}