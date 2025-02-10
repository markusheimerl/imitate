#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "sim/quad.h"

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define SIM_TIME    10.0

// Helper function to get random value in range [min, max]
double random_range(double min, double max) {
    return min + (double)rand() / RAND_MAX * (max - min);
}

int main() {
    srand(time(NULL) ^ getpid());
    
    double target[7] = {
        random_range(-2.0, 2.0),    // x: Range -2 to 2
        random_range(1.0, 3.0),     // y: Range 1 to 3    
        random_range(-2.0, 2.0),    // z: Range -2 to 2
        0.0, 0.0, 0.0,              // vx, vy, vz
        random_range(0.0, 2*M_PI)}; // yaw: Range 0 to 2π
    
    printf("Target position: (%.2f, %.2f, %.2f) with yaw: %.2f rad\n", 
           target[0], target[1], target[2], target[6]);
    
    Quad* quad = create_quad(0.0, 0.0, 0.0);
    double t_physics = 0.0;
    double t_control = 0.0;

    for(int i = 0; i < (int)(SIM_TIME / DT_PHYSICS); i++) {
        // Physics update
        if (t_physics >= DT_PHYSICS) {
            update_quad(quad, DT_PHYSICS);
            t_physics = 0.0;
        }
        
        // Control update
        if (t_control >= DT_CONTROL) {
            control_quad(quad, target);
            t_control = 0.0;
        }
        
        // Increment timers
        t_physics += DT_PHYSICS;
        t_control += DT_PHYSICS;
    }

    printf("Final position: (%.2f, %.2f, %.2f) with yaw: %.2f rad\n", 
           quad->linear_position_W[0], quad->linear_position_W[1], 
           quad->linear_position_W[2], atan2(quad->R_W_B[3], quad->R_W_B[0]));

    free(quad);
    return 0;
}