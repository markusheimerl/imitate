#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "grad/grad.h"
#include "grad/data.h"
#include "sim/sim.h"

#define STATE_DIM 12
#define ACTION_DIM 4
#define HIDDEN_DIM 64
#define MAX_STEPS 1000
#define DT_PHYSICS (1.0/1000.0)
#define DT_CONTROL (1.0/60.0)

void get_state(Quad* q, double* state) {
    memcpy(state, q->linear_position_W, 3 * sizeof(double));
    memcpy(state + 3, q->linear_velocity_W, 3 * sizeof(double));
    memcpy(state + 6, q->angular_velocity_B, 3 * sizeof(double));
    state[9] = q->R_W_B[0];
    state[10] = q->R_W_B[4];
    state[11] = q->R_W_B[8];
}

int main(int argc, char** argv) {
    srand(time(NULL));
    
    Net* policy = argc > 1 ? load_weights(argv[1]) : init_net(4, (int[]){STATE_DIM, HIDDEN_DIM, HIDDEN_DIM, ACTION_DIM});
    if(!policy) return 1;
    
    Sim* sim = init_sim(false);
    Data* data = malloc(sizeof(Data));
    data->n = MAX_STEPS;
    data->fx = STATE_DIM;
    data->fy = ACTION_DIM;
    data->X = malloc(MAX_STEPS * sizeof(double*));
    data->y = malloc(MAX_STEPS * sizeof(double*));
    
    double** act = malloc(4 * sizeof(double*));
    for(int i = 0; i < 4; i++) act[i] = malloc(policy->sz[i] * sizeof(double));
    for(int i = 0; i < MAX_STEPS; i++) {
        data->X[i] = malloc(STATE_DIM * sizeof(double));
        data->y[i] = malloc(ACTION_DIM * sizeof(double));
    }
    
    reset_quad(sim->quad, 0.0, 1.0, 0.0);
    printf("\nCollecting rollout...\n");
    
    double t_physics = 0.0, t_control = 0.0;
    int step = 0;
    
    while(step < MAX_STEPS) {
        update_quad(sim->quad, DT_PHYSICS);
        t_physics += DT_PHYSICS;
        
        if(t_control <= t_physics) {
            get_state(sim->quad, data->X[step]);
            fwd(policy, data->X[step], act);
            memcpy(data->y[step], act[3], ACTION_DIM * sizeof(double));
            
            for(int i = 0; i < ACTION_DIM; i++) {
                sim->quad->omega_next[i] = OMEGA_MIN + (OMEGA_MAX - OMEGA_MIN) * 
                                         (tanh(data->y[step][i]) + 1.0) / 2.0;
            }
            
            if(step % 100 == 0) printf("\rStep %d/%d", step, MAX_STEPS);
            t_control += DT_CONTROL;
            step++;
        }
    }
    
    const char* header = "px,py,pz,vx,vy,vz,wx,wy,wz,r11,r22,r33,m1,m2,m3,m4";
    char* fname = save_csv("rollout.csv", data, header);
    printf("\nSaved to: %s\n", fname);
    
    free(fname);
    free_data(data);
    for(int i = 0; i < 4; i++) free(act[i]);
    free(act);
    free_net(policy);
    free_sim(sim);
    
    return 0;
}