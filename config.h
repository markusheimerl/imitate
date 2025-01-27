#ifndef CONFIG_H
#define CONFIG_H

#define DT_PHYSICS (1.0/1000.0)
#define DT_CONTROL (1.0/60.0)
#define DT_RENDER (1.0/24.0)

#define STATE_DIM 15
#define ACTION_DIM 8
#define MAX_STEPS 1000
#define NUM_ROLLOUTS 64

#define GAMMA 0.999
#define MAX_STD 3.0
#define MIN_STD 1e-5

#define MAX_MEAN 55.0
#define MIN_MEAN 45.0

#define MIN_DISTANCE 0.1
#define MAX_DISTANCE 2.0

#endif