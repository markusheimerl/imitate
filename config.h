#ifndef CONFIG_H
#define CONFIG_H

#include "quad.h"

#define DT_PHYSICS (1.0/1000.0)
#define DT_CONTROL (1.0/60.0)
#define DT_RENDER (1.0/24.0)

#define STATE_DIM 6  // 3 accel + 3 gyro
#define ACTION_DIM 8
#define HISTORY_LENGTH 8
// At 60Hz control rate (DT_CONTROL = 1/60s), taking every 8th state means
// each history sample is 8/60 = 0.133s apart. With HISTORY_LENGTH = 16,
// this gives us a total history window of 16 * 8/60 = 2.133s
#define HISTORY_FREQUENCY 64
#define TOTAL_STATE_DIM (STATE_DIM * HISTORY_LENGTH)
#define MAX_STEPS 1000
#define NUM_ROLLOUTS 128

#define GAMMA 0.999
#define MAX_STD 3.0
#define MIN_STD 1e-5

#define MAX_MEAN (OMEGA_MAX - 4.0 * MAX_STD)
#define MIN_MEAN (OMEGA_MIN + 4.0 * MAX_STD)

#define MIN_DISTANCE 0.01
#define MAX_DISTANCE 0.5

#endif