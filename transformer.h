#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "grad.h"

void assert_float_eq(float a, float b, float eps, const char* msg) {
    if (fabsf(a - b) > eps) {
        printf("ASSERTION FAILED: %s\n", msg);
        printf("Expected: %f, Got: %f\n", b, a);
        exit(1);
    }
}

#endif // TRANSFORMER_H