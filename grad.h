#ifndef GRAD_H
#define GRAD_H
#include <stdlib.h>
#include <math.h>
#include <time.h>

double dot(double *v1, double *v2, int n) {
    double r = 0.0;
    for (int i = 0; i < n; i++) r += v1[i] * v2[i];
    return r;
}

double l_relu(double x) {
    return x > 0 ? x : 0.1 * x;
}

void he_init(double *W, int n_in, int n_out) {
    double stddev = sqrt(2.0 / n_in);
    for (int i = 0; i < n_in * n_out; i++) {
        double u1 = (double)rand() / RAND_MAX;
        double u2 = (double)rand() / RAND_MAX;
        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        W[i] = z * stddev;
    }
}

void init_linear(double **W, double **b, int n_in, int n_out) {
    *W = malloc(n_in * n_out * sizeof(double));
    *b = calloc(n_out, sizeof(double));
    he_init(*W, n_in, n_out);
}

#endif // GRAD_H