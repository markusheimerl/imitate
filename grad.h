#ifndef GRAD_H
#define GRAD_H

#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BETA1 0.9
#define BETA2 0.999
#define EPSILON 1e-8
#define DECAY 0.01

void save_weights(const char* filename, double** params, int* sizes, int num) {
    FILE *f = fopen(filename, "wb");
    for(int i = 0; i < num; i++) {
        fwrite(params[i], sizeof(double), sizes[i], f);
    }
    fclose(f);
}

void load_weights(const char* filename, double** params, int* sizes, int num) {
    FILE *f = fopen(filename, "rb");
    for(int i = 0; i < num; i++) {
        fread(params[i], sizeof(double), sizes[i], f);
    }
    fclose(f);
}

void adam_update(double *param, double *grad, double *m, double *v, int size, int step, double learning_rate) {
    double lr_t = learning_rate * sqrt(1.0 - pow(BETA2, step)) / (1.0 - pow(BETA1, step));
    for(int i = 0; i < size; i++) {
        m[i] = BETA1 * m[i] + (1-BETA1) * grad[i];
        v[i] = BETA2 * v[i] + (1-BETA2) * grad[i] * grad[i];
        param[i] -= lr_t * (m[i] / (sqrt(v[i]) + EPSILON) + DECAY * param[i]);
    }
}

double dot(double *v1, double *v2, int n) {
    double r = 0.0;
    for (int i = 0; i < n; i++) r += v1[i] * v2[i];
    return r;
}

void d_dot_right(double *result, double *grad, double *weights, int n_in, int n_out) {
    for(int i = 0; i < n_in; i++) {
        result[i] = 0;
        for(int j = 0; j < n_out; j++) {
            result[i] += grad[j] * weights[j*n_in + i];
        }
    }
}

void d_dot_left(double *weights, double *grad, double *input, int n_in, int n_out) {
    for(int i = 0; i < n_out; i++) {
        for(int j = 0; j < n_in; j++) {
            weights[i*n_in + j] = grad[i] * input[j];
        }
    }
}

double l_relu(double x) {
    return x > 0 ? x : 0.1 * x;
}

double d_l_relu(double x) {
    return x > 0 ? 1.0 : 0.1;
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