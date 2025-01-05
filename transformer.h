#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define CONDITION_FEATURES 4
#define SEQUENCE_FEATURES 10
#define INPUT_FEATURES (CONDITION_FEATURES + SEQUENCE_FEATURES)
#define BATCH_SIZE 8
#define SEQ_LENGTH 16
#define D_MODEL 16
#define N_HEAD 2
#define N_LAYERS 2
#define EPSILON 1e-6
#define LEARNING_RATE 0.01
#define TRAINING_STEPS 1000

typedef struct { double *data; int rows, cols; double *mins, *maxs; } Dataset;
typedef struct { double *data; double *m; double *v; int size; } Tensor;

double normalize(double v, double min, double max) { return max == min ? 0 : 2.0 * (v - min) / (max - min) - 1.0; }
double denormalize(double v, double min, double max) { return (v + 1.0) * (max - min) / 2.0 + min; }
double randn() { return sqrt(-2.0 * log((double)rand() / RAND_MAX)) * cos(2.0 * M_PI * (double)rand() / RAND_MAX); }

void save_loss(const char* filename, double loss, int step) {
    FILE* f = fopen(filename, step == 0 ? "w" : "a");
    if (!f) { printf("Error opening loss file\n"); return; }
    fprintf(f, "%d,%f\n", step, loss);
    fclose(f);
}

Dataset load_csv(const char* filename) {
    Dataset ds = {NULL, 0, INPUT_FEATURES, calloc(INPUT_FEATURES, sizeof(double)), calloc(INPUT_FEATURES, sizeof(double))};
    char line[1024];
    double* tmp = malloc(1000 * INPUT_FEATURES * sizeof(double));
    FILE* f = fopen(filename, "r");
    if (!f || !fgets(line, 1024, f)) { printf("File error\n"); exit(1); }
    
    for (int i = 0; i < INPUT_FEATURES; i++) ds.mins[i]=INFINITY, ds.maxs[i]=-INFINITY;
    
    while (fgets(line, 1024, f)) {
        if (ds.rows >= 1000) tmp = realloc(tmp, (ds.rows*2) * INPUT_FEATURES * sizeof(double));
        char* tok = strtok(line, ",");
        for (int i = 0; i < INPUT_FEATURES && tok; i++, tok = strtok(NULL, ",")) {
            tmp[ds.rows * INPUT_FEATURES + i] = atof(tok);
            ds.mins[i] = fmin(ds.mins[i], tmp[ds.rows * INPUT_FEATURES + i]);
            ds.maxs[i] = fmax(ds.maxs[i], tmp[ds.rows * INPUT_FEATURES + i]);
        }
        ds.rows++;
    }
    
    for (int i = 0; i < ds.rows * INPUT_FEATURES; i++) tmp[i] = normalize(tmp[i], ds.mins[i % INPUT_FEATURES], ds.maxs[i % INPUT_FEATURES]);
    
    ds.data = tmp;
    fclose(f);
    return ds;
}

void save_weights(const char* filename, const Tensor* ws, const Tensor* wc,
                 const Tensor* wq, const Tensor* wk, const Tensor* wv, const Tensor* wo,
                 const Tensor* wf1, const Tensor* wf2, const Tensor* wout) {
    FILE* f = fopen(filename, "wb");
    if (!f) { printf("Error opening file for writing weights\n"); return; }

    fwrite(ws->data, sizeof(double), ws->size, f);
    fwrite(wc->data, sizeof(double), wc->size, f);
    for (int l = 0; l < N_LAYERS; l++) {
        fwrite(wq[l].data, sizeof(double), wq[l].size, f);
        fwrite(wk[l].data, sizeof(double), wk[l].size, f);
        fwrite(wv[l].data, sizeof(double), wv[l].size, f);
        fwrite(wo[l].data, sizeof(double), wo[l].size, f);
        fwrite(wf1[l].data, sizeof(double), wf1[l].size, f);
        fwrite(wf2[l].data, sizeof(double), wf2[l].size, f);
    }
    fwrite(wout->data, sizeof(double), wout->size, f);
    fclose(f);
}


int load_weights(const char* filename, Tensor* ws, Tensor* wc,
                Tensor* wq, Tensor* wk, Tensor* wv, Tensor* wo,
                Tensor* wf1, Tensor* wf2, Tensor* wout) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        printf("Error opening file for reading weights\n");
        return 0;
    }

    size_t read = 0;
    read += fread(ws->data, sizeof(double), ws->size, f);
    read += fread(wc->data, sizeof(double), wc->size, f);
    for (int l = 0; l < N_LAYERS; l++) {
        read += fread(wq[l].data, sizeof(double), wq[l].size, f);
        read += fread(wk[l].data, sizeof(double), wk[l].size, f);
        read += fread(wv[l].data, sizeof(double), wv[l].size, f);
        read += fread(wo[l].data, sizeof(double), wo[l].size, f);
        read += fread(wf1[l].data, sizeof(double), wf1[l].size, f);
        read += fread(wf2[l].data, sizeof(double), wf2[l].size, f);
    }
    read += fread(wout->data, sizeof(double), wout->size, f);
    fclose(f);

    size_t expected_size = ws->size + wc->size + wout->size;
    for (int l = 0; l < N_LAYERS; l++) {
        expected_size += wq[l].size + wk[l].size + wv[l].size + wo[l].size + wf1[l].size + wf2[l].size;
    }

    if (read != expected_size) {
        printf("Warning: Read %zu elements, expected %zu\n", read, expected_size);
        return 0;
    }
    return 1;
}

#endif // TRANSFORMER_H