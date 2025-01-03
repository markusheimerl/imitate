#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_LINE_LENGTH 1024
#define CONDITION_FEATURES 4
#define SEQUENCE_FEATURES 10
#define INPUT_FEATURES (CONDITION_FEATURES + SEQUENCE_FEATURES)
#define BATCH_SIZE 2
#define SEQ_LENGTH 16
#define D_MODEL 16
#define N_HEAD 4
#define N_LAYERS 2
#define EPSILON 1e-6
#define LEARNING_RATE 0.0001
#define TRAINING_STEPS 1000

typedef struct { double *data; int *dims, ndims, size; } Tensor;
typedef struct { double* data; int rows, cols; double *mins, *maxs; } Dataset;

double normalize(double v, double min, double max) { return max == min ? 0 : 2.0 * (v - min) / (max - min) - 1.0; }
double denormalize(double v, double min, double max) { return (v + 1.0) * (max - min) / 2.0 + min; }
double randn() { return sqrt(-2.0 * log((double)rand() / RAND_MAX)) * cos(2.0 * M_PI * (double)rand() / RAND_MAX); }

void rmsnorm(Tensor *out, const Tensor *in) {
    for (int b = 0; b < BATCH_SIZE; b++)
        for (int s = 0; s < SEQ_LENGTH; s++) {
            double ss = 0.0;
            int idx = (b * SEQ_LENGTH + s) * D_MODEL;
            for (int i = 0; i < D_MODEL; i++) ss += in->data[idx + i] * in->data[idx + i];
            double rms = sqrt(ss / D_MODEL + 1e-5);
            for (int i = 0; i < D_MODEL; i++) out->data[idx + i] = in->data[idx + i] / rms;
        }
}

double gelu(double x) { return 0.5 * x * (1.0 + tanh(sqrt(2.0/M_PI) * (x + 0.044715 * x * x * x))); }

void feedforward(Tensor *out, const Tensor *w1, const Tensor *w2, const Tensor *in) {
    double *mid = malloc(BATCH_SIZE * SEQ_LENGTH * (D_MODEL * 4) * sizeof(double));
    for (int b = 0; b < BATCH_SIZE; b++)
        for (int s = 0; s < SEQ_LENGTH; s++) {
            int idx = (b * SEQ_LENGTH + s);
            for (int h = 0; h < D_MODEL * 4; h++) {
                double sum = 0.0;
                for (int d = 0; d < D_MODEL; d++) 
                    sum += in->data[idx * D_MODEL + d] * w1->data[h * D_MODEL + d];
                mid[idx * (D_MODEL * 4) + h] = gelu(sum);
            }
            for (int d = 0; d < D_MODEL; d++) {
                double sum = 0.0;
                for (int h = 0; h < D_MODEL * 4; h++)
                    sum += mid[idx * (D_MODEL * 4) + h] * w2->data[d * (D_MODEL * 4) + h];
                out->data[idx * D_MODEL + d] = sum;
            }
        }
    free(mid);
}

void multihead_attention(Tensor *out, const Tensor *in, const Tensor *wq, const Tensor *wk, const Tensor *wv, const Tensor *wo) {
    int hd = D_MODEL/N_HEAD;
    double *q = malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double *k = malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double *v = malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double));
    double *s = malloc(BATCH_SIZE * N_HEAD * SEQ_LENGTH * SEQ_LENGTH * sizeof(double));

    for (int b = 0; b < BATCH_SIZE; b++)
        for (int t = 0; t < SEQ_LENGTH; t++)
            for (int h = 0; h < N_HEAD; h++)
                for (int d = 0; d < hd; d++) {
                    int idx = (b*SEQ_LENGTH*D_MODEL) + (t*D_MODEL) + (h*hd) + d;
                    double sq=0, sk=0, sv=0;
                    for (int i = 0; i < D_MODEL; i++) {
                        int in_i = (b*SEQ_LENGTH*D_MODEL) + (t*D_MODEL) + i;
                        int w_i = (h*hd + d)*D_MODEL + i;
                        sq += in->data[in_i] * wq->data[w_i];
                        sk += in->data[in_i] * wk->data[w_i];
                        sv += in->data[in_i] * wv->data[w_i];
                    }
                    q[idx]=sq; k[idx]=sk; v[idx]=sv;
                }

    for (int b = 0; b < BATCH_SIZE; b++)
        for (int h = 0; h < N_HEAD; h++) {
            double m = pow(2.0, -(8.0*(h+1)/N_HEAD));
            for (int i = 0; i < SEQ_LENGTH; i++) {
                double max=-INFINITY, sum=0;
                for (int j = 0; j <= i; j++) {
                    int idx = (b*N_HEAD*SEQ_LENGTH*SEQ_LENGTH) + (h*SEQ_LENGTH*SEQ_LENGTH) + (i*SEQ_LENGTH) + j;
                    s[idx] = 0;
                    for (int d = 0; d < hd; d++)
                        s[idx] += q[(b*SEQ_LENGTH*D_MODEL)+(i*D_MODEL)+(h*hd)+d] * k[(b*SEQ_LENGTH*D_MODEL)+(j*D_MODEL)+(h*hd)+d];
                    s[idx] = s[idx]/sqrt(hd) - m*(i-j);
                    max = fmax(max, s[idx]);
                }
                for (int j = 0; j <= i; j++) {
                    int idx = (b*N_HEAD*SEQ_LENGTH*SEQ_LENGTH) + (h*SEQ_LENGTH*SEQ_LENGTH) + (i*SEQ_LENGTH) + j;
                    s[idx] = exp(s[idx]-max);
                    sum += s[idx];
                }
                for (int j = 0; j <= i; j++)
                    s[(b*N_HEAD*SEQ_LENGTH*SEQ_LENGTH)+(h*SEQ_LENGTH*SEQ_LENGTH)+(i*SEQ_LENGTH)+j] /= sum;
            }
        }

    for (int b = 0; b < BATCH_SIZE; b++)
        for (int t = 0; t < SEQ_LENGTH; t++) {
            double *ho = calloc(D_MODEL, sizeof(double));
            for (int h = 0; h < N_HEAD; h++)
                for (int d = 0; d < hd; d++) {
                    double sum = 0;
                    for (int j = 0; j <= t; j++)
                        sum += s[(b*N_HEAD*SEQ_LENGTH*SEQ_LENGTH)+(h*SEQ_LENGTH*SEQ_LENGTH)+(t*SEQ_LENGTH)+j] * v[(b*SEQ_LENGTH*D_MODEL)+(j*D_MODEL)+(h*hd)+d];
                    ho[h*hd+d] = sum;
                }
            for (int d = 0; d < D_MODEL; d++) {
                double sum = 0;
                for (int i = 0; i < D_MODEL; i++) sum += ho[i] * wo->data[d*D_MODEL+i];
                out->data[(b*SEQ_LENGTH*D_MODEL)+(t*D_MODEL)+d] = sum;
            }
            free(ho);
        }
    free(q); free(k); free(v); free(s);
}

Dataset load_csv(const char* filename) {
    Dataset ds = {NULL, 0, INPUT_FEATURES, calloc(INPUT_FEATURES, sizeof(double)), calloc(INPUT_FEATURES, sizeof(double))};
    char line[MAX_LINE_LENGTH];
    double* tmp = malloc(1000 * INPUT_FEATURES * sizeof(double));
    FILE* f = fopen(filename, "r");
    if (!f || !fgets(line, MAX_LINE_LENGTH, f)) { printf("File error\n"); exit(1); }
    
    for (int i = 0; i < INPUT_FEATURES; i++) ds.mins[i]=INFINITY, ds.maxs[i]=-INFINITY;
    
    while (fgets(line, MAX_LINE_LENGTH, f)) {
        if (ds.rows >= 1000) tmp = realloc(tmp, (ds.rows*2) * INPUT_FEATURES * sizeof(double));
        char* tok = strtok(line, ",");
        for (int i = 0; i < INPUT_FEATURES && tok; i++, tok = strtok(NULL, ",")) {
            tmp[ds.rows * INPUT_FEATURES + i] = atof(tok);
            ds.mins[i] = fmin(ds.mins[i], tmp[ds.rows * INPUT_FEATURES + i]);
            ds.maxs[i] = fmax(ds.maxs[i], tmp[ds.rows * INPUT_FEATURES + i]);
        }
        ds.rows++;
    }
    
    for (int i = 0; i < ds.rows * INPUT_FEATURES; i++)
        tmp[i] = normalize(tmp[i], ds.mins[i % INPUT_FEATURES], ds.maxs[i % INPUT_FEATURES]);
    
    ds.data = tmp;
    fclose(f);
    return ds;
}

void embed_sequence(Tensor* out, const double* in, const Tensor* ws, const Tensor* wc) {
    for (int b = 0; b < BATCH_SIZE; b++)
        for (int s = 0; s < SEQ_LENGTH; s++)
            for (int d = 0; d < D_MODEL; d++) {
                int idx = (b * SEQ_LENGTH * D_MODEL) + (s * D_MODEL) + d;
                int in_idx = (b * SEQ_LENGTH + s) * INPUT_FEATURES;
                out->data[idx] = 0;
                for (int f = 0; f < SEQUENCE_FEATURES; f++)
                    out->data[idx] += in[in_idx + f + CONDITION_FEATURES] * ws->data[f * D_MODEL + d];
                for (int f = 0; f < CONDITION_FEATURES; f++)
                    out->data[idx] += in[in_idx + f] * wc->data[f * D_MODEL + d];
            }
}

double forward_pass(const double* batch_data, Tensor* out, Tensor* hidden, Tensor* temp,
                   const Tensor* ws, const Tensor* wc, const Tensor* wq, const Tensor* wk, 
                   const Tensor* wv, const Tensor* wo, const Tensor* wf1, const Tensor* wf2, 
                   const Tensor* wout) {
    embed_sequence(hidden, batch_data, ws, wc);

    for (int l = 0; l < N_LAYERS; l++) {
        rmsnorm(temp, hidden);
        multihead_attention(temp, temp, &wq[l], &wk[l], &wv[l], &wo[l]);
        for (int i = 0; i < hidden->size; i++) 
            hidden->data[i] += temp->data[i];
        rmsnorm(temp, hidden);
        feedforward(temp, &wf1[l], &wf2[l], temp);
        for (int i = 0; i < hidden->size; i++) 
            hidden->data[i] += temp->data[i];
    }

    for (int b = 0; b < BATCH_SIZE; b++)
        for (int s = 0; s < SEQ_LENGTH; s++)
            for (int f = 0; f < SEQUENCE_FEATURES; f++) {
                double sum = 0;
                for (int d = 0; d < D_MODEL; d++)
                    sum += hidden->data[(b * SEQ_LENGTH * D_MODEL) + (s * D_MODEL) + d] * 
                          wout->data[d * SEQUENCE_FEATURES + f];
                out->data[(b * SEQ_LENGTH * SEQUENCE_FEATURES) + (s * SEQUENCE_FEATURES) + f] = sum;
            }

    double loss = 0;
    for (int b = 0; b < BATCH_SIZE; b++)
        for (int s = 0; s < SEQ_LENGTH; s++)
            for (int f = 0; f < SEQUENCE_FEATURES; f++) {
                double pred = out->data[(b * SEQ_LENGTH * SEQUENCE_FEATURES) + 
                                      (s * SEQUENCE_FEATURES) + f];
                double target = batch_data[(b * (SEQ_LENGTH + 1) * INPUT_FEATURES) + 
                                         ((s + 1) * INPUT_FEATURES) + f + CONDITION_FEATURES];
                double diff = pred - target;
                loss += diff * diff;
            }
    return loss / (BATCH_SIZE * SEQ_LENGTH * SEQUENCE_FEATURES);
}

void update_weights(Tensor* w, double base_loss, double lr, double* batch_data, Tensor* out, 
                   Tensor* hidden, Tensor* temp, Tensor* ws, Tensor* wc, Tensor* wq, 
                   Tensor* wk, Tensor* wv, Tensor* wo, Tensor* wf1, Tensor* wf2, 
                   Tensor* wout) {
    for (int i = 0; i < w->size; i++) {
        w->data[i] += EPSILON;
        double new_loss = forward_pass(batch_data, out, hidden, temp, ws, wc, wq, wk, wv, wo, wf1, wf2, wout);
        w->data[i] -= EPSILON;
        if (!isnan(new_loss)) {
            double grad = (new_loss - base_loss) / EPSILON;
            grad = fmax(-1.0, fmin(1.0, grad));
            w->data[i] -= lr * grad;
        }
    }
}

void train_finite_diff(Dataset* ds, Tensor* out, Tensor* hidden, Tensor* temp,
                      Tensor* ws, Tensor* wc, Tensor* wq, Tensor* wk, Tensor* wv, 
                      Tensor* wo, Tensor* wf1, Tensor* wf2, Tensor* wout) {
    const double lr = LEARNING_RATE;
    double* batch_data = malloc(BATCH_SIZE * (SEQ_LENGTH + 1) * INPUT_FEATURES * sizeof(double));
    
    for (int step = 0; step < TRAINING_STEPS; step++) {
        // Fill batch_data with sequences - independent random starts per batch
        for (int b = 0; b < BATCH_SIZE; b++) {
            int seq_start = rand() % (ds->rows - SEQ_LENGTH - 1); // Random start for each batch
            for (int s = 0; s < SEQ_LENGTH + 1; s++) {
                for (int f = 0; f < INPUT_FEATURES; f++) {
                    batch_data[(b * (SEQ_LENGTH + 1) * INPUT_FEATURES) + 
                              (s * INPUT_FEATURES) + f] = 
                        ds->data[(seq_start + s) * INPUT_FEATURES + f];
                }
            }
        }

        double base_loss = forward_pass(batch_data, out, hidden, temp, ws, wc, wq, wk, wv, wo, wf1, wf2, wout);
        
        if (isnan(base_loss)) {
            printf("NaN detected at step %d, skipping update\n", step);
            continue;
        }

        printf("Step %d, Loss: %f\n", step, base_loss);

        // Update weights with fixed learning rate
        for (int l = 0; l < N_LAYERS; l++) {
            update_weights(&wq[l], base_loss, lr, batch_data, out, hidden, temp, ws, wc, wq, wk, wv, wo, wf1, wf2, wout);
            update_weights(&wk[l], base_loss, lr, batch_data, out, hidden, temp, ws, wc, wq, wk, wv, wo, wf1, wf2, wout);
            update_weights(&wv[l], base_loss, lr, batch_data, out, hidden, temp, ws, wc, wq, wk, wv, wo, wf1, wf2, wout);
            update_weights(&wo[l], base_loss, lr, batch_data, out, hidden, temp, ws, wc, wq, wk, wv, wo, wf1, wf2, wout);
            update_weights(&wf1[l], base_loss, lr, batch_data, out, hidden, temp, ws, wc, wq, wk, wv, wo, wf1, wf2, wout);
            update_weights(&wf2[l], base_loss, lr, batch_data, out, hidden, temp, ws, wc, wq, wk, wv, wo, wf1, wf2, wout);
        }
        update_weights(ws, base_loss, lr, batch_data, out, hidden, temp, ws, wc, wq, wk, wv, wo, wf1, wf2, wout);
        update_weights(wc, base_loss, lr, batch_data, out, hidden, temp, ws, wc, wq, wk, wv, wo, wf1, wf2, wout);
        update_weights(wout, base_loss, lr, batch_data, out, hidden, temp, ws, wc, wq, wk, wv, wo, wf1, wf2, wout);

        // Print predictions every 10 steps
        if (step % 10 == 0) {
            printf("\nPredictions at step %d:\n", step);
            for (int s = 0; s < 5; s++) {
                printf("Step %d: ", s);
                for (int f = 0; f < SEQUENCE_FEATURES; f++) {
                    double pred = denormalize(out->data[s * SEQUENCE_FEATURES + f], 
                                           ds->mins[f + CONDITION_FEATURES], 
                                           ds->maxs[f + CONDITION_FEATURES]);
                    double actual = denormalize(batch_data[(s + 1) * INPUT_FEATURES + f + CONDITION_FEATURES],
                                             ds->mins[f + CONDITION_FEATURES], 
                                             ds->maxs[f + CONDITION_FEATURES]);
                    printf("F%d(P:%.2f,A:%.2f) ", f, pred, actual);
                }
                printf("\n");
            }
            printf("\n");
        }
    }
    
    free(batch_data);
}

int main() {
    srand(time(NULL));
    Dataset ds = load_csv("2024-12-29_6-25-1_control_data.csv");
    double ws = sqrt(2.0 / D_MODEL);
    
    Tensor W_seq = {malloc(SEQUENCE_FEATURES * D_MODEL * sizeof(double)), NULL, 0, SEQUENCE_FEATURES * D_MODEL};
    Tensor W_cond = {malloc(CONDITION_FEATURES * D_MODEL * sizeof(double)), NULL, 0, CONDITION_FEATURES * D_MODEL};
    W_seq.size = SEQUENCE_FEATURES * D_MODEL;
    W_cond.size = CONDITION_FEATURES * D_MODEL;
    
    Tensor W_q[N_LAYERS], W_k[N_LAYERS], W_v[N_LAYERS], W_o[N_LAYERS], W_ff1[N_LAYERS], W_ff2[N_LAYERS];
    for (int l = 0; l < N_LAYERS; l++) {
        W_q[l] = (Tensor){malloc(D_MODEL * D_MODEL * sizeof(double)), NULL, 0, D_MODEL * D_MODEL};
        W_k[l] = (Tensor){malloc(D_MODEL * D_MODEL * sizeof(double)), NULL, 0, D_MODEL * D_MODEL};
        W_v[l] = (Tensor){malloc(D_MODEL * D_MODEL * sizeof(double)), NULL, 0, D_MODEL * D_MODEL};
        W_o[l] = (Tensor){malloc(D_MODEL * D_MODEL * sizeof(double)), NULL, 0, D_MODEL * D_MODEL};
        W_ff1[l] = (Tensor){malloc(D_MODEL * (D_MODEL * 4) * sizeof(double)), NULL, 0, D_MODEL * (D_MODEL * 4)};
        W_ff2[l] = (Tensor){malloc((D_MODEL * 4) * D_MODEL * sizeof(double)), NULL, 0, (D_MODEL * 4) * D_MODEL};
        
        for (int i = 0; i < D_MODEL * D_MODEL; i++)
            W_q[l].data[i] = W_k[l].data[i] = W_v[l].data[i] = W_o[l].data[i] = randn() * ws;
        for (int i = 0; i < W_ff1[l].size; i++) W_ff1[l].data[i] = randn() * ws;
        for (int i = 0; i < W_ff2[l].size; i++) W_ff2[l].data[i] = randn() * ws;
    }
    
    for (int i = 0; i < W_seq.size; i++) W_seq.data[i] = randn() * ws;
    for (int i = 0; i < W_cond.size; i++) W_cond.data[i] = randn() * ws;
    
    Tensor hidden = {malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double)), NULL, 0, BATCH_SIZE * SEQ_LENGTH * D_MODEL};
    Tensor temp = {malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double)), NULL, 0, BATCH_SIZE * SEQ_LENGTH * D_MODEL};
    Tensor output = {malloc(BATCH_SIZE * SEQ_LENGTH * SEQUENCE_FEATURES * sizeof(double)), NULL, 0, BATCH_SIZE * SEQ_LENGTH * SEQUENCE_FEATURES};
    Tensor W_out = {malloc(D_MODEL * SEQUENCE_FEATURES * sizeof(double)), NULL, 0, D_MODEL * SEQUENCE_FEATURES};
    
    for (int i = 0; i < W_out.size; i++) W_out.data[i] = randn() * ws;

    train_finite_diff(&ds, &output, &hidden, &temp, &W_seq, &W_cond, W_q, W_k, W_v, W_o, W_ff1, W_ff2, &W_out);

    // Cleanup
    free(ds.data); free(ds.mins); free(ds.maxs);
    free(hidden.data); free(temp.data); free(output.data); free(W_out.data);
    for (int l = 0; l < N_LAYERS; l++) {
        free(W_q[l].data); free(W_k[l].data); free(W_v[l].data); free(W_o[l].data);
        free(W_ff1[l].data); free(W_ff2[l].data);
    }
    free(W_seq.data); free(W_cond.data);
    return 0;
}