#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#define OUTPUT_FEATURES 4
#define ROTOR_OFFSET 6
#define CONDITION_FEATURES 3
#define SEQUENCE_FEATURES 10
#define SEQ_LENGTH 64
#define D_MODEL 32
#define N_HEAD 4
#define N_LAYERS 2
#define LEARNING_RATE 0.00001
#define TRAINING_STEPS 10000

typedef struct { double *data; int rows, cols; } Dataset;

double randn() { return sqrt(-2.0 * log((double)rand() / RAND_MAX)) * cos(2.0 * M_PI * (double)rand() / RAND_MAX); }

Dataset load_csv(const char* filename) {
    Dataset ds = {malloc(1000 * (CONDITION_FEATURES + SEQUENCE_FEATURES) * sizeof(double)), 0, (CONDITION_FEATURES + SEQUENCE_FEATURES)};
    char line[1024]; FILE* f = fopen(filename, "r");
    if (!f || !fgets(line, 1024, f)) { printf("File error\n"); exit(1); }
    
    while (fgets(line, 1024, f)) {
        if (ds.rows >= 1000) ds.data = realloc(ds.data, (ds.rows*2) * (CONDITION_FEATURES + SEQUENCE_FEATURES) * sizeof(double));
        char* tok = strtok(line, ",");
        for (int i = 0; i < (CONDITION_FEATURES + SEQUENCE_FEATURES) && tok; i++, tok = strtok(NULL, ",")) ds.data[ds.rows * (CONDITION_FEATURES + SEQUENCE_FEATURES) + i] = atof(tok);
        ds.rows++;
    }
    fclose(f);
    return ds;
}

void save_weights(const double* ws, const double* wc, const double* wq, const double* wk, const double* wv, const double* wo, const double* wf1, const double* wf2, const double* wout) {
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    char filename[100]; 
    sprintf(filename, "%d-%d-%d_%d-%d-%d_weights.bin", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    
    FILE* f = fopen(filename, "wb");
    if (!f) return;
    
    fwrite(ws, sizeof(double), SEQUENCE_FEATURES * D_MODEL, f);
    fwrite(wc, sizeof(double), CONDITION_FEATURES * D_MODEL, f);
    for (int l = 0; l < N_LAYERS; l++) {
        const int layer_offset = l * D_MODEL * D_MODEL;
        const int ff_offset1 = l * D_MODEL * (D_MODEL * 4);
        const int ff_offset2 = l * (D_MODEL * 4) * D_MODEL;
        fwrite(wq + layer_offset, sizeof(double), D_MODEL * D_MODEL, f);
        fwrite(wk + layer_offset, sizeof(double), D_MODEL * D_MODEL, f);
        fwrite(wv + layer_offset, sizeof(double), D_MODEL * D_MODEL, f);
        fwrite(wo + layer_offset, sizeof(double), D_MODEL * D_MODEL, f);
        fwrite(wf1 + ff_offset1, sizeof(double), D_MODEL * (D_MODEL * 4), f);
        fwrite(wf2 + ff_offset2, sizeof(double), (D_MODEL * 4) * D_MODEL, f);
    }
    fwrite(wout, sizeof(double), D_MODEL * OUTPUT_FEATURES, f);
    fclose(f);
    printf("Saved weights to: %s\n", filename);
}

int load_weights(const char* filename, double* ws, double* wc, double* wq, double* wk, double* wv, double* wo, double* wf1, double* wf2, double* wout) {
    FILE* f = fopen(filename, "rb");
    if (!f) return 0;
    
    size_t read = fread(ws, sizeof(double), SEQUENCE_FEATURES * D_MODEL, f);
    read += fread(wc, sizeof(double), CONDITION_FEATURES * D_MODEL, f);
    
    for (int l = 0; l < N_LAYERS; l++) {
        const int layer_offset = l * D_MODEL * D_MODEL;
        const int ff_offset1 = l * D_MODEL * (D_MODEL * 4);
        const int ff_offset2 = l * (D_MODEL * 4) * D_MODEL;
        read += fread(wq + layer_offset, sizeof(double), D_MODEL * D_MODEL, f);
        read += fread(wk + layer_offset, sizeof(double), D_MODEL * D_MODEL, f);
        read += fread(wv + layer_offset, sizeof(double), D_MODEL * D_MODEL, f);
        read += fread(wo + layer_offset, sizeof(double), D_MODEL * D_MODEL, f);
        read += fread(wf1 + ff_offset1, sizeof(double), D_MODEL * (D_MODEL * 4), f);
        read += fread(wf2 + ff_offset2, sizeof(double), (D_MODEL * 4) * D_MODEL, f);
    }
    
    read += fread(wout, sizeof(double), D_MODEL * OUTPUT_FEATURES, f);
    fclose(f);
    
    size_t expected = SEQUENCE_FEATURES * D_MODEL + CONDITION_FEATURES * D_MODEL + D_MODEL * OUTPUT_FEATURES + N_LAYERS * (4 * D_MODEL * D_MODEL + D_MODEL * (D_MODEL * 4) + (D_MODEL * 4) * D_MODEL);
    return read == expected;
}

// Given input X of shape [seq_len, d_model]
// RMSNorm(X)_s,d = X_s,d / sqrt(1/D * sum_i(X_s,i^2) + eps)
// where s=sequence index, d=dimension index
void rmsnorm(double *out, const double *in) {
    #pragma omp parallel for
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* x = in + s * D_MODEL;
        double* y = out + s * D_MODEL;
        double ss = 0.0;
        for (int i = 0; i < D_MODEL; i++) ss += x[i] * x[i];
        double scale = 1.0 / sqrt(ss / D_MODEL + 1e-5);
        for (int i = 0; i < D_MODEL; i++) y[i] = x[i] * scale;
    }
}

// Given input X of shape [seq_len, d_model]
// For each position in sequence:
// 1. Linear: U = X*W1 where X:[1,d_model], W1:[d_model,4*d_model] -> U:[1,4*d_model]
// 2. GELU (elementwise): G(U) = 0.5 * U * (1 + tanh(sqrt(2/pi) * (U + 0.044715 * U^3)))
// 3. Linear: Y = G(U)*W2 where G(U):[1,4*d_model], W2:[4*d_model,d_model] -> Y:[1,d_model]
void feedforward(double *out, const double *w1, const double *w2, const double *in, double *mid) {
    #pragma omp parallel for
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* x = in + s * D_MODEL;
        double* y = out + s * D_MODEL;
        double* u = mid + s * (D_MODEL * 4);
        
        for (int i = 0; i < D_MODEL * 4; i++) {
            u[i] = 0.0;
            for (int j = 0; j < D_MODEL; j++) u[i] += x[j] * w1[i * D_MODEL + j];
            u[i] = 0.5 * u[i] * (1.0 + tanh(sqrt(2.0/M_PI) * u[i] + 0.044715 * u[i] * u[i] * u[i]));
        }
        
        for (int i = 0; i < D_MODEL; i++) {
            y[i] = 0.0;
            for (int j = 0; j < D_MODEL * 4; j++) y[i] += u[j] * w2[i * (D_MODEL * 4) + j];
        }
    }
}

// Given input X of shape [seq_len, d_model]
// 1. QKV projection for each head h:
//    Q_h = X * Wq_h, K_h = X * Wk_h, V_h = X * Wv_h
// 2. Scaled dot-product attention with ALiBi bias per head:
//    score = (Q_h * K_h^T)/sqrt(d_head) - ALiBi_slope_h * distance_matrix
//    A_h = softmax(score) * V_h  where softmax is causal (upper triangle masked)
// 3. Concatenate heads and project:
//    MultiHead(X) = concat(A_1,...,A_h) * Wo
void multihead_attention(double *out, const double *in, const double *wq, const double *wk, const double *wv, const double *wo, double *q, double *k, double *v, double *s) {
    const int hd = D_MODEL / N_HEAD;
    const double scale = 1.0 / sqrt(hd);

    #pragma omp parallel for
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* x = in + s * D_MODEL;
        for (int h = 0; h < N_HEAD; h++)
            for (int d = 0; d < hd; d++) {
                double sq = 0.0, sk = 0.0, sv = 0.0;
                const int w_idx = (h * hd + d) * D_MODEL;
                for (int i = 0; i < D_MODEL; i++) sq += x[i] * wq[w_idx + i], sk += x[i] * wk[w_idx + i], sv += x[i] * wv[w_idx + i];
                const int qkv_idx = s * D_MODEL + h * hd + d;
                q[qkv_idx] = sq, k[qkv_idx] = sk, v[qkv_idx] = sv;
            }
    }

    #pragma omp parallel for
    for (int h = 0; h < N_HEAD; h++) {
        const double slope = pow(2.0, -(8.0 * (h + 1) / N_HEAD));
        for (int i = 0; i < SEQ_LENGTH; i++) {
            double max = -1e9, sum = 0.0;
            for (int j = 0; j <= i; j++) {
                double dot = 0.0;
                for (int d = 0; d < hd; d++) dot += q[i * D_MODEL + h * hd + d] * k[j * D_MODEL + h * hd + d];
                s[(h * SEQ_LENGTH + i) * SEQ_LENGTH + j] = dot * scale - slope * (i - j);
                max = fmax(max, s[(h * SEQ_LENGTH + i) * SEQ_LENGTH + j]);
            }
            for (int j = 0; j <= i; j++) {
                s[(h * SEQ_LENGTH + i) * SEQ_LENGTH + j] = exp(s[(h * SEQ_LENGTH + i) * SEQ_LENGTH + j] - max);
                sum += s[(h * SEQ_LENGTH + i) * SEQ_LENGTH + j];
            }
            for (int j = 0; j <= i; j++) s[(h * SEQ_LENGTH + i) * SEQ_LENGTH + j] /= (sum + 1e-10);
        }
    }

    #pragma omp parallel for
    for (int t = 0; t < SEQ_LENGTH; t++) {
        double tmp[D_MODEL] = {0};
        for (int h = 0; h < N_HEAD; h++)
            for (int d = 0; d < hd; d++) {
                double sum = 0.0;
                for (int j = 0; j <= t; j++) sum += s[(h * SEQ_LENGTH + t) * SEQ_LENGTH + j] * v[j * D_MODEL + h * hd + d];
                tmp[h * hd + d] = sum;
            }
        for (int d = 0; d < D_MODEL; d++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++) sum += tmp[i] * wo[d * D_MODEL + i];
            out[t * D_MODEL + d] = sum;
        }
    }
}

// Forward pass through the transformer:
// 1. Input embedding: sequence_features * Ws + condition_features * Wc
// 2. N transformer layers of:
//    x = x + attention(rmsnorm(x))
//    x = x + ffn(rmsnorm(x))
// 3. Output projection to sequence_features
void forward_pass(const double* seq_data, double* out, double* hidden, double* temp, const double* ws, const double* wc, const double* wq, const double* wk, const double* wv, const double* wo, const double* wf1, const double* wf2, const double* wout, double* q_buf, double* k_buf, double* v_buf, double* s_buf, double* mid_buf) {
    #pragma omp parallel for
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* x = seq_data + s * (CONDITION_FEATURES + SEQUENCE_FEATURES);
        double* y = hidden + s * D_MODEL;
        for (int d = 0; d < D_MODEL; d++) {
            double sum = 0.0;
            for (int f = 0; f < SEQUENCE_FEATURES; f++) sum += x[f + CONDITION_FEATURES] * ws[f * D_MODEL + d];
            for (int f = 0; f < CONDITION_FEATURES; f++) sum += x[f] * wc[f * D_MODEL + d];
            y[d] = sum;
        }
    }

    for (int l = 0; l < N_LAYERS; l++) {
        const int layer_offset = l * D_MODEL * D_MODEL;
        const int ff_offset1 = l * D_MODEL * (D_MODEL * 4);
        const int ff_offset2 = l * (D_MODEL * 4) * D_MODEL;
        
        rmsnorm(temp, hidden);
        multihead_attention(temp, temp, wq + layer_offset, wk + layer_offset, wv + layer_offset, wo + layer_offset, q_buf, k_buf, v_buf, s_buf);
        for (int i = 0; i < SEQ_LENGTH * D_MODEL; i++) hidden[i] += temp[i];
        
        rmsnorm(temp, hidden);
        feedforward(temp, wf1 + ff_offset1, wf2 + ff_offset2, temp, mid_buf);
        for (int i = 0; i < SEQ_LENGTH * D_MODEL; i++) hidden[i] += temp[i];
    }

    #pragma omp parallel for
    for (int s = 0; s < SEQ_LENGTH; s++) {
        const double* h = hidden + s * D_MODEL;
        double* o = out + s * OUTPUT_FEATURES;
        for (int f = 0; f < OUTPUT_FEATURES; f++) {
            double sum = 0.0;
            for (int d = 0; d < D_MODEL; d++) sum += h[d] * wout[f * D_MODEL + d];
            o[f] = sum;
        }
    }
}

#endif // TRANSFORMER_H