#include "transformer.h"

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 3 || !strstr(argv[1], ".csv")) {
        printf("Usage: %s <training_data.csv> [weights.bin]\n", argv[0]);
        return 1;
    }

    char *csv_file = argv[1];
    char *weights_file = argc > 2 ? (strstr(argv[2], ".bin") ? argv[2] : NULL) : NULL;
    if (argc > 2 && !weights_file) {
        printf("Error: Invalid file '%s'\n", argv[2]);
        return 1;
    }

    srand(time(NULL));
    Dataset ds = load_csv(csv_file);
    const double ws = sqrt(2.0 / D_MODEL);
    
    Tensor W_seq = {malloc(SEQUENCE_FEATURES * D_MODEL * sizeof(double)), calloc(SEQUENCE_FEATURES * D_MODEL, sizeof(double)), calloc(SEQUENCE_FEATURES * D_MODEL, sizeof(double)), SEQUENCE_FEATURES * D_MODEL};
    Tensor W_cond = {malloc(CONDITION_FEATURES * D_MODEL * sizeof(double)), calloc(CONDITION_FEATURES * D_MODEL, sizeof(double)), calloc(CONDITION_FEATURES * D_MODEL, sizeof(double)), CONDITION_FEATURES * D_MODEL};
    Tensor W_out = {malloc(D_MODEL * SEQUENCE_FEATURES * sizeof(double)), calloc(D_MODEL * SEQUENCE_FEATURES, sizeof(double)), calloc(D_MODEL * SEQUENCE_FEATURES, sizeof(double)), D_MODEL * SEQUENCE_FEATURES};
    
    Tensor W_q[N_LAYERS], W_k[N_LAYERS], W_v[N_LAYERS], W_o[N_LAYERS], W_ff1[N_LAYERS], W_ff2[N_LAYERS];
    for (int l = 0; l < N_LAYERS; l++) {
        const int attn_size = D_MODEL * D_MODEL;
        const int ff_size1 = D_MODEL * (D_MODEL * 4);
        const int ff_size2 = (D_MODEL * 4) * D_MODEL;
        
        W_q[l] = (Tensor){malloc(attn_size * sizeof(double)), calloc(attn_size, sizeof(double)), calloc(attn_size, sizeof(double)), attn_size};
        W_k[l] = (Tensor){malloc(attn_size * sizeof(double)), calloc(attn_size, sizeof(double)), calloc(attn_size, sizeof(double)), attn_size};
        W_v[l] = (Tensor){malloc(attn_size * sizeof(double)), calloc(attn_size, sizeof(double)), calloc(attn_size, sizeof(double)), attn_size};
        W_o[l] = (Tensor){malloc(attn_size * sizeof(double)), calloc(attn_size, sizeof(double)), calloc(attn_size, sizeof(double)), attn_size};
        W_ff1[l] = (Tensor){malloc(ff_size1 * sizeof(double)), calloc(ff_size1, sizeof(double)), calloc(ff_size1, sizeof(double)), ff_size1};
        W_ff2[l] = (Tensor){malloc(ff_size2 * sizeof(double)), calloc(ff_size2, sizeof(double)), calloc(ff_size2, sizeof(double)), ff_size2};
    }

    if (weights_file && load_weights(weights_file, &W_seq, &W_cond, W_q, W_k, W_v, W_o, W_ff1, W_ff2, &W_out)) {
        printf("Successfully loaded weights\n");
    } else {
        if (weights_file) printf("Failed to load weights, initializing randomly\n");
        for (int i = 0; i < W_seq.size; i++) W_seq.data[i] = randn() * ws;
        for (int i = 0; i < W_cond.size; i++) W_cond.data[i] = randn() * ws;
        for (int i = 0; i < W_out.size; i++) W_out.data[i] = randn() * ws;
        
        for (int l = 0; l < N_LAYERS; l++) {
            for (int i = 0; i < W_q[l].size; i++) {
                W_q[l].data[i] = randn() * ws;
                W_k[l].data[i] = randn() * ws;
                W_v[l].data[i] = randn() * ws;
                W_o[l].data[i] = randn() * ws;
            }
            for (int i = 0; i < W_ff1[l].size; i++) W_ff1[l].data[i] = randn() * ws;
            for (int i = 0; i < W_ff2[l].size; i++) W_ff2[l].data[i] = randn() * ws;
        }
    }

    Tensor hidden = {malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double)), NULL, NULL, BATCH_SIZE * SEQ_LENGTH * D_MODEL};
    Tensor temp = {malloc(BATCH_SIZE * SEQ_LENGTH * D_MODEL * sizeof(double)), NULL, NULL, BATCH_SIZE * SEQ_LENGTH * D_MODEL};
    Tensor output = {malloc(BATCH_SIZE * SEQ_LENGTH * SEQUENCE_FEATURES * sizeof(double)), NULL, NULL, BATCH_SIZE * SEQ_LENGTH * SEQUENCE_FEATURES};
    
    train_finite_diff(&ds, &output, &hidden, &temp, &W_seq, &W_cond, W_q, W_k, W_v, W_o, W_ff1, W_ff2, &W_out);
    
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    char filename[100];
    sprintf(filename, "%d-%d-%d_%d-%d-%d_weights.bin", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    
    save_weights(filename, &W_seq, &W_cond, W_q, W_k, W_v, W_o, W_ff1, W_ff2, &W_out);
    printf("Saved weights to: %s\n", filename);

    free(ds.data); free(ds.mins); free(ds.maxs);
    free(hidden.data); free(temp.data); free(output.data);
    free(W_seq.data); free(W_seq.m); free(W_seq.v);
    free(W_cond.data); free(W_cond.m); free(W_cond.v);
    free(W_out.data); free(W_out.m); free(W_out.v);
    
    for (int l = 0; l < N_LAYERS; l++) {
        free(W_q[l].data); free(W_q[l].m); free(W_q[l].v);
        free(W_k[l].data); free(W_k[l].m); free(W_k[l].v);
        free(W_v[l].data); free(W_v[l].m); free(W_v[l].v);
        free(W_o[l].data); free(W_o[l].m); free(W_o[l].v);
        free(W_ff1[l].data); free(W_ff1[l].m); free(W_ff1[l].v);
        free(W_ff2[l].data); free(W_ff2[l].m); free(W_ff2[l].v);
    }
    
    return 0;
}