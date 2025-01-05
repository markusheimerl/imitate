#include "transformer.h"

int main(int argc, char *argv[]) {
    srand(time(NULL));
    Dataset ds = load_csv("2024-12-29_6-25-1_control_data.csv");
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

    if (argc > 1 && load_weights(argv[1], &W_seq, &W_cond, W_q, W_k, W_v, W_o, W_ff1, W_ff2, &W_out)) {
        printf("Successfully loaded weights from %s\n", argv[1]);
    } else {
        printf("Initializing random weights\n");
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
    save_weights("weights.bin", &W_seq, &W_cond, W_q, W_k, W_v, W_o, W_ff1, W_ff2, &W_out);

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