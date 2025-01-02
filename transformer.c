#include "transformer.h"

int main() {
    Dataset dataset = load_csv("2024-12-29_6-25-1_control_data.csv");
    
    int dims_e[] = {SEQUENCE_FEATURES, D_MODEL};
    int dims_cond[] = {CONDITION_FEATURES, D_MODEL};
    Tensor* W_e = tensor_randn_permanent(2, dims_e, 1);
    Tensor* W_cond = tensor_randn_permanent(2, dims_cond, 1);
    
    int dims_out[] = {D_MODEL, SEQUENCE_FEATURES};
    Tensor* W_out = tensor_randn_permanent(2, dims_out, 1);
    
    float w_scale = sqrtf(2.0f / D_MODEL);
    for (int i = 0; i < W_e->size; i++) W_e->data[i] *= w_scale;
    for (int i = 0; i < W_cond->size; i++) W_cond->data[i] *= w_scale;
    for (int i = 0; i < W_out->size; i++) W_out->data[i] *= w_scale;
    
    int scale_dims[] = {BATCH_SIZE, N_HEAD, SEQ_LENGTH, SEQ_LENGTH};
    Tensor* scale_tensor = tensor_zeros_permanent(4, scale_dims, 0);
    float scale_val = 1.0f / sqrtf(D_MODEL / N_HEAD);
    for (int i = 0; i < scale_tensor->size; i++) scale_tensor->data[i] = scale_val;
    Tensor* alibi_mask = create_alibi_mask(BATCH_SIZE, N_HEAD, SEQ_LENGTH);
    
    int attn_dims[] = {D_MODEL, D_MODEL};
    int ff_dims1[] = {D_MODEL, D_MODEL * 4};
    int ff_dims2[] = {D_MODEL * 4, D_MODEL};
    w_scale = sqrtf(2.0f / D_MODEL);
    
    Tensor* W_q[N_LAYERS], *W_k[N_LAYERS], *W_v[N_LAYERS], *W_o[N_LAYERS], *W_ff1[N_LAYERS], *W_ff2[N_LAYERS];
    
    for (int l = 0; l < N_LAYERS; l++) {
        W_q[l] = tensor_randn_permanent(2, attn_dims, 1);
        W_k[l] = tensor_randn_permanent(2, attn_dims, 1);
        W_v[l] = tensor_randn_permanent(2, attn_dims, 1);
        W_o[l] = tensor_randn_permanent(2, attn_dims, 1);
        W_ff1[l] = tensor_randn_permanent(2, ff_dims1, 1);
        W_ff2[l] = tensor_randn_permanent(2, ff_dims2, 1);
        
        for (int i = 0; i < D_MODEL * D_MODEL; i++) {
            W_q[l]->data[i] *= w_scale;
            W_k[l]->data[i] *= w_scale;
            W_v[l]->data[i] *= w_scale;
            W_o[l]->data[i] *= w_scale;
        }
        for (int i = 0; i < W_ff1[l]->size; i++) W_ff1[l]->data[i] *= w_scale;
        for (int i = 0; i < W_ff2[l]->size; i++) W_ff2[l]->data[i] *= w_scale;
    }
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        printf("\n=== Epoch %d/%d ===\n", epoch + 1, EPOCHS);
        train_epoch(&dataset, W_e, W_cond, W_q, W_k, W_v, W_o, W_ff1, W_ff2,
                   W_out, scale_tensor, alibi_mask, LEARNING_RATE);
    }

    save_weights(W_e, W_cond, W_out, W_q, W_k, W_v, W_o, W_ff1, W_ff2);
    
    free(dataset.data);
    clean_registry();
    return 0;
}