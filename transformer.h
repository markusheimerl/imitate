#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "grad.h"

Tensor* tensor_rms_norm(Tensor* x, float eps);

Tensor* tensor_masked_multihead_attention(Tensor* Q, Tensor* K, Tensor* V, Tensor* mask, int num_heads) {
    if (!Q || !K || !V || !mask || Q->ndims != 3 || K->ndims != 3 || V->ndims != 3 || mask->ndims != 4) return NULL;
    int batch_size = Q->dims[0], seq_len_q = Q->dims[1], seq_len_k = K->dims[1], d_model = Q->dims[2];
    if (d_model % num_heads != 0 || K->dims[2] != d_model || V->dims[2] != d_model || 
        seq_len_k != V->dims[1] || batch_size != K->dims[0] || batch_size != V->dims[0] ||
        mask->dims[0] != batch_size || mask->dims[1] != num_heads || 
        mask->dims[2] != seq_len_q || mask->dims[3] != seq_len_k) return NULL;
    
    int d_head = d_model/num_heads; tape_len = 0;
    int reshape_dims[] = {batch_size, -1, num_heads, d_head}, perm[] = {0, 2, 1, 3};
    
    reshape_dims[1] = seq_len_q;
    Tensor* Q_perm = tensor_permute(tensor_reshape(Q, 4, reshape_dims), perm, 4);
    reshape_dims[1] = seq_len_k;
    Tensor* K_perm = tensor_permute(tensor_reshape(K, 4, reshape_dims), perm, 4);
    Tensor* V_perm = tensor_permute(tensor_reshape(V, 4, reshape_dims), perm, 4);
    if (!Q_perm || !K_perm || !V_perm) return NULL;

    Tensor* K_transpose = tensor_permute(K_perm, (int[]){0,1,3,2}, 4);
    if (!K_transpose) return NULL;
    
    Tensor* scores = tensor_matmul(Q_perm, K_transpose);
    if (!scores) return NULL;
    
    float scale = 1.0f/sqrt(d_head);
    Tensor* scaled_scores = tensor_hadamard(scores, tensor_new(4, (int[]){1,1,1,1}, (float[]){scale}, 0));
    if (!scaled_scores) return NULL;

    // Apply mask
    Tensor* masked_scores = tensor_hadamard(scaled_scores, mask);
    if (!masked_scores) return NULL;

    Tensor* attention = tensor_matmul(tensor_softmax(masked_scores), V_perm);
    if (!attention) return NULL;

    return tensor_reshape(tensor_permute(attention, (int[]){0,2,1,3}, 4), 3, (int[]){batch_size,seq_len_q,d_model});
}

Tensor* tensor_feedforward(Tensor* input, Tensor* weights) {
    if (!input || !weights) return NULL;
    Tensor* matmul_result = tensor_matmul(input, weights);
    Tensor* output = tensor_gelu(matmul_result);
    return output;
}


#endif // TRANSFORMER_H