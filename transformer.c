#include "grad.h"

#define MAX_LINE_LENGTH 1024
#define INPUT_FEATURES 14
#define LEARNING_RATE 0.0001f
#define BATCH_SIZE 4
#define SEQ_LENGTH 64
#define D_MODEL 128
#define N_HEAD 4
#define N_LAYERS 2
#define EPOCHS 100

typedef struct {
    float* data;
    int rows;
    int cols;
} Dataset;

Tensor* feed_forward(Tensor* W_in, Tensor* W_out, Tensor* x) {
    return tensor_matmul(tensor_gelu(tensor_matmul(x, W_in)), W_out);
}

Tensor* attention(Tensor* W_q, Tensor* W_k, Tensor* W_v, Tensor* W_o, 
                 Tensor* x, Tensor* scale_tensor, Tensor* causal_mask,
                 int batch_size, int seq_len, int n_head, int d_model) {
    int d_head = d_model / n_head;
    int qkv_dims[] = {batch_size, seq_len, n_head, d_head};
    int perm[] = {0, 2, 1, 3};
    
    Tensor* Q = tensor_permute(tensor_reshape(tensor_matmul(x, W_q), 4, qkv_dims), perm, 4);
    Tensor* K = tensor_permute(tensor_reshape(tensor_matmul(x, W_k), 4, qkv_dims), perm, 4);
    Tensor* V = tensor_permute(tensor_reshape(tensor_matmul(x, W_v), 4, qkv_dims), perm, 4);
    
    int perm_k[] = {0, 1, 3, 2};
    Tensor* scores = tensor_hadamard(tensor_hadamard(tensor_matmul(Q, tensor_permute(K, perm_k, 4)), 
                                   scale_tensor), causal_mask);
    
    Tensor* attn_output = tensor_matmul(tensor_softmax(scores), V);
    int out_dims[] = {batch_size, seq_len, d_model};
    return tensor_matmul(tensor_reshape(tensor_permute(attn_output, perm, 4), 3, out_dims), W_o);
}

Tensor* transformer_block(Tensor* W_q, Tensor* W_k, Tensor* W_v, Tensor* W_o,
                        Tensor* W_ff1, Tensor* W_ff2, Tensor* x,
                        Tensor* scale_tensor, Tensor* causal_mask,
                        int batch_size, int seq_len, int n_head, int d_model) {
    Tensor* attn_out = attention(W_q, W_k, W_v, W_o, tensor_rms_norm(x, 1e-5f), 
                                scale_tensor, causal_mask,
                                batch_size, seq_len, n_head, d_model);
    Tensor* res1 = tensor_add(x, attn_out);
    return tensor_add(res1, feed_forward(W_ff1, W_ff2, tensor_rms_norm(res1, 1e-5f)));
}

Tensor* decoder_transformer(Tensor* input, Tensor** W_q, Tensor** W_k, 
                          Tensor** W_v, Tensor** W_o, Tensor** W_ff1, 
                          Tensor** W_ff2, Tensor* scale_tensor, 
                          Tensor* causal_mask, int batch_size, int seq_len, 
                          int n_head, int d_model, int n_layers) {
    Tensor* current = input;
    for (int l = 0; l < n_layers; l++) {
        current = transformer_block(W_q[l], W_k[l], W_v[l], W_o[l],
            W_ff1[l], W_ff2[l], current, scale_tensor, causal_mask,
            batch_size, seq_len, n_head, d_model);
    }
    return current;
}

Dataset load_csv(const char* filename) {
    printf("Loading CSV file: %s\n", filename);
    Dataset dataset = {NULL, 0, INPUT_FEATURES};
    char line[MAX_LINE_LENGTH];
    float* temp = malloc(1000 * INPUT_FEATURES * sizeof(float));
    int capacity = 1000;
    
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Failed to open file\n");
        exit(1);
    }
    
    if (fgets(line, MAX_LINE_LENGTH, file) == NULL) {
        printf("Failed to read header line\n");
        fclose(file);
        exit(1);
    }
    
    // Find min and max for normalization
    float* mins = calloc(INPUT_FEATURES, sizeof(float));
    float* maxs = calloc(INPUT_FEATURES, sizeof(float));
    for (int i = 0; i < INPUT_FEATURES; i++) {
        mins[i] = INFINITY;
        maxs[i] = -INFINITY;
    }
    
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        if (dataset.rows >= capacity) {
            capacity *= 2;
            temp = realloc(temp, capacity * INPUT_FEATURES * sizeof(float));
        }
        
        char* token = strtok(line, ",");
        for (int i = 0; i < INPUT_FEATURES; i++) {
            float val = atof(token);
            temp[dataset.rows * INPUT_FEATURES + i] = val;
            mins[i] = fminf(mins[i], val);
            maxs[i] = fmaxf(maxs[i], val);
            token = strtok(NULL, ",");
        }
        dataset.rows++;
    }
    
    // Normalize data to [-1, 1]
    for (int i = 0; i < dataset.rows; i++) {
        for (int j = 0; j < INPUT_FEATURES; j++) {
            float range = maxs[j] - mins[j];
            if (range > 0) {
                temp[i * INPUT_FEATURES + j] = 2.0f * (temp[i * INPUT_FEATURES + j] - mins[j]) / range - 1.0f;
            }
        }
    }
    
    free(mins);
    free(maxs);
    fclose(file);
    dataset.data = temp;
    printf("Loaded %d rows\n", dataset.rows);
    return dataset;
}

Tensor* embed_features(Tensor* W_e, float* data, int batch_size, int seq_len) {
    int input_dims[] = {batch_size, seq_len, INPUT_FEATURES};
    Tensor* input = tensor_new(3, input_dims, NULL, 1);
    memcpy(input->data, data, batch_size * seq_len * INPUT_FEATURES * sizeof(float));
    return tensor_matmul(input, W_e);
}

void train_epoch(Dataset* dataset, Tensor* W_e, Tensor** W_q, Tensor** W_k, 
                Tensor** W_v, Tensor** W_o, Tensor** W_ff1, Tensor** W_ff2, 
                Tensor* scale_tensor, Tensor* causal_mask, float learning_rate) {
    int n_batches = (dataset->rows - SEQ_LENGTH - 1) / BATCH_SIZE;
    float total_loss = 0.0f;
    
    int batch_dims[] = {BATCH_SIZE, SEQ_LENGTH, INPUT_FEATURES};
    Tensor* x_batch = tensor_zeros_permanent(3, batch_dims, 0);
    Tensor* y_batch = tensor_zeros_permanent(3, batch_dims, 0);

    for (int batch = 0; batch < n_batches; batch++) {
        clean_registry();
        
        for (int b = 0; b < BATCH_SIZE; b++) {
            int start_idx = (batch * BATCH_SIZE + b) % (dataset->rows - SEQ_LENGTH - 1);
            for (int s = 0; s < SEQ_LENGTH; s++) {
                for (int f = 0; f < INPUT_FEATURES; f++) {
                    int idx = (b * SEQ_LENGTH + s) * INPUT_FEATURES + f;
                    int data_idx = (start_idx + s) * INPUT_FEATURES + f;
                    x_batch->data[idx] = dataset->data[data_idx];
                    y_batch->data[idx] = dataset->data[data_idx + INPUT_FEATURES];
                }
            }
        }
        
        Tensor* output = decoder_transformer(embed_features(W_e, x_batch->data, BATCH_SIZE, SEQ_LENGTH),
                                          W_q, W_k, W_v, W_o, W_ff1, W_ff2, scale_tensor, causal_mask,
                                          BATCH_SIZE, SEQ_LENGTH, N_HEAD, D_MODEL, N_LAYERS);
        
        int perm[] = {1, 0};
        Tensor* pred = tensor_matmul(output, tensor_permute(W_e, perm, 2));
        
        float batch_loss = 0.0f;
        for (int i = 0; i < BATCH_SIZE * SEQ_LENGTH * INPUT_FEATURES; i++) {
            float diff = pred->data[i] - y_batch->data[i];
            diff = fmaxf(fminf(diff, 100.0f), -100.0f); // Clip gradients
            batch_loss += diff * diff;
            pred->grad[i] = 2.0f * diff;
        }
        batch_loss /= (BATCH_SIZE * SEQ_LENGTH * INPUT_FEATURES);
        
        if (isfinite(batch_loss)) {
            total_loss += batch_loss;
            
            backward();
            
            // Gradient clipping
            float max_grad = 1.0f;
            
            for (int i = 0; i < W_e->size; i++) {
                W_e->grad[i] = fmaxf(fminf(W_e->grad[i], max_grad), -max_grad);
                W_e->data[i] -= learning_rate * W_e->grad[i];
                W_e->grad[i] = 0.0f;
            }
            
            for (int l = 0; l < N_LAYERS; l++) {
                for (int i = 0; i < W_q[l]->size; i++) {
                    W_q[l]->grad[i] = fmaxf(fminf(W_q[l]->grad[i], max_grad), -max_grad);
                    W_k[l]->grad[i] = fmaxf(fminf(W_k[l]->grad[i], max_grad), -max_grad);
                    W_v[l]->grad[i] = fmaxf(fminf(W_v[l]->grad[i], max_grad), -max_grad);
                    W_o[l]->grad[i] = fmaxf(fminf(W_o[l]->grad[i], max_grad), -max_grad);
                    
                    W_q[l]->data[i] -= learning_rate * W_q[l]->grad[i];
                    W_k[l]->data[i] -= learning_rate * W_k[l]->grad[i];
                    W_v[l]->data[i] -= learning_rate * W_v[l]->grad[i];
                    W_o[l]->data[i] -= learning_rate * W_o[l]->grad[i];
                    
                    W_q[l]->grad[i] = W_k[l]->grad[i] = W_v[l]->grad[i] = W_o[l]->grad[i] = 0.0f;
                }
                
                for (int i = 0; i < W_ff1[l]->size; i++) {
                    W_ff1[l]->grad[i] = fmaxf(fminf(W_ff1[l]->grad[i], max_grad), -max_grad);
                    W_ff1[l]->data[i] -= learning_rate * W_ff1[l]->grad[i];
                    W_ff1[l]->grad[i] = 0.0f;
                }
                
                for (int i = 0; i < W_ff2[l]->size; i++) {
                    W_ff2[l]->grad[i] = fmaxf(fminf(W_ff2[l]->grad[i], max_grad), -max_grad);
                    W_ff2[l]->data[i] -= learning_rate * W_ff2[l]->grad[i];
                    W_ff2[l]->grad[i] = 0.0f;
                }
            }
        }
        
        if (batch % 10 == 0) {
            printf("Batch %d/%d, Loss: %f\n", batch + 1, n_batches, batch_loss);
        }
    }
    
    printf("Average loss: %f\n", total_loss / n_batches);
}

int main() {
    Dataset dataset = load_csv("2024-12-29_6-25-1_control_data.csv");
    
    int dims_e[] = {INPUT_FEATURES, D_MODEL};
    Tensor* W_e = tensor_randn_permanent(2, dims_e, 1);
    float w_scale = sqrtf(2.0f / INPUT_FEATURES);
    for (int i = 0; i < W_e->size; i++) W_e->data[i] *= w_scale;
    
    int scale_dims[] = {BATCH_SIZE, N_HEAD, SEQ_LENGTH, SEQ_LENGTH};
    Tensor* scale_tensor = tensor_zeros_permanent(4, scale_dims, 0);
    Tensor* causal_mask = tensor_zeros_permanent(4, scale_dims, 0);
    float scale_val = 1.0f / sqrtf(D_MODEL / N_HEAD);
    
    for (int i = 0; i < scale_tensor->size; i++) scale_tensor->data[i] = scale_val;
    
    for (int b = 0; b < BATCH_SIZE; b++)
        for (int h = 0; h < N_HEAD; h++)
            for (int i = 0; i < SEQ_LENGTH; i++)
                for (int j = 0; j < SEQ_LENGTH; j++)
                    causal_mask->data[((b*N_HEAD + h)*SEQ_LENGTH + i)*SEQ_LENGTH + j] = j <= i ? 1.0f : -1e9f;
    
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
        train_epoch(&dataset, W_e, W_q, W_k, W_v, W_o, W_ff1, W_ff2,
                   scale_tensor, causal_mask, LEARNING_RATE);
    }
    
    free(dataset.data);
    clean_registry();
    return 0;
}