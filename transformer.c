#include "grad.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH 1024
#define INPUT_FEATURES 14
#define BATCH_SIZE 4     // Further reduced
#define SEQ_LENGTH 2     // Further reduced
#define D_MODEL 32       // Further reduced
#define N_HEAD 2         // Reduced
#define N_LAYERS 1       // Single layer for testing
#define EPOCHS 100
#define LEARNING_RATE 0.001f

#define DEBUG_PRINT(fmt, ...) \
    do { printf("[DEBUG] %s:%d: " fmt "\n", __func__, __LINE__, ##__VA_ARGS__); } while (0)

typedef struct {
    float* data;
    int rows;
    int cols;
} Dataset;

void assert_float_eq(float a, float b, float eps, const char* msg) {
    if (fabsf(a - b) > eps) {
        printf("ASSERTION FAILED: %s\nExpected: %f, Got: %f\n", msg, b, a);
        exit(1);
    }
}

Tensor* feed_forward(Tensor* W_in, Tensor* W_out, Tensor* x) {
    DEBUG_PRINT("Input tensor dims: [%d, %d, %d]", x->dims[0], x->dims[1], x->dims[2]);
    Tensor* hidden = tensor_matmul(x, W_in);
    DEBUG_PRINT("After first matmul: [%d, %d, %d]", hidden->dims[0], hidden->dims[1], hidden->dims[2]);
    Tensor* activated = tensor_gelu(hidden);
    Tensor* output = tensor_matmul(activated, W_out);
    DEBUG_PRINT("Output tensor dims: [%d, %d, %d]", output->dims[0], output->dims[1], output->dims[2]);
    return output;
}

Tensor* attention(Tensor* W_q, Tensor* W_k, Tensor* W_v, Tensor* W_o, 
                 Tensor* x, Tensor* scale_tensor, Tensor* causal_mask,
                 int batch_size, int seq_len, int n_head, int d_model) {
    DEBUG_PRINT("Starting attention computation");
    
    // Validate input tensors
    if (!W_q || !W_k || !W_v || !W_o || !x || !scale_tensor || !causal_mask) {
        printf("ERROR: Null tensor passed to attention\n");
        exit(1);
    }
    
    DEBUG_PRINT("Input tensor dims: [%d, %d, %d]", x->dims[0], x->dims[1], x->dims[2]);
    DEBUG_PRINT("Weight matrix dims: Q[%d, %d], K[%d, %d], V[%d, %d], O[%d, %d]",
                W_q->dims[0], W_q->dims[1], W_k->dims[0], W_k->dims[1],
                W_v->dims[0], W_v->dims[1], W_o->dims[0], W_o->dims[1]);
    
    int d_head = d_model / n_head;
    DEBUG_PRINT("d_head = %d", d_head);
    
    int qkv_dims[] = {batch_size, seq_len, n_head, d_head};
    int perm[] = {0, 2, 1, 3};
    
    // Linear projections
    DEBUG_PRINT("Computing Q projection");
    Tensor* Q = tensor_matmul(x, W_q);
    if (!Q) { printf("ERROR: Q projection failed\n"); exit(1); }
    
    DEBUG_PRINT("Computing K projection");
    Tensor* K = tensor_matmul(x, W_k);
    if (!K) { printf("ERROR: K projection failed\n"); exit(1); }
    
    DEBUG_PRINT("Computing V projection");
    Tensor* V = tensor_matmul(x, W_v);
    if (!V) { printf("ERROR: V projection failed\n"); exit(1); }
    
    // Reshape
    DEBUG_PRINT("Reshaping Q");
    Q = tensor_reshape(Q, 4, qkv_dims);
    if (!Q) { printf("ERROR: Q reshape failed\n"); exit(1); }
    
    DEBUG_PRINT("Reshaping K");
    K = tensor_reshape(K, 4, qkv_dims);
    if (!K) { printf("ERROR: K reshape failed\n"); exit(1); }
    
    DEBUG_PRINT("Reshaping V");
    V = tensor_reshape(V, 4, qkv_dims);
    if (!V) { printf("ERROR: V reshape failed\n"); exit(1); }
    
    // Permute
    DEBUG_PRINT("Permuting Q");
    Q = tensor_permute(Q, perm, 4);
    if (!Q) { printf("ERROR: Q permute failed\n"); exit(1); }
    
    DEBUG_PRINT("Permuting K");
    K = tensor_permute(K, perm, 4);
    if (!K) { printf("ERROR: K permute failed\n"); exit(1); }
    
    DEBUG_PRINT("Permuting V");
    V = tensor_permute(V, perm, 4);
    if (!V) { printf("ERROR: V permute failed\n"); exit(1); }
    
    // Compute attention scores
    DEBUG_PRINT("Computing attention scores");
    int perm_k[] = {0, 1, 3, 2};
    Tensor* K_t = tensor_permute(K, perm_k, 4);
    if (!K_t) { printf("ERROR: K transpose failed\n"); exit(1); }
    
    Tensor* scores = tensor_matmul(Q, K_t);
    if (!scores) { printf("ERROR: QK matmul failed\n"); exit(1); }
    
    scores = tensor_hadamard(scores, scale_tensor);
    if (!scores) { printf("ERROR: Scale multiplication failed\n"); exit(1); }
    
    scores = tensor_hadamard(scores, causal_mask);
    if (!scores) { printf("ERROR: Mask application failed\n"); exit(1); }
    
    // Apply attention
    DEBUG_PRINT("Applying softmax and computing attention output");
    Tensor* attn_weights = tensor_softmax(scores);
    if (!attn_weights) { printf("ERROR: Softmax failed\n"); exit(1); }
    
    Tensor* attn_output = tensor_matmul(attn_weights, V);
    if (!attn_output) { printf("ERROR: Attention output computation failed\n"); exit(1); }
    
    // Reshape back
    DEBUG_PRINT("Reshaping output");
    attn_output = tensor_permute(attn_output, perm, 4);
    if (!attn_output) { printf("ERROR: Output permute failed\n"); exit(1); }
    
    int out_dims[] = {batch_size, seq_len, d_model};
    attn_output = tensor_reshape(attn_output, 3, out_dims);
    if (!attn_output) { printf("ERROR: Output reshape failed\n"); exit(1); }
    
    // Final projection
    DEBUG_PRINT("Final projection");
    Tensor* output = tensor_matmul(attn_output, W_o);
    if (!output) { printf("ERROR: Output projection failed\n"); exit(1); }
    
    DEBUG_PRINT("Attention computation complete");
    return output;
}

Tensor* transformer_block(Tensor* W_q, Tensor* W_k, Tensor* W_v, Tensor* W_o,
                        Tensor* W_ff1, Tensor* W_ff2, Tensor* x,
                        Tensor* scale_tensor, Tensor* causal_mask,
                        int batch_size, int seq_len, int n_head, int d_model) {
    DEBUG_PRINT("Starting transformer block");
    
    // Validate input tensors
    if (!W_q || !W_k || !W_v || !W_o || !W_ff1 || !W_ff2 || !x || !scale_tensor || !causal_mask) {
        printf("ERROR: Null tensor passed to transformer_block\n");
        exit(1);
    }
    
    DEBUG_PRINT("Input tensor dims: [%d, %d, %d]", x->dims[0], x->dims[1], x->dims[2]);
    DEBUG_PRINT("Weight matrix dims: FF1[%d, %d], FF2[%d, %d]",
                W_ff1->dims[0], W_ff1->dims[1], W_ff2->dims[0], W_ff2->dims[1]);
    
    // Self-attention sublayer
    DEBUG_PRINT("Computing layer normalization 1");
    Tensor* normed1 = tensor_rms_norm(x, 1e-5f);
    if (!normed1) { printf("ERROR: First layer norm failed\n"); exit(1); }
    
    DEBUG_PRINT("Computing self-attention");
    Tensor* attn_out = attention(W_q, W_k, W_v, W_o, normed1, 
                                scale_tensor, causal_mask,
                                batch_size, seq_len, n_head, d_model);
    if (!attn_out) { printf("ERROR: Attention computation failed\n"); exit(1); }
    
    DEBUG_PRINT("Computing first residual connection");
    Tensor* res1 = tensor_add(x, attn_out);
    if (!res1) { printf("ERROR: First residual connection failed\n"); exit(1); }
    
    // Feed-forward sublayer
    DEBUG_PRINT("Computing layer normalization 2");
    Tensor* normed2 = tensor_rms_norm(res1, 1e-5f);
    if (!normed2) { printf("ERROR: Second layer norm failed\n"); exit(1); }
    
    DEBUG_PRINT("Computing feed-forward network");
    Tensor* ff_out = feed_forward(W_ff1, W_ff2, normed2);
    if (!ff_out) { printf("ERROR: Feed-forward computation failed\n"); exit(1); }
    
    DEBUG_PRINT("Computing second residual connection");
    Tensor* output = tensor_add(res1, ff_out);
    if (!output) { printf("ERROR: Second residual connection failed\n"); exit(1); }
    
    DEBUG_PRINT("Transformer block complete");
    return output;
}

Tensor* decoder_transformer(Tensor* input, Tensor** W_q, Tensor** W_k, 
                          Tensor** W_v, Tensor** W_o, Tensor** W_ff1, 
                          Tensor** W_ff2, Tensor* scale_tensor, 
                          Tensor* causal_mask, int batch_size, int seq_len, 
                          int n_head, int d_model, int n_layers) {
    DEBUG_PRINT("Starting decoder transformer with %d layers", n_layers);
    
    // Validate input tensors
    if (!input) {
        printf("ERROR: Input tensor is null\n");
        exit(1);
    }
    if (!W_q || !W_k || !W_v || !W_o || !W_ff1 || !W_ff2) {
        printf("ERROR: Weight arrays are null\n");
        exit(1);
    }
    if (!scale_tensor || !causal_mask) {
        printf("ERROR: Scale tensor or causal mask is null\n");
        exit(1);
    }

    DEBUG_PRINT("Input tensor dims: [%d, %d, %d]", 
                input->dims[0], input->dims[1], input->dims[2]);
    
    // Validate layer weights
    for (int l = 0; l < n_layers; l++) {
        if (!W_q[l] || !W_k[l] || !W_v[l] || !W_o[l] || !W_ff1[l] || !W_ff2[l]) {
            printf("ERROR: Layer %d weights are null\n", l);
            exit(1);
        }
        DEBUG_PRINT("Layer %d weight dims:", l);
        DEBUG_PRINT("  W_q: [%d, %d]", W_q[l]->dims[0], W_q[l]->dims[1]);
        DEBUG_PRINT("  W_k: [%d, %d]", W_k[l]->dims[0], W_k[l]->dims[1]);
        DEBUG_PRINT("  W_v: [%d, %d]", W_v[l]->dims[0], W_v[l]->dims[1]);
        DEBUG_PRINT("  W_o: [%d, %d]", W_o[l]->dims[0], W_o[l]->dims[1]);
        DEBUG_PRINT("  W_ff1: [%d, %d]", W_ff1[l]->dims[0], W_ff1[l]->dims[1]);
        DEBUG_PRINT("  W_ff2: [%d, %d]", W_ff2[l]->dims[0], W_ff2[l]->dims[1]);
    }
    
    Tensor* current = input;
    
    for (int l = 0; l < n_layers; l++) {
        DEBUG_PRINT("Processing layer %d", l + 1);
        DEBUG_PRINT("Current tensor dims: [%d, %d, %d]", 
                    current->dims[0], current->dims[1], current->dims[2]);
        
        Tensor* next = transformer_block(W_q[l], W_k[l], W_v[l], W_o[l],
            W_ff1[l], W_ff2[l], current, scale_tensor, causal_mask,
            batch_size, seq_len, n_head, d_model);
            
        if (!next) {
            printf("ERROR: transformer_block returned null at layer %d\n", l);
            exit(1);
        }
        
        current = next;
    }
    
    DEBUG_PRINT("Decoder transformer complete");
    DEBUG_PRINT("Output tensor dims: [%d, %d, %d]", 
                current->dims[0], current->dims[1], current->dims[2]);
    return current;
}

Dataset load_csv(const char* filename) {
    DEBUG_PRINT("Loading CSV file: %s", filename);
    Dataset dataset = {NULL, 0, INPUT_FEATURES};
    char line[MAX_LINE_LENGTH];
    float* temp = NULL;
    int capacity = 1000;
    
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("ERROR: Failed to open file: %s\n", filename);
        exit(1);
    }

    if (fgets(line, MAX_LINE_LENGTH, file) == NULL) {
        printf("ERROR: Failed to read header\n");
        exit(1);
    }
    
    temp = malloc(capacity * INPUT_FEATURES * sizeof(float));
    if (!temp) {
        printf("ERROR: Failed to allocate initial memory for dataset\n");
        exit(1);
    }
    
    DEBUG_PRINT("Reading data rows...");
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        if (dataset.rows >= capacity) {
            capacity *= 2;
            DEBUG_PRINT("Expanding capacity to %d rows", capacity);
            float* new_temp = realloc(temp, capacity * INPUT_FEATURES * sizeof(float));
            if (!new_temp) {
                printf("ERROR: Failed to reallocate memory for dataset\n");
                free(temp);
                exit(1);
            }
            temp = new_temp;
        }
        
        char* token = strtok(line, ",");
        for (int i = 0; i < INPUT_FEATURES; i++) {
            if (token == NULL) {
                printf("ERROR: Invalid CSV format at row %d\n", dataset.rows + 1);
                free(temp);
                exit(1);
            }
            temp[dataset.rows * INPUT_FEATURES + i] = atof(token);
            token = strtok(NULL, ",");
        }
        dataset.rows++;
    }
    
    fclose(file);
    dataset.data = temp;
    DEBUG_PRINT("Successfully loaded %d rows", dataset.rows);
    return dataset;
}

Tensor* create_embedding_weights() {
    DEBUG_PRINT("Creating embedding weights [%d, %d]", INPUT_FEATURES, D_MODEL);
    int dims[] = {INPUT_FEATURES, D_MODEL};
    Tensor* W_e = tensor_randn(2, dims, 1);
    
    if (!W_e) {
        printf("ERROR: Failed to create embedding weights\n");
        exit(1);
    }
    
    // Validate dimensions
    if (W_e->dims[0] != INPUT_FEATURES || W_e->dims[1] != D_MODEL) {
        printf("ERROR: Embedding weights have wrong dimensions: [%d, %d], expected [%d, %d]\n",
               W_e->dims[0], W_e->dims[1], INPUT_FEATURES, D_MODEL);
        exit(1);
    }
    
    float w_scale = sqrtf(2.0f / INPUT_FEATURES);
    DEBUG_PRINT("Initializing embedding weights with scale %f", w_scale);
    
    for (int i = 0; i < W_e->size; i++) {
        W_e->data[i] *= w_scale;
    }
    
    DEBUG_PRINT("Created embedding weights with dims [%d, %d]", W_e->dims[0], W_e->dims[1]);
    return W_e;
}

Tensor* embed_features(Tensor* W_e, float* data, int batch_size, int seq_len) {
    DEBUG_PRINT("Embedding features for batch_size=%d, seq_len=%d", batch_size, seq_len);
    
    // Validate embedding matrix
    if (!W_e) {
        printf("ERROR: Embedding matrix is null\n");
        exit(1);
    }
    if (W_e->dims[0] != INPUT_FEATURES || W_e->dims[1] != D_MODEL) {
        printf("ERROR: Embedding matrix has wrong dimensions: [%d, %d], expected [%d, %d]\n",
               W_e->dims[0], W_e->dims[1], INPUT_FEATURES, D_MODEL);
        exit(1);
    }
    
    // Create input tensor
    int input_dims[] = {batch_size, seq_len, INPUT_FEATURES};
    DEBUG_PRINT("Creating input tensor with dims: [%d, %d, %d]", 
                input_dims[0], input_dims[1], input_dims[2]);
    
    Tensor* input = tensor_new(3, input_dims, NULL, 1);
    if (!input) {
        printf("ERROR: Failed to create input tensor\n");
        exit(1);
    }
    
    // Copy data
    DEBUG_PRINT("Copying data to input tensor");
    int total_elements = batch_size * seq_len * INPUT_FEATURES;
    for (int i = 0; i < total_elements; i++) {
        input->data[i] = data[i];
    }
    
    // Perform embedding
    DEBUG_PRINT("Performing embedding matrix multiplication");
    DEBUG_PRINT("Input tensor dims: [%d, %d, %d]", 
                input->dims[0], input->dims[1], input->dims[2]);
    DEBUG_PRINT("Embedding matrix dims: [%d, %d]",
                W_e->dims[0], W_e->dims[1]);
    
    // Validate dimensions for matrix multiplication
    if (input->dims[2] != W_e->dims[0]) {
        printf("ERROR: Incompatible dimensions for embedding: input[%d, %d, %d] @ W_e[%d, %d]\n",
               input->dims[0], input->dims[1], input->dims[2], W_e->dims[0], W_e->dims[1]);
        exit(1);
    }
    
    Tensor* embedded = tensor_matmul(input, W_e);
    if (!embedded) {
        printf("ERROR: Embedding matrix multiplication failed\n");
        exit(1);
    }
    
    DEBUG_PRINT("Embedded tensor dims: [%d, %d, %d]", 
                embedded->dims[0], embedded->dims[1], embedded->dims[2]);
    
    // Validate output dimensions
    if (embedded->dims[0] != batch_size || 
        embedded->dims[1] != seq_len || 
        embedded->dims[2] != D_MODEL) {
        printf("ERROR: Wrong embedded dimensions: [%d, %d, %d], expected [%d, %d, %d]\n",
               embedded->dims[0], embedded->dims[1], embedded->dims[2],
               batch_size, seq_len, D_MODEL);
        exit(1);
    }
    
    return embedded;
}

void train_epoch(Dataset* dataset, Tensor* W_e, Tensor** W_q, Tensor** W_k, 
                Tensor** W_v, Tensor** W_o, Tensor** W_ff1, Tensor** W_ff2, 
                Tensor* scale_tensor, Tensor* causal_mask, float learning_rate) {
    
    DEBUG_PRINT("Starting epoch");
    int n_batches = (dataset->rows - SEQ_LENGTH - 1) / BATCH_SIZE;
    DEBUG_PRINT("Will process %d batches", n_batches);
    float total_loss = 0.0f;
    
    // Allocate batch data
    float* x_data = malloc(BATCH_SIZE * SEQ_LENGTH * INPUT_FEATURES * sizeof(float));
    float* y_data = malloc(BATCH_SIZE * SEQ_LENGTH * INPUT_FEATURES * sizeof(float));
    
    if (!x_data || !y_data) {
        printf("ERROR: Failed to allocate batch data memory\n");
        exit(1);
    }

    // Store the original weights
    float* W_e_data = malloc(W_e->size * sizeof(float));
    memcpy(W_e_data, W_e->data, W_e->size * sizeof(float));
    
    float** W_q_data = malloc(N_LAYERS * sizeof(float*));
    float** W_k_data = malloc(N_LAYERS * sizeof(float*));
    float** W_v_data = malloc(N_LAYERS * sizeof(float*));
    float** W_o_data = malloc(N_LAYERS * sizeof(float*));
    float** W_ff1_data = malloc(N_LAYERS * sizeof(float*));
    float** W_ff2_data = malloc(N_LAYERS * sizeof(float*));
    
    for (int l = 0; l < N_LAYERS; l++) {
        W_q_data[l] = malloc(W_q[l]->size * sizeof(float));
        W_k_data[l] = malloc(W_k[l]->size * sizeof(float));
        W_v_data[l] = malloc(W_v[l]->size * sizeof(float));
        W_o_data[l] = malloc(W_o[l]->size * sizeof(float));
        W_ff1_data[l] = malloc(W_ff1[l]->size * sizeof(float));
        W_ff2_data[l] = malloc(W_ff2[l]->size * sizeof(float));
        
        memcpy(W_q_data[l], W_q[l]->data, W_q[l]->size * sizeof(float));
        memcpy(W_k_data[l], W_k[l]->data, W_k[l]->size * sizeof(float));
        memcpy(W_v_data[l], W_v[l]->data, W_v[l]->size * sizeof(float));
        memcpy(W_o_data[l], W_o[l]->data, W_o[l]->size * sizeof(float));
        memcpy(W_ff1_data[l], W_ff1[l]->data, W_ff1[l]->size * sizeof(float));
        memcpy(W_ff2_data[l], W_ff2[l]->data, W_ff2[l]->size * sizeof(float));
    }
    
    // Store attention mask data
    int scale_size = BATCH_SIZE * N_HEAD * SEQ_LENGTH * SEQ_LENGTH;
    float* scale_data = malloc(scale_size * sizeof(float));
    float* mask_data = malloc(scale_size * sizeof(float));
    memcpy(scale_data, scale_tensor->data, scale_size * sizeof(float));
    memcpy(mask_data, causal_mask->data, scale_size * sizeof(float));
    
    for (int batch = 0; batch < n_batches; batch++) {
        DEBUG_PRINT("\nProcessing batch %d/%d", batch + 1, n_batches);
        
        // Clear registry and recreate tensors
        clean_registry();
        
        // Recreate embedding tensor
        int dims_e[] = {INPUT_FEATURES, D_MODEL};
        W_e = tensor_new(2, dims_e, W_e_data, 1);
        
        // Recreate layer tensors
        int dims_attn[] = {D_MODEL, D_MODEL};
        int dims_ff1[] = {D_MODEL, D_MODEL * 4};
        int dims_ff2[] = {D_MODEL * 4, D_MODEL};
        
        for (int l = 0; l < N_LAYERS; l++) {
            W_q[l] = tensor_new(2, dims_attn, W_q_data[l], 1);
            W_k[l] = tensor_new(2, dims_attn, W_k_data[l], 1);
            W_v[l] = tensor_new(2, dims_attn, W_v_data[l], 1);
            W_o[l] = tensor_new(2, dims_attn, W_o_data[l], 1);
            W_ff1[l] = tensor_new(2, dims_ff1, W_ff1_data[l], 1);
            W_ff2[l] = tensor_new(2, dims_ff2, W_ff2_data[l], 1);
        }
        
        // Recreate attention masks
        int scale_dims[] = {BATCH_SIZE, N_HEAD, SEQ_LENGTH, SEQ_LENGTH};
        scale_tensor = tensor_new(4, scale_dims, scale_data, 0);
        causal_mask = tensor_new(4, scale_dims, mask_data, 0);
        
        // Fill batch data
        for (int b = 0; b < BATCH_SIZE; b++) {
            int start_idx = (batch * BATCH_SIZE + b) % (dataset->rows - SEQ_LENGTH - 1);
            for (int s = 0; s < SEQ_LENGTH; s++) {
                for (int f = 0; f < INPUT_FEATURES; f++) {
                    int idx = (b * SEQ_LENGTH + s) * INPUT_FEATURES + f;
                    int data_idx = (start_idx + s) * INPUT_FEATURES + f;
                    x_data[idx] = dataset->data[data_idx];
                    y_data[idx] = dataset->data[data_idx + INPUT_FEATURES];
                }
            }
        }
        
        // Forward pass
        Tensor* x_embedded = embed_features(W_e, x_data, BATCH_SIZE, SEQ_LENGTH);
        if (!x_embedded) {
            printf("ERROR: Failed to create embedded input\n");
            exit(1);
        }
        
        Tensor* output = decoder_transformer(x_embedded, W_q, W_k, W_v, W_o, W_ff1, W_ff2,
                                          scale_tensor, causal_mask, BATCH_SIZE, 
                                          SEQ_LENGTH, N_HEAD, D_MODEL, N_LAYERS);
        if (!output) {
            printf("ERROR: Transformer returned null output\n");
            exit(1);
        }
        
        int perm[] = {1, 0};
        Tensor* W_e_permuted = tensor_permute(W_e, perm, 2);
        if (!W_e_permuted) {
            printf("ERROR: Failed to permute embedding matrix\n");
            exit(1);
        }
        
        Tensor* pred = tensor_matmul(output, W_e_permuted);
        if (!pred) {
            printf("ERROR: Failed to compute predictions\n");
            exit(1);
        }
        
        // Compute loss and gradients
        float batch_loss = 0.0f;
        for (int i = 0; i < BATCH_SIZE * SEQ_LENGTH * INPUT_FEATURES; i++) {
            float diff = pred->data[i] - y_data[i];
            batch_loss += diff * diff;
            pred->grad[i] = 2.0f * diff;
        }
        batch_loss /= (BATCH_SIZE * SEQ_LENGTH * INPUT_FEATURES);
        total_loss += batch_loss;
        
        backward();
        
        // Update stored weights
        for (int i = 0; i < W_e->size; i++) {
            W_e_data[i] -= learning_rate * W_e->grad[i];
        }
        
        for (int l = 0; l < N_LAYERS; l++) {
            for (int i = 0; i < W_q[l]->size; i++) {
                W_q_data[l][i] -= learning_rate * W_q[l]->grad[i];
                W_k_data[l][i] -= learning_rate * W_k[l]->grad[i];
                W_v_data[l][i] -= learning_rate * W_v[l]->grad[i];
                W_o_data[l][i] -= learning_rate * W_o[l]->grad[i];
            }
            for (int i = 0; i < W_ff1[l]->size; i++) {
                W_ff1_data[l][i] -= learning_rate * W_ff1[l]->grad[i];
            }
            for (int i = 0; i < W_ff2[l]->size; i++) {
                W_ff2_data[l][i] -= learning_rate * W_ff2[l]->grad[i];
            }
        }
        
        if (batch % 10 == 0) {
            DEBUG_PRINT("Batch %d/%d, Loss: %f", batch + 1, n_batches, batch_loss);
        }
    }
    
    // Copy final weights back to original tensors
    memcpy(W_e->data, W_e_data, W_e->size * sizeof(float));
    for (int l = 0; l < N_LAYERS; l++) {
        memcpy(W_q[l]->data, W_q_data[l], W_q[l]->size * sizeof(float));
        memcpy(W_k[l]->data, W_k_data[l], W_k[l]->size * sizeof(float));
        memcpy(W_v[l]->data, W_v_data[l], W_v[l]->size * sizeof(float));
        memcpy(W_o[l]->data, W_o_data[l], W_o[l]->size * sizeof(float));
        memcpy(W_ff1[l]->data, W_ff1_data[l], W_ff1[l]->size * sizeof(float));
        memcpy(W_ff2[l]->data, W_ff2_data[l], W_ff2[l]->size * sizeof(float));
    }
    
    // Cleanup
    free(W_e_data);
    free(scale_data);
    free(mask_data);
    for (int l = 0; l < N_LAYERS; l++) {
        free(W_q_data[l]);
        free(W_k_data[l]);
        free(W_v_data[l]);
        free(W_o_data[l]);
        free(W_ff1_data[l]);
        free(W_ff2_data[l]);
    }
    free(W_q_data);
    free(W_k_data);
    free(W_v_data);
    free(W_o_data);
    free(W_ff1_data);
    free(W_ff2_data);
    
    free(x_data);
    free(y_data);
    
    DEBUG_PRINT("Epoch complete, average loss: %f", total_loss / n_batches);
}

int main() {
    DEBUG_PRINT("Starting program");
    
    DEBUG_PRINT("Loading dataset");
    Dataset dataset = load_csv("2024-12-29_6-25-1_control_data.csv");
    DEBUG_PRINT("Dataset loaded with %d rows", dataset.rows);
    
    DEBUG_PRINT("Creating embedding weights");
    Tensor* W_e = create_embedding_weights();
    if (!W_e) {
        printf("ERROR: Failed to create embedding weights\n");
        exit(1);
    }
    DEBUG_PRINT("Created embedding weights with dims [%d, %d]", W_e->dims[0], W_e->dims[1]);
    
    DEBUG_PRINT("Allocating transformer weights");
    Tensor **W_q = malloc(N_LAYERS * sizeof(Tensor*));
    Tensor **W_k = malloc(N_LAYERS * sizeof(Tensor*));
    Tensor **W_v = malloc(N_LAYERS * sizeof(Tensor*));
    Tensor **W_o = malloc(N_LAYERS * sizeof(Tensor*));
    Tensor **W_ff1 = malloc(N_LAYERS * sizeof(Tensor*));
    Tensor **W_ff2 = malloc(N_LAYERS * sizeof(Tensor*));
    
    if (!W_q || !W_k || !W_v || !W_o || !W_ff1 || !W_ff2) {
        printf("ERROR: Failed to allocate weight arrays\n");
        exit(1);
    }
    
    DEBUG_PRINT("Creating attention masks");
    int scale_dims[] = {BATCH_SIZE, N_HEAD, SEQ_LENGTH, SEQ_LENGTH};
    Tensor* scale_tensor = tensor_new(4, scale_dims, NULL, 0);
    Tensor* causal_mask = tensor_new(4, scale_dims, NULL, 0);
    float scale_val = 1.0f / sqrtf(D_MODEL / N_HEAD);
    
    DEBUG_PRINT("Initializing attention masks");
    for (int i = 0; i < scale_tensor->size; i++) {
        scale_tensor->data[i] = scale_val;
    }
    
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int h = 0; h < N_HEAD; h++) {
            for (int i = 0; i < SEQ_LENGTH; i++) {
                for (int j = 0; j < SEQ_LENGTH; j++) {
                    causal_mask->data[((b*N_HEAD + h)*SEQ_LENGTH + i)*SEQ_LENGTH + j] = 
                        j <= i ? 1.0f : -1e9f;
                }
            }
        }
    }
    
    DEBUG_PRINT("Initializing transformer weights");
    int attn_dims[] = {D_MODEL, D_MODEL};
    int ff_dims1[] = {D_MODEL, D_MODEL * 4};
    int ff_dims2[] = {D_MODEL * 4, D_MODEL};
    float w_scale = sqrtf(2.0f / D_MODEL);
    
    for (int l = 0; l < N_LAYERS; l++) {
        DEBUG_PRINT("Initializing layer %d", l + 1);
        W_q[l] = tensor_randn(2, attn_dims, 1);
        W_k[l] = tensor_randn(2, attn_dims, 1);
        W_v[l] = tensor_randn(2, attn_dims, 1);
        W_o[l] = tensor_randn(2, attn_dims, 1);
        W_ff1[l] = tensor_randn(2, ff_dims1, 1);
        W_ff2[l] = tensor_randn(2, ff_dims2, 1);
        
        // Validate tensor creation
        if (!W_q[l] || !W_k[l] || !W_v[l] || !W_o[l] || !W_ff1[l] || !W_ff2[l]) {
            printf("ERROR: Failed to create weight tensors for layer %d\n", l);
            exit(1);
        }
        
        DEBUG_PRINT("Layer %d weight dimensions:", l + 1);
        DEBUG_PRINT("  W_q: [%d, %d]", W_q[l]->dims[0], W_q[l]->dims[1]);
        DEBUG_PRINT("  W_k: [%d, %d]", W_k[l]->dims[0], W_k[l]->dims[1]);
        DEBUG_PRINT("  W_v: [%d, %d]", W_v[l]->dims[0], W_v[l]->dims[1]);
        DEBUG_PRINT("  W_o: [%d, %d]", W_o[l]->dims[0], W_o[l]->dims[1]);
        DEBUG_PRINT("  W_ff1: [%d, %d]", W_ff1[l]->dims[0], W_ff1[l]->dims[1]);
        DEBUG_PRINT("  W_ff2: [%d, %d]", W_ff2[l]->dims[0], W_ff2[l]->dims[1]);
        
        for (int i = 0; i < D_MODEL * D_MODEL; i++) {
            W_q[l]->data[i] *= w_scale;
            W_k[l]->data[i] *= w_scale;
            W_v[l]->data[i] *= w_scale;
            W_o[l]->data[i] *= w_scale;
        }
        for (int i = 0; i < D_MODEL * D_MODEL * 4; i++) {
            W_ff1[l]->data[i] *= w_scale;
        }
        for (int i = 0; i < D_MODEL * 4 * D_MODEL; i++) {
            W_ff2[l]->data[i] *= w_scale;
        }
    }
    
    DEBUG_PRINT("Starting training");
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        DEBUG_PRINT("\n=== Epoch %d/%d ===", epoch + 1, EPOCHS);
        train_epoch(&dataset, W_e, W_q, W_k, W_v, W_o, W_ff1, W_ff2,
                   scale_tensor, causal_mask, LEARNING_RATE);
    }
    
    DEBUG_PRINT("Cleaning up");
    free(W_q);
    free(W_k);
    free(W_v);
    free(W_o);
    free(W_ff1);
    free(W_ff2);
    free(dataset.data);
    clean_registry();
    
    DEBUG_PRINT("Program complete");
    return 0;
}