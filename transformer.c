#include "grad/grad.h"
#include <math.h>

// Hyperparameters
#define BATCH_SIZE 2
#define SEQ_LENGTH 4
#define EMBED_DIM 8
#define NUM_HEADS 2
#define HEAD_DIM (EMBED_DIM / NUM_HEADS)
#define FF_DIM (4 * EMBED_DIM)

void print_tensor(const char* name, Tensor* t) {
    printf("\n=== %s ===\n", name);
    printf("Shape: ");
    for (int i = 0; i < t->ndims; i++) {
        printf("%d", t->dims[i]);
        if (i < t->ndims - 1) printf("x");
    }
    printf("\nFirst few values: ");
    int print_count = MIN(8, t->size);
    for (int i = 0; i < print_count; i++) {
        printf("%.4f ", t->data[i]);
    }
    printf("\n");
}

Tensor* create_random_tensor(int ndims, int* dims) {
    int size = 1;
    for (int i = 0; i < ndims; i++) size *= dims[i];
    
    float* data = malloc(size * sizeof(float));
    float scale = 0.02f;
    for (int i = 0; i < size; i++) {
        data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
    }
    
    Tensor* t = tensor_new(ndims, dims, data, 1);
    free(data);
    return t;
}

Tensor* create_ones_tensor(int ndims, int* dims) {
    int size = 1;
    for (int i = 0; i < ndims; i++) size *= dims[i];
    
    float* data = malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        data[i] = 1.0f;
    }
    
    Tensor* t = tensor_new(ndims, dims, data, 1);
    free(data);
    return t;
}

Tensor* rms_norm(Tensor* x, Tensor* weight) {
    // Square all elements
    Tensor* squared = tensor_matmul(x, x);  // Element-wise multiplication
    
    // Compute mean
    int mean_dims[] = {1};
    Tensor* mean = tensor_reshape(squared, 1, mean_dims);
    
    // Add epsilon for numerical stability
    float eps_val = 1e-6f;
    float* eps_data = malloc(sizeof(float));
    *eps_data = eps_val;
    int eps_dims[] = {1};
    Tensor* eps = tensor_new(1, eps_dims, eps_data, 0);
    Tensor* stable_mean = tensor_add(mean, eps);
    
    // Compute RMS
    Tensor* rms = tensor_sigmoid(stable_mean);  // Using sigmoid as sqrt approximation
    
    // Normalize
    Tensor* normalized = tensor_matmul(x, rms);
    
    // Scale with learned parameters
    Tensor* output = tensor_matmul(normalized, weight);
    
    // Cleanup
    tensor_free(squared);
    tensor_free(mean);
    tensor_free(eps);
    tensor_free(stable_mean);
    tensor_free(rms);
    tensor_free(normalized);
    free(eps_data);
    
    return output;
}

Tensor* self_attention(Tensor* x, Tensor* w_q, Tensor* w_k, Tensor* w_v, Tensor* w_o) {
    printf("\n--- Self Attention Forward Pass ---\n");
    
    // Linear projections
    Tensor* q = tensor_matmul(x, w_q);
    Tensor* k = tensor_matmul(x, w_k);
    Tensor* v = tensor_matmul(x, w_v);
    
    print_tensor("Query", q);
    print_tensor("Key", k);
    print_tensor("Value", v);
    
    // Scaled dot-product attention
    float scale = 1.0f / sqrtf(EMBED_DIM);
    float* scale_data = malloc(sizeof(float));
    *scale_data = scale;
    int scale_dims[] = {1};
    Tensor* scale_tensor = tensor_new(1, scale_dims, scale_data, 0);
    
    Tensor* scores = tensor_matmul(q, k);
    Tensor* scaled_scores = tensor_matmul(scores, scale_tensor);
    
    // Create causal mask
    float* mask_data = malloc(SEQ_LENGTH * SEQ_LENGTH * sizeof(float));
    for (int i = 0; i < SEQ_LENGTH; i++) {
        for (int j = 0; j < SEQ_LENGTH; j++) {
            mask_data[i * SEQ_LENGTH + j] = j <= i ? 1.0f : 0.0f;
        }
    }
    int mask_dims[] = {SEQ_LENGTH, SEQ_LENGTH};
    Tensor* mask = tensor_new(2, mask_dims, mask_data, 0);
    
    // Apply mask
    Tensor* masked_scores = tensor_matmul(scaled_scores, mask);
    
    // Softmax
    Tensor* attention_probs = tensor_sigmoid(masked_scores);  // Approximate softmax with sigmoid
    
    // Apply attention to values
    Tensor* attention = tensor_matmul(attention_probs, v);
    Tensor* output = tensor_matmul(attention, w_o);
    
    print_tensor("Attention Output", output);
    
    // Cleanup
    tensor_free(q);
    tensor_free(k);
    tensor_free(v);
    tensor_free(scale_tensor);
    tensor_free(scores);
    tensor_free(scaled_scores);
    tensor_free(mask);
    tensor_free(masked_scores);
    tensor_free(attention_probs);
    tensor_free(attention);
    free(scale_data);
    free(mask_data);
    
    return output;
}

Tensor* ffn(Tensor* x, Tensor* w1, Tensor* w2) {
    printf("\n--- FFN Forward Pass ---\n");
    
    Tensor* hidden = tensor_matmul(x, w1);
    print_tensor("FFN Hidden (pre-activation)", hidden);
    
    Tensor* activated = tensor_relu(hidden);
    print_tensor("FFN Hidden (post-activation)", activated);
    
    Tensor* output = tensor_matmul(activated, w2);
    print_tensor("FFN Output", output);
    
    tensor_free(hidden);
    tensor_free(activated);
    
    return output;
}

Tensor* decoder_block(Tensor* x, 
                     Tensor* attn_weights[4],
                     Tensor* ffn_weights[2],
                     Tensor* norm_weights[2])
{
    printf("\n=== Decoder Block Forward Pass ===\n");
    print_tensor("Input", x);
    
    // Attention
    Tensor* norm1 = rms_norm(x, norm_weights[0]);
    print_tensor("Normalized (pre-attention)", norm1);
    
    Tensor* attn = self_attention(norm1, 
                                attn_weights[0], attn_weights[1], 
                                attn_weights[2], attn_weights[3]);
    Tensor* res1 = tensor_add(x, attn);
    print_tensor("Post-attention with residual", res1);
    
    // Feed-forward
    Tensor* norm2 = rms_norm(res1, norm_weights[1]);
    print_tensor("Normalized (pre-FFN)", norm2);
    
    Tensor* ff = ffn(norm2, ffn_weights[0], ffn_weights[1]);
    Tensor* output = tensor_add(res1, ff);
    print_tensor("Final output", output);
    
    tensor_free(norm1);
    tensor_free(attn);
    tensor_free(res1);
    tensor_free(norm2);
    tensor_free(ff);
    
    return output;
}

void transformer_test() {
    printf("\n=== Starting Transformer Test ===\n");
    printf("Configuration:\n");
    printf("Batch Size: %d\n", BATCH_SIZE);
    printf("Sequence Length: %d\n", SEQ_LENGTH);
    printf("Embedding Dimension: %d\n", EMBED_DIM);
    printf("Number of Heads: %d\n", NUM_HEADS);
    printf("FFN Dimension: %d\n", FF_DIM);
    
    // Initialize input
    int input_dims[] = {BATCH_SIZE, SEQ_LENGTH, EMBED_DIM};
    float* input_data = malloc(BATCH_SIZE * SEQ_LENGTH * EMBED_DIM * sizeof(float));
    for (int i = 0; i < BATCH_SIZE * SEQ_LENGTH * EMBED_DIM; i++) {
        input_data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 0.02f;
    }
    Tensor* input = tensor_new(3, input_dims, input_data, 1);
    
    // Initialize weights
    int qkv_dims[] = {EMBED_DIM, EMBED_DIM};
    int ff1_dims[] = {EMBED_DIM, FF_DIM};
    int ff2_dims[] = {FF_DIM, EMBED_DIM};
    int norm_dims[] = {EMBED_DIM};
    
    Tensor* attn_weights[4] = {
        create_random_tensor(2, qkv_dims),  // w_q
        create_random_tensor(2, qkv_dims),  // w_k
        create_random_tensor(2, qkv_dims),  // w_v
        create_random_tensor(2, qkv_dims)   // w_o
    };
    
    Tensor* ffn_weights[2] = {
        create_random_tensor(2, ff1_dims),  // w1
        create_random_tensor(2, ff2_dims)   // w2
    };
    
    Tensor* norm_weights[2] = {
        create_ones_tensor(1, norm_dims),  // norm1
        create_ones_tensor(1, norm_dims)   // norm2
    };
    
    // Forward pass
    Tensor* output = decoder_block(input, attn_weights, ffn_weights, norm_weights);
    
    // Cleanup
    printf("\n=== Cleaning up ===\n");
    tensor_free(output);
    tensor_free(input);
    for (int i = 0; i < 4; i++) tensor_free(attn_weights[i]);
    for (int i = 0; i < 2; i++) tensor_free(ffn_weights[i]);
    for (int i = 0; i < 2; i++) tensor_free(norm_weights[i]);
    free(input_data);
    
    printf("\n=== Test Complete ===\n");
}

int main() {
    srand(42);
    transformer_test();
    return 0;
}