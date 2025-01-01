#include "transformer.h"

Tensor* attention(Tensor* x, Tensor* mask, int num_heads) {
    return tensor_masked_multihead_attention(x, x, x, mask, num_heads);
}

Tensor* feed_forward(Tensor* x, Tensor* W1, Tensor* W2) {
    return tensor_feedforward(tensor_feedforward(x, W1), W2);
}

Tensor* transformer_block(Tensor* x, Tensor* mask, int num_heads, float eps, Tensor* W_ff1, Tensor* W_ff2) {
    x = tensor_add(x, attention(tensor_rms_norm(x, eps), mask, num_heads));
    x = tensor_add(x, feed_forward(tensor_rms_norm(x, eps), W_ff1, W_ff2));
    return x;
}

int main() {
    const int batch_size = 1;
    const int seq_len = 3;
    const int d_model = 8;
    const int num_heads = 2;
    const int ff_dim = 32;
    const float eps = 1e-5f;

    // Initialize input tensor
    const float input_data[] = {0.1f, 0.2f, -0.1f, 0.3f, 0.2f, 0.1f, -0.2f, 0.1f, 
                               0.2f, 0.1f, 0.3f, -0.1f, 0.1f, 0.2f, 0.3f, -0.2f, 
                               -0.1f, 0.3f, 0.2f, 0.1f, -0.2f, 0.3f, 0.1f, 0.2f};
    Tensor* x = tensor_new(3, (const int[3]){batch_size, seq_len, d_model}, input_data, 1);

    // Initialize mask
    Tensor* mask = tensor_new(4, (const int[4]){batch_size, num_heads, seq_len, seq_len}, 
                             (const float[18]){1,0,0, 1,1,0, 1,1,1, 1,0,0, 1,1,0, 1,1,1}, 0);

    // Initialize feedforward weights
    Tensor* W_ff1 = tensor_randn(2, (const int[2]){d_model, ff_dim}, 1);
    Tensor* W_ff2 = tensor_randn(2, (const int[2]){ff_dim, d_model}, 1);

    printf("Processing sequence through decoder layer...\n");
    Tensor* output = transformer_layer(x, mask, num_heads, eps, W_ff1, W_ff2);

    printf("\nFinal output (first sequence position):\n");
    for(int j = 0; j < d_model; j++) printf("%.3f ", output->data[j]);
    printf("\n");

    // Backpropagation
    for(int i = 0; i < output->size; i++) output->grad[i] = 1.0f;
    backward();

    printf("\nInput gradients (first sequence position):\n");
    for(int j = 0; j < d_model; j++) printf("%.3f ", x->grad[j]);
    printf("\n");

    clean_registry();
    return 0;
}