#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define SEQ_LEN 32
#define MODEL_DIM 64
#define NUM_MOTORS 4
#define LEARNING_RATE 0.001f
#define NUM_EPOCHS 1000

// Global model parameters
float* W_up;    // [MODEL_DIM x NUM_MOTORS]
float* b_up;    // [MODEL_DIM]
float* W_down;  // [NUM_MOTORS x MODEL_DIM]
float* b_down;  // [NUM_MOTORS]

// Global gradients
float* dW_up;
float* db_up;
float* dW_down;
float* db_down;

// Global buffers
float* hidden;
float* temp;
float** sequences;

void init_model() {
    // Allocate weights and biases
    W_up = calloc(MODEL_DIM * NUM_MOTORS, sizeof(float));
    b_up = calloc(MODEL_DIM, sizeof(float));
    W_down = calloc(NUM_MOTORS * MODEL_DIM, sizeof(float));
    b_down = calloc(NUM_MOTORS, sizeof(float));
    
    // Allocate gradients
    dW_up = calloc(MODEL_DIM * NUM_MOTORS, sizeof(float));
    db_up = calloc(MODEL_DIM, sizeof(float));
    dW_down = calloc(NUM_MOTORS * MODEL_DIM, sizeof(float));
    db_down = calloc(NUM_MOTORS, sizeof(float));
    
    // Allocate buffers
    hidden = calloc(MODEL_DIM, sizeof(float));
    temp = calloc(MODEL_DIM, sizeof(float));
    sequences = malloc(SEQ_LEN * sizeof(float*));
    for(int i = 0; i < SEQ_LEN; i++) {
        sequences[i] = malloc(NUM_MOTORS * sizeof(float));
    }
    
    // Initialize weights with small random values
    for(int i = 0; i < MODEL_DIM * NUM_MOTORS; i++) {
        W_up[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        W_down[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    }
}

void forward(float** input_sequence, float* output) {
    memset(hidden, 0, MODEL_DIM * sizeof(float));
    
    for(int seq = 0; seq < SEQ_LEN; seq++) {
        // Up projection
        for(int i = 0; i < MODEL_DIM; i++) {
            temp[i] = b_up[i];
            for(int j = 0; j < NUM_MOTORS; j++) {
                temp[i] += W_up[i * NUM_MOTORS + j] * input_sequence[seq][j];
            }
            if(temp[i] < 0) temp[i] = 0; // ReLU
            hidden[i] += temp[i];
        }
    }
    
    // Down projection
    for(int i = 0; i < NUM_MOTORS; i++) {
        output[i] = b_down[i];
        for(int j = 0; j < MODEL_DIM; j++) {
            output[i] += W_down[i * MODEL_DIM + j] * hidden[j];
        }
    }
}

void backward(float** input_sequence, float* output, float* target) {
    memset(dW_up, 0, MODEL_DIM * NUM_MOTORS * sizeof(float));
    memset(db_up, 0, MODEL_DIM * sizeof(float));
    memset(dW_down, 0, NUM_MOTORS * MODEL_DIM * sizeof(float));
    memset(db_down, 0, NUM_MOTORS * sizeof(float));
    
    // Compute gradients for down projection
    for(int i = 0; i < NUM_MOTORS; i++) {
        float d_output = 2 * (output[i] - target[i]);
        db_down[i] += d_output;
        
        for(int j = 0; j < MODEL_DIM; j++) {
            dW_down[i * MODEL_DIM + j] += d_output * hidden[j];
        }
    }
    
    // Compute gradients for up projection
    for(int seq = 0; seq < SEQ_LEN; seq++) {
        for(int i = 0; i < MODEL_DIM; i++) {
            float temp_val = b_up[i];
            for(int j = 0; j < NUM_MOTORS; j++) {
                temp_val += W_up[i * NUM_MOTORS + j] * input_sequence[seq][j];
            }
            
            if(temp_val > 0) {  // ReLU derivative
                for(int j = 0; j < NUM_MOTORS; j++) {
                    dW_up[i * NUM_MOTORS + j] += temp_val * input_sequence[seq][j];
                }
                db_up[i] += temp_val;
            }
        }
    }
    
    // Update weights and biases
    for(int i = 0; i < MODEL_DIM * NUM_MOTORS; i++) {
        W_up[i] -= LEARNING_RATE * dW_up[i];
        W_down[i] -= LEARNING_RATE * dW_down[i];
    }
    for(int i = 0; i < MODEL_DIM; i++) {
        b_up[i] -= LEARNING_RATE * db_up[i];
    }
    for(int i = 0; i < NUM_MOTORS; i++) {
        b_down[i] -= LEARNING_RATE * db_down[i];
    }
}

float** load_csv(const char* filename, int* num_rows) {
    FILE* file = fopen(filename, "r");
    if(!file) return NULL;
    
    char line[1024];
    *num_rows = 0;
    
    fgets(line, sizeof(line), file); // Skip header
    while(fgets(line, sizeof(line), file)) (*num_rows)++;
    
    float** data = malloc(*num_rows * sizeof(float*));
    rewind(file);
    fgets(line, sizeof(line), file); // Skip header again
    
    for(int row = 0; row < *num_rows; row++) {
        data[row] = malloc(NUM_MOTORS * sizeof(float));
        fgets(line, sizeof(line), file);
        char* token = strtok(line, ",");
        for(int i = 0; i < 10; i++) token = strtok(NULL, ","); // Skip non-motor columns
        for(int i = 0; i < NUM_MOTORS; i++) {
            data[row][i] = atof(token);
            token = strtok(NULL, ",");
        }
    }
    
    fclose(file);
    return data;
}

void save_weights(const char* filename) {
    FILE* file = fopen(filename, "wb");
    fwrite(W_up, sizeof(float), MODEL_DIM * NUM_MOTORS, file);
    fwrite(b_up, sizeof(float), MODEL_DIM, file);
    fwrite(W_down, sizeof(float), NUM_MOTORS * MODEL_DIM, file);
    fwrite(b_down, sizeof(float), NUM_MOTORS, file);
    fclose(file);
}

void load_weights(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if(!file) return;
    fread(W_up, sizeof(float), MODEL_DIM * NUM_MOTORS, file);
    fread(b_up, sizeof(float), MODEL_DIM, file);
    fread(W_down, sizeof(float), NUM_MOTORS * MODEL_DIM, file);
    fread(b_down, sizeof(float), NUM_MOTORS, file);
    fclose(file);
}

int main() {
    int num_rows;
    float** data = load_csv("sim/2025-1-10_10-19-44_control_data.csv", &num_rows);
    if(!data) return 1;
    
    init_model();
    float output[NUM_MOTORS];
    
    for(int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        float total_loss = 0;
        
        for(int i = SEQ_LEN; i < num_rows - 1; i++) {
            for(int j = 0; j < SEQ_LEN; j++) {
                memcpy(sequences[j], data[i - SEQ_LEN + j], NUM_MOTORS * sizeof(float));
            }
            
            forward(sequences, output);
            
            float loss = 0;
            for(int j = 0; j < NUM_MOTORS; j++) {
                float diff = output[j] - data[i][j];
                loss += diff * diff;
            }
            total_loss += loss;
            
            backward(sequences, output, data[i]);
        }
        
        printf("Epoch %d, Loss: %f\n", epoch, total_loss / (num_rows - SEQ_LEN));
        
    }
    
    save_weights("model_weights.bin");
    
    // Cleanup
    for(int i = 0; i < num_rows; i++) free(data[i]);
    free(data);
    for(int i = 0; i < SEQ_LEN; i++) free(sequences[i]);
    free(sequences);
    free(W_up); free(b_up); free(W_down); free(b_down);
    free(dW_up); free(db_up); free(dW_down); free(db_down);
    free(hidden); free(temp);
    
    return 0;
}