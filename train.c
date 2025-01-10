#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define SEQ_LEN 32
#define MODEL_DIM 64
#define NUM_MOTORS 4
#define LEARNING_RATE 0.001f
#define BATCH_SIZE 32
#define NUM_EPOCHS 1000

float W_up[MODEL_DIM * NUM_MOTORS];
float b_up[MODEL_DIM];
float W_down[NUM_MOTORS * MODEL_DIM];
float b_down[NUM_MOTORS];
float hidden[MODEL_DIM];
float sequences[SEQ_LEN][NUM_MOTORS];

void init_model() {
    for(int i = 0; i < MODEL_DIM * NUM_MOTORS; i++) {
        W_up[i] = (((float)rand() / (float)RAND_MAX) - 0.5f) * 0.02f;
        W_down[i] = (((float)rand() / (float)RAND_MAX) - 0.5f) * 0.02f;
    }
}

void forward(float* output) {
    memset(hidden, 0, sizeof(hidden));
    
    for(int seq = 0; seq < SEQ_LEN; seq++) {
        for(int i = 0; i < MODEL_DIM; i++) {
            float sum = b_up[i];
            for(int j = 0; j < NUM_MOTORS; j++) {
                sum += W_up[i * NUM_MOTORS + j] * sequences[seq][j];
            }
            hidden[i] += sum > 0 ? sum : 0; // ReLU
        }
    }
    
    for(int i = 0; i < NUM_MOTORS; i++) {
        float sum = b_down[i];
        for(int j = 0; j < MODEL_DIM; j++) {
            sum += W_down[i * MODEL_DIM + j] * hidden[j];
        }
        output[i] = sum;
    }
}

void backward(float* output, float* target) {
    for(int i = 0; i < NUM_MOTORS; i++) {
        float d_output = 2 * (output[i] - target[i]);
        b_down[i] -= LEARNING_RATE * d_output;
        
        for(int j = 0; j < MODEL_DIM; j++) {
            W_down[i * MODEL_DIM + j] -= LEARNING_RATE * d_output * hidden[j];
        }
    }
    
    for(int seq = 0; seq < SEQ_LEN; seq++) {
        for(int i = 0; i < MODEL_DIM; i++) {
            float sum = b_up[i];
            for(int j = 0; j < NUM_MOTORS; j++) {
                sum += W_up[i * NUM_MOTORS + j] * sequences[seq][j];
            }
            
            if(sum > 0) {
                b_up[i] -= LEARNING_RATE * sum;
                for(int j = 0; j < NUM_MOTORS; j++) {
                    W_up[i * NUM_MOTORS + j] -= LEARNING_RATE * sum * sequences[seq][j];
                }
            }
        }
    }
}

float** load_csv(const char* filename, int* num_rows) {
    FILE* file = fopen(filename, "r");
    if(!file) {
        printf("Failed to open file: %s\n", filename);
        return NULL;
    }
    
    char line[1024];
    *num_rows = 0;
    
    fgets(line, sizeof(line), file);
    while(fgets(line, sizeof(line), file)) (*num_rows)++;
    
    float** data = malloc(*num_rows * sizeof(float*));
    rewind(file);
    fgets(line, sizeof(line), file);
    
    for(int row = 0; row < *num_rows; row++) {
        data[row] = malloc(NUM_MOTORS * sizeof(float));
        fgets(line, sizeof(line), file);
        char* token = strtok(line, ",");
        for(int i = 0; i < 10; i++) token = strtok(NULL, ",");
        for(int i = 0; i < NUM_MOTORS; i++) {
            data[row][i] = atof(token);
            token = strtok(NULL, ",");
        }
    }
    
    fclose(file);
    return data;
}

int main() {
    srand(time(NULL));
    
    int num_rows;
    float** data = load_csv("2025-1-10_10-37-14_control_data.csv", &num_rows);
    if(!data) return 1;
    
    init_model();
    float output[NUM_MOTORS];
    
    // Create array of indices for shuffling
    int* indices = malloc((num_rows - SEQ_LEN) * sizeof(int));
    for(int i = 0; i < num_rows - SEQ_LEN; i++) {
        indices[i] = i + SEQ_LEN;
    }
    
    for(int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        float total_loss = 0;
        
        // Shuffle indices
        for(int i = 0; i < num_rows - SEQ_LEN; i++) {
            int j = i + rand() % (num_rows - SEQ_LEN - i);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        
        // Mini-batch training
        for(int batch = 0; batch < (num_rows - SEQ_LEN) / BATCH_SIZE; batch++) {
            float batch_loss = 0;
            
            for(int i = 0; i < BATCH_SIZE; i++) {
                int idx = indices[batch * BATCH_SIZE + i];
                
                // Load sequence
                for(int j = 0; j < SEQ_LEN; j++) {
                    memcpy(sequences[j], data[idx - SEQ_LEN + j], NUM_MOTORS * sizeof(float));
                }
                
                forward(output);
                
                float sample_loss = 0;
                for(int j = 0; j < NUM_MOTORS; j++) {
                    float diff = output[j] - data[idx][j];
                    sample_loss += diff * diff;
                }
                batch_loss += sample_loss;
                
                backward(output, data[idx]);
            }
            
            total_loss += batch_loss;
        }
        
        printf("Epoch %d, Loss: %f\n", epoch, total_loss / (num_rows - SEQ_LEN));
    }
    
    free(indices);
    for(int i = 0; i < num_rows; i++) free(data[i]);
    free(data);
    
    return 0;
}