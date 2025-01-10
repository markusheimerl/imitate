#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define SEQ_LEN 32
#define MODEL_DIM 64
#define NUM_MOTORS 4
#define LEARNING_RATE 0.001f
#define CLIP_THRESHOLD 1.0f

typedef struct {
    float* W_up;
    float* b_up;
    float* W_down;
    float* b_down;
    float* hidden;
    float* d_W_up;
    float* d_b_up;
    float* d_W_down;
    float* d_b_down;
} Model;

Model* init_model() {
    Model* m = malloc(sizeof(Model));
    m->W_up = malloc(MODEL_DIM * NUM_MOTORS * sizeof(float));
    m->b_up = calloc(MODEL_DIM, sizeof(float));
    m->W_down = malloc(NUM_MOTORS * MODEL_DIM * sizeof(float));
    m->b_down = calloc(NUM_MOTORS, sizeof(float));
    m->hidden = malloc(MODEL_DIM * sizeof(float));
    m->d_W_up = malloc(MODEL_DIM * NUM_MOTORS * sizeof(float));
    m->d_b_up = malloc(MODEL_DIM * sizeof(float));
    m->d_W_down = malloc(NUM_MOTORS * MODEL_DIM * sizeof(float));
    m->d_b_down = malloc(NUM_MOTORS * sizeof(float));

    float scale_up = sqrtf(2.0f/NUM_MOTORS);
    float scale_down = sqrtf(2.0f/MODEL_DIM);
    for(int i = 0; i < MODEL_DIM * NUM_MOTORS; i++) {
        m->W_up[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * scale_up;
        m->W_down[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * scale_down;
    }
    return m;
}

void forward(Model* m, float (*seq)[NUM_MOTORS], float* out) {
    memset(m->hidden, 0, MODEL_DIM * sizeof(float));
    
    for(int s = 0; s < SEQ_LEN; s++) {
        for(int i = 0; i < MODEL_DIM; i++) {
            float sum = m->b_up[i];
            for(int j = 0; j < NUM_MOTORS; j++) 
                sum += m->W_up[i * NUM_MOTORS + j] * seq[s][j];
            m->hidden[i] += fmaxf(0.0f, sum);
        }
    }
    
    for(int i = 0; i < NUM_MOTORS; i++) {
        float sum = m->b_down[i];
        for(int j = 0; j < MODEL_DIM; j++)
            sum += m->W_down[i * MODEL_DIM + j] * m->hidden[j];
        out[i] = sum;
    }
}

void clip_gradients(float* grad, int size) {
    float norm = 0;
    for(int i = 0; i < size; i++) 
        norm += grad[i] * grad[i];
    norm = sqrtf(norm);
    
    if(norm > CLIP_THRESHOLD) {
        float scale = CLIP_THRESHOLD / norm;
        for(int i = 0; i < size; i++)
            grad[i] *= scale;
    }
}

float train_step(Model* m, float (*seq)[NUM_MOTORS], float* target) {
    float out[NUM_MOTORS];
    forward(m, seq, out);
    
    float loss = 0;
    memset(m->d_W_up, 0, MODEL_DIM * NUM_MOTORS * sizeof(float));
    memset(m->d_b_up, 0, MODEL_DIM * sizeof(float));
    memset(m->d_W_down, 0, NUM_MOTORS * MODEL_DIM * sizeof(float));
    memset(m->d_b_down, 0, NUM_MOTORS * sizeof(float));
    
    // Backward pass
    for(int i = 0; i < NUM_MOTORS; i++) {
        float d_out = 2 * (out[i] - target[i]);
        loss += (out[i] - target[i]) * (out[i] - target[i]);
        m->d_b_down[i] = d_out;
        
        for(int j = 0; j < MODEL_DIM; j++) {
            m->d_W_down[i * MODEL_DIM + j] = d_out * m->hidden[j];
            float d_hidden = d_out * m->W_down[i * MODEL_DIM + j];
            
            for(int s = 0; s < SEQ_LEN; s++) {
                float sum = m->b_up[j];
                for(int k = 0; k < NUM_MOTORS; k++)
                    sum += m->W_up[j * NUM_MOTORS + k] * seq[s][k];
                
                if(sum > 0) {
                    m->d_b_up[j] += d_hidden;
                    for(int k = 0; k < NUM_MOTORS; k++)
                        m->d_W_up[j * NUM_MOTORS + k] += d_hidden * seq[s][k];
                }
            }
        }
    }
    
    // Clip and apply gradients
    clip_gradients(m->d_W_up, MODEL_DIM * NUM_MOTORS);
    clip_gradients(m->d_W_down, NUM_MOTORS * MODEL_DIM);
    clip_gradients(m->d_b_up, MODEL_DIM);
    clip_gradients(m->d_b_down, NUM_MOTORS);
    
    for(int i = 0; i < MODEL_DIM * NUM_MOTORS; i++) {
        m->W_up[i] -= LEARNING_RATE * m->d_W_up[i];
        m->W_down[i] -= LEARNING_RATE * m->d_W_down[i];
    }
    for(int i = 0; i < MODEL_DIM; i++)
        m->b_up[i] -= LEARNING_RATE * m->d_b_up[i];
    for(int i = 0; i < NUM_MOTORS; i++)
        m->b_down[i] -= LEARNING_RATE * m->d_b_down[i];
    
    return loss;
}

void free_model(Model* m) {
    free(m->W_up);
    free(m->b_up);
    free(m->W_down);
    free(m->b_down);
    free(m->hidden);
    free(m->d_W_up);
    free(m->d_b_up);
    free(m->d_W_down);
    free(m->d_b_down);
    free(m);
}

int main() {
    srand(time(NULL));
    Model* model = init_model();
    
    // Load data...
    FILE* f = fopen("2025-1-10_10-43-1_control_data.csv", "r");
    if(!f) return 1;
    
    int rows = 0;
    char line[1024];
    fgets(line, sizeof(line), f);
    while(fgets(line, sizeof(line), f)) rows++;
    
    float** data = malloc(rows * sizeof(float*));
    rewind(f);
    fgets(line, sizeof(line), f);
    
    for(int i = 0; i < rows; i++) {
        data[i] = malloc(NUM_MOTORS * sizeof(float));
        fgets(line, sizeof(line), f);
        char* token = strtok(line, ",");
        for(int j = 0; j < 10; j++) token = strtok(NULL, ",");
        for(int j = 0; j < NUM_MOTORS; j++)
            data[i][j] = atof(token), token = strtok(NULL, ",");
    }
    fclose(f);
    
    // Train
    float seq[SEQ_LEN][NUM_MOTORS];
    for(int epoch = 0; epoch < 1000; epoch++) {
        float loss = 0;
        int samples = 0;
        
        for(int i = SEQ_LEN; i < rows; i++) {
            for(int j = 0; j < SEQ_LEN; j++)
                memcpy(seq[j], data[i - SEQ_LEN + j], NUM_MOTORS * sizeof(float));
            loss += train_step(model, seq, data[i]);
            samples++;
        }
        printf("Epoch %d, Loss: %f\n", epoch, loss / samples);
    }
    
    for(int i = 0; i < rows; i++) free(data[i]);
    free(data);
    free_model(model);
    return 0;
}