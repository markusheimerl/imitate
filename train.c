#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define SEQ_LEN 32
#define MODEL_DIM 64
#define NUM_MOTORS 4
#define LEARNING_RATE 0.001f
#define BETA1 0.9f
#define BETA2 0.999f
#define EPSILON 1e-8f
#define WEIGHT_DECAY 0.01f
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
    // Adam parameters
    float* m_W_up;
    float* m_b_up;
    float* m_W_down;
    float* m_b_down;
    float* v_W_up;
    float* v_b_up;
    float* v_W_down;
    float* v_b_down;
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
    
    // Initialize Adam parameters
    m->m_W_up = calloc(MODEL_DIM * NUM_MOTORS, sizeof(float));
    m->m_b_up = calloc(MODEL_DIM, sizeof(float));
    m->m_W_down = calloc(NUM_MOTORS * MODEL_DIM, sizeof(float));
    m->m_b_down = calloc(NUM_MOTORS, sizeof(float));
    m->v_W_up = calloc(MODEL_DIM * NUM_MOTORS, sizeof(float));
    m->v_b_up = calloc(MODEL_DIM, sizeof(float));
    m->v_W_down = calloc(NUM_MOTORS * MODEL_DIM, sizeof(float));
    m->v_b_down = calloc(NUM_MOTORS, sizeof(float));

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

void adam_update(float* param, float* grad, float* m, float* v, int size, int t) {
    float lr_t = LEARNING_RATE * sqrtf(1.0f - powf(BETA2, t)) / (1.0f - powf(BETA1, t));
    
    for(int i = 0; i < size; i++) {
        m[i] = BETA1 * m[i] + (1.0f - BETA1) * grad[i];
        v[i] = BETA2 * v[i] + (1.0f - BETA2) * grad[i] * grad[i];
        float m_hat = m[i] / (1.0f - powf(BETA1, t));
        float v_hat = v[i] / (1.0f - powf(BETA2, t));
        param[i] -= lr_t * (m_hat / (sqrtf(v_hat) + EPSILON) + WEIGHT_DECAY * param[i]);
    }
}

float train_step(Model* m, float (*seq)[NUM_MOTORS], float* target, int step) {
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
    
    // Clip gradients
    clip_gradients(m->d_W_up, MODEL_DIM * NUM_MOTORS);
    clip_gradients(m->d_W_down, NUM_MOTORS * MODEL_DIM);
    clip_gradients(m->d_b_up, MODEL_DIM);
    clip_gradients(m->d_b_down, NUM_MOTORS);
    
    // Apply AdamW updates
    adam_update(m->W_up, m->d_W_up, m->m_W_up, m->v_W_up, MODEL_DIM * NUM_MOTORS, step);
    adam_update(m->b_up, m->d_b_up, m->m_b_up, m->v_b_up, MODEL_DIM, step);
    adam_update(m->W_down, m->d_W_down, m->m_W_down, m->v_W_down, NUM_MOTORS * MODEL_DIM, step);
    adam_update(m->b_down, m->d_b_down, m->m_b_down, m->v_b_down, NUM_MOTORS, step);
    
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
    free(m->m_W_up);
    free(m->m_b_up);
    free(m->m_W_down);
    free(m->m_b_down);
    free(m->v_W_up);
    free(m->v_b_up);
    free(m->v_W_down);
    free(m->v_b_down);
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
    int global_step = 1;
    for(int epoch = 0; epoch < 1000; epoch++) {
        float loss = 0;
        int samples = 0;
        
        for(int i = SEQ_LEN; i < rows; i++) {
            for(int j = 0; j < SEQ_LEN; j++)
                memcpy(seq[j], data[i - SEQ_LEN + j], NUM_MOTORS * sizeof(float));
            loss += train_step(model, seq, data[i], global_step++);
            samples++;
        }
        printf("Epoch %d, Loss: %f\n", epoch, loss / samples);
    }
    
    for(int i = 0; i < rows; i++) free(data[i]);
    free(data);
    free_model(model);
    return 0;
}