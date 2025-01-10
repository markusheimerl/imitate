#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define SEQ_LEN 32
#define MODEL_DIM 64
#define NUM_MOTORS 4
#define LR 0.001f
#define B1 0.9f
#define B2 0.999f
#define EPS 1e-8f
#define DECAY 0.01f
#define CLIP 1.0f

typedef struct {
    double *W_up, *b_up, *W_down, *b_down, *hidden;
    double *d_W_up, *d_b_up, *d_W_down, *d_b_down;
    double *m_W_up, *m_b_up, *m_W_down, *m_b_down;
    double *v_W_up, *v_b_up, *v_W_down, *v_b_down;
} Model;

double* alloc(int size) { return malloc(size * sizeof(double)); }
double* calloc_f(int size) { return calloc(size, sizeof(double)); }

Model* init_model() {
    Model* m = malloc(sizeof(Model));
    double scale_up = sqrtf(2.0f/NUM_MOTORS), scale_down = sqrtf(2.0f/MODEL_DIM);
    
    m->W_up = alloc(MODEL_DIM * NUM_MOTORS);
    m->W_down = alloc(MODEL_DIM * NUM_MOTORS);
    m->b_up = calloc_f(MODEL_DIM);
    m->b_down = calloc_f(NUM_MOTORS);
    m->hidden = alloc(MODEL_DIM);
    
    m->d_W_up = alloc(MODEL_DIM * NUM_MOTORS);
    m->d_W_down = alloc(MODEL_DIM * NUM_MOTORS);
    m->d_b_up = alloc(MODEL_DIM);
    m->d_b_down = alloc(NUM_MOTORS);
    
    for(int i = 0; i < 4; i++) {
        double **ptrs[] = {&m->m_W_up, &m->m_b_up, &m->m_W_down, &m->m_b_down,
                         &m->v_W_up, &m->v_b_up, &m->v_W_down, &m->v_b_down};
        int sizes[] = {MODEL_DIM * NUM_MOTORS, MODEL_DIM, MODEL_DIM * NUM_MOTORS, NUM_MOTORS};
        *ptrs[i] = calloc_f(sizes[i % 4]);
        *ptrs[i+4] = calloc_f(sizes[i % 4]);
    }
    
    for(int i = 0; i < MODEL_DIM * NUM_MOTORS; i++) {
        m->W_up[i] = ((double)rand()/RAND_MAX - 0.5f) * scale_up;
        m->W_down[i] = ((double)rand()/RAND_MAX - 0.5f) * scale_down;
    }
    return m;
}

void forward(Model* m, double (*seq)[NUM_MOTORS], double* out) {
    memset(m->hidden, 0, MODEL_DIM * sizeof(double));
    
    for(int s = 0; s < SEQ_LEN; s++)
        for(int i = 0; i < MODEL_DIM; i++) {
            double sum = m->b_up[i];
            for(int j = 0; j < NUM_MOTORS; j++) 
                sum += m->W_up[i * NUM_MOTORS + j] * seq[s][j];
            m->hidden[i] += fmaxf(0.0f, sum);
        }
    
    for(int i = 0; i < NUM_MOTORS; i++) {
        double sum = m->b_down[i];
        for(int j = 0; j < MODEL_DIM; j++)
            sum += m->W_down[i * MODEL_DIM + j] * m->hidden[j];
        out[i] = sum;
    }
}

void adam_update(double* p, double* g, double* m, double* v, int size, int t) {
    double lr_t = LR * sqrtf(1.0f - powf(B2, t)) / (1.0f - powf(B1, t));
    double norm = 0;
    for(int i = 0; i < size; i++) norm += g[i] * g[i];
    
    if(sqrtf(norm) > CLIP) {
        double scale = CLIP / sqrtf(norm);
        for(int i = 0; i < size; i++) g[i] *= scale;
    }
    
    for(int i = 0; i < size; i++) {
        m[i] = B1 * m[i] + (1-B1) * g[i];
        v[i] = B2 * v[i] + (1-B2) * g[i] * g[i];
        p[i] -= lr_t * (m[i]/(sqrtf(v[i]) + EPS) + DECAY * p[i]);
    }
}

double train_step(Model* m, double (*seq)[NUM_MOTORS], double* target, int step) {
    double out[NUM_MOTORS], loss = 0;
    forward(m, seq, out);
    
    memset(m->d_W_up, 0, MODEL_DIM * NUM_MOTORS * sizeof(double));
    memset(m->d_b_up, 0, MODEL_DIM * sizeof(double));
    memset(m->d_W_down, 0, NUM_MOTORS * MODEL_DIM * sizeof(double));
    memset(m->d_b_down, 0, NUM_MOTORS * sizeof(double));
    
    for(int i = 0; i < NUM_MOTORS; i++) {
        double d_out = 2 * (out[i] - target[i]);
        loss += (out[i] - target[i]) * (out[i] - target[i]);
        m->d_b_down[i] = d_out;
        
        for(int j = 0; j < MODEL_DIM; j++) {
            m->d_W_down[i * MODEL_DIM + j] = d_out * m->hidden[j];
            double d_hidden = d_out * m->W_down[i * MODEL_DIM + j];
            
            for(int s = 0; s < SEQ_LEN; s++) {
                double sum = m->b_up[j];
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
    
    adam_update(m->W_up, m->d_W_up, m->m_W_up, m->v_W_up, MODEL_DIM * NUM_MOTORS, step);
    adam_update(m->b_up, m->d_b_up, m->m_b_up, m->v_b_up, MODEL_DIM, step);
    adam_update(m->W_down, m->d_W_down, m->m_W_down, m->v_W_down, NUM_MOTORS * MODEL_DIM, step);
    adam_update(m->b_down, m->d_b_down, m->m_b_down, m->v_b_down, NUM_MOTORS, step);
    
    return loss;
}

int main() {
    srand(time(NULL));
    Model* model = init_model();
    FILE* f = fopen("2025-1-10_10-43-1_control_data.csv", "r");
    if(!f) return 1;
    
    int rows = 0;
    char line[1024];
    fgets(line, sizeof(line), f);
    while(fgets(line, sizeof(line), f)) rows++;
    
    double** data = malloc(rows * sizeof(double*));
    rewind(f);
    fgets(line, sizeof(line), f);
    
    for(int i = 0; i < rows; i++) {
        data[i] = alloc(NUM_MOTORS);
        fgets(line, sizeof(line), f);
        char* token = strtok(line, ",");
        for(int j = 0; j < 10; j++) token = strtok(NULL, ",");
        for(int j = 0; j < NUM_MOTORS; j++)
            data[i][j] = atof(token), token = strtok(NULL, ",");
    }
    fclose(f);
    
    double seq[SEQ_LEN][NUM_MOTORS];
    int step = 1;
    for(int epoch = 0; epoch < 1000; epoch++) {
        double loss = 0;
        int samples = 0;
        
        for(int i = SEQ_LEN; i < rows; i++) {
            for(int j = 0; j < SEQ_LEN; j++)
                memcpy(seq[j], data[i - SEQ_LEN + j], NUM_MOTORS * sizeof(double));
            loss += train_step(model, seq, data[i], step++);
            samples++;
        }
        printf("Epoch %d, Loss: %f\n", epoch, loss / samples);
    }
    
    for(int i = 0; i < rows; i++) free(data[i]);
    free(data);
    for(int i = 0; i < 16; i++) free(((double**)model)[i]);
    free(model);
    return 0;
}