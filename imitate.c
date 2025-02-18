#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "sim/quad.h"
#include "ssm/gpu/ssm.h"
#include "ssm/data.h"

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define INITIAL_SIM_TIME 0.1
#define TIME_INCREMENT 0.1
#define NUM_ITERATIONS 30
#define LOSS_THRESHOLD 0.01
#define MAX_EPOCHS 100000

// Helper function to get random value in range [min, max]
double random_range(double min, double max) {
    return min + (double)rand() / RAND_MAX * (max - min);
}

// Get difficulty-adjusted range based on current iteration
double get_range(double max_range, double sim_time) {
    // Start with 20% of max range at iteration 0, linearly increase to max_range
    double progress = (sim_time - INITIAL_SIM_TIME) / (NUM_ITERATIONS * TIME_INCREMENT);
    progress = fmin(1.0, fmax(0.0, progress)); // Clamp between 0 and 1
    return 0.2 * max_range + progress * 0.8 * max_range;
}

void cleanup_old_files(const char* base_fname, int current_iter) {
    if (current_iter <= 0) return;
    
    char old_data_fname[128], old_model_fname[128];
    sprintf(old_data_fname, "%s_flight_%d.csv", base_fname, current_iter - 1);
    sprintf(old_model_fname, "%s_policy_%d.bin", base_fname, current_iter - 1);
    
    if (remove(old_data_fname) == 0) {
        printf("Cleaned up: %s\n", old_data_fname);
    }
    if (remove(old_model_fname) == 0) {
        printf("Cleaned up: %s\n", old_model_fname);
    }
}

// Generate training data
void generate_training_data(const char* filename, double sim_time) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        printf("Error opening file: %s\n", filename);
        return;
    }
    
    // Write header
    fprintf(f, "vx,vy,vz,wx,wy,wz,tx,ty,tz,tyaw,m1,m2,m3,m4\n");
    
    int num_episodes = 16;//(int)(8 * (1 + 0.2 * log2(sim_time / INITIAL_SIM_TIME)));
    printf("Generating %d episodes for %.2f seconds each\n", num_episodes, sim_time);
    
    // Calculate ranges based on current simulation time
    double pos_range_xz = get_range(2.0, sim_time);  // Max range of ±2.0 for x,z
    double pos_range_y = get_range(2.0, sim_time);   // Max range of 2.0 for y above ground
    double target_height = get_range(2.0, sim_time);  // Max target height range
    
    printf("Current ranges - Position(x,z): ±%.2f, Height: 0-%.2f, Target height: 1-%.2f\n",
           pos_range_xz, pos_range_y, target_height + 1.0);
    
    for (int episode = 0; episode < num_episodes; episode++) {
        // Create quad with progressively increasing position ranges
        Quad* quad = create_quad(
            random_range(-pos_range_xz, pos_range_xz),     // x
            random_range(0.0, pos_range_y),                // y
            random_range(-pos_range_xz, pos_range_xz)      // z
        );
        
        // Generate target with progressively increasing ranges
        double target[7] = {
            random_range(-pos_range_xz, pos_range_xz),     // x
            random_range(1.0, 1.0 + target_height),        // y (always above ground)
            random_range(-pos_range_xz, pos_range_xz),     // z
            0.0, 0.0, 0.0,                                 // vx, vy, vz
            random_range(0.0, get_range(2*M_PI, sim_time)) // yaw (progressively increase angle)
        };
        
        double t_physics = 0.0;
        double t_control = 0.0;
        double total_time = 0.0;
        
        while (total_time < sim_time) {
            if (t_physics >= DT_PHYSICS) {
                update_quad(quad, DT_PHYSICS);
                t_physics = 0.0;
                total_time += DT_PHYSICS;
            }
            
            if (t_control >= DT_CONTROL) {
                control_quad(quad, target);
                
                fprintf(f, "%.6f,%.6f,%.6f,",
                       quad->linear_velocity_W[0],
                       quad->linear_velocity_W[1],
                       quad->linear_velocity_W[2]);
                       
                fprintf(f, "%.6f,%.6f,%.6f,",
                       quad->angular_velocity_B[0],
                       quad->angular_velocity_B[1],
                       quad->angular_velocity_B[2]);
                       
                fprintf(f, "%.6f,%.6f,%.6f,%.6f,",
                       target[0], target[1], target[2], target[6]);
                       
                fprintf(f, "%.6f,%.6f,%.6f,%.6f\n",
                       quad->omega_next[0],
                       quad->omega_next[1],
                       quad->omega_next[2],
                       quad->omega_next[3]);
                       
                t_control = 0.0;
            }
            
            t_physics += DT_PHYSICS;
            t_control += DT_PHYSICS;
        }
        
        free(quad);
        
        if ((episode + 1) % 10 == 0) {
            printf("Generated %d/%d episodes\n", episode + 1, num_episodes);
        }
    }
    
    fclose(f);
}

void train_policy(const char* data_file, const char* model_file, 
                 const char* previous_model, double sim_time) {
    printf("Loading training data from %s...\n", data_file);
    
    float *X, *y;
    int num_samples;
    load_csv(data_file, &X, &y, &num_samples, 10, 4);
    
    printf("Training data loaded: %d samples\n", num_samples);
    
    const int input_dim = 10;
    const int state_dim = 1024;
    const int output_dim = 4;
    const int batch_size = 16;//(int)(8 * (1 + 0.2 * log2(sim_time / INITIAL_SIM_TIME)));
    const int seq_length = (int)(sim_time / DT_CONTROL - ( 0.5));
    
    printf("Batch size: %d, Sequence length: %d\n", batch_size, seq_length);
    printf("Batch size * Sequence length: %d\n", batch_size * seq_length);
    
    SSM* ssm;
    if (previous_model) {
        ssm = load_model(previous_model);
        printf("Continuing training from previous model\n");
    } else {
        ssm = init_ssm(input_dim, state_dim, output_dim, batch_size);
        printf("Starting fresh training\n");
    }
    
    // Calculate adaptive learning rate
    const float initial_lr = 0.0003f;
    const float learning_rate = initial_lr * expf(-0.1f * (sim_time - INITIAL_SIM_TIME));
    printf("Adaptive learning rate: %.6f\n", learning_rate);
    
    // Transfer data to GPU
    float *d_X, *d_y;
    CHECK_CUDA(cudaMalloc(&d_X, num_samples * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, num_samples * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_X, X, num_samples * input_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, y, num_samples * output_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    float *d_batch_X, *d_batch_y;
    CHECK_CUDA(cudaMalloc(&d_batch_X, batch_size * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_batch_y, batch_size * output_dim * sizeof(float)));
    
    printf("Starting training for max %d epochs...\n", MAX_EPOCHS);
    
    float min_loss = INFINITY;
    int epoch;
    clock_t start_time = clock();
    
    for (epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        float epoch_loss = 0.0f;
        int num_batches = 0;
        
        for (int seq_start = 0; seq_start <= num_samples - seq_length; seq_start += seq_length) {
            CHECK_CUDA(cudaMemset(ssm->d_state, 0,
                                 ssm->batch_size * ssm->state_dim * sizeof(float)));
            
            for (int step = 0; step < seq_length; step += batch_size) {
                int current_batch_size = batch_size;
                if (seq_start + step + batch_size > num_samples) {
                    current_batch_size = num_samples - (seq_start + step);
                }
                
                CHECK_CUDA(cudaMemcpy(d_batch_X,
                                    &d_X[(seq_start + step) * input_dim],
                                    current_batch_size * input_dim * sizeof(float),
                                    cudaMemcpyDeviceToDevice));
                CHECK_CUDA(cudaMemcpy(d_batch_y,
                                    &d_y[(seq_start + step) * output_dim],
                                    current_batch_size * output_dim * sizeof(float),
                                    cudaMemcpyDeviceToDevice));
                
                forward_pass(ssm, d_batch_X);
                float loss = calculate_loss(ssm, d_batch_y);
                epoch_loss += loss;
                num_batches++;
                
                zero_gradients(ssm);
                backward_pass(ssm, d_batch_X);
                update_weights(ssm, learning_rate);
            }
        }
        
        float avg_loss = epoch_loss / num_batches;
        min_loss = fminf(min_loss, avg_loss);
        
        if ((epoch + 1) % 1000 == 0) {
            clock_t current_time = clock();
            double elapsed_seconds = (double)(current_time - start_time) / CLOCKS_PER_SEC;
            printf("Epoch [%d], Loss: %.8f, Time: %.1f seconds\n",
                   epoch + 1, avg_loss, elapsed_seconds);
        }
        
        if (avg_loss < LOSS_THRESHOLD) {
            printf("Reached target loss!\n");
            break;
        }
    }
    
    clock_t end_time = clock();
    double total_seconds = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    printf("Training completed in %.1f seconds\n", total_seconds);
    printf("Final loss: %.8f\n", min_loss);
    
    save_model(ssm, model_file);
    
    // Cleanup
    free(X);
    free(y);
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_batch_X);
    cudaFree(d_batch_y);
    free_ssm(ssm);
}

int main() {
    srand(time(NULL) ^ getpid());
    
    char base_fname[64];
    time_t now = time(NULL);
    strftime(base_fname, sizeof(base_fname), "%Y%m%d_%H%M%S", localtime(&now));
    
    char previous_model[128] = {0};
    
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        double current_sim_time = INITIAL_SIM_TIME + (i * TIME_INCREMENT);
        
        char data_fname[128], model_fname[128];
        sprintf(data_fname, "%s_flight_%d.csv", base_fname, i);
        sprintf(model_fname, "%s_policy_%d.bin", base_fname, i);
        
        printf("\nIteration %d/%d (Sim time: %.2f seconds)\n",
               i + 1, NUM_ITERATIONS, current_sim_time);
        
        printf("Phase 1: Generating training data...\n");
        generate_training_data(data_fname, current_sim_time);
        
        printf("Phase 2: Training policy network...\n");
        train_policy(data_fname, model_fname,
                    previous_model[0] ? previous_model : NULL,
                    current_sim_time);
        
        if (previous_model[0]) {
            if (remove(previous_model) == 0) {
                printf("Cleaned up previous model: %s\n", previous_model);
            }
        }
        
        cleanup_old_files(base_fname, i);
        
        strcpy(previous_model, model_fname);
        
        printf("Iteration complete!\n");
        printf("Current files:\n");
        printf("- Data: %s\n", data_fname);
        printf("- Model: %s\n", model_fname);
    }
    
    char final_data_fname[128];
    sprintf(final_data_fname, "%s_flight_%d.csv", base_fname, NUM_ITERATIONS - 1);
    if (remove(final_data_fname) == 0) {
        printf("Cleaned up final data file: %s\n", final_data_fname);
    }
    
    printf("\nProgressive training complete!\n");
    printf("Final model: %s\n", previous_model);
    
    return 0;
}