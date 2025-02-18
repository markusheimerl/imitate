#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "sim/quad.h"
#include "mlp/gpu/mlp.h"
#include "mlp/data.h"

#define DT_PHYSICS  (1.0 / 1000.0)
#define DT_CONTROL  (1.0 / 60.0)
#define INITIAL_SIM_TIME 2.0
#define TIME_INCREMENT 1.0
#define NUM_ITERATIONS 30
#define LOSS_THRESHOLD 0.01
#define MAX_EPOCHS 1000000

// Helper function to get random value in range [min, max]
double random_range(double min, double max) {
    return min + (double)rand() / RAND_MAX * (max - min);
}

// Check if quadcopter has reached target
int target_reached(Quad* quad, double* target) {
    double dx = quad->linear_position_W[0] - target[0];
    double dy = quad->linear_position_W[1] - target[1];
    double dz = quad->linear_position_W[2] - target[2];
    double distance = sqrt(dx*dx + dy*dy + dz*dz);
    double velocity_mag = sqrt(
        quad->linear_velocity_W[0] * quad->linear_velocity_W[0] +
        quad->linear_velocity_W[1] * quad->linear_velocity_W[1] +
        quad->linear_velocity_W[2] * quad->linear_velocity_W[2]
    );
    return (distance < 0.05 && velocity_mag < 0.02);
}

// Generate new random target
void generate_target(double* target) {
    target[0] = random_range(-2.0, 2.0);     // x
    target[1] = random_range(1.0, 3.0);      // y: Always above ground
    target[2] = random_range(-2.0, 2.0);     // z
    target[3] = target[4] = target[5] = 0.0; // vx, vy, vz
    target[6] = random_range(0.0, 2*M_PI);   // yaw
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
    
    fprintf(f, "vx,vy,vz,wx,wy,wz,tx,ty,tz,tyaw,m1,m2,m3,m4\n");
    
    Quad* quad = create_quad(
        random_range(-2.0, 2.0),
        random_range(1.0, 3.0),
        random_range(-2.0, 2.0)
    );
    
    double target[7];
    generate_target(target);
    
    int target_count = 0;
    double t_physics = 0.0;
    double t_control = 0.0;
    double total_time = 0.0;
    
    while (total_time < sim_time) {
        if (t_physics >= DT_PHYSICS) {
            update_quad(quad, DT_PHYSICS);
            t_physics = 0.0;
            total_time += DT_PHYSICS;
            
            if (target_reached(quad, target)) {
                target_count++;
                generate_target(target);
            }
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
    fclose(f);
    
    printf("Generated data for %.2f seconds with %d targets reached\n", total_time, target_count);
}

void train_policy(const char* data_file, const char* model_file, const char* previous_model, double sim_time) {
    printf("Loading training data from %s...\n", data_file);
    
    float *X, *y;
    int num_samples;
    load_csv(data_file, &X, &y, &num_samples, 10, 4);
    
    printf("Training data loaded: %d samples\n", num_samples);
    
    Net* net;
    if (previous_model) {
        net = load_model(previous_model);
        Net* old_net = net;
        
        net = init_net(net->input_dim, net->hidden_dim, net->output_dim, num_samples);
        
        CHECK_CUDA(cudaMemcpy(net->d_fc1_weight, old_net->d_fc1_weight,
                             net->hidden_dim * net->input_dim * sizeof(float),
                             cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(net->d_fc2_weight, old_net->d_fc2_weight,
                             net->output_dim * net->hidden_dim * sizeof(float),
                             cudaMemcpyDeviceToDevice));
        
        net->t = old_net->t;
        CHECK_CUDA(cudaMemcpy(net->d_fc1_m, old_net->d_fc1_m,
                             net->hidden_dim * net->input_dim * sizeof(float),
                             cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(net->d_fc1_v, old_net->d_fc1_v,
                             net->hidden_dim * net->input_dim * sizeof(float),
                             cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(net->d_fc2_m, old_net->d_fc2_m,
                             net->output_dim * net->hidden_dim * sizeof(float),
                             cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(net->d_fc2_v, old_net->d_fc2_v,
                             net->output_dim * net->hidden_dim * sizeof(float),
                             cudaMemcpyDeviceToDevice));
        
        free_net(old_net);
        
        printf("Continuing training from previous model with new batch size: %d\n", num_samples);
    } else {
        net = init_net(10, 4096, 4, num_samples);
        printf("Starting fresh training\n");
    }
    
    // Calculate adaptive learning rate based on simulation time
    // Start with 0.001 at INITIAL_SIM_TIME and decrease exponentially
    const float initial_lr = 0.001f;
    const float learning_rate = initial_lr * expf(-0.1f * (sim_time - INITIAL_SIM_TIME));
    printf("Adaptive learning rate: %.6f\n", learning_rate);
    
    float loss = INFINITY;
    int epoch = 0;
    
    printf("Training until loss < %.4f (max epochs: %d)...\n", LOSS_THRESHOLD, MAX_EPOCHS);
    
    clock_t start_time = clock();
    
    while (loss > LOSS_THRESHOLD && epoch < MAX_EPOCHS) {
        forward_pass(net, X);
        loss = calculate_loss(net, y);
        zero_gradients(net);
        backward_pass(net, X);
        update_weights(net, learning_rate);
        
        if (epoch == 0 || (epoch + 1) % 10000 == 0) {
            clock_t current_time = clock();
            double elapsed_seconds = (double)(current_time - start_time) / CLOCKS_PER_SEC;
            printf("Epoch [%d], Loss: %.8f, Time: %.1f seconds\n", 
                   epoch + 1, loss, elapsed_seconds);
        }
        
        epoch++;
    }
    
    clock_t end_time = clock();
    double total_seconds = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    if (loss <= LOSS_THRESHOLD) {
        printf("Reached target loss in %d epochs (%.1f seconds)\n", epoch, total_seconds);
    } else {
        printf("Reached max epochs. Final loss: %.8f (%.1f seconds)\n", loss, total_seconds);
    }
    
    net->batch_size = 1;
    save_model(net, model_file);
    
    free(X);
    free(y);
    free_net(net);
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