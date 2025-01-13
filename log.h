#ifndef LOG_H
#define LOG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>

typedef struct{
    FILE* csv_file;
    double* rewards;
    char** trajectory_lines;
    int reward_count;

    double** data;
    double* targets;
    int* indices;
    int rows;
} Logger;

Logger* init_logger(){
    Logger* logger = malloc(sizeof(Logger));
    return logger;
}

void log_trajectory(Logger* logger, Sim* sim, double* output, double reward) {
    static bool first = true;
    if(first){
        char filename[100];
        strftime(filename, 100, "%Y-%m-%d_%H-%M-%S_trajectory.csv", localtime(&(time_t){time(NULL)}));
        logger->csv_file = fopen(filename, "w");
        fprintf(logger->csv_file, "pos[0],pos[1],pos[2],vel[0],vel[1],vel[2],ang_vel[0],ang_vel[1],ang_vel[2],acc_s[0],acc_s[1],acc_s[2],gyro_s[0],gyro_s[1],gyro_s[2],mean[0],mean[1],mean[2],mean[3],var[0],var[1],var[2],var[3],omega[0],omega[1],omega[2],omega[3],reward,cumulative_discounted_reward\n");
        logger->rewards = NULL;
        logger->trajectory_lines = NULL;
        logger->reward_count = 0;
        first = false;
    }else{
        logger->rewards = realloc(logger->rewards, (logger->reward_count + 1) * sizeof(double));
        logger->rewards[logger->reward_count] = reward;
        logger->trajectory_lines = realloc(logger->trajectory_lines, (logger->reward_count + 1) * sizeof(char*));
        logger->trajectory_lines[logger->reward_count] = malloc(1024);
        snprintf(logger->trajectory_lines[logger->reward_count], 1024, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,0.0\n", sim->quad->linear_position_W[0], sim->quad->linear_position_W[1], sim->quad->linear_position_W[2], sim->quad->linear_velocity_W[0], sim->quad->linear_velocity_W[1], sim->quad->linear_velocity_W[2], sim->quad->angular_velocity_B[0], sim->quad->angular_velocity_B[1], sim->quad->angular_velocity_B[2], sim->quad->linear_acceleration_B_s[0], sim->quad->linear_acceleration_B_s[1], sim->quad->linear_acceleration_B_s[2], sim->quad->angular_velocity_B_s[0], sim->quad->angular_velocity_B_s[1], sim->quad->angular_velocity_B_s[2], output[0], output[1], output[2], output[3], output[4], output[5], output[6], output[7], sim->quad->omega_next[0], sim->quad->omega_next[1], sim->quad->omega_next[2], sim->quad->omega_next[3], reward);
        logger->reward_count++;
    }
}

void save_trajectories(Logger* logger){
    if (logger->reward_count > 0) {
        for(int i = 0; i < logger->reward_count; i++) {
            double discounted_return = 0.0;
            double gamma = 0.99;
            double discount = 1.0;
            for(int j = i; j < logger->reward_count; j++) {
                discounted_return += discount * logger->rewards[j];
                discount *= gamma;
            }
            char *line = logger->trajectory_lines[i];
            char *last_comma = strrchr(line, ',');
            sprintf(last_comma + 1, "%f\n", discounted_return);
            fprintf(logger->csv_file, "%s", line);
            free(line);
        }
        
        free(logger->trajectory_lines);
        free(logger->rewards);
        logger->rewards = NULL;
        logger->trajectory_lines = NULL;
        logger->reward_count = 0;
    }
}

void read_trajectories(Logger* logger, const char* filename){
    FILE *f = fopen(filename, "r");
    if (!f) { printf("Failed to open file\n"); return 1; }

    // Count rows (excluding header)
    char line[1024];
    logger->rows = -1;
    while (fgets(line, sizeof(line), f)) logger->rowsrows++;
    rewind(f);
    fgets(line, sizeof(line), f);  // Skip header

    // Allocate memory
    logger->data = malloc(logger->rows * sizeof(double*));
    logger->targets = malloc(logger->rows * sizeof(double));
    logger->indices = malloc(logger->rows * sizeof(int));
    for(int i = 0; i < logger->rows; i++) {
        data[i] = malloc(M_IN * sizeof(double));
        indices[i] = i;
    }

    // Read data
    for(int i = 0; i < logger->rows; i++) {
        if (!fgets(line, sizeof(line), f)) break;
        char *ptr = line;
        
        // Read position (3), velocity (3), and angular velocity (3)
        for(int j = 0; j < M_IN; j++) {
            logger->data[i][j] = atof(strsep(&ptr, ","));
        }
        
        // Skip acc_s (3), gyro_s (3), means (4), vars (4), omega (4)
        for(int j = 0; j < 18; j++) strsep(&ptr, ",");
        
        // Skip immediate reward
        strsep(&ptr, ",");
        
        // Get discounted return (target)
        logger->targets[i] = atof(strsep(&ptr, ",\n"));
    }
    fclose(f);
}

void free_logger(Logger* logger){
    if(logger->rewards != NULL) {
        for(int i = 0; i < logger->reward_count; i++) {
            free(logger->trajectory_lines[i]);
        }
        free(logger->rewards);
        free(logger->trajectory_lines);
        fclose(logger->csv_file);
    }else{
        free(logger->data);
        free(logger->targets);
        free(logger->indices);
    }
    free(logger);
}

#endif // LOG_H