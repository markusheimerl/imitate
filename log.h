#ifndef LOG_H
#define LOG_H

typedef struct{
    FILE* csv_file;
    double* rewards;
    char** trajectory_lines;
    int reward_count;
} Logger;

Logger* init_logger(){
    Logger* logger = malloc(sizeof(Logger));
    char filename[100];
    strftime(filename, 100, "%Y-%m-%d_%H-%M-%S_trajectory.csv", localtime(&(time_t){time(NULL)}));
    logger->csv_file = fopen(filename, "w");
    fprintf(logger->csv_file, "pos[0],pos[1],pos[2],vel[0],vel[1],vel[2],ang_vel[0],ang_vel[1],ang_vel[2],acc_s[0],acc_s[1],acc_s[2],gyro_s[0],gyro_s[1],gyro_s[2],mean[0],mean[1],mean[2],mean[3],var[0],var[1],var[2],var[3],omega[0],omega[1],omega[2],omega[3],reward,cumulative_discounted_reward\n");
    logger->rewards = NULL;
    logger->trajectory_lines = NULL;
    logger->reward_count = 0;
    return logger;
}

void log_trajectory(Logger* logger, Sim* sim, double* output, double reward) {
    logger->rewards = realloc(logger->rewards, (logger->reward_count + 1) * sizeof(double));
    logger->rewards[logger->reward_count] = reward;
    logger->trajectory_lines = realloc(logger->trajectory_lines, (logger->reward_count + 1) * sizeof(char*));
    logger->trajectory_lines[logger->reward_count] = malloc(1024);
    snprintf(logger->trajectory_lines[logger->reward_count], 1024, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,0.0\n", sim->quad->linear_position_W[0], sim->quad->linear_position_W[1], sim->quad->linear_position_W[2], sim->quad->linear_velocity_W[0], sim->quad->linear_velocity_W[1], sim->quad->linear_velocity_W[2], sim->quad->angular_velocity_B[0], sim->quad->angular_velocity_B[1], sim->quad->angular_velocity_B[2], sim->quad->linear_acceleration_B_s[0], sim->quad->linear_acceleration_B_s[1], sim->quad->linear_acceleration_B_s[2], sim->quad->angular_velocity_B_s[0], sim->quad->angular_velocity_B_s[1], sim->quad->angular_velocity_B_s[2], output[0], output[1], output[2], output[3], output[4], output[5], output[6], output[7], sim->quad->omega_next[0], sim->quad->omega_next[1], sim->quad->omega_next[2], sim->quad->omega_next[3], reward);
    logger->reward_count++;
}

void save_logger(Logger* logger){
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

void free_logger(Logger* logger){
    for(int i = 0; i < logger->reward_count; i++) {
        free(logger->trajectory_lines[i]);
    }
    free(logger->rewards);
    free(logger->trajectory_lines);
    fclose(logger->csv_file);
    free(logger);
}

#endif // LOG_H