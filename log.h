#ifndef LOG_H
#define LOG_H

FILE* create_log_file(){
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    char filename[100];
    sprintf(filename, "%d-%d-%d_%d-%d-%d_trajectory.csv", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    FILE* csv_file = fopen(filename, "w");
    fprintf(csv_file, "pos[0],pos[1],pos[2],vel[0],vel[1],vel[2],ang_vel[0],ang_vel[1],ang_vel[2],acc_s[0],acc_s[1],acc_s[2],gyro_s[0],gyro_s[1],gyro_s[2],mean[0],mean[1],mean[2],mean[3],var[0],var[1],var[2],var[3],omega[0],omega[1],omega[2],omega[3],reward,discounted_return\n");
    return csv_file;
}

#endif // LOG_H