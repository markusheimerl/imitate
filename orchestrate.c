#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/time.h>
#include "grad/grad.h"

#define NUM_PROCESSES 8
#define GENERATIONS 100
#define MUTATION_STRENGTH 0.1
#define ELITE_COUNT 2

typedef struct {
    double mean_return;
    double std_return;
    double min_return;
    double max_return;
    int pid;
    char weights_file[64];
} ProcessResult;

unsigned int get_unique_seed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (unsigned int)(tv.tv_sec * 1000000 + tv.tv_usec) ^ getpid();
}

void mutate_weights(Net* net, double strength) {
    for(int i = 0; i < net->n; i++) {
        int in = net->sz[i], out = net->sz[i+1];
        for(int j = 0; j < in*out; j++) {
            if((double)rand()/RAND_MAX < 0.3) {  // 30% mutation probability
                double noise = strength * ((2.0*(double)rand()/RAND_MAX) - 1.0);
                net->w[i][j] *= (1.0 + noise);
            }
        }
        for(int j = 0; j < out; j++) {
            if((double)rand()/RAND_MAX < 0.3) {
                double noise = strength * ((2.0*(double)rand()/RAND_MAX) - 1.0);
                net->b[i][j] *= (1.0 + noise);
            }
        }
    }
}

int compare_results(const void* a, const void* b) {
    ProcessResult* ra = (ProcessResult*)a;
    ProcessResult* rb = (ProcessResult*)b;
    if(ra->mean_return > rb->mean_return) return -1;
    if(ra->mean_return < rb->mean_return) return 1;
    return 0;
}

int main() {
    srand(get_unique_seed());
    
    ProcessResult* results = mmap(NULL, NUM_PROCESSES * sizeof(ProcessResult),
                                PROT_READ | PROT_WRITE,
                                MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    
    int layers[] = {12, 64, 64, 64, 8};
    Net* base_net = init_net(5, layers, adamw);
    if(!base_net) return 1;
    
    char base_weights[64];
    strftime(base_weights, sizeof(base_weights), "base_%Y%m%d_%H%M%S.bin", 
             localtime(&(time_t){time(NULL)}));
    save_weights(base_weights, base_net);
    
    for(int generation = 0; generation < GENERATIONS; generation++) {
        printf("\nGeneration %d/%d\n", generation + 1, GENERATIONS);
        
        for(int i = 0; i < NUM_PROCESSES; i++) {
            pid_t pid = fork();
            
            if(pid == 0) {  // Child process
                srand(get_unique_seed());  // Re-seed with unique value
                
                Net* net = load_weights(base_weights, adamw);
                if(i >= ELITE_COUNT) {
                    mutate_weights(net, MUTATION_STRENGTH);
                }
                
                char weights_file[64];
                sprintf(weights_file, "weights_%d.bin", i);
                save_weights(weights_file, net);
                strcpy(results[i].weights_file, weights_file);
                
                char command[256];
                sprintf(command, "./reinforce.out %s > process_%d.txt", weights_file, i);
                system(command);
                
                char filename[64];
                sprintf(filename, "process_%d.txt", i);
                FILE* fp = fopen(filename, "r");
                if(fp) {
                    char line[256];
                    while(fgets(line, sizeof(line), fp)) {
                        if(strstr(line, "Iteration")) {
                            sscanf(line, "Iteration %*d/%*d [n=%*d]: %lf ± %lf (min: %lf, max: %lf)",
                                   &results[i].mean_return,
                                   &results[i].std_return,
                                   &results[i].min_return,
                                   &results[i].max_return);
                        }
                    }
                    fclose(fp);
                }
                
                free_net(net);
                exit(0);
            } else {  // Parent process
                results[i].pid = pid;
            }
        }
        
        for(int i = 0; i < NUM_PROCESSES; i++) {
            waitpid(results[i].pid, NULL, 0);
        }
        
        qsort(results, NUM_PROCESSES, sizeof(ProcessResult), compare_results);
        
        printf("\nGeneration Results:\n");
        for(int i = 0; i < NUM_PROCESSES; i++) {
            printf("Process %d: %.2f ± %.2f (min: %.2f, max: %.2f)\n",
                   i, results[i].mean_return, results[i].std_return,
                   results[i].min_return, results[i].max_return);
        }
        
        rename(results[0].weights_file, base_weights);
        
        for(int i = 0; i < NUM_PROCESSES; i++) {
            char out_file[64];
            sprintf(out_file, "process_%d.txt", i);
            remove(out_file);
            if(i > 0) remove(results[i].weights_file);
        }
    }
    
    munmap(results, NUM_PROCESSES * sizeof(ProcessResult));
    free_net(base_net);
    return 0;
}