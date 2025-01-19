#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <sys/time.h>
#include "grad/grad.h"

#define NUM_PROCESSES 8
#define ELITE_COUNT 2

typedef struct {
    double mean_return;
    double std_return;
    pid_t pid;
    char weights_file[64];
} ProcessResult;

void interpolate_weights(Net* net, Net* elite, double alpha) {
    for(int i = 0; i < net->n; i++) {
        int in = net->sz[i], out = net->sz[i+1];
        for(int j = 0; j < in*out; j++) {
            net->w[i][j] = alpha * net->w[i][j] + (1.0 - alpha) * elite->w[i][j];
        }
        for(int j = 0; j < out; j++) {
            net->b[i][j] = alpha * net->b[i][j] + (1.0 - alpha) * elite->b[i][j];
        }
    }
}

int compare_results(const void* a, const void* b) {
    double diff = ((ProcessResult*)b)->mean_return - ((ProcessResult*)a)->mean_return;
    return diff > 0 ? 1 : (diff < 0 ? -1 : 0);
}

int main(int argc, char** argv) {
    if(argc != 2 && argc != 3) {
        printf("Usage: %s <num_generations> [initial_weights.bin]\n", argv[0]);
        return 1;
    }
    
    srand(time(NULL) ^ getpid());
    
    ProcessResult* results = mmap(NULL, NUM_PROCESSES * sizeof(ProcessResult),
                                PROT_READ | PROT_WRITE,
                                MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    
    for(int i = 0; i < NUM_PROCESSES; i++) {
        sprintf(results[i].weights_file, "weights_%d.bin", i);
    }
    
    char final_weights[64];
    strftime(final_weights, sizeof(final_weights), "%Y%m%d_%H%M%S_policy.bin", 
             localtime(&(time_t){time(NULL)}));
    
    if(argc == 3) {
        // If initial weights provided, copy them to our new file
        char command[256];
        sprintf(command, "cp %s %s", argv[2], final_weights);
        if(system(command) != 0) {
            fprintf(stderr, "Failed to copy initial weights from %s\n", argv[2]);
            return 1;
        }
        printf("Starting from weights in %s\n", argv[2]);
        printf("Will save results to %s\n", final_weights);
    } else {
        // Create fresh initial weights
        Net* initial_net = init_net(5, (int[]){12, 64, 64, 64, 8}, adamw);
        if(!initial_net) return 1;
        save_weights(final_weights, initial_net);
        free_net(initial_net);
        printf("Starting fresh, will save results to %s\n", final_weights);
    }
    
    int generations = atoi(argv[1]);
    for(int gen = 0; gen < generations; gen++) {
        printf("\nGeneration %d/%d\n", gen + 1, generations);
        
        int pipes[NUM_PROCESSES][2];
        
        for(int i = 0; i < NUM_PROCESSES; i++) {
            if(pipe(pipes[i]) == -1) {
                perror("pipe creation failed");
                exit(1);
            }
            
            if(fork() == 0) {
                close(pipes[i][0]);
                dup2(pipes[i][1], STDOUT_FILENO);
                close(pipes[i][1]);
                
                Net* net;
                if(gen == 0) {
                    // First generation: everyone starts from initial weights
                    net = load_weights(final_weights, adamw);
                    if(!net) {
                        fprintf(stderr, "Failed to load weights from %s\n", final_weights);
                        exit(1);
                    }
                } else {
                    // Everyone loads their previous weights
                    net = load_weights(results[i].weights_file, adamw);
                    if(i >= ELITE_COUNT) {
                        // Non-elites interpolate with the best
                        Net* elite = load_weights(final_weights, adamw);
                        interpolate_weights(net, elite, 0.5);  // Keep 50% of own weights
                        free_net(elite);
                    }
                }
                
                save_weights(results[i].weights_file, net);
                free_net(net);
                
                execl("./reinforce.out", "reinforce.out", results[i].weights_file, NULL);
                exit(1);
            }
            
            close(pipes[i][1]);
        }
        
        char buf[256], last_iteration[256] = {0};
        for(int i = 0; i < NUM_PROCESSES; i++) {
            FILE* fp = fdopen(pipes[i][0], "r");
            while(fgets(buf, sizeof(buf), fp)) {
                if(strstr(buf, "Iteration")) {
                    strcpy(last_iteration, buf);
                }
            }
            if(last_iteration[0]) {
                sscanf(last_iteration, "Iteration %*d/%*d [n=%*d]: %lf ± %lf",
                       &results[i].mean_return, &results[i].std_return);
            }
            fclose(fp);
            wait(NULL);
        }
        
        qsort(results, NUM_PROCESSES, sizeof(ProcessResult), compare_results);
        
        printf("\nGeneration Results:\n");
        for(int i = 0; i < NUM_PROCESSES; i++) {
            printf("Agent %d: %.2f ± %.2f\n", i, 
                   results[i].mean_return, results[i].std_return);
        }
        
        char command[256];
        sprintf(command, "cp %s %s", results[0].weights_file, final_weights);
        system(command);
    }
    
    for(int i = 0; i < NUM_PROCESSES; i++) {
        remove(results[i].weights_file);
    }
    
    munmap(results, NUM_PROCESSES * sizeof(ProcessResult));
    return 0;
}