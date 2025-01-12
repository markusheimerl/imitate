#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#define D1 64
#define D2 32
#define D3 16
#define M_IN 18  // pos[3], vel[3], ang_vel[3], R[9]

bool load_weights(const char* filename, double *W1, double *b1, double *W2, double *b2, 
                 double *W3, double *b3, double *W4, double *b4) {
    FILE *f = fopen(filename, "rb");
    if (!f) return false;
    size_t items_read = fread(W1, sizeof(double), D1*M_IN, f) +
                       fread(b1, sizeof(double), D1, f) +
                       fread(W2, sizeof(double), D2*D1, f) +
                       fread(b2, sizeof(double), D2, f) +
                       fread(W3, sizeof(double), D3*D2, f) +
                       fread(b3, sizeof(double), D3, f) +
                       fread(W4, sizeof(double), D3, f) +
                       fread(b4, sizeof(double), 1, f);
    fclose(f);
    return items_read == (D1*M_IN + D1 + D2*D1 + D2 + D3*D2 + D3 + D3 + 1);
}

void forward(double *W1, double *b1, double *W2, double *b2, double *W3, double *b3,
            double *W4, double *b4, double *input, double *h1, double *h2, double *h3, double *output) {
    for(int i = 0; i < D1; i++) {
        double sum = b1[i];
        for(int j = 0; j < M_IN; j++) sum += W1[i*M_IN + j] * input[j];
        h1[i] = sum > 0 ? sum : sum * 0.1;
    }

    for(int i = 0; i < D2; i++) {
        double sum = b2[i];
        for(int j = 0; j < D1; j++) sum += W2[i*D1 + j] * h1[j];
        h2[i] = sum > 0 ? sum : sum * 0.1;
    }

    for(int i = 0; i < D3; i++) {
        double sum = b3[i];
        for(int j = 0; j < D2; j++) sum += W3[i*D2 + j] * h2[j];
        h3[i] = sum > 0 ? sum : sum * 0.1;
    }

    *output = b4[0];
    for(int i = 0; i < D3; i++) *output += W4[i] * h3[i];
}

int main(int argc, char **argv) {
    if(argc != 3) {
        printf("Usage: %s <trajectory_csv> <value_weights>\n", argv[0]);
        return 1;
    }

    // Initialize network parameters
    double *W1 = malloc(D1*M_IN*sizeof(double)), *b1 = malloc(D1*sizeof(double));
    double *W2 = malloc(D2*D1*sizeof(double)), *b2 = malloc(D2*sizeof(double));
    double *W3 = malloc(D3*D2*sizeof(double)), *b3 = malloc(D3*sizeof(double));
    double *W4 = malloc(D3*sizeof(double)), *b4 = malloc(sizeof(double));
    double *h1 = malloc(D1*sizeof(double));
    double *h2 = malloc(D2*sizeof(double));
    double *h3 = malloc(D3*sizeof(double));

    if (!load_weights(argv[2], W1, b1, W2, b2, W3, b3, W4, b4)) {
        printf("Failed to load weights\n");
        return 1;
    }

    FILE *f = fopen(argv[1], "r");
    if (!f) { printf("Failed to open trajectory file\n"); return 1; }

    // Count rows (excluding header)
    char line[1024];
    int rows = -1;
    while (fgets(line, sizeof(line), f)) rows++;
    rewind(f);
    fgets(line, sizeof(line), f);  // Skip header

    // Process data
    FILE *temp = fopen("temp.csv", "w");
    if (!temp) { printf("Failed to create temporary file\n"); return 1; }

    // Write header (copy original header and add advantage column)
    line[strcspn(line, "\n")] = 0;
    fprintf(temp, "%s,advantage\n", line);

    printf("Processing %d rows...\n", rows);
    int processed = 0;

    while (fgets(line, sizeof(line), f)) {
        char *original = strdup(line);
        original[strcspn(original, "\n")] = 0;
        
        double state[M_IN];
        char *token = strtok(line, ",");
        token = strtok(NULL, ",");  // Skip rollout number
        
        // Read state
        for(int i = 0; i < M_IN; i++) {
            if (!token) { printf("Error: malformed input\n"); return 1; }
            state[i] = atof(token);
            token = strtok(NULL, ",");
        }
        
        // Skip to discounted return (acc_s, gyro_s, means, vars, omega, reward)
        for(int i = 0; i < 15; i++) {
            if (!token) { printf("Error: malformed input\n"); return 1; }
            token = strtok(NULL, ",");
        }
        
        // Skip immediate reward
        token = strtok(NULL, ",");
        
        double actual_return = atof(token);
        double value_pred;
        
        forward(W1, b1, W2, b2, W3, b3, W4, b4, state, h1, h2, h3, &value_pred);
        double advantage = actual_return - value_pred;
        
        fprintf(temp, "%s,%f\n", original, advantage);
        free(original);
        
        if (++processed % 1000 == 0) {
            printf("\rProcessed %d/%d rows", processed, rows);
            fflush(stdout);
        }
    }
    printf("\nCompleted processing %d rows\n", rows);

    fclose(f);
    fclose(temp);
    remove(argv[1]);
    rename("temp.csv", argv[1]);

    // Cleanup
    free(W1); free(b1); free(W2); free(b2);
    free(W3); free(b3); free(W4); free(b4);
    free(h1); free(h2); free(h3);

    return 0;
}