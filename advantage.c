#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define D1 64
#define D2 32
#define D3 16
#define M 18

void forward(const double *W1, const double *b1, const double *W2, const double *b2, 
            const double *W3, const double *b3, const double *W4, const double *b4, 
            const double *input, double *h1, double *h2, double *h3, double *out) {
    for(int i = 0; i < D1; i++) {
        double sum = b1[i];
        for(int j = 0; j < M; j++) sum += W1[i*M + j] * input[j];
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
    *out = b4[0];
    for(int i = 0; i < D3; i++) *out += W4[i] * h3[i];
}

int main(int argc, char **argv) {
    if(argc != 3) {
        printf("Usage: %s <trajectory_csv> <value_weights>\n", argv[0]);
        return 1;
    }

    // Allocate memory
    double *W1 = malloc(D1*M*sizeof(double));
    double *b1 = malloc(D1*sizeof(double));
    double *W2 = malloc(D2*D1*sizeof(double));
    double *b2 = malloc(D2*sizeof(double));
    double *W3 = malloc(D3*D2*sizeof(double));
    double *b3 = malloc(D3*sizeof(double));
    double *W4 = malloc(D3*sizeof(double));
    double *b4 = malloc(sizeof(double));
    double *h1 = malloc(D1*sizeof(double));
    double *h2 = malloc(D2*sizeof(double));
    double *h3 = malloc(D3*sizeof(double));

    // Load weights
    FILE *f = fopen(argv[2], "rb");
    if (!f) { printf("Failed to load weights\n"); return 1; }
    fread(W1, sizeof(double), D1*M, f);
    fread(b1, sizeof(double), D1, f);
    fread(W2, sizeof(double), D2*D1, f);
    fread(b2, sizeof(double), D2, f);
    fread(W3, sizeof(double), D3*D2, f);
    fread(b3, sizeof(double), D3, f);
    fread(W4, sizeof(double), D3, f);
    fread(b4, sizeof(double), 1, f);
    fclose(f);

    // Process CSV
    FILE *f_in = fopen(argv[1], "r");
    FILE *f_out = fopen("temp.csv", "w");
    if (!f_in || !f_out) { printf("Failed to open CSV files\n"); return 1; }

    char line[4096];
    // Copy header and add advantage column
    if (fgets(line, sizeof(line), f_in)) {
        line[strcspn(line, "\n")] = 0;
        fprintf(f_out, "%s,advantage\n", line);
    }

    int count = 0;
    printf("Processing trajectory...\n");
    
    while (fgets(line, sizeof(line), f_in)) {
        char *original = strdup(line);
        original[strcspn(original, "\n")] = 0;

        double state[M], value_pred;
        char *token = strtok(line, ",");
        
        // Read state (first M values)
        for(int i = 0; i < M; i++) {
            if (!token) { printf("Error: not enough columns\n"); return 1; }
            state[i] = atof(token);
            token = strtok(NULL, ",");
        }
        
        // Skip to discounted_return (29th column)
        for(int i = 0; i < 11; i++) {
            if (!token) { printf("Error: not enough columns\n"); return 1; }
            token = strtok(NULL, ",");
        }
        
        if (!token) { printf("Error: not enough columns\n"); return 1; }
        double actual_return = atof(token);
        
        forward(W1, b1, W2, b2, W3, b3, W4, b4, state, h1, h2, h3, &value_pred);
        double advantage = actual_return - value_pred;
        
        if (count == 0) {
            printf("First state values:\n");
            printf("Actual return: %f\n", actual_return);
            printf("Predicted value: %f\n", value_pred);
            printf("Advantage: %f\n", advantage);
        }
        
        fprintf(f_out, "%s,%f\n", original, advantage);
        free(original);
        count++;
    }

    printf("Completed processing %d rows\n", count);

    fclose(f_in);
    fclose(f_out);
    remove(argv[1]);
    rename("temp.csv", argv[1]);

    // Free memory
    free(W1); free(b1); free(W2); free(b2);
    free(W3); free(b3); free(W4); free(b4);
    free(h1); free(h2); free(h3);

    return 0;
}