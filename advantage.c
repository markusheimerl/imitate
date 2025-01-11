#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#define D1 64
#define D2 32
#define D3 16
#define M 18

bool load_weights(const char* filename, double *W1, double *b1, double *W2, double *b2, 
                 double *W3, double *b3, double *W4, double *b4) {
    FILE *f = fopen(filename, "rb");
    if (!f) return false;
    size_t items_read = fread(W1, sizeof(double), D1*M, f) + 
                        fread(b1, sizeof(double), D1, f) + 
                        fread(W2, sizeof(double), D2*D1, f) + 
                        fread(b2, sizeof(double), D2, f) + 
                        fread(W3, sizeof(double), D3*D2, f) + 
                        fread(b3, sizeof(double), D3, f) + 
                        fread(W4, sizeof(double), D3, f) + 
                        fread(b4, sizeof(double), 1, f);
    fclose(f);
    return items_read == (D1*M + D1 + D2*D1 + D2 + D3*D2 + D3 + D3 + 1);
}

void forward(double *W1, double *b1, double *W2, double *b2, double *W3, double *b3, 
            double *W4, double *b4, double *input, double *h1, double *h2, double *h3, 
            double *out) {
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

    // Allocate memory for the value network
    double *W1 = malloc(D1*M*sizeof(double)), *b1 = calloc(D1, sizeof(double));
    double *W2 = malloc(D2*D1*sizeof(double)), *b2 = calloc(D2, sizeof(double));
    double *W3 = malloc(D3*D2*sizeof(double)), *b3 = calloc(D3, sizeof(double));
    double *W4 = malloc(D3*sizeof(double)), *b4 = calloc(1, sizeof(double));
    double *h1 = malloc(D1*sizeof(double)), *h2 = malloc(D2*sizeof(double));
    double *h3 = malloc(D3*sizeof(double));

    // Load value network weights
    if (!load_weights(argv[2], W1, b1, W2, b2, W3, b3, W4, b4)) {
        printf("Failed to load value network weights\n");
        return 1;
    }

    // Open input CSV
    FILE *f_in = fopen(argv[1], "r");
    if (!f_in) {
        printf("Failed to open input CSV\n");
        return 1;
    }

    // Create temporary output file
    char temp_filename[1024];
    snprintf(temp_filename, sizeof(temp_filename), "%s.tmp", argv[1]);
    FILE *f_out = fopen(temp_filename, "w");
    if (!f_out) {
        printf("Failed to create temporary file\n");
        return 1;
    }

    char line[4096], header[4096];
    // Read and modify header
    fgets(header, sizeof(header), f_in);
    header[strlen(header)-1] = '\0';  // Remove newline
    fprintf(f_out, "%s,advantage\n", header);

    double state[M], value_pred, out;
    while (fgets(line, sizeof(line), f_in)) {
        char *token = strtok(line, ",");
        // Read state (first 18 values)
        for(int i = 0; i < M; i++) {
            state[i] = atof(token);
            token = strtok(NULL, ",");
        }
        
        // Skip to discounted_return (last value)
        for(int i = 0; i < 10; i++) token = strtok(NULL, ",");
        double discounted_return = atof(token);

        // Get value prediction
        forward(W1, b1, W2, b2, W3, b3, W4, b4, state, h1, h2, h3, &value_pred);

        // Calculate advantage
        double advantage = discounted_return - value_pred;

        // Write original line and advantage
        line[strlen(line)-1] = '\0';  // Remove newline
        fprintf(f_out, "%s,%f\n", line, advantage);
    }

    fclose(f_in);
    fclose(f_out);

    // Replace original file with new file
    remove(argv[1]);
    rename(temp_filename, argv[1]);

    // Free memory
    free(W1); free(b1); free(W2); free(b2);
    free(W3); free(b3); free(W4); free(b4);
    free(h1); free(h2); free(h3);

    return 0;
}