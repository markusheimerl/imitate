#ifndef UTIL_H
#define UTIL_H

void save_weights(const char* filename, double** params, int* sizes, int num) {
    FILE *f = fopen(filename, "wb");
    for(int i = 0; i < num; i++) {
        fwrite(params[i], sizeof(double), sizes[i], f);
    }
    fclose(f);
}

void load_weights(const char* filename, double** params, int* sizes, int num) {
    FILE *f = fopen(filename, "rb");
    for(int i = 0; i < num; i++) {
        fread(params[i], sizeof(double), sizes[i], f);
    }
    fclose(f);
}

#endif // UTIL_H