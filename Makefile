CC = clang
CFLAGS = -O3 -march=native -ffast-math -fomit-frame-pointer \
         -fno-signed-zeros -fno-trapping-math -funroll-loops \
         -fvectorize -fslp-vectorize -falign-functions=32 \
         -fstrict-aliasing -fmerge-all-constants \
         -fno-math-errno -freciprocal-math
LDFLAGS = -flto -lm
TARGET = transformer.out

.PHONY: all clean run

all: $(TARGET)

$(TARGET): transformer.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)