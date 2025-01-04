CC = clang
CFLAGS = -O3 -march=native -ffast-math -funroll-loops -fvectorize -fslp-vectorize -fstrict-aliasing -fstrict-return -Wall -Wextra
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