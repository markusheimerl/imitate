CC = clang
CFLAGS = -O3 -march=native -ffast-math -fopenmp
LDFLAGS = -flto -lm
INCLUDES = -I./sim -I./sim/rasterizer
TARGET = transformer.out
FLY_TARGET = fly.out

.PHONY: all clean run fly

all: $(TARGET) $(FLY_TARGET)

$(TARGET): transformer.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(FLY_TARGET): fly.c
	$(CC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

run: $(TARGET)
	./$(TARGET)

fly: $(FLY_TARGET)
	./$(FLY_TARGET) weights.bin

clean:
	rm -f $(TARGET) $(FLY_TARGET) training_loss.csv