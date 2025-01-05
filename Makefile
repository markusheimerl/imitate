CC = clang
CFLAGS = -O3 -march=native -ffast-math -fopenmp
LDFLAGS = -flto -lm
INCLUDES = -I./sim -I./sim/rasterizer
TRAIN_TARGET = train.out
FLY_TARGET = fly.out

.PHONY: all clean run fly

all: $(TRAIN_TARGET) $(FLY_TARGET)

$(TRAIN_TARGET): train.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(FLY_TARGET): fly.c
	$(CC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

run: $(TRAIN_TARGET)
	./$(TRAIN_TARGET)

clean:
	rm -f $(TRAIN_TARGET) $(FLY_TARGET) training_loss.csv transformer_flight.gif