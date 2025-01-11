CC = clang
CFLAGS = -O3 -march=native -ffast-math
LDFLAGS = -flto -lm
INCLUDES = -I./sim -I./sim/rasterizer

VALUE_TARGET = value.out
TRAJECTORY_TARGET = trajectory.out

.PHONY: clean run plot

all: $(VALUE_TARGET) $(TRAJECTORY_TARGET)

$(VALUE_TARGET): value.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(TRAJECTORY_TARGET): trajectory.c
	$(CC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

run: $(VALUE_TARGET) $(TRAJECTORY_TARGET)
	cd sim && make log && ./sim.out 100 && cp *_state_data.csv .. && make clean && cd ..
	./$(VALUE_TARGET) `ls -t *_state_data.csv | head -1`
	./$(TRAJECTORY_TARGET) `ls -t *_weights.bin | head -1`

clean:
	rm -f $(VALUE_TARGET) $(TRAJECTORY_TARGET) *_loss.csv *_flight.gif *_loss.png *_state_data.csv *_weights.bin *_flight_data.csv
	cd sim && make clean