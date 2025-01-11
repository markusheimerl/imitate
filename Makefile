CC = clang
CFLAGS = -O3 -march=native -ffast-math
LDFLAGS = -flto -lm
INCLUDES = -I./sim -I./sim/rasterizer

VALUE_TARGET = value.out
TRAJECTORY_TARGET = trajectory.out

.PHONY: clean run plot

$(VALUE_TARGET): value.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(TRAJECTORY_TARGET): trajectory.c
	$(CC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

render: CFLAGS += -DRENDER
render: trajectory.c
	$(CC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $(TRAJECTORY_TARGET)

log: CFLAGS += -DLOG
log: trajectory.c
	$(CC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $(TRAJECTORY_TARGET)

clean:
	rm -f $(VALUE_TARGET) $(TRAJECTORY_TARGET) *_flight.gif *_trajectory.csv *_weights.bin