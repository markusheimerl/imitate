CC = clang
CFLAGS = -O3 -march=native -ffast-math
LDFLAGS = -flto -lm
INCLUDES = -I./sim -I./sim/rasterizer

VALUE_TARGET = value.out
TRAJECTORY_TARGET = trajectory.out
ADVANTAGE_TARGET = advantage.out

.PHONY: clean run plot

$(VALUE_TARGET): value.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(ADVANTAGE_TARGET): advantage.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(TRAJECTORY_TARGET): trajectory.c
	$(CC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

render: CFLAGS += -DRENDER
render: trajectory.c
	$(CC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $(TRAJECTORY_TARGET)

log: CFLAGS += -DLOG
log: trajectory.c
	$(CC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $(TRAJECTORY_TARGET)

run: $(VALUE_TARGET) log $(ADVANTAGE_TARGET)
	./$(TRAJECTORY_TARGET)
	TRAJECTORY_FILE=$$(ls -t *_trajectory.csv | head -n1); \
	./$(VALUE_TARGET) $$TRAJECTORY_FILE; \
	VALUE_WEIGHTS=$$(ls -t *_value_weights.bin | head -n1); \
	./$(ADVANTAGE_TARGET) $$TRAJECTORY_FILE $$VALUE_WEIGHTS

clean:
	rm -f $(VALUE_TARGET) $(TRAJECTORY_TARGET) $(ADVANTAGE_TARGET) *_flight.gif *_trajectory.csv *_policy_weights.bin *_value_weights.bin