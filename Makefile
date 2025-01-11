CC = clang
CFLAGS = -O3 -march=native -ffast-math
LDFLAGS = -flto -lm
INCLUDES = -I./sim -I./sim/rasterizer

VALUE_TARGET = value.out
TRAJECTORY_TARGET = trajectory.out
ADVANTAGE_TARGET = advantage.out
POLICY_TARGET = policy.out

.PHONY: clean run

$(POLICY_TARGET): policy.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

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

run: $(VALUE_TARGET) log $(ADVANTAGE_TARGET) $(POLICY_TARGET)
	./$(TRAJECTORY_TARGET)
	TRAJECTORY_FILE=$$(ls -t *_trajectory.csv | head -n1); \
	./$(VALUE_TARGET) $$TRAJECTORY_FILE; \
	VALUE_WEIGHTS=$$(ls -t *_value_weights.bin | head -n1); \
	./$(ADVANTAGE_TARGET) $$TRAJECTORY_FILE $$VALUE_WEIGHTS; \
	POLICY_WEIGHTS=$$(ls -t *_policy_weights.bin | head -n1); \
	./$(POLICY_TARGET) $$TRAJECTORY_FILE $$POLICY_WEIGHTS

clean:
	rm -f $(VALUE_TARGET) $(TRAJECTORY_TARGET) $(ADVANTAGE_TARGET) $(POLICY_TARGET) *_flight.gif *_trajectory.csv *_policy_weights.bin *_value_weights.bin