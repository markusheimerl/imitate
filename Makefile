CC = clang
CFLAGS = -O3 -march=native -ffast-math
LDFLAGS = -flto -lm
INCLUDES = -I./sim -I./sim/rasterizer

VALUE_TARGET = value.out
TRAJECTORY_RENDER_TARGET = trajectory_render.out
TRAJECTORY_LOG_TARGET = trajectory_log.out
ADVANTAGE_TARGET = advantage.out
POLICY_TARGET = policy.out

ITERATIONS = 10

.PHONY: clean run train

$(POLICY_TARGET): policy.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(VALUE_TARGET): value.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(ADVANTAGE_TARGET): advantage.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

render: trajectory.c
	$(CC) $(CFLAGS) -DRENDER $(INCLUDES) $^ $(LDFLAGS) -o $(TRAJECTORY_RENDER_TARGET)

log: trajectory.c
	$(CC) $(CFLAGS) -DLOG $(INCLUDES) $^ $(LDFLAGS) -o $(TRAJECTORY_LOG_TARGET)

train: log $(VALUE_TARGET) $(ADVANTAGE_TARGET) $(POLICY_TARGET)
	@echo "Starting PPO training loop..."
	@for i in $$(seq 1 $(ITERATIONS)); do \
		echo "\n=== Iteration $$i of $(ITERATIONS) ==="; \
		if [ $$i -eq 1 ]; then \
			./$(TRAJECTORY_LOG_TARGET); \
		else \
			./$(TRAJECTORY_LOG_TARGET) $$POLICY_WEIGHTS; \
		fi; \
		TRAJECTORY_FILE=$$(ls -t *_trajectory.csv | head -n1); \
		echo "Using trajectory file: $$TRAJECTORY_FILE"; \
		if [ $$i -eq 1 ]; then \
			./$(VALUE_TARGET) $$TRAJECTORY_FILE; \
		else \
			./$(VALUE_TARGET) $$TRAJECTORY_FILE $$VALUE_WEIGHTS; \
		fi; \
		VALUE_WEIGHTS=$$(ls -t *_value_weights.bin | head -n1); \
		echo "Using value weights: $$VALUE_WEIGHTS"; \
		./$(ADVANTAGE_TARGET) $$TRAJECTORY_FILE $$VALUE_WEIGHTS; \
		if [ $$i -eq 1 ]; then \
			POLICY_WEIGHTS=$$(ls -t *_policy_weights.bin | head -n1); \
		fi; \
		./$(POLICY_TARGET) $$TRAJECTORY_FILE $$POLICY_WEIGHTS; \
		POLICY_WEIGHTS=$$(ls -t *_policy_weights.bin | head -n1); \
		echo "Using policy weights: $$POLICY_WEIGHTS"; \
	done

run: render train
	@echo "\nGenerating final render..."
	@POLICY_WEIGHTS=$$(ls -t *_policy_weights.bin | head -n1); \
	./$(TRAJECTORY_RENDER_TARGET) $$POLICY_WEIGHTS

clean:
	rm -f $(VALUE_TARGET) $(TRAJECTORY_LOG_TARGET) $(TRAJECTORY_RENDER_TARGET) \
		$(ADVANTAGE_TARGET) $(POLICY_TARGET) *_flight.gif *_trajectory.csv \
		*_policy_weights.bin *_value_weights.bin