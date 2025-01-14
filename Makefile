CC = clang
CFLAGS = -O3 -march=native -ffast-math -Isim -Igrad -Isim/rasterizer
LDFLAGS = -flto -lm

TARGETS = rollout.out reinforce.out
ITERATIONS = 100  # Number of training iterations

.PHONY: clean run

all: $(TARGETS)

rollout.out: rollout.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

reinforce.out: reinforce.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

# Training loop
run: $(TARGETS)
	@echo "Starting training loop..."
	@# First iteration without policy file
	@echo "\nIteration 1/$(ITERATIONS)"
	@./rollout.out
	@POLICY=$$(ls -t *_policy.bin | head -1) && \
	./reinforce.out $$POLICY
	@# Remaining iterations
	@for i in $$(seq 2 $(ITERATIONS)); do \
		echo "\nIteration $$i/$(ITERATIONS)"; \
		POLICY=$$(ls -t *_policy.bin | head -1) && \
		./rollout.out $$POLICY && \
		./reinforce.out $$POLICY || break; \
	done

clean:
	rm -f $(TARGETS) *.csv *.bin