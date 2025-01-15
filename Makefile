CC = clang
CFLAGS = -O3 -march=native -ffast-math -Isim -Igrad -Isim/rasterizer
LDFLAGS = -flto -lm

TARGET = reinforce.out
ITERATIONS = 2000

.PHONY: clean run

all: $(TARGET)

$(TARGET): reinforce.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

run: $(TARGET)
	@echo "Starting training loop..."
	@# First iteration without policy file
	@echo "\nIteration 1/$(ITERATIONS)"
	@./$(TARGET)
	@# Remaining iterations
	@for i in $$(seq 2 $(ITERATIONS)); do \
		echo "\nIteration $$i/$(ITERATIONS)"; \
		POLICY=$$(ls -t *_policy.bin | head -1) && \
		./$(TARGET) $$POLICY || break; \
	done

clean:
	rm -f $(TARGET) *.bin