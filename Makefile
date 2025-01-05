CC = clang
CFLAGS = -O3 -march=native -ffast-math -fopenmp
LDFLAGS = -flto -lm
INCLUDES = -I./sim -I./sim/rasterizer

TRAIN_TARGET = train.out
FLY_TARGET = fly.out

.PHONY: clean run

all: $(TRAIN_TARGET) $(FLY_TARGET)

$(TRAIN_TARGET): train.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(FLY_TARGET): fly.c
	$(CC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

run:
	@# First, clean up any old data
	rm -f *_control_data.csv
	
	@# Build and run simulation
	cd sim && \
		make clean && \
		make log && \
		./sim.out 100 && \
		cp *_control_data.csv .. && \
		rm -f *_control_data.csv && \
		cd ..
	
	@# Train on the most recent data file
	./$(TRAIN_TARGET) `ls -t *_control_data.csv | head -1`

clean:
	rm -f $(TRAIN_TARGET) $(FLY_TARGET)
	rm -f *_loss.csv *_flight.gif
	rm -f *_control_data.csv
	cd sim && make clean