CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -flto
CUDAFLAGS = --cuda-gpu-arch=sm_89 -x cuda -Wno-unknown-cuda-version
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

all: imitate.out visualize.out data.out world.out

imitate.out: imitate.c
	$(CC) $(CFLAGS) $(CUDAFLAGS) $< $(CUDALIBS) $(LDFLAGS) -o $@

visualize.out: visualize.c
	$(CC) $(CFLAGS) $< -lopenblas -lwebp -lwebpmux $(LDFLAGS) -o $@

data.out: data.c
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

world.out: world.c
	$(CC) $(CFLAGS) $(CUDAFLAGS) $< $(CUDALIBS) $(LDFLAGS) -o $@

data: data.out
	@./data.out 1000

run: imitate.out
	@time ./imitate.out 1000 $(shell ls -t *_data.csv | head -1)

viz: visualize.out
	@time ./visualize.out $(shell ls -t *_model_layer1.bin | head -1) $(shell ls -t *_model_layer2.bin | head -1)

world: world.out
	@time ./world.out $(shell ls -t *_dynamics.csv | head -1)

clean:
	rm -f *.out *.bin *.csv *.webp
