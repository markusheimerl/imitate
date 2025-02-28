CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -flto
CUDAFLAGS = --cuda-gpu-arch=sm_89 \
    -x cuda \
    -fcuda-flush-denormals-to-zero \
    -fcuda-approx-transcendentals \
    -Wno-unknown-cuda-version

CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

all: imitate.out visualize.out

imitate.out: imitate.c
	$(CC) $(CFLAGS) $(CUDAFLAGS) $< $(CUDALIBS) $(LDFLAGS) -o $@

visualize.out: visualize.c
	$(CC) $(CFLAGS) $< -lopenblas -lwebp -lwebpmux $(LDFLAGS) -o $@

run: imitate.out
	@time ./imitate.out

viz: visualize.out
	@./visualize.out $(shell ls -t *_policy.bin | head -1) $(shell ls -t *_estimator.bin | head -1)

clean:
	rm -f *.out *.bin *.csv *_flight.webp
