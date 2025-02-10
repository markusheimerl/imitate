CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -lopenblas -flto
CUDAFLAGS = --cuda-gpu-arch=sm_86 \
    -x cuda \
    -fcuda-flush-denormals-to-zero \
    -fcuda-approx-transcendentals \
    -Wno-unknown-cuda-version

CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

all: imitate.out visualize.out

imitate.out: imitate.c
	$(CC) $(CFLAGS) $(CUDAFLAGS) $< $(CUDALIBS) $(LDFLAGS) -o $@

visualize.out: visualize.c
	$(CC) $(CFLAGS) $< $(LDFLAGS) -lwebp -lwebpmux -lpthread -o $@

run: imitate.out
	@time ./imitate.out

viz: visualize.out
	@./visualize.out $(shell ls -t *_policy.bin | head -1)

clean:
	rm -f *.out *_policy.bin *_flight.csv *_flight.webp