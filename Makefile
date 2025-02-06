CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
CUDAFLAGS = --cuda-gpu-arch=sm_86 -x cuda -Wno-unknown-cuda-version
LDFLAGS = -lm -lpthread -flto
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -ldl -lrt

reinforce.out: %.out: %.c
	$(CC) $(CFLAGS) $(CUDAFLAGS) $< $(LDFLAGS) $(CUDALIBS) -o $@

visualize.out: %.out: %.c
	$(CC) $(CFLAGS) -Iraytracer $< $(LDFLAGS) -lwebp -lwebpmux -o $@

run: reinforce.out
	@time ./reinforce.out 1000

clean:
	rm -f *.out *_policy.bin *_flight.webp