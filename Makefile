CC = clang
NVCC = nvcc
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
CUDAFLAGS = -O3 -arch=sm_60
LDFLAGS = -lm -lpthread

reinforce.out: %.out: %.cu
	$(NVCC) $(CUDAFLAGS) $< $(LDFLAGS) -lcudart -o $@

visualize.out: %.out: %.c
	$(CC) $(CFLAGS) -Iraytracer $< -static $(LDFLAGS) -lwebp -lwebpmux -flto -o $@

run: reinforce.out
	@time ./reinforce.out 100

clean:
	rm -f *.out *_policy.bin *_flight.webp