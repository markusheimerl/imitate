CC = nvcc
CFLAGS = -O3 -arch=sm_60 -Igrad -Iraytracer
LDFLAGS = -lcudart -lcurand -lm -lwebp -lwebpmux -lpthread

reinforce.out visualize.out: %.out: %.cu
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

run: reinforce.out
	@time ./reinforce.out 1000

clean:
	rm -f *.out *_policy.bin *_flight.webp