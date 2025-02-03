CC = nvcc
CFLAGS = -O3
LDFLAGS = -lm

reinforce.out: %.out: %.cu
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

visualize.out: %.out: %.cu
	$(CC) $(CFLAGS) -Iraytracer $< $(LDFLAGS) -lwebp -lwebpmux -lpthread -o $@

run: reinforce.out
	@time ./reinforce.out 128000

clean:
	rm -f *.out *_policy.bin *_flight.webp