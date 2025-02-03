CC = nvcc
CFLAGS = -O3
LDFLAGS = -lm

reinforce.out: %.out: %.c
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

visualize.out: %.out: %.c
	$(CC) $(CFLAGS) -Iraytracer $< $(LDFLAGS) -lwebp -lwebpmux -lpthread -o $@

run: reinforce.out
	@time ./reinforce.out 1000

clean:
	rm -f *.out *_policy.bin *_flight.webp