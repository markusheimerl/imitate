CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra -Iraytracer
LDFLAGS = -static -lm -lwebp -lwebpmux -lpthread

reinforce.out visualize.out: %.out: %.c
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

run: reinforce.out
	@time ./reinforce.out 1000

clean:
	rm -f *.out *_policy.bin *_flight.webp