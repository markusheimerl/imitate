CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -static -lm -lpthread -flto

reinforce.out: %.out: %.c
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

visualize.out: %.out: %.c
	$(CC) $(CFLAGS) -Iraytracer $< $(LDFLAGS) -lwebp -lwebpmux -o $@

run: reinforce.out
	@time ./reinforce.out 1000

clean:
	rm -f *.out *_policy.bin *_flight.webp