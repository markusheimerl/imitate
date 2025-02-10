CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -static -lm -flto

reinforce.out: %.out: %.c
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

run: reinforce.out
	@time ./reinforce.out

clean:
	rm -f *.out *_policy.bin *_flight.webp