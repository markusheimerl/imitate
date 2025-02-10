CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -static -lm -lopenblas -flto

reinforce.out: reinforce.c
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

run: reinforce.out
	@time ./reinforce.out

clean:
	rm -f *.out *_policy.bin *_flight.csv *_flight.webp