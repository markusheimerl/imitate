CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -static -lm -lopenblas -flto

all: reinforce.out visualize.out

reinforce.out: reinforce.c
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

visualize.out: visualize.c
	$(CC) $(CFLAGS) $< $(LDFLAGS) -lwebp -lwebpmux -lpthread -o $@

run: reinforce.out
	@time ./reinforce.out

viz: visualize.out
	@./visualize.out $(shell ls -t *_policy.bin | head -1)

clean:
	rm -f *.out *_policy.bin *_flight.csv *_flight.webp