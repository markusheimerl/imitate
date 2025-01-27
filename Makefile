CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra -Igrad -Iraytracer
LDFLAGS = -static -lm -lwebp -lwebpmux -lpthread -flto

reinforce.out: reinforce.c
	$(CC) $(CFLAGS) reinforce.c $(LDFLAGS) -o reinforce.out

run: reinforce.out
	@time ./reinforce.out 1000

visualize.out: visualize.c
	$(CC) $(CFLAGS) visualize.c $(LDFLAGS) -o visualize.out

clean:
	rm -f *.out *_policy.bin *_flight.webp