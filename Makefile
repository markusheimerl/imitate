CC = clang
CFLAGS = -O3 -march=native -ffast-math -Isim -Igrad -Isim/rasterizer
LDFLAGS = -static -lm -flto

TARGETS = reinforce.out orchestrate.out visualize.out

.PHONY: clean run

all: $(TARGETS)

visualize.out: visualize.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

reinforce.out: reinforce.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

orchestrate.out: orchestrate.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

run: orchestrate.out reinforce.out
	./orchestrate.out

clean:
	rm -f *.out *.bin *.gif *.txt