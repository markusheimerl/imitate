CC = clang
CFLAGS = -O3 -march=native -ffast-math -Isim -Igrad -Isim/rasterizer
LDFLAGS = -flto -lm

TARGETS = reinforce.out orchestrate.out

.PHONY: clean run

all: $(TARGETS)

reinforce.out: reinforce.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

orchestrate.out: orchestrate.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

run: orchestrate.out reinforce.out
	./orchestrate.out

clean:
	rm -f *.out *.bin *.gif *.txt