CC = clang
CFLAGS = -O3 -march=native -ffast-math -Isim -Igrad -Isim/rasterizer
LDFLAGS = -static -lm -flto

.PHONY: clean run

all: $(TARGETS)

visualize.out: visualize.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

reinforce.out: reinforce.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

run: reinforce.out
	./reinforce.out 10000

clean:
	rm -f *.out *.bin *.gif