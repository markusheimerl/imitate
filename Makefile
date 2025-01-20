CC = clang
CFLAGS = -O3 -march=native -ffast-math -Isim -Igrad -Isim/rasterizer
LDFLAGS = -static -lm -flto

.PHONY: clean run

all: $(TARGETS)

visualize.out: visualize.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

reinforce.out: reinforce.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

orchestrate.out: orchestrate.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

joint.out: joint.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

sep.out: sep.c
	$(CC) $(CFLAGS) -fopenmp $^ -lm -flto -o $@

run: orchestrate.out reinforce.out
	./orchestrate.out 10

clean:
	rm -f *.out *.bin *.gif