CC = clang
CFLAGS = -O3 -march=native -ffast-math -Isim -Igrad -Isim/rasterizer
LDFLAGS = -flto -lm

TARGET = reinforce.out

.PHONY: clean run

all: $(TARGET)

$(TARGET): reinforce.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

visualize.out: visualize.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f *.out *.bin *.gif