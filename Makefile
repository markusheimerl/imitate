CC = clang
CFLAGS = -O3 -march=native -ffast-math -Isim -Igrad -Isim/rasterizer
LDFLAGS = -flto -lm

TARGET = rollout.out
SRC = rollout.c

.PHONY: clean run

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.csv *.bin