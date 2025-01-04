CC = clang
CFLAGS = -O3 -march=native -ffast-math -fomit-frame-pointer -fno-signed-zeros -fno-trapping-math -Wall -Wextra
LDFLAGS = -flto -lm
TARGET = transformer.out

.PHONY: all clean run

all: $(TARGET)

$(TARGET): transformer.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)