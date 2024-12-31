CC = gcc
CFLAGS = -Wall -Wextra -I./grad
LDFLAGS = -lm
TARGET = transformer.out

.PHONY: all clean run

all: $(TARGET)

$(TARGET): transformer.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)