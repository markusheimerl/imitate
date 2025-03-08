CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -flto
CUDAFLAGS = --cuda-gpu-arch=sm_89 -x cuda -Wno-unknown-cuda-version
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

all: data.out imitate.out visualize.out

data.out: data.c
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

imitate.out: imitate.c
	$(CC) $(CFLAGS) $(CUDAFLAGS) $< $(CUDALIBS) $(LDFLAGS) -o $@

visualize.out: visualize.c
	$(CC) $(CFLAGS) $(CUDAFLAGS) $< $(CUDALIBS) -lwebp -lwebpmux $(LDFLAGS) -o $@

data: data.out
	@time ./data.out 10000

run: imitate.out
	@time ./imitate.out $(shell ls -t *_data.csv | head -1) 10000

cont: imitate.out
	@time ./imitate.out $(shell ls -t *_data.csv | head -1) 10000 \
		$(shell ls -t *_layer1_model.bin | head -1) \
		$(shell ls -t *_layer2_model.bin | head -1) \
		$(shell ls -t *_layer3_model.bin | head -1) \
		$(shell ls -t *_layer4_model.bin | head -1)

viz: visualize.out
	@time ./visualize.out $(shell ls -t *_layer1_model.bin | head -1) \
		$(shell ls -t *_layer2_model.bin | head -1) \
		$(shell ls -t *_layer3_model.bin | head -1) \
		$(shell ls -t *_layer4_model.bin | head -1)

clean:
	rm -f *.out *.bin *.webp
