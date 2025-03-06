CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -flto
CUDAFLAGS = --cuda-gpu-arch=sm_89 -x cuda -Wno-unknown-cuda-version
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

all: imitate.out visualize.out

imitate.out: imitate.c
	$(CC) $(CFLAGS) $(CUDAFLAGS) $< -lwebp -lwebpmux $(CUDALIBS) $(LDFLAGS) -o $@

visualize.out: visualize.c
	$(CC) $(CFLAGS) $< -lopenblas -lwebp -lwebpmux $(LDFLAGS) -o $@

run: imitate.out
	@time ./imitate.out

viz: visualize.out
	@time ./visualize.out \
		$(shell ls -t *_layer1_model.bin | head -1) \
		$(shell ls -t *_layer2_model.bin | head -1) \
		$(shell ls -t *_layer3_model.bin | head -1) \
		$(shell ls -t *_layer4_model.bin | head -1)

clean:
	rm -f *.out *.bin *.csv *.webp
