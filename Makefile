CC = clang
CFLAGS = -O3 -march=native -ffast-math -fopenmp
LDFLAGS = -flto -lm
INCLUDES = -I./sim -I./sim/rasterizer

TRAIN_TARGET = train.out
FLY_TARGET = fly.out

.PHONY: clean run plot

all: $(TRAIN_TARGET) $(FLY_TARGET)

$(TRAIN_TARGET): train.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(FLY_TARGET): fly.c
	$(CC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

run: $(TRAIN_TARGET) $(FLY_TARGET)
	cd sim && make log && ./sim.out 100 && cp *_control_data.csv .. && make clean && cd ..
	./$(TRAIN_TARGET) `ls -t *_control_data.csv | head -1`
	@python -c 'import matplotlib.pyplot as plt, pandas as pd, os; \
	f = sorted([f for f in os.listdir(".") if f.endswith("_loss.csv")])[-1]; \
	ts = f.replace("_loss.csv", ""); \
	df = pd.read_csv(f); \
	plt.figure(figsize=(10, 6)); \
	plt.plot(df["step"], df["loss"], "b", alpha=0.5, label="Raw"); \
	plt.plot(df["step"], df["loss"].rolling(10, min_periods=1, center=True).mean(), "r", lw=2, label="Average"); \
	plt.title("Training Loss"); plt.xlabel("Step"); plt.ylabel("Loss"); \
	plt.yscale("log"); plt.grid(True); plt.legend(); \
	plt.savefig(f"{ts}_loss.png");'
	./$(FLY_TARGET) `ls -t *_weights.bin | head -1`

clean:
	rm -f $(TRAIN_TARGET) $(FLY_TARGET) *_loss.csv *_flight.gif *_loss.png *_control_data.csv *_weights.bin
	cd sim && make clean