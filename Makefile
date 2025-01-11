CC = clang
CFLAGS = -O3 -march=native -ffast-math
LDFLAGS = -flto -lm
INCLUDES = -I./sim -I./sim/rasterizer

VALUE_TARGET = value.out
TRAJECTORY_TARGET = trajectory.out

.PHONY: clean run plot

all: $(VALUE_TARGET) $(TRAJECTORY_TARGET)

$(VALUE_TARGET): value.c
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(TRAJECTORY_TARGET): trajectory.c
	$(CC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

run: $(VALUE_TARGET) $(TRAJECTORY_TARGET)
	cd sim && make log && ./sim.out 100 && cp *_state_data.csv .. && make clean && cd ..
	./$(VALUE_TARGET) `ls -t *_state_data.csv | head -1`
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
	./$(TRAJECTORY_TARGET) `ls -t *_weights.bin | head -1`

clean:
	rm -f $(VALUE_TARGET) $(TRAJECTORY_TARGET) *_loss.csv *_flight.gif *_loss.png *_state_data.csv *_weights.bin *_flight_data.csv
	cd sim && make clean