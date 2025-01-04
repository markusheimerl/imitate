import matplotlib.pyplot as plt
import pandas as pd

# Read the loss data
df = pd.read_csv('training_loss.csv', names=['step', 'loss'])

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(df['step'], df['loss'])
plt.title('Training Loss Over Time')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.yscale('log')  # Use log scale for better visualization
plt.grid(True)

# Save the plot
plt.savefig('training_loss.png')
plt.close()