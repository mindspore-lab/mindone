import sys

import matplotlib.pyplot as plt
import pandas as pd

# Read the log file
data = pd.read_csv("memory_usage.log", parse_dates=["Timestamp"])

# Plotting the memory usage
plt.figure(figsize=(10, 5))
plt.plot(data["Timestamp"], data["Memory_Usage_Percentage"], label="Memory Usage (%)", color="blue")
plt.title("Memory Usage Percentage Over Time")
plt.xlabel("Time")
plt.ylabel("Memory Usage (%)")
plt.xticks(rotation=45)
plt.ylim(0, 100)  # Set y-axis limits from 0 to 100%
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("memory_usage_plot.png")
plt.show()
