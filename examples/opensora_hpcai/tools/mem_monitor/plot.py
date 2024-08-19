import pandas as pd
import matplotlib.pyplot as plt

# Load the memory usage data
data = pd.read_csv('memory_usage.log', parse_dates=['Timestamp'])

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(data['Timestamp'], data['Memory_Usage(%)'])
plt.title('Memory Usage Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Memory Usage (%)')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()

# Save the plot
plt.savefig('memory_usage_plot.png')
plt.show()
