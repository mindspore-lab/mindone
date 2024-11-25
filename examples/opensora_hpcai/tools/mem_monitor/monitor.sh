#!/bin/bash

# Check if a PID is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <PID> <output_log_file>"
    exit 1
fi

PID=$1
LOG_FILE="memory_usage.log"

# Check if the process with the given PID exists
if ! ps -p $PID > /dev/null; then
    echo "Process with PID $PID does not exist."
    exit 1
fi

# Initialize the log file
echo "Timestamp,Memory_Usage_Percentage" > "$LOG_FILE"

# Monitor memory usage
echo "Monitoring memory usage for PID: $PID. Logging to $LOG_FILE"
echo "Press [CTRL+C] to stop."

# Loop to continuously monitor memory usage
while true; do
    # Get the total memory in KB
    TOTAL_MEM=$(grep MemTotal /proc/meminfo | awk '{print $2}')

    # Get the RSS memory of the process in KB
    MEMORY_INFO=$(pmap -x $PID | tail -n 1)
    RSS_MEMORY=$(echo $MEMORY_INFO | awk '{print $3}') # Get the total RSS memory

    # Calculate memory usage percentage
    if [ -n "$RSS_MEMORY" ]; then
        MEMORY_USAGE_PERCENTAGE=$(echo "scale=2; ($RSS_MEMORY / $TOTAL_MEM) * 100" | bc)
        TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

        # Log the timestamp and memory usage percentage
        echo "$TIMESTAMP,$MEMORY_USAGE_PERCENTAGE" >> "$LOG_FILE"

        # Print the memory usage percentage to the console
        echo "[$TIMESTAMP] Memory Usage: $MEMORY_USAGE_PERCENTAGE%"
    else
        echo "Unable to retrieve memory usage for PID $PID."
    fi

    # Sleep for a specified interval (e.g., 1 second)
    sleep 10
done
