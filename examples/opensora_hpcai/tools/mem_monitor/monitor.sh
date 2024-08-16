#!/bin/bash

# File to store memory usage data
LOG_FILE="memory_usage.log"

# Clear the log file at the start
echo "Timestamp,Memory_Usage(%)" > $LOG_FILE

# Monitor memory usage every second for a specified duration
DURATION=60  # Total duration in seconds
INTERVAL=1   # Interval in seconds

for ((i=0; i<DURATION; i+=INTERVAL)); do
    # Get memory usage percentage
    memory_usage=$(free | grep Mem | awk '{print $3/$2 * 100.0}')
    # Log the timestamp and memory usage
    echo "$(date +%Y-%m-%d\ %H:%M:%S),$memory_usage" >> $LOG_FILE
    sleep $INTERVAL
done
