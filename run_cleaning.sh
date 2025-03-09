#!/bin/bash

while true; do
    # Check if the script is running
    if ! pgrep -f "python 1_edf_cleaning.py" > /dev/null; then
        echo "Script is not running. Starting the script..."
        python 1_edf_cleaning.py &
    else
        echo "Script is already running."
    fi
    # Wait for a specified interval before checking again
    sleep 60
done