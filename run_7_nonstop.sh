#!/bin/bash

while true; do
    # Check if the script is running
    if ! pgrep -f "python 7_HEP.py" > /dev/null; then
        echo "Script is not running. Starting the script..."
        python 7_HEP.py &
    else
        echo "Script is already running."
    fi
    # Wait for a specified interval before checking again
    sleep 600
done