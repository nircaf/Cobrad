#!/bin/bash

# Check for a reset argument to clear existing runs
if [[ "$1" == "--reset" || "$1" == "-r" || "$1" == "reset" ]]; then
    echo "Reset request detected. Resetting current run..."
    echo "Killing existing tmux sessions 0-4..."
    for i in {0..4}; do
        tmux kill-session -t $i 2>/dev/null
    done
    echo "Removing temporary 'parquets_HEP' directory and logs to restart processing..."
    rm -rf parquets_HEP
    rm -rf logs
fi

# Define stages and session mapping
# Session 0 -> R
# Session 1 -> W
# Session 2 -> N1
# Session 3 -> N2
# Session 4 -> N3

# Function to setup session
setup_session() {
    SESSION=$1
    STAGE=$2
    echo "Setting up session $SESSION for stage $STAGE"
    tmux new-session -d -s $SESSION
    tmux send-keys -t $SESSION "source venv/bin/activate" C-m
    tmux send-keys -t $SESSION "python HEP_parquet_generation.py --stage $STAGE --edf_root EDF_Format/Berkeley_data" C-m
}
setup_session 1 W
setup_session 0 R
setup_session 2 N1
setup_session 3 N2
setup_session 4 N3

echo "All sessions initialized."
tmux ls
tmux a -t 1
