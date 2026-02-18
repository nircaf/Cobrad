#!/bin/bash

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
    tmux send-keys -t $SESSION "python HEP_parquet_generation.py --stage $STAGE" C-m
}

setup_session 0 R
setup_session 1 W
setup_session 2 N1
setup_session 3 N2
setup_session 4 N3

echo "All sessions initialized."
tmux ls
