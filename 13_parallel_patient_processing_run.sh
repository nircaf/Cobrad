#!/bin/bash

# Set the root directory — pass subfolder name inside EDF_Format (default: The_Human_Sleep_Project)
EDF_FOLDER=${1:-"Harvard_Electroencephalography"}
EDF_ROOT="EDF_Format/${EDF_FOLDER}"
SESSION_NAME="Parallel_HEP"

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Killing existing tmux session: $SESSION_NAME"
    tmux kill-session -t "$SESSION_NAME"
fi

echo "Starting new tmux session: $SESSION_NAME"

# Create a detached tmux session
tmux new-session -d -s "$SESSION_NAME"

# Send the commands to the tmux session
tmux send-keys -t "$SESSION_NAME" "source venv/bin/activate" Enter
tmux send-keys -t "$SESSION_NAME" "python parallel_patient_processing.py --edf_root \"$EDF_ROOT\"" Enter

echo "================================================================"
echo "Started parallel processing inside tmux session: $SESSION_NAME"
echo "You can safely log out and the process will continue."
echo ""
echo "To view the live output, attach to the session using:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "To detach from the session while viewing (without stopping it):"
echo "  press 'Ctrl+b' then 'd'"
echo "================================================================"
tmux attach -t "$SESSION_NAME"