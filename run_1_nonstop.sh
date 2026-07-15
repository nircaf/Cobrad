#!/bin/bash

# Re-launch under nohup if not already detached
if [ -z "$NOHUP_RUNNING" ]; then
    NOHUP_RUNNING=1 nohup bash "$0" "$@" > run_1_nonstop.log 2>&1 &
    echo "Started in background (PID $!). Attach with: tmux attach -t hep_parquet"
    exit 0
fi

RETRY_FAILED=""
for arg in "$@"; do
  if [ "$arg" = "--retry" ]; then
    RETRY_FAILED="--rerun-failed"
  fi
done

SESSION="hep_parquet"
STAGES=("N3" "R" "W" "light_sleep")
EDF_ROOTS=("EDF_Format/The_Human_Sleep_Project" "EDF_Format/EDF")

tmux new-session -d -s "$SESSION"

first=true
for stage in "${STAGES[@]}"; do
    for edf_root in "${EDF_ROOTS[@]}"; do
        window="${stage}_$(basename $edf_root)"
        if $first; then
            tmux rename-window -t "$SESSION:0" "$window"
            first=false
        else
            tmux new-window -t "$SESSION" -n "$window"
        fi
        tmux send-keys -t "$SESSION:$window" \
            "python HEP_parquet_generation.py --stage $stage --edf_root $edf_root $RETRY_FAILED" Enter
    done
done

echo "Started tmux session '$SESSION'. Attach with:"
echo "  tmux attach -t $SESSION"
