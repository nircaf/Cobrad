#!/bin/bash
# Launch normal and >=10-standard-10-20-channel HEP caches in one tmux session.
# Usage:
#   ./17_generate_hep_cache.sh          # uses 1/4 of available CPUs in total
#   ./17_generate_hep_cache.sh --force  # rebuild from scratch
#   ./17_generate_hep_cache.sh --workers 24 --force  # total budget override
#   HEP_TOTAL_WORKERS=40 ./17_generate_hep_cache.sh   # environment override
#
# Two subprocesses run in separate tmux windows. Per-stage locks prevent them
# from writing the same base cache concurrently. The tmux session closes after
# both subprocesses finish, so a later invocation can start a new update.
#
# Attach to session:  tmux attach -t hep-cache
# Switch windows:     Ctrl+B then N/P
# Watch live output:  tmux attach -t hep-cache   (Ctrl+B D to detach)
# Kill session:       tmux kill-session -t hep-cache

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="hep-cache"

# Kill existing session if already running
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists."
    echo "  Attach : tmux attach -t $SESSION"
    echo "  Kill   : tmux kill-session -t $SESSION"
    exit 0
fi

AVAILABLE_CPUS="$(nproc)"
TOTAL_WORKERS="${HEP_TOTAL_WORKERS:-$(( AVAILABLE_CPUS / 4 ))}"

# For this launcher, --workers means the total budget shared by both windows.
# Remove it from the common arguments before assigning each subprocess its share.
COMMON_ARGS=()
while (( $# > 0 )); do
    case "$1" in
        --workers)
            if (( $# < 2 )); then
                echo "--workers requires a positive integer." >&2
                exit 2
            fi
            TOTAL_WORKERS="$2"
            shift 2
            ;;
        --workers=*)
            TOTAL_WORKERS="${1#*=}"
            shift
            ;;
        *)
            COMMON_ARGS+=("$1")
            shift
            ;;
    esac
done

if ! [[ "$TOTAL_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Worker budget must be a positive integer, got: $TOTAL_WORKERS" >&2
    exit 2
fi
(( TOTAL_WORKERS >= 2 )) || TOTAL_WORKERS=2
if (( TOTAL_WORKERS > AVAILABLE_CPUS )); then
    TOTAL_WORKERS="$AVAILABLE_CPUS"
fi

# Raw-patient HEP processing is CPU-heavy; filtered-cache derivation is mostly I/O.
NORMAL_WORKERS=$(( TOTAL_WORKERS * 3 / 4 ))
COHORT_WORKERS=$(( TOTAL_WORKERS - NORMAL_WORKERS ))
(( COHORT_WORKERS >= 1 )) || COHORT_WORKERS=1
if (( NORMAL_WORKERS + COHORT_WORKERS > AVAILABLE_CPUS )); then
    NORMAL_WORKERS=$(( AVAILABLE_CPUS - COHORT_WORKERS ))
fi

QUOTED_ARGS=""
if (( ${#COMMON_ARGS[@]} > 0 )); then
    printf -v QUOTED_ARGS ' %q' "${COMMON_ARGS[@]}"
fi
COMMON_PREFIX="cd $(printf '%q' "$SCRIPT_DIR") && source venv/bin/activate &&"
THREAD_LIMITS="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1"
NORMAL_CMD="$COMMON_PREFIX $THREAD_LIMITS python 17_generate_hep_cache.py$QUOTED_ARGS --workers $NORMAL_WORKERS; status=\$?; echo '--- NORMAL CACHE DONE ---'; exit \$status"
COHORT_CMD="$COMMON_PREFIX $THREAD_LIMITS python 17_generate_hep_cache.py$QUOTED_ARGS --min-eeg-channels 10 --workers $COHORT_WORKERS; status=\$?; echo '--- >=10 STANDARD 10-20 EEG CACHE DONE ---'; exit \$status"

tmux new-session -d -s "$SESSION" -n "all-patients" -x 220 -y 50
tmux send-keys -t "$SESSION:all-patients" "$NORMAL_CMD" Enter
tmux new-window -d -t "$SESSION" -n "min10-1020eeg"
tmux send-keys -t "$SESSION:min10-1020eeg" "$COHORT_CMD" Enter

echo "Cache generation started in tmux session '$SESSION'."
echo "  CPUs   : $AVAILABLE_CPUS available; $TOTAL_WORKERS total workers requested"
echo "  Split  : $NORMAL_WORKERS all-patient workers + $COHORT_WORKERS min10-1020eeg workers"
echo "  Attach : tmux attach -t $SESSION"
echo "  Output : tmux attach -t $SESSION  (Ctrl+B D to detach without stopping)"
echo "  Windows: all-patients and min10-1020eeg (Ctrl+B N/P to switch)"

tmux attach -t "$SESSION"
