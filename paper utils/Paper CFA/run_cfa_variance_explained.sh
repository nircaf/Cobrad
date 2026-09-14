#!/usr/bin/env bash
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if [[ ! -f "$PROJECT_DIR/venv/bin/activate" ]]; then
    echo "ERROR: venv not found at $PROJECT_DIR/venv" >&2
    exec bash
fi

source "$PROJECT_DIR/venv/bin/activate"
cd "$PROJECT_DIR"
LOG="$SCRIPT_DIR/cfa_variance_explained.log"
echo "[$(date --iso-8601=seconds)] Starting CFA batch: $*" | tee -a "$LOG"
python "$SCRIPT_DIR/cfa_variance_explained.py" "$@" 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
if (( status != 0 )); then
    echo "CFA batch exited with status $status. This tmux pane is being kept open for inspection." | tee -a "$LOG"
    exec bash
fi
