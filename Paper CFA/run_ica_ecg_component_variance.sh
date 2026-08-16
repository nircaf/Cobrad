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
LOG="$SCRIPT_DIR/ica_ecg_component_variance.log"
echo "[$(date --iso-8601=seconds)] Starting ICA batch: $*" | tee -a "$LOG"
python "$SCRIPT_DIR/ica_ecg_component_variance.py" "$@" 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
if (( status != 0 )); then
    echo "ICA batch exited with status $status. This tmux pane is being kept open for inspection." | tee -a "$LOG"
    exec bash
fi
