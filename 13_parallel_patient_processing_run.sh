#!/usr/bin/env bash
set -euo pipefail

# Process Harvard EDF/PSG recordings, optionally restricted to an exact
# recording-level manifest written by script 18.
#
# Usage:
#   ./13_parallel_patient_processing_run.sh Harvard_Electroencephalography
#   EDF_FILES_FILE=/path/eligible_edfs.txt ./13_parallel_patient_processing_run.sh Harvard_Electroencephalography
#   RUN_FOREGROUND=1 EDF_FILES_FILE=/path/eligible_edfs.txt ./13_parallel_patient_processing_run.sh Harvard_Electroencephalography

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDF_FOLDER="${1:-Harvard_Electroencephalography}"
EDF_ROOT="${SCRIPT_DIR}/EDF_Format/${EDF_FOLDER}"
SESSION_NAME="${SESSION_NAME:-Parallel_HEP}"
RUN_FOREGROUND="${RUN_FOREGROUND:-0}"
EDF_FILES_FILE="${EDF_FILES_FILE:-}"
WATCH_MANIFEST="${WATCH_MANIFEST:-0}"
EDF_MANIFEST_DONE_FILE="${EDF_MANIFEST_DONE_FILE:-}"
WATCH_POLL_SECONDS="${WATCH_POLL_SECONDS:-5}"
PARALLEL_HEP_WORKERS="${PARALLEL_HEP_WORKERS:-}"

if [[ ! -d "$EDF_ROOT" ]]; then
    echo "EDF root does not exist: $EDF_ROOT" >&2
    exit 1
fi
if [[ -n "$EDF_FILES_FILE" ]]; then
    if [[ "$WATCH_MANIFEST" == "1" && ! -f "$EDF_FILES_FILE" ]]; then
        echo "EDF recording manifest does not exist: $EDF_FILES_FILE" >&2
        exit 1
    elif [[ "$WATCH_MANIFEST" != "1" && ! -s "$EDF_FILES_FILE" ]]; then
        echo "EDF recording manifest is missing or empty: $EDF_FILES_FILE" >&2
        exit 1
    fi
fi
if [[ "$WATCH_MANIFEST" == "1" && -z "$EDF_MANIFEST_DONE_FILE" ]]; then
    echo "WATCH_MANIFEST=1 requires EDF_MANIFEST_DONE_FILE." >&2
    exit 1
fi

PROCESS_ARGS=(
    python
    "${SCRIPT_DIR}/parallel_patient_processing.py"
    --edf_root
    "$EDF_ROOT"
)
if [[ -n "$EDF_FILES_FILE" ]]; then
    PROCESS_ARGS+=(--edf-files-file "$EDF_FILES_FILE")
fi
if [[ "$WATCH_MANIFEST" == "1" ]]; then
    PROCESS_ARGS+=(
        --watch-manifest
        --watch-done-file
        "$EDF_MANIFEST_DONE_FILE"
        --watch-poll-seconds
        "$WATCH_POLL_SECONDS"
    )
fi
if [[ -n "$PARALLEL_HEP_WORKERS" ]]; then
    PROCESS_ARGS+=(--workers "$PARALLEL_HEP_WORKERS")
fi

if [[ "$RUN_FOREGROUND" == "1" ]]; then
    cd "$SCRIPT_DIR"
    source venv/bin/activate
    exec "${PROCESS_ARGS[@]}"
fi

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not available on PATH." >&2
    exit 1
fi
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists; leaving it running."
    echo "Attach: tmux attach -t $SESSION_NAME"
    exit 0
fi

printf -v quoted_process ' %q' "${PROCESS_ARGS[@]}"
printf -v tmux_command \
    'cd %q && source venv/bin/activate && exec%s' \
    "$SCRIPT_DIR" "$quoted_process"

tmux new-session -d -s "$SESSION_NAME" -n processing "$tmux_command"

echo "Started parallel processing in tmux session: $SESSION_NAME"
echo "  EDF root : $EDF_ROOT"
if [[ -n "$EDF_FILES_FILE" ]]; then
    echo "  Manifest : $EDF_FILES_FILE"
fi
echo "  Attach   : tmux attach -t $SESSION_NAME"

if [[ -t 0 && -t 1 ]]; then
    tmux attach -t "$SESSION_NAME"
fi
