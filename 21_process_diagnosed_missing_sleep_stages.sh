#!/usr/bin/env bash
set -euo pipefail

# Incrementally process missing Light Sleep, N3, and REM pickles for the
# diagnosed cohort shown by dashboard 16. Each run re-discovers the cohort and
# schedules only missing patient-stage combinations.
#
# Usage:
#   ./21_process_diagnosed_missing_sleep_stages.sh
#   ./21_process_diagnosed_missing_sleep_stages.sh --dry-run
#   COHORT_SOURCE=all-local ./21_process_diagnosed_missing_sleep_stages.sh
#   STAGES="light_sleep N3 R W" ./21_process_diagnosed_missing_sleep_stages.sh
#
# The all-local cohort is intentionally opt-in: it currently contains thousands
# of diagnosed EDF subjects, whereas the dashboard cohort is much smaller.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
PYTHON_SCRIPT="${SCRIPT_DIR}/21_process_diagnosed_missing_sleep_stages.py"
SESSION="${SESSION:-diagnosed-missing-stages}"
COHORT_SOURCE="${COHORT_SOURCE:-dashboard}"
STAGES="${STAGES:-light_sleep N3 R}"
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/logs/21_diagnosed_missing_stages}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${LOG_ROOT}/${RUN_ID}"

read -r -a STAGE_ARGS <<< "$STAGES"

run_worker() {
    mkdir -p "$RUN_DIR"
    cd "$SCRIPT_DIR"
    source venv/bin/activate

    if [[ "${WAIT_FOR_PARALLEL_HEP:-0}" == "1" ]]; then
        while pgrep -f '[p]arallel_patient_processing.py' >/dev/null 2>&1; do
            echo "$(date --iso-8601=seconds) Waiting for the existing Parallel_HEP job to finish; manifests will be rediscovered afterward."
            sleep 60
        done
    fi

    local -a command=(
        python "$PYTHON_SCRIPT"
        --cohort "$COHORT_SOURCE"
        --stages "${STAGE_ARGS[@]}"
        --output-dir "$RUN_DIR"
        --execute
    )
    if [[ "${RETRY_PERMANENT:-0}" == "1" ]]; then
        command+=(--retry-permanent)
    fi
    "${command[@]}" 2>&1 | tee -a "${RUN_DIR}/run.log"
}

if [[ "${1:-}" == "--worker" ]]; then
    run_worker
    exit
fi

if [[ "${1:-}" == "--dry-run" ]]; then
    mkdir -p "$RUN_DIR"
    cd "$SCRIPT_DIR"
    source venv/bin/activate
    python "$PYTHON_SCRIPT" \
        --cohort "$COHORT_SOURCE" \
        --stages "${STAGE_ARGS[@]}" \
        --output-dir "$RUN_DIR"
    exit
fi

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is unavailable; running in the foreground."
    run_worker
    exit
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' is already running."
    echo "Attach: tmux attach -t $SESSION"
    exit 0
fi

mkdir -p "$RUN_DIR"
printf -v worker_command \
    'cd %q && RUN_ID=%q RUN_DIR=%q COHORT_SOURCE=%q STAGES=%q LOG_ROOT=%q RETRY_PERMANENT=%q WAIT_FOR_PARALLEL_HEP=%q bash %q --worker' \
    "$SCRIPT_DIR" "$RUN_ID" "$RUN_DIR" "$COHORT_SOURCE" "$STAGES" "$LOG_ROOT" \
    "${RETRY_PERMANENT:-0}" "${WAIT_FOR_PARALLEL_HEP:-0}" "$SCRIPT_PATH"

tmux new-session -d -s "$SESSION" -n stages "$worker_command"

echo "Diagnosed missing-stage processing started."
echo "  Cohort : $COHORT_SOURCE"
echo "  Stages : $STAGES"
echo "  Session: $SESSION"
echo "  Attach : tmux attach -t $SESSION"
echo "  Log    : ${RUN_DIR}/run.log"
