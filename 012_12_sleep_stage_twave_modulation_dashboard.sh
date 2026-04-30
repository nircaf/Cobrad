#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/venv/bin/activate"

COUNT="${1:-1}"
BASE_PORT="${STREAMLIT_BASE_PORT:-8501}"
SLEEP_STAGES=("W" "light_sleep" "N3" "R")

if ! [[ "$COUNT" =~ ^[0-9]+$ ]] || [[ "$COUNT" -lt 1 ]]; then
    echo "Usage: $0 [number_of_windows]"
    echo "Example: $0 4"
    exit 1
fi

pids=()
cleanup() {
    for pid in "${pids[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup INT TERM EXIT

for ((i = 0; i < COUNT; i++)); do
    port=$((BASE_PORT + i))
    sleep_stage="${SLEEP_STAGES[$((i % ${#SLEEP_STAGES[@]}))]}"
    echo "Starting dashboard $((i + 1))/$COUNT: sleep stage=${sleep_stage}, URL=http://localhost:${port}"
    STREAMLIT_DEFAULT_SLEEP_STAGE="$sleep_stage" \
    streamlit run "$SCRIPT_DIR/12_sleep_stage_twave_modulation_dashboard.py" \
        --server.port "$port" \
        --server.headless false &
    pids+=("$!")
    sleep 1
done

wait
