#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/venv/bin/activate"

COUNT="${1:-1}"
BASE_PORT="${STREAMLIT_BASE_PORT:-8501}"
SLEEP_STAGES=("W" "light_sleep" "N3" "R")

is_port_free() {
    local port="$1"

    if command -v ss >/dev/null 2>&1; then
        ! ss -ltn "( sport = :$port )" | tail -n +2 | grep -q .
    elif command -v lsof >/dev/null 2>&1; then
        ! lsof -iTCP:"$port" -sTCP:LISTEN -Pn >/dev/null 2>&1
    else
        ! python - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sys.exit(0 if sock.connect_ex(("127.0.0.1", port)) == 0 else 1)
PY
    fi
}

find_open_port() {
    local port="$1"

    while ! is_port_free "$port"; do
        port=$((port + 1))
    done

    echo "$port"
}

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

next_port="$BASE_PORT"
for ((i = 0; i < COUNT; i++)); do
    port="$(find_open_port "$next_port")"
    next_port=$((port + 1))
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
