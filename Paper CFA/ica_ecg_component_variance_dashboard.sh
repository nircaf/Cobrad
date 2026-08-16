#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

source venv/bin/activate

PORT="${PORT:-8501}"

is_port_free() {
    local port="$1"
    python - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    try:
        sock.bind(("", port))
    except OSError:
        sys.exit(1)
PY
}

next_free_port() {
    local port="$1"
    while ! is_port_free "$port"; do
        port=$((port + 1))
    done
    echo "$port"
}

wait_for_streamlit() {
    local port="$1"
    local health_url="http://localhost:${port}/_stcore/health"
    local i
    for ((i=1; i<=120; i++)); do
        if curl -fsS "$health_url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

open_url() {
    local url="$1"
    if command -v brave-browser >/dev/null 2>&1; then
        brave-browser --new-tab "$url" >/dev/null 2>&1 &
    elif command -v brave >/dev/null 2>&1; then
        brave --new-tab "$url" >/dev/null 2>&1 &
    elif command -v brave-browser-stable >/dev/null 2>&1; then
        brave-browser-stable --new-tab "$url" >/dev/null 2>&1 &
    else
        # Uses the desktop default browser (Brave on this workstation).
        python - "$url" <<'PY'
import sys
import webbrowser

webbrowser.open(sys.argv[1], new=2)
PY
    fi
}

PORT="$(next_free_port "$PORT")"
URL="http://localhost:${PORT}/"

echo "Starting ICA ECG-component variance dashboard on ${URL}"
streamlit run "Paper CFA/ica_ecg_component_variance_dashboard.py" \
    --server.port "$PORT" \
    --server.headless true \
    --server.fileWatcherType none \
    --browser.gatherUsageStats false &
PID="$!"

trap 'kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true' EXIT INT TERM

if wait_for_streamlit "$PORT"; then
    open_url "$URL"
else
    echo "Warning: dashboard did not become ready on ${URL}" >&2
fi

wait "$PID"
