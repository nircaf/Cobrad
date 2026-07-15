#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source venv/bin/activate

MODES=(
    "Single Group Analysis"
    "Compare Groups"
    "Compare Sleep Stages"
    "Compare Groups All Sleep Stages"
    "Compare Groups Non-EEG Channels"
    "Per Channel Group Comparison"
    "Diagnosis Group Comparison"
)

echo "Available modes:"
for i in "${!MODES[@]}"; do
    echo "  $((i+1)). ${MODES[$i]}"
done
echo "  A. All modes"
echo ""

echo "Auto-selecting A (all modes)."
SELECTED_MODES=("${MODES[@]}")

echo ""
echo "Opening ${#SELECTED_MODES[@]} window(s)..."

PIDS=()
URLS=()
BASE_PORT=${BASE_PORT:-8501}

cleanup() {
    if [[ ${#PIDS[@]} -gt 0 ]]; then
        echo ""
        echo "Stopping Streamlit servers..."
        kill "${PIDS[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

is_port_free() {
    local port="$1"
    python - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
for family, host in ((socket.AF_INET, ""), (socket.AF_INET6, "::")):
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        if family == socket.AF_INET6:
            try:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            except OSError:
                pass
        try:
            sock.bind((host, port))
        except OSError:
            sys.exit(1)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.2)
    try:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            sys.exit(1)
    except OSError:
        pass
PY
}

next_free_port() {
    local port="$1"
    while ! is_port_free "$port"; do
        port=$((port + 1))
    done
    echo "$port"
}

is_pid_running() {
    local pid="$1"
    kill -0 "$pid" 2>/dev/null
}

START_PID=""
START_URL=""
START_PORT=""

start_streamlit() {
    local mode="$1"
    local start_port="$2"
    local port
    local url
    local pid

    port="$(next_free_port "$start_port")"
    while true; do
        url="http://localhost:${port}/"
        echo "  Starting: $mode on $url" >&2
        streamlit run 6_hep_group_comparison.py \
            --server.port "$port" \
            --server.headless true \
            --browser.gatherUsageStats false \
            -- --mode "$mode" &
        pid=$!

        sleep 0.5
        if is_pid_running "$pid"; then
            START_PID="$pid"
            START_URL="$url"
            START_PORT="$port"
            return 0
        fi

        wait "$pid" 2>/dev/null || true
        if is_port_free "$port"; then
            echo "  Error: $mode exited immediately on $url" >&2
            return 1
        fi

        port=$((port + 1))
        port="$(next_free_port "$port")"
        echo "  Port was unavailable; retrying $mode on http://localhost:${port}/" >&2
    done
}

wait_for_streamlit() {
    local port="$1"
    local url="http://localhost:${port}/_stcore/health"
    local tries=120
    local i
    for ((i=1; i<=tries; i++)); do
        if curl -fsS "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

open_url() {
    local url="$1"
    python - "$url" <<'PY'
import sys
import webbrowser

webbrowser.open(sys.argv[1], new=2)
PY
}

port="$BASE_PORT"
for mode in "${SELECTED_MODES[@]}"; do
    start_streamlit "$mode" "$port"
    pid="$START_PID"
    url="$START_URL"
    port="$START_PORT"
    PIDS+=("$pid")
    URLS+=("$url")

    if wait_for_streamlit "$port"; then
        open_url "$url"
    else
        echo "  Warning: $mode did not become ready on $url"
    fi
    port=$((port + 1))
done

echo ""
echo "All windows launched. Press Ctrl+C to stop all."
printf '  %s\n' "${URLS[@]}"
wait
