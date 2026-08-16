#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source venv/bin/activate

# Open three independent dashboard modes, like 06_hep_group_comparison.sh.
# Each entry is "label|mode|gt10", gt10 overrides the ≥10-electrode toggle default.
MODES=(
    "Non-diagnosis patients — complete analysis (≥10 EEG channels)|Non-diagnosis patients — complete analysis|true"
    "Non-diagnosis patients — complete analysis (all channels)|Non-diagnosis patients — complete analysis|false"
    "Diagnosis comparison — HEP + HR/T-wave features (all channels)|Diagnosis comparison — HEP + HR/T-wave features|false"
)

BASE_PORT="${BASE_PORT:-${PORT:-8501}}"
PIDS=()
URLS=()

cleanup() {
    if [[ ${#PIDS[@]} -gt 0 ]]; then
        echo ""
        echo "Stopping Streamlit servers..."
        kill "${PIDS[@]}" 2>/dev/null || true
        wait "${PIDS[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

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

echo "Opening ${#MODES[@]} dashboard modes asynchronously..."

port="$BASE_PORT"
for entry in "${MODES[@]}"; do
    IFS='|' read -r label mode gt10 <<< "$entry"
    port="$(next_free_port "$port")"
    url="http://localhost:${port}/"

    echo "  Starting: ${label} on ${url}"
    streamlit run 16_diagnosis_sleep_stage_comparison_dashboard.py \
        --server.port "$port" \
        --server.headless true \
        --server.fileWatcherType none \
        --browser.gatherUsageStats false \
        -- --mode "$mode" --gt10 "$gt10" &
    PIDS+=("$!")
    URLS+=("$url")

    # Each readiness check and browser-tab launch runs independently.
    (
        if wait_for_streamlit "$port"; then
            open_url "$url"
        else
            echo "  Warning: ${mode} did not become ready on ${url}" >&2
        fi
    ) &

    port=$((port + 1))
done

echo ""
echo "All modes started. Press Ctrl+C to stop all."
printf '  %s\n' "${URLS[@]}"
wait "${PIDS[@]}"
