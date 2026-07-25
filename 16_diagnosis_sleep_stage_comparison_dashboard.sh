#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
source venv/bin/activate

# Use PORT when supplied; otherwise select the first available Streamlit port.
PORT="${PORT:-8501}"
while ss -ltnH "sport = :${PORT}" | grep -q .; do
  PORT=$((PORT + 1))
done
URL="http://localhost:${PORT}"

echo "Starting dashboard at ${URL}"

# Open browser once server is up
( until curl -s -o /dev/null "$URL"; do sleep 1; done
  xdg-open "$URL" >/dev/null 2>&1 || open "$URL" >/dev/null 2>&1 ) &

# Run Streamlit app
streamlit run 16_diagnosis_sleep_stage_comparison_dashboard.py \
  --server.port "$PORT" \
  --server.fileWatcherType none
