#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_COHORT_FILE="${CURRENT_COHORT_FILE:-${SCRIPT_DIR}/logs/21_diagnosed_missing_stages/20260731_064021/diagnosed_subjects.txt}"
RECOVERY_ROOT="${RECOVERY_ROOT:-${SCRIPT_DIR}/logs/22_diagnosed_three_stage_recovery}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RECOVERY_ROOT}/${RUN_ID}"
WORKERS="${WORKERS:-8}"

mkdir -p "$RUN_DIR"
cd "$SCRIPT_DIR"
source venv/bin/activate

while tmux has-session -t diagnosed-missing-stages 2>/dev/null; do
    echo "$(date --iso-8601=seconds) Waiting for the current diagnosed stage extraction."
    sleep 30
done

# Retry fragmented stages at a still-conservative five-minute minimum. The
# normal extraction remains at 7.5 minutes (15 YASA epochs).
HEP_MINIMUM_USABLE_STREAK_EPOCHS=10 \
python 21_process_diagnosed_missing_sleep_stages.py \
    --cohort dashboard \
    --stages light_sleep N3 R \
    --retry-permanent \
    --execute \
    --output-dir "${RUN_DIR}/stage_retry"

# Refresh the ordinary caches, then selectively retry the missing third stage
# for diagnosed patients with a patient-specific HF/LF cutoff.
python 17_generate_hep_cache.py \
    --groups Harvard_Electroencephalography \
    --stages light_sleep N3 R \
    --complete-three-stages light_sleep N3 R \
    --third-stage-candidates-file "$CURRENT_COHORT_FILE" \
    --workers "$WORKERS"

# Derive the dashboard's >=10-standard-electrode caches from the recovered
# ordinary caches.
python 17_generate_hep_cache.py \
    --groups Harvard_Electroencephalography \
    --stages light_sleep N3 R \
    --min-eeg-channels 10 \
    --workers "$WORKERS"

echo "$(date --iso-8601=seconds) Diagnosed three-stage recovery complete."
