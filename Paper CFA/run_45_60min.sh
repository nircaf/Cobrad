#!/bin/bash
# Resumable launcher for the 45min and 60min full-cohort CFA reruns.
# Per-EDF caching (cfa_variance_explained_cache/) means re-running this
# script after a crash/kill just resumes where it left off -- no lost work.
#
# Usage:
#   bash "Paper CFA/run_45_60min.sh"          # foreground
#   setsid nohup bash "Paper CFA/run_45_60min.sh" > "Paper CFA/run_45_60min.log" 2>&1 < /dev/null & disown
#
# To check status:
#   tail -f "Paper CFA/run_45_60min.log"
#   pgrep -af "cfa_variance_explained.py --window-minutes"
#
# To resume after a crash, just re-run this same command -- cached EDFs
# are skipped automatically.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
VENV=/storage/pblab_shared_data/Nir/Cobrad/venv/bin/python

for m in 45 60; do
    echo "=== window-minutes=$m starting $(date) ==="
    "$VENV" cfa_variance_explained.py --window-minutes "$m" \
        --output "$(pwd)/cfa_variance_explained_${m}min.parquet"
    echo "=== window-minutes=$m done $(date) ==="
done
