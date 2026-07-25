#!/usr/bin/env bash
set -euo pipefail

# Start the Harvard BIDS downloader in tmux with one shard (worker) per study.
#
# Usage:
#   ./run_download_new_harvard_bids_one_parallel_tmux.sh
#   STUDIES="I0002 I0003" ./run_download_new_harvard_bids_one_parallel_tmux.sh
#   DRY_RUN=1 ./run_download_new_harvard_bids_one_parallel_tmux.sh
#
# Monitor:
#   tmux attach -t harvard-bids-downloads-1parallel

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOADER="${SCRIPT_DIR}/download_new_harvard_bids_patients_gt10_electrodes.sh"

if [[ ! -f "$DOWNLOADER" ]]; then
    echo "Downloader not found: $DOWNLOADER" >&2
    exit 1
fi

export SHARDS_PER_STUDY=1
export TMUX_SESSION="${TMUX_SESSION:-harvard-bids-downloads-1parallel}"

exec bash "$DOWNLOADER" start-tmux
