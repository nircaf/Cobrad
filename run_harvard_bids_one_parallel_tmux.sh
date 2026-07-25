#!/usr/bin/env bash
set -euo pipefail

# Launch one download worker (one shard) per Harvard BIDS study in tmux.
# Override STUDIES or TMUX_SESSION when needed, for example:
#   STUDIES="I0002" TMUX_SESSION=harvard-i0002 ./run_harvard_bids_one_parallel_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SHARDS_PER_STUDY=1 \
TMUX_SESSION="${TMUX_SESSION:-harvard-bids-downloads-one-parallel}" \
bash "${SCRIPT_DIR}/download_new_harvard_bids_patients_gt10_electrodes.sh" start-tmux
