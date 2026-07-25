#!/usr/bin/env bash
set -euo pipefail

# Download every configured Harvard BIDS study, wait for all download workers,
# and then launch scripts 17 and 13.
#
# Usage:
#   ./18_download_all_bids_and_run_17_13.sh
#   STUDIES="I0002 I0003 I0004 I0006" SHARDS_PER_STUDY=2 ./18_download_all_bids_and_run_17_13.sh
#
# Monitor:
#   tmux attach -t harvard-bids-download-process
#
# The following variables are passed through to the downloader when set:
# AWS_BIN, S3_ROOT, S3_BIDS, ROOT_DEST, MIN_ELECTRODES, USE_REGISTRY,
# REFRESH_REGISTRY, REGISTRY_VERSION, REGISTRY_DIR, LOCK_DIR, and MAX_DOWNLOADS.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
DOWNLOADER="${SCRIPT_DIR}/download_new_harvard_bids_patients_gt10_electrodes.sh"
SCRIPT_17="${SCRIPT_DIR}/17_generate_hep_cache.sh"
SCRIPT_13="${SCRIPT_DIR}/13_parallel_patient_processing_run.sh"

SESSION="${SESSION:-harvard-bids-download-process}"
STUDIES="${STUDIES:-I0002 I0003 I0004 I0006}"
SHARDS_PER_STUDY="${SHARDS_PER_STUDY:-1}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/18_download_all_bids}"

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required script not found: $1" >&2
        exit 1
    fi
}

require_file "$DOWNLOADER"
require_file "$SCRIPT_17"
require_file "$SCRIPT_13"

if ! [[ "$SHARDS_PER_STUDY" =~ ^[1-9][0-9]*$ ]]; then
    echo "SHARDS_PER_STUDY must be a positive integer." >&2
    exit 2
fi

run_pipeline() {
    mkdir -p "$LOG_DIR"

    local -a pids=()
    local -a names=()
    local study
    local shard
    local status=0

    echo "Downloading all configured Harvard BIDS studies: $STUDIES"
    echo "Download workers per study: $SHARDS_PER_STUDY"
    echo "Logs: $LOG_DIR"
    echo

    for study in $STUDIES; do
        for ((shard = 0; shard < SHARDS_PER_STUDY; shard++)); do
            names+=("${study}-${shard}")
            (
                cd "$SCRIPT_DIR"
                RUN_WORKER=1 \
                STUDY="$study" \
                SHARDS_PER_STUDY="$SHARDS_PER_STUDY" \
                SHARD_INDEX="$shard" \
                bash "$DOWNLOADER"
            ) > >(tee -a "${LOG_DIR}/${study}-${shard}.log") 2>&1 &
            pids+=("$!")
        done
    done

    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            echo "Download worker ${names[$i]} finished successfully."
        else
            echo "Download worker ${names[$i]} failed." >&2
            status=1
        fi
    done

    if (( status != 0 )); then
        echo "At least one download failed; scripts 17 and 13 will not be started." >&2
        return "$status"
    fi

    echo
    echo "All BIDS downloads finished successfully."
    echo "Launching script 17..."
    # Both launchers try to attach after creating their own tmux session. Because
    # this pipeline already runs inside tmux, that attach is intentionally
    # rejected while the newly created sessions continue in the background.
    bash "$SCRIPT_17" </dev/null || true

    echo
    echo "Launching script 13 for Harvard_Electroencephalography..."
    bash "$SCRIPT_13" Harvard_Electroencephalography </dev/null || true

    echo
    echo "Pipeline launch complete."
    echo "  Script 17 session: hep-cache"
    echo "  Script 13 session: Parallel_HEP"
}

if [[ "${1:-}" == "--pipeline-worker" ]]; then
    run_pipeline
    exit
fi

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not available on PATH." >&2
    exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists."
    echo "Attach: tmux attach -t $SESSION"
    exit 0
fi

mkdir -p "$LOG_DIR"
printf -v worker_command \
    'cd %q && STUDIES=%q SHARDS_PER_STUDY=%q LOG_DIR=%q bash %q --pipeline-worker 2>&1 | tee -a %q' \
    "$SCRIPT_DIR" "$STUDIES" "$SHARDS_PER_STUDY" "$LOG_DIR" "$SCRIPT_PATH" \
    "${LOG_DIR}/pipeline.log"

tmux new-session -d -s "$SESSION" -n pipeline "$worker_command"

echo "Download-and-process pipeline started."
echo "  Studies : $STUDIES"
echo "  Shards  : $SHARDS_PER_STUDY per study"
echo "  Session : $SESSION"
echo "  Attach  : tmux attach -t $SESSION"
echo "  Log     : ${LOG_DIR}/pipeline.log"
