#!/usr/bin/env bash
set -euo pipefail

# Download every diagnosed Harvard BIDS PSG recording that is at least 7 hours
# long and has at least 5 canonical scalp EEG electrodes. Process recordings
# with script 13 as each download completes, then refresh HEP caches with
# script 17 after the live queue drains.
#
# Usage:
#   ./18_download_all_bids_and_run_17_13.sh
#   STUDIES="I0002 I0003 I0004 I0006" SHARDS_PER_STUDY=2 ./18_download_all_bids_and_run_17_13.sh
#
# Monitor:
#   tmux attach -t harvard-bids-download-process
#
# The following variables are passed through to the downloader when set:
# AWS_BIN, S3_ROOT, S3_BIDS, ROOT_DEST, MIN_ELECTRODES,
# MIN_DURATION_HOURS, USE_REGISTRY, REFRESH_REGISTRY, REGISTRY_VERSION,
# REGISTRY_DIR, LOCK_DIR, and MAX_DOWNLOADS.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
DOWNLOADER="${SCRIPT_DIR}/download_new_harvard_bids_patients_gt10_electrodes.sh"
SCRIPT_17="${SCRIPT_DIR}/17_generate_hep_cache.sh"
SCRIPT_13="${SCRIPT_DIR}/13_parallel_patient_processing_run.sh"
ELIGIBILITY_HELPER="${SCRIPT_DIR}/18_bids_psg_eligibility.py"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/venv/bin/python}"

SESSION="${SESSION:-harvard-bids-download-process}"
STUDIES="${STUDIES:-I0002 I0003 I0004 I0006}"
read -r -a STUDY_ARRAY <<< "$STUDIES"
MIN_ELECTRODES="${MIN_ELECTRODES:-5}"
MIN_DURATION_HOURS="${MIN_DURATION_HOURS:-7}"
BIDS_ROOT="${ROOT_DEST:-${SCRIPT_DIR}/EDF_Format/Harvard_Electroencephalography/bids}"
AWS_BIN="${AWS_BIN:-/storage/pblab_shared_data/Nir/bin/aws}"
S3_ROOT="${S3_ROOT:-s3://arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-credentialed-access-point}"
S3_BIDS="${S3_BIDS:-${S3_ROOT}/PSG/bids}"
DRY_RUN="${DRY_RUN:-0}"
MAX_DOWNLOADS="${MAX_DOWNLOADS:-0}"
USE_REGISTRY="${USE_REGISTRY:-1}"
REFRESH_REGISTRY="${REFRESH_REGISTRY:-0}"
REGISTRY_VERSION="${REGISTRY_VERSION:-3}"
LOCK_DIR="${LOCK_DIR:-${BIDS_ROOT}/.download_locks}"
REGISTRY_DIR="${REGISTRY_DIR:-${BIDS_ROOT}/.checked_subjects/v${REGISTRY_VERSION}/min-${MIN_ELECTRODES}_hours-${MIN_DURATION_HOURS}}"
AVAILABLE_CPUS="$(nproc)"
SHARDS_PER_STUDY="${SHARDS_PER_STUDY:-4}"
HEP_TOTAL_WORKERS="${HEP_TOTAL_WORKERS:-$(( AVAILABLE_CPUS / 2 ))}"
PROCESSING_WORKERS="${PROCESSING_WORKERS:-$(( AVAILABLE_CPUS / 3 ))}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/18_download_all_bids}"
MANIFEST_DIR="${MANIFEST_DIR:-${LOG_DIR}/manifests}"
DIAGNOSIS_MANIFEST_DIR="${DIAGNOSIS_MANIFEST_DIR:-${MANIFEST_DIR}/diagnosed_subjects}"
ELIGIBLE_EDF_MANIFEST="${ELIGIBLE_EDF_MANIFEST:-${MANIFEST_DIR}/eligible_local_psg_edfs.txt}"
ELIGIBILITY_REPORT="${ELIGIBILITY_REPORT:-${MANIFEST_DIR}/eligible_local_psg_report.csv}"
DOWNLOAD_DONE_FILE="${DOWNLOAD_DONE_FILE:-${MANIFEST_DIR}/downloads_complete}"
PROCESSING_LOG="${PROCESSING_LOG:-${LOG_DIR}/live_processing.log}"

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required script not found: $1" >&2
        exit 1
    fi
}

require_file "$DOWNLOADER"
require_file "$SCRIPT_17"
require_file "$SCRIPT_13"
require_file "$ELIGIBILITY_HELPER"
require_file "$PYTHON_BIN"

if ! [[ "$SHARDS_PER_STUDY" =~ ^[1-9][0-9]*$ ]]; then
    echo "SHARDS_PER_STUDY must be a positive integer." >&2
    exit 2
fi

run_pipeline() {
    mkdir -p "$LOG_DIR"
    mkdir -p "$MANIFEST_DIR"

    local -a pids=()
    local -a names=()
    local study
    local shard
    local status=0
    local processor_pid=""

    echo "Building diagnosed-subject manifests from local EHR parquet data..."
    "$PYTHON_BIN" "$ELIGIBILITY_HELPER" diagnosis-manifests \
        --output-dir "$DIAGNOSIS_MANIFEST_DIR" \
        --studies "${STUDY_ARRAY[@]}"

    if [[ "$DRY_RUN" != "1" ]]; then
        echo
        if [[ -s "$ELIGIBLE_EDF_MANIFEST" ]]; then
            echo "Reusing the existing eligible-PSG manifest while downloads run."
        else
            : > "$ELIGIBLE_EDF_MANIFEST"
            echo "Starting with an empty live manifest; downloads will append EDFs."
        fi
        rm -f "$DOWNLOAD_DONE_FILE"
        echo "Starting live PSG processing (${PROCESSING_WORKERS} workers)..."
        RUN_FOREGROUND=1 \
        WATCH_MANIFEST=1 \
        EDF_FILES_FILE="$ELIGIBLE_EDF_MANIFEST" \
        EDF_MANIFEST_DONE_FILE="$DOWNLOAD_DONE_FILE" \
        PARALLEL_HEP_WORKERS="$PROCESSING_WORKERS" \
            bash "$SCRIPT_13" Harvard_Electroencephalography \
            > >(tee -a "$PROCESSING_LOG") 2>&1 &
        processor_pid=$!
    fi

    echo
    echo "Downloading eligible Harvard BIDS PSGs from studies: $STUDIES"
    echo "Eligibility: diagnosed, >= ${MIN_DURATION_HOURS}h, >= ${MIN_ELECTRODES} scalp EEG electrodes"
    echo "Download workers per study: $SHARDS_PER_STUDY"
    echo "Logs: $LOG_DIR"
    echo

    for study in "${STUDY_ARRAY[@]}"; do
        for ((shard = 0; shard < SHARDS_PER_STUDY; shard++)); do
            names+=("${study}-${shard}")
            (
                cd "$SCRIPT_DIR"
                RUN_WORKER=1 \
                STUDY="$study" \
                SHARDS_PER_STUDY="$SHARDS_PER_STUDY" \
                SHARD_INDEX="$shard" \
                MIN_ELECTRODES="$MIN_ELECTRODES" \
                MIN_DURATION_HOURS="$MIN_DURATION_HOURS" \
                DIAGNOSIS_MANIFEST_DIR="$DIAGNOSIS_MANIFEST_DIR" \
                ROOT_DEST="$BIDS_ROOT" \
                AWS_BIN="$AWS_BIN" \
                S3_ROOT="$S3_ROOT" \
                S3_BIDS="$S3_BIDS" \
                DRY_RUN="$DRY_RUN" \
                MAX_DOWNLOADS="$MAX_DOWNLOADS" \
                USE_REGISTRY="$USE_REGISTRY" \
                REFRESH_REGISTRY="$REFRESH_REGISTRY" \
                REGISTRY_VERSION="$REGISTRY_VERSION" \
                LOCK_DIR="$LOCK_DIR" \
                REGISTRY_DIR="$REGISTRY_DIR" \
                PROCESSING_MANIFEST="$ELIGIBLE_EDF_MANIFEST" \
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
        if [[ -n "$processor_pid" ]]; then
            touch "$DOWNLOAD_DONE_FILE"
            wait "$processor_pid" || true
        fi
        echo "At least one download failed; cache generation will not start." >&2
        return "$status"
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "Dry run complete; local processing and cache generation were not started."
        return 0
    fi

    echo
    echo "All BIDS downloads finished successfully."
    echo "Refreshing the final local eligibility manifest..."
    "$PYTHON_BIN" "$ELIGIBILITY_HELPER" eligible-local \
        --bids-root "$BIDS_ROOT" \
        --diagnosis-dir "$DIAGNOSIS_MANIFEST_DIR" \
        --output "$ELIGIBLE_EDF_MANIFEST" \
        --report "$ELIGIBILITY_REPORT" \
        --min-duration-hours "$MIN_DURATION_HOURS" \
        --min-eeg-electrodes "$MIN_ELECTRODES" \
        --studies "${STUDY_ARRAY[@]}"

    if [[ ! -s "$ELIGIBLE_EDF_MANIFEST" ]]; then
        touch "$DOWNLOAD_DONE_FILE"
        if [[ -n "$processor_pid" ]]; then
            wait "$processor_pid" || true
        fi
        echo "No eligible local PSG recordings were found; processing stopped." >&2
        return 1
    fi

    echo
    echo "Downloads are complete; waiting for the live processing queue to drain..."
    touch "$DOWNLOAD_DONE_FILE"
    if ! wait "$processor_pid"; then
        echo "Live PSG processing failed; cache generation will not start." >&2
        return 1
    fi
    processor_pid=""

    echo
    echo "Eligible PSG processing finished. Refreshing HEP caches with script 17..."
    RUN_FOREGROUND=1 \
    HEP_TOTAL_WORKERS="$HEP_TOTAL_WORKERS" \
    HEP_MIN_EEG_CHANNELS="$MIN_ELECTRODES" \
        bash "$SCRIPT_17" --groups Harvard_Electroencephalography

    echo
    echo "Pipeline complete."
    echo "  Eligible EDF manifest: $ELIGIBLE_EDF_MANIFEST"
    echo "  Eligibility report   : $ELIGIBILITY_REPORT"
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
    'cd %q && STUDIES=%q MIN_ELECTRODES=%q MIN_DURATION_HOURS=%q ROOT_DEST=%q AWS_BIN=%q S3_ROOT=%q S3_BIDS=%q DRY_RUN=%q MAX_DOWNLOADS=%q USE_REGISTRY=%q REFRESH_REGISTRY=%q REGISTRY_VERSION=%q LOCK_DIR=%q REGISTRY_DIR=%q SHARDS_PER_STUDY=%q HEP_TOTAL_WORKERS=%q PROCESSING_WORKERS=%q LOG_DIR=%q MANIFEST_DIR=%q DIAGNOSIS_MANIFEST_DIR=%q ELIGIBLE_EDF_MANIFEST=%q ELIGIBILITY_REPORT=%q DOWNLOAD_DONE_FILE=%q PROCESSING_LOG=%q bash %q --pipeline-worker 2>&1 | tee -a %q' \
    "$SCRIPT_DIR" "$STUDIES" "$MIN_ELECTRODES" "$MIN_DURATION_HOURS" \
    "$BIDS_ROOT" "$AWS_BIN" "$S3_ROOT" "$S3_BIDS" "$DRY_RUN" \
    "$MAX_DOWNLOADS" "$USE_REGISTRY" "$REFRESH_REGISTRY" "$REGISTRY_VERSION" \
    "$LOCK_DIR" "$REGISTRY_DIR" "$SHARDS_PER_STUDY" "$HEP_TOTAL_WORKERS" \
    "$PROCESSING_WORKERS" "$LOG_DIR" \
    "$MANIFEST_DIR" "$DIAGNOSIS_MANIFEST_DIR" "$ELIGIBLE_EDF_MANIFEST" \
    "$ELIGIBILITY_REPORT" "$DOWNLOAD_DONE_FILE" "$PROCESSING_LOG" "$SCRIPT_PATH" \
    "${LOG_DIR}/pipeline.log"

tmux new-session -d -s "$SESSION" -n pipeline "$worker_command"

echo "Download-and-process pipeline started."
echo "  Studies : $STUDIES"
echo "  Eligible: diagnosed, >=${MIN_DURATION_HOURS}h, >=${MIN_ELECTRODES} EEG electrodes"
echo "  Shards  : $SHARDS_PER_STUDY per study"
echo "  HEP CPUs: $HEP_TOTAL_WORKERS (script 17)"
echo "  Process : $PROCESSING_WORKERS live PSG workers during download"
echo "  Session : $SESSION"
echo "  Attach  : tmux attach -t $SESSION"
echo "  Log     : ${LOG_DIR}/pipeline.log"
