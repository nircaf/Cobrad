#!/usr/bin/env bash
set -euo pipefail

# Download all configured Harvard BIDS studies, process the downloaded EDFs
# (the work launched by script 13), and finally build both HEP caches (script 17).
#
# Usage:
#   ./18_download_process_all_harvard_bids.sh
#   STUDIES="I0002 I0003 I0004 I0006" ./18_download_process_all_harvard_bids.sh
#   SHARDS_PER_STUDY=4 HEP_TOTAL_WORKERS=32 ./18_download_process_all_harvard_bids.sh
#
# Monitor:
#   tmux attach -t harvard-bids-full-pipeline
#   tail -f logs/harvard_bids_full_pipeline/pipeline.log
#
# Re-running is safe: the downloader skips existing sessions, patient processing
# skips completed output, and the cache generator updates its existing caches.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
SESSION="${TMUX_SESSION:-harvard-bids-full-pipeline}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/harvard_bids_full_pipeline}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/pipeline.log}"

STUDIES="${STUDIES:-I0002 I0003 I0004 I0006}"
SHARDS_PER_STUDY="${SHARDS_PER_STUDY:-4}"
EDF_ROOT="${EDF_ROOT:-${SCRIPT_DIR}/EDF_Format/Harvard_Electroencephalography/bids}"
AVAILABLE_CPUS="$(nproc)"
TOTAL_WORKERS="${HEP_TOTAL_WORKERS:-$(( AVAILABLE_CPUS / 4 ))}"
(( TOTAL_WORKERS >= 2 )) || TOTAL_WORKERS=2
(( TOTAL_WORKERS <= AVAILABLE_CPUS )) || TOTAL_WORKERS="$AVAILABLE_CPUS"

run_pipeline() {
    cd "$SCRIPT_DIR"
    source venv/bin/activate

    echo "[$(date --iso-8601=seconds)] Phase 1/3: downloading Harvard BIDS studies: ${STUDIES}"
    download_pids=()
    download_labels=()
    study_list=()
    IFS=' ' read -r -a study_list <<< "$STUDIES"
    for study in "${study_list[@]}"; do
        for (( shard = 0; shard < SHARDS_PER_STUDY; shard++ )); do
            worker_log="${LOG_DIR}/download-${study}-${shard}.log"
            echo "Starting ${study}, shard ${shard}/${SHARDS_PER_STUDY}; log: ${worker_log}"
            RUN_WORKER=1 \
                STUDY="$study" \
                SHARDS_PER_STUDY="$SHARDS_PER_STUDY" \
                SHARD_INDEX="$shard" \
                bash "${SCRIPT_DIR}/download_new_harvard_bids_patients_gt10_electrodes.sh" \
                > >(tee -a "$worker_log") 2>&1 &
            download_pids+=("$!")
            download_labels+=("${study}-${shard}")
        done
    done

    failed=0
    for i in "${!download_pids[@]}"; do
        if ! wait "${download_pids[$i]}"; then
            echo "ERROR: download worker ${download_labels[$i]} failed." >&2
            failed=1
        fi
    done
    if (( failed )); then
        echo "At least one download failed; processing was not started." >&2
        return 1
    fi

    echo "[$(date --iso-8601=seconds)] Phase 2/3: running script 13 patient processing"
    python parallel_patient_processing.py --edf_root "$EDF_ROOT"

    echo "[$(date --iso-8601=seconds)] Phase 3/3: running script 17 cache generation"
    normal_workers=$(( TOTAL_WORKERS * 3 / 4 ))
    cohort_workers=$(( TOTAL_WORKERS - normal_workers ))
    (( normal_workers >= 1 )) || normal_workers=1
    (( cohort_workers >= 1 )) || cohort_workers=1

    thread_limits=(
        OMP_NUM_THREADS=1
        MKL_NUM_THREADS=1
        OPENBLAS_NUM_THREADS=1
        NUMEXPR_NUM_THREADS=1
    )
    env "${thread_limits[@]}" python 17_generate_hep_cache.py \
        --groups Harvard_Electroencephalography \
        --workers "$normal_workers" &
    normal_pid="$!"
    env "${thread_limits[@]}" python 17_generate_hep_cache.py \
        --groups Harvard_Electroencephalography \
        --min-eeg-channels 10 \
        --workers "$cohort_workers" &
    cohort_pid="$!"

    cache_failed=0
    wait "$normal_pid" || cache_failed=1
    wait "$cohort_pid" || cache_failed=1
    if (( cache_failed )); then
        echo "At least one cache-generation job failed." >&2
        return 1
    fi

    echo "[$(date --iso-8601=seconds)] Harvard BIDS pipeline completed successfully."
}

if [[ "${RUN_PIPELINE:-0}" == "1" ]]; then
    mkdir -p "$LOG_DIR"
    run_pipeline
    exit
fi

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not available on PATH." >&2
    exit 1
fi
if ! [[ "$SHARDS_PER_STUDY" =~ ^[1-9][0-9]*$ ]]; then
    echo "SHARDS_PER_STUDY must be a positive integer." >&2
    exit 2
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Pipeline session '$SESSION' is already running."
    echo "Attach: tmux attach -t $SESSION"
    exit 0
fi

mkdir -p "$LOG_DIR"
printf -v command \
    'RUN_PIPELINE=1 STUDIES=%q SHARDS_PER_STUDY=%q EDF_ROOT=%q HEP_TOTAL_WORKERS=%q LOG_DIR=%q LOG_FILE=%q bash %q 2>&1 | tee -a %q; status=${PIPESTATUS[0]}; echo; if (( status == 0 )); then echo "Pipeline finished successfully."; else echo "Pipeline failed with status ${status}."; fi; read -r -p "Press Enter to close..."; exit "$status"' \
    "$STUDIES" "$SHARDS_PER_STUDY" "$EDF_ROOT" "$TOTAL_WORKERS" \
    "$LOG_DIR" "$LOG_FILE" "$SCRIPT_PATH" "$LOG_FILE"

tmux new-session -d -s "$SESSION" -n pipeline -x 220 -y 50
tmux send-keys -t "$SESSION:pipeline" "$command" Enter

echo "Started the complete Harvard BIDS pipeline in tmux session '$SESSION'."
echo "  Studies : $STUDIES"
echo "  EDF root: $EDF_ROOT"
echo "  Shards  : $SHARDS_PER_STUDY per study"
echo "  Workers : $TOTAL_WORKERS total cache workers"
echo "  Log     : $LOG_FILE"
echo "  Attach  : tmux attach -t $SESSION"
