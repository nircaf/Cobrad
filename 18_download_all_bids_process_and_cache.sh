#!/usr/bin/env bash
set -euo pipefail

# Download every configured Harvard BIDS study, process the downloaded patients,
# and refresh both HEP caches. The pipeline runs in tmux and continues after
# logout.
#
# Usage:
#   ./18_download_all_bids_process_and_cache.sh
#   STUDIES="I0002 I0003 I0004 I0006" ./18_download_all_bids_process_and_cache.sh
#   SHARDS_PER_STUDY=4 HEP_TOTAL_WORKERS=24 ./18_download_all_bids_process_and_cache.sh
#   ./18_download_all_bids_process_and_cache.sh --force
#
# Monitor:
#   tmux attach -t harvard-bids-full-pipeline

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${TMUX_SESSION:-harvard-bids-full-pipeline}"
STUDIES="${STUDIES:-I0002 I0003 I0004 I0006}"
SHARDS_PER_STUDY="${SHARDS_PER_STUDY:-4}"
EDF_FOLDER="${EDF_FOLDER:-Harvard_Electroencephalography}"
EDF_ROOT="${EDF_ROOT:-EDF_Format/${EDF_FOLDER}}"
DOWNLOADER="${SCRIPT_DIR}/download_new_harvard_bids_patients_gt10_electrodes.sh"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/harvard_bids_full_pipeline}"
AVAILABLE_CPUS="$(nproc)"
TOTAL_WORKERS="${HEP_TOTAL_WORKERS:-$(( AVAILABLE_CPUS / 4 ))}"

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not available on PATH." >&2
    exit 1
fi
if [[ ! -f "$DOWNLOADER" ]]; then
    echo "Downloader not found: $DOWNLOADER" >&2
    exit 1
fi
if [[ ! -f "${SCRIPT_DIR}/parallel_patient_processing.py" ]]; then
    echo "Patient processor not found: ${SCRIPT_DIR}/parallel_patient_processing.py" >&2
    exit 1
fi
if [[ ! -f "${SCRIPT_DIR}/17_generate_hep_cache.py" ]]; then
    echo "Cache generator not found: ${SCRIPT_DIR}/17_generate_hep_cache.py" >&2
    exit 1
fi
if [[ ! -f "${SCRIPT_DIR}/venv/bin/activate" ]]; then
    echo "Virtual environment not found: ${SCRIPT_DIR}/venv/bin/activate" >&2
    exit 1
fi
if ! [[ "$SHARDS_PER_STUDY" =~ ^[1-9][0-9]*$ ]]; then
    echo "SHARDS_PER_STUDY must be a positive integer." >&2
    exit 2
fi
if ! [[ "$TOTAL_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "HEP_TOTAL_WORKERS must be a positive integer." >&2
    exit 2
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists."
    echo "  Attach: tmux attach -t $SESSION"
    echo "  Kill  : tmux kill-session -t $SESSION"
    exit 0
fi

(( TOTAL_WORKERS >= 2 )) || TOTAL_WORKERS=2
(( TOTAL_WORKERS <= AVAILABLE_CPUS )) || TOTAL_WORKERS="$AVAILABLE_CPUS"
NORMAL_WORKERS=$(( TOTAL_WORKERS * 3 / 4 ))
COHORT_WORKERS=$(( TOTAL_WORKERS - NORMAL_WORKERS ))
(( NORMAL_WORKERS >= 1 )) || NORMAL_WORKERS=1
(( COHORT_WORKERS >= 1 )) || COHORT_WORKERS=1

CACHE_ARGS=()
for argument in "$@"; do
    CACHE_ARGS+=("$argument")
done

printf -v Q_SCRIPT_DIR '%q' "$SCRIPT_DIR"
printf -v Q_DOWNLOADER '%q' "$DOWNLOADER"
printf -v Q_EDF_ROOT '%q' "$EDF_ROOT"
printf -v Q_SHARDS '%q' "$SHARDS_PER_STUDY"
printf -v Q_NORMAL_WORKERS '%q' "$NORMAL_WORKERS"
printf -v Q_COHORT_WORKERS '%q' "$COHORT_WORKERS"
printf -v Q_LOG_DIR '%q' "$LOG_DIR"
Q_STUDY_WORDS=""
while IFS= read -r study; do
    [[ -n "$study" ]] || continue
    printf -v Q_STUDY_WORDS '%s %q' "$Q_STUDY_WORDS" "$study"
done < <(xargs -n1 <<< "$STUDIES")

Q_CACHE_ARGS=""
if (( ${#CACHE_ARGS[@]} > 0 )); then
    printf -v Q_CACHE_ARGS ' %q' "${CACHE_ARGS[@]}"
fi

PIPELINE_CMD=$(cat <<EOF
set -euo pipefail
cd ${Q_SCRIPT_DIR}
mkdir -p ${Q_LOG_DIR}
exec > >(tee -a ${Q_LOG_DIR}/pipeline.log) 2>&1

echo "=== Stage 1/3: download all configured BIDS studies ==="
pids=()
for study in${Q_STUDY_WORDS}; do
    for ((shard=0; shard<${Q_SHARDS}; shard++)); do
        RUN_WORKER=1 STUDY="\$study" SHARDS_PER_STUDY=${Q_SHARDS} SHARD_INDEX="\$shard" \
            bash ${Q_DOWNLOADER} > >(tee -a "${Q_LOG_DIR}/download-\${study}-\${shard}.log") 2>&1 &
        pids+=("\$!")
    done
done
download_status=0
for pid in "\${pids[@]}"; do
    wait "\$pid" || download_status=1
done
if (( download_status != 0 )); then
    echo "One or more BIDS download workers failed." >&2
    exit 1
fi

echo "=== Stage 2/3: run patient processing (script 13 workflow) ==="
source venv/bin/activate
python parallel_patient_processing.py --edf_root ${Q_EDF_ROOT}

echo "=== Stage 3/3: generate HEP caches (script 17 workflow) ==="
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    python 17_generate_hep_cache.py${Q_CACHE_ARGS} --workers ${Q_NORMAL_WORKERS}
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    python 17_generate_hep_cache.py${Q_CACHE_ARGS} --min-eeg-channels 10 --workers ${Q_COHORT_WORKERS}

echo "=== Full BIDS pipeline completed successfully ==="
EOF
)

mkdir -p "$LOG_DIR"
tmux new-session -d -s "$SESSION" -n pipeline -x 220 -y 50
tmux send-keys -t "$SESSION:pipeline" "$PIPELINE_CMD" Enter

echo "Full Harvard BIDS pipeline started in tmux session '$SESSION'."
echo "  Studies : $STUDIES"
echo "  Download: $SHARDS_PER_STUDY workers per study"
echo "  HEP CPUs: $TOTAL_WORKERS total ($NORMAL_WORKERS normal, $COHORT_WORKERS min10)"
echo "  Logs    : $LOG_DIR"
echo "  Attach  : tmux attach -t $SESSION"
