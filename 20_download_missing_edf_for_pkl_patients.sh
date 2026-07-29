#!/usr/bin/env bash
set -euo pipefail

# Re-download the .edf files for every Harvard patient we already have a
# >10-electrode pickle for (pickles_sleep_stage/Harvard_Electroencephalography),
# but whose local BIDS session is missing the .edf (it was deleted after the
# .h5/pkl conversion to save space). Only the missing .edf is fetched; every
# other already-present file in the session is left untouched.
#
# Usage:
#   ./20_download_missing_edf_for_pkl_patients.sh
#   DRY_RUN=1 ./20_download_missing_edf_for_pkl_patients.sh
#   PARALLEL_JOBS=8 ./20_download_missing_edf_for_pkl_patients.sh

AWS_BIN="${AWS_BIN:-/storage/pblab_shared_data/Nir/bin/aws}"
S3_ROOT="${S3_ROOT:-s3://arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-credentialed-access-point}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PICKLE_DIR="${PICKLE_DIR:-${SCRIPT_DIR}/pickles_sleep_stage/Harvard_Electroencephalography}"
BIDS_ROOT="${BIDS_ROOT:-${SCRIPT_DIR}/EDF_Format/Harvard_Electroencephalography/bids}"
DRY_RUN="${DRY_RUN:-0}"
PARALLEL_JOBS="${PARALLEL_JOBS:-$(( $(nproc) / 2 > 0 ? $(nproc) / 2 : 1 ))}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/20_download_missing_edf}"

mkdir -p "$LOG_DIR"

download_edf_for_subject() {
    local subject="$1"
    local study="${subject:0:5}"
    local subject_dir="${BIDS_ROOT}/${study}/sub-${subject}"

    if [[ ! -d "$subject_dir" ]]; then
        echo "SKIP ${subject}: no local BIDS subject dir at ${subject_dir}"
        return
    fi

    local ses_dir
    for ses_dir in "$subject_dir"/ses-*; do
        [[ -d "$ses_dir" ]] || continue
        local eeg_dir="${ses_dir}/eeg"
        [[ -d "$eeg_dir" ]] || continue

        if compgen -G "${eeg_dir}/*.edf" > /dev/null; then
            continue
        fi

        local channels_file
        channels_file="$(find "$eeg_dir" -iname '*_task-PSG_channels.tsv' -print -quit)"
        if [[ -z "$channels_file" ]]; then
            echo "SKIP ${subject}/$(basename "$ses_dir"): no *_task-PSG_channels.tsv to derive edf name"
            continue
        fi

        local edf_name
        edf_name="$(basename "${channels_file/_task-PSG_channels.tsv/_task-PSG_eeg.edf}")"
        local rel_key="${eeg_dir#${BIDS_ROOT}/}"
        local s3_key="PSG/bids/${rel_key}/${edf_name}"

        if [[ "$DRY_RUN" == "1" ]]; then
            echo "WOULD DOWNLOAD ${subject}/$(basename "$ses_dir"): ${s3_key}"
            continue
        fi

        echo "DOWNLOAD ${subject}/$(basename "$ses_dir"): ${edf_name}"
        "$AWS_BIN" s3 cp "${S3_ROOT}/${s3_key}" "${eeg_dir}/${edf_name}" \
            --request-payer requester \
            --no-cli-pager
    done
}
export -f download_edf_for_subject
export AWS_BIN S3_ROOT BIDS_ROOT DRY_RUN

find "$PICKLE_DIR" -iname 'SUB-*.pkl' -printf '%f\n' \
    | sed -E 's/^SUB-([A-Za-z0-9]+)_.*/\1/' \
    | sort -u \
    | xargs -P "$PARALLEL_JOBS" -I{} bash -c 'download_edf_for_subject "$@"' _ {} \
    2>&1 | tee -a "${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
