#!/usr/bin/env bash
set -euo pipefail

# Download diagnosed Harvard BIDS PSG recordings that are at least the requested
# duration and contain the requested number of real 10-20 scalp EEG electrodes.
# Physiology channels such as ECG/EKG, EMG, CHIN, CO2, SaO2, flow, belts,
# snore, etc. do not count even if BIDS labels them as EEG.
#
# Usage:
#   ./download_new_harvard_bids_patients_gt10_electrodes.sh start-tmux
#   DRY_RUN=1 ./download_new_harvard_bids_patients_gt10_electrodes.sh start-tmux
#   STUDIES="I0002 I0003" ./download_new_harvard_bids_patients_gt10_electrodes.sh start-tmux
#   SHARDS_PER_STUDY=4 ./download_new_harvard_bids_patients_gt10_electrodes.sh start-tmux
#   REFRESH_REGISTRY=1 ./download_new_harvard_bids_patients_gt10_electrodes.sh start-tmux
#
# Single-study worker/debug mode:
#   RUN_WORKER=1 STUDY=I0002 ./download_new_harvard_bids_patients_gt10_electrodes.sh
#   RUN_WORKER=1 STUDY=I0002 SHARDS_PER_STUDY=4 SHARD_INDEX=0 ./download_new_harvard_bids_patients_gt10_electrodes.sh

AWS_BIN="${AWS_BIN:-/storage/pblab_shared_data/Nir/bin/aws}"
S3_ROOT="${S3_ROOT:-s3://arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-credentialed-access-point}"
S3_BIDS="${S3_BIDS:-${S3_ROOT}/PSG/bids}"
STUDY="${STUDY:-I0002}"
ROOT_DEST="${ROOT_DEST:-/storage/pblab_shared_data2/Nir/Cobrad/EDF_Format/Harvard_Electroencephalography/bids}"
DEST="${DEST:-${ROOT_DEST}/${STUDY}}"
MIN_ELECTRODES="${MIN_ELECTRODES:-5}"
MIN_DURATION_HOURS="${MIN_DURATION_HOURS:-7}"
DRY_RUN="${DRY_RUN:-0}"
MAX_DOWNLOADS="${MAX_DOWNLOADS:-0}"
STUDIES="${STUDIES:-I0002 I0003 I0004 I0006}"
TMUX_SESSION="${TMUX_SESSION:-harvard-bids-downloads}"
RUN_WORKER="${RUN_WORKER:-0}"
SHARDS_PER_STUDY="${SHARDS_PER_STUDY:-4}"
SHARD_INDEX="${SHARD_INDEX:-0}"
USE_REGISTRY="${USE_REGISTRY:-1}"
REFRESH_REGISTRY="${REFRESH_REGISTRY:-0}"
REGISTRY_VERSION="${REGISTRY_VERSION:-3}"

SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
ELIGIBILITY_HELPER="${SCRIPT_DIR}/18_bids_psg_eligibility.py"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/venv/bin/python}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/harvard_bids_downloads}"
LOCK_DIR="${LOCK_DIR:-${ROOT_DEST}/.download_locks}"
DIAGNOSIS_MANIFEST_DIR="${DIAGNOSIS_MANIFEST_DIR:-${ROOT_DEST}/.diagnosed_subjects}"
DIAGNOSIS_MANIFEST="${DIAGNOSIS_MANIFEST:-${DIAGNOSIS_MANIFEST_DIR}/${STUDY}.txt}"
PROCESSING_MANIFEST="${PROCESSING_MANIFEST:-}"
REGISTRY_DIR="${REGISTRY_DIR:-${ROOT_DEST}/.checked_subjects/v${REGISTRY_VERSION}/min-${MIN_ELECTRODES}_hours-${MIN_DURATION_HOURS}}"
S3_ROOT_WITHOUT_SCHEME="${S3_ROOT#s3://}"
if [[ "$S3_ROOT_WITHOUT_SCHEME" == arn:* ]]; then
    # An S3 access-point ARN contains a slash before its access-point name;
    # that slash is part of the bucket identifier and must not be truncated.
    S3API_BUCKET="$S3_ROOT_WITHOUT_SCHEME"
else
    S3API_BUCKET="${S3_ROOT_WITHOUT_SCHEME%%/*}"
fi

unique_studies() {
    awk '
        {
            for (i = 1; i <= NF; i++) {
                if (!seen[$i]++) print $i
            }
        }
    ' <<< "$STUDIES"
}

start_tmux_downloads() {
    local -a configured_studies=()
    if ! command -v tmux >/dev/null 2>&1; then
        echo "tmux is not available on PATH." >&2
        exit 1
    fi
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "tmux session '${TMUX_SESSION}' already exists."
        echo "Attach: tmux attach -t ${TMUX_SESSION}"
        echo "Kill  : tmux kill-session -t ${TMUX_SESSION}"
        exit 0
    fi

    mkdir -p "$LOG_DIR"
    mapfile -t configured_studies < <(unique_studies)
    "$PYTHON_BIN" "$ELIGIBILITY_HELPER" diagnosis-manifests \
        --output-dir "$DIAGNOSIS_MANIFEST_DIR" \
        --studies "${configured_studies[@]}"

    if ! [[ "$SHARDS_PER_STUDY" =~ ^[0-9]+$ ]] || (( SHARDS_PER_STUDY < 1 )); then
        echo "SHARDS_PER_STUDY must be a positive integer." >&2
        exit 1
    fi

    local first_window=1
    local study
    local shard
    local window_name
    while IFS= read -r study; do
        [[ -n "$study" ]] || continue
        for (( shard = 0; shard < SHARDS_PER_STUDY; shard++ )); do
            window_name="${study}-${shard}"
            if (( first_window )); then
                tmux new-session -d -s "$TMUX_SESSION" -n "$window_name"
                first_window=0
            else
                tmux new-window -t "$TMUX_SESSION" -n "$window_name"
            fi

            local worker_cmd
            worker_cmd=$(
                printf 'cd %q && set -o pipefail && RUN_WORKER=1 STUDY=%q SHARDS_PER_STUDY=%q SHARD_INDEX=%q AWS_BIN=%q S3_ROOT=%q S3_BIDS=%q ROOT_DEST=%q MIN_ELECTRODES=%q MIN_DURATION_HOURS=%q DIAGNOSIS_MANIFEST_DIR=%q PROCESSING_MANIFEST=%q DRY_RUN=%q MAX_DOWNLOADS=%q LOCK_DIR=%q USE_REGISTRY=%q REFRESH_REGISTRY=%q REGISTRY_VERSION=%q REGISTRY_DIR=%q bash %q 2>&1 | tee -a %q; echo; echo "--- %s shard %s worker done ---"; read -r -p "Press Enter to close..."' \
                    "$SCRIPT_DIR" "$study" "$SHARDS_PER_STUDY" "$shard" \
                    "$AWS_BIN" "$S3_ROOT" "$S3_BIDS" "$ROOT_DEST" \
                    "$MIN_ELECTRODES" "$MIN_DURATION_HOURS" "$DIAGNOSIS_MANIFEST_DIR" \
                    "$PROCESSING_MANIFEST" "$DRY_RUN" "$MAX_DOWNLOADS" "$LOCK_DIR" \
                    "$USE_REGISTRY" "$REFRESH_REGISTRY" "$REGISTRY_VERSION" "$REGISTRY_DIR" \
                    "$SCRIPT_PATH" "${LOG_DIR}/${window_name}.log" "$study" "$shard"
            )
            tmux send-keys -t "${TMUX_SESSION}:${window_name}" "$worker_cmd" Enter
        done
    done < <(unique_studies)

    echo "Started parallel Harvard BIDS downloads in tmux session '${TMUX_SESSION}'."
    echo "Studies: $(unique_studies | paste -sd ' ' -)"
    echo "Shards per study: ${SHARDS_PER_STUDY}"
    echo "Logs: ${LOG_DIR}/"
    echo "Attach: tmux attach -t ${TMUX_SESSION}"
    echo "Detach after attaching: Ctrl+B then D"
}

case "${1:-}" in
    start-tmux|tmux|start)
        start_tmux_downloads
        exit 0
        ;;
esac

mkdir -p "$DEST"
mkdir -p "$LOCK_DIR"
mkdir -p "${REGISTRY_DIR}/${STUDY}"
TMP_DIR="$(mktemp -d)"
CURRENT_LOCK=""

if [[ ! -s "$DIAGNOSIS_MANIFEST" ]]; then
    echo "Diagnosed-subject manifest is missing or empty: $DIAGNOSIS_MANIFEST" >&2
    echo "Run 18_download_all_bids_and_run_17_13.sh to generate it." >&2
    exit 1
fi

cleanup() {
    if [[ -n "$CURRENT_LOCK" ]]; then
        rm -rf "$CURRENT_LOCK"
    fi
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

downloaded=0
eligible=0
skipped_existing=0
skipped_too_few_electrodes=0
skipped_missing_channels=0
skipped_not_diagnosed=0
skipped_short_recording=0
skipped_duration_unavailable=0
skipped_locked=0
registry_hits=0
registry_writes=0

if ! [[ "$MIN_ELECTRODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "MIN_ELECTRODES must be a positive integer." >&2
    exit 1
fi
if ! awk -v value="$MIN_DURATION_HOURS" \
    'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value > 0) }'; then
    echo "MIN_DURATION_HOURS must be a positive number." >&2
    exit 1
fi
if ! [[ "$SHARDS_PER_STUDY" =~ ^[0-9]+$ ]] || (( SHARDS_PER_STUDY < 1 )); then
    echo "SHARDS_PER_STUDY must be a positive integer." >&2
    exit 1
fi
if ! [[ "$SHARD_INDEX" =~ ^[0-9]+$ ]] || (( SHARD_INDEX < 0 || SHARD_INDEX >= SHARDS_PER_STUDY )); then
    echo "SHARD_INDEX must be an integer in [0, SHARDS_PER_STUDY)." >&2
    exit 1
fi

scalp_eeg_summary() {
    awk -F '\t' '
        BEGIN {
            split("Fp1,Fp2,F7,F3,Fz,F4,F8,T3,T7,C3,Cz,C4,T4,T8,T5,P7,P3,Pz,P4,T6,P8,O1,Oz,O2", ordered, ",")
            montage_count = 24
            for (i = 1; i <= montage_count; i++) {
                montage[toupper(ordered[i])] = ordered[i]
            }
            split("ECG,EKG,EMG,EOG,CHIN,SNORE,NPT,C-FLOW,CHEST,ABDOMINAL,LAT,RAT,SAO2,PLETH,SENTEC-TC,C PRESS,RR,ETCO2,THERM,FLOW,AIRFLOW,THOR,ABD,POSITION,BODY", physio, ",")
            for (i in physio) physio_like[physio[i]] = 1

        }
        NR == 1 {
            for (i = 1; i <= NF; i++) {
                column = tolower($i)
                gsub(/\r/, "", column)
                if (column == "name") name_col = i
                if (column == "type") type_col = i
                if (column == "status") status_col = i
            }
            next
        }
        {
            name_value = name_col ? $name_col : $1
            status_value = status_col ? tolower($status_col) : "good"
            gsub(/\r/, "", name_value)
            gsub(/\r/, "", status_value)
            if (status_value == "bad") next

            normalized = normalize_channel(name_value)
            if (normalized != "") found[normalized] = 1
        }

        function normalize_channel(value, raw, upper_raw, split_parts, candidates, n_candidates, candidate, key, i) {
            raw = value
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", raw)
            sub(/^EEG[[:space:]]+/, "", raw)
            upper_raw = toupper(raw)
            if (upper_raw in physio_like) return ""

            n_candidates = 1
            candidates[1] = raw
            if (index(raw, "-") > 0) {
                split(raw, split_parts, "-")
                candidates[++n_candidates] = split_parts[1]
            }
            if (index(raw, "_") > 0) {
                split(raw, split_parts, "_")
                candidates[++n_candidates] = split_parts[1]
            }
            if (index(raw, " ") > 0) {
                split(raw, split_parts, " ")
                candidates[++n_candidates] = split_parts[1]
            }
            for (i = 1; i <= n_candidates; i++) {
                candidate = candidates[i]
                gsub(/[^A-Za-z0-9]/, "", candidate)
                key = toupper(candidate)
                if (key in montage) return montage[key]
            }
            return ""
        }

        END {
            count = 0
            electrode_list = ""
            for (i = 1; i <= montage_count; i++) {
                electrode = ordered[i]
                if (electrode in found) {
                    count++
                    electrode_list = electrode_list (electrode_list == "" ? "" : ",") electrode
                }
            }

            print count "\t" electrode_list
        }
    '
}

remote_subjects() {
    "$AWS_BIN" s3 ls "${S3_BIDS}/${STUDY}/" \
        --request-payer requester \
        --no-cli-pager |
        awk '/PRE[[:space:]]+sub-/ { subject=$2; sub(/\/$/, "", subject); print subject }'
}

shard_subjects() {
    awk -v shards="$SHARDS_PER_STUDY" -v shard="$SHARD_INDEX" '
        (NR - 1) % shards == shard { print }
    '
}

channel_keys_for_subject() {
    local subject="$1"
    "$AWS_BIN" s3 ls "${S3_BIDS}/${STUDY}/${subject}/" \
        --recursive \
        --request-payer requester \
        --no-cli-pager |
        awk '$4 ~ /_task-PSG_channels.tsv$/ { print $4 }'
}

edf_duration_seconds_for_key() {
    local edf_key="$1"
    local header_file
    local n_records
    local record_seconds
    local duration_seconds

    header_file="$(mktemp "${TMP_DIR}/edf_header.XXXXXX")"
    if ! "$AWS_BIN" s3api get-object \
        --bucket "$S3API_BUCKET" \
        --key "$edf_key" \
        --range "bytes=0-255" \
        "$header_file" \
        --request-payer requester \
        --no-cli-pager >/dev/null; then
        rm -f "$header_file"
        echo "Failed to read remote EDF header: s3://${S3API_BUCKET}/${edf_key}" >&2
        return 2
    fi

    n_records="$(
        dd if="$header_file" bs=1 skip=236 count=8 status=none |
            tr -d '[:space:]'
    )"
    record_seconds="$(
        dd if="$header_file" bs=1 skip=244 count=8 status=none |
            tr -d '[:space:]'
    )"
    rm -f "$header_file"

    if ! duration_seconds="$(
        awk -v records="$n_records" -v seconds="$record_seconds" '
            BEGIN {
                if (records !~ /^-?[0-9]+([.][0-9]+)?$/ ||
                    seconds !~ /^[0-9]+([.][0-9]+)?$/ ||
                    records < 0 || seconds <= 0) {
                    exit 1
                }
                printf "%.6f", records * seconds
            }
        '
    )"; then
        return 1
    fi
    printf '%s\n' "$duration_seconds"
}

qualifying_sessions_for_subject() {
    local subject="$1"
    local max_count=0
    local best_electrodes=""
    local found_channels=0
    local channel_key
    local summary
    local count
    local electrodes
    local session
    local edf_key
    local duration_seconds
    local duration_status
    local duration_hours
    local qualified_sessions=0
    local channel_keys_file="${TMP_DIR}/${subject}_channel_keys.txt"

    if ! channel_keys_for_subject "$subject" > "$channel_keys_file"; then
        echo "Failed to list PSG channel metadata for ${subject}." >&2
        return 1
    fi
    while IFS= read -r channel_key; do
        [[ -n "$channel_key" ]] || continue
        found_channels=1
        summary="$(
            "$AWS_BIN" s3 cp "${S3_ROOT}/${channel_key}" - \
                --request-payer requester \
                --no-cli-pager |
                scalp_eeg_summary
        )"
        IFS=$'\t' read -r count electrodes <<< "$summary"
        if (( count > max_count )); then
            max_count="$count"
            best_electrodes="$electrodes"
        fi
        session="$(awk -F '/' '{ for (i = 1; i <= NF; i++) if ($i ~ /^ses-/) { print $i; exit } }' <<< "$channel_key")"
        edf_key="${channel_key%_channels.tsv}_eeg.edf"
        if duration_seconds="$(edf_duration_seconds_for_key "$edf_key")"; then
            :
        else
            duration_status=$?
            if (( duration_status == 2 )); then
                return 1
            fi
            echo "DURATION_UNAVAILABLE"$'\t'"${session}"$'\t'"${edf_key}"
            continue
        fi
        duration_hours="$(awk -v seconds="$duration_seconds" 'BEGIN { printf "%.3f", seconds / 3600 }')"
        if ! awk -v seconds="$duration_seconds" -v hours="$MIN_DURATION_HOURS" \
            'BEGIN { exit !(seconds >= hours * 3600) }'; then
            echo "SHORT"$'\t'"${session}"$'\t'"${duration_hours}"$'\t'"${edf_key}"
            continue
        fi
        if (( count >= MIN_ELECTRODES )); then
            qualified_sessions=1
            echo "QUALIFIED"$'\t'"${session}"$'\t'"${count}"$'\t'"${electrodes}"$'\t'"${duration_seconds}"$'\t'"${edf_key}"$'\t'"${channel_key}"
        fi
    done < "$channel_keys_file"

    if (( found_channels == 0 )); then
        echo "SUMMARY"$'\t'"-1"$'\t'
    elif (( qualified_sessions == 0 )); then
        echo "SUMMARY"$'\t'"${max_count}"$'\t'"${best_electrodes}"
    fi
}

checked_result_for_subject() {
    local subject="$1"
    local output_file="$2"
    local registry_file="${REGISTRY_DIR}/${STUDY}/${subject}.tsv"

    if [[ "$USE_REGISTRY" == "1" && "$REFRESH_REGISTRY" != "1" && -s "$registry_file" ]]; then
        ((registry_hits += 1))
        echo "REGISTRY hit: ${subject}"
        cp "$registry_file" "$output_file"
        return 0
    fi

    qualifying_sessions_for_subject "$subject" > "$output_file"

    if [[ "$USE_REGISTRY" == "1" ]]; then
        # mv is atomic on this filesystem, so parallel shards never see a
        # partially written registry entry.
        cp "$output_file" "${registry_file}.tmp.$$"
        mv "${registry_file}.tmp.$$" "$registry_file"
        ((registry_writes += 1))
    fi
}

download_subject_level_files() {
    local subject="$1"
    local final_dest="${DEST}/${subject}"
    local partial_dest="${final_dest}.partial_subject_files"

    rm -rf "$partial_dest"
    mkdir -p "$partial_dest"

    "$AWS_BIN" s3 cp "${S3_BIDS}/${STUDY}/${subject}/" "$partial_dest/" \
        --recursive \
        --exclude "ses-*/*" \
        --request-payer requester \
        --no-cli-pager

    mkdir -p "$final_dest"
    cp -a "$partial_dest"/. "$final_dest"/
    rm -rf "$partial_dest"
}

download_session() {
    local subject="$1"
    local session="$2"
    local final_dest="${DEST}/${subject}/${session}"
    local partial_dest="${final_dest}.partial"

    rm -rf "$partial_dest"
    mkdir -p "$(dirname "$partial_dest")"

    "$AWS_BIN" s3 cp "${S3_BIDS}/${STUDY}/${subject}/${session}/" "$partial_dest/" \
        --recursive \
        --request-payer requester \
        --no-cli-pager

    mkdir -p "$final_dest"
    cp -a "$partial_dest"/. "$final_dest"/
    rm -rf "$partial_dest"
}

lock_subject() {
    local subject="$1"
    local lock_path="${LOCK_DIR}/${STUDY}_${subject}.lock"
    if mkdir "$lock_path" 2>/dev/null; then
        echo "$lock_path"
        return 0
    fi
    return 1
}

append_processing_edf() {
    local edf_path="$1"
    if [[ -z "$PROCESSING_MANIFEST" ]]; then
        return 0
    fi
    if [[ ! -s "$edf_path" ]]; then
        echo "Cannot queue missing downloaded EDF: $edf_path" >&2
        return 1
    fi
    mkdir -p "$(dirname "$PROCESSING_MANIFEST")"
    (
        flock -x 9
        printf '%s\n' "$(readlink -f "$edf_path")" >> "$PROCESSING_MANIFEST"
    ) 9>>"${PROCESSING_MANIFEST}.lock"
}

echo "Source: ${S3_BIDS}/${STUDY}/"
echo "Destination: ${DEST}"
echo "Minimum 10-20 scalp EEG electrodes: >= ${MIN_ELECTRODES}"
echo "Minimum PSG duration: >= ${MIN_DURATION_HOURS} hours"
echo "Diagnosed subjects: ${DIAGNOSIS_MANIFEST}"
echo "Shard: ${SHARD_INDEX}/${SHARDS_PER_STUDY}"
echo "Dry run: ${DRY_RUN}"
echo "Persistent check registry: ${USE_REGISTRY} (${REGISTRY_DIR}/${STUDY})"
echo "Refresh registry: ${REFRESH_REGISTRY}"
echo

remote_shard_list="${TMP_DIR}/remote_subjects_shard.txt"
subject_list="${TMP_DIR}/diagnosed_remote_subjects.txt"
remote_subjects | shard_subjects > "$remote_shard_list"
awk '
    NR == FNR { diagnosed[$0] = 1; next }
    $0 in diagnosed { print }
' "$DIAGNOSIS_MANIFEST" "$remote_shard_list" > "$subject_list"
remote_count="$(wc -l < "$remote_shard_list")"
diagnosed_remote_count="$(wc -l < "$subject_list")"
skipped_not_diagnosed=$((remote_count - diagnosed_remote_count))

while IFS= read -r subject; do
    [[ -n "$subject" ]] || continue

    lock_path=""
    if ! lock_path="$(lock_subject "$subject")"; then
        ((skipped_locked += 1))
        echo "SKIP locked by another worker: ${subject}"
        continue
    fi
    CURRENT_LOCK="$lock_path"

    qualifying_file="${TMP_DIR}/${subject}_qualifying_sessions.tsv"
    summary_file="${TMP_DIR}/${subject}_summary.tsv"
    short_file="${TMP_DIR}/${subject}_short.tsv"
    duration_unavailable_file="${TMP_DIR}/${subject}_duration_unavailable.tsv"
    checked_file="${TMP_DIR}/${subject}_checked.tsv"
    checked_result_for_subject "$subject" "$checked_file"
    awk -F '\t' \
        -v q="$qualifying_file" \
        -v s="$summary_file" \
        -v short="$short_file" \
        -v unavailable="$duration_unavailable_file" '
            $1 == "QUALIFIED" { print > q; next }
            $1 == "SUMMARY" { print > s; next }
            $1 == "SHORT" { print > short; next }
            $1 == "DURATION_UNAVAILABLE" { print > unavailable; next }
        ' "$checked_file"

    if [[ -s "$short_file" ]]; then
        short_count="$(wc -l < "$short_file")"
        ((skipped_short_recording += short_count))
        while IFS=$'\t' read -r _tag session duration_hours _edf_key; do
            echo "SKIP ${subject}/${session}: PSG duration ${duration_hours}h is below ${MIN_DURATION_HOURS}h"
        done < "$short_file"
    fi
    if [[ -s "$duration_unavailable_file" ]]; then
        unavailable_count="$(wc -l < "$duration_unavailable_file")"
        ((skipped_duration_unavailable += unavailable_count))
        while IFS=$'\t' read -r _tag session _edf_key; do
            echo "SKIP ${subject}/${session}: EDF duration unavailable"
        done < "$duration_unavailable_file"
    fi

    if [[ ! -s "$qualifying_file" ]]; then
        if [[ -s "$summary_file" ]]; then
            IFS=$'\t' read -r _summary_tag electrode_count electrodes < "$summary_file"
        else
            electrode_count="-1"
            electrodes=""
        fi
        if (( electrode_count < 0 )); then
            ((skipped_missing_channels += 1))
            if [[ ! -s "$duration_unavailable_file" && ! -s "$short_file" ]]; then
                echo "SKIP no PSG channels.tsv: ${subject}"
            fi
        elif (( electrode_count < MIN_ELECTRODES )); then
            ((skipped_too_few_electrodes += 1))
            echo "SKIP ${subject}: ${electrode_count} scalp EEG electrodes (${electrodes}); need ${MIN_ELECTRODES}"
        fi
        rm -rf "$lock_path"
        CURRENT_LOCK=""
        continue
    fi

    missing_sessions_file="${TMP_DIR}/${subject}_missing_sessions.tsv"
    : > "$missing_sessions_file"
    declare -A queued_sessions=()
    while IFS=$'\t' read -r _tag session electrode_count electrodes duration_seconds edf_key channel_key; do
        relative_edf="${edf_key#PSG/bids/"${STUDY}"/"${subject}"/"${session}"/}"
        local_edf="${DEST}/${subject}/${session}/${relative_edf}"
        existing_local_edf=""
        if [[ -d "$(dirname "$local_edf")" ]]; then
            existing_local_edf="$(
                find "$(dirname "$local_edf")" -maxdepth 1 -type f \
                    -iname "$(basename "$local_edf")" -print -quit
            )"
        fi
        if [[ -n "$existing_local_edf" && -s "$existing_local_edf" ]]; then
            echo "SKIP existing PSG: ${subject}/${session}/$(basename "$existing_local_edf")"
            append_processing_edf "$existing_local_edf"
            continue
        fi
        if [[ -n "${queued_sessions[$session]:-}" ]]; then
            continue
        fi
        queued_sessions["$session"]=1
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$_tag" "$session" "$electrode_count" "$electrodes" \
            "$duration_seconds" "$edf_key" "$channel_key" >> "$missing_sessions_file"
    done < "$qualifying_file"

    if [[ ! -s "$missing_sessions_file" ]]; then
        ((skipped_existing += 1))
        echo "SKIP existing qualifying sessions: ${subject}"
        rm -rf "$lock_path"
        CURRENT_LOCK=""
        continue
    fi

    ((eligible += 1))
    while IFS=$'\t' read -r _tag session electrode_count electrodes duration_seconds edf_key channel_key; do
        duration_hours="$(awk -v seconds="$duration_seconds" 'BEGIN { printf "%.3f", seconds / 3600 }')"
        echo "DOWNLOAD ${subject}/${session}: ${duration_hours}h PSG; ${electrode_count} scalp EEG electrodes (${electrodes})"
    done < "$missing_sessions_file"

    if [[ "$DRY_RUN" == "1" ]]; then
        rm -rf "$lock_path"
        CURRENT_LOCK=""
        if (( MAX_DOWNLOADS > 0 && eligible >= MAX_DOWNLOADS )); then
            echo "Reached MAX_DOWNLOADS=${MAX_DOWNLOADS}; stopping."
            break
        fi
        continue
    fi

    download_subject_level_files "$subject"
    while IFS=$'\t' read -r _tag session electrode_count electrodes duration_seconds edf_key channel_key; do
        download_session "$subject" "$session"
        # A session download is recursive and can fetch several qualifying PSG
        # EDFs (for example, segmented recordings). Queue every qualifying EDF
        # from the newly completed session, not only the row that triggered the
        # one-per-session download.
        while IFS=$'\t' read -r _qualified_tag qualified_session _qualified_count \
            _qualified_electrodes _qualified_duration qualified_edf_key _qualified_channel_key; do
            [[ "$qualified_session" == "$session" ]] || continue
            relative_edf="${qualified_edf_key#PSG/bids/"${STUDY}"/"${subject}"/"${session}"/}"
            local_edf="${DEST}/${subject}/${session}/${relative_edf}"
            append_processing_edf "$local_edf"
        done < "$qualifying_file"
    done < "$missing_sessions_file"
    rm -rf "$lock_path"
    CURRENT_LOCK=""
    ((downloaded += 1))

    if (( MAX_DOWNLOADS > 0 && downloaded >= MAX_DOWNLOADS )); then
        echo "Reached MAX_DOWNLOADS=${MAX_DOWNLOADS}; stopping."
        break
    fi
done < "$subject_list"

echo
echo "Done."
echo "  Downloaded subjects with missing qualifying sessions: ${downloaded}"
echo "  Eligible subjects with missing qualifying sessions: ${eligible}"
echo "  Skipped existing qualifying sessions: ${skipped_existing}"
echo "  Skipped locked by another worker: ${skipped_locked}"
echo "  Skipped subjects without diagnosis: ${skipped_not_diagnosed}"
echo "  Skipped PSG recordings shorter than ${MIN_DURATION_HOURS}h: ${skipped_short_recording}"
echo "  Skipped PSG recordings with unavailable duration: ${skipped_duration_unavailable}"
echo "  Skipped too few scalp EEG electrodes: ${skipped_too_few_electrodes}"
echo "  Skipped missing PSG channels.tsv: ${skipped_missing_channels}"
echo "  Reused registry entries: ${registry_hits}"
echo "  New/refreshed registry entries: ${registry_writes}"
