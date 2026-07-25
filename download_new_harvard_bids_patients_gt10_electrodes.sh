#!/usr/bin/env bash
set -euo pipefail

# Download new Harvard BIDS patients that match one of the local scalp
# EEG montage templates already seen in this folder. This uses real 10-20 EEG
# electrode names only; physiology channels such as ECG/EKG, EMG, CHIN, CO2,
# SaO2, flow, belts, snore, etc. do not count even if BIDS labels them as EEG.
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
MIN_ELECTRODES="${MIN_ELECTRODES:-10}"
DRY_RUN="${DRY_RUN:-0}"
MAX_DOWNLOADS="${MAX_DOWNLOADS:-0}"
STUDIES="${STUDIES:-I0002 I0003 I0004 I0006}"
TMUX_SESSION="${TMUX_SESSION:-harvard-bids-downloads}"
RUN_WORKER="${RUN_WORKER:-0}"
SHARDS_PER_STUDY="${SHARDS_PER_STUDY:-4}"
SHARD_INDEX="${SHARD_INDEX:-0}"
USE_REGISTRY="${USE_REGISTRY:-1}"
REFRESH_REGISTRY="${REFRESH_REGISTRY:-0}"
REGISTRY_VERSION="${REGISTRY_VERSION:-1}"

SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/harvard_bids_downloads}"
LOCK_DIR="${LOCK_DIR:-${ROOT_DEST}/.download_locks}"
REGISTRY_DIR="${REGISTRY_DIR:-${ROOT_DEST}/.checked_subjects/v${REGISTRY_VERSION}/min-${MIN_ELECTRODES}}"

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
                printf 'cd %q && set -o pipefail && RUN_WORKER=1 STUDY=%q SHARDS_PER_STUDY=%q SHARD_INDEX=%q AWS_BIN=%q S3_ROOT=%q S3_BIDS=%q ROOT_DEST=%q MIN_ELECTRODES=%q DRY_RUN=%q MAX_DOWNLOADS=%q LOCK_DIR=%q USE_REGISTRY=%q REFRESH_REGISTRY=%q REGISTRY_VERSION=%q REGISTRY_DIR=%q bash %q 2>&1 | tee -a %q; echo; echo "--- %s shard %s worker done ---"; read -r -p "Press Enter to close..."' \
                    "$SCRIPT_DIR" "$study" "$SHARDS_PER_STUDY" "$shard" \
                    "$AWS_BIN" "$S3_ROOT" "$S3_BIDS" "$ROOT_DEST" \
                    "$MIN_ELECTRODES" "$DRY_RUN" "$MAX_DOWNLOADS" "$LOCK_DIR" \
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
skipped_no_matching_montage=0
skipped_missing_channels=0
skipped_locked=0
registry_hits=0
registry_writes=0

if ! [[ "$SHARDS_PER_STUDY" =~ ^[0-9]+$ ]] || (( SHARDS_PER_STUDY < 1 )); then
    echo "SHARDS_PER_STUDY must be a positive integer." >&2
    exit 1
fi
if ! [[ "$SHARD_INDEX" =~ ^[0-9]+$ ]] || (( SHARD_INDEX < 0 || SHARD_INDEX >= SHARDS_PER_STUDY )); then
    echo "SHARD_INDEX must be an integer in [0, SHARDS_PER_STUDY)." >&2
    exit 1
fi

matching_1020_montage_summary() {
    awk -F '\t' '
        BEGIN {
            split("Fp1,Fp2,F7,F3,Fz,F4,F8,T3,T7,C3,Cz,C4,T4,T8,T5,P7,P3,Pz,P4,T6,P8,O1,Oz,O2", ordered, ",")
            montage_count = 24
            for (i = 1; i <= montage_count; i++) {
                montage[toupper(ordered[i])] = ordered[i]
            }
            split("ECG,EKG,EMG,EOG,CHIN,SNORE,NPT,C-FLOW,CHEST,ABDOMINAL,LAT,RAT,SAO2,PLETH,SENTEC-TC,C PRESS,RR,ETCO2,THERM,FLOW,AIRFLOW,THOR,ABD,POSITION,BODY", physio, ",")
            for (i in physio) physio_like[physio[i]] = 1

            template_count = 9
            templates[1] = "F3,F4,T3,C3,C4,T4,T5,P3,P4,T6"
            templates[2] = "F7,F3,F8,T3,C3,T4,T5,T6,O1,O2"
            templates[3] = "Fp1,Fp2,F7,F3,F8,T3,C3,C4,T4,P3"
            templates[4] = "F3,F4,T3,C3,C4,T4,T5,T6,O1,O2"
            templates[5] = "F7,F8,T3,C3,C4,T4,T5,P3,P4,T6"
            templates[6] = "F3,F4,F8,T3,C3,C4,T4,T5,T6,O1,O2"
            templates[7] = "F7,F3,F4,F8,T3,C3,C4,T4,T6,O1,O2"
            templates[8] = "F7,F3,F4,F8,T3,C3,C4,T4,T5,T6,O1,O2"
            templates[9] = "Fp1,Fp2,F7,F3,F4,F8,T3,C3,C4,T4,T5,P3,P4,T6"
            for (i = 1; i <= template_count; i++) {
                n = split(templates[i], parts, ",")
                template_size[i] = n
                for (j = 1; j <= n; j++) template_member[i, parts[j]] = 1
            }
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

            best_template = ""
            best_size = 0
            for (i = 1; i <= template_count; i++) {
                matched = 1
                n = split(templates[i], parts, ",")
                for (j = 1; j <= n; j++) {
                    if (!(parts[j] in found)) {
                        matched = 0
                        break
                    }
                }
                if (matched && template_size[i] > best_size) {
                    best_size = template_size[i]
                    best_template = templates[i]
                }
            }
            if (best_size > 0) {
                print count "\t" electrode_list "\t" best_size "\t" best_template
            } else {
                print count "\t" electrode_list "\t0\t"
            }
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
        awk '$4 ~ /_channels.tsv$/ { print $4 }'
}

qualifying_sessions_for_subject() {
    local subject="$1"
    local max_count=0
    local best_electrodes=""
    local best_template_size=0
    local best_template=""
    local found_channels=0
    local channel_key
    local summary
    local count
    local electrodes
    local template_size
    local template
    local session
    local qualified_sessions=0

    while IFS= read -r channel_key; do
        [[ -n "$channel_key" ]] || continue
        found_channels=1
        summary="$(
            "$AWS_BIN" s3 cp "${S3_ROOT}/${channel_key}" - \
                --request-payer requester \
                --no-cli-pager |
                matching_1020_montage_summary
        )"
        IFS=$'\t' read -r count electrodes template_size template <<< "$summary"
        if (( count > max_count )); then
            max_count="$count"
            best_electrodes="$electrodes"
        fi
        if (( template_size > best_template_size )); then
            best_template_size="$template_size"
            best_template="$template"
        fi
        session="$(awk -F '/' '{ for (i = 1; i <= NF; i++) if ($i ~ /^ses-/) { print $i; exit } }' <<< "$channel_key")"
        if (( count >= MIN_ELECTRODES && template_size >= MIN_ELECTRODES )); then
            qualified_sessions=1
            echo "QUALIFIED"$'\t'"${session}"$'\t'"${count}"$'\t'"${electrodes}"$'\t'"${template_size}"$'\t'"${template}"$'\t'"${channel_key}"
        fi
    done < <(channel_keys_for_subject "$subject")

    if (( found_channels == 0 )); then
        echo "SUMMARY"$'\t'"-1"$'\t'$'\t'"0"$'\t'
    elif (( qualified_sessions == 0 )); then
        echo "SUMMARY"$'\t'"${max_count}"$'\t'"${best_electrodes}"$'\t'"${best_template_size}"$'\t'"${best_template}"
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

    mv "$partial_dest" "$final_dest"
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

echo "Source: ${S3_BIDS}/${STUDY}/"
echo "Destination: ${DEST}"
echo "Minimum 10-20 scalp EEG electrodes: >= ${MIN_ELECTRODES}"
echo "Must contain one existing local montage template: yes"
echo "Shard: ${SHARD_INDEX}/${SHARDS_PER_STUDY}"
echo "Dry run: ${DRY_RUN}"
echo "Persistent check registry: ${USE_REGISTRY} (${REGISTRY_DIR}/${STUDY})"
echo "Refresh registry: ${REFRESH_REGISTRY}"
echo

subject_list="${TMP_DIR}/remote_subjects.txt"
remote_subjects | shard_subjects > "$subject_list"

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
    checked_file="${TMP_DIR}/${subject}_checked.tsv"
    checked_result_for_subject "$subject" "$checked_file"
    awk -F '\t' -v q="$qualifying_file" -v s="$summary_file" '
            $1 == "QUALIFIED" { print > q; next }
            $1 == "SUMMARY" { print > s; next }
        ' "$checked_file"

    if [[ ! -s "$qualifying_file" ]]; then
        if [[ -s "$summary_file" ]]; then
            IFS=$'\t' read -r _summary_tag electrode_count electrodes template_size template < "$summary_file"
        else
            electrode_count="-1"
            electrodes=""
            template_size="0"
            template=""
        fi
        if (( electrode_count < 0 )); then
            ((skipped_missing_channels += 1))
            echo "SKIP no channels.tsv: ${subject}"
        else
            ((skipped_no_matching_montage += 1))
            if (( electrode_count < MIN_ELECTRODES )); then
                echo "SKIP ${subject}: ${electrode_count} scalp EEG electrodes (${electrodes})"
            else
                echo "SKIP ${subject}: ${electrode_count} scalp EEG electrodes, no matching local montage (${electrodes})"
            fi
        fi
        rm -rf "$lock_path"
        CURRENT_LOCK=""
        continue
    fi

    missing_sessions_file="${TMP_DIR}/${subject}_missing_sessions.tsv"
    : > "$missing_sessions_file"
    while IFS=$'\t' read -r _tag session electrode_count electrodes template_size template channel_key; do
        if [[ -d "${DEST}/${subject}/${session}" ]]; then
            echo "SKIP existing session: ${subject}/${session}"
            continue
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$_tag" "$session" "$electrode_count" "$electrodes" \
            "$template_size" "$template" "$channel_key" >> "$missing_sessions_file"
    done < "$qualifying_file"

    if [[ ! -s "$missing_sessions_file" ]]; then
        ((skipped_existing += 1))
        echo "SKIP existing qualifying sessions: ${subject}"
        rm -rf "$lock_path"
        CURRENT_LOCK=""
        continue
    fi

    ((eligible += 1))
    while IFS=$'\t' read -r _tag session electrode_count electrodes template_size template channel_key; do
        echo "DOWNLOAD ${subject}/${session}: ${electrode_count} scalp EEG electrodes; matched ${template_size}-electrode template (${template})"
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
    while IFS=$'\t' read -r _tag session electrode_count electrodes template_size template channel_key; do
        download_session "$subject" "$session"
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
echo "  Skipped no matching local scalp montage: ${skipped_no_matching_montage}"
echo "  Skipped missing channels.tsv: ${skipped_missing_channels}"
echo "  Reused registry entries: ${registry_hits}"
echo "  New/refreshed registry entries: ${registry_writes}"
