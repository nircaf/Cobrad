#!/bin/bash

# Ensure ~/.local/bin is in PATH (where zellij lives)
export PATH="$HOME/.local/bin:$PATH"

# ─── Config ────────────────────────────────────────────────────────────────────
ZELLIJ_SESSION="Berkeley_data" # Default fallback

# Parse session argument if provided
if [[ "$1" == "1" || "$1" == "EDF" ]]; then
    ZELLIJ_SESSION="EDF"
    shift
elif [[ "$1" == "2" || "$1" == "Berkeley_data" ]]; then
    ZELLIJ_SESSION="Berkeley_data"
    shift
elif [[ "$1" == "3" || "$1" == "Seeg" ]]; then
    ZELLIJ_SESSION="Seeg"
    shift
elif [[ -n "$1" && "$1" != -* && "$1" != "reset" && "$1" != "kill" ]]; then
    ZELLIJ_SESSION="$1"
    shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDF_ROOT="EDF_Format/$ZELLIJ_SESSION"   # switch to the new dataset folder EDF
STAGE_ORDER=(W N3 R N1 N2)

LAYOUT_FILE="$SCRIPT_DIR/logs/session_layout.kdl"

# ─── Reset / Kill ─────────────────────────────────────────────────────────────
if [[ "$1" == "--reset" || "$1" == "-reset" || "$1" == "-r" || "$1" == "reset" ]]; then
    echo "Reset request detected. Cleaning up..."
    timeout 5 zellij delete-session "$ZELLIJ_SESSION" --force 2>/dev/null || true
    pkill -9 -f "zellij.*$ZELLIJ_SESSION" 2>/dev/null || true
    rm -rf "$SCRIPT_DIR/logs"
    echo "Reset complete."
    exit 0
fi

if [[ "$1" == "--kill" || "$1" == "-k" || "$1" == "kill" ]]; then
    echo "Kill all sessions request detected..."
    # 'zellij delete-all-sessions --force --yes' deletes all state so it doesn't resurrect later
    timeout 5 zellij kill-all-sessions --yes 2>/dev/null || true
    timeout 5 zellij delete-all-sessions --force --yes 2>/dev/null || true
    echo "All Zellij sessions killed."
    exit 0
fi

mkdir -p "$SCRIPT_DIR/logs"

QUEUE_FILE="$SCRIPT_DIR/logs/file_queue.txt"
LOCK_FILE="$SCRIPT_DIR/logs/queue.lock"

# ─── Remove Duplicate Pickles ──────────────────────────────────────────────────
echo "Checking for duplicate pickles in pickles_sleep_stage/$ZELLIJ_SESSION..."
"$SCRIPT_DIR/venv/bin/python" -c '
import os, glob, sys
from collections import defaultdict

base_dir = sys.argv[1]
if os.path.exists(base_dir):
    for stage in os.listdir(base_dir):
        stage_dir = os.path.join(base_dir, stage)
        if not os.path.isdir(stage_dir): continue
        files = glob.glob(os.path.join(stage_dir, "*.pkl"))
        groups = defaultdict(list)
        for f in files:
            bn = os.path.basename(f)
            parts = bn.split("_")
            if len(parts) >= 3:
                groups[parts[0]].append(f)
        count = 0
        for pid, flist in groups.items():
            if len(flist) > 1:
                # Sort by duration
                flist.sort(key=lambda x: int(os.path.basename(x).split("_")[2]) if len(os.path.basename(x).split("_")) > 2 and os.path.basename(x).split("_")[2].isdigit() else 0, reverse=True)
                for fdel in flist[1:]:
                    os.remove(fdel)
                    count += 1
        if count > 0:
            print(f"Removed {count} duplicate(s) from {stage}")
' "pickles_sleep_stage/$ZELLIJ_SESSION"

# ─── File Discovery ────────────────────────────────────────────────────────────
# Support relative and absolute EDF_ROOT
EDF_SEARCH_DIR="$EDF_ROOT"
if [[ "$EDF_SEARCH_DIR" != /* ]]; then
    EDF_SEARCH_DIR="$SCRIPT_DIR/$EDF_SEARCH_DIR"
fi

get_edf_files() {
    if [[ -d "$EDF_SEARCH_DIR" ]]; then
        find "$EDF_SEARCH_DIR" -type f -iname "*.edf" | sort -r
    fi
}

populate_queue() {
    echo "Filtering EDF files with ECG channel from $EDF_SEARCH_DIR..."
    "$SCRIPT_DIR/venv/bin/python" "$SCRIPT_DIR/filter_ecg_edfs.py" "$EDF_SEARCH_DIR" > "$QUEUE_FILE"
    
    # Filter out already fully processed files
    echo "Filtering out already processed files..."
    "$SCRIPT_DIR/venv/bin/python" -c "
import os
import re
import sys
import glob

STAGE_ORDER = ['W', 'N3', 'R', 'N1', 'N2']
EXPECTED_BANDS = ['alpha', 'beta', 'delta', 'gamma', 'theta']
PROJECT_NAME = '$ZELLIJ_SESSION'
TEMPS_BASE = 'parquets_HEP'

def get_patient_id(fpath):
    m = re.search(r'(\d{4}-\d{3})', fpath)
    if m:
        return m.group(1)
    else:
        return os.path.basename(fpath).split('.')[0]

def is_stage_processed(patient_id, stage):
    temps_dir = os.path.join(TEMPS_BASE, f'{PROJECT_NAME}_{stage}')
    for band in EXPECTED_BANDS:
        parquet_file = os.path.join(temps_dir, f'{patient_id}_{stage}_results_{band}.parquet')
        if not os.path.exists(parquet_file):
            return False
    return True

def is_stage_errored(patient_id, stage):
    log_files = glob.glob('logs/*.log')
    for log_file in log_files:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                for line in f:
                    if f'[{stage}] Patient {patient_id}:' in line and ('ERROR' in line or 'WARNING' in line):
                        return True
    return False

def is_fully_processed(patient_id):
    return all(is_stage_processed(patient_id, stage) or is_stage_errored(patient_id, stage) for stage in STAGE_ORDER)

with open('$QUEUE_FILE', 'r') as f:
    files = [line.strip() for line in f if line.strip()]

filtered_files = []
for fpath in files:
    patient_id = get_patient_id(fpath)
    if not is_fully_processed(patient_id):
        filtered_files.append(fpath)
    else:
        print(f'Skipping already processed or errored: {fpath}')

with open('$QUEUE_FILE', 'w') as f:
    for fpath in filtered_files:
        f.write(fpath + '\n')

print(f'Filtered to {len(filtered_files)} files after removing processed or errored ones.')
"
    
    NUM_FILES=$(wc -l < "$QUEUE_FILE" | tr -d ' ')
    echo "Found $NUM_FILES files with ECG channel and not fully processed or errored."
}

# ─── Layout Generator ──────────────────────────────────────────────────────────
# Generates a separate layout file containing a generic worker tab
generate_worker_tab() {
    local WORKER_ID=$1
    local IS_FULL_SESSION=${2:-false}
    local TAB_NAME="Worker_$WORKER_ID"
    
    # We use a double-quoted script cmd that passes single-quoted variables around to prevent expansion issues in KDL
    local SCRIPT_CMD="cd $SCRIPT_DIR && source venv/bin/activate && while true; do "
    SCRIPT_CMD+="FILE=\$(flock -x 200; head -n 1 '$QUEUE_FILE' | tee /dev/stderr; sed -i '1d' '$QUEUE_FILE') 200>'$LOCK_FILE'; "
    SCRIPT_CMD+="if [ -z \\\"\$FILE\\\" ]; then echo 'Queue empty. Worker finished.'; zellij --session $ZELLIJ_SESSION action close-pane; exit 0; fi; "
    for STAGE in "${STAGE_ORDER[@]}"; do
        SCRIPT_CMD+="zellij action rename-tab \"${STAGE}_\$(basename \"\$FILE\")\"; "
        SCRIPT_CMD+="$SCRIPT_DIR/venv/bin/python HEP_parquet_generation.py --stage $STAGE --edf_root \\\"\$FILE\\\"; "
    done
    SCRIPT_CMD+="echo 'File processed. Fetching next...'; sleep 1; done"

    local SCRIPT_CMD_ESCAPED="${SCRIPT_CMD//\\/\\\\}"
    SCRIPT_CMD_ESCAPED="${SCRIPT_CMD_ESCAPED//\"/\\\"}"

    local OUT=""
    if $IS_FULL_SESSION; then
        OUT+="    tab name=\"$TAB_NAME\" {\n"
        OUT+="        pane command=\"bash\" {\n"
        OUT+="            args {\n"
        OUT+="                \"-c\"\n"
        OUT+="                \"$SCRIPT_CMD_ESCAPED\"\n"
        OUT+="            }\n"
        OUT+="            start_suspended false\n"
        OUT+="            close_on_exit true\n"
        OUT+="        }\n"
        OUT+="    }\n"
    else
        OUT+="layout {\n"
        OUT+="    tab name=\"$TAB_NAME\" {\n"
        OUT+="        pane command=\"bash\" {\n"
        OUT+="            args {\n"
        OUT+="                \"-c\"\n"
        OUT+="                \"$SCRIPT_CMD_ESCAPED\"\n"
        OUT+="            }\n"
        OUT+="            start_suspended false\n"
        OUT+="            close_on_exit true\n"
        OUT+="        }\n"
        OUT+="    }\n"
        OUT+="}\n"
    fi
    echo -e "$OUT"
}

generate_full_layout() {
    local NUM_WORKERS=$1
    echo "layout {" > "$LAYOUT_FILE"
    echo "    default_tab_template {" >> "$LAYOUT_FILE"
    echo "        pane size=1 borderless=true {" >> "$LAYOUT_FILE"
    echo "            plugin location=\"zellij:tab-bar\"" >> "$LAYOUT_FILE"
    echo "        }" >> "$LAYOUT_FILE"
    echo "        children" >> "$LAYOUT_FILE"
    echo "        pane size=1 borderless=true {" >> "$LAYOUT_FILE"
    echo "            plugin location=\"zellij:status-bar\"" >> "$LAYOUT_FILE"
    echo "        }" >> "$LAYOUT_FILE"
    echo "    }" >> "$LAYOUT_FILE"

    for ((i=1; i<=NUM_WORKERS; i++)); do
        generate_worker_tab "$i" true >> "$LAYOUT_FILE"
    done

    echo "}" >> "$LAYOUT_FILE"
}

# ─── Launch / attach ───────────────────────────────────────────────────────────

# 50% of logical cores (e.g. 128 cores → 64 workers)
WORKERS=$(( $(nproc 2>/dev/null || echo 1) / 3 ))
if [[ "$WORKERS" -eq 0 ]]; then WORKERS=1; fi
if [[ "$WORKERS" -gt 64 ]]; then WORKERS=64; fi

# Strip the "EXITED" text to just check if the session name exists as an active session
if zellij list-sessions 2>/dev/null | grep -v "EXITED" | grep -q "^$ZELLIJ_SESSION"; then
    echo "Session '$ZELLIJ_SESSION' already exists. Adding $WORKERS new worker tabs..."
    populate_queue
    for ((i=1; i<=WORKERS; i++)); do
        echo "  Adding worker $i..."
        # Build the same worker command as generate_worker_tab
        SCRIPT_CMD="cd $SCRIPT_DIR && source venv/bin/activate && while true; do "
        SCRIPT_CMD+="FILE=\$(flock -x 200; head -n 1 '$QUEUE_FILE' | tee /dev/stderr; sed -i '1d' '$QUEUE_FILE') 200>'$LOCK_FILE'; "
        SCRIPT_CMD+="if [ -z \"\$FILE\" ]; then echo 'Queue empty. Worker finished.'; exit 0; fi; "
        for STAGE in "${STAGE_ORDER[@]}"; do
            SCRIPT_CMD+="zellij action rename-tab \"${STAGE}_\$(basename \"\$FILE\")\"; "
            SCRIPT_CMD+="$SCRIPT_DIR/venv/bin/python HEP_parquet_generation.py --stage $STAGE --edf_root \"\$FILE\"; "
        done
        SCRIPT_CMD+="echo 'File processed. Fetching next...'; sleep 1; done"
        echo "SCRIPT_CMD: $SCRIPT_CMD"
        # zellij run opens a new tab and immediately starts the command — no start_suspended prompt
        zellij --session "$ZELLIJ_SESSION" run --name "Worker_$i" --new-tab -- bash -c "$SCRIPT_CMD"
        sleep 0.5
    done
else
    # Delete the session in case it exists in a resurrectable (EXITED) state
    zellij delete-session "$ZELLIJ_SESSION" --force 2>/dev/null || true
    
    echo "Generating session layout for $WORKERS workers..."
    populate_queue
    generate_full_layout "$WORKERS"
    
    echo "Layout file contents: $LAYOUT_FILE"
    sed -n '1,80p' "$LAYOUT_FILE"
    
    ZELLIJ_CMD=(zellij --layout "$LAYOUT_FILE" --session "$ZELLIJ_SESSION")
    echo "Running: ${ZELLIJ_CMD[*]}"
    "${ZELLIJ_CMD[@]}" > "$SCRIPT_DIR/logs/zellij_start.log" 2>&1 &
    ZELLIJ_PID=$!
    echo "Started zellij background PID=$ZELLIJ_PID"
    
    echo "Waiting for Zellij to initialize worker tabs..."
    sleep 5
    
    # Check if it stayed alive
    if ! zellij list-sessions 2>/dev/null | grep -q "^$ZELLIJ_SESSION"; then
        echo "Warning: Zellij background spawn might have failed or is taking longer than 5s."
    fi
fi

echo ""
echo "All worker tabs initialised in session '$ZELLIJ_SESSION'."
# delay 3 seconds
sleep 5

zellij attach $ZELLIJ_SESSION
# zellij attach cobrad
echo "To attach:  zellij attach $ZELLIJ_SESSION"
