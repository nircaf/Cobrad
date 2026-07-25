#!/bin/bash

# Runs HEP_parquet_generation.py for patients with exactly 4 stages,
# processing only their single missing stage (derived from 2026-03-15T16-50_export.csv).
# Updated 2026-03-17: removed 8 entries already completed (pickles present in pickles_sleep_stage/EDF).
# Remaining 11 jobs: 0345-016/N3, 0345-017/N3, 0345-028/R, 0345-071/N1, 0345-076/W,
#                    0345-078/N1, 0345-081/W, 0345-083/N3, 0345-091/N3, 0345-098/N3, 0345-104/N1

export PATH="$HOME/.local/bin:$PATH"

ZELLIJ_SESSION="EDF_missing"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Reset / Kill ─────────────────────────────────────────────────────────────
if [[ "$1" == "--reset" || "$1" == "-r" || "$1" == "reset" ]]; then
    echo "Reset request detected. Cleaning up..."
    zellij delete-session "$ZELLIJ_SESSION" --force 2>/dev/null || true
    pkill -9 -f "zellij.*$ZELLIJ_SESSION" 2>/dev/null || true
    rm -rf "$SCRIPT_DIR/logs"
    echo "Reset complete."
    exit 0
fi

if [[ "$1" == "--kill" || "$1" == "-k" || "$1" == "kill" ]]; then
    echo "Kill all sessions request detected..."
    zellij kill-all-sessions --yes 2>/dev/null || true
    zellij delete-all-sessions --force --yes 2>/dev/null || true
    echo "All Zellij sessions killed."
    exit 0
fi

mkdir -p "$SCRIPT_DIR/logs"

QUEUE_FILE="$SCRIPT_DIR/logs/patients_queue.txt"
LOCK_FILE="$SCRIPT_DIR/logs/patients_queue.lock"
LAYOUT_FILE="$SCRIPT_DIR/logs/patients_session_layout.kdl"

EDF_ROOT="/storage/pblab_shared_data2/Nir/Cobrad/EDF_Format/EDF"

# ─── Queue: one "edf_path|stage" entry per missing stage ─────────────────────
populate_queue() {
    cat > "$QUEUE_FILE" <<'EOF'
/storage/pblab_shared_data2/Nir/Cobrad/EDF_Format/EDF/0345-016.EDF|N3
/storage/pblab_shared_data2/Nir/Cobrad/EDF_Format/EDF/0345-017  EDF/0345-017.EDF|N3
/storage/pblab_shared_data2/Nir/Cobrad/EDF_Format/EDF/0345-028  EDF/0345-028.EDF|R
/storage/pblab_shared_data2/Nir/Cobrad/EDF_Format/EDF/0345-071 EDF.EDF|N1
/storage/pblab_shared_data2/Nir/Cobrad/EDF_Format/EDF/0345-076 EDF.EDF|W
/storage/pblab_shared_data2/Nir/Cobrad/EDF_Format/EDF/0345-078.EDF|N1
/storage/pblab_shared_data2/Nir/Cobrad/EDF_Format/EDF/0345-081 EDF.EDF|W
/storage/pblab_shared_data2/Nir/Cobrad/EDF_Format/EDF/0345-083 EDF.EDF|N3
/storage/pblab_shared_data2/Nir/Cobrad/EDF_Format/EDF/0345-091 EDF.EDF|N3
/storage/pblab_shared_data2/Nir/Cobrad/EDF_Format/EDF/0345-098 EDF.EDF|N3
/storage/pblab_shared_data2/Nir/Cobrad/EDF_Format/EDF/0345-104.EDF|N1
EOF
    NUM=$(wc -l < "$QUEUE_FILE" | tr -d ' ')
    echo "Queue populated with $NUM patient/stage jobs."
}

# ─── Layout Generator ─────────────────────────────────────────────────────────
generate_worker_tab() {
    local WORKER_ID=$1
    local IS_FULL_SESSION=${2:-false}
    local TAB_NAME="Worker_$WORKER_ID"

    local SCRIPT_CMD="cd $SCRIPT_DIR && source venv/bin/activate && while true; do "
    SCRIPT_CMD+="ENTRY=\$(flock -x 200; head -n 1 '$QUEUE_FILE' | tee /dev/stderr; sed -i '1d' '$QUEUE_FILE') 200>'$LOCK_FILE'; "
    SCRIPT_CMD+="if [ -z \\\"\$ENTRY\\\" ]; then echo 'Queue empty. Worker finished.'; zellij --session $ZELLIJ_SESSION action close-pane; exit 0; fi; "
    SCRIPT_CMD+="FILE=\\\"\${ENTRY%%|*}\\\"; STAGE=\\\"\${ENTRY##*|}\\\"; "
    SCRIPT_CMD+="echo \\\"Processing \$FILE  stage \$STAGE\\\"; "
    SCRIPT_CMD+="$SCRIPT_DIR/venv/bin/python HEP_parquet_generation.py --stage \\\"\$STAGE\\\" --edf_root \\\"\$FILE\\\"; "
    SCRIPT_CMD+="echo 'Job done. Fetching next...'; sleep 1; done"

    local OUT=""
    if $IS_FULL_SESSION; then
        OUT+="    tab name=\"$TAB_NAME\" {\n"
        OUT+="        pane command=\"bash\" {\n"
        OUT+="            args \"-c\" \"$SCRIPT_CMD\"\n"
        OUT+="            start_suspended false\n"
        OUT+="            close_on_exit true\n"
        OUT+="        }\n"
        OUT+="    }\n"
    else
        OUT+="layout {\n"
        OUT+="    tab name=\"$TAB_NAME\" {\n"
        OUT+="        pane command=\"bash\" {\n"
        OUT+="            args \"-c\" \"$SCRIPT_CMD\"\n"
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

# ─── Launch / Attach ──────────────────────────────────────────────────────────
# Cap workers at number of jobs (11) so we don't spawn idle tabs
TOTAL_JOBS=11
WORKERS=$(( $(nproc 2>/dev/null || echo 1) / 2 ))
if [[ "$WORKERS" -eq 0 ]]; then WORKERS=1; fi
if [[ "$WORKERS" -gt 64 ]]; then WORKERS=64; fi
if [[ "$WORKERS" -gt "$TOTAL_JOBS" ]]; then WORKERS=$TOTAL_JOBS; fi

if zellij list-sessions 2>/dev/null | grep -v "EXITED" | grep -q "^$ZELLIJ_SESSION"; then
    echo "Session '$ZELLIJ_SESSION' already exists. Adding $WORKERS new worker tabs..."
    populate_queue
    for ((i=1; i<=WORKERS; i++)); do
        echo "  Adding worker $i..."
        SCRIPT_CMD="cd $SCRIPT_DIR && source venv/bin/activate && while true; do "
        SCRIPT_CMD+="ENTRY=\$(flock -x 200; head -n 1 '$QUEUE_FILE' | tee /dev/stderr; sed -i '1d' '$QUEUE_FILE') 200>'$LOCK_FILE'; "
        SCRIPT_CMD+="if [ -z \"\$ENTRY\" ]; then echo 'Queue empty. Worker finished.'; exit 0; fi; "
        SCRIPT_CMD+="FILE=\"\${ENTRY%%|*}\"; STAGE=\"\${ENTRY##*|}\"; "
        SCRIPT_CMD+="echo \"Processing \$FILE  stage \$STAGE\"; "
        SCRIPT_CMD+="$SCRIPT_DIR/venv/bin/python HEP_parquet_generation.py --stage \"\$STAGE\" --edf_root \"\$FILE\"; "
        SCRIPT_CMD+="echo 'Job done. Fetching next...'; sleep 1; done"
        zellij --session "$ZELLIJ_SESSION" run --name "Worker_$i" --new-tab -- bash -c "$SCRIPT_CMD"
        sleep 0.5
    done
else
    zellij delete-session "$ZELLIJ_SESSION" --force 2>/dev/null || true

    echo "Generating session layout for $WORKERS workers ($TOTAL_JOBS jobs total)..."
    populate_queue
    generate_full_layout "$WORKERS"

    zellij --layout "$LAYOUT_FILE" attach --create-background "$ZELLIJ_SESSION" > /dev/null 2>&1

    echo "Waiting for Zellij to initialize worker tabs..."
    sleep 5

    if ! zellij list-sessions 2>/dev/null | grep -q "^$ZELLIJ_SESSION"; then
        echo "Warning: Zellij background spawn might have failed or is taking longer than 5s."
    fi
fi

echo ""
echo "All worker tabs initialised in session '$ZELLIJ_SESSION'."
echo "To attach:  zellij attach $ZELLIJ_SESSION"
sleep 3

zellij attach $ZELLIJ_SESSION
