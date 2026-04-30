#!/bin/bash

# ─── Config ────────────────────────────────────────────────────────────────────
TMUX_SESSION="EDF" # Default fallback

# Parse session argument if provided
if [[ "$1" == "1" || "$1" == "EDF" ]]; then
    TMUX_SESSION="EDF"
    shift
elif [[ "$1" == "2" || "$1" == "Berkeley_data" ]]; then
    TMUX_SESSION="Berkeley_data"
    shift
elif [[ "$1" == "3" || "$1" == "Seeg" ]]; then
    TMUX_SESSION="Seeg"
    shift
elif [[ "$1" == "4" || "$1" == "CAP_Sleep_Database" ]]; then
    TMUX_SESSION="CAP_Sleep_Database"
    shift
elif [[ "$1" == "5" ]]; then
    TMUX_SESSION="PSG_IPA"
    shift
elif [[ -n "$1" && "$1" != -* ]]; then
    TMUX_SESSION="$1"
    shift
fi

# Parse --reset, --dashboard, and --full_reset flags
RESET=0
DASHBOARD=0
FULL_RESET=0
REVERSE=0
for arg in "$@"; do
    if [[ "$arg" == "--reset" ]]; then
        RESET=1
    fi
    if [[ "$arg" == "--dashboard" ]]; then
        DASHBOARD=1
    fi
    if [[ "$arg" == "--full_reset" ]]; then
        FULL_RESET=1
    fi
    if [[ "$arg" == "--reverse" ]]; then
        REVERSE=1
    fi
done

REVERSE_FLAG=""
[[ "$REVERSE" == "1" ]] && REVERSE_FLAG="--reverse"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDF_ROOT="EDF_Format/$TMUX_SESSION"
STAGE_ORDER=(light_sleep W N3 R)

cd "$SCRIPT_DIR"

# ─── Dashboard ─────────────────────────────────────────────────────────────────
run_dashboard() {
    local GREEN='\033[32m'
    local RED='\033[31m'
    local YELLOW='\033[33m'
    local CYAN='\033[36m'
    local RESET='\033[0m'

    # Compute relative time from a "YYYY-MM-DD HH:MM:SS" string
    relative_time() {
        local ts="$1"
        local epoch
        epoch=$(date -d "$ts" +%s 2>/dev/null) || { echo "unknown"; return; }
        local now
        now=$(date +%s)
        local diff=$(( now - epoch ))
        if (( diff < 60 )); then
            echo "${diff}s ago"
        elif (( diff < 3600 )); then
            echo "$(( diff / 60 ))m ago"
        elif (( diff < 86400 )); then
            echo "$(( diff / 3600 ))h ago"
        else
            echo "$(( diff / 86400 ))d ago"
        fi
    }

    trap 'echo -e "\n${RESET}Dashboard stopped."; exit 0' INT

    while true; do
        clear
        echo -e "${CYAN}=== HEP Run Dashboard - ${TMUX_SESSION} ===${RESET}  $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""

        local most_active_stage=""
        local most_active_epoch=0

        for STAGE in "${STAGE_ORDER[@]}"; do
            # Tmux window status
            if tmux list-windows -t "$TMUX_SESSION" 2>/dev/null | grep -q "^[0-9]*: $STAGE "; then
                local status_str="${GREEN}RUNNING${RESET}"
            else
                local status_str="${RED}STOPPED${RESET}"
            fi

            echo -e "${YELLOW}[ $STAGE ]${RESET}  tmux: $status_str"

            # Find latest log file for this stage
            local logfile=""
            # Try session-specific log first
            local specific
            specific=$(ls -t "${SCRIPT_DIR}/logs/run_${TMUX_SESSION}_stage_${STAGE}_"*.log 2>/dev/null | head -1)
            if [[ -n "$specific" ]]; then
                logfile="$specific"
            else
                # Fallback: any session
                local fallback
                fallback=$(ls -t "${SCRIPT_DIR}/logs/run_"*"_stage_${STAGE}_"*.log 2>/dev/null | head -1)
                [[ -n "$fallback" ]] && logfile="$fallback"
            fi

            if [[ -z "$logfile" ]]; then
                echo "  Log:           no log found"
                echo "  Finished:      -"
                echo "  Current:       -"
                echo "  Last step:     -"
                echo "  Last activity: -"
            else
                echo "  Log:           $(basename "$logfile")"

                # Finished count
                local finished
                finished=$(grep -c "successfully finished processing" "$logfile" 2>/dev/null) || finished=0
                echo "  Finished:      $finished"

                # Current patient
                local current_patient
                current_patient=$(grep "started processing" "$logfile" 2>/dev/null | tail -1 | sed 's/.*\] Patient \(.*\): started.*/\1/')
                [[ -z "$current_patient" ]] && current_patient="-"
                echo "  Current:       $current_patient"

                # Last step — last INFO line, message after final " - "
                local last_info_line
                last_info_line=$(grep " - INFO - " "$logfile" 2>/dev/null | tail -1)
                if [[ -n "$last_info_line" ]]; then
                    local last_msg
                    last_msg="${last_info_line##* - }"
                    # Trim to 100 chars
                    last_msg="${last_msg:0:100}"
                    echo "  Last step:     $last_msg"

                    # Last activity timestamp
                    local last_ts
                    last_ts=$(echo "$last_info_line" | awk '{print $1" "$2}')
                    local rel
                    rel=$(relative_time "$last_ts")
                    echo "  Last activity: $rel  ($last_ts)"

                    # Track most active
                    local epoch
                    epoch=$(date -d "$last_ts" +%s 2>/dev/null || echo 0)
                    if (( epoch > most_active_epoch )); then
                        most_active_epoch=$epoch
                        most_active_stage="$STAGE"
                    fi
                else
                    echo "  Last step:     -"
                    echo "  Last activity: -"
                fi
            fi

            echo "  ---"
        done

        echo ""
        if [[ -n "$most_active_stage" ]]; then
            echo -e "${CYAN}Most active:${RESET} $most_active_stage  (last activity $(relative_time "$(date -d "@$most_active_epoch" '+%Y-%m-%d %H:%M:%S')"))"
        else
            echo -e "${CYAN}Most active:${RESET} -"
        fi

        sleep 5
    done
}

session_is_live() {
    tmux has-session -t "$TMUX_SESSION" 2>/dev/null
}

mkdir -p "$SCRIPT_DIR/logs"

add_windows() {
    for STAGE in "${STAGE_ORDER[@]}"; do
        CMD="cd $SCRIPT_DIR && source venv/bin/activate && python HEP_parquet_generation.py --stage $STAGE --edf_root \"$EDF_ROOT\" $REVERSE_FLAG"
        echo "CMD: $CMD"
        # Kill any existing window with the same name to avoid duplicate-name ambiguity
        tmux kill-window -t "$TMUX_SESSION:$STAGE" 2>/dev/null || true
        tmux new-window -t "$TMUX_SESSION" -n "$STAGE"
        tmux send-keys -t "$TMUX_SESSION:$STAGE" "$CMD" Enter
        sleep 0.3
    done
}

if [[ "$DASHBOARD" == "1" ]]; then
    run_dashboard
    exit 0
fi

if [[ "$FULL_RESET" == "1" ]]; then
    echo "Full reset for session '$TMUX_SESSION': killing processes, removing parquets, run_status files and cache..."
    # Kill any background Python processes running HEP_parquet_generation.py for this session
    pkill -f "HEP_parquet_generation.py.*${EDF_ROOT}" 2>/dev/null || true
    sleep 1
    # Force-kill any that didn't exit cleanly
    pkill -9 -f "HEP_parquet_generation.py.*${EDF_ROOT}" 2>/dev/null || true
    find "$SCRIPT_DIR/parquets_HEP" -maxdepth 2 -name "run_status.json" -path "*/${TMUX_SESSION}_*/*" -delete
    find "$SCRIPT_DIR/parquets_HEP" -maxdepth 2 -name "*.parquet" -path "*/${TMUX_SESSION}_*/*" -delete
    find "$SCRIPT_DIR/cache/${TMUX_SESSION}" -maxdepth 1 -name "*.pkl" -delete 2>/dev/null || true
    echo "Full reset cleanup done."
    tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
fi

if session_is_live; then
    if [[ "$RESET" == "1" ]]; then
        echo "Resetting session '$TMUX_SESSION'..."
        tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
        # fall through to the create-fresh block below
    else
        echo "Session '$TMUX_SESSION' exists. Adding windows for each sleep stage..."
        add_windows
        exit 0
    fi
fi

# Create fresh session (reached when session is not live, or after --reset kill)
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true

echo "Creating tmux session '$TMUX_SESSION' with windows: ${STAGE_ORDER[*]}"

# Create detached session with the first stage window
FIRST="${STAGE_ORDER[0]}"
CMD="cd $SCRIPT_DIR && source venv/bin/activate && python HEP_parquet_generation.py --stage $FIRST --edf_root \"$EDF_ROOT\" $REVERSE_FLAG"
echo "CMD: $CMD"
tmux new-session -d -s "$TMUX_SESSION" -n "$FIRST"
tmux send-keys -t "$TMUX_SESSION:$FIRST" "$CMD" Enter

# Add remaining stage windows
for STAGE in "${STAGE_ORDER[@]:1}"; do
    CMD="cd $SCRIPT_DIR && source venv/bin/activate && python HEP_parquet_generation.py --stage $STAGE --edf_root \"$EDF_ROOT\" $REVERSE_FLAG"
    echo "CMD: $CMD"
    tmux new-window -t "$TMUX_SESSION" -n "$STAGE"
    tmux send-keys -t "$TMUX_SESSION:$STAGE" "$CMD" Enter
    sleep 0.3
done

# echo tmux attach -t "$TMUX_SESSION"
echo "Session '$TMUX_SESSION' setup complete. "
echo "tmux attach -t \"$TMUX_SESSION\""