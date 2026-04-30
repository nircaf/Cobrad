#!/bin/bash

source venv/bin/activate

MODES=(
    "Single Group Analysis"
    "Compare Groups"
    "Compare Sleep Stages"
    "Compare Groups All Sleep Stages"
    "Compare Groups Non-EEG Channels"
)

echo "Available modes:"
for i in "${!MODES[@]}"; do
    echo "  $((i+1)). ${MODES[$i]}"
done
echo "  A. All modes"
echo ""

read -p "Select modes to open (e.g. '1 3 5' or 'A' for all) [A]: " SELECTION
SELECTION=${SELECTION:-A}

if [[ "$SELECTION" == "A" || "$SELECTION" == "a" ]]; then
    SELECTED_MODES=("${MODES[@]}")
else
    SELECTED_MODES=()
    for num in $SELECTION; do
        idx=$((num - 1))
        if [[ $idx -ge 0 && $idx -lt ${#MODES[@]} ]]; then
            SELECTED_MODES+=("${MODES[$idx]}")
        fi
    done
fi

echo ""
echo "Opening ${#SELECTED_MODES[@]} window(s)..."

for mode in "${SELECTED_MODES[@]}"; do
    echo "  Starting: $mode"
    streamlit run 6_hep_group_comparison.py -- --mode "$mode" &
    sleep 0.5
done

echo ""
echo "All windows launched. Press Ctrl+C to stop all."
wait
