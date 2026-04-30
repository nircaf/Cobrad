#!/usr/bin/env bash
# =============================================================================
# generate_sleep_stage_cache.sh
#
# Generate individuals_cache.pkl and individuals_cache_ica.pkl for COBRAD
# sleep stage data. Replaces any existing cache files.
#
# Usage:
#   bash generate_sleep_stage_cache.sh                           # all projects, all stages
#   bash generate_sleep_stage_cache.sh --project Berkeley_data   # one project, all stages
#   bash generate_sleep_stage_cache.sh --stage N2                # all projects, one stage
#   bash generate_sleep_stage_cache.sh --project EDF --stage W   # specific combo
#   bash generate_sleep_stage_cache.sh --no-ica                  # skip ICA cache
#   bash generate_sleep_stage_cache.sh --help                    # show this help
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/generate_sleep_stage_cache.py"

# Show help if requested
for arg in "$@"; do
    if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
        head -20 "${BASH_SOURCE[0]}" | grep "^#" | sed 's/^# \{0,2\}//'
        exit 0
    fi
done

# Detect Python interpreter (prefer venv)
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
if [[ -x "$VENV_PYTHON" ]]; then
    PYTHON="$VENV_PYTHON"
    echo "[INFO] Using venv Python: $PYTHON"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
    echo "[INFO] Using system python3"
else
    echo "[ERROR] No Python interpreter found. Install Python 3 or activate the venv." >&2
    exit 1
fi

# Ensure Python script exists
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "[ERROR] Python script not found: $PYTHON_SCRIPT" >&2
    exit 1
fi

echo "============================================="
echo " COBRAD Sleep Stage Cache Generator"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="
echo ""

# Change to project directory so relative imports resolve correctly
cd "$SCRIPT_DIR"

# Run the Python script, passing through all arguments
"$PYTHON" "$PYTHON_SCRIPT" "$@"
EXIT_CODE=$?

echo ""
echo "============================================="
if [[ $EXIT_CODE -eq 0 ]]; then
    echo " Done. $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo " Finished with errors (exit code $EXIT_CODE). $(date '+%Y-%m-%d %H:%M:%S')"
fi
echo "============================================="

exit $EXIT_CODE
