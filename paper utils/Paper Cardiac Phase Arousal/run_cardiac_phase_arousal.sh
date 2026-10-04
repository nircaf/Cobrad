#!/usr/bin/env bash
# Full cardiac-phase-of-arousal pipeline. Resumable: build_dataset.py caches one
# parquet per session, so re-running only processes sessions not yet done.
#   bash "paper utils/Paper Cardiac Phase Arousal/run_cardiac_phase_arousal.sh"
set -euo pipefail
cd "$(dirname "$0")/../.."
source venv/bin/activate
D="paper utils/Paper Cardiac Phase Arousal"
python3 "$D/build_dataset.py"
python3 "$D/cardiac_phase_arousal_stats.py"
python3 "$D/make_figures.py"
python3 "$D/make_pdf.py"
