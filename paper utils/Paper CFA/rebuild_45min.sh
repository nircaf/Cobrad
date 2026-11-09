#!/usr/bin/env bash
# Rebuild every table, figure, stat and the PDF on the 45-min main-analysis windows.
# Waits for the 45-min ICA run and the full cleaning-demo run (pids in *.pid), then runs in dependency order.
# Each step's exit code is logged; a failing step does not stop the rest.
#   bash "paper utils/Paper CFA/rebuild_45min.sh" > "paper utils/Paper CFA/rebuild_45min.log" 2>&1 &
cd "$(dirname "$0")/../.." || exit 1
D="paper utils/Paper CFA"
PY=venv/bin/python

for pid_file in "$D/ica_45min.pid" "$D/cleaning_demo_45min.pid"; do
    pid=$(cat "$pid_file" 2>/dev/null)
    while [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; do sleep 120; done
done

run() { echo "=== $(date +%T) $*"; "$PY" "$D/$1" "${@:2}"; echo "=== $1 exit $?"; }

run build_dataset.py
run build_bmi_cache.py
# table-only figures and stats
run make_figures.py
run make_topomap.py
run make_topomap_groups.py
run make_topomap_bmi_adjusted.py
run make_dose_response.py
run make_window_optimization.py
run window_stage_sensitivity_report.py
run make_v2_figures.py
run make_fig3.py
# supplementary analyses that reload the (now 45-min) EDF windows
run make_non_locked_control.py --workers 32
run make_entropy_control.py --workers 32
run make_post_ica_variance.py
run make_crosscorr_mi.py --workers 32
run make_crosscorr_mi_figure.py
run make_psd_raw_hep_cfar.py
run make_psd_figure.py
# cleaning: benchmark on all patients, pipeline figures, then Fig 4 + stats from the saved results
run benchmark_ica_cfar.py --all
run make_cfar_pipeline_figure.py
run make_cleaning_demo.py --plot-only
run make_pdf.py
echo "=== done $(date +%T)"
