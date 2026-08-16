"""
Figure 4 -- Diagnosis differences: grand-average HEP waveforms.

Grand-average, channel-averaged HEP waveform (+-SEM) for the largest
diagnostic categories by observed N (recomputed here, not assumed --
see build_hep_diagnosis_dataset.py category counts), overlaid with the
undiagnosed reference ("Unknown") cohort, one small-multiple panel per
sleep stage. Significance is assessed with 6_hep_group_comparison.
permutation_cluster_jitter_test (cluster permutation with pynapple jitter,
QRS window +-50ms excluded, Fisher-combined per-patient p-values) -- run
separately, per stage, on (a) the diagnostic group's patient-average HEP
matrix and (b) the reference cohort's patient-average HEP matrix, each
tested against the zero baseline. Time windows significant (p<0.05) in the
diagnostic group but NOT in the reference cohort (or vice versa) are marked
as candidate diagnosis-associated windows -- this is the only comparison
the existing cluster-permutation function supports (a one-sample cluster
test vs. 0 per group; it is not a two-sample cluster test), and is used
here without modification per the task's instruction not to write a new
cluster statistic.

Reuses mod16.load_patient_data (disk cache) and hep_mod.
permutation_cluster_jitter_test directly.

Run: source venv/bin/activate && python3 Paper1/build_fig4_diagnosis_waveforms.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
import types

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

REPO = "/storage/pblab_shared_data2/Nir/Cobrad"
OUT_DIR = os.path.join(REPO, "Paper1")
FIG_DIR = os.path.join(OUT_DIR, "figures")
LONG_DF_PKL = os.path.join(OUT_DIR, "hep_diagnosis_long_df.pkl")
FIG4_JSON = os.path.join(OUT_DIR, "fig4_diagnosis_waveforms_results.json")
FIG4_PKL = os.path.join(OUT_DIR, "fig4_diagnosis_waveforms_data.pkl")
FIG_PDF = os.path.join(FIG_DIR, "fig4_diagnosis_waveforms.pdf")

GROUP = "Harvard_Electroencephalography"
STAGES = ["light_sleep", "N3", "R"]
STAGE_LABELS = {"light_sleep": "Light Sleep", "N3": "N3 (SWS)", "R": "REM"}
N_TOP_DIAGNOSES = 6
N_PERMUTATIONS = 100
REFERENCE_LABEL = "Unknown"
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]

sys.path.insert(0, OUT_DIR)
from build_hep_diagnosis_dataset import _make_mock_streamlit, load_hep_module, load_mod16  # noqa: E402


def main():
    t0 = time.time()
    if "--plot-only" in sys.argv and os.path.exists(FIG4_PKL):
        with open(FIG4_PKL, "rb") as f:
            cached = pickle.load(f)
        render(cached["times"], cached["per_group_matrices"], cached["diff_windows"], cached["top_diagnoses"])
        print(f"Re-rendered from cache. Elapsed: {time.time() - t0:.1f}s")
        return
    with open(LONG_DF_PKL, "rb") as f:
        cached = pickle.load(f)
    meta_df = cached["meta_df"]
    selected_channels = cached["selected_channels"]

    from collections import Counter
    cat_counts = Counter()
    for cats in meta_df["categories"]:
        cat_counts.update(cats)
    top_diagnoses = [c for c, _ in cat_counts.most_common(N_TOP_DIAGNOSES)]
    print(f"Top {N_TOP_DIAGNOSES} diagnostic categories by observed N (patients, any stage): "
          f"{[(c, cat_counts[c]) for c in top_diagnoses]}")

    categories_by_patient = dict(zip(meta_df["patient_id"], meta_df["categories"]))
    reference_patients = set(meta_df.loc[meta_df["n_categories"] == 0, "patient_id"])

    sys.modules["streamlit"] = _make_mock_streamlit()
    for submod in ["streamlit.components", "streamlit.components.v1"]:
        sys.modules.setdefault(submod, types.ModuleType(submod))
    print("Loading hep_module / mod16 and patient pool (disk cache)...")
    hep_mod = load_hep_module()
    mod16 = load_mod16()
    raw_df = mod16.load_patient_data(hep_mod, [GROUP], STAGES, min_eeg_channels=None)
    raw_df = raw_df.copy()
    raw_df["patient_id"] = raw_df["patient_id"].map(mod16._canonical_patient_id)
    print(f"  raw_df: {len(raw_df)} rows, {raw_df['patient_id'].nunique()} unique patients")

    # ---- Per-patient channel-averaged full trace, per stage ----
    print("Computing channel-averaged full traces per (patient, stage)...")
    trace_by_stage = {s: {} for s in STAGES}  # stage -> {patient_id: (times, trace)}
    common_times = None
    for row in raw_df.itertuples(index=False):
        if row.patient_id in trace_by_stage[row.stage]:
            continue  # dedupe multi-session recordings
        result = mod16._patient_hep_trace(row.individual, selected_channels, False)
        if result is None:
            continue
        trace, times_arr = result
        if common_times is None:
            common_times = times_arr
        elif len(times_arr) != len(common_times):
            trace = np.interp(common_times, times_arr, trace)
        trace_by_stage[row.stage][row.patient_id] = trace

    groups_to_test = top_diagnoses + [REFERENCE_LABEL]
    waveform_results = {}
    per_group_matrices = {s: {} for s in STAGES}
    for stage in STAGES:
        patient_trace_map = trace_by_stage[stage]
        for g in groups_to_test:
            if g == REFERENCE_LABEL:
                ids = [p for p in patient_trace_map if p in reference_patients]
            else:
                ids = [p for p in patient_trace_map
                       if g in categories_by_patient.get(p, [])]
            mat = np.vstack([patient_trace_map[p] for p in ids]) if ids else np.empty((0, len(common_times)))
            per_group_matrices[stage][g] = mat
            waveform_results.setdefault(g, {})[stage] = dict(n=int(mat.shape[0]))
            print(f"  {stage} / {g}: n={mat.shape[0]}")

    # ---- Cluster-permutation test per (group, stage) vs. 0 ----
    print("\nRunning permutation_cluster_jitter_test per (group, stage)...")
    sig_windows_by_group_stage = {}
    for stage in STAGES:
        for g in groups_to_test:
            mat = per_group_matrices[stage][g]
            if mat.shape[0] < 5:
                sig_windows_by_group_stage[(g, stage)] = []
                continue
            sig_windows, _t_obs, per_patient_info = hep_mod.permutation_cluster_jitter_test(
                mat, common_times, n_permutations=N_PERMUTATIONS, p_threshold=0.05,
                jitter_sec=0.1, qrs_exclude_sec=0.05,
            )
            sig_windows_by_group_stage[(g, stage)] = sig_windows
            waveform_results[g][stage]["fisher_p"] = per_patient_info["fisher_p"]
            waveform_results[g][stage]["n_significant_patients"] = per_patient_info["n_significant"]
            waveform_results[g][stage]["cluster_windows"] = sig_windows
            print(f"  {stage} / {g}: {len(sig_windows)} significant cluster window(s), "
                  f"Fisher p={per_patient_info['fisher_p']:.3g}")

    # ---- Diagnosis-vs-reference candidate windows: sig in diagnosis, not in reference ----
    diff_windows = {}
    for stage in STAGES:
        ref_windows = sig_windows_by_group_stage.get((REFERENCE_LABEL, stage), [])
        ref_mask = np.zeros(len(common_times), dtype=bool)
        for w in ref_windows:
            ref_mask |= (common_times >= w["start"]) & (common_times <= w["end"])
        for g in top_diagnoses:
            g_windows = sig_windows_by_group_stage.get((g, stage), [])
            g_mask = np.zeros(len(common_times), dtype=bool)
            for w in g_windows:
                g_mask |= (common_times >= w["start"]) & (common_times <= w["end"])
            candidate_mask = g_mask & ~ref_mask
            diff_windows[(g, stage)] = candidate_mask

    with open(FIG4_PKL, "wb") as f:
        pickle.dump({
            "times": common_times, "per_group_matrices": per_group_matrices,
            "sig_windows_by_group_stage": sig_windows_by_group_stage,
            "diff_windows": diff_windows, "top_diagnoses": top_diagnoses,
        }, f)
    print(f"Saved {FIG4_PKL}")

    def _json_default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not JSON serializable: {type(o)}")

    results = {
        "top_diagnoses": [(c, int(cat_counts[c])) for c in top_diagnoses],
        "n_reference_patients_any_stage": len(reference_patients),
        "waveform_results": waveform_results,
        "n_permutations": N_PERMUTATIONS,
    }
    with open(FIG4_JSON, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"Saved {FIG4_JSON}")

    render(common_times, per_group_matrices, diff_windows, top_diagnoses)
    print(f"Elapsed: {time.time() - t0:.1f}s")


def render(times, per_group_matrices, diff_windows, top_diagnoses):
    fig, axes = plt.subplots(len(STAGES), 1, figsize=(9.0, 10.0), sharex=True)
    colors = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(top_diagnoses)}
    colors[REFERENCE_LABEL] = "#000000"
    panel_letters = "ABC"
    for i, (ax, stage) in enumerate(zip(axes, STAGES)):
        # Reference cohort first (bottom layer), then diagnoses
        for g in [REFERENCE_LABEL] + top_diagnoses:
            mat = per_group_matrices[stage][g]
            if mat.shape[0] == 0:
                continue
            mean = np.nanmean(mat, axis=0)
            sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(mat.shape[0])
            lw = 2.2 if g == REFERENCE_LABEL else 1.4
            ls = "-" if g == REFERENCE_LABEL else "-"
            ax.plot(times, mean, color=colors[g], linewidth=lw, linestyle=ls,
                     label=f"{g} (n={mat.shape[0]})", zorder=3 if g == REFERENCE_LABEL else 2)
            ax.fill_between(times, mean - sem, mean + sem, color=colors[g], alpha=0.15, zorder=1)
        ax.axvline(0, color="grey", linewidth=0.6, linestyle=":")
        ax.axhline(0, color="grey", linewidth=0.5, alpha=0.5)
        ax.axvspan(-0.05, 0.05, color="grey", alpha=0.12, zorder=0)
        ax.set_ylabel("HEP amplitude (µV)", fontsize=9.5)
        ax.set_title(STAGE_LABELS[stage], fontsize=10.5, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.09, 1.08, panel_letters[i], transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="top", ha="left")

        # Significance bars beneath the trace: one row per diagnosis group
        y0 = ax.get_ylim()[0]
        y_span = ax.get_ylim()[1] - y0
        bar_h = y_span * 0.035
        for gi, g in enumerate(top_diagnoses):
            mask = diff_windows.get((g, stage))
            if mask is None or not mask.any():
                continue
            y_bar = y0 - bar_h * (gi + 1.5)
            segs = np.where(np.diff(np.concatenate(([0], mask.astype(int), [0]))))[0]
            for s0, s1 in zip(segs[::2], segs[1::2]):
                ax.plot(times[s0:s1], np.full(s1 - s0, y_bar), color=colors[g], linewidth=4, solid_capstyle="butt")
        ax.set_ylim(y0 - bar_h * (len(top_diagnoses) + 2), ax.get_ylim()[1])
        if i == 0:
            ax.legend(fontsize=7.3, loc="upper right", ncol=2, frameon=False)
    axes[-1].set_xlabel("Time from R-peak (s)", fontsize=10)
    fig.suptitle(
        "Grand-average HEP waveforms by diagnostic category, vs. undiagnosed reference cohort",
        fontsize=11)
    fig.text(0.5, 0.005,
              "Colored bars beneath each trace: windows where the diagnostic group shows a "
              "significant cluster (permutation_cluster_jitter_test, p<0.05, QRS ±50ms excluded) "
              "not also present in the reference cohort at that stage.",
              ha="center", fontsize=7.2, color="#333333")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(FIG_PDF, format="pdf")
    fig.savefig(FIG_PDF.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)
    print(f"Saved {FIG_PDF}")


if __name__ == "__main__":
    main()
