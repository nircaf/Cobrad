"""
Figure 1 -- "One HEP per patient, state, and electrode."

3x3 grid: rows = representative electrodes (F3 frontal, C3 central, O1
occipital -- the three channels present in essentially the entire sparse
6-electrode montage that dominates this cohort, see build_hep_diagnosis_
dataset.py channel-presence counts), columns = sleep stage (light_sleep,
N3, R). Each panel overlays a patient subsample of single-channel HEP
traces (thin, low-alpha) with the full-cohort grand average (bold, +-SEM)
on top -- the "single-trial-level" transparency figure.

Reuses mod16.load_patient_data (disk cache) -- no raw EEG re-processing.
Caches trace arrays to Paper1/fig1_overview_data.pkl; renders the vector
PDF panel to Paper1/figures/fig1_overview.pdf.

Run: source venv/bin/activate && python3 Paper1/build_fig1_overview.py
"""
from __future__ import annotations

import os
import pickle
import sys
import time
import types
import importlib.util

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

REPO = "/storage/pblab_shared_data2/Nir/Cobrad"
OUT_DIR = os.path.join(REPO, "Paper1")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

GROUP = "Harvard_Electroencephalography"
STAGES = ["light_sleep", "N3", "R"]
STAGE_LABELS = {"light_sleep": "Light Sleep", "N3": "N3 (SWS)", "R": "REM"}
REP_ELECTRODES = ["F3", "C3", "O1"]  # near-universal in the sparse 6-ch montage
N_SUBSAMPLE = 60
RNG_SEED = 42
STAGE_COLOR = {"light_sleep": "#0072B2", "N3": "#D55E00", "R": "#009E73"}

CACHE_PKL = os.path.join(OUT_DIR, "fig1_overview_data.pkl")
FIG_PDF = os.path.join(FIG_DIR, "fig1_overview.pdf")

sys.path.insert(0, OUT_DIR)
from build_hep_diagnosis_dataset import _make_mock_streamlit, load_hep_module, load_mod16  # noqa: E402


def main():
    t0 = time.time()
    if "--plot-only" in sys.argv and os.path.exists(CACHE_PKL):
        with open(CACHE_PKL, "rb") as f:
            cached = pickle.load(f)
        panels = cached["panels"]
        render(panels)
        print(f"Re-rendered from cache. Elapsed: {time.time() - t0:.1f}s")
        return

    sys.modules["streamlit"] = _make_mock_streamlit()
    for submod in ["streamlit.components", "streamlit.components.v1"]:
        sys.modules.setdefault(submod, types.ModuleType(submod))
    print("Loading hep_module / mod16 and patient pool (disk cache)...")
    hep_mod = load_hep_module()
    mod16 = load_mod16()
    raw_df = mod16.load_patient_data(hep_mod, [GROUP], STAGES, min_eeg_channels=None)
    print(f"  raw_df: {len(raw_df)} rows, {raw_df['patient_id'].nunique()} unique patients")

    rng = np.random.default_rng(RNG_SEED)
    panels = {}  # (electrode, stage) -> dict(times, traces=[(pid, trace)], subsample_idx)
    for electrode in REP_ELECTRODES:
        for stage in STAGES:
            traces = []
            common_times = None
            seen_patients = set()
            for row in raw_df[raw_df["stage"] == stage].itertuples(index=False):
                canonical_pid = mod16._canonical_patient_id(row.patient_id)
                if canonical_pid in seen_patients:
                    continue  # one recording per patient (dedupe multi-session recordings)
                individual = row.individual
                if individual is None or len(individual) <= 3 or individual[3] is None:
                    continue
                hep_data, times, ch_names = individual[1], individual[2], individual[3]
                if hep_data is None or times is None:
                    continue
                ch_lower = [str(c).lower() for c in ch_names]
                if electrode.lower() not in ch_lower:
                    continue
                ci = ch_lower.index(electrode.lower())
                trace_uv = np.asarray(hep_data[ci], dtype=float) * 1e6
                times_arr = np.asarray(times, dtype=float)
                if not mod16._is_valid_hep_trace(trace_uv):
                    continue
                if common_times is None:
                    common_times = times_arr
                elif len(times_arr) != len(common_times):
                    trace_uv = np.interp(common_times, times_arr, trace_uv)
                traces.append((canonical_pid, trace_uv))
                seen_patients.add(canonical_pid)
            if not traces or common_times is None:
                panels[(electrode, stage)] = None
                continue
            mat = np.vstack([t for _, t in traces])
            grand_mean = np.nanmean(mat, axis=0)
            grand_sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(mat.shape[0])
            n_total = mat.shape[0]
            sub_idx = rng.choice(n_total, size=min(N_SUBSAMPLE, n_total), replace=False)
            panels[(electrode, stage)] = dict(
                times=common_times, grand_mean=grand_mean, grand_sem=grand_sem,
                n_total=n_total, subsample_traces=mat[sub_idx],
            )
            print(f"  {electrode} / {stage}: n_patients={n_total}, subsample={len(sub_idx)}")

    with open(CACHE_PKL, "wb") as f:
        pickle.dump({"panels": panels, "electrodes": REP_ELECTRODES, "stages": STAGES,
                     "n_subsample": N_SUBSAMPLE}, f)
    print(f"Saved {CACHE_PKL}")

    render(panels)
    print(f"Elapsed: {time.time() - t0:.1f}s")


def render(panels):
    # ---- Render figure ----
    fig, axes = plt.subplots(len(REP_ELECTRODES), len(STAGES), figsize=(10.2, 7.6), sharex=True)
    fig.subplots_adjust(left=0.1, wspace=0.38, hspace=0.45, top=0.90)
    panel_letters = "ABCDEFGHI"
    letter_i = 0
    for ri, electrode in enumerate(REP_ELECTRODES):
        for ci, stage in enumerate(STAGES):
            ax = axes[ri, ci]
            panel = panels.get((electrode, stage))
            letter = panel_letters[letter_i]
            letter_i += 1
            if panel is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            times = panel["times"]
            color = STAGE_COLOR[stage]
            for trace in panel["subsample_traces"]:
                ax.plot(times, trace, color=color, alpha=0.06, linewidth=0.5, zorder=1)
            ax.plot(times, panel["grand_mean"], color=color, linewidth=2.0, zorder=3)
            ax.fill_between(times, panel["grand_mean"] - panel["grand_sem"],
                             panel["grand_mean"] + panel["grand_sem"], color=color, alpha=0.3, zorder=2)
            ax.axvline(0, color="black", linewidth=0.6, linestyle=":", alpha=0.6)
            ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
            ax.axvspan(-0.05, 0.05, color="grey", alpha=0.15, zorder=0)
            ax.set_ylim(-15, 15)
            ax.spines[["top", "right"]].set_visible(False)
            ax.text(-0.34, 1.12, letter, transform=ax.transAxes, fontsize=13,
                    fontweight="bold", va="top", ha="left")
            if ri == 0:
                ax.set_title(f"{STAGE_LABELS[stage]}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"{electrode}\nHEP amplitude (µV)", fontsize=9)
            if ri == len(REP_ELECTRODES) - 1:
                ax.set_xlabel("Time from R-peak (s)", fontsize=9)
            ax.text(0.97, 0.06, f"n={panel['n_total']}", transform=ax.transAxes,
                    fontsize=7.5, ha="right", color="#444444")
    fig.suptitle("Single-patient HEP traces and grand average, by electrode and sleep stage",
                  fontsize=11)
    fig.savefig(FIG_PDF, format="pdf")
    fig.savefig(FIG_PDF.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)
    print(f"Saved {FIG_PDF}")


if __name__ == "__main__":
    main()
