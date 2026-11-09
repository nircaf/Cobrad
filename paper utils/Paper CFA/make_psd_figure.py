#!/usr/bin/env python3
"""Supplementary Figure S5: power spectral density comparison across four
conditions — pre-ICA EEG, post-ICA EEG, non-heartbeat-locked EEG control,
and the patient's own ECG — for the same evoked waveforms used in Figure
S4's cross-correlation/mutual-information analysis.

(a) Group-overall: core 4-electrode average PSD, all four conditions
    overlaid (mean +/- SEM across patients/channels).
(b) Per-electrode: one small panel per canonical site (6), same four-way
    overlay, so a reader can see whether spectral shape differs by region.

Run: venv/bin/python "Paper CFA/make_psd_figure.py"
(requires make_crosscorr_mi.py to have been run first)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from channel_utils import canonicalize

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00", "#000000"]
CORE_FOUR = ["F3", "F4", "C3", "C4"]
CONDITIONS = [("raw", PALETTE[1], "Raw EEG"), ("hep", PALETTE[0], "HEP"),
              ("cfar", PALETTE[3], "Patient-specific CFAR"), ("ica", PALETTE[5], "ICA")]
LS = {"ica": "--"}  # dashed: ICA sits on top of raw/CFAR

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300,
})

psd = pd.read_parquet(os.path.join(HERE, "psd_raw_hep_cfar.parquet"))
psd = psd[psd.freq_hz <= 99]  # just below the 100 Hz band-pass edge (§2.2); the edge bin is filter roll-off
psd["canon"] = psd["site"]
n_patients = psd.patient_id.nunique()
print(f"{len(psd):,} PSD rows, {n_patients:,} patients")

fig = plt.figure(figsize=(11, 8))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.28)
left = gs[0].subgridspec(2, 1, hspace=0.35)

# ---------------------------------------------------------------------
# Panel b: group-overall (core-4 EEG average) + ECG
# ---------------------------------------------------------------------
ax = fig.add_subplot(left[1])
for condition, color, label in CONDITIONS:
    sub = psd[(psd.canon.isin(CORE_FOUR)) & (psd.condition == condition)]
    stats = sub.groupby("freq_hz")["power_db"].agg(["mean", "sem"])
    ax.plot(stats.index, stats["mean"], color=color, lw=1.8, ls=LS.get(condition, "-"), label=label)
    ax.fill_between(stats.index, stats["mean"] - stats["sem"], stats["mean"] + stats["sem"],
                     color=color, alpha=0.2, linewidth=0)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power (dB)")
ax.set_title("b", loc="left", fontsize=10, fontweight="bold")
ax.set_title("Power spectrum, core electrodes", fontsize=9)
ax.legend(fontsize=7.5, frameon=False)

# ---------------------------------------------------------------------
# Panel c: per-electrode small multiples, each with the same 4-way overlay
# ---------------------------------------------------------------------
inner = gs[1].subgridspec(3, 2, hspace=0.55, wspace=0.35)
for idx, ch in enumerate(CORE_FOUR + ["O1", "O2"]):
    ax = fig.add_subplot(inner[idx // 2, idx % 2])
    for condition, color, label in CONDITIONS:
        sub = psd[(psd.canon == ch) & (psd.condition == condition)]
        stats = sub.groupby("freq_hz")["power_db"].agg(["mean", "sem"])
        ax.plot(stats.index, stats["mean"], color=color, lw=1.3, ls=LS.get(condition, "-"), label=label)
        ax.fill_between(stats.index, stats["mean"] - stats["sem"], stats["mean"] + stats["sem"],
                         color=color, alpha=0.2, linewidth=0)
    ax.set_title(ch, fontsize=8.5, fontweight="bold")
    ax.tick_params(labelsize=6.5)
    if idx >= 4:
        ax.set_xlabel("Hz", fontsize=7)
    if idx % 2 == 0:
        ax.set_ylabel("dB", fontsize=7)
fig.text(0.53, 0.9, "c", fontsize=10, fontweight="bold")
fig.text(0.76, 0.935, "Power spectrum by electrode", fontsize=9, ha="center")

# ---------------------------------------------------------------------
# Panel a: one patient (Figure 4a example), continuous EEG: raw vs ICA vs patient-specific CFAR
# ---------------------------------------------------------------------
import make_cleaning_demo as cd
from scipy.signal import welch
from ica_ecg_component_variance import epoch_around_r_peaks

row = pd.read_parquet(os.path.join(HERE, "cfa_combined.parquet"), columns=["patient_id", "edf_path", "window_start_s"])
row = row[row.patient_id == cd.EXAMPLE].iloc[0]
_, raw_ex, ecg_ex, peaks_ex, sfreq = cd.load((row.patient_id, row.edf_path, float(row.window_start_s)))
u_own = cd.cfa_pattern(raw_ex, peaks_ex, sfreq)  # patient-specific CFAR: the patient's own ECG-derived pattern
ch = cd.SITES.index(cd.EXAMPLE_SITE)
ax = fig.add_subplot(left[0])
hep_ex = epoch_around_r_peaks(raw_ex, peaks_ex, sfreq).mean(0)
for x, color, label, nper in [(raw_ex[ch], PALETTE[1], "Raw EEG", int(4 * sfreq)),
                              (hep_ex[ch], PALETTE[0], "HEP", min(hep_ex.shape[1], 128)),
                              (cd.project_out(raw_ex, u_own)[ch], PALETTE[3], "Patient-specific CFAR", int(4 * sfreq)),
                              (cd.ica_clean(raw_ex, ecg_ex, sfreq)[ch], PALETTE[5], "ICA", int(4 * sfreq))]:
    f_ex, p_ex = welch(x, sfreq, nperseg=nper)
    keep = (f_ex >= 0.5) & (f_ex <= 40)
    ax.plot(f_ex[keep], 10 * np.log10(p_ex[keep]), color=color, lw=1.2, ls="--" if label == "ICA" else "-", label=label)
ax.set_xlim(0.5, 40)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power (dB re µV²/Hz)")
ax.set_title("a", loc="left", fontsize=10, fontweight="bold")
ax.set_title(f"Example patient (Figure 4a), {cd.EXAMPLE_SITE}, 10-min EEG", fontsize=9)
ax.legend(fontsize=7.5, frameon=False)

# no suptitle: the manuscript caption carries the figure number
fig.savefig(os.path.join(FIG_DIR, "figS5_psd_comparison.pdf"))
fig.savefig(os.path.join(FIG_DIR, "figS5_psd_comparison.png"))
plt.close(fig)

with open(os.path.join(HERE, "paper_stats.json")) as f:
    S = json.load(f)
S["psd_comparison"] = {"n_patients": int(n_patients), "core_electrodes": CORE_FOUR}
with open(os.path.join(HERE, "paper_stats.json"), "w") as f:
    json.dump(S, f, indent=2, default=float)
print("Figure written to", os.path.join(FIG_DIR, "figS5_psd_comparison.png"))
