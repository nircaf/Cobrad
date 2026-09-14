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
CONDITIONS = [("pre_ica", PALETTE[1], "Pre-ICA EEG"), ("post_ica", PALETTE[2], "Post-ICA EEG"),
              ("non_locked", PALETTE[7], "Non-locked EEG control"), ("ecg", PALETTE[0], "ECG")]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300,
})

psd = pd.read_parquet(os.path.join(HERE, "psd_curve.parquet"))
psd["canon"] = np.where(psd.eeg_channel == "ECG", "ECG", canonicalize(psd["eeg_channel"]))
n_patients = psd.patient_id.nunique()
print(f"{len(psd):,} PSD rows, {n_patients:,} patients")

fig = plt.figure(figsize=(11, 8))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.28)

# ---------------------------------------------------------------------
# Panel a: group-overall (core-4 EEG average) + ECG
# ---------------------------------------------------------------------
ax = fig.add_subplot(gs[0])
ecg_psd = psd[psd.canon == "ECG"]
for condition, color, label in CONDITIONS:
    if condition == "ecg":
        sub = ecg_psd
    else:
        sub = psd[(psd.canon.isin(CORE_FOUR)) & (psd.condition == condition)]
    stats = sub.groupby("freq_hz")["power_db"].agg(["mean", "sem"])
    ax.plot(stats.index, stats["mean"], color=color, lw=1.8, label=label)
    ax.fill_between(stats.index, stats["mean"] - stats["sem"], stats["mean"] + stats["sem"],
                     color=color, alpha=0.2, linewidth=0)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power (dB)")
ax.set_title(f"a  Group-overall PSD ({'/'.join(CORE_FOUR)} average + ECG)", loc="left", fontsize=10, fontweight="bold")
ax.legend(fontsize=7.5, frameon=False)

# ---------------------------------------------------------------------
# Panel b: per-electrode small multiples, each with the same 4-way overlay
# ---------------------------------------------------------------------
inner = gs[1].subgridspec(3, 2, hspace=0.55, wspace=0.35)
for idx, ch in enumerate(CORE_FOUR + ["O1", "O2"]):
    ax = fig.add_subplot(inner[idx // 2, idx % 2])
    for condition, color, label in CONDITIONS:
        sub = ecg_psd if condition == "ecg" else psd[(psd.canon == ch) & (psd.condition == condition)]
        if sub.empty:
            continue
        stats = sub.groupby("freq_hz")["power_db"].agg(["mean", "sem"])
        ax.plot(stats.index, stats["mean"], color=color, lw=1.3, label=label)
        ax.fill_between(stats.index, stats["mean"] - stats["sem"], stats["mean"] + stats["sem"],
                         color=color, alpha=0.2, linewidth=0)
    ax.set_title(ch, fontsize=8.5, fontweight="bold")
    ax.tick_params(labelsize=6.5)
    if idx >= 4:
        ax.set_xlabel("Hz", fontsize=7)
    if idx % 2 == 0:
        ax.set_ylabel("dB", fontsize=7)
    if idx == 0:
        ax.legend(fontsize=6, frameon=False, loc="upper right")
fig.text(0.63, 0.93, "b  Per-electrode PSD (6 sites)", fontsize=10, fontweight="bold")

fig.suptitle(f"Figure S5. Power spectral density: pre-ICA, post-ICA, non-locked control, and ECG\n"
             f"(n = {n_patients:,} patients, subsample)", fontsize=10)
fig.savefig(os.path.join(FIG_DIR, "figS5_psd_comparison.pdf"))
fig.savefig(os.path.join(FIG_DIR, "figS5_psd_comparison.png"))
plt.close(fig)

with open(os.path.join(HERE, "paper_stats.json")) as f:
    S = json.load(f)
S["psd_comparison"] = {"n_patients": int(n_patients), "core_electrodes": CORE_FOUR}
with open(os.path.join(HERE, "paper_stats.json"), "w") as f:
    json.dump(S, f, indent=2, default=float)
print("Figure written to", os.path.join(FIG_DIR, "figS5_psd_comparison.png"))
