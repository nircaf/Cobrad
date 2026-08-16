#!/usr/bin/env python3
"""Figure 5: scalp topomap of mean CFA R^2 per 10-20 electrode site, plus the
full per-channel R^2 distribution (not just the top-15 bar chart in Fig 2b).

Channel labels in the corpus are a mix of monopolar ("F3") and
mastoid/ear-referenced bipolar ("F3-M2", "F8-A1") derivations; both are
canonicalised to their first (scalp-side) 10-20 site so every row contributes
to that site's topomap value and distribution box, matching the canonical
24-channel montage order this project already uses
(16_diagnosis_sleep_stage_comparison_dashboard.py's MONTAGE_1020_CHANNEL_ORDER).

Run: venv/bin/python "Paper CFA/make_topomap.py"
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from channel_utils import CANON, canonicalize, filter_min_coverage

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
MIN_COVERAGE = 0.0
MIN_PATIENTS = 500

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300,
})

df = pd.read_parquet(os.path.join(HERE, "cfa_combined.parquet"))
n_raw_channels = df.eeg_channel.nunique()
df["canon"] = canonicalize(df["eeg_channel"])
df = df.dropna(subset=["canon"])
df = filter_min_coverage(df, "patient_id", "canon", min_frac=MIN_COVERAGE)
site_n_patients = df.groupby("canon")["patient_id"].nunique()
sparse_sites = site_n_patients[site_n_patients < MIN_PATIENTS].index.tolist()
df = df[~df["canon"].isin(sparse_sites)]
print(f"{len(df):,} rows canonicalised to {df.canon.nunique()} scalp sites with >= {MIN_PATIENTS} "
      f"patients (of {n_raw_channels} raw channel labels; dropped {sparse_sites} for <{MIN_PATIENTS} patients)")

per_site = df.groupby("canon")["cfa_r2_excl_qrs"].agg(["mean", "median", "std", "count"])
present = [c for c in dict.fromkeys(list(CANON.values())) if c in per_site.index]

# ---------------------------------------------------------------------
# Panel a: topomap of mean CFA R^2 (excl. QRS) per site
# ---------------------------------------------------------------------
montage = mne.channels.make_standard_montage("standard_1020")
info = mne.create_info(present, sfreq=1.0, ch_types="eeg")
info.set_montage(montage)
values = np.array([per_site.loc[c, "mean"] for c in present])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.6]})

ax = axes[0]
im, cn = mne.viz.plot_topomap(
    values, info, axes=ax, show=False, cmap="Reds", vlim=(0, np.nanmax(values)),
    names=present, sensors=True, contours=4, size=4,
)
cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.04)
cbar.set_label("Mean CFA R² (outside QRS)", fontsize=8)
ax.set_title("a  Scalp topography of CFA variance explained", loc="left", fontsize=10, fontweight="bold")

ax = axes[1]
order = per_site.loc[present, "median"].sort_values(ascending=True).index.tolist()
box_data = [df.loc[df.canon == c, "cfa_r2_excl_qrs"].dropna().values for c in order]
order_labels = [f"{c} (n={len(v):,})" for c, v in zip(order, box_data)]
bp = ax.boxplot(box_data, tick_labels=order_labels, vert=False, patch_artist=True, widths=0.65,
                 showfliers=False, whis=0)
for line in bp["whiskers"] + bp["caps"]:
    line.set_visible(False)
cmap = plt.get_cmap("Reds")
norm_vals = (per_site.loc[order, "mean"].values - values.min()) / (values.max() - values.min() + 1e-12)
for patch, v in zip(bp["boxes"], norm_vals):
    patch.set_facecolor(cmap(0.25 + 0.65 * v))
    patch.set_alpha(0.9)
q1 = np.array([np.percentile(v, 25) for v in box_data])
q3 = np.array([np.percentile(v, 75) for v in box_data])
ax.set_xlim(q1.min() - 0.05 * (q3.max() - q1.min()), q3.max() + 0.05 * (q3.max() - q1.min()))
ax.set_xlabel("CFA R² (outside QRS window)")
ax.set_title(f"b  Per-channel distribution ({len(order)} sites)", loc="left", fontsize=10, fontweight="bold")
ax.tick_params(axis="y", labelsize=7.5)

fig.suptitle(
    f"Figure 2. Where on the scalp is cardiac-field variance concentrated? "
    f"(n = {df.patient_id.nunique():,} patients, {len(df):,} channel-recordings, {len(present)} sites)",
    fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(FIG_DIR, "fig2_topomap_channel_distribution.pdf"))
fig.savefig(os.path.join(FIG_DIR, "fig2_topomap_channel_distribution.png"))
plt.close(fig)

import json
with open(os.path.join(HERE, "paper_stats.json")) as f:
    S = json.load(f)
S["topomap"] = {
    "n_sites": len(present), "n_rows": int(len(df)), "min_patients": MIN_PATIENTS,
    "dropped_sites": sparse_sites,
    "site_means": {c: float(per_site.loc[c, "mean"]) for c in present},
    "highest_site": str(per_site.loc[present, "mean"].idxmax()),
    "highest_mean": float(per_site.loc[present, "mean"].max()),
    "lowest_site": str(per_site.loc[present, "mean"].idxmin()),
    "lowest_mean": float(per_site.loc[present, "mean"].min()),
}
with open(os.path.join(HERE, "paper_stats.json"), "w") as f:
    json.dump(S, f, indent=2)
print("Figure written to", os.path.join(FIG_DIR, "fig2_topomap_channel_distribution.png"))
print(per_site.loc[order])
