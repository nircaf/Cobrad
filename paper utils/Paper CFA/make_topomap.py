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
import json
import os
import shutil
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import colors
import numpy as np
import pandas as pd

from channel_utils import CANON, canonicalize, filter_min_coverage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from eeg_brain_blender import CMAPS  # noqa: E402 (bpy-free import: colour stops only)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
MIN_COVERAGE = 0.0
MIN_PATIENTS = 500
BRAIN_BLENDER = os.path.join(HERE, "..", "utils", "eeg_brain_blender.py")
BLENDER_BIN = os.environ.get("BLENDER_BIN") or shutil.which("blender") or "blender"

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
# Panel a: 3D scalp/cortex render of mean CFA R^2 per site, via
# utils/eeg_brain_blender.py (blender -b headless; see that file's docstring).
# ---------------------------------------------------------------------
values = np.array([per_site.loc[c, "mean"] for c in present])

with tempfile.TemporaryDirectory() as tmp:
    cfg_path = os.path.join(tmp, "channels.json")
    brain_png = os.path.join(tmp, "brain.png")
    json.dump({
        "channels": {c: float(v) for c, v in zip(present, values)},
        "view": "top", "cmap": "graphite", "vmin": 0.0, "vmax": float(np.nanmax(values)),
        "show_electrodes": False,
        # top view is portrait (brain is longer anterior-posterior than side to
        # side); match the render frame to that aspect and fill it tightly, or
        # most of a landscape frame is wasted white margin and the brain reads
        # small/soft at print size
        "resolution": [1100, 1450], "fill": 0.94,
        # denser mesh so smooth-shaded normals still resolve the sulci detail
        # instead of interpolating it away at figure print size
        "subdiv": 9,
    }, open(cfg_path, "w"))
    subprocess.run(
        [BLENDER_BIN, "-b", "-P", BRAIN_BLENDER, "--", cfg_path, brain_png],
        check=True, capture_output=True,
    )
    brain_img = mpimg.imread(brain_png)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.6]})

ax = axes[0]
ax.imshow(brain_img)
ax.axis("off")
graphite_cmap = colors.LinearSegmentedColormap.from_list("graphite", [(p, rgb) for p, rgb in CMAPS["graphite"]])
sm = plt.cm.ScalarMappable(cmap=graphite_cmap, norm=plt.Normalize(0, np.nanmax(values)))
cbar = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.04)
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
