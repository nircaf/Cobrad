#!/usr/bin/env python3
"""Figure 5: scalp CFA topography, no-diagnosis reference vs. clinically obese,
restricted to patients with a dense montage (>10 canonical 10-20 sites) so
each site's per-patient average is well powered within each subgroup.

Run: venv/bin/python "Paper CFA/make_topomap_groups.py"
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

from channel_utils import CANON, canonicalize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from eeg_brain_blender import CMAPS  # noqa: E402 (bpy-free import: colour stops only)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
MIN_ELECTRODES = 10   # dense-montage patient filter
MIN_PATIENTS = 10     # per-site minimum within a subgroup
BRAIN_BLENDER = os.path.join(HERE, "..", "utils", "eeg_brain_blender.py")
BLENDER_BIN = os.environ.get("BLENDER_BIN") or shutil.which("blender") or "blender"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300,
})

cfa = pd.read_parquet(os.path.join(HERE, "cfa_combined.parquet"))
demo = pd.read_parquet(os.path.join(HERE, "demographics_combined.parquet"))
cfa["canon"] = canonicalize(cfa["eeg_channel"])
cfa = cfa.dropna(subset=["canon"])

n_electrodes = cfa.groupby("patient_id")["canon"].nunique()
dense_ids = set(n_electrodes[n_electrodes > MIN_ELECTRODES].index)
cfa_dense = cfa[cfa.patient_id.isin(dense_ids)]

unknown_ids = set(demo.loc[
    demo.patient_id.isin(dense_ids) & (demo["n_diagnoses"].fillna(0) == 0), "patient_id"])
obesity_ids = set(demo[demo.patient_id.isin(dense_ids)]
                   .explode("diagnosis_categories")
                   .query("diagnosis_categories == 'Obesity'").patient_id)

GROUPS = [("unknown", "No diagnosis on record", unknown_ids),
          ("obesity", "Clinical obesity", obesity_ids)]

per_group = {}
for key, label, ids in GROUPS:
    sub = cfa_dense[cfa_dense.patient_id.isin(ids)]
    site_stats = sub.groupby("canon").agg(
        mean=("cfa_r2_excl_qrs", "mean"), n_patients=("patient_id", "nunique"))
    present = [c for c in dict.fromkeys(CANON.values())
               if c in site_stats.index and site_stats.loc[c, "n_patients"] >= MIN_PATIENTS]
    per_group[key] = {
        "label": label, "n_patients": len(ids), "present": present,
        "values": np.array([site_stats.loc[c, "mean"] for c in present]),
    }
    print(f"{label}: {len(ids):,} patients, {len(present)} sites "
          f">= {MIN_PATIENTS} patients/site")

vmax = float(max(g["values"].max() for g in per_group.values()))

# ---------------------------------------------------------------------
# Render both groups via utils/eeg_brain_blender.py, same styling as Figure 2
# (portrait top view, graphite cmap, shared colour scale for a fair comparison).
# ---------------------------------------------------------------------
brain_imgs = {}
with tempfile.TemporaryDirectory() as tmp:
    for key, g in per_group.items():
        cfg_path = os.path.join(tmp, f"{key}.json")
        png_path = os.path.join(tmp, f"{key}.png")
        json.dump({
            "channels": {c: float(v) for c, v in zip(g["present"], g["values"])},
            "view": "top", "cmap": "graphite", "vmin": 0.0, "vmax": vmax,
            "show_electrodes": False, "resolution": [1100, 1450], "fill": 0.94,
            "subdiv": 9,
        }, open(cfg_path, "w"))
        subprocess.run(
            [BLENDER_BIN, "-b", "-P", BRAIN_BLENDER, "--", cfg_path, png_path],
            check=True, capture_output=True,
        )
        brain_imgs[key] = mpimg.imread(png_path)

graphite_cmap = colors.LinearSegmentedColormap.from_list(
    "graphite", [(p, rgb) for p, rgb in CMAPS["graphite"]])

img_h, img_w = next(iter(brain_imgs.values())).shape[:2]
panel_w = 4.4
fig, axes = plt.subplots(1, 2, figsize=(2 * panel_w + 1.0, panel_w * img_h / img_w),
                          gridspec_kw={"wspace": 0.05})
for ax, (key, label, _) in zip(axes, GROUPS):
    g = per_group[key]
    ax.imshow(brain_imgs[key])
    ax.axis("off")
    ax.set_title(f"{'a' if key == 'unknown' else 'b'}  {label} (n = {g['n_patients']:,})",
                 loc="left", fontsize=10, fontweight="bold")

sm = plt.cm.ScalarMappable(cmap=graphite_cmap, norm=plt.Normalize(0, vmax))
cbar = fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.02, aspect=25)
cbar.set_label("Mean CFA R² (outside QRS)", fontsize=8)

fig.suptitle("Figure 5. Scalp CFA topography by clinical subgroup", fontsize=10, y=0.98)
fig.savefig(os.path.join(FIG_DIR, "fig5_group_topomap.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "fig5_group_topomap.png"), bbox_inches="tight")
plt.close(fig)

with open(os.path.join(HERE, "paper_stats.json")) as f:
    S = json.load(f)
S["topomap_groups"] = {
    "min_electrodes": MIN_ELECTRODES, "min_patients_per_site": MIN_PATIENTS,
}
for key, g in per_group.items():
    S["topomap_groups"][key] = {
        "label": g["label"], "n_patients": g["n_patients"], "n_sites": len(g["present"]),
        "site_means": {c: float(v) for c, v in zip(g["present"], g["values"])},
    }
with open(os.path.join(HERE, "paper_stats.json"), "w") as f:
    json.dump(S, f, indent=2)
print("Figure written to", os.path.join(FIG_DIR, "fig5_group_topomap.png"))
