#!/usr/bin/env python3
"""Figure 5: scalp topography of BMI-adjusted CFA R^2. A single OLS model
(cfa_r2_excl_qrs ~ bmi) is fit across every channel-recording with a known
BMI; the residual from that model is the BMI-adjusted CFA measure -- positive
means more cardiac field artifact than expected for that BMI, negative means
less. Residuals are averaged per canonical site for two groups: patients with
clinical obesity, and patients with depression and/or anxiety.

Run: venv/bin/python "Paper CFA/make_topomap_bmi_adjusted.py"
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
import statsmodels.formula.api as smf

from channel_utils import CANON, canonicalize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from eeg_brain_blender import CMAPS  # noqa: E402 (bpy-free import: colour stops only)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
MIN_PATIENTS = 10  # per-site minimum within a group
BRAIN_BLENDER = os.path.join(HERE, "..", "utils", "eeg_brain_blender.py")
BLENDER_BIN = os.environ.get("BLENDER_BIN") or shutil.which("blender") or "blender"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300,
})

cfa = pd.read_parquet(os.path.join(HERE, "cfa_combined.parquet"))
demo = pd.read_parquet(os.path.join(HERE, "demographics_combined.parquet"))
bmi = pd.read_parquet(os.path.join(HERE, "bmi_combined.parquet"))
cfa["canon"] = canonicalize(cfa["eeg_channel"])
cfa = cfa.dropna(subset=["canon"])
cfa = cfa.merge(demo[["patient_id", "bdsp_patient_id", "diagnosis_categories"]], on="patient_id", how="left")
cfa = cfa.merge(bmi[["bdsp_patient_id", "bmi"]], on="bdsp_patient_id", how="inner")
cfa = cfa.dropna(subset=["bmi", "cfa_r2_excl_qrs"])

model = smf.ols("cfa_r2_excl_qrs ~ bmi", data=cfa).fit()
cfa["resid"] = model.resid
print(f"cfa_r2_excl_qrs ~ bmi: n = {len(cfa):,}, slope = {model.params['bmi']:.4g}, "
      f"p = {model.pvalues['bmi']:.2g}")

dx = cfa.explode("diagnosis_categories")
obese_ids = set(dx.loc[dx.diagnosis_categories == "Obesity", "patient_id"])
depanx_ids = set(dx.loc[dx.diagnosis_categories.isin(["Depression", "Anxiety"]), "patient_id"])

GROUPS = [
    ("obesity", "Clinical obesity", cfa[cfa.patient_id.isin(obese_ids)]),
    ("dep_anx", "Depression / anxiety", cfa[cfa.patient_id.isin(depanx_ids)]),
]

per_group = {}
for key, label, sub in GROUPS:
    site_stats = sub.groupby("canon").agg(
        mean=("resid", "mean"), n_patients=("patient_id", "nunique"))
    present = [c for c in dict.fromkeys(CANON.values())
               if c in site_stats.index and site_stats.loc[c, "n_patients"] >= MIN_PATIENTS]
    per_group[key] = {
        "label": label, "n_patients": int(sub.patient_id.nunique()), "present": present,
        "values": np.array([site_stats.loc[c, "mean"] for c in present]),
    }
    print(f"{label}: {sub.patient_id.nunique():,} patients, {len(present)} sites "
          f">= {MIN_PATIENTS} patients/site")

vabs = float(max(np.abs(g["values"]).max() for g in per_group.values()))

# ---------------------------------------------------------------------
# Render both groups via utils/eeg_brain_blender.py, diverging colour
# scale shared across panels (0 = expected CFA for that BMI).
# ---------------------------------------------------------------------
brain_imgs = {}
with tempfile.TemporaryDirectory() as tmp:
    for key, g in per_group.items():
        cfg_path = os.path.join(tmp, f"{key}.json")
        png_path = os.path.join(tmp, f"{key}.png")
        json.dump({
            "channels": {c: float(v) for c, v in zip(g["present"], g["values"])},
            "view": "top", "cmap": "blue_white_red", "vmin": -vabs, "vmax": vabs,
            "show_electrodes": False, "resolution": [1100, 1450], "fill": 0.94,
            "subdiv": 9,
        }, open(cfg_path, "w"))
        subprocess.run(
            [BLENDER_BIN, "-b", "-P", BRAIN_BLENDER, "--", cfg_path, png_path],
            check=True, capture_output=True,
        )
        brain_imgs[key] = mpimg.imread(png_path)

bwr_cmap = colors.LinearSegmentedColormap.from_list(
    "blue_white_red", [(p, rgb) for p, rgb in CMAPS["blue_white_red"]])

img_h, img_w = next(iter(brain_imgs.values())).shape[:2]
panel_w = 4.4
fig, axes = plt.subplots(1, 2, figsize=(2 * panel_w + 1.0, panel_w * img_h / img_w),
                          gridspec_kw={"wspace": 0.05})
for ax, (key, label, _) in zip(axes, GROUPS):
    g = per_group[key]
    ax.imshow(brain_imgs[key])
    ax.axis("off")
    letter = {"obesity": "a", "dep_anx": "b"}[key]
    ax.set_title(f"{letter}  {label} (n = {g['n_patients']:,})",
                 loc="left", fontsize=10, fontweight="bold")

sm = plt.cm.ScalarMappable(cmap=bwr_cmap, norm=plt.Normalize(-vabs, vabs))
cbar = fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.02, aspect=25)
cbar.set_label("BMI-adjusted CFA R² (residual)", fontsize=8)

fig.suptitle("Figure 5. BMI-adjusted scalp CFA topography", fontsize=10, y=0.98)
fig.savefig(os.path.join(FIG_DIR, "fig5_bmi_adjusted_topomap.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "fig5_bmi_adjusted_topomap.png"), bbox_inches="tight")
plt.close(fig)

with open(os.path.join(HERE, "paper_stats.json")) as f:
    S = json.load(f)
S["topomap_bmi_adjusted"] = {
    "min_patients_per_site": MIN_PATIENTS,
    "model_n": int(len(cfa)), "model_slope": float(model.params["bmi"]),
    "model_p": float(model.pvalues["bmi"]), "model_r2": float(model.rsquared),
}
for key, g in per_group.items():
    S["topomap_bmi_adjusted"][key] = {
        "label": g["label"], "n_patients": g["n_patients"], "n_sites": len(g["present"]),
        "site_means": {c: float(v) for c, v in zip(g["present"], g["values"])},
    }
with open(os.path.join(HERE, "paper_stats.json"), "w") as f:
    json.dump(S, f, indent=2)
print("Figure written to", os.path.join(FIG_DIR, "fig5_bmi_adjusted_topomap.png"))
