"""
Build Fig 1-3 for the age x sleep-stage HEP paper from Paper1/analysis_results_clean.pkl.
Run: source venv/bin/activate && python3 Paper1/make_figures.py
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

REPO = "/storage/pblab_shared_data2/Nir/Cobrad"
OUT_DIR = os.path.join(REPO, "Paper1")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

STAGES = ["light_sleep", "N3", "R"]
STAGE_LABELS = {"light_sleep": "Light Sleep", "N3": "N3 (SWS)", "R": "REM"}
PALETTE = {"light_sleep": "#0072B2", "N3": "#D55E00", "R": "#009E73"}
AGEBAND_PALETTE = ["#56B4E9", "#E69F00", "#CC79A7"]
QRS_WINDOW = (-0.05, 0.05)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

with open(os.path.join(OUT_DIR, "analysis_results_clean.pkl"), "rb") as f:
    R = pickle.load(f)

df = R["df"]
times_ref = R["times_ref"]
per_stage_waveform = R["per_stage_waveform"]
rms_age_stats = R["rms_age_stats"]
contrast_stats = R["contrast_stats"]

# tertile age bands (data-driven, robust to whatever n this cohort has)
q = df["age"].quantile([1 / 3, 2 / 3]).values
band_edges = [df["age"].min(), q[0], q[1], df["age"].max()]
band_labels = [
    f"{band_edges[0]:.0f}-{band_edges[1]:.0f} yr",
    f"{band_edges[1]:.0f}-{band_edges[2]:.0f} yr",
    f"{band_edges[2]:.0f}-{band_edges[3]:.0f} yr",
]


def age_band(age):
    if age <= band_edges[1]:
        return 0
    elif age <= band_edges[2]:
        return 1
    return 2


df["age_band"] = df["age"].apply(age_band)

# ---------------------------------------------------------------------
# Figure 1: 3-panel grand-average waveform by stage, split by age tertile
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
for ax, stage in zip(axes, STAGES):
    wf_map = per_stage_waveform[stage]
    for band in range(3):
        pids = df.loc[df["age_band"] == band, "patient_id"]
        mats = [wf_map[pid] for pid in pids if pid in wf_map]
        if not mats:
            continue
        mats = np.vstack(mats)
        mean_wf = np.nanmean(mats, axis=0)
        sem_wf = np.nanstd(mats, axis=0) / np.sqrt(mats.shape[0])
        ax.plot(times_ref, mean_wf, color=AGEBAND_PALETTE[band], lw=1.4,
                label=f"{band_labels[band]} (n={mats.shape[0]})")
        ax.fill_between(times_ref, mean_wf - sem_wf, mean_wf + sem_wf,
                         color=AGEBAND_PALETTE[band], alpha=0.25, linewidth=0)
    ax.axvspan(QRS_WINDOW[0], QRS_WINDOW[1], color="grey", alpha=0.25, zorder=0)
    ax.axvline(0, color="k", lw=0.7, linestyle="--")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_title(STAGE_LABELS[stage], fontsize=10, fontweight="bold")
    ax.set_xlabel("Time from R-peak (s)")
    ax.legend(fontsize=6.5, loc="upper right", frameon=False)
axes[0].set_ylabel("HEP amplitude (µV)")
fig.suptitle("Figure 1. Grand-average HEP by sleep stage and age tertile", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(FIG_DIR, "fig1_waveforms_by_age_band.pdf"))
fig.savefig(os.path.join(FIG_DIR, "fig1_waveforms_by_age_band.png"))
plt.close(fig)

# ---------------------------------------------------------------------
# Figure 2: RMS vs age scatter + regression, one panel per stage
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
for ax, stage in zip(axes, STAGES):
    x = df["age"].values
    y = df[f"rms_{stage}"].values
    ax.scatter(x, y, s=8, alpha=0.35, color=PALETTE[stage], edgecolor="none")
    lr = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, lr.intercept + lr.slope * xs, color="black", lw=1.5)
    st_ = rms_age_stats[stage]
    ax.text(0.03, 0.95, f"r={st_['r']:.2f}, p={st_['p_r']:.2g}\nn={st_['n']}",
            transform=ax.transAxes, va="top", fontsize=8)
    ax.set_title(STAGE_LABELS[stage], fontsize=10, fontweight="bold")
    ax.set_xlabel("Age (years)")
axes[0].set_ylabel("Post-QRS RMS amplitude (µV)")
fig.suptitle("Figure 2. HEP RMS amplitude vs age, by sleep stage", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(FIG_DIR, "fig2_rms_vs_age.pdf"))
fig.savefig(os.path.join(FIG_DIR, "fig2_rms_vs_age.png"))
plt.close(fig)

# ---------------------------------------------------------------------
# Figure 3: age x stage interaction -- mean RMS per age band, one line/stage
#           + contrast-vs-age panel
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
ax = axes[0]
band_centers = []
for band in range(3):
    sub = df[df["age_band"] == band]
    band_centers.append(sub["age"].mean())
for stage in STAGES:
    means, sems = [], []
    for band in range(3):
        sub = df.loc[df["age_band"] == band, f"rms_{stage}"]
        means.append(sub.mean())
        sems.append(sub.std() / np.sqrt(len(sub)))
    ax.errorbar(band_centers, means, yerr=sems, marker="o", capsize=3,
                color=PALETTE[stage], label=STAGE_LABELS[stage], lw=1.6)
ax.set_xlabel("Mean age in band (years)")
ax.set_ylabel("Post-QRS RMS amplitude (µV)")
ax.set_title("A. Stage modulation across age bands", fontsize=9, fontweight="bold")
ax.legend(fontsize=7.5, frameon=False)

ax2 = axes[1]
labels = []
rvals = []
pvals = []
for k in ["rms_R_minus_N3", "rms_light_sleep_minus_N3", "rms_R_minus_light_sleep"]:
    v = contrast_stats[k]
    labels.append(k.replace("rms_", "").replace("_minus_", " − ").replace("light_sleep", "Light"))
    rvals.append(v["r"])
    pvals.append(v["p_r"])
colors = [PALETTE["R"], PALETTE["light_sleep"], "#000000"]
bars = ax2.bar(labels, rvals, color=colors[: len(labels)])
for b, p in zip(bars, pvals):
    ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + (0.01 if b.get_height() >= 0 else -0.03),
              f"p={p:.2g}", ha="center", fontsize=7)
ax2.axhline(0, color="grey", lw=0.6)
ax2.set_ylabel("Pearson r (contrast RMS vs age)")
ax2.set_title("B. Does the stage gap change with age?", fontsize=9, fontweight="bold")
ax2.tick_params(axis="x", rotation=20)

fig.suptitle("Figure 3. Age x sleep-stage HEP modulation", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(FIG_DIR, "fig3_interaction.pdf"))
fig.savefig(os.path.join(FIG_DIR, "fig3_interaction.png"))
plt.close(fig)

print("Figures written to", FIG_DIR)
for f in sorted(os.listdir(FIG_DIR)):
    print(" ", f, os.path.getsize(os.path.join(FIG_DIR, f)))
