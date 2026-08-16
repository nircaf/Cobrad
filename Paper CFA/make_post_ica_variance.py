#!/usr/bin/env python3
"""Figure 5: HEP-evoked EEG variance and spectral entropy before vs. after
ICA-based ECG-component removal, plus a non-heartbeat-locked control
(make_non_locked_control.py / make_entropy_control.py: the same
quality-controlled windows re-epoched around random pseudo-events instead of
true R-peaks) — (a) variance, core 4-electrode average (F3, F4, C3, C4);
(b) variance, per electrode; (c) spectral entropy, core 4-electrode average;
(d) spectral entropy, per electrode.

Variance distributions are heavily right-skewed (a handful of high-variance
channels dominate a raw mean), so this reports medians on a log axis rather
than means for panels a/b; entropy (bounded 0-1) uses linear axes. Uses the
same channel-canonicalisation as make_topomap.py.

Run: venv/bin/python "Paper CFA/make_post_ica_variance.py"
(requires make_non_locked_control.py and make_entropy_control.py to have
been run first)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from channel_utils import canonicalize, filter_min_coverage

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00", "#000000"]
MIN_COVERAGE = 0.5
CORE_FOUR = ["F3", "F4", "C3", "C4"]  # canonical bilateral frontal-central quad: majority-montage coverage

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300,
})

df = pd.read_parquet(os.path.join(HERE, "ica_combined.parquet"))
df["canon"] = canonicalize(df["eeg_channel"])
df = df.dropna(subset=["canon"])
# One row per channel per (patient, component) currently; restrict to the
# ECG-flagged component's channel-level pre/post variance, which is constant
# across components for a given channel — dedupe to one row per
# patient/channel so channels aren't double-counted per component.
df = df.drop_duplicates(subset=["patient_id", "recording_id", "eeg_channel"])
df = filter_min_coverage(df, "patient_id", "canon", min_frac=MIN_COVERAGE)
print(f"{len(df):,} patient-channel rows, {df.patient_id.nunique():,} patients, "
      f"{df.canon.nunique()} scalp sites at >= {MIN_COVERAGE:.0%} patient coverage")

nl = pd.read_parquet(os.path.join(HERE, "non_locked_control.parquet"))
nl["canon"] = canonicalize(nl["eeg_channel"])
nl = nl.dropna(subset=["canon"])
nl = filter_min_coverage(nl, "patient_id", "canon", min_frac=MIN_COVERAGE)
print(f"non-locked control: {len(nl):,} patient-channel rows, {nl.patient_id.nunique():,} patients")

ent = pd.read_parquet(os.path.join(HERE, "entropy_control.parquet"))
ent["canon"] = canonicalize(ent["eeg_channel"])
ent = ent.dropna(subset=["canon"])
ent = filter_min_coverage(ent, "patient_id", "canon", min_frac=MIN_COVERAGE)
print(f"entropy control: {len(ent):,} patient-channel rows, {ent.patient_id.nunique():,} patients")

fig, axes2d = plt.subplots(2, 2, figsize=(11, 8.8))
axes = axes2d[0]

# ---------------------------------------------------------------------
# Panel a: core 4-electrode average, pre vs post vs non-locked control
# ---------------------------------------------------------------------
core = df[df.canon.isin(CORE_FOUR)]
core_nl = nl[nl.canon.isin(CORE_FOUR)]
ax = axes[0]
data = [core["channel_hep_variance_pre_ica"].clip(lower=1e-3).values,
        core["channel_hep_variance_post_ica"].clip(lower=1e-3).values,
        core_nl["channel_hep_variance_non_locked"].clip(lower=1e-3).values]
bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
for patch, color in zip(bp["boxes"], [PALETTE[1], PALETTE[2], PALETTE[7]]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for median in bp["medians"]:
    median.set_color("white")
    median.set_linewidth(1.2)
ax.set_yscale("log")
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["Pre-ICA", "Post-ICA\n(ECG component removed)", "Non-heartbeat-locked\ncontrol"])
ax.set_ylabel("Evoked variance (µV², log scale)")
ax.set_title(f"a  Core 4-electrode average ({'/'.join(CORE_FOUR)})", loc="left", fontsize=10, fontweight="bold")
pre_med, post_med = core["channel_hep_variance_pre_ica"].median(), core["channel_hep_variance_post_ica"].median()
nl_med = core_nl["channel_hep_variance_non_locked"].median()
pct_drop = (pre_med - post_med) / pre_med * 100
ax.text(0.5, 0.02, f"median {pre_med:.2f} → {post_med:.2f} → {nl_med:.2f} µV²  ({pct_drop:.0f}% drop pre→post)",
        transform=ax.transAxes, ha="center", fontsize=7.5)

# ---------------------------------------------------------------------
# Panel b: per-electrode median pre/post/non-locked, all canonical sites
# ---------------------------------------------------------------------
per_site = df.groupby("canon").agg(
    pre_median=("channel_hep_variance_pre_ica", "median"),
    post_median=("channel_hep_variance_post_ica", "median"),
    n=("channel_hep_variance_post_ica", "size"),
)
nl_per_site = nl.groupby("canon")["channel_hep_variance_non_locked"].median()
per_site["non_locked_median"] = nl_per_site.reindex(per_site.index)
order = per_site["pre_median"].sort_values(ascending=True).index.tolist()
y = np.arange(len(order))
ax = axes[1]
h = 0.26
ax.barh(y + h, per_site.loc[order, "pre_median"], height=h, color=PALETTE[1], label="Pre-ICA")
ax.barh(y, per_site.loc[order, "post_median"], height=h, color=PALETTE[2], label="Post-ICA")
ax.barh(y - h, per_site.loc[order, "non_locked_median"], height=h, color=PALETTE[7], label="Non-locked control")
ax.set_yticks(y)
ax.set_yticklabels(order, fontsize=7.5)
ax.set_xscale("log")
xmin = per_site.loc[order, ["pre_median", "post_median", "non_locked_median"]].min().min()
xmax = per_site.loc[order, ["pre_median", "post_median", "non_locked_median"]].max().max()
ax.set_xlim(xmin * 0.7, xmax * 1.3)
ax.set_xlabel("Median evoked variance (µV², log scale)")
ax.set_title(f"b  Per-electrode ({len(order)} sites)", loc="left", fontsize=10, fontweight="bold")
ax.legend(fontsize=7.5, frameon=False, loc="lower right")

# ---------------------------------------------------------------------
# Panel c: spectral entropy, core 4-electrode average, pre vs post vs non-locked
# ---------------------------------------------------------------------
axes = axes2d[1]
core_ent = ent[ent.canon.isin(CORE_FOUR)]
ax = axes[0]
data = [core_ent["entropy_pre_ica"].dropna().values,
        core_ent["entropy_post_ica"].dropna().values,
        core_ent["entropy_non_locked"].dropna().values]
bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
for patch, color in zip(bp["boxes"], [PALETTE[1], PALETTE[2], PALETTE[7]]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for median in bp["medians"]:
    median.set_color("white")
    median.set_linewidth(1.2)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["Pre-ICA", "Post-ICA\n(ECG component removed)", "Non-heartbeat-locked\ncontrol"])
ax.set_ylabel("Spectral entropy (normalised, 0-1)")
ax.set_title(f"c  Entropy, core 4-electrode average ({'/'.join(CORE_FOUR)})", loc="left", fontsize=10, fontweight="bold")
ent_pre_med = core_ent["entropy_pre_ica"].median()
ent_post_med = core_ent["entropy_post_ica"].median()
ent_nl_med = core_ent["entropy_non_locked"].median()
ax.text(0.5, 0.03, f"median {ent_pre_med:.2f} → {ent_post_med:.2f} → {ent_nl_med:.2f}",
        transform=ax.transAxes, ha="center", fontsize=7.5)

# ---------------------------------------------------------------------
# Panel d: spectral entropy, per electrode, all canonical sites
# ---------------------------------------------------------------------
ent_per_site = ent.groupby("canon").agg(
    pre_entropy=("entropy_pre_ica", "median"),
    post_entropy=("entropy_post_ica", "median"),
    non_locked_entropy=("entropy_non_locked", "median"),
)
ent_order = [c for c in order if c in ent_per_site.index]
y2 = np.arange(len(ent_order))
ax = axes[1]
ax.barh(y2 + h, ent_per_site.loc[ent_order, "pre_entropy"], height=h, color=PALETTE[1], label="Pre-ICA")
ax.barh(y2, ent_per_site.loc[ent_order, "post_entropy"], height=h, color=PALETTE[2], label="Post-ICA")
ax.barh(y2 - h, ent_per_site.loc[ent_order, "non_locked_entropy"], height=h, color=PALETTE[7], label="Non-locked control")
ax.set_yticks(y2)
ax.set_yticklabels(ent_order, fontsize=7.5)
ax.set_xlabel("Median spectral entropy (0-1)")
ax.set_title(f"d  Entropy, per-electrode ({len(ent_order)} sites)", loc="left", fontsize=10, fontweight="bold")
ax.legend(fontsize=7.5, frameon=False, loc="lower right")

fig.suptitle(
    f"Figure S3. HEP-evoked EEG variance and spectral entropy vs. a non-heartbeat-locked noise floor\n"
    f"(n = {df.patient_id.nunique():,} patients ICA/variance; {nl.patient_id.nunique():,} patients "
    f"variance control; {ent.patient_id.nunique():,} patients entropy control)",
    fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(FIG_DIR, "figS3_post_ica_variance.pdf"))
fig.savefig(os.path.join(FIG_DIR, "figS3_post_ica_variance.png"))
plt.close(fig)

with open(os.path.join(HERE, "paper_stats.json")) as f:
    S = json.load(f)
S["post_ica_variance"] = {
    "n_patients": int(df.patient_id.nunique()), "core_electrodes": CORE_FOUR,
    "core_pre_median": float(pre_med), "core_post_median": float(post_med), "core_pct_drop": float(pct_drop),
    "core_non_locked_median": float(nl_med),
    "n_patients_control": int(nl.patient_id.nunique()),
    "per_site_pre_median": {c: float(per_site.loc[c, "pre_median"]) for c in order},
    "per_site_post_median": {c: float(per_site.loc[c, "post_median"]) for c in order},
    "per_site_non_locked_median": {c: float(per_site.loc[c, "non_locked_median"]) for c in order},
    "n_patients_entropy": int(ent.patient_id.nunique()),
    "core_entropy_pre_median": float(ent_pre_med), "core_entropy_post_median": float(ent_post_med),
    "core_entropy_non_locked_median": float(ent_nl_med),
    "per_site_entropy_pre_median": {c: float(ent_per_site.loc[c, "pre_entropy"]) for c in ent_order},
    "per_site_entropy_post_median": {c: float(ent_per_site.loc[c, "post_entropy"]) for c in ent_order},
    "per_site_entropy_non_locked_median": {c: float(ent_per_site.loc[c, "non_locked_entropy"]) for c in ent_order},
}
with open(os.path.join(HERE, "paper_stats.json"), "w") as f:
    json.dump(S, f, indent=2, default=float)
print(f"Core 4-electrode median: {pre_med:.3f} -> {post_med:.3f} uV^2 ({pct_drop:.0f}% drop)")
print("Figure written to", os.path.join(FIG_DIR, "figS3_post_ica_variance.png"))
