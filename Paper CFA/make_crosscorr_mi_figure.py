#!/usr/bin/env python3
"""Figure 5: EEG-ECG cross-correlation (lag-resolved) and mutual information,
pre- vs post-ICA — the direct answer to "how much of the EEG is heart
data": (a) cross-correlation vs. lag, core 4-electrode average; (b) peak
|cross-correlation| per electrode; (c) mutual information, core 4-electrode
average; (d) mutual information per electrode.

Correlation sign depends on channel/reference polarity (e.g. an "M1" vs "M2"
mastoid reference inverts the waveform), which is irrelevant to CFA
magnitude, so per-(patient, channel) curves are sign-aligned to a positive
peak before being averaged into the population curve in panel (a); panel (b)
already summarises with |r|.

Run: venv/bin/python "Paper CFA/make_crosscorr_mi_figure.py"
(requires make_crosscorr_mi.py to have been run first)
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
CORE_FOUR = ["F3", "F4", "C3", "C4"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300,
})

curve = pd.read_parquet(os.path.join(HERE, "crosscorr_curve.parquet"))
summary = pd.read_parquet(os.path.join(HERE, "crosscorr_mi_summary.parquet"))
curve["canon"] = canonicalize(curve["eeg_channel"])
summary["canon"] = canonicalize(summary["eeg_channel"])
curve = curve.dropna(subset=["canon"])
summary = summary.dropna(subset=["canon"])
curve = filter_min_coverage(curve, "patient_id", "canon", min_frac=MIN_COVERAGE)
summary = filter_min_coverage(summary, "patient_id", "canon", min_frac=MIN_COVERAGE)
print(f"curve: {len(curve):,} rows, {curve.patient_id.nunique():,} patients, {curve.canon.nunique()} sites")
print(f"summary: {len(summary):,} rows, {summary.patient_id.nunique():,} patients")

# Sign-align: within each (patient, recording, channel, condition), flip the
# whole lag curve if its peak (max |r|) is negative, so population averaging
# doesn't cancel out real structure via reference-polarity flips.
def sign_align(group: pd.DataFrame) -> pd.DataFrame:
    peak_idx = group["r"].abs().idxmax()
    sign = np.sign(group.loc[peak_idx, "r"]) or 1.0
    group = group.copy()
    group["r"] = group["r"] * sign
    return group

curve_aligned = curve.groupby(["patient_id", "recording_id", "eeg_channel", "condition"], group_keys=False).apply(sign_align)

fig, axes2d = plt.subplots(2, 2, figsize=(11, 8.8))

# ---------------------------------------------------------------------
# Panel a: cross-correlation vs. lag, core 4-electrode average, pre vs post
# ---------------------------------------------------------------------
ax = axes2d[0][0]
core_curve = curve_aligned[curve_aligned.canon.isin(CORE_FOUR)]
CONDITIONS = [("pre_ica", PALETTE[1], "Pre-ICA"), ("post_ica", PALETTE[2], "Post-ICA"),
              ("non_locked", PALETTE[7], "Non-heartbeat-locked control")]
for condition, color, label in CONDITIONS:
    sub = core_curve[core_curve.condition == condition]
    stats = sub.groupby("lag_ms")["r"].agg(["mean", "sem"])
    ax.plot(stats.index, stats["mean"], color=color, lw=1.8, label=label)
    ax.fill_between(stats.index, stats["mean"] - stats["sem"], stats["mean"] + stats["sem"],
                     color=color, alpha=0.25, linewidth=0)
ax.axvline(0, color="grey", lw=0.7, linestyle="--")
ax.axhline(0, color="grey", lw=0.5)
ax.set_xlabel("Lag (ms); ECG relative to EEG")
ax.set_ylabel("Cross-correlation r (sign-aligned, mean ± SEM)")
ax.set_title(f"a  Cross-correlation vs. lag ({'/'.join(CORE_FOUR)} average)", loc="left", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, frameon=False)

# ---------------------------------------------------------------------
# Panel b: peak |cross-correlation|, per electrode, pre vs post
# ---------------------------------------------------------------------
ax = axes2d[0][1]
per_site_cc = summary.groupby("canon").agg(
    pre=("peak_r_pre", lambda s: s.abs().mean()),
    post=("peak_r_post", lambda s: s.abs().mean()),
    non_locked=("peak_r_non_locked", lambda s: s.abs().mean()),
)
order = per_site_cc["pre"].sort_values(ascending=True).index.tolist()
y = np.arange(len(order))
h = 0.26
ax.barh(y + h, per_site_cc.loc[order, "pre"], height=h, color=PALETTE[1], label="Pre-ICA")
ax.barh(y, per_site_cc.loc[order, "post"], height=h, color=PALETTE[2], label="Post-ICA")
ax.barh(y - h, per_site_cc.loc[order, "non_locked"], height=h, color=PALETTE[7], label="Non-locked control")
ax.set_yticks(y)
ax.set_yticklabels(order, fontsize=7.5)
ax.set_xlabel("Mean peak |cross-correlation|")
ax.set_title(f"b  Peak |cross-correlation|, per electrode ({len(order)} sites)", loc="left", fontsize=10, fontweight="bold")
ax.legend(fontsize=7.5, frameon=False, loc="lower right")

# ---------------------------------------------------------------------
# Panel c: mutual information, core 4-electrode average, pre vs post
# ---------------------------------------------------------------------
ax = axes2d[1][0]
core_summary = summary[summary.canon.isin(CORE_FOUR)]
data = [core_summary["mi_pre"].dropna().values, core_summary["mi_post"].dropna().values,
        core_summary["mi_non_locked"].dropna().values]
bp = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=False)
for patch, color in zip(bp["boxes"], [PALETTE[1], PALETTE[2], PALETTE[7]]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for median in bp["medians"]:
    median.set_color("white")
    median.set_linewidth(1.2)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["Pre-ICA", "Post-ICA", "Non-locked\ncontrol"])
ax.set_ylabel("Mutual information (nats)")
ax.set_title(f"c  Mutual information ({'/'.join(CORE_FOUR)} average)", loc="left", fontsize=10, fontweight="bold")
mi_pre_med, mi_post_med = core_summary["mi_pre"].median(), core_summary["mi_post"].median()
mi_nl_med = core_summary["mi_non_locked"].median()
ax.text(0.5, 0.03, f"median {mi_pre_med:.2f} → {mi_post_med:.2f} → {mi_nl_med:.2f} nats",
        transform=ax.transAxes, ha="center", fontsize=7.5)

# ---------------------------------------------------------------------
# Panel d: mutual information, per electrode, pre vs post
# ---------------------------------------------------------------------
ax = axes2d[1][1]
per_site_mi = summary.groupby("canon").agg(pre=("mi_pre", "median"), post=("mi_post", "median"),
                                            non_locked=("mi_non_locked", "median"))
order_mi = [c for c in order if c in per_site_mi.index]
y2 = np.arange(len(order_mi))
ax.barh(y2 + h, per_site_mi.loc[order_mi, "pre"], height=h, color=PALETTE[1], label="Pre-ICA")
ax.barh(y2, per_site_mi.loc[order_mi, "post"], height=h, color=PALETTE[2], label="Post-ICA")
ax.barh(y2 - h, per_site_mi.loc[order_mi, "non_locked"], height=h, color=PALETTE[7], label="Non-locked control")
ax.set_yticks(y2)
ax.set_yticklabels(order_mi, fontsize=7.5)
ax.set_xlabel("Median mutual information (nats)")
ax.set_title(f"d  Mutual information, per electrode ({len(order_mi)} sites)", loc="left", fontsize=10, fontweight="bold")
ax.legend(fontsize=7.5, frameon=False, loc="lower right")

fig.suptitle(
    f"Figure S4. How much of the EEG is heart data? EEG-ECG cross-correlation and mutual\n"
    f"information: pre-ICA, post-ICA, and a non-heartbeat-locked control (n = {summary.patient_id.nunique():,} patients)",
    fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(FIG_DIR, "figS4_crosscorr_mi.pdf"))
fig.savefig(os.path.join(FIG_DIR, "figS4_crosscorr_mi.png"))
plt.close(fig)

with open(os.path.join(HERE, "paper_stats.json")) as f:
    S = json.load(f)
S["crosscorr_mi"] = {
    "n_patients": int(summary.patient_id.nunique()),
    "core_electrodes": CORE_FOUR,
    "core_peak_r_pre_mean": float(core_summary["peak_r_pre"].abs().mean()),
    "core_peak_r_post_mean": float(core_summary["peak_r_post"].abs().mean()),
    "core_peak_lag_ms_pre_median": float(core_summary["peak_lag_ms_pre"].median()),
    "core_peak_lag_ms_post_median": float(core_summary["peak_lag_ms_post"].median()),
    "core_peak_r_non_locked_mean": float(core_summary["peak_r_non_locked"].abs().mean()),
    "core_mi_pre_median": float(mi_pre_med), "core_mi_post_median": float(mi_post_med),
    "core_mi_non_locked_median": float(mi_nl_med),
    "per_site_peak_r_pre": {c: float(per_site_cc.loc[c, "pre"]) for c in order},
    "per_site_peak_r_post": {c: float(per_site_cc.loc[c, "post"]) for c in order},
    "per_site_peak_r_non_locked": {c: float(per_site_cc.loc[c, "non_locked"]) for c in order},
    "per_site_mi_pre": {c: float(per_site_mi.loc[c, "pre"]) for c in order_mi},
    "per_site_mi_post": {c: float(per_site_mi.loc[c, "post"]) for c in order_mi},
    "per_site_mi_non_locked": {c: float(per_site_mi.loc[c, "non_locked"]) for c in order_mi},
}
with open(os.path.join(HERE, "paper_stats.json"), "w") as f:
    json.dump(S, f, indent=2, default=float)
print(f"Core peak |r|: pre={core_summary['peak_r_pre'].abs().mean():.3f} post={core_summary['peak_r_post'].abs().mean():.3f}")
print(f"Core MI: pre={mi_pre_med:.3f} post={mi_post_med:.3f} nats")
print("Figure written to", os.path.join(FIG_DIR, "figS4_crosscorr_mi.png"))
