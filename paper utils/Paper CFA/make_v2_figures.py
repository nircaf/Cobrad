#!/usr/bin/env python3
"""Figures and stats for the v2 (general CFA-in-EEG) manuscript.

  Figure 1  fig1_patient_examples.png  two example patients (low BMI / low CFA,
            high BMI / high CFA): MNE topomaps of per-channel CFA R^2 at every
            segment duration, plus the 60 - 5 min difference map.
  Figure 2  fig2_cfa_overview.png      old Figures 1 + 2 combined: R^2
            distribution, R^2 vs. ICA SNR, MNE scalp topomap of mean R^2
            per site, and the per-site distribution.

Also writes paper_stats.json["v2"]: the example-patient metadata and a
Friedman test (+ paired Wilcoxon, Holm-corrected) across segment durations.

Run: venv/bin/python "paper utils/Paper CFA/make_v2_figures.py"
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from channel_utils import CHANNEL_ORDER, canonicalize

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00", "#000000"]
MIN_PATIENTS = 500  # same site floor as make_topomap.py
DURATION_FILES = {
    5: "cfa_variance_explained_5min.parquet", 10: "cfa_combined.parquet",
    20: "cfa_variance_explained_20min.parquet", 30: "cfa_variance_explained_30min.parquet",
    45: "cfa_variance_explained_45min.parquet", 60: "cfa_variance_explained_60min.parquet",
}
# Chosen from the HSP patients with BMI and data at every duration (one
# recording each): sex-matched men at the two BMI extremes with low vs. high CFA.
EXAMPLES = [("A", "I0002150033042"), ("B", "I0002150024023")]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300,
})
MONTAGE = mne.channels.make_standard_montage("standard_1020")


def topo(ax, site_values, vlim, cmap, names=True):
    """MNE head-circle topomap of {site: value}, electrode names drawn on it."""
    sites = [c for c in CHANNEL_ORDER if c in site_values]
    info = mne.create_info(sites, 100.0, "eeg")
    info.set_montage(MONTAGE)
    im, _ = mne.viz.plot_topomap(
        np.array([site_values[c] for c in sites]), info, axes=ax, show=False, cmap=cmap,
        vlim=vlim, contours=0, sensors=not names, names=sites if names else None, extrapolate="head",
    )
    for t in ax.texts:
        t.set_fontsize(5.5)
    return im


durations = {}
for minutes, fname in DURATION_FILES.items():
    d = pd.read_parquet(os.path.join(HERE, fname),
                        columns=["patient_id", "recording_id", "eeg_channel", "cfa_r2_excl_qrs"])
    d["canon"] = canonicalize(d["eeg_channel"])
    durations[minutes] = d

# =============================================================================
# Figure 1: example patients across segment durations
# =============================================================================
demo = pd.read_parquet(os.path.join(HERE, "demographics_combined.parquet"))
bmi = pd.read_parquet(os.path.join(HERE, "bmi_combined.parquet"))
minutes_all = sorted(DURATION_FILES)
fig, axes = plt.subplots(2, len(minutes_all) + 1, figsize=(11, 3.9))
examples_meta = []
for row, (label, pid) in enumerate(EXAMPLES):
    rec = durations[60].loc[durations[60].patient_id == pid, "recording_id"].iat[0]
    maps = {}
    for m in minutes_all:
        d = durations[m]
        d = d[(d.patient_id == pid) & (d.recording_id == rec)].dropna(subset=["canon"])
        maps[m] = d.groupby("canon")["cfa_r2_excl_qrs"].mean().to_dict()
    sites = set.intersection(*(set(v) for v in maps.values()))
    maps = {m: {c: v[c] for c in sites} for m, v in maps.items()}
    for col, m in enumerate(minutes_all):
        im_r2 = topo(axes[row, col], maps[m], (0, 1), "Reds")
        axes[row, col].set_title(f"{m} min\nmean R² = {np.mean(list(maps[m].values())):.2f}", fontsize=8)
    diff = {c: maps[60][c] - maps[5][c] for c in sites}
    im_diff = topo(axes[row, -1], diff, (-0.3, 0.3), "RdBu_r")
    axes[row, -1].set_title(f"Δ 60 − 5 min\nmean Δ = {np.mean(list(diff.values())):+.2f}", fontsize=8)
    p = demo[demo.patient_id == pid].iloc[0]
    p_bmi = float(bmi.loc[bmi.bdsp_patient_id == p.bdsp_patient_id, "bmi"].iat[0])
    axes[row, 0].text(-0.45, 0.5, f"Patient {label}\n{p.sex}, {p.age:.0f} y\nBMI {p_bmi:.1f}",
                      transform=axes[row, 0].transAxes, ha="center", va="center",
                      fontsize=8.5, fontweight="bold")
    examples_meta.append({
        "label": label, "sex": str(p.sex), "age": float(p.age), "bmi": p_bmi, "n_sites": len(sites),
        "mean_r2": {str(m): float(np.mean(list(maps[m].values()))) for m in minutes_all},
    })
fig.subplots_adjust(left=0.1, right=0.86, top=0.84, bottom=0.04, wspace=0.15, hspace=0.45)
cb = fig.colorbar(im_r2, cax=fig.add_axes([0.875, 0.12, 0.01, 0.62]))
cb.set_label("CFA R² (outside QRS)", fontsize=7.5)
cb2 = fig.colorbar(im_diff, cax=fig.add_axes([0.945, 0.12, 0.01, 0.62]))
cb2.set_label("Δ R² (60 − 5 min)", fontsize=7.5)
fig.suptitle("Figure 1. Cardiac field artifact in two individual patients across segment durations",
             fontsize=10)
fig.savefig(os.path.join(FIG_DIR, "fig1_patient_examples.png"), bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "fig1_patient_examples.pdf"), bbox_inches="tight")
plt.close(fig)

# =============================================================================
# Segment-duration test: patient-mean R^2 (all channels) in patients present
# at every duration, as in make_dose_response.py
# =============================================================================
common = set.intersection(*(set(d.patient_id) for d in durations.values()))
pt = pd.DataFrame({m: d[d.patient_id.isin(common)].groupby("patient_id")["cfa_r2_excl_qrs"].mean()
                   for m, d in durations.items()}).dropna()
fr_stat, fr_p = stats.friedmanchisquare(*(pt[m] for m in minutes_all))
pairs = list(zip(minutes_all[:-1], minutes_all[1:])) + [(5, 60), (10, 60)]
wil = [stats.wilcoxon(pt[a], pt[b]) for a, b in pairs]
p_holm = multipletests([w.pvalue for w in wil], method="holm")[1]
window_test = {
    "n": int(len(pt)), "friedman_chi2": float(fr_stat), "friedman_p": float(fr_p),
    "kendall_w": float(fr_stat / (len(pt) * (len(minutes_all) - 1))),
    "pairs": [{"a": a, "b": b, "mean_diff": float((pt[b] - pt[a]).mean()),
               "pct_patients_increase": float((pt[b] > pt[a]).mean() * 100), "p_holm": float(ph)}
              for (a, b), ph in zip(pairs, p_holm)],
}
print(json.dumps(window_test, indent=1))

# =============================================================================
# Figure 2: cohort-wide CFA (old Figures 1 + 2)
# =============================================================================
cfa = durations[10]
cfa_full = pd.read_parquet(os.path.join(HERE, "cfa_combined.parquet"))
ica = pd.read_parquet(os.path.join(HERE, "ica_combined.parquet"))
fig = plt.figure(figsize=(11, 8.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.35], hspace=0.38, wspace=0.3)

ax = fig.add_subplot(gs[0, 0])
data = [cfa_full["cfa_r2_full_epoch"].dropna().values, cfa_full["cfa_r2_excl_qrs"].dropna().values]
parts = ax.violinplot(data, showmedians=True, widths=0.8)
for pc, color in zip(parts["bodies"], [PALETTE[1], PALETTE[0]]):
    pc.set_facecolor(color)
    pc.set_alpha(0.6)
    pc.set_edgecolor("black")
    pc.set_linewidth(0.6)
for key in ("cmedians", "cbars", "cmins", "cmaxes"):
    parts[key].set_color("black")
    parts[key].set_linewidth(0.8)
ax.set_xticks([1, 2])
ax.set_xticklabels(["Full HEP epoch\n(−300 to 400 ms)", "Outside ±50 ms\nQRS window"])
ax.set_ylabel("Channel CFA R² (HEP vs. ECG evoked average)")
ax.set_title("a", loc="left", fontsize=10, fontweight="bold")
_lo, _hi = ax.get_ylim()
ax.set_ylim(_lo, _hi + 0.18 * (_hi - _lo))
ax.text(0.03, 0.97,
        f"full: {data[0].mean():.2f} ± {data[0].std():.2f}\nexcl. QRS: {data[1].mean():.2f} ± {data[1].std():.2f}",
        transform=ax.transAxes, va="top", fontsize=7.5)

ax = fig.add_subplot(gs[0, 1])
ica_all = ica[ica.ecg_artifact_component].copy()
ica_all["snr"] = ica_all["channel_component_variance_fraction"] / (1 - ica_all["channel_component_variance_fraction"])
ica_all = ica_all.sort_values("channel_component_variance_fraction", ascending=False).drop_duplicates(
    ["patient_id", "edf_path", "eeg_channel"])
merged = ica_all[["patient_id", "edf_path", "eeg_channel", "snr"]].merge(
    cfa_full[["patient_id", "edf_path", "eeg_channel", "cfa_r2_excl_qrs"]],
    on=["patient_id", "edf_path", "eeg_channel"]).dropna(subset=["snr", "cfa_r2_excl_qrs"])
x_r2, y_snr = merged["cfa_r2_excl_qrs"].values, merged["snr"].values
r_sc, p_sc = stats.pearsonr(x_r2, np.log10(y_snr))
slope, intercept = np.polyfit(x_r2, np.log10(y_snr), 1)
ax.scatter(x_r2, y_snr, s=3, alpha=0.08, color=PALETTE[2], edgecolor="none", rasterized=True)
xs = np.linspace(0, 1, 100)
ax.plot(xs, 10 ** (slope * xs + intercept), color=PALETTE[1], lw=1.8)
ax.set_yscale("log")
ax.set_xlabel("Model-free CFA R² (outside QRS)")
ax.set_ylabel("ICA-attributed CFA SNR\n(variance ratio, CFA:residual)")
ax.set_title("b", loc="left", fontsize=10, fontweight="bold")
ax.text(0.03, 0.05, f"r (log SNR) = {r_sc:.2f}, p {'< 1e-300' if p_sc == 0 else f'= {p_sc:.2g}'}\n"
        f"n = {len(merged):,}", transform=ax.transAxes, va="bottom", fontsize=7.5)

site = cfa.dropna(subset=["canon"])
site_n = site.groupby("canon")["patient_id"].nunique()
site = site[site.canon.isin(site_n[site_n >= MIN_PATIENTS].index)]
per_site = site.groupby("canon")["cfa_r2_excl_qrs"].agg(["mean", "median"])

ax = fig.add_subplot(gs[1, 0])
im = topo(ax, per_site["mean"].to_dict(), (0, float(per_site["mean"].max())), "Reds")
cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
cb.set_label("Mean CFA R² (outside QRS)", fontsize=8)
ax.set_title("c", loc="left", fontsize=10, fontweight="bold")

ax = fig.add_subplot(gs[1, 1])
order = per_site["median"].sort_values().index.tolist()
box_data = [site.loc[site.canon == c, "cfa_r2_excl_qrs"].dropna().values for c in order]
bp = ax.boxplot(box_data, tick_labels=[f"{c} (n={len(v):,})" for c, v in zip(order, box_data)],
                vert=False, patch_artist=True, widths=0.65, showfliers=False, whis=0,
                medianprops={"color": PALETTE[6]})
for line in bp["whiskers"] + bp["caps"]:
    line.set_visible(False)
m = per_site.loc[order, "mean"].values
for patch, v in zip(bp["boxes"], (m - m.min()) / (m.max() - m.min())):
    patch.set_facecolor(plt.get_cmap("Reds")(0.25 + 0.65 * v))
ax.set_xlabel("CFA R² (outside QRS window)")
ax.set_title("d", loc="left", fontsize=10, fontweight="bold")
ax.tick_params(axis="y", labelsize=7.5)

fig.suptitle(f"Figure 2. Cardiac field artifact across the cohort "
             f"(n = {cfa_full.patient_id.nunique():,} patients, {len(cfa_full):,} channel-recordings)",
             fontsize=10)
fig.savefig(os.path.join(FIG_DIR, "fig2_cfa_overview.png"), bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "fig2_cfa_overview.pdf"), bbox_inches="tight")
plt.close(fig)

# Does heart rate (beats per window) explain the BMI association? 30-min
# windows, patient mean over the six well-covered sites, as in Figure 3c.
import statsmodels.formula.api as smf
from channel_utils import filter_min_coverage
hr = pd.read_parquet(os.path.join(HERE, DURATION_FILES[30]),
                     columns=["patient_id", "eeg_channel", "cfa_r2_excl_qrs", "qc_ecg_bpm"])
hr["canon"] = canonicalize(hr["eeg_channel"])
hr = filter_min_coverage(hr.dropna(subset=["canon"]), "patient_id", "canon", min_frac=0.5)
hr = hr.groupby("patient_id", as_index=False).agg(r2=("cfa_r2_excl_qrs", "mean"), bpm=("qc_ecg_bpm", "mean"))
hr["bdsp_patient_id"] = pd.to_numeric(hr.patient_id.str.extract(r"I\d{4}(\d{9})", expand=False))
hr = hr.merge(bmi, on="bdsp_patient_id").dropna()
r_bmi_bpm, p_bmi_bpm = stats.pearsonr(hr.bmi, hr.bpm)
m_unadj = smf.ols("r2 ~ bmi", hr).fit()
m_adj = smf.ols("r2 ~ bmi + bpm", hr).fit()
bmi_hr = {"n": int(len(hr)), "r_bmi_bpm": float(r_bmi_bpm), "p_bmi_bpm": float(p_bmi_bpm),
          "bmi_coef_unadj": float(m_unadj.params["bmi"]), "bmi_coef_adj": float(m_adj.params["bmi"]),
          "bmi_p_adj": float(m_adj.pvalues["bmi"])}
print(bmi_hr)

with open(os.path.join(HERE, "paper_stats.json")) as f:
    S = json.load(f)
S["v2"] = {"examples": examples_meta, "window_test": window_test, "bmi_hr": bmi_hr}
with open(os.path.join(HERE, "paper_stats.json"), "w") as f:
    json.dump(S, f, indent=2)
print(json.dumps(examples_meta, indent=1))
