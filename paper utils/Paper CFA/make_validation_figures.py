#!/usr/bin/env python3
"""Figures for validate_ps_cfar.py (one PNG per check) + summary JSON. Writes figures/validation/*.png
and validate_ps_cfar_summary.json.

Run: venv/bin/python "paper utils/Paper CFA/make_validation_figures.py"
"""
import json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from make_cleaning_demo import PALETTE
from validate_ps_cfar import BANDS, F_GRID

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "validation")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})
d = pd.read_parquet(os.path.join(HERE, "validate_ps_cfar.parquet"))
S = {"n": int(len(d))}
RAW, CLEAN = PALETTE[1], PALETTE[2]


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.png"), bbox_inches="tight")
    plt.close(fig)


def wil(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(stats.wilcoxon(a[m], b[m]).pvalue)


def box(ax, data, labels, colors, log=False):
    data = [np.asarray(x)[np.isfinite(x)] for x in data]
    bp = ax.boxplot(data, widths=0.6, showfliers=False, patch_artist=True, medianprops={"color": "black"})
    for p, c in zip(bp["boxes"], colors):
        p.set(facecolor=c, alpha=0.75, lw=0)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    if log:
        ax.set_yscale("log")
    for i, x in enumerate(data, 1):
        ax.text(i, np.median(x), f" {np.median(x):.2g}", va="bottom", ha="left", fontsize=7.5)


def paired(ax, a, b, label, lim=None):
    m = np.isfinite(a) & np.isfinite(b)
    ax.scatter(a[m], b[m], s=8, alpha=0.5, color=CLEAN, lw=0)
    lo, hi = lim or (min(a[m].min(), b[m].min()), max(a[m].max(), b[m].max()))
    ax.plot([lo, hi], [lo, hi], color="0.5", ls="--", lw=0.8)
    ax.set_xlabel(f"Raw EEG: {label}")
    ax.set_ylabel(f"After ps-CFAR: {label}")


# 1 + 3b: gain on held-out half
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8, 3.4), gridspec_kw={"width_ratios": [2, 1]})
keys = [("in", "ps-CFAR\nsame half"), ("cross", "ps-CFAR\nother half"), ("pop", "CFAR\n(cohort)"), ("fake", "ps-CFAR from\nrandom events")]
box(ax, [d[f"gain_{k}"] for k, _ in keys], [l for _, l in keys], [CLEAN, CLEAN, "0.15", "0.7"], log=True)
ax.axhline(1, color="0.5", ls="--", lw=0.8)
ax.set_ylabel("Neural ÷ cardiac gain vs. raw (log)")
ax.set_title("Scored on held-out half B", fontsize=9)
ax2.hist(d.cos_ab, bins=30, color=CLEAN, alpha=0.8)
ax2.set_xlabel("|cos| pattern half A vs. B")
ax2.set_ylabel("Patients")
ax2.set_title("Pattern stability", fontsize=9)
save(fig, "1_cross_half")
S.update({f"gain_{k}_median": float(d[f"gain_{k}"].median()) for k, _ in keys})
S["cos_ab_median"] = float(d.cos_ab.median())
S["p_cross_vs_pop"] = wil(d.gain_cross.values, d.gain_pop.values)

# 2a: ECG reconstruction
fig, ax = plt.subplots(figsize=(3.6, 3.4))
paired(ax, d.ecg_r2_raw.values, d.ecg_r2_clean.values, "ECG R²", (0, 1))
ax.set_title("ECG rebuilt from EEG (held-out half)", fontsize=9)
save(fig, "2a_ecg_reconstruction")
S.update(ecg_r2_raw_median=float(d.ecg_r2_raw.median()), ecg_r2_clean_median=float(d.ecg_r2_clean.median()),
         p_ecg_r2=wil(d.ecg_r2_raw.values, d.ecg_r2_clean.values))

# 2b: coherence spectrum
fig, ax = plt.subplots(figsize=(6, 3.2))
for k, c, lab in [("raw", RAW, "Raw EEG"), ("clean", CLEAN, "After ps-CFAR")]:
    C = np.vstack(d[f"coh_{k}"].values)
    q = np.percentile(C, [25, 50, 75], axis=0)
    ax.fill_between(F_GRID, q[0], q[2], color=c, alpha=0.2, lw=0)
    ax.plot(F_GRID, q[1], color=c, lw=1.4, label=lab)
hr = d.hr_bpm.median() / 60
for h in range(1, 8):
    ax.axvline(h * hr, color="0.8", lw=0.6, zorder=0)
ax.set_xlabel("Frequency (Hz); grey lines: median heart rate × 1-7")
ax.set_ylabel("EEG-ECG coherence (median, IQR)")
ax.set_xlim(0.5, 40)
ax.legend(frameon=False)
save(fig, "2b_coherence")
cr, cc = np.vstack(d.coh_raw.values), np.vstack(d.coh_clean.values)
S.update(coh_raw_median=float(np.median(cr.mean(1))), coh_clean_median=float(np.median(cc.mean(1))))

# 3a: fake R-peak floor
fig, ax = plt.subplots(figsize=(5, 3.4))
box(ax, [d.hv_real_raw, d.hv_real_clean, d.hv_fake_raw, d.hv_fake_clean],
    ["R-peaks\nraw", "R-peaks\nps-CFAR", "random\nraw", "random\nps-CFAR"], [RAW, CLEAN, "0.75", "0.75"], log=True)
ax.set_ylabel("Event-locked variance outside QRS (µV², log)")
ax.set_title("Is what's left at the noise floor?", fontsize=9)
save(fig, "3a_fake_peaks")
S.update(hv_real_raw=float(d.hv_real_raw.median()), hv_real_clean=float(d.hv_real_clean.median()),
         hv_fake_raw=float(d.hv_fake_raw.median()), hv_fake_clean=float(d.hv_fake_clean.median()),
         pct_at_floor=float((d.hv_real_clean <= 2 * d.hv_fake_clean).mean() * 100))

# 4: injected artifact
low = d.r2_raw < d.r2_raw.quantile(0.25)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))
ax.hist(d.inj_left[low] * 100, bins=30, color=CLEAN, alpha=0.8)
ax.set_xlabel("Injected artifact left (%)")
ax.set_ylabel("Patients (lowest-CFA quartile)")
ax2.hist(d.inj_cos[low], bins=30, color=CLEAN, alpha=0.8)
ax2.set_xlabel("|cos| learned vs. injected pattern")
fig.suptitle("Known artifact injected into low-CFA patients", fontsize=9)
save(fig, "4_injection")
S.update(inj_n=int(low.sum()), inj_left_median=float(d.inj_left[low].median()),
         inj_cos_median=float(d.inj_cos[low].median()), inj_brain_kept_median=float(d.inj_brain_kept[low].median()))

# 5: brain features
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2), gridspec_kw={"width_ratios": [1.6, 1]})
box(ax, [d[f"bp_{b}_db"] for b in BANDS], list(BANDS), [CLEAN] * len(BANDS))
ax.axhline(0, color="0.5", ls="--", lw=0.8)
ax.set_ylabel("Power change after ps-CFAR (dB)")
if "spindles_raw" in d:
    dur = d.spindles_raw.notna()
    paired(ax2, d.spindles_raw[dur].values.astype(float), d.spindles_clean[dur].values.astype(float), "spindles")
    S.update(spindles_raw_median=float(d.spindles_raw.median()), spindles_clean_median=float(d.spindles_clean.median()),
             p_spindles=wil(d.spindles_raw.values.astype(float), d.spindles_clean.values.astype(float)))
save(fig, "5_brain_features")
S.update({f"bp_{b}_db_median": float(d[f"bp_{b}_db"].median()) for b in BANDS})

# 6a + 6b
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7, 3.3))
box(ax, [d.beat_r_raw, d.beat_r_clean], ["Raw", "ps-CFAR"], [RAW, CLEAN])
ax.axhline(0, color="0.5", ls="--", lw=0.8)
ax.set_ylabel("r(per-beat CFA, QRS amplitude)")
ax.set_title("Beat-to-beat tracking", fontsize=9)
pm = d.prem_left.notna() if "prem_left" in d else pd.Series(False, index=d.index)
if pm.any():
    paired(ax2, d.norm_left[pm].values * 100, d.prem_left[pm].values * 100, "")
    ax2.set_xlabel("Cardiac power left, normal beats (%)")
    ax2.set_ylabel("…premature beats (%)")
    ax2.set_title(f"Premature beats (n = {pm.sum()} patients)", fontsize=9)
    S.update(prem_n=int(pm.sum()), prem_left_median=float(d.prem_left.median()), norm_left_median=float(d.norm_left[pm].median()),
             p_prem=wil(d.prem_left[pm].values, d.norm_left[pm].values))
save(fig, "6_physiology")
S.update(beat_r_raw_median=float(d.beat_r_raw.median()), beat_r_clean_median=float(d.beat_r_clean.median()),
         p_beat_r=wil(d.beat_r_raw.values, d.beat_r_clean.values))

json.dump(S, open(os.path.join(HERE, "validate_ps_cfar_summary.json"), "w"), indent=2)
print(json.dumps(S, indent=1))
