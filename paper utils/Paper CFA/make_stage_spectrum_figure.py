#!/usr/bin/env python3
"""Which cleaned signal keeps the EEG's sleep-stage spectrum? Raw EEG is the reference; HEP, ICA and
patient-specific CFAR (ps-CFAR) are compared with it per expert-scored stage.

Spectra are relative (each divided by its 1-30 Hz power, dB), since the HEP averages away ~30 dB of
non-locked EEG and only the shape can be compared.
  a  mean spectrum, all electrodes, all stages      b  per stage
  c  peak frequency (1/f-corrected) by stage         d  agreement with raw: spectral-shape r, peak error

Inputs: stage_band_cache/<pid>.parquet (stage_band_cleaning.py), stage_band_cache/hep/<pid>.parquet
(stage_band_hep.py). Writes figures/figS_stage_spectrum.png and stage_spectrum_summary.json.
"""
import glob, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from make_cleaning_demo import FIG_DIR, HERE, PALETTE
from stage_band_cleaning import CACHE, FREQS, STAGES

CONDS = [("raw", "Raw EEG (reference)", PALETTE[1], "-"), ("hep", "HEP", PALETTE[0], "-"),
         ("ica", "ICA", PALETTE[5], "--"), ("own", "Patient-specific CFAR", PALETTE[2], "-")]
BAND = (FREQS >= 1.5) & (FREQS <= 30)  # HEP (0.7 s) has nothing real below ~1.4 Hz
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 220})

rows = []
for f in glob.glob(os.path.join(CACHE, "hep", "*.parquet")):
    h = pd.read_parquet(f)
    if h.empty:
        continue
    r = pd.read_parquet(os.path.join(CACHE, os.path.basename(f)))
    rows.append(pd.concat([r[(r.staging == "expert") & r.condition.isin(["raw", "ica", "own"])], h]))
d = pd.concat(rows)
d = d[d.channel == "mean"]
d["rel"] = [10 * np.log10(p[BAND] / p[BAND].sum()) for p in d.psd]
# keep patient-stages that have all four conditions
d = d.groupby(["patient_id", "stage"]).filter(lambda g: g.condition.nunique() == len(CONDS))
n_pat = d.patient_id.nunique()
F = np.where((FREQS[BAND] > 19) & (FREQS[BAND] < 21), np.nan, FREQS[BAND])  # hide 20-Hz device line (features skip it too)

fig = plt.figure(figsize=(11, 7.6))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.05], hspace=0.38)
top = gs[0].subgridspec(1, 6, wspace=0.35, width_ratios=[1.6, 1, 1, 1, 1, 1])
bot = gs[1].subgridspec(1, 3, wspace=0.35, width_ratios=[1.3, 1, 1])


def letter(ax, t):
    ax.text(-0.14, 1.06, t, transform=ax.transAxes, fontsize=10, fontweight="bold")


def spec_mean(sub):
    M = np.vstack(sub.rel.values)
    return M.mean(0), M.std(0) / np.sqrt(len(M))


# a: all stages pooled (patient mean over stages first)
ax = fig.add_subplot(top[0])
for c, lab, col, ls in CONDS:
    pm = d[d.condition == c].groupby("patient_id").rel.apply(lambda s: np.mean(np.vstack(s.values), 0))
    M = np.vstack(pm.values)
    m, se = M.mean(0), M.std(0) / np.sqrt(len(M))
    ax.fill_between(F, m - se, m + se, color=col, alpha=0.2, lw=0)
    ax.plot(F, m, color=col, ls=ls, lw=1.4, label=lab)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Relative power (dB re 1.5-30 Hz total)")
letter(ax, "a")
ax.set_title(f"All electrodes, all stages (n = {n_pat})", fontsize=8)
ax.legend(frameon=False, fontsize=7)

# b: per stage
for i, st in enumerate(STAGES):
    ax = fig.add_subplot(top[i + 1], sharey=fig.axes[0])
    for c, lab, col, ls in CONDS:
        m, se = spec_mean(d[(d.stage == st) & (d.condition == c)])
        ax.fill_between(F, m - se, m + se, color=col, alpha=0.2, lw=0)
        ax.plot(F, m, color=col, ls=ls, lw=1.1)
    ax.set_title(st, fontsize=8)
    if i == 0:
        letter(ax, "b")
    ax.set_xlabel("Hz")
    ax.tick_params(labelleft=False)

# c: peak frequency by stage
ax = fig.add_subplot(bot[0])
x = np.arange(len(STAGES))
S = {"n": int(n_pat)}
for j, (c, lab, col, ls) in enumerate(CONDS):
    g = d[d.condition == c].groupby("stage").peak_hz
    med = g.median().reindex(STAGES)
    ci = g.apply(lambda v: np.percentile([np.median(np.random.default_rng(k).choice(v.values, len(v))) for k in range(500)], [2.5, 97.5])).reindex(STAGES)
    off = (j - 1.5) * 0.08
    ax.errorbar(x + off, med, yerr=np.abs(np.vstack(ci.values).T - med.values), color=col, ls=ls, marker="o", ms=4, capsize=2, lw=1.3, label=lab)
    S[f"peak_{c}"] = med.round(2).to_dict()
ax.set_xticks(x, STAGES)
ax.set_ylabel("Peak frequency (Hz; median, 95% CI)")
letter(ax, "c")
ax.set_title("Dominant (1/f-corrected) peak by stage", fontsize=8)

# d: agreement with raw, per patient-stage
w = d.pivot_table(index=["patient_id", "stage"], columns="condition", values=["rel", "peak_hz"], aggfunc="first")
agree = {}
for c, *_ in CONDS[1:]:
    r = np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(w["rel"]["raw"], w["rel"][c])])
    e = np.abs(w["peak_hz"][c] - w["peak_hz"]["raw"]).values
    agree[c] = (r, e)
    S[f"shape_r_{c}"], S[f"peak_err_{c}"] = float(np.median(r)), float(np.median(e))
    S[f"pct_same_peak_{c}"] = float((e <= 0.5).mean() * 100)
for k, (idx, ylabel, title, fmt) in enumerate([(0, "Spectral-shape r with raw", "Shape agreement", "{:.2f}"),
                                               (1, "|peak − raw peak| (Hz)", "Peak-frequency error", "{:.2f} Hz")]):
    ax = fig.add_subplot(bot[1 + k])
    vals = [agree[c][idx] for c, *_ in CONDS[1:]]
    bp = ax.boxplot(vals, widths=0.6, showfliers=False, patch_artist=True, medianprops={"color": "black"})
    for p, (_, _, col, _) in zip(bp["boxes"], CONDS[1:]):
        p.set(facecolor=col, alpha=0.8, lw=0)
    for i, v in enumerate(vals, 1):
        ax.text(i, np.median(v), " " + fmt.format(np.median(v)), fontsize=7, va="bottom")
    ax.set_xticks([1, 2, 3], ["HEP", "ICA", "Patient-\nspecific CFAR"])
    ax.set_ylabel(ylabel)
    if k == 0:
        letter(ax, "d")
    ax.set_title(title + " (per patient × stage)", fontsize=8)

fig.savefig(os.path.join(FIG_DIR, "figS_stage_spectrum.png"), bbox_inches="tight")
json.dump(S, open(os.path.join(HERE, "stage_spectrum_summary.json"), "w"), indent=1)
print(json.dumps(S, indent=1))
