#!/usr/bin/env python3
"""HEP-cleaning figure from hep_cleaning_compare.parquet: raw vs. ICA vs. cohort CFAR vs. ps-CFAR.

  a  grand-average frontal HEP (F3/F4) with the ECG average
  b  ECG-explained HEP power over time (per channel: HEP regressed on the ECG average outside QRS)
  c  ECG-explained power in the 200-600 ms HEP window, % of raw
  d  ECG-likeness: R² between frontal HEP and ECG average within 200-600 ms, per patient
  e  synthetic neural HEP recovered when each method is re-fitted on the injected data (amplitude kept, waveform r)
Writes figures/figS_hep_cleaning.png and hep_cleaning_summary.json.
"""
import json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from make_cleaning_demo import FIG_DIR, HERE, PALETTE, QRS_EXCLUDE_SEC, SITES

M = [("raw", "Raw EEG", PALETTE[1]), ("ica", "ICA", PALETTE[5]), ("cfar", "CFAR", "0.15"), ("own", "Patient-\nspecific CFAR", PALETTE[2])]
FRONT = [SITES.index("F3"), SITES.index("F4")]
WIN = (0.2, 0.6)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 220})
rng = np.random.default_rng(0)

d = pd.read_parquet(os.path.join(HERE, "hep_cleaning_compare.parquet"))
d = d[d.sfreq == d.sfreq.mode()[0]].reset_index(drop=True)  # one sampling rate, so waveforms share a time axis
tt = np.array(d.tt[0])
T = len(tt)
H = {m: np.stack([np.asarray(v).reshape(6, T) for v in d[f"hep_{m}"]]) for m, *_ in M}  # (pat, ch, t)
E = np.stack(d.ecg.values)
keep = np.abs(tt) > QRS_EXCLUDE_SEC
win = (tt >= WIN[0]) & (tt < WIN[1])

# demographics
d["bdsp_patient_id"] = pd.to_numeric(d.patient_id.str.extract(r"I\d{4}(\d{9})", expand=False))
demo = pd.read_parquet(os.path.join(HERE, "demographics_combined.parquet"))[["patient_id", "sex"]]
bmi = pd.read_parquet(os.path.join(HERE, "bmi_combined.parquet")).drop_duplicates("bdsp_patient_id")
d = d.merge(demo.drop_duplicates("patient_id"), on="patient_id", how="left").merge(bmi, on="bdsp_patient_id", how="left")


def fitted(h, e):
    """Per channel LS fit of the HEP on the ECG average outside QRS -> ECG-explained part (ch, t)."""
    e0 = e - e[keep].mean()
    h0 = h - h[:, keep].mean(1, keepdims=True)
    beta = h0[:, keep] @ e0[keep] / (e0[keep] @ e0[keep])
    return np.outer(beta, e0)


FIT = {m: np.stack([fitted(H[m][i], E[i]) for i in range(len(d))]) for m, *_ in M}
S = {"n": int(len(d)), "sfreq": float(d.sfreq[0])}

fig = plt.figure(figsize=(11.5, 7.2))
gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.38)
bot = gs[1, :].subgridspec(1, 2, wspace=0.3)


def letter(ax, t, title):
    ax.text(-0.16, 1.06, t, transform=ax.transAxes, fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=8.5)


def boot_ci(f, *arrs, n=1000):
    idx = [rng.integers(0, len(arrs[0]), len(arrs[0])) for _ in range(n)]
    return np.percentile([f(*(a[i] for a in arrs)) for i in idx], [2.5, 97.5])


# a
ax = fig.add_subplot(gs[0, 0])
ax.axvspan(-QRS_EXCLUDE_SEC * 1e3, QRS_EXCLUDE_SEC * 1e3, color="0.92", lw=0)
ax.axvspan(WIN[0] * 1e3, WIN[1] * 1e3, color="0.96", lw=0)
for m, lab, col in M:
    x = H[m][:, FRONT].mean(1)
    mu, se = x.mean(0), x.std(0) / np.sqrt(len(x))
    ax.plot(tt * 1e3, mu, color=col, lw=1.3, label=lab.replace("\n", " "))
    ax.fill_between(tt * 1e3, mu - se, mu + se, color=col, alpha=0.2, lw=0)
ax2 = ax.twinx()
ax2.plot(tt * 1e3, E.mean(0), color="0.6", lw=0.8, ls=":")
ax2.set_yticks([])
ax2.spines["right"].set_visible(False)
ax.set_xlabel("Time from R-peak (ms)")
ax.set_ylabel("Frontal HEP (µV; mean ± SEM)")
ax.legend(frameon=False, fontsize=6.5, loc="upper right")
letter(ax, "a", "Grand-average HEP (dotted: ECG)")

# b
ax = fig.add_subplot(gs[0, 1])
for m, lab, col in M:
    p = (FIT[m] ** 2).mean(1)
    ax.plot(tt * 1e3, np.median(p, 0), color=col, lw=1.3)
ax.axvspan(-QRS_EXCLUDE_SEC * 1e3, QRS_EXCLUDE_SEC * 1e3, color="0.92", lw=0)
ax.axvspan(WIN[0] * 1e3, WIN[1] * 1e3, color="0.96", lw=0, zorder=0)
ax.set_yscale("log")
ax.set_xlabel("Time from R-peak (ms)")
ax.set_ylabel("ECG-explained HEP power (µV², median)")
letter(ax, "b", "Where cardiac artifact remains")

# c
ax = fig.add_subplot(gs[0, 2])
pw = {m: (FIT[m][:, :, win] ** 2).mean((1, 2)) for m, *_ in M}
vals = [pw[m] / pw["raw"] * 100 for m, *_ in M[1:]]
for i, (v, (m, lab, col)) in enumerate(zip(vals, M[1:])):
    med = np.median(v)
    ci = boot_ci(np.median, v)
    ax.bar(i, med, color=col, alpha=0.85, yerr=np.abs(ci - med)[:, None], capsize=2)
    ax.text(i, ci[1], f"{med:.0f}%", ha="center", va="bottom", fontsize=7.5)
    S[f"winpow_pct_{m}"] = float(med)
ax.set_xticks(range(3), [l for _, l, _ in M[1:]])
ax.set_ylabel("ECG-explained power left, 200-600 ms\n(% of raw; median, 95% CI)")
letter(ax, "c", "Artifact in the HEP window")

# d: how ECG-like the HEP is inside the HEP window (per patient, frontal HEP vs ECG average)
amp = None
ax = fig.add_subplot(bot[0])
Ew = E[:, win]
shape = {m: np.array([np.corrcoef(H[m][i, FRONT][:, win].mean(0), Ew[i])[0, 1] ** 2 for i in range(len(d))]) for m, *_ in M}
bp = ax.boxplot([shape[m] for m, *_ in M], widths=0.6, showfliers=False, patch_artist=True, medianprops={"color": "black"})
for p_, (_, _, col) in zip(bp["boxes"], M):
    p_.set(facecolor=col, alpha=0.8, lw=0)
for j, (m, *_) in enumerate(M, 1):
    ax.text(j, np.median(shape[m]), f" {np.median(shape[m]):.2f}", fontsize=7, va="bottom")
    S[f"shape_r2_{m}"] = float(np.median(shape[m]))
S["p_shape_own_vs_ica"] = float(stats.wilcoxon(shape["own"], shape["ica"]).pvalue)
ax.set_xticks(range(1, 5), [l for _, l, _ in M])
ax.set_ylabel("R², frontal HEP vs. ECG average\n(200-600 ms, per patient)")
letter(ax, "d", "How much the HEP looks like the ECG")

# e: synthetic HEP recovery
ax = fig.add_subplot(bot[1])
for j, (m, lab, col) in enumerate(M[1:]):
    for k, (key, mk) in enumerate([("inj_kept", "o"), ("inj_r", "s")]):
        v = d[f"{key}_{m}"].values
        med, ci = np.median(v), boot_ci(np.median, v)
        ax.errorbar(j + (k - 0.5) * 0.25, med, yerr=np.abs(ci - med)[:, None], color=col, marker=mk, ms=5, capsize=2, lw=1.2)
        S[f"{key}_{m}"] = float(med)
ax.plot([], [], "ko", label="amplitude kept")
ax.plot([], [], "ks", label="waveform r")
ax.set_xticks(range(3), [l for _, l, _ in M[1:]])
ax.set_ylabel("Synthetic neural HEP recovered (median)")
ax.legend(frameon=False, fontsize=6.5)
letter(ax, "e", "Known brain response recovered")

fig.savefig(os.path.join(FIG_DIR, "figS_hep_cleaning.png"), bbox_inches="tight")
json.dump(S, open(os.path.join(HERE, "hep_cleaning_summary.json"), "w"), indent=1)
print(json.dumps(S, indent=1))
