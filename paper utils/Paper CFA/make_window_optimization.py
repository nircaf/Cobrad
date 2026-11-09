#!/usr/bin/env python3
"""Fig S8: drill-down of Fig 3d -- which segment duration should we test?

Uses every full-cohort length available (10 min = cfa_combined.parquet, others
= cfa_variance_explained_{N}min.parquet, incl. 75/90 min once those runs land).
Restricted to (patient, channel) pairs present at every length, so channel-set
differences between runs don't masquerade as a duration effect. Note each length
draws its own random QC-passing window, so length and location are confounded.

  a  per-patient R² distribution per duration
  b  rank agreement (Spearman) with the longest duration and with the next one
  c  per-patient |ΔR²| between consecutive durations (convergence)
  d  patient retention: fraction of the 7,230-patient list with a valid window
  e  example patients whose R² moves differently across durations
  f  trade-off: agreement with longest x retention -> recommended duration

Run: venv/bin/python "paper utils/Paper CFA/make_window_optimization.py"
"""
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00", "#000000"]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300, "savefig.dpi": 300})
N_LIST = sum(1 for line in open(os.path.join(HERE, "figure5_7230_patients.txt")) if line.strip())

files = {10: os.path.join(HERE, "cfa_combined_10min.parquet")}
for p in glob.glob(os.path.join(HERE, "cfa_variance_explained_*min.parquet")):
    files[int(re.search(r"_(\d+)min\.parquet$", p).group(1))] = p
L = sorted(files)
cols = ["patient_id", "eeg_channel", "cfa_r2_excl_qrs"]
long = pd.concat([pd.read_parquet(files[m], columns=cols).assign(minutes=m) for m in L])
long = long.groupby(["patient_id", "eeg_channel", "minutes"], as_index=False).cfa_r2_excl_qrs.mean()
listed = set(open(os.path.join(HERE, "figure5_7230_patients.txt")).read().split())
retention = {m: len(set(long.patient_id[long.minutes == m]) & listed) / N_LIST for m in L}

# keep (patient, channel) pairs observed at every length, then average channels
complete = long.groupby(["patient_id", "eeg_channel"]).minutes.transform("nunique") == len(L)
wide = long[complete].groupby(["patient_id", "minutes"]).cfa_r2_excl_qrs.mean().unstack()
print(f"Lengths: {L}; patients complete at every length: {len(wide):,}")

ref = L[-1]
rho_ref = {m: stats.spearmanr(wide[m], wide[ref])[0] for m in L[:-1]}
rho_next = {a: stats.spearmanr(wide[a], wide[b])[0] for a, b in zip(L, L[1:])}
dabs = {a: (wide[b] - wide[a]).abs() for a, b in zip(L, L[1:])}
score = {m: rho_ref[m] * retention[m] for m in L[:-1]}
best = max(score, key=score.get)
table = pd.DataFrame({"retention": retention, "rho_vs_longest": rho_ref, "rho_vs_next": rho_next,
                      "median_abs_delta_to_next": {k: v.median() for k, v in dabs.items()},
                      "mean_r2": wide.mean(), "score": score})
print(table.round(3).to_string())
print(f"Recommended duration (max rho_vs_{ref}min x retention): {best} min")

fig, axes = plt.subplots(2, 3, figsize=(11, 7.4))
for ax, letter in zip(axes.flat, "abcdef"):
    ax.set_title(letter, loc="left", fontsize=10, fontweight="bold")

ax = axes[0, 0]
ax.boxplot([wide[m].values for m in L], positions=L, widths=[m * 0.18 + 1 for m in L], showfliers=False,
           patch_artist=True, boxprops={"facecolor": PALETTE[5], "alpha": 0.5}, medianprops={"color": "black"})
ax.plot(L, wide.mean(), "o-", color=PALETTE[0], ms=4, lw=1.6, label="Mean")
ax.set_xticks(L)
ax.set_ylabel("Patient-mean CFA R²")
ax.legend(frameon=False, fontsize=7.5)
ax.set_title(f"R² by duration (n = {len(wide):,})", fontsize=9)

ax = axes[0, 1]
ax.plot(L[:-1], list(rho_ref.values()), "o-", color=PALETTE[0], ms=4, lw=1.6, label=f"vs. {ref} min")
ax.plot(L[:-1], list(rho_next.values()), "s--", color=PALETTE[1], ms=4, lw=1.4, label="vs. next-longer duration")
ax.set_xticks(L[:-1])
ax.set_ylabel("Spearman ρ (patient ranks)")
ax.legend(frameon=False, fontsize=7.5)
ax.set_title("Rank agreement between durations", fontsize=9)

ax = axes[0, 2]
q = pd.DataFrame({m: d.quantile([0.25, 0.5, 0.75]) for m, d in dabs.items()}).T
ax.fill_between(q.index, q[0.25], q[0.75], color=PALETTE[2], alpha=0.25, lw=0)
ax.plot(q.index, q[0.5], "o-", color=PALETTE[2], ms=4, lw=1.6)
ax.set_xticks(L[:-1])
ax.set_xticklabels([f"{a}→{b}" for a, b in zip(L, L[1:])], fontsize=7, rotation=30)
ax.set_ylabel("|ΔR²| to next duration (median, IQR)")
ax.set_title("Within-patient convergence", fontsize=9)

ax = axes[1, 0]
ax.bar(L, [100 * retention[m] for m in L], width=[m * 0.18 + 1 for m in L], color=PALETTE[6], alpha=0.8)
ax.set_xticks(L)
ax.set_ylim(0, 105)
ax.set_ylabel(f"Patients with a valid window (% of {N_LIST:,})")
ax.set_title("Retention", fontsize=9)

# examples: steepest rise, steepest fall, most stable, interior peak, interior trough, most variable
ax = axes[1, 1]
x = np.log(np.array(L, float))
Y = wide[L].values
slope = np.polyfit(x, Y.T, 1)[0]
spread = Y.max(1) - Y.min(1)
inner = Y[:, 1:-1].max(1) - np.maximum(Y[:, 0], Y[:, -1])
dip = np.minimum(Y[:, 0], Y[:, -1]) - Y[:, 1:-1].min(1)
crit = {"Rising": slope, "Falling": -slope, "Stable": -(spread + (Y.mean(1) < np.median(Y)) * 9),
        "Mid peak": inner, "Mid trough": dip, "Erratic": np.abs(np.diff(Y, axis=1)).sum(1)}
picks = {}
for lab, v in crit.items():  # highest-scoring patient not already shown
    picks[lab] = next(i for i in np.argsort(-v) if i not in picks.values())
for (lab, i), c in zip(picks.items(), PALETTE[:3] + PALETTE[3:4] + PALETTE[5:7]):
    ax.plot(L, Y[i], "o-", ms=3.5, lw=1.4, color=c, label=f"{lab} ({wide.index[i]})")
ax.set_xticks(L)
ax.set_ylabel("Patient-mean CFA R²")
ax.legend(frameon=False, fontsize=6.5, loc="best")
ax.set_title("Example patients", fontsize=9)

ax = axes[1, 2]
ax.scatter([100 * retention[m] for m in L[:-1]], list(rho_ref.values()), s=30, color=PALETTE[0], zorder=3)
for m in L[:-1]:
    ax.annotate(f"{m} min", (100 * retention[m], rho_ref[m]), xytext=(4, 3), textcoords="offset points", fontsize=7.5,
                fontweight="bold" if m == best else "normal", color=PALETTE[1] if m == best else "black")
ax.set_xlabel("Retention (%)")
ax.set_ylabel(f"Spearman ρ vs. {ref} min")
ax.set_title(f"Trade-off: best ρ × retention = {best} min", fontsize=9)

for ax in [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]:
    ax.set_xlabel("Segment duration (min)")
fig.tight_layout(h_pad=2.2, w_pad=2.0)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, "figures", f"figS8_window_optimization.{ext}"), bbox_inches="tight")
with open(os.path.join(HERE, "paper_stats.json")) as f:
    S = json.load(f)
S["window_optimization"] = {"lengths": L, "n_complete_patients": len(wide), "recommended_minutes": int(best),
                            "examples": {k: str(wide.index[i]) for k, i in picks.items()},
                            "table": table.reset_index(names="minutes").to_dict(orient="records")}
with open(os.path.join(HERE, "paper_stats.json"), "w") as f:
    json.dump(S, f, indent=2, default=float)
print("Figure written to figures/figS8_window_optimization.png")
