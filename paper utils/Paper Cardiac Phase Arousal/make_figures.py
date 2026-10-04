#!/usr/bin/env python3
"""Figures for the cardiac-phase-of-arousal paper.

  source venv/bin/activate && python3 "paper utils/Paper Cardiac Phase Arousal/make_figures.py"
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

S = json.loads((HERE / "paper_stats.json").read_text())
RES = pd.read_parquet(HERE / "cardiac_phase_arousal.parquet")
EV = pd.read_parquet(HERE / "events.parquet")
N_BINS = S["cohort"]["n_bins"]
EDGES = np.linspace(0, 2 * np.pi, N_BINS + 1)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
WIDTH = 2 * np.pi / N_BINS


def rose(ax, phi, title, color="#3b6ea5"):
    counts, _ = np.histogram(phi % (2 * np.pi), bins=EDGES)
    rel = counts / counts.mean() if counts.mean() else counts
    ax.bar(CENTERS, rel, width=WIDTH * 0.95, bottom=0.0, color=color,
           edgecolor="white", linewidth=0.6, alpha=0.9)
    ax.plot(np.linspace(0, 2 * np.pi, 200), np.ones(200), color="#c0392b", lw=1.0, ls="--")
    C, Sx = np.cos(phi).mean(), np.sin(phi).mean()
    mu = np.arctan2(Sx, C) % (2 * np.pi)
    R = float(np.hypot(C, Sx))
    lo, hi = rel.min(), rel.max()
    ax.annotate("", xy=(mu, hi), xytext=(mu, 0),
                arrowprops=dict(color="#111111", width=1.2, headwidth=6, headlength=6))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, max(hi * 1.12, 1.05))
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.set_xticklabels(["R-peak\n0", "45", "90", "135", "180", "225", "270", "315"], fontsize=6.5)
    ax.set_title(f"{title}\nn={len(phi):,}  R={R:.4f}", fontsize=8.5, pad=9)
    return lo, hi


# ---------------------------------------------------------------- Figure 1
fig = plt.figure(figsize=(11, 4.2))
ax = fig.add_subplot(1, 3, 1, projection="polar")
rose(ax, EV["phi"].to_numpy(), "All arousal onsets (pooled)")

ax2 = fig.add_subplot(1, 3, 2)
counts = np.asarray(S["pooled_bin_counts"], dtype=float)
exp = counts.mean()
ax2.bar(np.degrees(CENTERS), counts, width=np.degrees(WIDTH) * 0.9,
        color="#3b6ea5", edgecolor="white")
ax2.axhline(exp, color="#c0392b", ls="--", lw=1.2,
            label=f"uniform expectation ({exp:,.0f})")
ax2.set_xlabel("Cardiac phase of arousal onset (deg; 0 = R-peak)", fontsize=9)
ax2.set_ylabel("Arousal onsets", fontsize=9)
ax2.set_ylim(0, counts.max() * 1.12)
ax2.legend(fontsize=7.5, frameon=False)
ax2.set_title(f"Exposure-normalised counts\n$\\chi^2$={S['pooled']['chi2'] if 'chi2' in S['pooled'] else RES.iloc[0]['chi2']:.1f}, "
              f"p={RES.iloc[0]['p_chi2']:.3g}", fontsize=8.5)
ax2.spines[["top", "right"]].set_visible(False)

ax3 = fig.add_subplot(1, 3, 3)
sub = RES[RES["stratum"] == "stage"]
ax3.bar(sub["group"], sub["R"], color="#6b9ac4", edgecolor="#33556f")
ax3.axhline(RES.iloc[0]["R"], color="#c0392b", ls="--", lw=1.2, label="pooled R")
for x, (_, r) in enumerate(sub.iterrows()):
    ax3.text(x, r["R"], f"p={r['p_rayleigh']:.2g}", ha="center", va="bottom", fontsize=6.5)
ax3.set_ylabel("Mean resultant length R", fontsize=9)
ax3.set_xlabel("Sleep stage", fontsize=9)
ax3.set_title("Effect size by sleep stage", fontsize=8.5)
ax3.legend(fontsize=7.5, frameon=False)
ax3.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_pooled_phase.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
stages = list(RES[RES["stratum"] == "stage"]["group"])
fig, axes = plt.subplots(1, max(len(stages), 1), figsize=(2.5 * len(stages), 3.2),
                         subplot_kw=dict(projection="polar"))
axes = np.atleast_1d(axes)
for ax, st in zip(axes, stages):
    rose(ax, EV[EV["stage"] == st]["phi"].to_numpy(), st, color="#7a9e7e")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_by_stage.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Figure 3
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, strat, title in [(axes[0], "sex", "Sex"), (axes[1], "age", "Age (median split)")]:
    sub = RES[RES["stratum"] == strat]
    ax.bar(sub["group"], sub["Z"], color="#b08968", edgecolor="#6b4f3a")
    for x, (_, r) in enumerate(sub.iterrows()):
        ax.text(x, r["Z"], f"n={r['n_events']:,}\np={r['p_rayleigh']:.2g}",
                ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Rayleigh Z", fontsize=9)
    ax.set_title(title, fontsize=9.5)
    ax.set_ylim(0, max(sub["Z"].max() * 1.35, 4))
    ax.axhline(-np.log(0.05), color="#c0392b", ls="--", lw=1.0,
               label="Z at p = 0.05")
    ax.legend(fontsize=7.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_strata.png", dpi=200)
plt.close(fig)

print("wrote", *[p.name for p in sorted(FIG_DIR.glob("*.png"))])
