#!/usr/bin/env python3
"""Figure 3: CFA vs. age, sex, BMI (a-c) and segment duration (d-f), one figure.

Panel data: fig3_strat.parquet / fig3_bmi.parquet (written by make_figures.py)
and paper_stats.json["dose_response"] (window_stage_sensitivity_report.py).

Run: venv/bin/python "paper utils/Paper CFA/make_fig3.py"
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00", "#000000"]
SEX_COLORS = {"Female": PALETTE[3], "Male": PALETTE[0]}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300, "savefig.dpi": 300})

strat = pd.read_parquet(os.path.join(HERE, "fig3_strat.parquet"))
bmi = pd.read_parquet(os.path.join(HERE, "fig3_bmi.parquet"))
with open(os.path.join(HERE, "paper_stats.json")) as f:
    S = json.load(f)
dose = pd.DataFrame(S["dose_response"]["lengths"])
dstrat = pd.DataFrame(S["dose_response"]["stratified_by_length"])
TICKS = [5, 10, 20, 30, 45, 60]
YLAB = "Patient-mean CFA R² (outside QRS)"

fig, axes = plt.subplots(2, 3, figsize=(11, 7.4))
for ax, letter in zip(axes.flat, "abcdef"):
    ax.set_title(letter, loc="left", fontsize=10, fontweight="bold")

ax = axes[0, 0]
x, y = strat["age"].values, strat["cfa_r2_excl_qrs"].values
ax.scatter(x, y, s=5, alpha=0.12, color=PALETTE[0], edgecolor="none", rasterized=True)
k, b = np.polyfit(x, y, 1)
xs = np.linspace(x.min(), x.max(), 100)
ax.plot(xs, k * xs + b, color=PALETTE[1], lw=1.8)
ax.set_xlabel("Age (years)")
ax.set_ylabel(YLAB)

ax = axes[0, 1]
groups = [strat.loc[strat.sex == s, "cfa_r2_excl_qrs"].dropna().values for s in SEX_COLORS]
bp = ax.boxplot(groups, tick_labels=list(SEX_COLORS), patch_artist=True, widths=0.5, showfliers=False,
                medianprops={"color": "black", "lw": 1.4})
for patch, c in zip(bp["boxes"], SEX_COLORS.values()):
    patch.set_facecolor(c)
    patch.set_alpha(0.65)
ax.set_ylabel(YLAB)

ax = axes[0, 2]
for sex, c in SEX_COLORS.items():
    g = bmi[bmi.sex == sex]
    ax.scatter(g.bmi, g.cfa_r2_excl_qrs, s=5, alpha=0.12, color=c, edgecolor="none", rasterized=True)
    k, b = np.polyfit(g.bmi, g.cfa_r2_excl_qrs, 1)
    xs = np.linspace(g.bmi.min(), g.bmi.max(), 100)
    ax.plot(xs, k * xs + b, color=c, lw=1.8, label=sex)
ax.set_xlabel("BMI (kg/m²)")
ax.set_ylabel(YLAB)
ax.legend(frameon=False, fontsize=7.5)

ax = axes[1, 0]
ax.errorbar(dose.window_minutes, dose["mean"], yerr=1.96 * dose["sem"], marker="o", ms=4, capsize=3,
            color=PALETTE[0], lw=1.6)
ax.set_ylabel("Mean CFA R² (± 95% CI)")

ax = axes[1, 1]
for col, sex in [("male_mean", "Male"), ("female_mean", "Female")]:
    ax.plot(dstrat.window_minutes, dstrat[col], marker="o", ms=4, color=SEX_COLORS[sex], lw=1.6, label=sex)
ax.set_ylabel("Mean CFA R²")
ax.legend(frameon=False, fontsize=7.5)

ax = axes[1, 2]
ax.plot(dstrat.window_minutes, dstrat.any_dx_mean, marker="s", ms=4, color=PALETTE[1], lw=1.6, label="≥1 diagnosis")
ax.plot(dstrat.window_minutes, dstrat.no_dx_mean, marker="s", ms=4, color=PALETTE[2], lw=1.6, label="No diagnosis")
ax.set_ylabel("Mean CFA R²")
ax.legend(frameon=False, fontsize=7.5)

for ax in axes[1]:
    ax.set_xlabel("Segment duration (min)")
    ax.set_xticks(TICKS)
    ax.set_xlim(2, 63)

fig.tight_layout(h_pad=2.2, w_pad=2.0)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, "figures", f"fig3_bmi_sex_duration.{ext}"), bbox_inches="tight")
print("Figure written to figures/fig3_bmi_sex_duration.png")
