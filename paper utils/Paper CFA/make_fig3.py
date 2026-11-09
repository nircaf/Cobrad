#!/usr/bin/env python3
"""Figure 3: CFA vs. age, sex, BMI (a-c) and segment duration (d-f), one figure.

Panel data: fig3_strat.parquet / fig3_bmi.parquet (written by make_figures.py)
and paper_stats.json["dose_response"] (window_stage_sensitivity_report.py).

Run: venv/bin/python "paper utils/Paper CFA/make_fig3.py"
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

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

def dx_by_length():
    """Mean CFA R2 of patients with exactly 0, exactly 1, and >=2 diagnoses at each length (common cohort)."""
    import glob, re
    files = {10: os.path.join(HERE, "cfa_combined_10min.parquet")}
    for f in glob.glob(os.path.join(HERE, "cfa_variance_explained_*min.parquet")):
        files[int(re.search(r"_(\d+)min\.parquet$", f).group(1))] = f
    demo = pd.read_parquet(os.path.join(HERE, "demographics_combined.parquet"))[["patient_id", "age", "n_diagnoses"]]
    dfs = {m: pd.read_parquet(f, columns=["patient_id", "cfa_r2_excl_qrs"]) for m, f in files.items()}
    common = set.intersection(*[set(d.patient_id) for d in dfs.values()])
    out = {"zero": ([], 0), "one": ([], 0), "two": ([], 0)}
    for m in sorted(dfs):
        pt = dfs[m][dfs[m].patient_id.isin(common)].groupby("patient_id", as_index=False).mean().merge(demo, on="patient_id").dropna(subset=["age"])
        nd = pt.n_diagnoses.fillna(0)
        for key, mask in [("zero", nd == 0), ("one", nd == 1), ("two", nd >= 2)]:
            g = pt.loc[mask, "cfa_r2_excl_qrs"]
            out[key][0].append(g.mean()); out[key] = (out[key][0], len(g))
    return out


DX = dx_by_length()
TICKS = [5, 10, 20, 30, 45, 60, 75, 90]
YLAB = "Patient-mean CFA R² (outside QRS)"


def title(ax, text):
    ax.set_title(text, fontsize=9)

fig, axes = plt.subplots(2, 3, figsize=(11, 7.4))
for ax, letter in zip(axes.flat, "abcdef"):
    ax.set_title(letter, loc="left", fontsize=10, fontweight="bold")

ax = axes[0, 0]
x, y = strat["age"].values, strat["cfa_r2_excl_qrs"].values
title(ax, f"CFA vs. age (n = {len(x):,}; r = {np.corrcoef(x, y)[0, 1]:.2f})")
ax.scatter(x, y, s=5, alpha=0.12, color=PALETTE[0], edgecolor="none", rasterized=True)
k, b = np.polyfit(x, y, 1)
xs = np.linspace(x.min(), x.max(), 100)
ax.plot(xs, k * xs + b, color=PALETTE[1], lw=1.8)
ax.set_xlabel("Age (years)")
ax.set_ylabel(YLAB)

ax = axes[0, 1]
groups = [strat.loc[strat.sex == s, "cfa_r2_excl_qrs"].dropna().values for s in SEX_COLORS]
bp = ax.boxplot(groups, tick_labels=[f"{s}\n(n = {len(g):,})" for s, g in zip(SEX_COLORS, groups)], patch_artist=True, widths=0.5, showfliers=False,
                medianprops={"color": "black", "lw": 1.4})
for patch, c in zip(bp["boxes"], SEX_COLORS.values()):
    patch.set_facecolor(c)
    patch.set_alpha(0.65)
ax.set_ylabel(YLAB)
title(ax, f"CFA by sex (p = {S['stratified']['p_sex_mannwhitney']:.1e})")

ax = axes[0, 2]
rs = {}
for sex, c in SEX_COLORS.items():
    g = bmi[bmi.sex == sex]
    rs[sex] = (np.corrcoef(g.bmi, g.cfa_r2_excl_qrs)[0, 1], len(g))
    ax.scatter(g.bmi, g.cfa_r2_excl_qrs, s=5, alpha=0.12, color=c, edgecolor="none", rasterized=True)
    k, b = np.polyfit(g.bmi, g.cfa_r2_excl_qrs, 1)
    xs = np.linspace(g.bmi.min(), g.bmi.max(), 100)
    ax.plot(xs, k * xs + b, color=c, lw=1.8, label=f"{sex} (n = {len(g):,}; r = {rs[sex][0]:.2f})")
ax.set_xlabel("BMI (kg/m²)")
ax.set_ylabel(YLAB)
ax.legend(frameon=False, fontsize=7.5)
# Fisher r-to-z test for difference between two independent correlations
(r_f, n_f), (r_m, n_m) = rs["Female"], rs["Male"]
z = (np.arctanh(r_f) - np.arctanh(r_m)) / math.sqrt(1 / (n_f - 3) + 1 / (n_m - 3))
p_diff = math.erfc(abs(z) / math.sqrt(2))
print(f"Fig 3c: r_female = {r_f:.3f} (n={n_f}), r_male = {r_m:.3f} (n={n_m}), Fisher z = {z:.2f}, p = {p_diff:.2e}")
title(ax, f"CFA vs. BMI by sex (Δr p = {p_diff:.1e})")

ax = axes[1, 0]
ax.errorbar(dose.window_minutes, dose["mean"], yerr=1.96 * dose["sem"], marker="o", ms=4, capsize=3,
            color=PALETTE[0], lw=1.6)


def sat(t, A, k, c):
    """Saturating hyperbola: c at t = 0, half-way to the asymptote A at t = k."""
    return c + (A - c) * t / (t + k)


p, C = curve_fit(sat, dose.window_minutes, dose["mean"], p0=(0.45, 10, 0.2), sigma=dose["sem"], absolute_sigma=True)
A, k, c = p
A_ci = 1.96 * math.sqrt(C[0, 0])
chi2 = float((((dose["mean"] - sat(dose.window_minutes, *p)) / dose["sem"]) ** 2).sum())
print(f"Fig 3d fit: R2_inf = {A:.4f} ± {A_ci:.4f}, k = {k:.2f} min, c = {c:.4f}, chi2 = {chi2:.1f} (df {len(dose) - 3})")
S["dose_fit"] = {"asymptote": A, "asymptote_ci_lo": A - A_ci, "asymptote_ci_hi": A + A_ci, "k_min": k, "c": c,
                 "chi2": chi2, "df": len(dose) - 3, "pct_of_asymptote_at_10": 100 * sat(10, *p) / A,
                 "pct_of_asymptote_at_max": 100 * sat(dose.window_minutes.max(), *p) / A}
with open(os.path.join(HERE, "paper_stats.json"), "w") as f:
    json.dump(S, f, indent=2)
title(ax, f"CFA vs. segment duration (n = {S['dose_response']['n_common_patients']:,})")
# inset: same mean CFA, x = compute = minutes x mean channels per patient (channel-minutes)
cm = dose.window_minutes * dose.n_rows / dose.n_patients
ins = ax.inset_axes([0.5, 0.1, 0.46, 0.38])
ins.errorbar(cm, dose["mean"], yerr=1.96 * dose["sem"], marker="o", ms=3, capsize=2, color=PALETTE[0], lw=1.2)
ins.set_xscale("log")
ins.set_xlabel("Compute (channel-min)", fontsize=6.5, labelpad=1)
ins.set_ylabel("Mean CFA R²", fontsize=6.5, labelpad=1)
ins.tick_params(labelsize=6, length=2)
ax.set_ylabel("Mean CFA R² (± 95% CI)")

ax = axes[1, 1]
for col, sex in [("male_mean", "Male"), ("female_mean", "Female")]:
    ax.plot(dstrat.window_minutes, dstrat[col], marker="o", ms=4, color=SEX_COLORS[sex], lw=1.6,
            label=f"{sex} (n = {int(dstrat[f'n_{sex.lower()}'].iloc[0]):,})")
ax.set_ylabel("Mean CFA R²")
ax.legend(frameon=False, fontsize=7.5)
title(ax, "Duration effect by sex")

ax = axes[1, 2]
for key, lab, c in [("zero", "Undiagnosed", PALETTE[2]), ("one", "1 diagnosis", PALETTE[1]), ("two", "2+ diagnoses", PALETTE[3])]:
    ax.plot(dstrat.window_minutes, DX[key][0], marker="s", ms=4, color=c, lw=1.6, label=f"{lab} (n = {DX[key][1]:,})")
ax.set_ylabel("Mean CFA R²")
ax.legend(frameon=False, fontsize=7.5)
title(ax, "Duration effect by diagnosis")

for ax in axes[1]:
    ax.set_xlabel("Segment duration (min)")
    ax.set_xticks(TICKS)
    ax.set_xlim(2, 93)

fig.tight_layout(h_pad=2.2, w_pad=2.0)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, "figures", f"fig3_bmi_sex_duration.{ext}"), bbox_inches="tight")
print("Figure written to figures/fig3_bmi_sex_duration.png")
