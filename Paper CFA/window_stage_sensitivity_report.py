#!/usr/bin/env python3
"""Summarize window_stage_sensitivity.parquet (does sleep stage matter, small
subsample) and the full-cohort window-length dose-response (does window
length matter, full cohort) in one Supplementary figure.

Panel a uses the small 23-patient subsample (only place stage is available).
Panels b-d use the full cohort at four window lengths (5/10/20/30 min,
restricted to patients present at every length) — more statistical power
than a small-subsample length sweep, so that comparison is made there
instead of on the subsample.

Requires cfa_combined.parquet (10 min) plus cfa_variance_explained_{N}min.parquet
for each additional length (produced by cfa_variance_explained.py reruns).

Run: venv/bin/python "Paper CFA/window_stage_sensitivity_report.py"
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
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00", "#000000"]
STAGE_ORDER = ["W", "light_sleep", "N3", "R"]
STAGE_LABELS = {"W": "Wake", "light_sleep": "Light (N1+N2)", "N3": "N3", "R": "REM"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300,
})

# =============================================================================
# Part 1: sleep-stage and window-length sensitivity, small subsample
# =============================================================================
df = pd.read_parquet(os.path.join(HERE, "window_stage_sensitivity.parquet"))
long_path = os.path.join(HERE, "window_stage_sensitivity_long.parquet")
if os.path.exists(long_path):
    df = pd.concat([df, pd.read_parquet(long_path)], ignore_index=True)
print(f"{len(df):,} rows, {df.patient_id.nunique()} patients, "
      f"{df.groupby(['patient_id']).ngroups} patient-recordings")

by_stage = df.groupby("stage")["cfa_r2_excl_qrs_mean"].agg(["mean", "std", "count"])
by_stage = by_stage.reindex(STAGE_ORDER)
print("\n=== CFA R^2 (excl. QRS) by sleep stage, pooled across all window lengths ===")
print(by_stage)
stage_groups = [df.loc[df.stage == s, "cfa_r2_excl_qrs_mean"].dropna().values for s in STAGE_ORDER]
stage_groups_nonempty = [g for g in stage_groups if len(g) > 1]
f_stat, p_stage = stats.f_oneway(*stage_groups_nonempty) if len(stage_groups_nonempty) > 1 else (np.nan, np.nan)
print(f"One-way ANOVA across stages: F={f_stat:.3g}, p={p_stage:.3g}")

by_len = df.groupby("window_minutes")["cfa_r2_excl_qrs_mean"].agg(["mean", "std", "count"])
sub_lens = sorted(df.window_minutes.unique())
sub_len_groups = [df.loc[df.window_minutes == m, "cfa_r2_excl_qrs_mean"].dropna().values for m in sub_lens]
f_len, p_len = stats.f_oneway(*[g for g in sub_len_groups if len(g) > 1])
print(f"One-way ANOVA across window lengths (subsample): F={f_len:.3g}, p={p_len:.3g}")

cell = df.groupby(["patient_id", "stage", "window_minutes"])["cfa_r2_excl_qrs_mean"]
location_spread = cell.std().dropna()  # within-cell SD across draws (location effect)
length_spread = df.groupby(["patient_id", "stage"])["cfa_r2_excl_qrs_mean"].std().dropna()  # across-length SD
print(f"\nWithin-cell (same patient/stage/length, different random location) SD: "
      f"mean={location_spread.mean():.3f}, median={location_spread.median():.3f}, n_cells={len(location_spread)}")
print(f"Across-length (same patient/stage, pooled over lengths+locations) SD: "
      f"mean={length_spread.mean():.3f}, median={length_spread.median():.3f}, n_cells={len(length_spread)}")

# =============================================================================
# Part 2: full-cohort window-length dose-response (cohort-matched)
# =============================================================================
demo = pd.read_parquet(os.path.join(HERE, "demographics_combined.parquet"))
length_files = {10: os.path.join(HERE, "cfa_combined.parquet")}
for path in glob.glob(os.path.join(HERE, "cfa_variance_explained_*min.parquet")):
    m = re.search(r"_(\d+)min\.parquet$", path)
    if m:
        length_files[int(m.group(1))] = path
lengths = sorted(length_files)
print("\nFull-cohort window lengths found:", lengths)

patient_sets = {m: set(pd.read_parquet(f, columns=["patient_id"]).patient_id.unique())
                 for m, f in length_files.items()}
common_patients = set.intersection(*patient_sets.values())
print(f"Common patients across all {len(lengths)} full-cohort lengths: {len(common_patients):,}")

overall_rows, strat_rows = [], []
for minutes in lengths:
    fdf = pd.read_parquet(length_files[minutes])
    fdf = fdf[fdf.patient_id.isin(common_patients)]
    overall_rows.append({
        "window_minutes": minutes, "n_patients": fdf.patient_id.nunique(), "n_rows": len(fdf),
        "mean": fdf.cfa_r2_excl_qrs.mean(), "sem": fdf.cfa_r2_excl_qrs.sem(),
        "median": fdf.cfa_r2_excl_qrs.median(),
    })
    pt = fdf.groupby("patient_id", as_index=False)["cfa_r2_excl_qrs"].mean().merge(
        demo[["patient_id", "age", "sex", "n_diagnoses"]], on="patient_id", how="left")
    pt = pt.dropna(subset=["age"])
    sex_groups = [pt.loc[pt.sex == s, "cfa_r2_excl_qrs"].dropna().values for s in ["Male", "Female"] if (pt.sex == s).any()]
    _, p_sex = stats.mannwhitneyu(*sex_groups) if len(sex_groups) == 2 else (np.nan, np.nan)
    dx_groups = [pt.loc[pt.n_diagnoses.fillna(0) == 0, "cfa_r2_excl_qrs"].dropna().values,
                 pt.loc[pt.n_diagnoses.fillna(0) > 0, "cfa_r2_excl_qrs"].dropna().values]
    _, p_dx = stats.mannwhitneyu(*dx_groups) if all(len(g) > 1 for g in dx_groups) else (np.nan, np.nan)
    strat_rows.append({
        "window_minutes": minutes, "n_with_age": len(pt),
        "male_mean": np.mean(sex_groups[0]) if sex_groups else np.nan,
        "female_mean": np.mean(sex_groups[1]) if len(sex_groups) > 1 else np.nan,
        "p_sex": p_sex,
        "no_dx_mean": np.mean(dx_groups[0]) if len(dx_groups[0]) else np.nan,
        "any_dx_mean": np.mean(dx_groups[1]) if len(dx_groups[1]) else np.nan,
        "p_dx": p_dx,
    })
    print(f"{minutes} min: n={fdf.patient_id.nunique():,} mean R2={fdf.cfa_r2_excl_qrs.mean():.3f} "
          f"| p_sex={p_sex:.2g} p_dx={p_dx:.2g}")

overall = pd.DataFrame(overall_rows).sort_values("window_minutes")
strat = pd.DataFrame(strat_rows).sort_values("window_minutes")

# =============================================================================
# Figure: a) dose-response (full cohort), b) sex by length (full cohort),
#         c) diagnosis by length (full cohort)
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(11.5, 4))

ax = axes[0]
ax.errorbar(overall.window_minutes, overall["mean"], yerr=overall["sem"] * 1.96,
            marker="o", capsize=4, color=PALETTE[0], lw=1.6)
ax.set_xlabel("Window length (min)")
ax.set_ylabel("Mean CFA R², full cohort ± 95% CI")
ax.set_title("a  Dose-response", loc="left", fontsize=9.5, fontweight="bold")

ax = axes[1]
ax.plot(strat.window_minutes, strat.male_mean, marker="o", color=PALETTE[0], label="Male")
ax.plot(strat.window_minutes, strat.female_mean, marker="o", color=PALETTE[3], label="Female")
ax.set_xlabel("Window length (min)")
ax.set_ylabel("Mean CFA R²")
p_sex_range = f"{strat.p_sex.max():.2g}"
ax.set_title(f"b  By sex (all p < {p_sex_range})", loc="left", fontsize=9.5, fontweight="bold")
ax.legend(fontsize=7, frameon=False)

ax = axes[2]
ax.plot(strat.window_minutes, strat.any_dx_mean, marker="s", color=PALETTE[1], label="≥1 diagnosis")
ax.plot(strat.window_minutes, strat.no_dx_mean, marker="s", color=PALETTE[2], label="No diagnosis")
ax.set_xlabel("Window length (min)")
ax.set_ylabel("Mean CFA R²")
p_dx_range = f"{strat.p_dx.max():.2g}"
ax.set_title(f"c  By diagnosis (all p < {p_dx_range})", loc="left", fontsize=9.5, fontweight="bold")
ax.legend(fontsize=7, frameon=False)

fig.suptitle(
    f"Figure S2. Does window length matter? Full-cohort dose-response, "
    f"n={len(common_patients):,} common to all lengths", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(os.path.join(FIG_DIR, "figS2_window_stage_sensitivity.pdf"))
fig.savefig(os.path.join(FIG_DIR, "figS2_window_stage_sensitivity.png"))
plt.close(fig)
print("\nFigure written to", os.path.join(FIG_DIR, "figS2_window_stage_sensitivity.png"))

stats_out = {
    "n_patients": int(df.patient_id.nunique()), "n_rows": int(len(df)),
    "window_lengths_min": [int(m) for m in sub_lens], "draws_per_cell": int(df.draw.nunique()),
    "by_stage_mean": {s: float(by_stage.loc[s, "mean"]) for s in STAGE_ORDER if s in by_stage.index and pd.notna(by_stage.loc[s, "mean"])},
    "p_stage_anova": float(p_stage),
    "by_length_mean": {str(int(m)): float(by_len.loc[m, "mean"]) for m in sub_lens},
    "p_length_anova": float(p_len),
    "location_sd_mean": float(location_spread.mean()), "location_sd_median": float(location_spread.median()),
    "length_sd_mean": float(length_spread.mean()), "length_sd_median": float(length_spread.median()),
}
with open(os.path.join(HERE, "window_stage_sensitivity_stats.json"), "w") as f:
    json.dump(stats_out, f, indent=2)
print("Stats written to", os.path.join(HERE, "window_stage_sensitivity_stats.json"))

with open(os.path.join(HERE, "paper_stats.json")) as f:
    S = json.load(f)
S["dose_response"] = {
    "n_common_patients": len(common_patients),
    "lengths": overall.to_dict(orient="records"),
    "stratified_by_length": strat.to_dict(orient="records"),
}
with open(os.path.join(HERE, "paper_stats.json"), "w") as f:
    json.dump(S, f, indent=2, default=float)
print("Wrote paper_stats.json['dose_response']")
