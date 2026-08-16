#!/usr/bin/env python3
"""Full-cohort window-length dose-response: rather than reporting CFA R^2 at
a single arbitrary 10-minute window (shown by window_stage_sensitivity.py's
n=23 subsample to depend significantly on window length), this reruns the
regression-based estimator at several lengths across the *entire* corpus and
reports R^2 as a function of length, and re-checks whether the age/sex/
diagnosis stratification (paper_stats.json's "stratified" block) replicates
at every length -- not only at 10 minutes.

Requires cfa_combined.parquet (10 min, from build_dataset.py) plus
cfa_variance_explained_{N}min.parquet for each additional length produced by
cfa_multi_length.log's reruns.

Run: venv/bin/python "Paper CFA/make_dose_response.py"
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
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300,
})

demo = pd.read_parquet(os.path.join(HERE, "demographics_combined.parquet"))

length_files = {10: os.path.join(HERE, "cfa_combined.parquet")}
for path in glob.glob(os.path.join(HERE, "cfa_variance_explained_*min.parquet")):
    m = re.search(r"_(\d+)min\.parquet$", path)
    if m:
        length_files[int(m.group(1))] = path

lengths = sorted(length_files)
print("Window lengths found:", lengths)

# Cohort-match: each length's raw output comes from a different eligible-patient
# snapshot (10min via the paper's QC'd build_dataset.py pipeline, others straight
# from cfa_variance_explained.py), so compare only patients present at every length.
patient_sets = {m: set(pd.read_parquet(f, columns=["patient_id"]).patient_id.unique())
                 for m, f in length_files.items()}
common_patients = set.intersection(*patient_sets.values())
print(f"Common patients across all {len(lengths)} lengths: {len(common_patients):,}")

overall_rows, strat_rows = [], []
for minutes in lengths:
    df = pd.read_parquet(length_files[minutes])
    df = df[df.patient_id.isin(common_patients)]
    overall_rows.append({
        "window_minutes": minutes, "n_patients": df.patient_id.nunique(), "n_rows": len(df),
        "mean": df.cfa_r2_excl_qrs.mean(), "sem": df.cfa_r2_excl_qrs.sem(),
        "median": df.cfa_r2_excl_qrs.median(),
    })
    pt = df.groupby("patient_id", as_index=False)["cfa_r2_excl_qrs"].mean().merge(
        demo[["patient_id", "age", "sex", "n_diagnoses"]], on="patient_id", how="left")
    pt = pt.dropna(subset=["age"])
    pt["age_tertile"] = pd.qcut(pt["age"], 3, labels=["Younger", "Middle", "Older"])
    age_groups = [pt.loc[pt.age_tertile == g, "cfa_r2_excl_qrs"].dropna().values for g in ["Younger", "Middle", "Older"]]
    _, p_age = stats.f_oneway(*age_groups) if all(len(g) > 1 for g in age_groups) else (np.nan, np.nan)
    sex_groups = [pt.loc[pt.sex == s, "cfa_r2_excl_qrs"].dropna().values for s in ["Male", "Female"] if (pt.sex == s).any()]
    _, p_sex = stats.mannwhitneyu(*sex_groups) if len(sex_groups) == 2 else (np.nan, np.nan)
    dx_groups = [pt.loc[pt.n_diagnoses.fillna(0) == 0, "cfa_r2_excl_qrs"].dropna().values,
                 pt.loc[pt.n_diagnoses.fillna(0) > 0, "cfa_r2_excl_qrs"].dropna().values]
    _, p_dx = stats.mannwhitneyu(*dx_groups) if all(len(g) > 1 for g in dx_groups) else (np.nan, np.nan)
    strat_rows.append({
        "window_minutes": minutes, "n_with_age": len(pt),
        "younger_mean": np.mean(age_groups[0]) if len(age_groups[0]) else np.nan,
        "middle_mean": np.mean(age_groups[1]) if len(age_groups[1]) else np.nan,
        "older_mean": np.mean(age_groups[2]) if len(age_groups[2]) else np.nan,
        "p_age": p_age,
        "male_mean": np.mean(sex_groups[0]) if sex_groups else np.nan,
        "female_mean": np.mean(sex_groups[1]) if len(sex_groups) > 1 else np.nan,
        "p_sex": p_sex,
        "no_dx_mean": np.mean(dx_groups[0]) if len(dx_groups[0]) else np.nan,
        "any_dx_mean": np.mean(dx_groups[1]) if len(dx_groups[1]) else np.nan,
        "p_dx": p_dx,
    })
    print(f"{minutes} min: n={df.patient_id.nunique():,} mean R2={df.cfa_r2_excl_qrs.mean():.3f} "
          f"| p_age={p_age:.2g} p_sex={p_sex:.2g} p_dx={p_dx:.2g}")

overall = pd.DataFrame(overall_rows).sort_values("window_minutes")
strat = pd.DataFrame(strat_rows).sort_values("window_minutes")

# ---------------------------------------------------------------------
# Figure: dose-response curve (full cohort, all lengths) + stratified panels
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

ax = axes[0]
ax.errorbar(overall.window_minutes, overall["mean"], yerr=overall["sem"] * 1.96,
            marker="o", capsize=4, color=PALETTE[0], lw=1.6)
for _, r in overall.iterrows():
    ax.annotate(f"n={r.n_patients:,.0f}", (r.window_minutes, r["mean"]), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=7)
ax.set_xlabel("Window length (minutes)")
ax.set_ylabel("Mean CFA R² (outside QRS), full cohort ± 95% CI")
ax.set_title("a  Dose-response, patients common to all lengths", loc="left", fontsize=10, fontweight="bold")

ax = axes[1]
ax.plot(strat.window_minutes, strat.male_mean, marker="o", color=PALETTE[0], label="Male")
ax.plot(strat.window_minutes, strat.female_mean, marker="o", color=PALETTE[3], label="Female")
ax.plot(strat.window_minutes, strat.any_dx_mean, marker="s", color=PALETTE[1], linestyle="--", label="≥1 diagnosis")
ax.plot(strat.window_minutes, strat.no_dx_mean, marker="s", color=PALETTE[2], linestyle="--", label="No diagnosis")
ax.set_xlabel("Window length (minutes)")
ax.set_ylabel("Mean CFA R²")
ax.set_title("b  Do sex/diagnosis gaps hold at every length?", loc="left", fontsize=10, fontweight="bold")
ax.legend(fontsize=7, frameon=False)

fig.suptitle(
    f"Figure 5. Time-domain CFA R² vs. window length "
    f"(n={len(common_patients):,} patients common to all 4 lengths)",
    fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(FIG_DIR, "figS7_dose_response.pdf"))
fig.savefig(os.path.join(FIG_DIR, "figS7_dose_response.png"))
plt.close(fig)

with open(os.path.join(HERE, "paper_stats.json")) as f:
    S = json.load(f)
S["dose_response"] = {
    "n_common_patients": len(common_patients),
    "lengths": overall.to_dict(orient="records"),
    "stratified_by_length": strat.to_dict(orient="records"),
}
with open(os.path.join(HERE, "paper_stats.json"), "w") as f:
    json.dump(S, f, indent=2, default=float)
print("\nWrote figure and paper_stats.json['dose_response']")
