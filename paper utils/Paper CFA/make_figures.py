#!/usr/bin/env python3
"""Build Fig 1-4 for the CFA variance-explained paper from the parquet tables
written by build_dataset.py.

Run: venv/bin/python "Paper CFA/make_figures.py"
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

from channel_utils import canonicalize, filter_min_coverage

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Wong (2011) colorblind-safe palette, reused verbatim from
# 16_diagnosis_sleep_stage_comparison_dashboard.py so figures match the
# rest of the project's visual language.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00", "#000000"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "svg.fonttype": "none",
})

cfa = pd.read_parquet(os.path.join(HERE, "cfa_combined.parquet"))
ica = pd.read_parquet(os.path.join(HERE, "ica_combined.parquet"))
demo = pd.read_parquet(os.path.join(HERE, "demographics_combined.parquet"))

# Patient-level CFA summary: mean R^2 across electrodes per patient, restricted
# to the same 6 well-covered canonical sites (>=50% patient coverage, §2.7)
# used by every other per-electrode figure — averaging over whatever raw
# channels each patient happens to have would let differing channel
# composition across patients/diagnoses confound the patient-level metric.
cfa["canon"] = canonicalize(cfa["eeg_channel"])
cfa_6ch = filter_min_coverage(cfa.dropna(subset=["canon"]), "patient_id", "canon", min_frac=0.5)
cfa_pt = cfa_6ch.groupby("patient_id", as_index=False).agg(
    cfa_r2_full=("cfa_r2_full_epoch", "mean"),
    cfa_r2_excl_qrs=("cfa_r2_excl_qrs", "mean"),
    n_channels=("canon", "nunique"),
)
cfa_pt = cfa_pt.merge(demo[["patient_id", "bdsp_patient_id", "age", "sex", "diagnosis_categories", "n_diagnoses"]], on="patient_id", how="left")

# BMI, cached from the I0002/I0004/I0006 vitals & flowsheet tables
# (§ build_bmi_cache.py) — the linked sites with height/weight or BMI data.
bmi = pd.read_parquet(os.path.join(HERE, "bmi_combined.parquet"))
cfa_pt = cfa_pt.merge(bmi[["bdsp_patient_id", "bmi"]], on="bdsp_patient_id", how="left")

# Patient-level ICA summary: the flagged ECG-artifact component's variance
# fraction of the channel's HEP evoked variance, and the actual pre/post
# cleaning drop — averaged over channels per patient.
ica_flag = ica[ica.ecg_artifact_component]
ica_pt = ica_flag.groupby("patient_id", as_index=False).agg(
    ecg_component_variance_fraction=("channel_component_variance_fraction", "mean"),
    hep_variance_pct_drop=("channel_hep_variance_pct_drop", "mean"),
)
ica_pt = ica_pt.merge(demo[["patient_id", "age", "sex", "diagnosis_categories", "n_diagnoses"]], on="patient_id", how="left")

STATS = {}

# =============================================================================
# Figure 1: cohort / population overview
# =============================================================================
demo_age = demo.dropna(subset=["age"])
demo_sex = demo.dropna(subset=["sex"])
dx_counts = (
    demo.explode("diagnosis_categories")
    .dropna(subset=["diagnosis_categories"])
    .diagnosis_categories.value_counts()
    .sort_values(ascending=True)
)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))

ax = axes[0]
ax.hist(demo_age.age, bins=30, color=PALETTE[0], edgecolor="white", linewidth=0.4)
ax.set_xlabel("Age (years)")
ax.set_ylabel("Patients")
ax.set_title("a", loc="left", fontsize=10, fontweight="bold")
# ponytail: headroom so the n/median label clears the tallest bars
_lo, _hi = ax.get_ylim()
ax.set_ylim(_lo, _hi * 1.15)
ax.text(0.97, 0.95, f"n = {len(demo_age):,}\nmedian = {demo_age.age.median():.0f} yr",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.5)

ax = axes[1]
sex_counts = demo_sex.sex.value_counts()
ax.pie(sex_counts.values, labels=[f"{i} ({v:,})" for i, v in sex_counts.items()],
       colors=[PALETTE[0], PALETTE[1], PALETTE[2]][:len(sex_counts)], autopct="%1.0f%%",
       textprops={"fontsize": 8}, wedgeprops={"linewidth": 0.6, "edgecolor": "white"})
ax.set_title("b", loc="left", fontsize=10, fontweight="bold")

ax = axes[2]
top_dx = dx_counts.tail(10)
ax.barh(top_dx.index, top_dx.values, color=PALETTE[3])
ax.set_xlabel("Patients with diagnosis")
ax.set_title("c", loc="left", fontsize=10, fontweight="bold")
ax.tick_params(axis="y", labelsize=7.5)

fig.suptitle(f"Figure S1. Cohort composition (Harvard EHR-linked subset, n = {len(demo):,})", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(FIG_DIR, "figS1_cohort.pdf"))
fig.savefig(os.path.join(FIG_DIR, "figS1_cohort.png"))
plt.close(fig)

STATS["cohort"] = {
    "n_demographics": int(len(demo)), "n_age": int(len(demo_age)),
    "age_median": float(demo_age.age.median()), "age_mean": float(demo_age.age.mean()),
    "age_sd": float(demo_age.age.std()), "age_min": float(demo_age.age.min()), "age_max": float(demo_age.age.max()),
    "n_sex": int(len(demo_sex)), "sex_counts": {str(k): int(v) for k, v in sex_counts.items()},
    "top_diagnoses": {str(k): int(v) for k, v in dx_counts.sort_values(ascending=False).items()},
}

# =============================================================================
# Figure 2: CFA variance explained — distribution, full epoch vs. outside QRS
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

ax = axes[0]
data = [cfa["cfa_r2_full_epoch"].dropna().values, cfa["cfa_r2_excl_qrs"].dropna().values]
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
ax.set_ylabel("Channel HEP-evoked R² vs. ECG evoked average")
ax.set_title("a", loc="left", fontsize=10, fontweight="bold")
# ponytail: headroom so the mean±sd label clears the violins' max bars
_lo, _hi = ax.get_ylim()
ax.set_ylim(_lo, _hi + 0.18 * (_hi - _lo))
ax.text(0.03, 0.97,
        f"full: {cfa['cfa_r2_full_epoch'].mean():.2f} ± {cfa['cfa_r2_full_epoch'].std():.2f}\n"
        f"excl. QRS: {cfa['cfa_r2_excl_qrs'].mean():.2f} ± {cfa['cfa_r2_excl_qrs'].std():.2f}",
        transform=ax.transAxes, va="top", fontsize=7.5)

ax = axes[1]
vals_raw = ica_flag["channel_component_variance_fraction"].dropna().values
vals = vals_raw
snr_vals = vals_raw / (1 - vals_raw)
median_snr = float(np.median(snr_vals))

ica_all = ica_flag.copy()
ica_all["snr"] = ica_all["channel_component_variance_fraction"] / (1 - ica_all["channel_component_variance_fraction"])
ica_all = ica_all.sort_values("channel_component_variance_fraction", ascending=False).drop_duplicates(
    ["patient_id", "edf_path", "eeg_channel"])
merged = ica_all[["patient_id", "edf_path", "eeg_channel", "snr"]].merge(
    cfa[["patient_id", "edf_path", "eeg_channel", "cfa_r2_excl_qrs"]],
    on=["patient_id", "edf_path", "eeg_channel"], how="inner"
).dropna(subset=["snr", "cfa_r2_excl_qrs"])
x_r2 = merged["cfa_r2_excl_qrs"].values
y_snr = merged["snr"].values
y_log = np.log10(y_snr)
r_scatter, p_scatter = stats.pearsonr(x_r2, y_log)
slope, intercept = np.polyfit(x_r2, y_log, 1)
ax.scatter(x_r2, y_snr, s=4, alpha=0.10, color=PALETTE[2], edgecolor="none")
xs = np.linspace(x_r2.min(), x_r2.max(), 100)
ax.plot(xs, 10 ** (slope * xs + intercept), color=PALETTE[1], lw=1.8)
ax.set_yscale("log")
ax.set_xlabel("Model-free CFA R² (outside QRS)")
ax.set_ylabel("ICA-attributed CFA SNR\n(variance ratio, CFA:residual)")
ax.set_title("b", loc="left", fontsize=10, fontweight="bold")
p_scatter_str = "< 1e-300" if p_scatter == 0 else f"= {p_scatter:.2g}"
ax.text(0.03, 0.05, f"r (log SNR) = {r_scatter:.2f}, p {p_scatter_str}\nn = {len(merged):,}",
        transform=ax.transAxes, va="bottom", fontsize=7.5)

fig.suptitle(
    f"Figure 1. Model-free and ICA-based cardiac-field-artifact (CFA) variance explained\n"
    f"(n = {cfa.patient_id.nunique():,} patients, {len(cfa):,} channel-recordings)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(os.path.join(FIG_DIR, "fig1_cfa_r2.pdf"))
fig.savefig(os.path.join(FIG_DIR, "fig1_cfa_r2.png"))
plt.close(fig)

STATS["cfa"] = {
    "n_patients": int(cfa.patient_id.nunique()), "n_rows": int(len(cfa)),
    "r2_full_mean": float(cfa.cfa_r2_full_epoch.mean()), "r2_full_sd": float(cfa.cfa_r2_full_epoch.std()),
    "r2_full_median": float(cfa.cfa_r2_full_epoch.median()),
    "r2_excl_qrs_mean": float(cfa.cfa_r2_excl_qrs.mean()), "r2_excl_qrs_sd": float(cfa.cfa_r2_excl_qrs.std()),
    "r2_excl_qrs_median": float(cfa.cfa_r2_excl_qrs.median()),
    "bpm_quartiles": {str(q): float(v) for q, v in
                      cfa.drop_duplicates("edf_path").qc_ecg_bpm.quantile([.25, .5, .75]).items()},
}

drop = ica_flag["channel_hep_variance_pct_drop"].dropna()
drop = drop[np.isfinite(drop)]

STATS["ica"] = {
    "n_patients": int(ica.patient_id.nunique()), "n_rows": int(len(ica)),
    "component_variance_fraction_median": float(np.median(vals)), "component_variance_fraction_mean": float(np.mean(vals)),
    "snr_median": median_snr,
    "r2_vs_snr_r": float(r_scatter), "r2_vs_snr_p": float(p_scatter), "r2_vs_snr_n": int(len(merged)),
    "component_variance_fraction_median_unfiltered": float(np.median(vals_raw)),
    "hep_pct_drop_median": float(drop.median()), "hep_pct_drop_mean": float(drop.mean()),
}

# =============================================================================
# Figure 3: CFA variance explained by diagnosis category
# =============================================================================
dx_exploded = cfa_pt.dropna(subset=["cfa_r2_excl_qrs"]).explode("diagnosis_categories")
top_categories = list(STATS["cohort"]["top_diagnoses"].keys())
# EHR-linked patients (age known) with no recorded diagnosis; unlinked patients have unknown status
no_dx = cfa_pt.loc[(cfa_pt["n_diagnoses"] == 0) & cfa_pt["age"].notna(), "cfa_r2_excl_qrs"].dropna().values
no_dx_mean, no_dx_n = float(np.mean(no_dx)), len(no_dx)
no_dx_var = float(np.var(no_dx, ddof=1))

group_data = {}
for cat in top_categories:
    vals_cat = dx_exploded.loc[dx_exploded["diagnosis_categories"] == cat, "cfa_r2_excl_qrs"].dropna().values
    if len(vals_cat) >= 10:
        group_data[cat] = vals_cat

# Absolute mean CFA R^2 per category (not a difference from the reference),
# with a Welch (unequal-variance) 95% CI on the mean — easier to read
# directly than a pre-subtracted effect size; the no-diagnosis mean is drawn
# as a reference line rather than baked into each point.
rows = []
for cat, vals_cat in group_data.items():
    n_cat = len(vals_cat)
    mean_cat = float(np.mean(vals_cat))
    var_cat = float(np.var(vals_cat, ddof=1))
    se_mean = np.sqrt(var_cat / n_cat)
    ci_lo, ci_hi = mean_cat - 1.96 * se_mean, mean_cat + 1.96 * se_mean
    diff = mean_cat - no_dx_mean
    _, p_cat = stats.mannwhitneyu(vals_cat, no_dx)
    rows.append({"category": cat, "n": n_cat, "mean": mean_cat, "ci_lo": ci_lo, "ci_hi": ci_hi,
                 "diff": diff, "p": p_cat})

forest = pd.DataFrame(rows).sort_values("mean", ascending=True).reset_index(drop=True)
_, forest["q"], _, _ = multipletests(forest["p"].values, method="fdr_bh")
kw_stat, p_kw = stats.kruskal(*group_data.values(), no_dx)

# All pairwise category-vs-category comparisons (not just vs. no-diagnosis),
# BH-FDR corrected across the full pairwise family.
pair_cats = list(group_data.keys())
pair_rows = []
for i in range(len(pair_cats)):
    for j in range(i + 1, len(pair_cats)):
        a, b = pair_cats[i], pair_cats[j]
        _, p_ab = stats.mannwhitneyu(group_data[a], group_data[b])
        pair_rows.append({"cat_a": a, "cat_b": b, "p": p_ab})
pairwise = pd.DataFrame(pair_rows)
_, pairwise["q"], _, _ = multipletests(pairwise["p"].values, method="fdr_bh")

sig_pairwise = pairwise[pairwise["q"] < 0.05].sort_values("q")
n_cats = len(pair_cats)
qmat = pd.DataFrame(np.nan, index=pair_cats, columns=pair_cats)
for row in pairwise.itertuples():
    qmat.loc[row.cat_a, row.cat_b] = row.q
    qmat.loc[row.cat_b, row.cat_a] = row.q
order_cats = forest["category"].tolist()
qmat = qmat.loc[order_cats, order_cats]

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1, 1.15]})

ax = axes[0]
y = np.arange(len(forest))
colors_pt = [PALETTE[1] if row.diff < 0 else PALETTE[0] for row in forest.itertuples()]
ax.errorbar(forest["mean"], y, xerr=[forest["mean"] - forest["ci_lo"], forest["ci_hi"] - forest["mean"]],
            fmt="none", ecolor="black", elinewidth=1, capsize=3, zorder=1)
ax.scatter(forest["mean"], y, c=colors_pt, s=45, zorder=2, edgecolor="black", linewidth=0.5)
ax.set_yticks(y)
ax.set_yticklabels(forest["category"], fontsize=8)
data_lo = float(forest["ci_lo"].min())
data_hi = float(forest["ci_hi"].max())
data_range = data_hi - data_lo
left = data_lo - 0.04 * data_range
n_col_x = data_hi + 0.14 * data_range
ax.set_xlim(left, data_hi + 0.28 * data_range)
for yi, row in zip(y, forest.itertuples()):
    ax.text(n_col_x, yi, f"n={row.n:,}", va="center", ha="left", fontsize=7.5)
ax.text(n_col_x, len(forest) - 0.3, "N", va="center", ha="left", fontsize=7.5, fontweight="bold")
ax.set_xlabel(f"Mean CFA R² (outside QRS)\n(no linked diagnosis: mean = {no_dx_mean:.2f}, n = {no_dx_n:,})")
ax.set_title("a", loc="left", fontsize=10, fontweight="bold")

ax = axes[1]
mask = np.triu(np.ones_like(qmat, dtype=bool))
plot_mat = qmat.mask(mask)
norm = colors.Normalize(vmin=0, vmax=0.5, clip=True)
im = ax.imshow(plot_mat.values, cmap="Reds_r", norm=norm)
ax.set_xticks(range(n_cats))
ax.set_xticklabels(order_cats, rotation=90, fontsize=7.5)
ax.set_yticks(range(n_cats))
ax.set_yticklabels(order_cats, fontsize=7.5)
cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.03, extend="max")
cbar.set_label("FDR p-value", fontsize=8)
ax.set_title("b", loc="left", fontsize=10, fontweight="bold")

fig.suptitle(
    f"Figure 3. CFA variance explained by diagnosis category\n"
    f"(Kruskal-Wallis across all groups p = {p_kw:.2g})", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(FIG_DIR, "fig3_diagnosis.pdf"))
fig.savefig(os.path.join(FIG_DIR, "fig3_diagnosis.png"))
plt.close(fig)

STATS["diagnosis"] = {
    "categories": {row.category: {"n": int(row.n), "mean": float(row.mean), "ci_lo": float(row.ci_lo),
                                    "ci_hi": float(row.ci_hi), "diff": float(row.diff), "p": float(row.p),
                                    "q": float(row.q)}
                   for row in forest.itertuples()},
    "p_kruskal": float(p_kw),
    "highest_category": forest.iloc[-1]["category"], "highest_diff": float(forest.iloc[-1]["diff"]),
    "lowest_category": forest.iloc[0]["category"], "lowest_diff": float(forest.iloc[0]["diff"]),
    "no_dx_mean": no_dx_mean, "no_dx_n": no_dx_n,
    "n_categories_significant": int((forest["p"] < 0.05).sum()),
    "n_categories_significant_fdr": int((forest["q"] < 0.05).sum()),
    "n_categories_total": int(len(forest)),
    "pairwise": {
        "n_pairs": int(len(pairwise)), "n_significant_fdr": int(len(sig_pairwise)),
        "significant_pairs": [{"a": r.cat_a, "b": r.cat_b, "p": float(r.p), "q": float(r.q)}
                               for r in sig_pairwise.itertuples()],
    },
}

# =============================================================================
# Figure 4: variance explained stratified by age tertile, sex, diagnosis burden
# =============================================================================
strat = cfa_pt.dropna(subset=["age"]).copy()
strat["age_tertile"] = pd.qcut(strat["age"], 3, labels=["Younger", "Middle", "Older"])

fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))

ax = axes[0]
groups = [strat.loc[strat.age_tertile == g, "cfa_r2_excl_qrs"].dropna().values for g in ["Younger", "Middle", "Older"]]
f_stat, p_age = stats.f_oneway(*groups)

age_x = strat["age"].values
r2_y = strat["cfa_r2_excl_qrs"].values
r_age, p_age_pearson = stats.pearsonr(age_x, r2_y)
slope, intercept = np.polyfit(age_x, r2_y, 1)
ax.scatter(age_x, r2_y, s=6, alpha=0.15, color=PALETTE[0], edgecolor="none")
xs = np.linspace(age_x.min(), age_x.max(), 100)
ax.plot(xs, slope * xs + intercept, color=PALETTE[1], lw=1.8)
ax.set_xlabel("Age (years)")
ax.set_ylabel("Patient-mean CFA R² (outside QRS)")
ax.set_title("a", loc="left", fontsize=10, fontweight="bold")
ax.text(0.03, 0.97, f"n = {len(age_x):,}", transform=ax.transAxes, va="top", fontsize=7.5)

ax = axes[1]
sex_groups = {
    s: strat.loc[strat.sex == s, "cfa_r2_excl_qrs"].dropna().values
    for s in ["Female", "Male"]
    if (strat.sex == s).any()
}
bp = ax.boxplot(list(sex_groups.values()), labels=list(sex_groups.keys()), patch_artist=True, widths=0.5, showfliers=False)
sex_colors = {"Female": PALETTE[3], "Male": PALETTE[0]}
for patch, sex in zip(bp["boxes"], sex_groups):
    color = sex_colors[sex]
    patch.set_facecolor(color)
    patch.set_alpha(0.65)
p_sex = np.nan
if len(sex_groups) == 2:
    vals2 = list(sex_groups.values())
    _, p_sex = stats.mannwhitneyu(vals2[0], vals2[1])
ax.set_title("b", loc="left", fontsize=10, fontweight="bold")

# Diagnostic-burden stats (used in text/abstract) computed but no longer plotted as a fig4 panel.
strat["any_dx"] = strat["n_diagnoses"].fillna(0) > 0
dx_groups = [strat.loc[~strat.any_dx, "cfa_r2_excl_qrs"].dropna().values,
             strat.loc[strat.any_dx, "cfa_r2_excl_qrs"].dropna().values]
p_dx = np.nan
if all(len(g) > 1 for g in dx_groups):
    _, p_dx = stats.mannwhitneyu(dx_groups[0], dx_groups[1])

ax = axes[2]
# Maximise measured-BMI coverage without duplicating patients: select each
# patient's longest available CFA window (30, then 20, 10, and 5 min). The
# longer windows are preferred because Figure 5 shows more stable CFA R² with
# increasing duration. Each duration is reduced to the same six-site patient
# mean before patient-level selection.
bmi_duration_frames = []
for window_minutes, filename in [
    (30, "cfa_variance_explained_30min.parquet"),
    (20, "cfa_variance_explained_20min.parquet"),
    (10, "cfa_combined.parquet"),
    (5, "cfa_variance_explained_5min.parquet"),
]:
    duration_cfa = pd.read_parquet(os.path.join(HERE, filename))
    duration_cfa["canon"] = canonicalize(duration_cfa["eeg_channel"])
    duration_cfa = filter_min_coverage(
        duration_cfa.dropna(subset=["canon"]), "patient_id", "canon", min_frac=0.5
    )
    duration_pt = duration_cfa.groupby("patient_id", as_index=False).agg(
        cfa_r2_excl_qrs=("cfa_r2_excl_qrs", "mean"),
        n_channels=("canon", "nunique"),
    )
    duration_pt["window_minutes"] = window_minutes
    bmi_duration_frames.append(duration_pt)

bmi_data = (
    pd.concat(bmi_duration_frames, ignore_index=True)
    .sort_values(["patient_id", "window_minutes"], ascending=[True, False])
    .drop_duplicates("patient_id", keep="first")
)
bmi_data["bdsp_patient_id"] = pd.to_numeric(
    bmi_data["patient_id"].astype(str).str.extract(r"I\d{4}(\d{9})", expand=False),
    errors="coerce",
)
target_bmi_ids = bmi_data["bdsp_patient_id"].dropna().astype("int64").unique().tolist()
ehr_sex = pd.read_parquet(
    os.path.join(HERE, "..", "..", "cache", "ehr_demographics.parquet"),
    columns=["patient_id", "sex"],
    filters=[("patient_id", "in", target_bmi_ids)],
)
ehr_sex["patient_id"] = pd.to_numeric(ehr_sex["patient_id"], errors="coerce")
ehr_sex = (
    ehr_sex.dropna(subset=["patient_id"])
    .assign(patient_id=lambda d: d["patient_id"].astype("int64"))
    .groupby("patient_id")["sex"]
    .agg(lambda s: s.dropna().mode().iat[0] if s.notna().any() else None)
    .rename("sex")
    .reset_index()
    .rename(columns={"patient_id": "bdsp_patient_id"})
)
bmi_data = (
    bmi_data.dropna(subset=["bdsp_patient_id"])
    .assign(bdsp_patient_id=lambda d: d["bdsp_patient_id"].astype("int64"))
    .merge(bmi[["bdsp_patient_id", "bmi"]], on="bdsp_patient_id", how="inner")
    .merge(ehr_sex, on="bdsp_patient_id", how="left")
    .dropna(subset=["bmi", "cfa_r2_excl_qrs"])
)
bmi_x = bmi_data["bmi"].values
r2_y_bmi = bmi_data["cfa_r2_excl_qrs"].values
r_bmi, p_bmi = stats.pearsonr(bmi_x, r2_y_bmi)
slope_bmi, intercept_bmi = np.polyfit(bmi_x, r2_y_bmi, 1)
bmi_by_sex = {}
for sex, color in [("Female", PALETTE[3]), ("Male", PALETTE[0])]:
    group = bmi_data[bmi_data["sex"] == sex]
    x_group = group["bmi"].values
    y_group = group["cfa_r2_excl_qrs"].values
    r_group, p_group = stats.pearsonr(x_group, y_group)
    slope_group, intercept_group = np.polyfit(x_group, y_group, 1)
    bmi_by_sex[sex] = {
        "n": int(len(group)), "r_pearson": float(r_group), "p_pearson": float(p_group),
        "slope": float(slope_group), "bmi_mean": float(np.mean(x_group)),
    }
    ax.scatter(x_group, y_group, s=6, alpha=0.10, color=color, edgecolor="none")
    xs_group = np.linspace(x_group.min(), x_group.max(), 100)
    ax.plot(xs_group, slope_group * xs_group + intercept_group, color=color, lw=1.8,
            label=f"{sex}: n={len(group):,}, r={r_group:.2f}, p={p_group:.1e}")

interaction_data = bmi_data[bmi_data["sex"].isin(["Male", "Female"])].copy()
interaction_data["sex_male"] = (interaction_data["sex"] == "Male").astype(int)
bmi_sex_interaction = smf.ols(
    "cfa_r2_excl_qrs ~ bmi * sex_male", data=interaction_data
).fit()
p_bmi_sex_interaction = float(bmi_sex_interaction.pvalues["bmi:sex_male"])
ax.set_xlabel("BMI (kg/m²)")
ax.set_ylabel("Patient-mean CFA R² (outside QRS)")
ax.set_title("c",
             loc="left", fontsize=10, fontweight="bold")
ax.legend(frameon=False, fontsize=7, loc="upper left")

fig.suptitle("Figure 4. CFA variance explained shifts modestly with sex, age, and BMI",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(FIG_DIR, "fig4_stratified.pdf"))
fig.savefig(os.path.join(FIG_DIR, "fig4_stratified.png"))
plt.close(fig)

STATS["stratified"] = {
    "n_with_age": int(len(strat)),
    "age_tertile_means": {g: float(strat.loc[strat.age_tertile == g, "cfa_r2_excl_qrs"].mean()) for g in ["Younger", "Middle", "Older"]},
    "p_age_anova": float(p_age),
    "r_age_pearson": float(r_age), "p_age_pearson": float(p_age_pearson), "age_slope": float(slope),
    "sex_means": {str(k): float(np.mean(v)) for k, v in sex_groups.items()},
    "p_sex_mannwhitney": float(p_sex) if p_sex == p_sex else None,
    "no_dx_mean": float(np.mean(dx_groups[0])) if len(dx_groups[0]) else None,
    "any_dx_mean": float(np.mean(dx_groups[1])) if len(dx_groups[1]) else None,
    "p_dx_mannwhitney": float(p_dx) if p_dx == p_dx else None,
    "bmi": {
        "n": int(len(bmi_data)), "r_pearson": float(r_bmi), "p_pearson": float(p_bmi),
        "slope": float(slope_bmi), "bmi_mean": float(np.mean(bmi_x)),
        "window_counts": {
            str(int(k)): int(v) for k, v in bmi_data["window_minutes"].value_counts().sort_index().items()
        },
        "underweight_n": int((bmi_data["bmi"] < 18.5).sum()),
        "underweight_pct": float((bmi_data["bmi"] < 18.5).mean() * 100),
        "by_sex": bmi_by_sex,
        "sex_interaction_p": p_bmi_sex_interaction,
        "n_with_sex": int(len(interaction_data)),
    },
}
# Diagnosis tables exist only for some HSP sites, so "no diagnosis" is partly
# a site marker; report the reference split by site (patient-ID prefix).
_site = strat["patient_id"].astype(str).str[:5]
_dx = np.where(strat["n_diagnoses"] > 0, "any_dx", "no_dx")
STATS["stratified"]["dx_by_site"] = {
    f"{s}_{d}": {"mean": float(g["cfa_r2_excl_qrs"].mean()), "n": int(len(g))}
    for (s, d), g in strat.groupby([_site, _dx])
}
STATS["cohort"]["source_counts"] = (
    cfa.drop_duplicates("patient_id")["edf_path"].str.split("EDF_Format/").str[1]
    .str.split("/").str[0].value_counts().to_dict()
)

# =============================================================================
# Figure S5: is the sex effect on CFA R^2 confounded by BMI?
# Male patients skew heavier; if BMI drives CFA R^2 and men have higher BMI,
# the raw sex gap (Figure 4b) could just be BMI in disguise. Fit the same
# unadjusted sex comparison as an OLS (same estimate as Fig 4b's group means,
# now with a CI) restricted to the BMI-available subset, then add BMI as a
# covariate in the same subset and see whether the sex coefficient survives.
# =============================================================================
confound = bmi_data.dropna(subset=["bmi", "sex", "cfa_r2_excl_qrs"])
confound = confound[confound["sex"].isin(["Male", "Female"])].copy()
confound["sex_male"] = (confound["sex"] == "Male").astype(int)

m_unadj = smf.ols("cfa_r2_excl_qrs ~ sex_male", data=confound).fit()
m_adj = smf.ols("cfa_r2_excl_qrs ~ sex_male + bmi", data=confound).fit()

sex_unadj_coef = float(m_unadj.params["sex_male"])
sex_unadj_p = float(m_unadj.pvalues["sex_male"])
sex_unadj_ci = m_unadj.conf_int().loc["sex_male"].values.astype(float)
sex_adj_coef = float(m_adj.params["sex_male"])
sex_adj_p = float(m_adj.pvalues["sex_male"])
sex_adj_ci = m_adj.conf_int().loc["sex_male"].values.astype(float)
bmi_coef = float(m_adj.params["bmi"])
bmi_p = float(m_adj.pvalues["bmi"])

fig, ax = plt.subplots(figsize=(5, 3.6))
y = [1, 0]
coefs = [sex_unadj_coef, sex_adj_coef]
los = [sex_unadj_coef - sex_unadj_ci[0], sex_adj_coef - sex_adj_ci[0]]
his = [sex_unadj_ci[1] - sex_unadj_coef, sex_adj_ci[1] - sex_adj_coef]
colors_pt = [PALETTE[0], PALETTE[1]]
ax.errorbar(coefs, y, xerr=[los, his], fmt="none", ecolor="black", elinewidth=1, capsize=4, zorder=1)
ax.scatter(coefs, y, c=colors_pt, s=70, zorder=2, edgecolor="black", linewidth=0.6)
ax.set_yticks(y)
ax.set_yticklabels([f"Unadjusted\n(p = {sex_unadj_p:.2g})", f"+ BMI covariate\n(p = {sex_adj_p:.2g})"])
ax.set_xlabel("Male − Female CFA R² (outside QRS)\ncoefficient, 95% CI")
ax.set_ylim(-0.6, 1.6)
data_lo = min(sex_unadj_ci[0], sex_adj_ci[0])
data_hi = max(sex_unadj_ci[1], sex_adj_ci[1])
data_range = data_hi - data_lo
ax.set_xlim(data_lo - 0.08 * data_range, data_hi + 0.08 * data_range)
ax.text(0.03, 0.03, f"n = {len(confound):,}\nBMI coefficient = {bmi_coef:.4f} (p = {bmi_p:.2g})",
        transform=ax.transAxes, va="bottom", fontsize=7.5)
fig.suptitle(
    "Figure S5. Adjusting for BMI does not shrink the sex coefficient\n"
    "(BMI-available subsample only, underpowered vs. Fig. 4b)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(os.path.join(FIG_DIR, "figS6_sex_bmi_confound.pdf"))
fig.savefig(os.path.join(FIG_DIR, "figS6_sex_bmi_confound.png"))
plt.close(fig)

STATS["sex_bmi_confound"] = {
    "n": int(len(confound)),
    "sex_unadj_coef": sex_unadj_coef, "sex_unadj_p": sex_unadj_p,
    "sex_unadj_ci_lo": float(sex_unadj_ci[0]), "sex_unadj_ci_hi": float(sex_unadj_ci[1]),
    "sex_adj_coef": sex_adj_coef, "sex_adj_p": sex_adj_p,
    "sex_adj_ci_lo": float(sex_adj_ci[0]), "sex_adj_ci_hi": float(sex_adj_ci[1]),
    "bmi_coef": bmi_coef, "bmi_p": bmi_p,
}

# =============================================================================
# Graphical abstract: headline "how much does the heart show up in the EEG"
# summary, gathering the paper's core numbers into one stat-card panel for
# readers who only look at one figure.
# =============================================================================
cards = [
    (f"{STATS['cfa']['r2_excl_qrs_mean']*100:.0f}%",
     "of HEP variance still explained\nby the ECG, outside the QRS\nexclusion window", PALETTE[1]),
    (f"{STATS['ica']['component_variance_fraction_median']*100:.0f}%",
     "median share of HEP variance\ncarried by the ECG-flagged\nICA component", PALETTE[0]),
    (f"{STATS['ica']['hep_pct_drop_median']*100:.0f}%",
     "median drop in HEP variance\nafter removing that component\n(realised cleaning effect)", PALETTE[2]),
    (f"{STATS['cfa']['n_patients']:,}",
     "patients analysed — the\nlargest cohort in which CFA's\nHEP contribution is quantified", PALETTE[6]),
]

fig, axes = plt.subplots(1, 4, figsize=(11, 2.6))
for ax, (big, label, color) in zip(axes, cards):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.62, big, ha="center", va="center", fontsize=30, fontweight="bold", color=color)
    ax.text(0.5, 0.22, label, ha="center", va="center", fontsize=8.3, color="#222222", linespacing=1.4)
    ax.axhline(0.42, xmin=0.18, xmax=0.82, color=color, linewidth=2.4)
fig.suptitle(
    "Graphical Abstract. The heart is still in the EEG: cardiac-field contribution to the HEP, "
    f"this cohort (n = {STATS['cfa']['n_patients']:,} patients)",
    fontsize=10.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.86])
fig.savefig(os.path.join(FIG_DIR, "fig0_headline.pdf"))
fig.savefig(os.path.join(FIG_DIR, "fig0_headline.png"))
plt.close(fig)

with open(os.path.join(HERE, "paper_stats.json")) as f:
    ALL_STATS = json.load(f)  # keep keys written by the other figure scripts
ALL_STATS.update(STATS)
with open(os.path.join(HERE, "paper_stats.json"), "w") as f:
    json.dump(ALL_STATS, f, indent=2)

print("Figures written to", FIG_DIR)
for fn in sorted(os.listdir(FIG_DIR)):
    print(" ", fn, os.path.getsize(os.path.join(FIG_DIR, fn)))
print("Stats written to", os.path.join(HERE, "paper_stats.json"))
