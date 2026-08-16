"""
Figure 3 -- Mixed-effects model of HEP amplitude.

HEP_amplitude ~ NeuroGroup * Stage + Region + Age + Sex + HR + CFA + (1|Patient)

fit via statsmodels.formula.api.mixedlm on the per-(patient,stage,electrode)
long-format table (build_hep_diagnosis_dataset.py). Diagnosis is represented
by the same 3-level neuro_group used in Fig 2 (Neurological / Non-neurological
/ Unknown reference) rather than the 15 non-exclusive diagnosis categories --
a categorical fixed effect needs mutually-exclusive levels, and 15 non-
exclusive binary indicators would blow up collinearity/interpretability for a
single coefficient plot; this simplification is documented here and in the
manuscript Methods. Electrode is binned into 5 scalp regions (frontal,
central, parietal, temporal, occipital) per the task's electrode-simplification
option, again to keep the fixed-effect count legible in one forest plot.
Full per-electrode summary statistics are exported to a supplementary table.

Reads the cached long-format dataset (Paper1/hep_diagnosis_long_df.pkl) --
no raw data reload.

Run: source venv/bin/activate && python3 Paper1/build_fig3_mixedmodel.py
"""
from __future__ import annotations

import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

REPO = "/storage/pblab_shared_data2/Nir/Cobrad"
OUT_DIR = os.path.join(REPO, "Paper1")
FIG_DIR = os.path.join(OUT_DIR, "figures")
LONG_DF_PKL = os.path.join(OUT_DIR, "hep_diagnosis_long_df.pkl")
FIG3_JSON = os.path.join(OUT_DIR, "fig3_mixedmodel_results.json")
FIG3_ELECTRODE_CSV = os.path.join(OUT_DIR, "fig3_electrode_supplementary_table.csv")
FIG_PDF = os.path.join(FIG_DIR, "fig3_mixedmodel.pdf")

TERM_COLOR_NEURO = "#D55E00"
TERM_COLOR_STAGE = "#0072B2"
TERM_COLOR_REGION = "#009E73"
TERM_COLOR_INTERACTION = "#CC79A7"
TERM_COLOR_COVAR = "#555555"


def assign_neuro_group(row):
    if row["is_neuro"]:
        return "Neurological"
    if row["is_nonneuro"]:
        return "Non-neurological"
    return "Unknown"


def term_color(name):
    if "neuro_group" in name and ":" in name:
        return TERM_COLOR_INTERACTION
    if "neuro_group" in name:
        return TERM_COLOR_NEURO
    if "stage" in name:
        return TERM_COLOR_STAGE
    if "region" in name:
        return TERM_COLOR_REGION
    return TERM_COLOR_COVAR


def main():
    with open(LONG_DF_PKL, "rb") as f:
        cached = pickle.load(f)
    long_df = cached["long_df"].copy()

    long_df["neuro_group"] = long_df.apply(assign_neuro_group, axis=1)
    model_df = long_df[long_df["sex"].isin(["Male", "Female"])].copy()
    model_df = model_df.dropna(subset=["hep_amplitude_uv", "age", "hr_bpm", "cfa_r2", "region", "stage", "neuro_group"])
    model_df["sex_bin"] = (model_df["sex"] == "Male").astype(int)
    model_df["age_z"] = (model_df["age"] - model_df["age"].mean()) / model_df["age"].std()
    model_df["hr_z"] = (model_df["hr_bpm"] - model_df["hr_bpm"].mean()) / model_df["hr_bpm"].std()
    model_df["cfa_z"] = (model_df["cfa_r2"] - model_df["cfa_r2"].mean()) / model_df["cfa_r2"].std()

    n_obs = len(model_df)
    n_patients = model_df["patient_id"].nunique()
    print(f"Model dataset: {n_obs} observations, {n_patients} unique patients")
    print(model_df.groupby(["neuro_group", "stage"], observed=True)["patient_id"].nunique().unstack())

    formula = (
        "hep_amplitude_uv ~ C(neuro_group, Treatment('Unknown')) * C(stage, Treatment('light_sleep')) "
        "+ C(region, Treatment('central')) + age_z + sex_bin + hr_z + cfa_z"
    )
    print(f"\nFormula: {formula}")
    print("Fitting MixedLM (random intercept per patient)...")
    # ponytail: statsmodels' lbfgs optimizer reliably converges MixedLM to a
    # degenerate boundary solution (Group Var -> 0, Log-Likelihood -> inf) on
    # this dataset -- verified on a 1500-patient subsample where lbfgs alone
    # diverges while bfgs/cg/powell agree to 5 decimal places. Use bfgs, with
    # cg as a fallback if it fails to converge.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(formula, data=model_df, groups=model_df["patient_id"])
        result = model.fit(reml=True, method="bfgs", maxiter=300)
        if not result.converged or not np.isfinite(result.llf):
            print("  bfgs did not converge cleanly, retrying with cg...")
            result = model.fit(reml=True, method="cg", maxiter=300)
    print(result.summary())

    # ---- Coefficient table ----
    params = result.params.drop("Group Var", errors="ignore")
    conf = result.conf_int()
    coef_rows = []
    for name in params.index:
        est = float(params[name])
        lo = float(conf.loc[name, 0])
        hi = float(conf.loc[name, 1])
        pval = float(result.pvalues[name])
        coef_rows.append({"term": name, "estimate": est, "ci_lo": lo, "ci_hi": hi, "p_value": pval})
    coef_df = pd.DataFrame(coef_rows)
    coef_df = coef_df[coef_df["term"] != "Intercept"].copy()
    coef_df["abs_estimate"] = coef_df["estimate"].abs()
    coef_df = coef_df.sort_values("abs_estimate", ascending=True).reset_index(drop=True)

    def pretty_term(t):
        t = t.replace(":", " \x00 ")  # placeholder for interaction separator, applied last
        t = t.replace("C(neuro_group, Treatment('Unknown'))[T.", "Diagnosis=")
        t = t.replace("C(stage, Treatment('light_sleep'))[T.", "Stage=")
        t = t.replace("C(region, Treatment('central'))[T.", "Region=")
        t = t.replace("]", "")
        t = t.replace("age_z", "Age (z)").replace("sex_bin", "Sex (Male=1)")
        t = t.replace("hr_z", "Heart rate (z)").replace("cfa_z", "CFA contamination (z)")
        t = t.replace("\x00", "x")
        return t
    coef_df["label"] = coef_df["term"].map(pretty_term)

    # ---- Model fit summary ----
    ll = float(result.llf)
    aic = float(-2 * ll + 2 * (len(params) + 1))
    resid_var = float(result.scale)
    group_var = float(result.cov_re.iloc[0, 0])
    fitted = result.fittedvalues
    var_fitted = float(np.var(fitted))
    var_resid = float(np.var(model_df["hep_amplitude_uv"] - fitted))
    total_var = var_fitted + group_var + resid_var
    marginal_r2 = var_fitted / total_var if total_var > 0 else np.nan
    conditional_r2 = (var_fitted + group_var) / total_var if total_var > 0 else np.nan

    results = {
        "formula": formula,
        "n_obs": int(n_obs), "n_patients": int(n_patients),
        "log_likelihood": ll, "aic": aic,
        "group_var": group_var, "resid_var": resid_var,
        "marginal_r2_approx": marginal_r2, "conditional_r2_approx": conditional_r2,
        "coefficients": coef_df.to_dict(orient="records"),
        "n_by_group_stage": {
            f"{g}|{s}": int(model_df.loc[(model_df["neuro_group"] == g) & (model_df["stage"] == s), "patient_id"].nunique())
            for g in ["Unknown", "Non-neurological", "Neurological"] for s in ["light_sleep", "N3", "R"]
        },
    }
    with open(FIG3_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {FIG3_JSON}")
    print(f"AIC={aic:.1f}  logLik={ll:.1f}  marginal R2~{marginal_r2:.3f}  conditional R2~{conditional_r2:.3f}")

    # ---- Supplementary: per-electrode descriptive summary (full 24-electrode detail) ----
    elec_rows = []
    for (electrode, stage), grp in long_df.groupby(["electrode", "stage"], observed=True):
        elec_rows.append({
            "electrode": electrode, "stage": stage, "n_patients": grp["patient_id"].nunique(),
            "median_uv": float(grp["hep_amplitude_uv"].median()),
            "mean_uv": float(grp["hep_amplitude_uv"].mean()),
            "sd_uv": float(grp["hep_amplitude_uv"].std()),
        })
    pd.DataFrame(elec_rows).to_csv(FIG3_ELECTRODE_CSV, index=False)
    print(f"Saved {FIG3_ELECTRODE_CSV}")

    # ---- Render forest/coefficient plot ----
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    y = np.arange(len(coef_df))
    colors = [term_color(t) for t in coef_df["term"]]
    ax.errorbar(coef_df["estimate"], y,
                xerr=[coef_df["estimate"] - coef_df["ci_lo"], coef_df["ci_hi"] - coef_df["estimate"]],
                fmt="o", markersize=5, capsize=3, elinewidth=1.3, color="black", ecolor="grey", zorder=2)
    for yi, (est, color) in enumerate(zip(coef_df["estimate"], colors)):
        ax.scatter([est], [yi], color=color, s=45, zorder=3)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(coef_df["label"], fontsize=9)
    ax.set_xlabel("Fixed-effect estimate (µV) ± 95% CI", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        f"Mixed-effects model of HEP amplitude\n"
        f"N={n_obs:,} observations, {n_patients:,} patients | AIC={aic:.0f}",
        fontsize=10.5)
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TERM_COLOR_NEURO, markersize=8, label="Diagnosis"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TERM_COLOR_INTERACTION, markersize=8, label="Diagnosis x Stage"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TERM_COLOR_STAGE, markersize=8, label="Stage"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TERM_COLOR_REGION, markersize=8, label="Region"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TERM_COLOR_COVAR, markersize=8, label="Covariate"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_PDF, format="pdf")
    fig.savefig(FIG_PDF.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)
    print(f"Saved {FIG_PDF}")


if __name__ == "__main__":
    main()
