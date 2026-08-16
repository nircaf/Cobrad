"""
Stats recompute on the full matched cohort.

An earlier pass of this script applied a post-hoc amplitude exclusion because
the raw RMS distribution had an implausible heavy tail (max ~1.1e7 uV). Root
cause was found and fixed upstream in analysis_age_by_stage.py: the per-patient
channel average was being computed over ALL channels in the cache tuple,
including non-EEG PSG channels (ECG, CHEST, ABDOMINAL, SaO2, Pleth, etc.)
whose amplitude scales are orders of magnitude larger than scalp EEG. With
metrics now restricted to true 10-20 EEG derivations only, RMS values are in
the expected sub-microvolt-to-low-microvolt range (max ~6.8 uV) and no
amplitude-based exclusion is needed. This script now just recomputes stats on
the full matched cohort and writes analysis_results_clean.pkl, keeping the
same output shape so make_figures.py / make_pdf.py need no changes.

Run: source venv/bin/activate && python3 Paper1/recompute_stats.py
"""
import os
import pickle
import numpy as np
from scipy import stats

REPO = "/storage/pblab_shared_data2/Nir/Cobrad"
OUT_DIR = os.path.join(REPO, "Paper1")
STAGES = ["light_sleep", "N3", "R"]

with open(os.path.join(OUT_DIR, "analysis_results.pkl"), "rb") as f:
    R = pickle.load(f)

df = R["df"]
per_stage_waveform = R["per_stage_waveform"]
n_excluded = 0
print(f"Full matched cohort, n={len(df)}. No amplitude-based exclusion needed "
      f"(EEG-only channel fix resolved the earlier heavy-tailed distribution).")


def stage_regressions(metric_col):
    out = {}
    for stage in STAGES:
        x = df["age"].values
        y = df[f"{metric_col}_{stage}"].values
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        n = len(x)
        r, p_r = stats.pearsonr(x, y)
        rho, p_rho = stats.spearmanr(x, y)
        lr = stats.linregress(x, y)
        tval = stats.t.ppf(0.975, df=n - 2)
        slope_ci = (lr.slope - tval * lr.stderr, lr.slope + tval * lr.stderr)
        out[stage] = dict(n=n, r=r, p_r=p_r, rho=rho, p_rho=p_rho,
                           slope=lr.slope, slope_ci=slope_ci, intercept=lr.intercept,
                           p_slope=lr.pvalue)
    return out


rms_age_stats = stage_regressions("rms")
trough_age_stats = stage_regressions("trough")

print("\n=== Per-stage age correlation: RMS ===")
for s, v in rms_age_stats.items():
    print(f"  {s}: n={v['n']} r={v['r']:.3f} (p={v['p_r']:.4g}) rho={v['rho']:.3f} "
          f"(p={v['p_rho']:.4g}) slope={v['slope']:.4f} uV/yr [{v['slope_ci'][0]:.4f},{v['slope_ci'][1]:.4f}]")

print("\n=== Per-stage age correlation: Trough ===")
for s, v in trough_age_stats.items():
    print(f"  {s}: n={v['n']} r={v['r']:.3f} (p={v['p_r']:.4g}) rho={v['rho']:.3f} "
          f"(p={v['p_rho']:.4g}) slope={v['slope']:.4f} uV/yr [{v['slope_ci'][0]:.4f},{v['slope_ci'][1]:.4f}]")

contrasts = [("R", "N3"), ("light_sleep", "N3"), ("R", "light_sleep")]
contrast_stats = {}
for metric in ["rms", "trough"]:
    for a, b in contrasts:
        key = f"{metric}_{a}_minus_{b}"
        diff = df[f"{metric}_{a}"].values - df[f"{metric}_{b}"].values
        x = df["age"].values
        mask = np.isfinite(x) & np.isfinite(diff)
        r, p_r = stats.pearsonr(x[mask], diff[mask])
        rho, p_rho = stats.spearmanr(x[mask], diff[mask])
        lr = stats.linregress(x[mask], diff[mask])
        contrast_stats[key] = dict(n=int(mask.sum()), r=r, p_r=p_r, rho=rho, p_rho=p_rho,
                                    slope=lr.slope, p_slope=lr.pvalue)

print("\n=== Stage-contrast vs age ===")
for k, v in contrast_stats.items():
    print(f"  {k}: n={v['n']} r={v['r']:.3f} (p={v['p_r']:.4g}) slope={v['slope']:.4f} uV/yr")

rng = np.random.default_rng(42)
n_perm = 2000


def compute_slopes(metric_col):
    ages = df["age"].values
    slopes = {}
    for stage in STAGES:
        y = df[f"{metric_col}_{stage}"].values
        slopes[stage] = stats.linregress(ages, y).slope
    return slopes


def permutation_slope_diff_test(metric_col, pair):
    a, b = pair
    ages = df["age"].values
    obs_slopes = compute_slopes(metric_col)
    obs_diff = obs_slopes[a] - obs_slopes[b]
    mat = df[[f"{metric_col}_{s}" for s in STAGES]].values.copy()
    stage_idx = {s: i for i, s in enumerate(STAGES)}
    null_diffs = np.empty(n_perm)
    for i in range(n_perm):
        perm_mat = mat.copy()
        for row in range(perm_mat.shape[0]):
            perm = rng.permutation(3)
            perm_mat[row] = perm_mat[row][perm]
        y_a = perm_mat[:, stage_idx[a]]
        y_b = perm_mat[:, stage_idx[b]]
        slope_a = stats.linregress(ages, y_a).slope
        slope_b = stats.linregress(ages, y_b).slope
        null_diffs[i] = slope_a - slope_b
    p_perm = float(np.mean(np.abs(null_diffs) >= np.abs(obs_diff)))
    return dict(obs_diff=obs_diff, p_perm=p_perm, null_diffs=null_diffs)


interaction_results = {}
for metric_col in ["rms", "trough"]:
    for pair in [("R", "N3"), ("R", "light_sleep"), ("light_sleep", "N3")]:
        key = f"{metric_col}_{pair[0]}_vs_{pair[1]}"
        interaction_results[key] = permutation_slope_diff_test(metric_col, pair)

print("\n=== Age x stage interaction (permutation test, n_perm=2000) ===")
for k, v in interaction_results.items():
    print(f"  {k}: observed slope diff={v['obs_diff']:.4f} uV/yr, perm p={v['p_perm']:.4g}")

cohort_summary = {
    "n": len(df), "n_excluded_artifact": n_excluded, "n_raw": len(df),
    "artifact_threshold_uv": None,
    "age_mean": df["age"].mean(), "age_median": df["age"].median(),
    "age_sd": df["age"].std(), "age_min": df["age"].min(), "age_max": df["age"].max(),
    "n_male": int((df["sex"] == "Male").sum()),
    "n_female": int((df["sex"] == "Female").sum()),
    "n_unknown_sex": int((~df["sex"].isin(["Male", "Female"])).sum()),
}
print("\n=== Cohort summary ===")
for k, v in cohort_summary.items():
    print(f"  {k}: {v}")

results = dict(
    df=df, times_ref=R["times_ref"], per_stage_waveform=per_stage_waveform,
    rms_age_stats=rms_age_stats, trough_age_stats=trough_age_stats,
    contrast_stats=contrast_stats, interaction_results=interaction_results,
    cohort_summary=cohort_summary,
)
with open(os.path.join(OUT_DIR, "analysis_results_clean.pkl"), "wb") as f:
    pickle.dump(results, f)
print("\nSaved results to", os.path.join(OUT_DIR, "analysis_results_clean.pkl"))
