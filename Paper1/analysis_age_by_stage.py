"""
Age x sleep-stage HEP-magnitude analysis (Harvard_Electroencephalography cohort).

Loads per-patient HEP waveforms for light_sleep / N3 / R from the existing
on-disk caches (via 6_hep_group_comparison.get_group_individuals, mock
streamlit import pattern copied from 17_generate_hep_cache.py), joins age/sex
from the EHR demographics parquet (join logic copied from dashboard 16's
_pid_to_bdsp_ehr / _patient_demographics), computes RMS/trough amplitude in
the post-QRS window per patient x stage, and runs:
  - per-stage age correlation (Pearson/Spearman/OLS)
  - stage-contrast (REM-N3 etc.) vs age correlation
  - permutation test for age x stage slope interaction

Outputs: cache/results pickle + figures under Paper1/figures/.
Run with the project venv: source venv/bin/activate && python3 Paper1/analysis_age_by_stage.py
"""
from __future__ import annotations

import os
import re
import sys
import types
import pickle
import importlib.util
import time

import numpy as np
import pandas as pd
from scipy import stats

REPO = "/storage/pblab_shared_data2/Nir/Cobrad"
BASE_PATH = os.path.join(REPO, "pickles_sleep_stage")
GROUP = "Harvard_Electroencephalography"
STAGES = ["light_sleep", "N3", "R"]
STAGE_LABELS = {"light_sleep": "Light Sleep", "N3": "N3 (SWS)", "R": "REM"}
# ch_names in the raw cache mixes EEG derivations with non-EEG PSG channels
# (CHIN/ECG/SNORE/NPT/C-FLOW/CHEST/ABDOMINAL/Flexor/Extensor/LAT/RAT/SaO2/
# Pleth/C PRESS/LEAK), whose amplitude scales are orders of magnitude larger
# than scalp EEG. Restrict every waveform/metric computation to true 10-20
# EEG derivations only (list copied from dashboard 16's MONTAGE_1020_CHANNEL_ORDER).
EEG_CHANNELS = {
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "T7", "C3", "Cz", "C4", "T4", "T8",
    "T5", "P7", "P3", "Pz", "P4", "T6", "P8",
    "O1", "Oz", "O2",
}
EHR_DEMOGRAPHICS_CACHE = os.path.join(REPO, "cache", "ehr_demographics.parquet")
POST_QRS_WINDOW = (0.05, 0.40)  # exclusion end -> epoch end
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00", "#000000"]
OUT_DIR = os.path.join(REPO, "Paper1")
FIG_DIR = os.path.join(OUT_DIR, "figures")
RESULTS_PKL = os.path.join(OUT_DIR, "analysis_results.pkl")


# ---------------------------------------------------------------------------
# Mock streamlit + module loader (copied pattern from 17_generate_hep_cache.py)
# ---------------------------------------------------------------------------
def _make_mock_streamlit():
    class _NoOp:
        def __init__(self, *a, **kw):
            pass
        def __call__(self, *a, **kw):
            return _NoOp()
        def progress(self, *a, **kw):
            pass
        def empty(self, *a, **kw):
            pass
        def text(self, *a, **kw):
            pass
        def warning(self, *a, **kw):
            pass
        def error(self, *a, **kw):
            pass
        def info(self, *a, **kw):
            pass
        def __getattr__(self, name):
            return _NoOp()

    def _cache_data(func=None, **kwargs):
        if func is not None:
            return func
        return lambda f: f

    def _cache_resource(func=None, **kwargs):
        if func is not None:
            return func
        return lambda f: f

    st = types.ModuleType("streamlit")
    st.cache_data = _cache_data
    st.cache_resource = _cache_resource
    st.session_state = {}
    st.progress = lambda value=0, text=None, **kw: _NoOp()
    st.empty = lambda *a, **kw: _NoOp()
    st.warning = lambda msg, *a, **kw: print(f"[WARN] {msg}", file=sys.stderr)
    st.error = lambda msg, *a, **kw: print(f"[ERROR] {msg}", file=sys.stderr)
    st.info = lambda msg, *a, **kw: None
    st.write = lambda *a, **kw: None
    st.title = lambda *a, **kw: None
    st.header = lambda *a, **kw: None
    st.subheader = lambda *a, **kw: None
    st.markdown = lambda *a, **kw: None
    st.text = lambda *a, **kw: None
    st.pyplot = lambda *a, **kw: None
    st.sidebar = _NoOp()
    st.columns = lambda *a, **kw: [_NoOp() for _ in range(a[0] if a else 2)]
    st.tabs = lambda labels: [_NoOp() for _ in labels]
    st.expander = lambda *a, **kw: _NoOp()
    st.spinner = lambda *a, **kw: _NoOp()
    st.set_page_config = lambda *a, **kw: None
    st.stop = lambda: None
    st.experimental_rerun = lambda: None
    st.rerun = lambda: None
    st.__enter__ = lambda s: s
    st.__exit__ = lambda s, *a: False
    components = types.ModuleType("streamlit.components")
    components.v1 = _NoOp()
    st.components = components
    return st


def load_hep_module():
    mock_st = _make_mock_streamlit()
    sys.modules["streamlit"] = mock_st
    for submod in ["streamlit.components", "streamlit.components.v1"]:
        if submod not in sys.modules:
            sys.modules[submod] = types.ModuleType(submod)
    hep_path = os.path.join(REPO, "6_hep_group_comparison.py")
    spec = importlib.util.spec_from_file_location("hep_module", hep_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hep_module"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# ID / demographics helpers (copied logic from 16_diagnosis_sleep_stage_comparison_dashboard.py)
# ---------------------------------------------------------------------------
def _extract_cobrad_id(raw_id):
    if raw_id is None:
        return None
    m = re.search(r"(\d{3}-\d{3})", str(raw_id))
    return m.group(1) if m else None


def canonical_patient_id(patient_id):
    value = str(patient_id)
    m = re.search(r"(I\d{13})", value)
    if m:
        return m.group(1)
    return _extract_cobrad_id(value) or value


def pid_to_bdsp_ehr(pid_str):
    m = re.search(r"I\d{4}(\d{9})", str(pid_str))
    return int(m.group(1)) if m else None


def load_ehr_demographics():
    df = pd.read_parquet(EHR_DEMOGRAPHICS_CACHE)
    return df


# ---------------------------------------------------------------------------
# Metrics (style of extract_metric in dashboard 16, but new: RMS/trough over
# post-QRS window, averaged across all available EEG channels)
# ---------------------------------------------------------------------------
def compute_patient_metrics(individual, window=POST_QRS_WINDOW):
    """Return dict(rms_uv, trough_uv, mean_uv, n_channels) for one patient tuple,
    or None if unusable. hep_data is volts, n_channels x n_times."""
    patient_id, hep_data, times, ch_names, rpeaks, ecg_hep_data, ecg_ch_names, log_msg, flipped = individual[:9]
    if hep_data is None or times is None or ch_names is None:
        return None
    hep_data = np.asarray(hep_data, dtype=float)
    times_arr = np.asarray(times, dtype=float)
    if hep_data.ndim != 2 or hep_data.shape[0] == 0:
        return None
    eeg_idx = [i for i, ch in enumerate(ch_names) if ch in EEG_CHANNELS]
    if not eeg_idx:
        return None
    hep_data = hep_data[eeg_idx]
    t_mask = (times_arr >= window[0]) & (times_arr <= window[1])
    if not t_mask.any():
        return None
    win_uv = hep_data[:, t_mask] * 1e6  # V -> uV, shape (n_ch, n_t)
    if not np.isfinite(win_uv).any():
        return None
    # channel-average waveform first (grand per-patient waveform over the window),
    # then RMS/trough on that averaged waveform -- avoids channels with opposite
    # polarity cancelling out is a concern, but this mirrors extract_metric's
    # "average over selected channels" convention used for scalar summaries.
    chan_avg_waveform = np.nanmean(win_uv, axis=0)
    if not np.isfinite(chan_avg_waveform).any():
        return None
    rms_uv = float(np.sqrt(np.nanmean(chan_avg_waveform ** 2)))
    trough_uv = float(np.nanmin(chan_avg_waveform))
    mean_uv = float(np.nanmean(chan_avg_waveform))
    return {"rms_uv": rms_uv, "trough_uv": trough_uv, "mean_uv": mean_uv,
            "n_channels": int(hep_data.shape[0])}


def full_waveform_channel_avg(individual):
    """Channel-averaged waveform across the FULL epoch (for Fig 1), in uV."""
    patient_id, hep_data, times, ch_names, rpeaks, ecg_hep_data, ecg_ch_names, log_msg, flipped = individual[:9]
    if hep_data is None or times is None or ch_names is None:
        return None, None
    hep_data = np.asarray(hep_data, dtype=float)
    if hep_data.ndim != 2 or hep_data.shape[0] == 0:
        return None, None
    eeg_idx = [i for i, ch in enumerate(ch_names) if ch in EEG_CHANNELS]
    if not eeg_idx:
        return None, None
    return np.asarray(times, dtype=float), np.nanmean(hep_data[eeg_idx], axis=0) * 1e6


# ---------------------------------------------------------------------------
# Main data assembly
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("Loading hep_module (mock streamlit)...")
    hep_mod = load_hep_module()

    print("Loading EHR demographics parquet...")
    ehr_demo_df = load_ehr_demographics()
    ehr_age_map = dict(zip(ehr_demo_df["patient_id"], ehr_demo_df["age"]))
    ehr_sex_map = dict(zip(ehr_demo_df["patient_id"], ehr_demo_df["sex"]))
    print(f"  EHR demographics rows: {len(ehr_demo_df)}, unique patient_id: {ehr_demo_df['patient_id'].nunique()}")

    # per-stage: canonical_id -> (metrics dict, full waveform)
    per_stage_metrics = {s: {} for s in STAGES}
    per_stage_waveform = {s: {} for s in STAGES}
    times_ref = None

    for stage in STAGES:
        print(f"Loading {GROUP} / {stage} individuals from cache...")
        individuals = hep_mod.get_group_individuals(
            GROUP, stage, BASE_PATH,
            test_run=False, recompute_cache=False, apply_ica=False,
            parallel_workers=hep_mod.DEFAULT_PATIENT_WORKERS,
            min_eeg_channels=None,
        )
        print(f"  {stage}: {len(individuals)} raw individuals loaded")
        n_ok = 0
        for ind in individuals:
            if ind is None or len(ind) == 0:
                continue
            cid = canonical_patient_id(ind[0])
            m = compute_patient_metrics(ind)
            if m is None:
                continue
            # keep first valid occurrence per canonical id (dedupe within stage)
            if cid not in per_stage_metrics[stage]:
                per_stage_metrics[stage][cid] = m
                t_arr, wf = full_waveform_channel_avg(ind)
                if wf is not None:
                    per_stage_waveform[stage][cid] = wf
                    if times_ref is None:
                        times_ref = t_arr
                n_ok += 1
        print(f"  {stage}: {n_ok} usable patients (unique canonical ids)")

    # matched cohort: present + usable in all 3 stages
    common_ids = set(per_stage_metrics[STAGES[0]])
    for s in STAGES[1:]:
        common_ids &= set(per_stage_metrics[s])
    print(f"Patients present in all 3 stages: {len(common_ids)}")

    # attach age/sex
    rows = []
    for cid in sorted(common_ids):
        bdsp = pid_to_bdsp_ehr(cid)
        age = ehr_age_map.get(bdsp, np.nan) if bdsp is not None else np.nan
        sex = ehr_sex_map.get(bdsp) if bdsp is not None else None
        if pd.isna(age):
            continue
        row = {"patient_id": cid, "bdsp_id": bdsp, "age": float(age), "sex": sex or "Unknown"}
        for stage in STAGES:
            m = per_stage_metrics[stage][cid]
            row[f"rms_{stage}"] = m["rms_uv"]
            row[f"trough_{stage}"] = m["trough_uv"]
            row[f"nch_{stage}"] = m["n_channels"]
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Matched cohort with valid age: n={len(df)}")
    if len(df) < 20:
        print("!!! Cohort too small -- something is likely broken. Inspect before proceeding.")
    print(df[["age"] + [f"rms_{s}" for s in STAGES]].describe())

    # sanity: raw sample
    print("Sample rows:")
    print(df.head(5).to_string())

    df.to_pickle(os.path.join(OUT_DIR, "matched_cohort.pkl"))

    # -----------------------------------------------------------------
    # Long format for interaction test
    # -----------------------------------------------------------------
    long_rows = []
    for _, r in df.iterrows():
        for stage in STAGES:
            long_rows.append({
                "patient_id": r["patient_id"], "age": r["age"], "sex": r["sex"],
                "stage": stage, "rms_uv": r[f"rms_{stage}"], "trough_uv": r[f"trough_{stage}"],
            })
    long_df = pd.DataFrame(long_rows)

    # -----------------------------------------------------------------
    # Per-stage age correlations
    # -----------------------------------------------------------------
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
            # 95% CI on slope via t-distribution
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

    # -----------------------------------------------------------------
    # Stage contrasts vs age (does the gap widen/narrow with age)
    # -----------------------------------------------------------------
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
            contrast_stats[key] = dict(n=mask.sum(), r=r, p_r=p_r, rho=rho, p_rho=p_rho,
                                        slope=lr.slope, p_slope=lr.pvalue)

    print("\n=== Stage-contrast vs age ===")
    for k, v in contrast_stats.items():
        print(f"  {k}: n={v['n']} r={v['r']:.3f} (p={v['p_r']:.4g}) slope={v['slope']:.4f} uV/yr")

    # -----------------------------------------------------------------
    # Age x stage interaction: permutation test on slope differences
    # Shuffle stage labels WITHIN each patient (permutes which stage's
    # metric value goes with which stage label), recompute per-stage OLS
    # slope-vs-age, and build a null distribution for slope differences
    # between stage pairs (REM-N3, REM-light, light-N3).
    # -----------------------------------------------------------------
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
        # matrix n_patients x n_stages of metric values
        mat = df[[f"{metric_col}_{s}" for s in STAGES]].values.copy()  # columns ordered as STAGES
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
            res = permutation_slope_diff_test(metric_col, pair)
            interaction_results[key] = res

    print("\n=== Age x stage interaction (permutation test, n_perm=2000) ===")
    for k, v in interaction_results.items():
        print(f"  {k}: observed slope diff={v['obs_diff']:.4f} uV/yr, perm p={v['p_perm']:.4g}")

    # -----------------------------------------------------------------
    # Cohort table
    # -----------------------------------------------------------------
    cohort_summary = {
        "n": len(df),
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
        df=df, long_df=long_df, times_ref=times_ref,
        per_stage_waveform=per_stage_waveform,
        rms_age_stats=rms_age_stats, trough_age_stats=trough_age_stats,
        contrast_stats=contrast_stats, interaction_results=interaction_results,
        cohort_summary=cohort_summary,
    )
    with open(RESULTS_PKL, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved results to {RESULTS_PKL}")
    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
