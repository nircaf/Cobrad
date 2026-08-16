"""
HEP-diagnosis long-format dataset builder.

Builds one cached DataFrame with a row per (patient_id, stage, electrode):
HEP amplitude (uV, mean over AMP_WINDOW), plus per-patient covariates
(age, sex, HR, CFA-contamination, diagnosis categories, neuro/non-neuro
flags). No fabricated numbers -- everything is read from the existing
pickle cache (6_hep_group_comparison.get_group_individuals via mod16) and
from Paper CFA/{demographics,cfa}_combined.parquet.

Reuses (does not reimplement): mod16.load_patient_data, mod16._patient_hep_trace,
mod16._canonical_patient_id, mod16._patient_demographics, mod16._DIAG_CATEGORIES
(for cross-check only -- category assignment is read pre-computed from
demographics_combined.parquet, built by Paper CFA/build_dataset.py using the
identical category functions).

Run: source venv/bin/activate && python3 Paper1/build_hep_diagnosis_dataset.py
"""
from __future__ import annotations

import os
import pickle
import sys
import time
import types
import importlib.util

import numpy as np
import pandas as pd

REPO = "/storage/pblab_shared_data2/Nir/Cobrad"
OUT_DIR = os.path.join(REPO, "Paper1")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

LONG_DF_PKL = os.path.join(OUT_DIR, "hep_diagnosis_long_df.pkl")
PATIENT_META_PKL = os.path.join(OUT_DIR, "hep_diagnosis_patient_meta.pkl")

GROUP = "Harvard_Electroencephalography"
STAGES = ["light_sleep", "N3", "R"]
AMP_WINDOW = (0.15, 0.5)  # standard HEP amplitude window used throughout this repo (Paper1/build_diagnosis_alignment_analysis.py)

NEURO_CATEGORIES = {"Cognitive Impairment / Dementia", "Stroke / Cerebrovascular"}

REGION_MAP = {
    "Fp1": "frontal", "Fp2": "frontal", "F7": "frontal", "F3": "frontal",
    "Fz": "frontal", "F4": "frontal", "F8": "frontal",
    "C3": "central", "Cz": "central", "C4": "central",
    "T3": "temporal", "T7": "temporal", "T4": "temporal", "T8": "temporal",
    "T5": "temporal", "P7": "temporal", "T6": "temporal", "P8": "temporal",
    "P3": "parietal", "Pz": "parietal", "P4": "parietal",
    "O1": "occipital", "Oz": "occipital", "O2": "occipital",
}


# ---------------------------------------------------------------------------
# Mock streamlit + module loaders (copied pattern from build_diagnosis_alignment_analysis.py)
# ---------------------------------------------------------------------------
def _make_mock_streamlit():
    class _NoOp:
        def __init__(self, *a, **kw):
            pass
        def __call__(self, *a, **kw):
            return _NoOp()
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
    st.warning = lambda msg, *a, **kw: print(f"[WARN] {msg}", file=sys.stderr)
    st.error = lambda msg, *a, **kw: print(f"[ERROR] {msg}", file=sys.stderr)
    st.info = lambda msg, *a, **kw: None
    for name in ("write", "title", "header", "subheader", "markdown", "text", "pyplot",
                 "set_page_config", "stop", "experimental_rerun", "rerun"):
        setattr(st, name, lambda *a, **kw: None)
    st.sidebar = _NoOp()
    st.columns = lambda *a, **kw: [_NoOp() for _ in range(a[0] if a else 2)]
    st.tabs = lambda labels: [_NoOp() for _ in labels]
    st.expander = lambda *a, **kw: _NoOp()
    st.spinner = lambda *a, **kw: _NoOp()
    st.progress = lambda *a, **kw: _NoOp()
    st.empty = lambda *a, **kw: _NoOp()
    st.__enter__ = lambda s: s
    st.__exit__ = lambda s, *a: False
    components = types.ModuleType("streamlit.components")
    components.v1 = _NoOp()
    st.components = components
    return st


def load_hep_module():
    hep_path = os.path.join(REPO, "6_hep_group_comparison.py")
    spec = importlib.util.spec_from_file_location("hep_module", hep_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hep_module"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_mod16():
    dash_path = os.path.join(REPO, "16_diagnosis_sleep_stage_comparison_dashboard.py")
    spec = importlib.util.spec_from_file_location("dash16_module", dash_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dash16_module"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    t0 = time.time()
    mock_st = _make_mock_streamlit()
    sys.modules["streamlit"] = mock_st
    for submod in ["streamlit.components", "streamlit.components.v1"]:
        sys.modules.setdefault(submod, types.ModuleType(submod))

    print("Loading hep_module and mod16 (mock streamlit)...")
    hep_mod = load_hep_module()
    mod16 = load_mod16()

    print("\n=== Loading full patient pool (min_eeg_channels=None) ===")
    raw_df = mod16.load_patient_data(hep_mod, [GROUP], STAGES, min_eeg_channels=None)
    print(f"  raw_df: {len(raw_df)} rows, {raw_df['patient_id'].nunique()} unique patients")
    raw_df = raw_df.copy()
    raw_df["patient_id"] = raw_df["patient_id"].map(mod16._canonical_patient_id)

    # ---- Selected channels: montage channels actually present anywhere ----
    montage_by_lower = {c.lower(): c for c in mod16.MONTAGE_1020_CHANNEL_ORDER}
    channel_counter = {}
    for individual in raw_df["individual"]:
        if individual is None or len(individual) <= 3 or individual[3] is None:
            continue
        for ch in individual[3]:
            key = str(ch).lower()
            if key in montage_by_lower:
                channel_counter[montage_by_lower[key]] = channel_counter.get(montage_by_lower[key], 0) + 1
    print("  Channel presence counts (rows) across full pool:")
    for ch, cnt in sorted(channel_counter.items(), key=lambda kv: -kv[1]):
        print(f"    {ch}: {cnt}")
    selected_channels = sorted(
        channel_counter.keys(), key=lambda c: mod16.MONTAGE_1020_CHANNEL_ORDER.index(c)
    )
    print(f"  Selected channels ({len(selected_channels)}): {selected_channels}")

    # ---- Diagnosis categories + demographics from Paper CFA cache ----
    print("\n=== Loading diagnosis categories + demographics (Paper CFA/demographics_combined.parquet) ===")
    demo_path = os.path.join(REPO, "Paper CFA", "demographics_combined.parquet")
    demo_df = pd.read_parquet(demo_path)
    demo_df["patient_id"] = demo_df["patient_id"].astype(str)
    demo_map = demo_df.set_index("patient_id")

    all_patient_ids = sorted(raw_df["patient_id"].unique())
    covered = sum(1 for p in all_patient_ids if p in demo_map.index)
    print(f"  {covered}/{len(all_patient_ids)} unique HEP patients have a demographics_combined.parquet row")

    # Fallback for the small remainder: EHR-derived categories + clinical/EHR demographics
    missing_ids = [p for p in all_patient_ids if p not in demo_map.index]
    fallback_categories = {}
    fallback_demo = pd.DataFrame()
    if missing_ids:
        print(f"  Falling back to mod16.load_ehr_data / _patient_demographics for {len(missing_ids)} patients")
        ehr_df = mod16.load_ehr_data(tuple(missing_ids))
        if not ehr_df.empty:
            fallback_categories = dict(zip(ehr_df["patient_id"], ehr_df["categories"]))
        clinical_df = mod16.load_cobrad_clinical()
        fallback_demo = mod16._patient_demographics(missing_ids, clinical_df).set_index("patient_id")

    def get_categories(pid):
        if pid in demo_map.index:
            cats = demo_map.loc[pid, "diagnosis_categories"]
            if isinstance(cats, np.ndarray):
                return list(cats)
            if isinstance(cats, list):
                return cats
            return []
        return fallback_categories.get(pid, [])

    def get_age_sex(pid):
        if pid in demo_map.index:
            row = demo_map.loc[pid]
            sex = row.get("sex")
            sex_label = sex if sex in ("Male", "Female") else "Unknown"
            return row.get("age", np.nan), sex_label
        if pid in fallback_demo.index:
            row = fallback_demo.loc[pid]
            return row.get("age", np.nan), row.get("sex", "Unknown")
        return np.nan, "Unknown"

    # ---- HR + CFA covariates from Paper CFA/cfa_combined.parquet ----
    print("\n=== Loading HR (qc_ecg_bpm) + CFA contamination (cfa_r2_excl_qrs) from cfa_combined.parquet ===")
    cfa_df = pd.read_parquet(
        os.path.join(REPO, "Paper CFA", "cfa_combined.parquet"),
        columns=["patient_id", "qc_ecg_bpm", "cfa_r2_excl_qrs"],
    )
    cfa_df["patient_id"] = cfa_df["patient_id"].astype(str)
    cfa_agg = cfa_df.groupby("patient_id").agg(
        hr_bpm=("qc_ecg_bpm", "mean"), cfa_r2=("cfa_r2_excl_qrs", "mean")
    )
    covered_cfa = sum(1 for p in all_patient_ids if p in cfa_agg.index)
    print(f"  {covered_cfa}/{len(all_patient_ids)} unique HEP patients have CFA/HR coverage")

    # ---- Build per-patient metadata table ----
    print("\n=== Building per-patient metadata ===")
    meta_rows = []
    for pid in all_patient_ids:
        cats = [str(c) for c in get_categories(pid)]
        age, sex = get_age_sex(pid)
        hr = cfa_agg.loc[pid, "hr_bpm"] if pid in cfa_agg.index else np.nan
        cfa = cfa_agg.loc[pid, "cfa_r2"] if pid in cfa_agg.index else np.nan
        is_neuro = any(c in NEURO_CATEGORIES for c in cats)
        is_nonneuro = any(c not in NEURO_CATEGORIES for c in cats)
        if not cats:
            broad_group = "Unknown"
        elif is_neuro and is_nonneuro:
            broad_group = "Both"
        elif is_neuro:
            broad_group = "Neurological"
        else:
            broad_group = "Non-neurological"
        meta_rows.append({
            "patient_id": pid, "categories": cats, "n_categories": len(cats),
            "age": age, "sex": sex, "hr_bpm": hr, "cfa_r2": cfa,
            "is_neuro": is_neuro, "is_nonneuro": is_nonneuro, "broad_group": broad_group,
        })
    meta_df = pd.DataFrame(meta_rows)
    print(meta_df["broad_group"].value_counts().to_string())
    print("\n  N patients per diagnosis category (non-exclusive membership):")
    from collections import Counter
    cat_counts = Counter()
    for cats in meta_df["categories"]:
        cat_counts.update(cats)
    for cat, n in cat_counts.most_common():
        print(f"    {cat}: {n}")

    # ---- Long-format HEP amplitude table: one row per (patient, stage, electrode) ----
    print("\n=== Computing per-electrode HEP amplitudes ===")
    long_rows = []
    for row in raw_df.itertuples(index=False):
        individual = row.individual
        if individual is None or len(individual) <= 3 or individual[3] is None:
            continue
        hep_data, times, ch_names = individual[1], individual[2], individual[3]
        data = np.asarray(hep_data, dtype=float)
        times_arr = np.asarray(times, dtype=float)
        if data.ndim != 2 or times_arr.ndim != 1 or data.shape[1] != len(times_arr):
            continue
        window_mask = (times_arr >= AMP_WINDOW[0]) & (times_arr <= AMP_WINDOW[1])
        if window_mask.sum() < 2:
            continue
        for ci, ch in enumerate(ch_names):
            ch_key = str(ch).lower()
            if ch_key not in montage_by_lower:
                continue
            electrode = montage_by_lower[ch_key]
            trace_uv = data[ci] * 1e6
            if not mod16._is_valid_hep_trace(trace_uv):
                continue
            amp = float(np.nanmean(trace_uv[window_mask]))
            if not np.isfinite(amp):
                continue
            long_rows.append({
                "patient_id": row.patient_id, "stage": row.stage,
                "electrode": electrode, "region": REGION_MAP.get(electrode, "other"),
                "hep_amplitude_uv": amp,
            })
    long_df = pd.DataFrame(long_rows)
    long_df = long_df.drop_duplicates(subset=["patient_id", "stage", "electrode"])
    print(f"  long_df: {len(long_df)} rows, {long_df['patient_id'].nunique()} unique patients, "
          f"{long_df['electrode'].nunique()} electrodes")
    print(f"  HEP amplitude range: [{long_df['hep_amplitude_uv'].min():.3f}, {long_df['hep_amplitude_uv'].max():.3f}] uV, "
          f"median={long_df['hep_amplitude_uv'].median():.3f} uV")

    long_df = long_df.merge(meta_df, on="patient_id", how="left")

    # ---- Channel-averaged amplitude (Fig2 scalar metric) via mod16._patient_hep_trace ----
    print("\n=== Computing channel-averaged HEP amplitude per (patient, stage) ===")
    avg_rows = []
    for row in raw_df.itertuples(index=False):
        trace_result = mod16._patient_hep_trace(row.individual, selected_channels, False)
        if trace_result is None:
            continue
        trace, times_arr = trace_result
        window_mask = (times_arr >= AMP_WINDOW[0]) & (times_arr <= AMP_WINDOW[1])
        if window_mask.sum() < 2:
            continue
        avg_rows.append({
            "patient_id": row.patient_id, "stage": row.stage,
            "hep_amplitude_uv": float(np.nanmean(trace[window_mask])),
        })
    avg_df = pd.DataFrame(avg_rows).drop_duplicates(subset=["patient_id", "stage"])
    avg_df = avg_df.merge(meta_df, on="patient_id", how="left")
    print(f"  avg_df (channel-averaged): {len(avg_df)} rows, {avg_df['patient_id'].nunique()} unique patients")
    print(f"  Channel-averaged HEP amplitude range: [{avg_df['hep_amplitude_uv'].min():.3f}, "
          f"{avg_df['hep_amplitude_uv'].max():.3f}] uV, median={avg_df['hep_amplitude_uv'].median():.3f} uV")

    with open(LONG_DF_PKL, "wb") as f:
        pickle.dump({
            "long_df": long_df, "avg_df": avg_df, "meta_df": meta_df,
            "selected_channels": selected_channels, "amp_window": AMP_WINDOW,
            "group": GROUP, "stages": STAGES,
        }, f)
    print(f"\nSaved {LONG_DF_PKL}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
