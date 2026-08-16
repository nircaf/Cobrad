"""
Diagnosis-cohort HEP stage-delta alignment analysis.

Three analyses, all run on real cached data (no fabricated numbers):
  1. Stage-delta pattern alignment: does any of the 4 diagnosis sub-cohorts
     (sparse ~6-electrode montage, Cohort B) show a within-subject HEP
     stage-delta vector (light_sleep->N3, N3->R, light_sleep->R) that matches
     the undiagnosed reference cohort's (Cohort A, >10 electrodes) pattern,
     more closely than chance (label-shuffle permutation test)?
  2. Age comparison: Cohort A vs Cohort B overall, and Cohort A vs each of the
     4 diagnosis sub-cohorts (Mann-Whitney U).
  3. PSD comparison: per-stage EEG band power, Cohort A vs the 4 diagnosis
     sub-cohorts (Kruskal-Wallis per band x stage).

Reuses existing dashboard machinery (mod16 = 16_diagnosis_sleep_stage_comparison_
dashboard.py, hep_mod = 6_hep_group_comparison.py) via the mock-streamlit import
pattern from Paper1/analysis_age_by_stage.py / Paper1/build_stage_delta_age_
analysis.py. Statistics (benjamini_hochberg, _patient_hep_trace, PSD helpers,
select_non_diagnosis_cohort, _standard_1020_channel_count) are called directly,
not reimplemented, following 23_brain_body_synchrony_dysrhythmia_dashboard.py's
build_grouped_df / compute_amplitudes pattern.

Run with the project venv: source venv/bin/activate && python3 Paper1/build_diagnosis_alignment_analysis.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
import types
import importlib.util

import numpy as np
import pandas as pd
from scipy import stats
from scipy import signal as scipy_signal

REPO = "/storage/pblab_shared_data2/Nir/Cobrad"
OUT_DIR = os.path.join(REPO, "Paper1")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
RESULTS_JSON = os.path.join(OUT_DIR, "diagnosis_alignment_results.json")
RESULTS_PKL = os.path.join(OUT_DIR, "diagnosis_alignment_results.pkl")

GROUP = "Harvard_Electroencephalography"
STAGES = ["light_sleep", "N3", "R"]
REF_GROUP = "Susp. Epilepsy"
DIAG_GROUPS = [
    "Atrial Fibrillation",
    "Heart Failure",
    "Stroke / Cerebrovascular",
    "Cognitive Impairment / Dementia",
]
AMP_WINDOW = (0.15, 0.5)
DELTA_PAIRS = [("light_sleep", "N3"), ("N3", "R"), ("light_sleep", "R")]
N_PERM = 5000
PSD_MAX_PER_GROUP_STAGE = 40
PSD_BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0), "beta": (12.0, 30.0)}
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Mock streamlit + module loaders (copied pattern from Paper1/analysis_age_by_stage.py)
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


# ---------------------------------------------------------------------------
# Cohort construction
# ---------------------------------------------------------------------------
def build_cohort_a(mod16, hep_mod):
    """Cohort A: undiagnosed ('Susp. Epilepsy') reference cohort, >10 EEG electrodes.
    Uses the existing individuals_cache_min11eeg_* cache (>=11 == >10 electrodes,
    via hep_mod._standard_1020_channel_count) -- no cache regeneration needed."""
    raw_df = mod16.load_patient_data(hep_mod, [GROUP], STAGES, min_eeg_channels=11)
    print(f"  Cohort-A candidate pool (>10 EEG ch): {raw_df['patient_id'].nunique()} unique patients, {len(raw_df)} rows")
    ehr_df = mod16.load_ehr_data(tuple(sorted(raw_df["patient_id"].unique())))
    cohort_a_df = mod16.select_non_diagnosis_cohort(raw_df, ehr_df)
    print(f"  Cohort A ('{REF_GROUP}', >10 EEG ch): {cohort_a_df['patient_id'].nunique()} unique patients")
    return raw_df, cohort_a_df


def build_cohort_b(mod16, hep_mod):
    """Cohort B: diagnosed patients, sparse (~6-electrode) montage.
    Loads the full unfiltered cache (min_eeg_channels=None), computes the actual
    per-recording standard-1020-channel count via hep_mod._standard_1020_channel_count,
    and filters explicitly to that count == 6 (verified against the observed
    distribution below), then splits into the 4 non-exclusive diagnosis groups."""
    raw_df = mod16.load_patient_data(hep_mod, [GROUP], STAGES, min_eeg_channels=None)
    print(f"  Full unfiltered pool: {raw_df['patient_id'].nunique()} unique patients, {len(raw_df)} rows")

    raw_df = raw_df.copy()
    raw_df["n_eeg_channels"] = raw_df["individual"].map(
        lambda ind: hep_mod._standard_1020_channel_count(ind[3]) if ind is not None and len(ind) > 3 else 0
    )
    counts = raw_df.drop_duplicates("patient_id")["n_eeg_channels"].value_counts().sort_index()
    print("  Distribution of per-patient standard-10-20 EEG channel count (rows, one per patient x first stage seen):")
    for n_ch, cnt in counts.items():
        print(f"    {n_ch} channels: {cnt} patients")

    six_ch_ids = set(
        raw_df.loc[raw_df["n_eeg_channels"] == 6, "patient_id"].unique()
    )
    print(f"  Patients with exactly 6 standard 10-20 EEG channels: {len(six_ch_ids)}")
    pool = raw_df[raw_df["patient_id"].isin(six_ch_ids)].copy()

    ehr_df = mod16.load_ehr_data(tuple(sorted(pool["patient_id"].unique())))
    categories_by_patient = (
        dict(zip(ehr_df["patient_id"], ehr_df["categories"])) if not ehr_df.empty else {}
    )
    frames = []
    for diag_group in DIAG_GROUPS:
        matched_ids = {
            pid for pid, cats in categories_by_patient.items()
            if isinstance(cats, list) and diag_group in cats
        }
        subset = pool[pool["patient_id"].isin(matched_ids)].copy()
        subset["diagnosis_group"] = diag_group
        n_unique = subset["patient_id"].nunique()
        print(f"  Cohort B / {diag_group}: {n_unique} unique patients (6-electrode montage)")
        frames.append(subset)
    grouped_b = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return raw_df, grouped_b


# ---------------------------------------------------------------------------
# 1) Stage-delta amplitude + alignment
# ---------------------------------------------------------------------------
def compute_amp_df(grouped_df, mod16, selected_channels):
    """One HEP amplitude scalar per (diagnosis_group, stage, patient): mean of the
    channel-averaged HEP trace (mod16._patient_hep_trace) inside AMP_WINDOW.
    Identical logic to 23_brain_body_synchrony_dysrhythmia_dashboard.compute_amplitudes."""
    rows = []
    for _, row in grouped_df.iterrows():
        trace_result = mod16._patient_hep_trace(row["individual"], selected_channels, False)
        if trace_result is None:
            continue
        trace, times = trace_result
        window_mask = (times >= AMP_WINDOW[0]) & (times <= AMP_WINDOW[1])
        if window_mask.sum() < 2:
            continue
        rows.append({
            "diagnosis_group": row["diagnosis_group"],
            "stage": row["stage"],
            "patient_id": mod16._canonical_patient_id(row["patient_id"]),
            "amplitude": float(np.nanmean(trace[window_mask])),
        })
    amp_df = pd.DataFrame(rows)
    if not amp_df.empty:
        amp_df = amp_df.drop_duplicates(subset=["diagnosis_group", "stage", "patient_id"])
    return amp_df


def patient_delta_matrix(amp_df, group_label):
    """Return (patient_ids, deltas[n_patients, 3]) for patients present in all
    3 stages within this diagnosis_group. Column order = DELTA_PAIRS."""
    sub = amp_df[amp_df["diagnosis_group"] == group_label]
    wide = sub.pivot_table(index="patient_id", columns="stage", values="amplitude", aggfunc="first")
    wide = wide.dropna(subset=STAGES)
    if wide.empty:
        return [], np.empty((0, 3))
    deltas = np.column_stack([
        (wide[b] - wide[a]).to_numpy() for a, b in DELTA_PAIRS
    ])
    return wide.index.tolist(), deltas


def alignment_test(delta_a, delta_sub, n_perm, rng):
    """Permutation test: shuffle group labels among the pooled patients
    (cohort A matched-3-stage patients U sub-cohort matched-3-stage patients),
    recompute the two group-mean delta vectors and their Euclidean distance,
    build a null distribution, and return the observed distance's upper-tail
    p-value (fraction of null distances >= observed -- i.e. probability of
    seeing at least this much divergence between the two group means if the
    two groups were really drawn from the same underlying stage-delta pattern)."""
    n_a, n_sub = len(delta_a), len(delta_sub)
    if n_a < 3 or n_sub < 3:
        return dict(n_a=n_a, n_sub=n_sub, obs_distance=np.nan, p_perm=np.nan,
                     pearson_r=np.nan, pearson_p=np.nan, cosine_sim=np.nan,
                     mean_delta_a=[np.nan] * 3, mean_delta_sub=[np.nan] * 3)

    mean_a = delta_a.mean(axis=0)
    mean_sub = delta_sub.mean(axis=0)
    obs_distance = float(np.linalg.norm(mean_a - mean_sub))
    r, p_r = stats.pearsonr(mean_a, mean_sub) if np.std(mean_a) > 0 and np.std(mean_sub) > 0 else (np.nan, np.nan)
    denom = (np.linalg.norm(mean_a) * np.linalg.norm(mean_sub))
    cosine_sim = float(np.dot(mean_a, mean_sub) / denom) if denom > 0 else np.nan

    pooled = np.vstack([delta_a, delta_sub])
    n_total = n_a + n_sub
    null_distances = np.empty(n_perm)
    for i in range(n_perm):
        perm_idx = rng.permutation(n_total)
        grp_a_idx, grp_b_idx = perm_idx[:n_a], perm_idx[n_a:]
        m_a = pooled[grp_a_idx].mean(axis=0)
        m_b = pooled[grp_b_idx].mean(axis=0)
        null_distances[i] = np.linalg.norm(m_a - m_b)
    p_perm = float(np.mean(null_distances >= obs_distance))

    return dict(
        n_a=n_a, n_sub=n_sub, obs_distance=obs_distance, p_perm=p_perm,
        pearson_r=float(r) if np.isfinite(r) else np.nan,
        pearson_p=float(p_r) if np.isfinite(p_r) else np.nan,
        cosine_sim=cosine_sim,
        mean_delta_a=mean_a.tolist(), mean_delta_sub=mean_sub.tolist(),
        null_distances_summary=dict(
            mean=float(np.mean(null_distances)), sd=float(np.std(null_distances)),
            p5=float(np.percentile(null_distances, 5)), p95=float(np.percentile(null_distances, 95)),
        ),
    )


# ---------------------------------------------------------------------------
# 2) Age comparison
# ---------------------------------------------------------------------------
def get_ages(mod16, patient_ids, clinical_df):
    demo = mod16._patient_demographics(list(patient_ids), clinical_df)
    return demo.set_index("patient_id")["age"].dropna()


def mwu_summary(a, b):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float)
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return dict(n1=len(a), n2=len(b), U=np.nan, p=np.nan,
                     median1=float(np.median(a)) if len(a) else np.nan,
                     median2=float(np.median(b)) if len(b) else np.nan,
                     iqr1=[np.nan, np.nan], iqr2=[np.nan, np.nan])
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return dict(
        n1=len(a), n2=len(b), U=float(u), p=float(p),
        median1=float(np.median(a)), median2=float(np.median(b)),
        iqr1=[float(np.percentile(a, 25)), float(np.percentile(a, 75))],
        iqr2=[float(np.percentile(b, 25)), float(np.percentile(b, 75))],
    )


# ---------------------------------------------------------------------------
# 3) PSD comparison
# ---------------------------------------------------------------------------
def psd_candidate_rows(grouped_df):
    """One source recording per (diagnosis_group, stage, patient) with enough
    R-peaks, source_path resolved the same way as mod16._hr_psd_candidate_rows."""
    rows = []
    seen = set()
    for row in grouped_df.itertuples(index=False):
        key = (row.patient_id, row.diagnosis_group, row.stage)
        if key in seen:
            continue
        individual = row.individual
        if individual is None or len(individual) <= 4 or individual[4] is None:
            continue
        n_rpeaks = len(np.asarray(individual[4]).ravel())
        if n_rpeaks < 20:
            continue
        source_path = os.path.join(BASE_PATH, GROUP, str(row.stage), f"{individual[0]}.pkl")
        if not os.path.exists(source_path):
            continue
        seen.add(key)
        rows.append({
            "patient_id": str(row.patient_id), "diagnosis_group": str(row.diagnosis_group),
            "stage": str(row.stage), "source_path": source_path, "n_rpeaks": n_rpeaks,
        })
    return pd.DataFrame(rows)


def compute_patient_psd(hep_mod, source_path):
    """Load one patient's raw pickle, reprocess to clean continuous EEG (same
    pipeline as mod16._compute_hr_psd_patient_correlation), and return
    (freqs, mean_psd_across_channels, band_powers dict)."""
    try:
        with open(source_path, "rb") as f:
            source_raw = pickle.load(f)
        raw = hep_mod.drop_non_eeg_channels(source_raw.copy())
        processed = hep_mod.process_file_data(raw, "psd_tmp")
        if processed is None:
            return None
        clean_raw, sfreq, _rpeak_ts, rpeaks, _minmax, _log = processed
        eeg_data = clean_raw.get_data()
        if eeg_data.shape[1] < int(sfreq * 4):
            return None
        nperseg = min(eeg_data.shape[1], int(sfreq * 8))
        freqs, psd = scipy_signal.welch(eeg_data, fs=float(sfreq), axis=-1, nperseg=nperseg)
        mean_psd = np.nanmean(psd, axis=0)
        band_powers = {}
        for band, (fmin, fmax) in PSD_BANDS.items():
            mask = (freqs >= fmin) & (freqs <= min(fmax, sfreq / 2.0))
            if not mask.any():
                band_powers[band] = np.nan
                continue
            band_powers[band] = float(np.trapz(mean_psd[mask], freqs[mask]))
        return freqs, mean_psd, band_powers
    except Exception as exc:
        print(f"    [PSD] failed on {source_path}: {exc}", file=sys.stderr)
        return None


def build_psd_dataset(hep_mod, grouped_all_df, group_labels):
    candidates = psd_candidate_rows(grouped_all_df[grouped_all_df["diagnosis_group"].isin(group_labels)])
    if candidates.empty:
        return pd.DataFrame(), {}
    sampled = (
        candidates.sort_values("n_rpeaks", ascending=False)
        .groupby(["diagnosis_group", "stage"], group_keys=False)
        .head(PSD_MAX_PER_GROUP_STAGE)
    )
    print(f"  PSD: sampling {len(sampled)} (group,stage,patient) recordings "
          f"(cap {PSD_MAX_PER_GROUP_STAGE}/group/stage) out of {len(candidates)} candidates")
    rows = []
    curves = {}  # (group,stage) -> list of (freqs, psd)
    t0 = time.time()
    for i, row in enumerate(sampled.itertuples(index=False)):
        result = compute_patient_psd(hep_mod, row.source_path)
        if result is None:
            continue
        freqs, mean_psd, band_powers = result
        rows.append({
            "diagnosis_group": row.diagnosis_group, "stage": row.stage,
            "patient_id": row.patient_id, **band_powers,
        })
        curves.setdefault((row.diagnosis_group, row.stage), []).append((freqs, mean_psd))
        if (i + 1) % 25 == 0:
            print(f"    PSD progress {i + 1}/{len(sampled)} ({time.time() - t0:.0f}s elapsed)")
    return pd.DataFrame(rows), curves


def kruskal_band_stage(psd_df, group_labels):
    rows = []
    for band in PSD_BANDS:
        for stage in STAGES:
            samples = [
                psd_df.loc[(psd_df["diagnosis_group"] == g) & (psd_df["stage"] == stage), band].dropna().to_numpy()
                for g in group_labels
            ]
            samples = [s for s in samples if len(s) >= 2]
            if len(samples) >= 2:
                try:
                    h, p = stats.kruskal(*samples)
                except Exception:
                    h, p = np.nan, np.nan
            else:
                h, p = np.nan, np.nan
            rows.append({"band": band, "stage": stage, "H_stat": h, "p_value": p,
                         "n_groups": len(samples)})
    table = pd.DataFrame(rows)
    table["q_value"] = np.nan
    finite = table["p_value"].notna() & np.isfinite(table["p_value"])
    if finite.any():
        from_bh = _bh(table.loc[finite, "p_value"].to_numpy(dtype=float))
        table.loc[finite, "q_value"] = from_bh
    return table


def _bh(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    n = len(ranked)
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0, 1)
    return out


def plot_psd_curves(curves, group_labels, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    fig, axes = plt.subplots(1, len(STAGES), figsize=(15, 4.2), sharey=True)
    for ax, stage in zip(axes, STAGES):
        for gi, group in enumerate(group_labels):
            pairs = curves.get((group, stage), [])
            if not pairs:
                continue
            common_freqs = pairs[0][0]
            mats = [np.interp(common_freqs, f, p) for f, p in pairs]
            mean_psd = np.mean(mats, axis=0)
            mask = common_freqs <= 45.0
            ax.plot(common_freqs[mask], np.log10(mean_psd[mask] + 1e-20), color=palette[gi % len(palette)],
                     label=f"{group} (n={len(pairs)})", linewidth=1.5)
        ax.set_title(stage)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xlim(0, 45)
    axes[0].set_ylabel("log10 power (a.u.)")
    axes[-1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global BASE_PATH
    t0 = time.time()
    mock_st = _make_mock_streamlit()
    sys.modules["streamlit"] = mock_st
    for submod in ["streamlit.components", "streamlit.components.v1"]:
        if submod not in sys.modules:
            sys.modules[submod] = types.ModuleType(submod)

    print("Loading hep_module and mod16 (mock streamlit)...")
    hep_mod = load_hep_module()
    mod16 = load_mod16()
    BASE_PATH = mod16.BASE_PATH

    print("\n=== Building Cohort A (undiagnosed reference, >10 EEG channels) ===")
    raw_a, cohort_a_df = build_cohort_a(mod16, hep_mod)

    print("\n=== Building Cohort B (diagnosed, 6-electrode montage, 4 sub-cohorts) ===")
    raw_b, grouped_b_df = build_cohort_b(mod16, hep_mod)

    n_cohort_a = cohort_a_df["patient_id"].nunique()
    n_cohort_b_pool = raw_b.loc[raw_b["n_eeg_channels"] == 6, "patient_id"].nunique()
    diag_n = {g: grouped_b_df.loc[grouped_b_df["diagnosis_group"] == g, "patient_id"].nunique() for g in DIAG_GROUPS}

    cohort_a_df = cohort_a_df.copy()
    cohort_a_df["diagnosis_group"] = REF_GROUP  # already set by select_non_diagnosis_cohort, kept explicit

    # ---- Selected channels: union of montage channels actually present -----
    montage_by_lower = {c.lower(): c for c in mod16.MONTAGE_1020_CHANNEL_ORDER}
    def available_channels(df):
        return {
            montage_by_lower[str(ch).lower()]
            for individual in df["individual"]
            if individual is not None and len(individual) > 3 and individual[3] is not None
            for ch in individual[3]
            if str(ch).lower() in montage_by_lower
        }
    selected_channels = sorted(
        available_channels(cohort_a_df) | available_channels(grouped_b_df),
        key=lambda c: mod16.MONTAGE_1020_CHANNEL_ORDER.index(c),
    )
    print(f"\nSelected channels (union across cohorts, {len(selected_channels)}): {selected_channels}")

    # =========================================================================
    # Analysis 1: stage-delta amplitude + alignment permutation test
    # =========================================================================
    print("\n=== Analysis 1: HEP amplitude + stage-delta alignment ===")
    grouped_all = pd.concat([cohort_a_df, grouped_b_df], ignore_index=True)
    amp_df = compute_amp_df(grouped_all, mod16, selected_channels)
    print(f"  amp_df rows: {len(amp_df)}")

    ids_a, delta_a = patient_delta_matrix(amp_df, REF_GROUP)
    print(f"  Cohort A patients with usable amplitude in all 3 stages: {len(ids_a)}")

    rng = np.random.default_rng(RNG_SEED)
    alignment_results = {}
    for group in DIAG_GROUPS:
        ids_sub, delta_sub = patient_delta_matrix(amp_df, group)
        print(f"  {group}: {len(ids_sub)} patients with usable amplitude in all 3 stages")
        res = alignment_test(delta_a, delta_sub, N_PERM, rng)
        alignment_results[group] = res
        print(f"    obs_distance={res['obs_distance']}, p_perm={res['p_perm']}, "
              f"pearson_r={res['pearson_r']}, cosine_sim={res['cosine_sim']}")

    p_list = [alignment_results[g]["p_perm"] for g in DIAG_GROUPS]
    finite_mask = [np.isfinite(p) for p in p_list]
    q_list = [np.nan] * len(DIAG_GROUPS)
    finite_p = [p for p, m in zip(p_list, finite_mask) if m]
    if finite_p:
        q_vals = _bh(np.array(finite_p))
        qi = 0
        for i, m in enumerate(finite_mask):
            if m:
                q_list[i] = float(q_vals[qi])
                qi += 1
    for g, q in zip(DIAG_GROUPS, q_list):
        alignment_results[g]["q_perm"] = q
        verdict = "ALIGNED with Cohort A (not significantly divergent)" if (np.isfinite(q) and q >= 0.05) else (
            "DIVERGES from Cohort A (significant)" if np.isfinite(q) else "not testable (insufficient N)")
        print(f"  {g}: q_perm={q} -> {verdict}")

    # =========================================================================
    # Analysis 2: age comparison
    # =========================================================================
    print("\n=== Analysis 2: age comparison ===")
    clinical_df = mod16.load_cobrad_clinical()
    ages_a = get_ages(mod16, cohort_a_df["patient_id"].unique(), clinical_df)
    ages_b_all = get_ages(mod16, grouped_b_df["patient_id"].unique(), clinical_df)
    print(f"  Cohort A: n_age={len(ages_a)} median={ages_a.median() if len(ages_a) else np.nan}")
    print(f"  Cohort B (overall, union of 4 diagnosis groups): n_age={len(ages_b_all)} "
          f"median={ages_b_all.median() if len(ages_b_all) else np.nan}")

    age_overall = mwu_summary(ages_a.to_numpy(), ages_b_all.to_numpy())
    print(f"  A vs B overall: {age_overall}")

    age_results = {}
    age_p_list = []
    for group in DIAG_GROUPS:
        ids_g = grouped_b_df.loc[grouped_b_df["diagnosis_group"] == group, "patient_id"].unique()
        ages_g = get_ages(mod16, ids_g, clinical_df)
        res = mwu_summary(ages_a.to_numpy(), ages_g.to_numpy())
        age_results[group] = res
        age_p_list.append(res["p"])
        print(f"  A vs {group}: n1={res['n1']} n2={res['n2']} median_A={res['median1']} "
              f"median_{group}={res['median2']} p={res['p']}")

    finite_mask = [np.isfinite(p) for p in age_p_list]
    age_q_list = [np.nan] * len(DIAG_GROUPS)
    finite_p = [p for p, m in zip(age_p_list, finite_mask) if m]
    if finite_p:
        q_vals = _bh(np.array(finite_p))
        qi = 0
        for i, m in enumerate(finite_mask):
            if m:
                age_q_list[i] = float(q_vals[qi])
                qi += 1
    for g, q in zip(DIAG_GROUPS, age_q_list):
        age_results[g]["q"] = q
        print(f"  A vs {g}: q={q}")

    # =========================================================================
    # Analysis 3: PSD comparison
    # =========================================================================
    print("\n=== Analysis 3: PSD band-power comparison ===")
    psd_group_labels = [REF_GROUP] + DIAG_GROUPS
    psd_df, curves = build_psd_dataset(hep_mod, grouped_all, psd_group_labels)
    print(f"  psd_df rows: {len(psd_df)}")
    if not psd_df.empty:
        n_by_group_stage = psd_df.groupby(["diagnosis_group", "stage"])["patient_id"].nunique()
        print(n_by_group_stage.to_string())
        kw_table = kruskal_band_stage(psd_df, psd_group_labels)
        print(kw_table.to_string())
        psd_fig_path = os.path.join(FIG_DIR, "diagnosis_alignment_psd_curves.png")
        plot_psd_curves(curves, psd_group_labels, psd_fig_path)
    else:
        kw_table = pd.DataFrame()
        psd_fig_path = None
        print("  !!! No usable PSD recordings -- PSD analysis will be reported as infeasible with available data.")

    # =========================================================================
    # Save results
    # =========================================================================
    results = dict(
        n_cohort_a=int(n_cohort_a),
        n_cohort_b_pool_6ch=int(n_cohort_b_pool),
        diag_n=diag_n,
        selected_channels=selected_channels,
        alignment_results=alignment_results,
        age_overall=age_overall,
        age_results=age_results,
        psd_kw_table=kw_table.to_dict(orient="records") if not kw_table.empty else [],
        psd_n_by_group_stage=(
            {f"{g}|{s}": int(n) for (g, s), n in n_by_group_stage.items()} if not psd_df.empty else {}
        ),
        psd_fig_path=psd_fig_path,
        n_perm=N_PERM,
        psd_max_per_group_stage=PSD_MAX_PER_GROUP_STAGE,
    )

    def _json_default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not JSON serializable: {type(o)}")

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"\nSaved JSON results to {RESULTS_JSON}")

    with open(RESULTS_PKL, "wb") as f:
        pickle.dump(dict(results, amp_df=amp_df, psd_df=psd_df, curves=curves), f)
    print(f"Saved pickle (incl. dataframes) to {RESULTS_PKL}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
