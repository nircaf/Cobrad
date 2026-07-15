"""
HEP Diagnosis × Sleep Stage Comparison Dashboard.

PhD-oriented dashboard comparing Heartbeat-Evoked Potential (HEP) T-wave
coupling across sleep stages, grouped by clinical diagnosis. Supports 6
grouping modes to compare heart-brain modulation across diseases.

Run:
  streamlit run 16_diagnosis_sleep_stage_comparison_dashboard.py
"""

from __future__ import annotations

import glob as _glob
import importlib.util
import os
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from scipy.ndimage import label as connected_components
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =============================================================================
# Import 6_hep_group_comparison.py as a module (reuse its loading/stats code)
# =============================================================================
_HEP_MODULE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "6_hep_group_comparison.py")


@st.cache_resource(show_spinner="Loading HEP analysis module...")
def _load_hep_module():
    """
    Import 6_hep_group_comparison.py against the REAL streamlit module (this
    dashboard IS a Streamlit app, so we cannot mock streamlit out like
    generate_sleep_stage_cache.py does).

    6_hep_group_comparison.py calls st.set_page_config(layout="wide") and
    reassigns st.pyplot at module import time. Streamlit only allows
    set_page_config to be called once, and dashboard 16 must be the one that
    calls it (as its actual first Streamlit command). So we transiently
    no-op both problematic top-level calls only for the duration of the
    import, then restore the real streamlit attributes afterward.
    """
    previous_mod = sys.modules.get("hep_group_comparison_mod")
    real_set_page_config = st.set_page_config
    real_pyplot = getattr(previous_mod, "_original_st_pyplot", st.pyplot)
    real_plotly_chart = getattr(
        previous_mod, "_original_st_plotly_chart", st.plotly_chart
    )
    st.set_page_config = lambda *a, **kw: None
    try:
        spec = importlib.util.spec_from_file_location("hep_group_comparison_mod", _HEP_MODULE_PATH)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["hep_group_comparison_mod"] = mod
        spec.loader.exec_module(mod)
    finally:
        st.set_page_config = real_set_page_config
        # The imported module wraps both rendering functions and appends
        # "(ICA cleaned)" to every Plotly annotation. Restore the dashboard's
        # original renderers so statistical labels are not mistaken for titles.
        st.pyplot = real_pyplot
        st.plotly_chart = real_plotly_chart
    return mod

# =============================================================================
# Constants
# =============================================================================
BASE_PATH = "/storage/pblab_shared_data/Nir/Cobrad/pickles_sleep_stage"
CLINICAL_FILE = Path("/storage/pblab_shared_data/Nir/Cobrad/COBRAD_clinical_24022025.xlsx")
_EHR_BASE = "/storage/pblab_shared_data/Nir/Cobrad/EDF_Format/Harvard_Electroencephalography/EHR"
_EHR_DIAG_BASE = os.path.join(
    _EHR_BASE, "data_Structured", "Parquet", "Parquet", "bdsp_i0006_diagnosis"
)
_EHR_I0002_ICD10_BASE = os.path.join(
    _EHR_BASE, "I0002-EHR", "I0002_structured", "icd10_codes_nax_2024_parquet"
)

STAGE_ORDER = ["W", "light_sleep", "N3", "R"]
STAGE_LABELS = {"W": "Wake", "light_sleep": "Light Sleep", "N3": "N3 (SWS)", "R": "REM"}
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00", "#000000"]
PAIRWISE_CLUSTER_ALPHA = 0.01
ELECTRODE_MIN_PATIENT_COVERAGE_FRACTION = 0.05
ELECTRODE_MIN_MATCHED_PATIENTS = 2
WAVEFORM_PLOT_SIZE = 900
AI_FEATURE_CACHE = Path("/storage/pblab_shared_data/Nir/Cobrad/cache/14_autoencoder_clusters")
AI_PHYSIOLOGICAL_FEATURES = [
    "eeg_abs_delta_mean", "eeg_rel_delta_mean",
    "eeg_abs_theta_mean", "eeg_rel_theta_mean",
    "eeg_abs_alpha_mean", "eeg_rel_alpha_mean",
    "eeg_abs_sigma_mean", "eeg_rel_sigma_mean",
    "eeg_abs_beta_mean", "eeg_rel_beta_mean",
    "eeg_abs_gamma_mean", "eeg_rel_gamma_mean",
    "eeg_delta_theta_ratio_mean", "eeg_theta_alpha_ratio_mean",
    "eeg_spectral_entropy_mean", "eeg_hjorth_activity_mean",
    "eeg_hjorth_mobility_mean", "eeg_hjorth_complexity_mean",
    "eeg_line_length_mean", "ecg_rpeak_count", "ecg_mean_hr",
    "ecg_sdnn", "ecg_rmssd", "ecg_pnn50", "ecg_lf_power",
    "ecg_hf_power", "ecg_lf_hf_ratio",
]
NON_PATIENT_CACHE_PREFIXES = ("individuals_cache", "non_eeg_individuals_cache")
DEFAULT_HEP_GROUPS = [
    "The_Human_Sleep_Project",
    "Young_The_Human_Sleep_Project",
    "Harvard_Electroencephalography",
]
# NOTE: per-file disk caching / checkpointing is now handled entirely inside
# 6_hep_group_comparison.get_group_individuals (per group/stage individuals_cache*.pkl).
# No separate cache/checkpoint machinery is needed here.

# =============================================================================
# Diagnosis categories (copied verbatim from file 15)
# =============================================================================
_DIAG_CATEGORIES: Dict[str, Any] = {
    "Obstructive Sleep Apnea": lambda n: "sleep apnea" in n.lower(),
    "Hypertension": lambda n: "hypertension" in n.lower(),
    "Diabetes": lambda n: "diabetes" in n.lower(),
    "Heart Failure": lambda n: "heart failure" in n.lower(),
    "Coronary Artery Disease": lambda n: "coronary" in n.lower() or "atherosclerotic heart" in n.lower(),
    "Heart Transplant": lambda n: "heart transplant" in n.lower() or "transplant status" in n.lower(),
    "Atrial Fibrillation": lambda n: "atrial fibrillation" in n.lower() or "atrial flutter" in n.lower(),
    "COPD": lambda n: "chronic obstructive" in n.lower(),
    "Obesity": lambda n: "obesity" in n.lower() or "obese" in n.lower(),
    "Depression": lambda n: "depressive" in n.lower() or "depression" in n.lower(),
    "Anxiety": lambda n: "anxiety" in n.lower(),
    "Cognitive Impairment / Dementia": lambda n: "cognitive" in n.lower() or "alzheimer" in n.lower() or "dementia" in n.lower(),
    "Stroke / Cerebrovascular": lambda n: "infarction" in n.lower() or "cerebrovascular" in n.lower() or "stroke" in n.lower() or "hemorrhage" in n.lower(),
    "Kidney Disease": lambda n: "kidney" in n.lower() or "renal" in n.lower(),
    "Anemia": lambda n: "anemia" in n.lower(),
}

# =============================================================================
# EHR helpers (copied verbatim from file 15)
# =============================================================================

def _extract_cobrad_id(raw_id: str) -> Optional[str]:
    """Try to extract a COBRAD-style id like ``345-002`` from a longer id string."""
    if raw_id is None:
        return None
    s = str(raw_id)
    m = re.search(r"(\d{3}-\d{3})", s)
    if m:
        return m.group(1)
    return None


def _safe_encode_sex(series: pd.Series) -> pd.Series:
    """Encode sex column to 0/1 numeric."""
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "m": 1, "male": 1, "1": 1, "1.0": 1,
        "f": 0, "female": 0, "0": 0, "0.0": 0,
    }
    out = s.map(mapping)
    return pd.to_numeric(out, errors="coerce")


def _pid_to_bdsp_ehr(pid_str: str) -> Optional[int]:
    """Convert SUB-I0006179000017_... patient ID to 9-digit BDSPPatientID int."""
    import re as _re_ehr
    m = _re_ehr.search(r'I\d{4}(\d{9})', str(pid_str))
    if m:
        return int(m.group(1))
    return None


def _assign_categories(dx_names_list: List[str]) -> List[str]:
    """Map list of diagnosis names to sorted list of broad category names."""
    cats: set = set()
    for dx in dx_names_list:
        for cat_name, test_fn in _DIAG_CATEGORIES.items():
            if test_fn(dx):
                cats.add(cat_name)
    return sorted(cats)


def _add_dx_rows_to_patient_map(
    patient_map: dict, df: pd.DataFrame, pid_col: str, dx_col: str, target_ids=None
) -> None:
    if df is None or df.empty or pid_col not in df.columns or dx_col not in df.columns:
        return
    work = df[[pid_col, dx_col]].copy()
    work = work[work[pid_col].notna() & work[dx_col].notna()]
    if work.empty:
        return
    work[pid_col] = pd.to_numeric(work[pid_col], errors="coerce")
    work = work.dropna(subset=[pid_col])
    work[pid_col] = work[pid_col].astype(int)
    if target_ids is not None:
        work = work[work[pid_col].isin(target_ids)]
    work[dx_col] = work[dx_col].astype(str).str.strip()
    work = work[(work[dx_col] != "") & (work[dx_col].str.lower() != "not recorded")]
    for pid_val, grp in work.groupby(pid_col):
        patient_map.setdefault(int(pid_val), set()).update(grp[dx_col].dropna().unique().tolist())


def _read_ehr_parquet(path: str, columns: List[str], pid_col: str, target_ids=None) -> pd.DataFrame:
    if target_ids:
        try:
            return pd.read_parquet(path, columns=columns, filters=[(pid_col, "in", list(target_ids))])
        except Exception:
            pass
    return pd.read_parquet(path, columns=columns)


def _load_ehr_diagnosis_map(target_ids=None) -> dict:
    """Load I0006 + I0002 parquet chunks -> {bdsp_patient_id (int): [dx_name, ...]}."""
    target_ids = set(int(x) for x in target_ids) if target_ids is not None else None
    patient_map: dict = {}
    for ch in sorted(_glob.glob(os.path.join(_EHR_DIAG_BASE, "*.parquet"))):
        try:
            df_ch = _read_ehr_parquet(
                ch, columns=["bdsp_patient_id", "dx_name", "dx_code_icd_10"],
                pid_col="bdsp_patient_id", target_ids=target_ids,
            )
            _add_dx_rows_to_patient_map(patient_map, df_ch, "bdsp_patient_id", "dx_name", target_ids)
        except Exception:
            pass
    for ch in sorted(_glob.glob(os.path.join(_EHR_I0002_ICD10_BASE, "*.parquet"))):
        try:
            df_ch = _read_ehr_parquet(
                ch, columns=["BDSPPatientID", "LongDescription"],
                pid_col="BDSPPatientID", target_ids=target_ids,
            )
            _add_dx_rows_to_patient_map(patient_map, df_ch, "BDSPPatientID", "LongDescription", target_ids)
        except Exception:
            pass
    return {pid: sorted(dx_names) for pid, dx_names in patient_map.items()}


# =============================================================================
# Clinical data loading (copied verbatim from file 15)
# =============================================================================

@st.cache_data(show_spinner=False)
def load_cobrad_clinical() -> pd.DataFrame:
    """Load the clinical sheet from the COBRAD spreadsheet."""
    if not CLINICAL_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_excel(CLINICAL_FILE, sheet_name="clinical")
    except Exception:
        return pd.DataFrame()

    keep = [
        "record_id", "age", "sex", "mmse", "moca",
        "cdr_sum_of_boxes", "ApoE4", "Class_bvftd", "Class_LBD",
    ]
    present = [c for c in keep if c in df.columns]
    df = df[present].copy()

    if "record_id" not in df.columns:
        return pd.DataFrame()

    df["record_id"] = df["record_id"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["record_id"]).reset_index(drop=True)

    for c in ["age", "mmse", "moca", "cdr_sum_of_boxes", "ApoE4", "Class_bvftd", "Class_LBD"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "sex" in df.columns:
        df["sex"] = _safe_encode_sex(df["sex"])

    for c in ["Class_bvftd", "Class_LBD"]:
        if c in df.columns:
            df[c] = (df[c].fillna(0) > 0).astype(int)

    return df


# =============================================================================
# File discovery helpers (copied verbatim from file 12)
# =============================================================================

def list_groups(base_path: str) -> List[str]:
    if not os.path.isdir(base_path):
        return []
    return sorted(d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)))


def list_stages(base_path: str, groups: Sequence[str]) -> List[str]:
    stages: set = set()
    for group in groups:
        gdir = os.path.join(base_path, group)
        if os.path.isdir(gdir):
            stages.update(d for d in os.listdir(gdir) if os.path.isdir(os.path.join(gdir, d)))
    return sorted(stages, key=lambda s: STAGE_ORDER.index(s) if s in STAGE_ORDER else 99)


# =============================================================================
# Statistical functions
# =============================================================================
# ponytail: 6_hep_group_comparison.py has no reusable top-level equivalents of
# these generic stats helpers (grepped for benjamini_hochberg/rank_biserial_r/
# bootstrap_median_ci/cohens_d/format_p/pval_to_stars — none exist there; its
# own significance testing is permutation_cluster_jitter_test /
# permutation_two_group_cluster_test, which operate on HEP waveforms across
# time, not on a single scalar metric across diagnosis groups like dashboard
# 16 needs). Kept here — this part of file 16 was not bespoke/broken, only
# the pickle-loading layer was. Only the loading layer is delegated below.

def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    n = len(ranked)
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0, 1)
    return out


def rank_biserial_r(a: np.ndarray, b: np.ndarray) -> float:
    """Non-parametric effect size matching the Mann-Whitney U statistic."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return np.nan
    u, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(2.0 * u / (n1 * n2) - 1.0)


def bootstrap_median_ci(data: np.ndarray, n_boot: int = 1000, ci: float = 0.95) -> "tuple[float, float]":
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(42)
    boot = np.array([np.median(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)


def format_p(p: Optional[float]) -> str:
    if p is None or not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def pval_to_stars(p: Optional[float]) -> str:
    if p is None or not np.isfinite(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# =============================================================================
# Metric extraction — reads the (patient_id, hep_data, times, ch_names,
# rpeaks, ecg_hep_data, ecg_ch_names, log_msg, flipped_channels) tuples
# returned by 6_hep_group_comparison.get_group_individuals, instead of the
# old (broken) PatientResult dataclass fields.
# =============================================================================

def extract_metric(hep_mod, individual: tuple, metric: str, t_window: tuple, selected_channels: list) -> float:
    (patient_id, hep_data, times, ch_names, rpeaks, ecg_hep_data, ecg_ch_names,
     log_msg, flipped_channels) = individual[:9]

    if metric == "N_epochs kept":
        # ponytail: get_group_individuals does not expose kept-vs-total epoch
        # counts (that split happens internally in compute_hep_avg and isn't
        # returned). Closest available proxy: number of detected R-peaks
        # feeding into the HEP average. Not faithful to the old "kept after
        # quality filtering" semantics — see summary.
        return float(len(rpeaks)) if rpeaks is not None else np.nan

    if metric == "ECG T-peak time":
        t_peak = hep_mod.find_ecg_t_peak_time(ecg_hep_data, times, t_window=t_window)
        return float(t_peak) * 1000 if t_peak is not None else np.nan  # ms

    # EEG metrics need channel index mapping
    if hep_data is None or times is None or ch_names is None:
        return np.nan
    ch_indices = [i for i, ch in enumerate(ch_names) if ch in selected_channels]
    if not ch_indices:
        return np.nan
    times_arr = np.asarray(times, dtype=float)
    t_mask = (times_arr >= t_window[0]) & (times_arr <= t_window[1])
    if not t_mask.any():
        return np.nan
    eeg_win = np.asarray(hep_data)[ch_indices][:, t_mask] * 1e6  # V -> µV
    if metric == "EEG T-wave amplitude (mean)":
        return float(np.nanmean(eeg_win))
    if metric == "EEG T-wave amplitude (peak)":
        return float(np.nanmax(np.abs(eeg_win)))
    if metric == "Signal quality score":
        # ponytail: no total-epochs-attempted count is returned either, so a
        # kept/total ratio cannot be faithfully reproduced. Not available —
        # excluded from the metric selectbox (see main()).
        return np.nan
    return np.nan


# =============================================================================
# Data loading — delegates entirely to 6_hep_group_comparison.get_group_individuals,
# which already does multicore ProcessPoolExecutor loading + its own on-disk
# per-(group,stage) cache (individuals_cache*.pkl next to the source pickles).
# No separate disk cache / checkpoint / session-state streaming is needed here.
# =============================================================================

def load_patient_data(
    hep_mod,
    groups: Sequence[str],
    stages: Sequence[str],
    force_rebuild: bool = False,
    apply_ica: bool = False,
    min_eeg_channels: Optional[int] = None,
) -> pd.DataFrame:
    """
    Calls get_group_individuals once per (group, stage) pair and flattens the
    results into a DataFrame of small (patient_id, tuple) rows — never holds
    raw EEG signals beyond what get_group_individuals itself returns.
    """
    rows: List[dict] = []
    for g in groups:
        for s in stages:
            individuals = hep_mod.get_group_individuals(
                g, s, BASE_PATH,
                test_run=False,
                recompute_cache=force_rebuild,
                apply_ica=apply_ica,
                parallel_workers=hep_mod.DEFAULT_PATIENT_WORKERS,
                min_eeg_channels=min_eeg_channels,
            )
            for ind in individuals:
                if ind is None or len(ind) == 0:
                    continue
                rows.append({
                    "group": g,
                    "stage": s,
                    "patient_id": ind[0],
                    "individual": ind,
                })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["group", "stage", "patient_id", "individual"])


@st.cache_data(ttl=3600, show_spinner=False)
def load_ehr_data(patient_ids: tuple) -> pd.DataFrame:
    """Load EHR diagnosis data for Harvard patients."""
    bdsp_ids: Dict[str, int] = {}
    for pid in patient_ids:
        bdsp = _pid_to_bdsp_ehr(pid)
        if bdsp is not None:
            bdsp_ids[pid] = bdsp
    if not bdsp_ids:
        return pd.DataFrame()
    target_ids = set(bdsp_ids.values())
    diag_map = _load_ehr_diagnosis_map(target_ids)
    rows = []
    for pid, bdsp in bdsp_ids.items():
        dx_names = diag_map.get(bdsp, [])
        categories = _assign_categories(dx_names)
        rows.append({"patient_id": pid, "bdsp_id": bdsp, "dx_names": dx_names, "categories": categories})
    return pd.DataFrame(rows)


def select_non_diagnosis_cohort(
    metric_df: pd.DataFrame,
    ehr_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return rows whose patient has no diagnosis in the available EHR mapping."""
    diagnosis_names_by_patient = (
        dict(zip(ehr_df["patient_id"], ehr_df["dx_names"]))
        if not ehr_df.empty else {}
    )
    selected = metric_df[
        metric_df["patient_id"].map(
            lambda patient_id: not diagnosis_names_by_patient.get(patient_id, [])
        )
    ].copy()
    selected["diagnosis_group"] = "No diagnosis"
    return selected


@st.cache_data(ttl=3600, show_spinner=False)
def load_ai_physiological_features(
    groups: tuple,
    stages: tuple,
) -> pd.DataFrame:
    """Load cached interpretable EEG/ECG features and pivot stages per subject."""
    if not AI_FEATURE_CACHE.exists():
        return pd.DataFrame()
    try:
        import pyarrow.parquet as pq
    except Exception:
        pq = None

    frames = []
    for path in sorted(AI_FEATURE_CACHE.glob("*.parquet")):
        try:
            if pq is not None:
                available = set(pq.ParquetFile(path).schema.names)
                columns = [
                    column for column in
                    ["group", "stage", "patient_id"] + AI_PHYSIOLOGICAL_FEATURES
                    if column in available
                ]
                frame = pd.read_parquet(path, columns=columns)
            else:
                frame = pd.read_parquet(path)
            if not {"group", "stage", "patient_id"}.issubset(frame.columns):
                continue
            frame = frame[
                frame["group"].isin(groups) & frame["stage"].isin(stages)
            ].copy()
            if frame.empty:
                continue
            for feature in AI_PHYSIOLOGICAL_FEATURES:
                if feature not in frame.columns:
                    frame[feature] = np.nan
                frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
            frames.append(
                frame[["patient_id", "stage"] + AI_PHYSIOLOGICAL_FEATURES]
            )
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["patient_id"] = combined["patient_id"].map(_canonical_patient_id)
    aggregated = combined.groupby(
        ["patient_id", "stage"], as_index=False
    )[AI_PHYSIOLOGICAL_FEATURES].mean()
    wide = aggregated.pivot(index="patient_id", columns="stage")
    wide.columns = [f"{stage}__{feature}" for feature, stage in wide.columns]
    return wide.sort_index()


def consolidate_ai_ehr(ehr_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse stage-specific EHR rows to one diagnosis record per subject."""
    if ehr_df is None or ehr_df.empty:
        return pd.DataFrame(columns=["patient_id", "categories", "dx_names"])
    rows = []
    work = ehr_df.copy()
    work["canonical_id"] = work["patient_id"].map(_canonical_patient_id)
    for patient_id, patient_rows in work.groupby("canonical_id"):
        categories = sorted({
            category
            for values in patient_rows["categories"]
            if isinstance(values, list)
            for category in values
        })
        diagnoses = sorted({
            diagnosis
            for values in patient_rows["dx_names"]
            if isinstance(values, list)
            for diagnosis in values
        })
        rows.append({
            "patient_id": patient_id,
            "categories": categories,
            "dx_names": diagnoses,
        })
    return pd.DataFrame(rows).set_index("patient_id")


def build_ai_target_labels(
    ehr_subjects: pd.DataFrame,
    target_kind: str,
    selected_diseases: Sequence[str],
    index_disease: Optional[str] = None,
) -> pd.Series:
    """Create mutually exclusive disease or exact comorbidity-subgroup labels."""
    labels: Dict[str, str] = {}
    selected = set(selected_diseases)
    for patient_id, row in ehr_subjects.iterrows():
        categories = set(row.get("categories", []))
        if target_kind == "Different diseases":
            matches = sorted(categories & selected)
            if len(matches) == 1:
                labels[str(patient_id)] = matches[0]
        else:
            if not index_disease or index_disease not in categories:
                continue
            other = sorted(categories - {index_disease})
            labels[str(patient_id)] = (
                f"{index_disease} only"
                if not other
                else f"{index_disease} + " + " + ".join(other)
            )
    return pd.Series(labels, name="diagnosis_label", dtype="object")


def _train_dense_autoencoder_features(
    x_train: np.ndarray,
    x_test: np.ndarray,
    x_all: np.ndarray,
    latent_dim: int,
    epochs: int,
) -> tuple:
    """Train an unsupervised dense autoencoder on training subjects only."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(42)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    np.random.seed(42)
    input_dim = x_train.shape[1]
    hidden_dim = max(16, min(128, input_dim // 2))
    latent_dim = max(2, min(int(latent_dim), hidden_dim, input_dim))

    class TabularAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, values):
            return self.decoder(self.encoder(values))

    model = TabularAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.MSELoss()
    train_tensor = torch.tensor(x_train, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(train_tensor, train_tensor),
        batch_size=min(64, max(8, len(train_tensor) // 4)),
        shuffle=True,
    )
    model.train()
    for _ in range(int(epochs)):
        for batch, target in loader:
            optimizer.zero_grad()
            loss = loss_function(model(batch), target)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        def encode(values):
            tensor = torch.tensor(values, dtype=torch.float32)
            return model.encoder(tensor).cpu().numpy()
        latent_train = encode(x_train)
        latent_test = encode(x_test)
        latent_all = encode(x_all)
    return latent_train, latent_test, latent_all


def train_ai_feature_models(
    feature_df: pd.DataFrame,
    labels: pd.Series,
    test_fraction: float,
    top_feature_count: int,
    latent_dim: int,
    autoencoder_epochs: int,
    n_estimators: int,
) -> dict:
    """Train leakage-controlled classifiers on engineered and autoencoder features."""
    common_ids = feature_df.index.intersection(labels.index)
    x_frame = feature_df.loc[common_ids].replace([np.inf, -np.inf], np.nan)
    y_text = labels.loc[common_ids].astype(str)
    missing = x_frame.isna().mean()
    usable_features = missing[missing <= 0.60].index.tolist()
    x_frame = x_frame[usable_features]
    variable = x_frame.nunique(dropna=True) > 1
    x_frame = x_frame.loc[:, variable]
    if len(x_frame) < 12 or x_frame.shape[1] < 2 or y_text.nunique() < 2:
        raise ValueError("Insufficient samples, classes, or usable features.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)
    indices = np.arange(len(x_frame))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_fraction, random_state=42, stratify=y,
    )
    y_train, y_test = y[train_idx], y[test_idx]
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(x_frame.iloc[train_idx])
    x_test = imputer.transform(x_frame.iloc[test_idx])
    x_all = imputer.transform(x_frame)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    x_all_scaled = scaler.transform(x_all)

    def fit_random_forest(train_values, test_values, names):
        model = RandomForestClassifier(
            n_estimators=int(n_estimators), random_state=42,
            class_weight="balanced_subsample",
            n_jobs=max(1, min(8, os.cpu_count() or 1)),
            min_samples_leaf=2,
        )
        model.fit(train_values, y_train)
        prediction = model.predict(test_values)
        return {
            "model": model,
            "prediction": prediction,
            "feature_names": list(names),
            "accuracy": float(accuracy_score(y_test, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
            "macro_f1": float(f1_score(y_test, prediction, average="macro")),
        }

    all_model = fit_random_forest(x_train, x_test, x_frame.columns)
    importance_order = np.argsort(all_model["model"].feature_importances_)[::-1]
    top_count = max(2, min(int(top_feature_count), len(importance_order)))
    top_indices = importance_order[:top_count]
    top_names = x_frame.columns[top_indices].tolist()
    top_model = fit_random_forest(
        x_train[:, top_indices], x_test[:, top_indices], top_names
    )

    latent_train, latent_test, latent_all = _train_dense_autoencoder_features(
        x_train_scaled, x_test_scaled, x_all_scaled,
        latent_dim=latent_dim, epochs=autoencoder_epochs,
    )
    latent_names = [f"AE latent {index + 1}" for index in range(latent_train.shape[1])]
    latent_model = fit_random_forest(latent_train, latent_test, latent_names)
    combined_model = fit_random_forest(
        np.column_stack([x_train[:, top_indices], latent_train]),
        np.column_stack([x_test[:, top_indices], latent_test]),
        top_names + latent_names,
    )

    correlations = np.corrcoef(x_all_scaled.T, latent_all.T)
    feature_latent_correlations = correlations[
        :x_all_scaled.shape[1], x_all_scaled.shape[1]:
    ]
    return {
        "models": {
            "All engineered features": all_model,
            "Top engineered features": top_model,
            "Autoencoder latent features": latent_model,
            "Top features + autoencoder": combined_model,
        },
        "y_test": y_test,
        "class_names": encoder.classes_.tolist(),
        "feature_frame": x_frame,
        "labels_text": y_text,
        "top_feature_names": top_names,
        "latent_all": latent_all,
        "latent_names": latent_names,
        "feature_latent_correlations": feature_latent_correlations,
        "all_feature_names": x_frame.columns.tolist(),
        "train_size": len(train_idx),
        "test_size": len(test_idx),
    }


# =============================================================================
# Statistical comparison
# =============================================================================

def compute_group_stats(grouped_df: pd.DataFrame, metric_col: str, ref_group: str) -> pd.DataFrame:
    """Per group: N, mean, std, median, CI, effect size, p-value vs ref, q-value."""
    groups = grouped_df["diagnosis_group"].unique().tolist()
    ref_vals = grouped_df.loc[grouped_df["diagnosis_group"] == ref_group, metric_col].dropna().values
    rows_list = []
    p_vals = []
    for g in groups:
        vals = grouped_df.loc[grouped_df["diagnosis_group"] == g, metric_col].dropna().values
        n = len(vals)
        mean = np.nanmean(vals) if n > 0 else np.nan
        std = np.nanstd(vals, ddof=1) if n > 1 else np.nan
        med = np.nanmedian(vals) if n > 0 else np.nan
        ci_lo, ci_hi = bootstrap_median_ci(vals) if n >= 2 else (np.nan, np.nan)
        if g == ref_group or len(ref_vals) < 2 or n < 2:
            p = np.nan
            eff = np.nan
        else:
            try:
                _, p = stats.mannwhitneyu(vals, ref_vals, alternative="two-sided")
                eff = rank_biserial_r(vals, ref_vals)
            except Exception:
                p = np.nan
                eff = np.nan
        p_vals.append(p)
        rows_list.append({
            "Group": g, "N": n, "Mean": mean, "Std": std, "Median": med,
            "CI_lo": ci_lo, "CI_hi": ci_hi, "Effect_size": eff, "p_value": p,
        })
    stat_df = pd.DataFrame(rows_list)
    # BH correction over finite p-values only
    q_arr = np.full(len(rows_list), np.nan)
    valid_idx = [i for i, p in enumerate(p_vals) if np.isfinite(p)]
    if valid_idx:
        valid_p = np.array([p_vals[i] for i in valid_idx])
        q_corr = benjamini_hochberg(valid_p)
        for ii, vi in enumerate(valid_idx):
            q_arr[vi] = q_corr[ii]
    stat_df["q_value"] = q_arr
    return stat_df


# =============================================================================
# Plotting
# =============================================================================

def _metric_unit(metric: str) -> str:
    if "amplitude" in metric.lower():
        return "µV"
    if "time" in metric.lower():
        return "ms"
    return ""


def plot_stage_comparison(metric_df: pd.DataFrame, stages: list, metric: str, grouping_mode: str) -> go.Figure:
    """Box plots per stage, groups on x-axis."""
    n_stages = len(stages)
    n_cols = min(2, n_stages)
    n_rows = (n_stages + n_cols - 1) // n_cols
    groups = sorted(metric_df["diagnosis_group"].unique())

    # Compute KW result per stage first so it can be merged into the subplot title
    # (a separate annotation at y=1.08 collides with Plotly's auto-placed subplot_titles).
    stage_title_list = []
    for stage in stages:
        stage_label = STAGE_LABELS.get(stage, stage)
        stage_df = metric_df[metric_df["stage"] == stage]
        group_vals = [
            stage_df[stage_df["diagnosis_group"] == g]["metric_value"].dropna().values
            for g in groups
            if len(stage_df[stage_df["diagnosis_group"] == g]) >= 2
        ]
        title = stage_label
        if len(group_vals) >= 2:
            try:
                _, kw_p = stats.kruskal(*group_vals)
                stars = pval_to_stars(kw_p)
                title = f"{stage_label}<br>KW {stars} (p={kw_p:.3f})"
            except Exception:
                pass
        stage_title_list.append(title)

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=stage_title_list,
        vertical_spacing=0.35 if n_rows > 1 else 0.15,
        horizontal_spacing=0.1,
    )
    unit = _metric_unit(metric)
    metric_label = f"{metric} ({unit})" if unit else metric

    for si, stage in enumerate(stages):
        row = si // n_cols + 1
        col = si % n_cols + 1
        stage_df = metric_df[metric_df["stage"] == stage]
        for gi, group in enumerate(groups):
            gdf = stage_df[stage_df["diagnosis_group"] == group]
            vals = gdf["metric_value"].dropna().values
            # unique patients in this group/stage, not row count
            n = gdf["patient_id"].nunique()
            color = PALETTE[gi % len(PALETTE)]
            fig.add_trace(go.Box(
                y=vals,
                name=f"{group} (N={n})",
                marker_color=color,
                line_color=color,
                boxmean="sd",
                showlegend=(si == 0),
                legendgroup=group,
            ), row=row, col=col)

    fig.update_yaxes(title_text=metric_label)
    fig.update_layout(
        height=420 * n_rows,
        title_text=f"HEP T-Wave Comparison by {grouping_mode}",
        title_font_size=16,
        font_size=12,
        legend_title="Group",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="black",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee", zeroline=True)
    return fig


def plot_cross_stage(metric_df: pd.DataFrame, stages: list, metric: str) -> go.Figure:
    """One trace per group, x=stages (mean ± bootstrap CI)."""
    groups = sorted(metric_df["diagnosis_group"].unique())
    stage_label_list = [STAGE_LABELS.get(s, s) for s in stages]
    unit = _metric_unit(metric)
    metric_label = f"{metric} ({unit})" if unit else metric
    fig = go.Figure()
    for gi, group in enumerate(groups):
        color = PALETTE[gi % len(PALETTE)]
        gdf = metric_df[metric_df["diagnosis_group"] == group]
        y_means, y_err_lo, y_err_hi = [], [], []
        for stage in stages:
            vals = gdf[gdf["stage"] == stage]["metric_value"].dropna().values
            if len(vals) >= 2:
                m = np.nanmean(vals)
                lo, hi = bootstrap_median_ci(vals)
                y_means.append(m)
                y_err_lo.append(m - lo if np.isfinite(lo) else 0)
                y_err_hi.append(hi - m if np.isfinite(hi) else 0)
            else:
                y_means.append(np.nan)
                y_err_lo.append(0)
                y_err_hi.append(0)
        fig.add_trace(go.Scatter(
            x=stage_label_list,
            y=y_means,
            error_y=dict(type="data", symmetric=False, array=y_err_hi, arrayminus=y_err_lo),
            mode="lines+markers",
            name=group,
            line=dict(color=color, width=2),
            marker=dict(size=8, color=color),
        ))
    fig.update_layout(
        title=f"HEP {metric} Across Sleep Stages by Group",
        title_font_size=16,
        font_size=12,
        xaxis_title="Sleep Stage",
        yaxis_title=metric_label,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="black",
        xaxis=dict(showgrid=True, gridcolor="#eee"),
        yaxis=dict(showgrid=True, gridcolor="#eee", zeroline=True),
        legend_title="Group",
    )
    return fig


def _comorbidity_stage_statistics(
    overview_df: pd.DataFrame,
    stages: Sequence[str],
    index_group: str,
    subgroup_order: Sequence[str],
) -> pd.DataFrame:
    """Per subgroup/stage: delta from all-index mean and p vs the index remainder."""
    rows = []
    for subgroup in subgroup_order:
        for stage in stages:
            main_stage = overview_df[
                (overview_df["diagnosis_group"] == index_group)
                & (overview_df["stage"] == stage)
            ].copy()
            subgroup_stage = overview_df[
                (overview_df["diagnosis_group"] == subgroup)
                & (overview_df["stage"] == stage)
            ].copy()
            main_vals = main_stage["metric_value"].dropna().to_numpy(dtype=float)
            subgroup_vals = subgroup_stage["metric_value"].dropna().to_numpy(dtype=float)
            if main_stage.empty or subgroup_stage.empty:
                rows.append({
                    "Group": subgroup,
                    "Stage": stage,
                    "N": 0,
                    "Mean": np.nan,
                    "Delta_vs_index_mean": np.nan,
                    "p_value": np.nan,
                })
                continue
            subgroup_ids = set(
                subgroup_stage["patient_id"].map(_canonical_patient_id).astype(str)
            )
            rest_stage = main_stage[
                ~main_stage["patient_id"].map(_canonical_patient_id).astype(str).isin(subgroup_ids)
            ]
            rest_vals = rest_stage["metric_value"].dropna().to_numpy(dtype=float)
            if len(subgroup_vals) >= 2 and len(rest_vals) >= 2:
                try:
                    _, p_value = stats.mannwhitneyu(
                        subgroup_vals, rest_vals, alternative="two-sided"
                    )
                except Exception:
                    p_value = np.nan
            else:
                p_value = np.nan
            rows.append({
                "Group": subgroup,
                "Stage": stage,
                "N": subgroup_stage["patient_id"].nunique(),
                "Mean": np.nanmean(subgroup_vals) if len(subgroup_vals) else np.nan,
                "Delta_vs_index_mean": (
                    np.nanmean(subgroup_vals) - np.nanmean(main_vals)
                    if len(subgroup_vals) and len(main_vals) else np.nan
                ),
                "p_value": p_value,
            })
    stat_df = pd.DataFrame(rows)
    stat_df["q_value"] = np.nan
    finite = stat_df["p_value"].notna() & np.isfinite(stat_df["p_value"])
    if finite.any():
        stat_df.loc[finite, "q_value"] = benjamini_hochberg(
            stat_df.loc[finite, "p_value"].to_numpy(dtype=float)
        )
    return stat_df


def plot_comorbidity_overview(
    overview_df: pd.DataFrame,
    stages: Sequence[str],
    metric: str,
    index_group: str,
    subgroup_order: Sequence[str],
) -> go.Figure:
    """One interactive figure: subgroup p-values above index/subgroup stage lines."""
    stage_labels = [STAGE_LABELS.get(stage, stage) for stage in stages]
    unit = _metric_unit(metric)
    metric_label = f"{metric} ({unit})" if unit else metric
    subgroup_order = [g for g in subgroup_order if g in set(overview_df["diagnosis_group"])]
    stats_df = _comorbidity_stage_statistics(
        overview_df, stages, index_group, subgroup_order
    )

    heatmap_z = []
    heatmap_text = []
    for subgroup in subgroup_order:
        row_z = []
        row_text = []
        for stage in stages:
            stat_row = stats_df[
                (stats_df["Group"] == subgroup) & (stats_df["Stage"] == stage)
            ]
            if stat_row.empty:
                row_z.append(np.nan)
                row_text.append("NA")
                continue
            item = stat_row.iloc[0]
            delta = item["Delta_vs_index_mean"]
            p_value = item["p_value"]
            q_value = item["q_value"]
            row_z.append(delta)
            if np.isfinite(delta):
                row_text.append(
                    f"{pval_to_stars(p_value)}<br>"
                    f"d={delta:+.3f}<br>"
                    f"p={format_p(p_value)}<br>"
                    f"q={format_p(q_value)}"
                )
            else:
                row_text.append("NA")
        heatmap_z.append(row_z)
        heatmap_text.append(row_text)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.36, 0.64],
        vertical_spacing=0.08,
        specs=[[{"type": "heatmap"}], [{"type": "xy"}]],
        subplot_titles=(
            "Subgroup statistics: delta from all-index mean; p/q vs remaining index patients",
            "Index diagnosis and largest exact comorbidity subgroups",
        ),
    )

    if subgroup_order:
        finite_deltas = np.abs(np.asarray(heatmap_z, dtype=float))
        finite_deltas = finite_deltas[np.isfinite(finite_deltas)]
        max_abs_delta = float(np.max(finite_deltas)) if finite_deltas.size else 1.0
        max_abs_delta = max(max_abs_delta, 1.0e-12)
        fig.add_trace(go.Heatmap(
            z=heatmap_z,
            x=stage_labels,
            y=subgroup_order,
            text=heatmap_text,
            texttemplate="%{text}",
            colorscale="RdBu_r",
            zmid=0,
            zmin=-max_abs_delta,
            zmax=max_abs_delta,
            colorbar=dict(title="Delta", len=0.34, y=0.80),
            hovertemplate=(
                "Subgroup: %{y}<br>Stage: %{x}<br>%{text}"
                "<extra></extra>"
            ),
        ), row=1, col=1)

    groups_for_lines = [index_group] + list(subgroup_order)
    stats_lookup = {
        (row["Group"], row["Stage"]): row
        for _, row in stats_df.iterrows()
    }
    for group_index, group in enumerate(groups_for_lines):
        gdf = overview_df[overview_df["diagnosis_group"] == group]
        y_values = []
        err_plus = []
        err_minus = []
        custom = []
        for stage in stages:
            vals = gdf[gdf["stage"] == stage]["metric_value"].dropna().to_numpy(dtype=float)
            n_patients = gdf[gdf["stage"] == stage]["patient_id"].nunique()
            if len(vals) >= 2:
                mean_val = float(np.nanmean(vals))
                lo, hi = bootstrap_median_ci(vals)
                y_values.append(mean_val)
                err_minus.append(mean_val - lo if np.isfinite(lo) else 0)
                err_plus.append(hi - mean_val if np.isfinite(hi) else 0)
            elif len(vals) == 1:
                y_values.append(float(vals[0]))
                err_minus.append(0)
                err_plus.append(0)
            else:
                y_values.append(np.nan)
                err_minus.append(0)
                err_plus.append(0)

            stat_row = stats_lookup.get((group, stage))
            if group == index_group or stat_row is None:
                custom.append([n_patients, np.nan, np.nan, np.nan])
            else:
                custom.append([
                    n_patients,
                    stat_row["Delta_vs_index_mean"],
                    stat_row["p_value"],
                    stat_row["q_value"],
                ])

        is_index = group == index_group
        color = "#111111" if is_index else PALETTE[(group_index - 1) % len(PALETTE)]
        line = dict(color=color, width=4 if is_index else 2.2)
        if not is_index:
            line["dash"] = "solid"
        fig.add_trace(go.Scatter(
            x=stage_labels,
            y=y_values,
            mode="lines+markers",
            name=group,
            legendgroup=group,
            customdata=np.asarray(custom, dtype=float),
            error_y=dict(
                type="data",
                symmetric=False,
                array=err_plus,
                arrayminus=err_minus,
                thickness=1,
                width=3,
            ),
            line=line,
            marker=dict(size=10 if is_index else 8, color=color),
            hovertemplate=(
                "Group: %{fullData.name}<br>"
                "Stage: %{x}<br>"
                f"{metric}: " + "%{y:.4f}<br>"
                "N: %{customdata[0]:.0f}<br>"
                "Delta vs index: %{customdata[1]:+.4f}<br>"
                "p vs index remainder: %{customdata[2]:.4g}<br>"
                "q: %{customdata[3]:.4g}<extra></extra>"
            ) if not is_index else (
                "Group: %{fullData.name}<br>"
                "Stage: %{x}<br>"
                f"{metric}: " + "%{y:.4f}<br>"
                "N: %{customdata[0]:.0f}<extra></extra>"
            ),
        ), row=2, col=1)

    fig.update_yaxes(title_text="Subgroup", row=1, col=1, automargin=True)
    fig.update_yaxes(title_text=metric_label, row=2, col=1, showgrid=True, gridcolor="#eee")
    fig.update_xaxes(title_text="Sleep Stage", row=2, col=1, showgrid=True, gridcolor="#eee")
    fig.update_layout(
        height=max(700, 520 + 34 * max(1, len(subgroup_order))),
        title=f"Comorbidity Contribution Overview | {index_group}",
        title_font_size=16,
        font_size=12,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="black",
        legend=dict(
            title="Click legend items to hide/show lines",
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        hovermode="x unified",
        uirevision=f"comorbidity-overview-{index_group}-{metric}-{tuple(stages)}",
    )
    return fig


def plot_heatmap(metric_df: pd.DataFrame, stages: list, ref_group: str) -> go.Figure:
    """Group × Stage heatmap of mean metric, with MW p-value stars vs ref."""
    groups = sorted(metric_df["diagnosis_group"].unique())
    stage_label_list = [STAGE_LABELS.get(s, s) for s in stages]
    ref_data = {
        stage: metric_df[
            (metric_df["diagnosis_group"] == ref_group) & (metric_df["stage"] == stage)
        ]["metric_value"].dropna().values
        for stage in stages
    }
    z_vals = []
    text_vals = []
    for group in groups:
        row_z = []
        row_t = []
        for stage in stages:
            vals = metric_df[
                (metric_df["diagnosis_group"] == group) & (metric_df["stage"] == stage)
            ]["metric_value"].dropna().values
            mean_val = np.nanmean(vals) if len(vals) > 0 else np.nan
            row_z.append(mean_val)
            ref_vals = ref_data.get(stage, np.array([]))
            if len(vals) >= 2 and len(ref_vals) >= 2 and group != ref_group:
                try:
                    _, p = stats.mannwhitneyu(vals, ref_vals, alternative="two-sided")
                    stars = pval_to_stars(p)
                except Exception:
                    stars = ""
            else:
                stars = ""
            row_t.append(f"{mean_val:.3f}<br>{stars}" if np.isfinite(mean_val) else "NA")
        z_vals.append(row_z)
        text_vals.append(row_t)
    fig = go.Figure(go.Heatmap(
        z=z_vals,
        x=stage_label_list,
        y=groups,
        text=text_vals,
        texttemplate="%{text}",
        colorscale="RdBu_r",
        colorbar_title="Mean",
        hoverongaps=False,
    ))
    fig.update_layout(
        title=f"Group × Stage Heatmap (ref: {ref_group})",
        title_font_size=16,
        font_size=12,
        xaxis_title="Sleep Stage",
        yaxis_title="Group",
        height=max(300, 60 * len(groups) + 100),
    )
    return fig


# =============================================================================
# Pairwise HEP waveform comparison
# =============================================================================

def _pointwise_t_components(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    paired: bool,
) -> dict:
    """Compute the exact numerator, standard error, and T statistic per time point."""
    a = np.asarray(matrix_a, dtype=float)
    b = np.asarray(matrix_b, dtype=float)
    mean_a = np.nanmean(a, axis=0)
    mean_b = np.nanmean(b, axis=0)
    sem_a = stats.sem(a, axis=0, nan_policy="omit")
    sem_b = stats.sem(b, axis=0, nan_policy="omit")
    mean_difference = mean_a - mean_b
    if paired:
        differences = a - b
        difference_se = stats.sem(differences, axis=0, nan_policy="omit")
    else:
        difference_se = np.sqrt(
            np.nanvar(a, axis=0, ddof=1) / a.shape[0]
            + np.nanvar(b, axis=0, ddof=1) / b.shape[0]
        )

    finite_scale_values = np.concatenate([
        np.abs(mean_a[np.isfinite(mean_a)]),
        np.abs(mean_b[np.isfinite(mean_b)]),
    ])
    signal_scale = (
        float(np.nanmax(finite_scale_values)) if finite_scale_values.size else 1.0
    )
    numerical_floor = np.finfo(float).eps * max(1.0, signal_scale) * 100
    valid_se = np.isfinite(difference_se) & (difference_se > numerical_floor)
    t_stat = np.full(mean_difference.shape, np.nan, dtype=float)
    np.divide(mean_difference, difference_se, out=t_stat, where=valid_se)
    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "sem_a": sem_a,
        "sem_b": sem_b,
        "mean_difference": mean_difference,
        "difference_se": difference_se,
        "t_stat": t_stat,
        "valid_se": valid_se,
    }

def _patient_hep_trace(
    individual: tuple,
    selected_channels: Sequence[str],
    zscore_subjects: bool = False,
) -> Optional[tuple]:
    """Return one channel-averaged HEP trace (µV) and its time axis."""
    if individual is None or len(individual) < 4:
        return None
    hep_data, times, ch_names = individual[1], individual[2], individual[3]
    if hep_data is None or times is None or ch_names is None:
        return None
    selected_channel_keys = {str(channel).lower() for channel in selected_channels}
    indices = [
        i for i, ch in enumerate(ch_names)
        if str(ch).lower() in selected_channel_keys
    ]
    if not indices:
        return None
    data = np.asarray(hep_data, dtype=float)
    times_arr = np.asarray(times, dtype=float)
    if data.ndim != 2 or times_arr.ndim != 1 or data.shape[1] != len(times_arr):
        return None
    trace = np.nanmean(data[indices], axis=0) * 1e6
    finite = np.isfinite(trace) & np.isfinite(times_arr)
    if finite.sum() < 3:
        return None
    trace, times_arr = trace[finite], times_arr[finite]
    order = np.argsort(times_arr)
    trace, times_arr = trace[order], times_arr[order]
    unique_times, unique_idx = np.unique(times_arr, return_index=True)
    trace, times_arr = trace[unique_idx], unique_times
    if zscore_subjects:
        sd = float(np.nanstd(trace, ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            return None
        trace = (trace - np.nanmean(trace)) / sd
    return trace, times_arr


def _patient_channel_hep_trace(
    individual: tuple,
    channel: str,
    zscore_subjects: bool = False,
) -> Optional[tuple]:
    """Return one patient's HEP trace for a single EEG electrode."""
    if individual is None or len(individual) < 4:
        return None
    hep_data, times, ch_names = individual[1], individual[2], individual[3]
    if hep_data is None or times is None or ch_names is None:
        return None
    channel_indices = {
        str(name).lower(): index for index, name in enumerate(ch_names)
    }
    channel_index = channel_indices.get(str(channel).lower())
    if channel_index is None:
        return None
    data = np.asarray(hep_data, dtype=float)
    times_arr = np.asarray(times, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(times_arr):
        return None
    trace = data[channel_index] * 1e6
    finite = np.isfinite(trace) & np.isfinite(times_arr)
    if finite.sum() < 3:
        return None
    trace, times_arr = trace[finite], times_arr[finite]
    order = np.argsort(times_arr)
    trace, times_arr = trace[order], times_arr[order]
    unique_times, unique_idx = np.unique(times_arr, return_index=True)
    trace, times_arr = trace[unique_idx], unique_times
    if zscore_subjects:
        sd = float(np.nanstd(trace, ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            return None
        trace = (trace - np.nanmean(trace)) / sd
    return trace, times_arr


def _canonical_patient_id(patient_id: Any) -> str:
    """Remove stage/segment suffixes so repeated sleep stages match by subject."""
    value = str(patient_id)
    bdsp_match = re.search(r"(I\d{13})", value)
    if bdsp_match:
        return bdsp_match.group(1)
    cobrad_id = _extract_cobrad_id(value)
    return cobrad_id or value


def _collect_hep_traces(
    grouped_df: pd.DataFrame,
    diagnosis_group: str,
    stage: str,
    selected_channels: Sequence[str],
    zscore_subjects: bool,
) -> Dict[str, tuple]:
    """Collect at most one valid waveform per patient for a group/stage cell."""
    subset = grouped_df[
        (grouped_df["diagnosis_group"] == diagnosis_group)
        & (grouped_df["stage"] == stage)
    ]
    traces: Dict[str, tuple] = {}
    for _, row in subset.iterrows():
        patient_id = _canonical_patient_id(row["patient_id"])
        if patient_id in traces:
            continue
        result = _patient_hep_trace(row.get("individual"), selected_channels, zscore_subjects)
        if result is not None:
            traces[patient_id] = result
    return traces


def _collect_channel_hep_traces(
    grouped_df: pd.DataFrame,
    diagnosis_group: str,
    stage: str,
    channel: str,
    zscore_subjects: bool,
) -> Dict[str, tuple]:
    """Collect one electrode's HEP trace for each patient in a group/stage cell."""
    subset = grouped_df[
        (grouped_df["diagnosis_group"] == diagnosis_group)
        & (grouped_df["stage"] == stage)
    ]
    traces: Dict[str, tuple] = {}
    for _, row in subset.iterrows():
        patient_id = _canonical_patient_id(row["patient_id"])
        if patient_id in traces:
            continue
        result = _patient_channel_hep_trace(
            row.get("individual"), channel, zscore_subjects
        )
        if result is not None:
            traces[patient_id] = result
    return traces


def _align_trace_maps(
    traces_a: Dict[str, tuple],
    traces_b: Dict[str, tuple],
    paired: bool,
) -> Optional[tuple]:
    """Interpolate two collections onto a shared observed time grid."""
    if paired:
        patient_ids = sorted(set(traces_a) & set(traces_b))
        traces_a = {pid: traces_a[pid] for pid in patient_ids}
        traces_b = {pid: traces_b[pid] for pid in patient_ids}
    if not traces_a or not traces_b:
        return None

    all_items = list(traces_a.values()) + list(traces_b.values())
    overlap_start = max(float(times[0]) for _, times in all_items)
    overlap_end = min(float(times[-1]) for _, times in all_items)
    if overlap_start >= overlap_end:
        return None

    # Use the coarsest observed time grid to avoid inventing extra samples.
    ref_trace, ref_times = min(
        all_items,
        key=lambda item: np.sum((item[1] >= overlap_start) & (item[1] <= overlap_end)),
    )
    del ref_trace
    common_times = ref_times[(ref_times >= overlap_start) & (ref_times <= overlap_end)]
    if len(common_times) < 3:
        return None

    def stack(trace_map: Dict[str, tuple]) -> np.ndarray:
        return np.vstack([
            np.interp(common_times, times, trace)
            for trace, times in trace_map.values()
        ])

    return stack(traces_a), stack(traces_b), common_times, list(traces_a), list(traces_b)


def _electrode_patient_coverage_threshold(n_patients: int) -> int:
    """Minimum patient count required for a per-electrode contrast."""
    if n_patients <= 0:
        return 0
    return min(
        int(n_patients),
        max(
            ELECTRODE_MIN_MATCHED_PATIENTS,
            int(np.ceil(ELECTRODE_MIN_PATIENT_COVERAGE_FRACTION * n_patients)),
        ),
    )


def _patient_channel_sets(
    grouped_df: pd.DataFrame,
    diagnosis_group: str,
    stage: str,
) -> Dict[str, set]:
    """Map canonical patient IDs to the EEG channels available in one cell."""
    subset = grouped_df[
        (grouped_df["diagnosis_group"] == diagnosis_group)
        & (grouped_df["stage"] == stage)
    ]
    patient_channels: Dict[str, set] = {}
    for _, row in subset.iterrows():
        individual = row.get("individual")
        if individual is None or len(individual) <= 3 or individual[3] is None:
            continue
        patient_id = _canonical_patient_id(row["patient_id"])
        patient_channels.setdefault(patient_id, set()).update(
            str(channel).lower() for channel in individual[3]
        )
    return patient_channels


def _channels_meeting_comparison_coverage(
    grouped_df: pd.DataFrame,
    group_a: str,
    stage_a: str,
    group_b: str,
    stage_b: str,
    channels: Sequence[str],
    paired: bool,
) -> List[str]:
    """Keep channels represented in at least 5% of both comparison cells."""
    channels = list(dict.fromkeys(channels))
    patient_channels_a = _patient_channel_sets(grouped_df, group_a, stage_a)
    patient_channels_b = _patient_channel_sets(grouped_df, group_b, stage_b)
    if paired:
        patient_ids_a = patient_ids_b = sorted(
            set(patient_channels_a) & set(patient_channels_b)
        )
    else:
        patient_ids_a = list(patient_channels_a)
        patient_ids_b = list(patient_channels_b)
    required_a = _electrode_patient_coverage_threshold(len(patient_ids_a))
    required_b = _electrode_patient_coverage_threshold(len(patient_ids_b))
    if paired:
        return [
            channel for channel in channels
            if sum(
                str(channel).lower() in patient_channels_a[pid]
                and str(channel).lower() in patient_channels_b[pid]
                for pid in patient_ids_a
            ) >= max(required_a, required_b)
        ]
    return [
        channel for channel in channels
        if sum(str(channel).lower() in patient_channels_a[pid] for pid in patient_ids_a) >= required_a
        and sum(str(channel).lower() in patient_channels_b[pid] for pid in patient_ids_b) >= required_b
    ]


def cluster_permutation_contrast(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    times: np.ndarray,
    paired: bool,
    n_permutations: int = 200,
    cluster_alpha: float = PAIRWISE_CLUSTER_ALPHA,
    random_seed: int = 42,
) -> tuple:
    """Cluster-mass permutation test for independent or paired HEP traces."""
    a = np.asarray(matrix_a, dtype=float)
    b = np.asarray(matrix_b, dtype=float)
    if paired:
        if a.shape != b.shape or a.shape[0] < 2:
            raise ValueError("Paired contrasts require at least two matched patients.")
        differences = a - b
        degrees_freedom = a.shape[0] - 1
    else:
        if a.shape[0] < 2 or b.shape[0] < 2:
            raise ValueError("Independent contrasts require at least two patients per group.")
        degrees_freedom = a.shape[0] + b.shape[0] - 2

    t_components = _pointwise_t_components(a, b, paired)
    t_obs = np.nan_to_num(
        t_components["t_stat"], nan=0.0, posinf=0.0, neginf=0.0
    )
    t_threshold = stats.t.ppf(1 - cluster_alpha / 2, df=max(1, degrees_freedom))
    observed_labels, n_clusters = connected_components(np.abs(t_obs) > t_threshold)
    observed_masses = np.array([
        np.sum(np.abs(t_obs[observed_labels == cluster_id]))
        for cluster_id in range(1, n_clusters + 1)
    ])

    rng = np.random.default_rng(random_seed)
    null_max_masses = np.zeros(int(n_permutations), dtype=float)
    pooled = np.vstack([a, b]) if not paired else None
    for permutation in range(int(n_permutations)):
        if paired:
            signs = rng.choice((-1.0, 1.0), size=(a.shape[0], 1))
            signed_differences = differences * signs
            null_mean = np.nanmean(signed_differences, axis=0)
            null_se = stats.sem(signed_differences, axis=0, nan_policy="omit")
            t_null = np.divide(
                null_mean, null_se,
                out=np.full(null_mean.shape, np.nan, dtype=float),
                where=np.isfinite(null_se) & (null_se > 0),
            )
        else:
            permuted = pooled[rng.permutation(len(pooled))]
            t_null = _pointwise_t_components(
                permuted[:len(a)], permuted[len(a):], paired=False
            )["t_stat"]
        t_null = np.nan_to_num(t_null, nan=0.0, posinf=0.0, neginf=0.0)
        null_labels, n_null = connected_components(np.abs(t_null) > t_threshold)
        if n_null:
            null_max_masses[permutation] = max(
                np.sum(np.abs(t_null[null_labels == cluster_id]))
                for cluster_id in range(1, n_null + 1)
            )

    clusters = []
    for cluster_idx, mass in enumerate(observed_masses, start=1):
        indices = np.flatnonzero(observed_labels == cluster_idx)
        p_value = float((np.sum(null_max_masses >= mass) + 1) / (n_permutations + 1))
        clusters.append({
            "start": float(times[indices[0]]),
            "end": float(times[indices[-1]]),
            "p_value": p_value,
            "significant": p_value < cluster_alpha,
            "direction": "A > B" if np.nanmean(t_obs[indices]) > 0 else "B > A",
            "peak_t": float(t_obs[indices[np.argmax(np.abs(t_obs[indices]))]]),
            "cluster_mass": float(mass),
        })
    contrast_p = min((row["p_value"] for row in clusters), default=1.0)
    return clusters, t_obs, float(t_threshold), contrast_p


def prepare_waveform_contrast(
    grouped_df: pd.DataFrame,
    group_a: str,
    stage_a: str,
    group_b: str,
    stage_b: str,
    selected_channels: Sequence[str],
    zscore_subjects: bool,
    n_permutations: int,
    paired: bool,
) -> Optional[dict]:
    """Build aligned matrices and run one diagnosis or sleep-stage contrast."""
    selected_channels = _channels_meeting_comparison_coverage(
        grouped_df,
        group_a, stage_a,
        group_b, stage_b,
        selected_channels,
        paired,
    )
    if not selected_channels:
        return None
    traces_a = _collect_hep_traces(
        grouped_df, group_a, stage_a, selected_channels, zscore_subjects
    )
    traces_b = _collect_hep_traces(
        grouped_df, group_b, stage_b, selected_channels, zscore_subjects
    )
    overlap_removed = 0
    if not paired:
        # Non-exclusive diagnosis modes can assign one patient to both groups.
        # Removing those patients preserves the independent-samples test.
        shared = set(traces_a) & set(traces_b)
        overlap_removed = len(shared)
        traces_a = {pid: value for pid, value in traces_a.items() if pid not in shared}
        traces_b = {pid: value for pid, value in traces_b.items() if pid not in shared}

    aligned = _align_trace_maps(traces_a, traces_b, paired=paired)
    if aligned is None:
        return None
    matrix_a, matrix_b, times, patient_ids_a, patient_ids_b = aligned
    if matrix_a.shape[0] < 2 or matrix_b.shape[0] < 2:
        return None
    clusters, t_stat, t_threshold, contrast_p = cluster_permutation_contrast(
        matrix_a, matrix_b, times, paired=paired,
        n_permutations=n_permutations,
    )
    return {
        "matrix_a": matrix_a,
        "matrix_b": matrix_b,
        "channels_used": list(selected_channels),
        "times": times,
        "patient_ids_a": patient_ids_a,
        "patient_ids_b": patient_ids_b,
        "clusters": clusters,
        "t_stat": t_stat,
        "t_threshold": t_threshold,
        "contrast_p": contrast_p,
        "overlap_removed": overlap_removed,
        "paired": paired,
    }


def prepare_electrode_contrasts(
    grouped_df: pd.DataFrame,
    group_a: str,
    stage_a: str,
    group_b: str,
    stage_b: str,
    channels: Sequence[str],
    zscore_subjects: bool,
    n_permutations: int,
    paired: bool,
    progress_callback=None,
) -> Dict[str, dict]:
    """Run the pairwise waveform contrast independently for every electrode."""
    channel_results: Dict[str, dict] = {}
    channels = _channels_meeting_comparison_coverage(
        grouped_df,
        group_a, stage_a,
        group_b, stage_b,
        channels,
        paired,
    )

    reference_traces_a = _collect_hep_traces(
        grouped_df, group_a, stage_a, channels, zscore_subjects
    )
    reference_traces_b = _collect_hep_traces(
        grouped_df, group_b, stage_b, channels, zscore_subjects
    )
    shared_reference_patients: set = set()
    if not paired:
        shared_reference_patients = set(reference_traces_a) & set(reference_traces_b)
        reference_traces_a = {
            pid: value for pid, value in reference_traces_a.items()
            if pid not in shared_reference_patients
        }
        reference_traces_b = {
            pid: value for pid, value in reference_traces_b.items()
            if pid not in shared_reference_patients
        }
    reference_aligned = _align_trace_maps(
        reference_traces_a, reference_traces_b, paired=paired
    )
    if reference_aligned is not None:
        _, _, _, reference_patient_ids_a, reference_patient_ids_b = reference_aligned
    elif paired:
        matched = sorted(set(reference_traces_a) & set(reference_traces_b))
        reference_patient_ids_a = matched
        reference_patient_ids_b = matched
    else:
        reference_patient_ids_a = list(reference_traces_a)
        reference_patient_ids_b = list(reference_traces_b)
    denominator_a = len(reference_patient_ids_a)
    denominator_b = len(reference_patient_ids_b)
    required_a = _electrode_patient_coverage_threshold(denominator_a)
    required_b = _electrode_patient_coverage_threshold(denominator_b)

    for channel_index, channel in enumerate(channels):
        if progress_callback is not None:
            progress_callback(channel_index, len(channels), channel)
        traces_a = _collect_channel_hep_traces(
            grouped_df, group_a, stage_a, channel, zscore_subjects
        )
        traces_b = _collect_channel_hep_traces(
            grouped_df, group_b, stage_b, channel, zscore_subjects
        )
        overlap_removed = 0
        if not paired:
            overlap_removed = len(
                (set(traces_a) | set(traces_b)) & shared_reference_patients
            )
            traces_a = {
                pid: value for pid, value in traces_a.items()
                if pid not in shared_reference_patients
            }
            traces_b = {
                pid: value for pid, value in traces_b.items()
                if pid not in shared_reference_patients
            }
        aligned = _align_trace_maps(traces_a, traces_b, paired=paired)
        if aligned is None:
            continue
        matrix_a, matrix_b, times, patient_ids_a, patient_ids_b = aligned
        if matrix_a.shape[0] < 2 or matrix_b.shape[0] < 2:
            continue
        n_a, n_b = len(patient_ids_a), len(patient_ids_b)
        coverage_a = n_a / denominator_a if denominator_a else np.nan
        coverage_b = n_b / denominator_b if denominator_b else np.nan
        if n_a < required_a or n_b < required_b:
            continue
        clusters, t_stat, t_threshold, contrast_p = cluster_permutation_contrast(
            matrix_a, matrix_b, times, paired=paired,
            n_permutations=n_permutations,
        )
        channel_results[channel] = {
            "matrix_a": matrix_a,
            "matrix_b": matrix_b,
            "times": times,
            "patient_ids_a": patient_ids_a,
            "patient_ids_b": patient_ids_b,
            "coverage_denominator_a": denominator_a,
            "coverage_denominator_b": denominator_b,
            "coverage_fraction_a": float(coverage_a),
            "coverage_fraction_b": float(coverage_b),
            "coverage_required_a": required_a,
            "coverage_required_b": required_b,
            "clusters": clusters,
            "t_stat": t_stat,
            "t_threshold": t_threshold,
            "contrast_p": contrast_p,
            "overlap_removed": overlap_removed,
            "paired": paired,
        }

    if channel_results:
        channels_with_data = list(channel_results)
        p_values = np.array([
            channel_results[channel]["contrast_p"]
            for channel in channels_with_data
        ], dtype=float)
        q_values = benjamini_hochberg(p_values)
        for channel, q_value in zip(channels_with_data, q_values):
            result = channel_results[channel]
            result["q_value"] = float(q_value)
            result["cluster_significant"] = any(
                cluster["significant"] for cluster in result["clusters"]
            )
            result["fdr_significant"] = bool(
                result["cluster_significant"] and q_value < 0.05
            )
    return channel_results


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    color = hex_color.lstrip("#")
    red, green, blue = (int(color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def plot_waveform_contrast(
    result: dict,
    label_a: str,
    label_b: str,
    zscore_subjects: bool,
    ica_cleaned: bool = False,
) -> go.Figure:
    """Mean HEPs, their exact T-test numerator, and cluster-forming T statistics."""
    a, b, times = result["matrix_a"], result["matrix_b"], result["times"]
    t_components = _pointwise_t_components(a, b, result["paired"])
    mean_a, mean_b = t_components["mean_a"], t_components["mean_b"]
    sem_a, sem_b = t_components["sem_a"], t_components["sem_b"]
    difference = t_components["mean_difference"]
    difference_se = t_components["difference_se"]
    # Keep undefined zero-variance samples as gaps in the visual instead of
    # drawing artificial plunges to T=0. The test treats those samples as
    # non-cluster-forming.
    t_stat = t_components["t_stat"]
    t_threshold = result["t_threshold"]
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.46, 0.25, 0.29], vertical_spacing=0.08,
        subplot_titles=(
            "Mean HEP ± SEM",
            "Mean HEP difference (A − B): numerator of the T statistic",
            (
                "Observed T = mean difference / standard error "
                f"(two-sided p < {PAIRWISE_CLUSTER_ALPHA:.2f} threshold)"
            ),
        ),
    )

    for mean, sem, label_text, color in (
        (mean_a, sem_a, label_a, PALETTE[0]),
        (mean_b, sem_b, label_b, PALETTE[1]),
    ):
        fig.add_trace(go.Scatter(
            x=times, y=mean - sem, mode="lines", line=dict(width=0),
            hoverinfo="skip", showlegend=False, legendgroup=label_text,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=times, y=mean + sem, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=_hex_to_rgba(color, 0.18),
            hoverinfo="skip", showlegend=False, legendgroup=label_text,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=times, y=mean, mode="lines", name=f"{label_text} mean",
            line=dict(color=color, width=2.5), legendgroup=label_text,
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=times, y=difference, mode="lines", name="Mean difference (A − B)",
        line=dict(color="#222222", width=2.2),
        fill="tozeroy", fillcolor="rgba(50,50,50,0.10)",
        customdata=np.column_stack([difference_se, t_stat]),
        hovertemplate=(
            "Time: %{x:.3f} s<br>Mean A − B: %{y:.4f}<br>"
            "SE(A − B): %{customdata[0]:.4f}<br>"
            "T = difference / SE: %{customdata[1]:.3f}<extra></extra>"
        ),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=times, y=t_stat, mode="lines", name="Observed T statistic",
        line=dict(color="#6F42C1", width=2),
    ), row=3, col=1)
    superthreshold_t = np.where(np.abs(t_stat) > t_threshold, t_stat, np.nan)
    fig.add_trace(go.Scatter(
        x=times, y=superthreshold_t, mode="lines",
        name="T exceeds cluster threshold",
        line=dict(color="#C0392B", width=3.5),
    ), row=3, col=1)
    fig.add_hline(
        y=t_threshold, line_color="#C0392B", line_dash="dash", line_width=1.5,
        annotation_text=f"+T threshold = {t_threshold:.2f}",
        annotation_position="top right", row=3, col=1,
    )
    fig.add_hline(
        y=-t_threshold, line_color="#C0392B", line_dash="dash", line_width=1.5,
        annotation_text=f"−T threshold = {-t_threshold:.2f}",
        annotation_position="bottom right", row=3, col=1,
    )

    significant_clusters = [
        cluster for cluster in result["clusters"] if cluster["significant"]
    ]
    for cluster_number, cluster in enumerate(significant_clusters, start=1):
        annotation = f"C{cluster_number}"
        fig.add_vrect(
            x0=cluster["start"], x1=cluster["end"], fillcolor="#F0A202",
            opacity=0.22, line_width=0, row=1, col=1,
        )
        fig.add_vrect(
            x0=cluster["start"], x1=cluster["end"], fillcolor="#F0A202",
            opacity=0.18, line_width=0, row=2, col=1,
        )
        fig.add_vrect(
            x0=cluster["start"], x1=cluster["end"], fillcolor="#F0A202",
            opacity=0.18, line_width=0, annotation_text=annotation,
            annotation_position="top", row=3, col=1,
        )
    fig.add_vline(x=0, line_color="red", line_dash="dash", opacity=0.55, row="all", col=1)
    fig.add_hline(y=0, line_color="#555", opacity=0.4, row="all", col=1)
    fig.update_yaxes(
        title_text="Amplitude (z-score)" if zscore_subjects else "Amplitude (µV)",
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text="A − B (z-score)" if zscore_subjects else "A − B (µV)",
        row=2, col=1,
    )
    fig.update_yaxes(title_text="T statistic", row=3, col=1)
    fig.update_xaxes(title_text="Time relative to R-peak (s)", row=3, col=1)
    fig.update_layout(
        width=WAVEFORM_PLOT_SIZE,
        height=WAVEFORM_PLOT_SIZE,
        title=(
            f"{label_a} vs {label_b} | "
            f"N={len(result['patient_ids_a'])} vs {len(result['patient_ids_b'])}"
            + (" | ICA cleaned" if ica_cleaned else "")
        ),
        plot_bgcolor="white", paper_bgcolor="white", font_color="black",
        hovermode="x unified", legend_title="Waveform",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    return fig


def plot_electrode_overview(
    channel_results: Dict[str, dict],
    label_a: str,
    label_b: str,
    zscore_subjects: bool,
) -> go.Figure:
    """Small-multiple mean HEP and difference traces for every electrode."""
    channels = list(channel_results)
    n_cols = 3
    n_rows = max(1, (len(channels) + n_cols - 1) // n_cols)
    subplot_titles = []
    for channel in channels:
        result = channel_results[channel]
        coverage_a = result.get("coverage_fraction_a", np.nan)
        coverage_b = result.get("coverage_fraction_b", np.nan)
        subplot_titles.append(
            f"{channel} | "
            f"N={len(result['patient_ids_a'])}/{result.get('coverage_denominator_a', '?')} "
            f"({coverage_a:.0%}) vs "
            f"{len(result['patient_ids_b'])}/{result.get('coverage_denominator_b', '?')} "
            f"({coverage_b:.0%}) | p={format_p(result['contrast_p'])}, "
            f"q={format_p(result.get('q_value'))}"
        )
    fig = make_subplots(
        rows=n_rows, cols=n_cols, shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=min(0.12, 0.32 / n_rows),
        horizontal_spacing=0.08,
    )
    for channel_index, channel in enumerate(channels):
        row = channel_index // n_cols + 1
        col = channel_index % n_cols + 1
        result = channel_results[channel]
        components = _pointwise_t_components(
            result["matrix_a"], result["matrix_b"], result["paired"]
        )
        times = result["times"]
        show_legend = channel_index == 0
        fig.add_trace(go.Scatter(
            x=times, y=components["mean_a"], mode="lines",
            name=f"{label_a} mean", legendgroup="electrode_a",
            showlegend=show_legend, line=dict(color=PALETTE[0], width=1.5),
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=times, y=components["mean_b"], mode="lines",
            name=f"{label_b} mean", legendgroup="electrode_b",
            showlegend=show_legend, line=dict(color=PALETTE[1], width=1.5),
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=times, y=components["mean_difference"], mode="lines",
            name="Mean A − B", legendgroup="electrode_difference",
            showlegend=show_legend,
            line=dict(color="#222222", width=1.2, dash="dash"),
        ), row=row, col=col)
        for cluster in result["clusters"]:
            if cluster["significant"]:
                fig.add_vrect(
                    x0=cluster["start"], x1=cluster["end"],
                    fillcolor="#F0A202", opacity=0.20, line_width=0,
                    row=row, col=col,
                )
        fig.add_vline(
            x=0, line_color="red", line_dash="dash", opacity=0.4,
            row=row, col=col,
        )
        fig.add_hline(
            y=0, line_color="#777", opacity=0.3, row=row, col=col
        )
    y_title = "Amplitude (z-score)" if zscore_subjects else "Amplitude (µV)"
    fig.update_yaxes(title_text=y_title, showgrid=True, gridcolor="#eee")
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_layout(
        height=max(420, 285 * n_rows),
        title=(
            "Per-electrode HEP means and A − B differences "
            f"(≥{ELECTRODE_MIN_PATIENT_COVERAGE_FRACTION:.0%} patient coverage)"
        ),
        plot_bgcolor="white", paper_bgcolor="white", font_color="black",
        legend_title="Trace", hovermode="x unified",
    )
    return fig


def make_significant_channel_topomaps(
    channel_results: Dict[str, dict],
    label_a: str,
    label_b: str,
):
    """Create signed peak-T and signed -log10(q) maps for FDR-significant electrodes."""
    import matplotlib.pyplot as plt
    import mne

    montage = mne.channels.make_standard_montage("standard_1020")
    montage_lookup = {channel.upper(): channel for channel in montage.ch_names}
    aliases = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    montage_names = []
    display_names = []
    peak_t_values = []
    signed_log_q_values = []
    significant_mask = []
    significant_channels = []

    for channel, result in channel_results.items():
        lookup_name = aliases.get(channel.upper(), channel.upper())
        montage_name = montage_lookup.get(lookup_name)
        if montage_name is None or montage_name in montage_names:
            continue
        significant_clusters = [
            cluster for cluster in result["clusters"] if cluster["significant"]
        ]
        is_significant = bool(result.get("fdr_significant") and significant_clusters)
        if is_significant:
            best_cluster = min(significant_clusters, key=lambda cluster: cluster["p_value"])
            peak_t = float(best_cluster["peak_t"])
            signed_log_q = np.sign(peak_t) * -np.log10(
                max(float(result["q_value"]), 1e-12)
            )
            significant_channels.append(channel)
        else:
            peak_t = 0.0
            signed_log_q = 0.0
        montage_names.append(montage_name)
        display_names.append(channel if is_significant else "")
        peak_t_values.append(peak_t)
        signed_log_q_values.append(signed_log_q)
        significant_mask.append(is_significant)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    if len(montage_names) < 4:
        for axis in axes:
            axis.text(0.5, 0.5, "Too few montage-compatible electrodes", ha="center", va="center")
            axis.axis("off")
        return fig, significant_channels

    info = mne.create_info(montage_names, sfreq=250.0, ch_types="eeg")
    info.set_montage(montage, on_missing="ignore")
    valid_positions = np.array([
        np.all(np.isfinite(channel_info["loc"][:3]))
        and np.any(channel_info["loc"][:3] != 0)
        for channel_info in info["chs"]
    ])
    if valid_positions.sum() < 4:
        for axis in axes:
            axis.text(0.5, 0.5, "Too few valid electrode positions", ha="center", va="center")
            axis.axis("off")
        return fig, significant_channels

    info = mne.pick_info(info, np.flatnonzero(valid_positions))
    names = np.asarray(display_names, dtype=object)[valid_positions].tolist()
    mask = np.asarray(significant_mask, dtype=bool)[valid_positions]
    map_specs = (
        (np.asarray(peak_t_values)[valid_positions], "Signed peak T", "Peak T"),
        (
            np.asarray(signed_log_q_values)[valid_positions],
            "Signed significance: −log10(q)",
            "signed −log10(q)",
        ),
    )
    mask_params = dict(
        marker="o", markerfacecolor="none", markeredgecolor="black",
        linewidth=1.5, markersize=8,
    )
    for axis, (data, title, colorbar_label) in zip(axes, map_specs):
        vmax = float(np.nanmax(np.abs(data))) if np.any(np.isfinite(data)) else 1.0
        vmax = max(vmax, 1e-6)
        topomap = mne.viz.plot_topomap(
            data, info, axes=axis, cmap="RdBu_r", vlim=(-vmax, vmax),
            names=names, mask=mask, mask_params=mask_params,
            extrapolate="head", contours=0, show=False,
        )
        image = topomap[0] if isinstance(topomap, tuple) else topomap
        colorbar = fig.colorbar(image, ax=axis, shrink=0.8)
        colorbar.set_label(colorbar_label)
        axis.set_title(title)
    fig.suptitle(
        f"Significant electrodes only: {label_a} vs {label_b}\n"
        f"cluster p < {PAIRWISE_CLUSTER_ALPHA:.2f} and electrode FDR q < 0.05",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return fig, significant_channels


def make_red_significance_topomap(
    channel_results: Dict[str, dict],
    label_a: str,
    label_b: str,
):
    """Map raw cluster-permutation p-values: red at zero, white at/above p=0.01."""
    import matplotlib.pyplot as plt
    import mne

    montage = mne.channels.make_standard_montage("standard_1020")
    montage_lookup = {channel.upper(): channel for channel in montage.ch_names}
    aliases = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    montage_names = []
    display_names = []
    displayed_p_values = []
    display_p_threshold = 0.01

    for channel, result in channel_results.items():
        lookup_name = aliases.get(channel.upper(), channel.upper())
        montage_name = montage_lookup.get(lookup_name)
        if montage_name is None or montage_name in montage_names:
            continue
        p_value = float(result.get("contrast_p", 1.0))
        is_significant = bool(
            result.get("cluster_significant", False)
            and np.isfinite(p_value)
            and p_value < display_p_threshold
        )
        displayed_p = max(0.0, p_value) if is_significant else display_p_threshold
        montage_names.append(montage_name)
        display_names.append(channel)
        displayed_p_values.append(displayed_p)

    fig, axis = plt.subplots(figsize=(7.2, 6.2))
    if len(montage_names) < 4:
        axis.text(
            0.5, 0.5, "Too few montage-compatible electrodes",
            ha="center", va="center",
        )
        axis.axis("off")
        return fig

    info = mne.create_info(montage_names, sfreq=250.0, ch_types="eeg")
    info.set_montage(montage, on_missing="ignore")
    valid_positions = np.array([
        np.all(np.isfinite(channel_info["loc"][:3]))
        and np.any(channel_info["loc"][:3] != 0)
        for channel_info in info["chs"]
    ])
    if valid_positions.sum() < 4:
        axis.text(0.5, 0.5, "Too few valid electrode positions", ha="center", va="center")
        axis.axis("off")
        return fig

    info = mne.pick_info(info, np.flatnonzero(valid_positions))
    data = np.asarray(displayed_p_values, dtype=float)[valid_positions]
    names = np.asarray(display_names, dtype=object)[valid_positions].tolist()
    topomap = mne.viz.plot_topomap(
        data,
        info,
        axes=axis,
        cmap="Reds_r",
        vlim=(0.0, display_p_threshold),
        names=names,
        sensors=True,
        extrapolate="head",
        contours=0,
        show=False,
    )
    image = topomap[0] if isinstance(topomap, tuple) else topomap
    colorbar = fig.colorbar(image, ax=axis, shrink=0.82)
    colorbar.set_ticks(np.linspace(0.0, display_p_threshold, 6))
    colorbar.set_label("p-value")
    axis.set_title(f"{label_a} vs {label_b}")
    fig.tight_layout()
    return fig


def make_peak_significance_time_topomap(
    channel_results: Dict[str, dict],
    label_a: str,
    label_b: str,
):
    """Color each significant electrode by its strongest significant time."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import mne

    montage = mne.channels.make_standard_montage("standard_1020")
    montage_lookup = {channel.upper(): channel for channel in montage.ch_names}
    montage_names = []
    display_names = []
    peak_times = []
    all_observed_times = []
    aliases = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

    for channel, result in channel_results.items():
        lookup_name = aliases.get(channel.upper(), channel.upper())
        montage_name = montage_lookup.get(lookup_name)
        if montage_name is None or montage_name in montage_names:
            continue

        times = np.asarray(result.get("times", []), dtype=float)
        t_stat = np.asarray(result.get("t_stat", []), dtype=float)
        finite_times = times[np.isfinite(times)]
        if finite_times.size:
            all_observed_times.extend(finite_times.tolist())

        significant_clusters = [
            cluster for cluster in result.get("clusters", [])
            if cluster.get("significant", False)
        ]
        peak_time = np.nan
        if significant_clusters and times.size and t_stat.shape == times.shape:
            best_cluster = min(
                significant_clusters,
                key=lambda cluster: cluster.get("p_value", 1.0),
            )
            cluster_mask = (
                (times >= float(best_cluster["start"]))
                & (times <= float(best_cluster["end"]))
                & np.isfinite(t_stat)
            )
            cluster_indices = np.flatnonzero(cluster_mask)
            if cluster_indices.size:
                peak_index = cluster_indices[
                    np.argmax(np.abs(t_stat[cluster_indices]))
                ]
                peak_time = float(times[peak_index])
            else:
                peak_time = 0.5 * (
                    float(best_cluster["start"]) + float(best_cluster["end"])
                )
        montage_names.append(montage_name)
        display_names.append(channel)
        peak_times.append(peak_time)

    fig, axis = plt.subplots(figsize=(7.2, 6.2))
    if len(montage_names) < 4:
        axis.text(
            0.5, 0.5, "Too few montage-compatible electrodes",
            ha="center", va="center",
        )
        axis.axis("off")
        return fig

    info = mne.create_info(montage_names, sfreq=250.0, ch_types="eeg")
    info.set_montage(montage, on_missing="ignore")
    valid_positions = np.array([
        np.all(np.isfinite(channel_info["loc"][:3]))
        and np.any(channel_info["loc"][:3] != 0)
        for channel_info in info["chs"]
    ])
    if valid_positions.sum() < 4:
        axis.text(0.5, 0.5, "Too few valid electrode positions", ha="center", va="center")
        axis.axis("off")
        return fig

    info = mne.pick_info(info, np.flatnonzero(valid_positions))
    names = np.asarray(display_names, dtype=object)[valid_positions].tolist()
    peak_times = np.asarray(peak_times, dtype=float)[valid_positions]

    if all_observed_times:
        time_min = float(np.nanmin(all_observed_times))
        time_max = float(np.nanmax(all_observed_times))
    else:
        time_min, time_max = -0.3, 0.4
    if not np.isfinite(time_min) or not np.isfinite(time_max) or time_min >= time_max:
        time_min, time_max = -0.3, 0.4

    jet = mpl.colormaps["jet"]
    significant_mask = np.isfinite(peak_times)
    # Keep missingness separate from the time values. Encoding "not significant"
    # as a value beyond time_max creates a false red halo because interpolation
    # traverses the red end of jet before reaching the sentinel's white color.
    # The neutral fill below is hidden by the independent white mask.
    if significant_mask.any():
        neutral_time = float(np.nanmedian(peak_times[significant_mask]))
    else:
        neutral_time = 0.5 * (time_min + time_max)
    map_data = np.where(significant_mask, peak_times, neutral_time)
    mne.viz.plot_topomap(
        map_data,
        info,
        axes=axis,
        cmap=jet,
        vlim=(time_min, time_max),
        names=names,
        sensors=False,
        extrapolate="head",
        contours=0,
        show=False,
    )

    # A discrete overlay makes regions supported primarily by non-significant
    # electrodes fully white, without routing them through any jet color.
    white_or_clear = mpl.colors.ListedColormap(
        [(1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 0.0)],
        name="white_nonsignificant_mask",
    )
    mne.viz.plot_topomap(
        significant_mask.astype(float),
        info,
        axes=axis,
        cmap=white_or_clear,
        vlim=(0.0, 1.0),
        sensors=False,
        extrapolate="head",
        contours=0,
        outlines=None,
        show=False,
    )

    norm = mpl.colors.Normalize(vmin=time_min, vmax=time_max)
    scalar_map = mpl.cm.ScalarMappable(norm=norm, cmap=jet)
    scalar_map.set_array([])
    colorbar = fig.colorbar(scalar_map, ax=axis, shrink=0.82)
    colorbar.set_label("Time relative to R-peak (s)")
    axis.set_title(f"{label_a} vs {label_b}")
    fig.tight_layout()
    return fig


def _standard_1020_positions(channels: Sequence[str]) -> Dict[str, np.ndarray]:
    """Return standard 10-20 3D positions for known EEG channel names."""
    import mne

    montage = mne.channels.make_standard_montage("standard_1020")
    aliases = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    lookup = {name.upper(): np.asarray(pos, dtype=float) for name, pos in montage.get_positions()["ch_pos"].items()}
    positions: Dict[str, np.ndarray] = {}
    for channel in channels:
        lookup_name = aliases.get(channel.upper(), channel.upper())
        pos = lookup.get(lookup_name)
        if pos is not None and np.all(np.isfinite(pos)):
            positions[channel] = pos
    return positions


def _channel_window_statistics(
    channel_results: Dict[str, dict],
    time_window: tuple[float, float],
) -> pd.DataFrame:
    """Summarize per-electrode differences and p-values inside a selected time window."""
    rows = []
    for channel, result in channel_results.items():
        times = np.asarray(result["times"], dtype=float)
        window_mask = (times >= time_window[0]) & (times <= time_window[1])
        if window_mask.sum() < 2:
            window_mask = np.isfinite(times)
        a_window = np.nanmean(result["matrix_a"][:, window_mask], axis=1)
        b_window = np.nanmean(result["matrix_b"][:, window_mask], axis=1)
        if result["paired"] and len(a_window) == len(b_window):
            test = stats.ttest_rel(a_window, b_window, nan_policy="omit")
            paired_differences = a_window - b_window
            effect = (
                np.nanmean(paired_differences) / np.nanstd(paired_differences, ddof=1)
                if len(paired_differences) > 1 and np.nanstd(paired_differences, ddof=1) > 0
                else np.nan
            )
        else:
            test = stats.ttest_ind(a_window, b_window, equal_var=False, nan_policy="omit")
            pooled_sd = np.sqrt((np.nanvar(a_window, ddof=1) + np.nanvar(b_window, ddof=1)) / 2)
            effect = (
                (np.nanmean(a_window) - np.nanmean(b_window)) / pooled_sd
                if np.isfinite(pooled_sd) and pooled_sd > 0
                else np.nan
            )
        components = _pointwise_t_components(
            result["matrix_a"], result["matrix_b"], result["paired"]
        )
        t_stat = np.asarray(components["t_stat"], dtype=float)
        difference = np.asarray(components["mean_difference"], dtype=float)
        finite_window_t = t_stat[window_mask & np.isfinite(t_stat)]
        finite_window_difference = difference[window_mask & np.isfinite(difference)]
        significant_clusters = [
            cluster for cluster in result["clusters"] if cluster["significant"]
        ]
        best_cluster = (
            min(significant_clusters, key=lambda cluster: cluster["p_value"])
            if significant_clusters else None
        )
        rows.append({
            "Electrode": channel,
            "N A": len(result["patient_ids_a"]),
            "N B": len(result["patient_ids_b"]),
            "A coverage": result.get("coverage_fraction_a", np.nan),
            "B coverage": result.get("coverage_fraction_b", np.nan),
            "Window mean A-B": float(np.nanmean(finite_window_difference)) if finite_window_difference.size else np.nan,
            "Window t": float(test.statistic) if np.isfinite(test.statistic) else np.nan,
            "Window p": float(test.pvalue) if np.isfinite(test.pvalue) else np.nan,
            "Window effect": float(effect) if np.isfinite(effect) else np.nan,
            "Peak |T| in window": float(np.nanmax(np.abs(finite_window_t))) if finite_window_t.size else np.nan,
            "Cluster p": result["contrast_p"],
            "Cluster q": result.get("q_value", np.nan),
            "FDR significant": bool(result.get("fdr_significant", False)),
            "Significant windows": "; ".join(
                f"{cluster['start']:.3f}-{cluster['end']:.3f}s"
                for cluster in significant_clusters
            ),
            "Best cluster peak T": best_cluster["peak_t"] if best_cluster else np.nan,
        })
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["Window q"] = np.nan
    finite = summary["Window p"].notna()
    if finite.any():
        summary.loc[finite, "Window q"] = benjamini_hochberg(
            summary.loc[finite, "Window p"].to_numpy(dtype=float)
        )
    summary["Window significant"] = summary["Window q"] < 0.05
    return summary.sort_values(
        ["Window significant", "FDR significant", "Window p"],
        ascending=[False, False, True],
    )


def _with_positions(summary: pd.DataFrame) -> pd.DataFrame:
    """Attach standard montage coordinates to a per-electrode summary table."""
    if summary.empty:
        return summary.copy()
    positions = _standard_1020_positions(summary["Electrode"].tolist())
    rows = []
    for _, row in summary.iterrows():
        pos = positions.get(row["Electrode"])
        if pos is None:
            continue
        enriched = row.to_dict()
        enriched.update({"x": pos[0], "y": pos[1], "z": pos[2]})
        rows.append(enriched)
    return pd.DataFrame(rows)


def make_testing_window_topomaps(
    summary: pd.DataFrame,
    label_a: str,
    label_b: str,
    time_window: tuple[float, float],
):
    """Topographic brain maps for selected-window difference and significance."""
    import matplotlib.pyplot as plt
    import mne

    positioned = _with_positions(summary)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    if positioned.empty or len(positioned) < 4:
        for axis in axes:
            axis.text(0.5, 0.5, "Too few montage-compatible electrodes", ha="center", va="center")
            axis.axis("off")
        return fig

    montage = mne.channels.make_standard_montage("standard_1020")
    aliases = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    montage_lookup = {channel.upper(): channel for channel in montage.ch_names}
    montage_names = [
        montage_lookup.get(aliases.get(channel.upper(), channel.upper()), channel)
        for channel in positioned["Electrode"]
    ]
    info = mne.create_info(montage_names, sfreq=250.0, ch_types="eeg")
    info.set_montage(montage, on_missing="ignore")
    difference = positioned["Window mean A-B"].to_numpy(dtype=float)
    window_q = np.nan_to_num(
        positioned["Window q"].to_numpy(dtype=float), nan=1.0, posinf=1.0
    )
    signed_log_q = (
        np.sign(difference)
        * -np.log10(np.maximum(window_q, 1e-12))
    )
    window_mask = positioned["Window significant"].fillna(False).to_numpy(dtype=bool)
    cluster_mask = positioned["FDR significant"].fillna(False).to_numpy(dtype=bool)
    map_specs = (
        (difference, "Mean A-B in selected window", "A-B"),
        (signed_log_q, "Signed selected-window significance", "signed -log10(q)"),
    )
    for axis, (values, title, colorbar_label), mask in zip(
        axes, map_specs, (window_mask, cluster_mask | window_mask)
    ):
        vmax = float(np.nanmax(np.abs(values))) if np.any(np.isfinite(values)) else 1.0
        vmax = max(vmax, 1e-6)
        topomap = mne.viz.plot_topomap(
            values, info, axes=axis, cmap="RdBu_r", vlim=(-vmax, vmax),
            names=positioned["Electrode"].tolist(),
            mask=mask,
            mask_params=dict(
                marker="o", markerfacecolor="none", markeredgecolor="black",
                linewidth=1.5, markersize=8,
            ),
            extrapolate="head", contours=0, show=False,
        )
        image = topomap[0] if isinstance(topomap, tuple) else topomap
        colorbar = fig.colorbar(image, ax=axis, shrink=0.8)
        colorbar.set_label(colorbar_label)
        axis.set_title(title)
    fig.suptitle(
        f"{label_a} vs {label_b}, {time_window[0]:.3f}-{time_window[1]:.3f}s",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_3d_electrode_brain_map(
    summary: pd.DataFrame,
    label_a: str,
    label_b: str,
    time_window: tuple[float, float],
    color_metric: str,
) -> go.Figure:
    """Interactive 3D scalp-style electrode map."""
    positioned = _with_positions(summary)
    fig = go.Figure()
    theta = np.linspace(0, 2 * np.pi, 48)
    phi = np.linspace(0, np.pi, 24)
    radius = 0.105
    sphere_x = radius * np.outer(np.cos(theta), np.sin(phi))
    sphere_y = radius * np.outer(np.sin(theta), np.sin(phi))
    sphere_z = radius * np.outer(np.ones_like(theta), np.cos(phi))
    fig.add_trace(go.Surface(
        x=sphere_x, y=sphere_y, z=sphere_z,
        colorscale=[[0, "#eeeeee"], [1, "#eeeeee"]],
        showscale=False, opacity=0.20, hoverinfo="skip",
    ))
    if positioned.empty:
        fig.update_layout(title="No montage-compatible electrodes")
        return fig
    values = positioned[color_metric].to_numpy(dtype=float)
    finite_abs = np.abs(values[np.isfinite(values)])
    cmax = float(finite_abs.max()) if finite_abs.size else 1.0
    cmax = max(cmax, 1e-6)
    signed_window_q = (
        np.sign(positioned["Window mean A-B"].to_numpy(dtype=float))
        * -np.log10(np.maximum(
            np.nan_to_num(positioned["Window q"].to_numpy(dtype=float), nan=1.0, posinf=1.0),
            1e-12,
        ))
    )
    finite_signed_q = np.abs(signed_window_q[np.isfinite(signed_window_q)])
    signed_q_scale = max(float(finite_signed_q.max()) if finite_signed_q.size else 1.0, 1.0)
    marker_sizes = 8 + 10 * np.clip(
        np.nan_to_num(np.abs(signed_window_q), nan=0.0) / signed_q_scale,
        0,
        1,
    )
    fig.add_trace(go.Scatter3d(
        x=positioned["x"], y=positioned["y"], z=positioned["z"],
        mode="markers+text",
        text=positioned["Electrode"],
        textposition="top center",
        customdata=np.column_stack([
            positioned["Window mean A-B"],
            positioned["Window p"],
            positioned["Window q"],
            positioned["Cluster q"],
        ]),
        marker=dict(
            size=marker_sizes,
            color=values,
            colorscale="RdBu_r",
            cmin=-cmax,
            cmax=cmax,
            colorbar=dict(title=color_metric),
            line=dict(color="black", width=2),
        ),
        hovertemplate=(
            "%{text}<br>A-B: %{customdata[0]:.4f}<br>"
            "Window p: %{customdata[1]:.4g}<br>"
            "Window q: %{customdata[2]:.4g}<br>"
            "Cluster q: %{customdata[3]:.4g}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=(
            f"3D electrode map: {label_a} vs {label_b} | "
            f"{time_window[0]:.3f}-{time_window[1]:.3f}s"
        ),
        height=720,
        scene=dict(
            aspectmode="cube",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(eye=dict(x=0, y=-1.7, z=0.75)),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def plot_circular_electrode_statistics(
    summary: pd.DataFrame,
    label_a: str,
    label_b: str,
    time_window: tuple[float, float],
) -> go.Figure:
    """Circular electrode view: angle from scalp position, color by A-B, size by q."""
    positioned = _with_positions(summary)
    fig = go.Figure()
    if positioned.empty:
        fig.update_layout(title="No montage-compatible electrodes")
        return fig
    angles = np.arctan2(positioned["y"].to_numpy(dtype=float), positioned["x"].to_numpy(dtype=float))
    radius = np.ones(len(positioned))
    difference = positioned["Window mean A-B"].to_numpy(dtype=float)
    window_q = np.nan_to_num(
        positioned["Window q"].to_numpy(dtype=float), nan=1.0, posinf=1.0
    )
    signed_log_q = (
        np.sign(difference)
        * -np.log10(np.maximum(window_q, 1e-12))
    )
    cmax = max(float(np.nanmax(np.abs(difference))) if np.any(np.isfinite(difference)) else 1.0, 1e-6)
    finite_signed_q = np.abs(signed_log_q[np.isfinite(signed_log_q)])
    signed_q_scale = max(float(finite_signed_q.max()) if finite_signed_q.size else 1.0, 1.0)
    sizes = 12 + 18 * np.clip(
        np.nan_to_num(np.abs(signed_log_q), nan=0.0) / signed_q_scale,
        0,
        1,
    )
    fig.add_trace(go.Scatterpolar(
        r=radius,
        theta=np.degrees(angles),
        mode="markers+text",
        text=positioned["Electrode"],
        textposition="top center",
        customdata=np.column_stack([
            difference,
            positioned["Window p"],
            positioned["Window q"],
            positioned["FDR significant"],
        ]),
        marker=dict(
            size=sizes,
            color=difference,
            colorscale="RdBu_r",
            cmin=-cmax,
            cmax=cmax,
            colorbar=dict(title="A-B"),
            line=dict(color="#333", width=1.5),
        ),
        hovertemplate=(
            "%{text}<br>A-B: %{customdata[0]:.4f}<br>"
            "Window p: %{customdata[1]:.4g}<br>"
            "Window q: %{customdata[2]:.4g}<br>"
            "Waveform FDR significant: %{customdata[3]}<extra></extra>"
        ),
    ))
    for angle, significant in zip(np.degrees(angles), positioned["Window significant"].fillna(False)):
        if significant:
            fig.add_trace(go.Scatterpolar(
                r=[0, 0.86], theta=[angle, angle], mode="lines",
                line=dict(color="rgba(0,0,0,0.22)", width=2),
                hoverinfo="skip", showlegend=False,
            ))
    fig.update_layout(
        title=(
            f"Circular electrode statistics: {label_a} vs {label_b} | "
            f"{time_window[0]:.3f}-{time_window[1]:.3f}s"
        ),
        height=700,
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1.18]),
            angularaxis=dict(rotation=90, direction="clockwise", showticklabels=False),
        ),
        showlegend=False,
    )
    return fig


def _clusters_overlap(cluster_a: dict, cluster_b: dict) -> tuple[bool, float]:
    start = max(float(cluster_a["start"]), float(cluster_b["start"]))
    end = min(float(cluster_a["end"]), float(cluster_b["end"]))
    return end > start, max(0.0, end - start)


def plot_significant_window_network(
    channel_results: Dict[str, dict],
    summary: pd.DataFrame,
    label_a: str,
    label_b: str,
) -> tuple[go.Figure, pd.DataFrame]:
    """Brain-shaped electrode graph; edges connect channels with overlapping significant windows."""
    positioned = _with_positions(summary)
    fig = go.Figure()
    if positioned.empty:
        fig.update_layout(title="No montage-compatible electrodes")
        return fig, pd.DataFrame()
    position_lookup = {
        row["Electrode"]: (float(row["x"]), float(row["y"]))
        for _, row in positioned.iterrows()
    }
    node_cluster_lookup = {
        channel: [cluster for cluster in result["clusters"] if cluster["significant"]]
        for channel, result in channel_results.items()
    }
    edge_rows = []
    for channel_a, channel_b in combinations(position_lookup, 2):
        overlaps = []
        for cluster_a in node_cluster_lookup.get(channel_a, []):
            for cluster_b in node_cluster_lookup.get(channel_b, []):
                overlaps.append(_clusters_overlap(cluster_a, cluster_b))
        overlap_durations = [duration for has_overlap, duration in overlaps if has_overlap]
        if not overlap_durations:
            continue
        edge_rows.append({
            "Electrode A": channel_a,
            "Electrode B": channel_b,
            "Overlapping windows": len(overlap_durations),
            "Total overlap seconds": float(np.sum(overlap_durations)),
        })
    for row in edge_rows:
        xa, ya = position_lookup[row["Electrode A"]]
        xb, yb = position_lookup[row["Electrode B"]]
        width = 1 + 10 * min(row["Total overlap seconds"], 0.25) / 0.25
        fig.add_trace(go.Scatter(
            x=[xa, xb], y=[ya, yb], mode="lines",
            line=dict(color="rgba(20,20,20,0.22)", width=width),
            hovertemplate=(
                f"{row['Electrode A']} - {row['Electrode B']}<br>"
                f"overlap windows: {row['Overlapping windows']}<br>"
                f"total overlap: {row['Total overlap seconds']:.3f}s<extra></extra>"
            ),
            showlegend=False,
        ))
    difference = positioned["Window mean A-B"].to_numpy(dtype=float)
    cmax = max(float(np.nanmax(np.abs(difference))) if np.any(np.isfinite(difference)) else 1.0, 1e-6)
    node_sizes = np.where(positioned["FDR significant"].fillna(False), 24, 13)
    fig.add_trace(go.Scatter(
        x=positioned["x"], y=positioned["y"],
        mode="markers+text",
        text=positioned["Electrode"],
        textposition="top center",
        customdata=np.column_stack([
            positioned["Window mean A-B"],
            positioned["Window q"],
            positioned["Cluster q"],
            positioned["Significant windows"],
        ]),
        marker=dict(
            size=node_sizes,
            color=difference,
            colorscale="RdBu_r",
            cmin=-cmax,
            cmax=cmax,
            colorbar=dict(title="A-B"),
            line=dict(
                color=np.where(positioned["FDR significant"].fillna(False), "black", "#777"),
                width=np.where(positioned["FDR significant"].fillna(False), 3, 1),
            ),
        ),
        hovertemplate=(
            "%{text}<br>A-B: %{customdata[0]:.4f}<br>"
            "Window q: %{customdata[1]:.4g}<br>"
            "Cluster q: %{customdata[2]:.4g}<br>"
            "Cluster windows: %{customdata[3]}<extra></extra>"
        ),
        showlegend=False,
    ))
    head_theta = np.linspace(0, 2 * np.pi, 160)
    fig.add_trace(go.Scatter(
        x=0.115 * np.cos(head_theta), y=0.145 * np.sin(head_theta),
        mode="lines", line=dict(color="#444", width=2),
        hoverinfo="skip", showlegend=False,
    ))
    fig.update_layout(
        title=f"Significant-window electrode graph: {label_a} vs {label_b}",
        height=720,
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig, pd.DataFrame(edge_rows).sort_values(
        "Total overlap seconds", ascending=False
    ) if edge_rows else pd.DataFrame()


def _pairwise_summary_table(
    grouped_df: pd.DataFrame,
    comparisons_to_run: list,
    selected_channels: Sequence[str],
    zscore_subjects: bool,
    n_permutations: int,
    paired: bool,
) -> pd.DataFrame:
    """Run all requested contrasts and FDR-correct their minimum cluster p-values."""
    rows = []
    for group_a, stage_a, group_b, stage_b in comparisons_to_run:
        result = prepare_waveform_contrast(
            grouped_df, group_a, stage_a, group_b, stage_b,
            selected_channels, zscore_subjects, n_permutations, paired,
        )
        if result is None:
            rows.append({
                "Group A": group_a, "Stage A": STAGE_LABELS.get(stage_a, stage_a),
                "Group B": group_b, "Stage B": STAGE_LABELS.get(stage_b, stage_b),
                "N A": 0, "N B": 0, "p_value": np.nan,
                "Significant windows": 0,
            })
            continue
        rows.append({
            "Group A": group_a, "Stage A": STAGE_LABELS.get(stage_a, stage_a),
            "Group B": group_b, "Stage B": STAGE_LABELS.get(stage_b, stage_b),
            "N A": len(result["patient_ids_a"]), "N B": len(result["patient_ids_b"]),
            "p_value": result["contrast_p"],
            "Significant windows": sum(c["significant"] for c in result["clusters"]),
        })
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["q_value"] = np.nan
    finite = summary["p_value"].notna()
    if finite.any():
        summary.loc[finite, "q_value"] = benjamini_hochberg(
            summary.loc[finite, "p_value"].to_numpy(dtype=float)
        )
    return summary


def render_pairwise_waveform_mode(
    grouped_df: pd.DataFrame,
    all_groups: Sequence[str],
    selected_stages: Sequence[str],
    selected_channels: Sequence[str],
    apply_ica: bool = False,
) -> None:
    """Automatically run and rank every diagnosis and sleep-stage contrast."""
    st.header("Pairwise HEP waveform comparisons")
    st.caption(
        "Automatically compares every disease pair within the same sleep stage and "
        "every sleep-stage pair within the same disease. Significant comparisons "
        "are shown first; non-significant comparisons follow below."
    )
    with st.expander("How these comparisons are tested", expanded=False):
        st.markdown(
            "**Diagnosis contrasts** use an independent-samples cluster-mass "
            "permutation test. If a patient belongs to both diagnoses, that patient "
            "is removed from both cells before testing.\n\n"
            "**Sleep-stage contrasts** retain only patients observed in both stages "
            "and use a paired sign-flip cluster-mass permutation test. This preserves "
            "the repeated-measures design.\n\n"
            f"The cluster-forming T threshold and cluster significance cutoff both "
            f"use two-sided **p < {PAIRWISE_CLUSTER_ALPHA:.2f}**. The cluster test "
            "controls the family-wise error rate across waveform "
            "time points. Because all disease and stage pairs are tested automatically, "
            "the summary also reports `q_value` after Benjamini–Hochberg FDR "
            "correction across every valid contrast on this page. Section placement "
            "follows the cluster-permutation significance returned by the test."
        )
    settings_col, normalize_col = st.columns(2)
    with settings_col:
        n_permutations = st.slider(
            "Cluster permutations", 100, 2000, 200, 100,
            help="More permutations give finer p-values but take longer.",
        )
    with normalize_col:
        zscore_subjects = st.checkbox(
            "Z-score each patient's waveform", value=False,
            help="Useful for waveform shape; leave off to retain amplitude in µV.",
        )

    comparison_definitions = []
    for stage in selected_stages:
        for group_a, group_b in combinations(all_groups, 2):
            comparison_definitions.append({
                "contrast_type": "Disease pair within stage",
                "group_a": group_a, "stage_a": stage,
                "group_b": group_b, "stage_b": stage,
                "paired": False,
                "label_a": group_a,
                "label_b": group_b,
            })
    for group in all_groups:
        for stage_a, stage_b in combinations(selected_stages, 2):
            comparison_definitions.append({
                "contrast_type": "Sleep-stage pair within disease",
                "group_a": group, "stage_a": stage_a,
                "group_b": group, "stage_b": stage_b,
                "paired": True,
                "label_a": f"{group} — {STAGE_LABELS.get(stage_a, stage_a)}",
                "label_b": f"{group} — {STAGE_LABELS.get(stage_b, stage_b)}",
            })

    if not comparison_definitions:
        st.warning(
            "Automatic pairwise analysis needs at least two diagnosis groups or "
            "at least two sleep stages."
        )
        return

    patient_signature = tuple(sorted(
        (
            str(row.diagnosis_group), str(row.stage), str(row.patient_id)
        )
        for row in grouped_df[["diagnosis_group", "stage", "patient_id"]]
        .drop_duplicates().itertuples(index=False)
    ))
    cache_key = (
        "mean_sem_t_p01_canonical_subject_5pct_channels_v5",
        tuple(all_groups), tuple(selected_stages), tuple(selected_channels),
        bool(zscore_subjects), bool(apply_ica), int(n_permutations), patient_signature,
    )
    cache_entry = st.session_state.get("_automatic_pairwise_hep_cache")
    if cache_entry and cache_entry.get("key") == cache_key:
        comparison_results = cache_entry["results"]
        st.success(
            f"Loaded {len(comparison_results)} automatic pairwise results from this session."
        )
    else:
        comparison_results = []
        progress = st.progress(0, text="Starting automatic pairwise comparisons…")
        for index, definition in enumerate(comparison_definitions):
            progress.progress(
                (index + 1) / len(comparison_definitions),
                text=(
                    f"Comparison {index + 1}/{len(comparison_definitions)}: "
                    f"{definition['label_a']} vs {definition['label_b']}"
                ),
            )
            result = prepare_waveform_contrast(
                grouped_df,
                definition["group_a"], definition["stage_a"],
                definition["group_b"], definition["stage_b"],
                selected_channels, zscore_subjects, n_permutations,
                definition["paired"],
            )
            comparison_results.append({**definition, "result": result})
        progress.empty()
        st.session_state["_automatic_pairwise_hep_cache"] = {
            "key": cache_key,
            "results": comparison_results,
        }

    valid_results = [item for item in comparison_results if item["result"] is not None]
    valid_p_values = np.array([
        item["result"]["contrast_p"] for item in valid_results
    ], dtype=float)
    q_values = (
        benjamini_hochberg(valid_p_values)
        if len(valid_p_values) else np.array([], dtype=float)
    )
    for item, q_value in zip(valid_results, q_values):
        item["q_value"] = float(q_value)
        item["fdr_significant"] = bool(q_value < 0.05)
        item["cluster_significant"] = any(
            cluster["significant"] for cluster in item["result"]["clusters"]
        )

    significant_results = sorted(
        [item for item in valid_results if item["cluster_significant"]],
        key=lambda item: (item["result"]["contrast_p"], item["q_value"]),
    )
    nonsignificant_results = sorted(
        [item for item in valid_results if not item["cluster_significant"]],
        key=lambda item: (item["result"]["contrast_p"], item["q_value"]),
    )
    unavailable_results = [
        item for item in comparison_results if item["result"] is None
    ]

    summary_rows = []
    for item in comparison_results:
        result = item["result"]
        summary_rows.append({
            "Contrast type": item["contrast_type"],
            "Group A": item["group_a"],
            "Stage A": STAGE_LABELS.get(item["stage_a"], item["stage_a"]),
            "Group B": item["group_b"],
            "Stage B": STAGE_LABELS.get(item["stage_b"], item["stage_b"]),
            "N A": len(result["patient_ids_a"]) if result else 0,
            "N B": len(result["patient_ids_b"]) if result else 0,
            "cluster_p": result["contrast_p"] if result else np.nan,
            "q_value": item.get("q_value", np.nan),
            "Cluster significant": item.get("cluster_significant", False),
            "FDR significant": item.get("fdr_significant", False),
            "Significant clusters": (
                sum(cluster["significant"] for cluster in result["clusters"])
                if result else 0
            ),
            "Overlapping patients removed": result["overlap_removed"] if result else 0,
        })
    summary = pd.DataFrame(summary_rows)

    metric_columns = st.columns(4)
    metric_columns[0].metric("All contrasts", len(comparison_results))
    metric_columns[1].metric("Cluster significant", len(significant_results))
    metric_columns[2].metric("Non-significant", len(nonsignificant_results))
    metric_columns[3].metric("Insufficient data", len(unavailable_results))
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download all automatic pairwise statistics",
        summary.to_csv(index=False),
        "hep_all_pairwise_waveform_statistics.csv",
        "text/csv",
    )
    st.caption(
        "In every plot, the top lines are group means and the middle panel is their "
        "exact difference A − B. The bottom line is computed from those same values as "
        "T = mean difference / standard error. Red T segments cross the cluster-forming "
        f"two-sided p < {PAIRWISE_CLUSTER_ALPHA:.2f} threshold; orange bands survived "
        "the cluster permutation test at the same p-value cutoff. "
        "The table separately reports FDR q-values across all contrasts."
    )

    for plot_index, item in enumerate(comparison_results):
        item["_plot_index"] = plot_index

    def render_result(item: dict, expanded: bool) -> None:
        result = item["result"]
        stage_description = (
            STAGE_LABELS.get(item["stage_a"], item["stage_a"])
            if item["stage_a"] == item["stage_b"]
            else (
                f"{STAGE_LABELS.get(item['stage_a'], item['stage_a'])} vs "
                f"{STAGE_LABELS.get(item['stage_b'], item['stage_b'])}"
            )
        )
        title = (
            f"{item['label_a']} vs {item['label_b']} | {stage_description} | "
            f"p={format_p(result['contrast_p'])}, q={format_p(item['q_value'])}"
        )
        with st.expander(title, expanded=expanded):
            if result["overlap_removed"]:
                st.info(
                    f"Removed {result['overlap_removed']} patient(s) present in both "
                    "diagnosis groups before the independent-samples test."
                )
            st.plotly_chart(
                plot_waveform_contrast(
                    result, item["label_a"], item["label_b"], zscore_subjects,
                    ica_cleaned=apply_ica,
                ),
                use_container_width=False,
                theme=None,
                key=f"automatic_pairwise_plot_{item['_plot_index']}",
            )

            electrode_key = (
                "automatic_pairwise_electrodes_5pct_coverage_v4",
                item["group_a"], item["stage_a"], item["group_b"], item["stage_b"],
                tuple(selected_channels), bool(zscore_subjects),
                int(n_permutations), bool(apply_ica), item["paired"],
                patient_signature,
            )
            show_electrodes = st.checkbox(
                "Show differences for every electrode",
                value=expanded,
                key=f"automatic_pairwise_show_electrodes_{item['_plot_index']}",
                help=(
                    "Runs the same cluster test separately for each selected electrode "
                    "and shows the per-electrode A-B differences in this comparison."
                ),
            )
            if show_electrodes:
                electrode_cache = st.session_state.setdefault(
                    "_automatic_pairwise_electrode_cache", {}
                )
                if electrode_key in electrode_cache:
                    channel_results = electrode_cache[electrode_key]
                    st.success("Loaded per-electrode differences from this session.")
                else:
                    electrode_progress = st.progress(
                        0.0, text="Testing electrodes for this comparison..."
                    )

                    def update_electrode_progress(index: int, total: int, channel: str) -> None:
                        electrode_progress.progress(
                            (index + 1) / max(total, 1),
                            text=f"Testing electrode {index + 1}/{total}: {channel}",
                        )

                    channel_results = prepare_electrode_contrasts(
                        grouped_df,
                        item["group_a"], item["stage_a"],
                        item["group_b"], item["stage_b"],
                        selected_channels, zscore_subjects, n_permutations,
                        item["paired"],
                        progress_callback=update_electrode_progress,
                    )
                    electrode_progress.empty()
                    electrode_cache[electrode_key] = channel_results

                if channel_results:
                    st.subheader("Differences for every electrode")
                    st.plotly_chart(
                        plot_electrode_overview(
                            channel_results,
                            item["label_a"],
                            item["label_b"],
                            zscore_subjects,
                        ),
                        use_container_width=True,
                        theme=None,
                        key=f"automatic_pairwise_electrode_plot_{item['_plot_index']}",
                    )
                    st.markdown("**Electrode significance topomap**")
                    red_topomap = make_red_significance_topomap(
                        channel_results,
                        item["label_a"],
                        item["label_b"],
                    )
                    st.pyplot(red_topomap, use_container_width=True)
                    plt.close(red_topomap)
                    st.caption(
                        "Topomap: darker red means a smaller p-value; white means "
                        f"p≥{PAIRWISE_CLUSTER_ALPHA:.2f}. Electrodes are present in at least "
                        f"{ELECTRODE_MIN_PATIENT_COVERAGE_FRACTION:.0%} of patients "
                        "in both comparison cells."
                    )
                    st.markdown("**Peak significant time topomap**")
                    time_topomap = make_peak_significance_time_topomap(
                        channel_results,
                        item["label_a"],
                        item["label_b"],
                    )
                    st.pyplot(time_topomap, use_container_width=True)
                    plt.close(time_topomap)
                    st.caption(
                        "Jet color shows the strongest significant time; white means "
                        "no significant cluster."
                    )
                else:
                    st.info(
                        "No selected electrode has valid traces in at least "
                        f"{ELECTRODE_MIN_PATIENT_COVERAGE_FRACTION:.0%} of patients "
                        "in both comparison cells."
                    )

            significant_clusters = [
                cluster for cluster in result["clusters"] if cluster["significant"]
            ]
            if significant_clusters:
                cluster_df = pd.DataFrame(significant_clusters).drop(columns="significant")
                cluster_df.insert(
                    0, "Cluster",
                    [f"C{number}" for number in range(1, len(cluster_df) + 1)],
                )
                cluster_df["duration_ms"] = (
                    cluster_df["end"] - cluster_df["start"]
                ) * 1000
                st.dataframe(cluster_df, use_container_width=True, hide_index=True)
            else:
                st.info("No time cluster survived the cluster-mass permutation test.")

    tab_definitions = []
    cross_stage_items = [
        item for item in comparison_results if item["stage_a"] != item["stage_b"]
    ]
    if cross_stage_items:
        # Streamlit opens the first tab by default, so put Cross-stage first.
        tab_definitions.append(("Cross-stage", cross_stage_items))
    for stage in selected_stages:
        tab_definitions.append((
            STAGE_LABELS.get(stage, stage),
            [
                item for item in comparison_results
                if item["stage_a"] == stage and item["stage_b"] == stage
            ],
        ))

    stage_tabs = st.tabs([label for label, _ in tab_definitions])
    for stage_tab, (tab_label, tab_items) in zip(stage_tabs, tab_definitions):
        with stage_tab:
            display_items = tab_items
            if tab_label == "Cross-stage":
                show_wake_cross_stage = st.checkbox(
                    "Show Wake comparisons",
                    value=False,
                    key="pairwise_cross_stage_show_wake",
                    help=(
                        "Wake is hidden in Cross-stage by default so sleep-stage "
                        "comparisons focus on Light Sleep, N3, and REM."
                    ),
                )
                if not show_wake_cross_stage:
                    display_items = [
                        item for item in tab_items
                        if item["stage_a"] != "W" and item["stage_b"] != "W"
                    ]
                    hidden_wake_count = len(tab_items) - len(display_items)
                    if hidden_wake_count:
                        st.caption(
                            f"Hidden by default: {hidden_wake_count} Cross-stage "
                            "comparison(s) involving Wake."
                        )

            tab_valid = [item for item in display_items if item["result"] is not None]
            tab_significant = sorted(
                [item for item in tab_valid if item.get("cluster_significant")],
                key=lambda item: (item["result"]["contrast_p"], item["q_value"]),
            )
            tab_nonsignificant = sorted(
                [item for item in tab_valid if not item.get("cluster_significant")],
                key=lambda item: (item["result"]["contrast_p"], item["q_value"]),
            )
            tab_unavailable = [item for item in display_items if item["result"] is None]

            st.subheader(
                f"{tab_label}: significant comparisons ({len(tab_significant)})"
            )
            if not tab_significant:
                st.info(
                    "No comparison in this tab contained a cluster with permutation "
                    f"p < {PAIRWISE_CLUSTER_ALPHA:.2f}."
                )
            for item in tab_significant:
                render_result(item, expanded=True)

            st.subheader(
                f"{tab_label}: non-significant comparisons ({len(tab_nonsignificant)})"
            )
            st.caption(
                "Collapsed by default; open any comparison to see its complete plot."
            )
            for item in tab_nonsignificant:
                render_result(item, expanded=False)

            if tab_unavailable:
                st.subheader(
                    f"{tab_label}: insufficient data ({len(tab_unavailable)})"
                )
                unavailable_rows = [{
                    "Group A": item["group_a"],
                    "Stage A": STAGE_LABELS.get(item["stage_a"], item["stage_a"]),
                    "Group B": item["group_b"],
                    "Stage B": STAGE_LABELS.get(item["stage_b"], item["stage_b"]),
                } for item in tab_unavailable]
                st.dataframe(
                    pd.DataFrame(unavailable_rows),
                    use_container_width=True,
                    hide_index=True,
                )


def render_electrode_topomap_mode(
    grouped_df: pd.DataFrame,
    all_groups: Sequence[str],
    selected_stages: Sequence[str],
    selected_channels: Sequence[str],
    apply_ica: bool = False,
) -> None:
    """Render one electrode-resolved contrast with per-channel topographic statistics."""
    st.header("Electrode-resolved HEP differences and significant-channel topomaps")
    st.caption(
        "Shows the existing channel-averaged comparison, the same group difference "
        "for every electrode, and scalp maps containing only statistically significant channels."
    )
    if len(all_groups) < 2 and len(selected_stages) < 2:
        st.warning("Select at least two groups or two sleep stages.")
        return

    settings_col, normalization_col = st.columns(2)
    with settings_col:
        n_permutations = st.slider(
            "Per-electrode cluster permutations", 100, 1000, 200, 100,
            key="electrode_n_permutations",
            help=(
                "Each selected electrode is tested separately. At least 100 "
                "permutations are required for a p < 0.01 result."
            ),
        )
    with normalization_col:
        zscore_subjects = st.checkbox(
            "Z-score each patient for electrode analysis", value=False,
            key="electrode_zscore_subjects",
        )

    contrast_options = []
    if len(all_groups) >= 2:
        contrast_options.append("Diagnoses within one sleep stage")
    if len(selected_stages) >= 2:
        contrast_options.append("Sleep stages within one diagnosis")
    contrast_kind = st.radio(
        "Electrode contrast", contrast_options, horizontal=True,
        key="electrode_contrast_kind",
    )

    if contrast_kind == "Diagnoses within one sleep stage":
        control_a, control_b, control_stage = st.columns(3)
        with control_a:
            group_a = st.selectbox(
                "Diagnosis A", all_groups, index=0, key="electrode_group_a"
            )
        with control_b:
            group_b = st.selectbox(
                "Diagnosis B", all_groups, index=1, key="electrode_group_b"
            )
        with control_stage:
            stage_a = st.selectbox(
                "Sleep stage", selected_stages, key="electrode_same_stage",
                format_func=lambda value: STAGE_LABELS.get(value, value),
            )
        stage_b = stage_a
        paired = False
        label_a, label_b = group_a, group_b
    else:
        control_group, control_a, control_b = st.columns(3)
        with control_group:
            group_a = st.selectbox(
                "Diagnosis", all_groups, index=0, key="electrode_same_group"
            )
        group_b = group_a
        with control_a:
            stage_a = st.selectbox(
                "Sleep stage A", selected_stages, index=0,
                key="electrode_stage_a",
                format_func=lambda value: STAGE_LABELS.get(value, value),
            )
        with control_b:
            stage_b = st.selectbox(
                "Sleep stage B", selected_stages, index=1,
                key="electrode_stage_b",
                format_func=lambda value: STAGE_LABELS.get(value, value),
            )
        paired = True
        label_a = f"{group_a} — {STAGE_LABELS.get(stage_a, stage_a)}"
        label_b = f"{group_b} — {STAGE_LABELS.get(stage_b, stage_b)}"

    if group_a == group_b and stage_a == stage_b:
        st.warning("Choose two different diagnoses or sleep stages.")
        return

    patient_signature = tuple(sorted(
        (
            str(row.diagnosis_group), str(row.stage), str(row.patient_id)
        )
        for row in grouped_df[["diagnosis_group", "stage", "patient_id"]]
        .drop_duplicates().itertuples(index=False)
    ))
    cache_key = (
        "electrode_topomap_p01_canonical_subject_5pct_coverage_v5",
        group_a, stage_a, group_b, stage_b, tuple(selected_channels),
        zscore_subjects, n_permutations, bool(apply_ica), patient_signature,
    )
    cache_entry = st.session_state.get("_electrode_pairwise_hep_cache")
    if cache_entry and cache_entry.get("key") == cache_key:
        average_result = cache_entry["average_result"]
        channel_results = cache_entry["channel_results"]
        st.success("Loaded the electrode-resolved comparison from this session.")
    else:
        average_progress = st.progress(
            0.0, text="Computing the channel-averaged comparison…"
        )
        average_result = prepare_waveform_contrast(
            grouped_df, group_a, stage_a, group_b, stage_b,
            selected_channels, zscore_subjects, n_permutations, paired,
        )

        def update_electrode_progress(index: int, total: int, channel: str) -> None:
            average_progress.progress(
                (index + 1) / max(total, 1),
                text=f"Testing electrode {index + 1}/{total}: {channel}",
            )

        channel_results = prepare_electrode_contrasts(
            grouped_df, group_a, stage_a, group_b, stage_b,
            selected_channels, zscore_subjects, n_permutations, paired,
            progress_callback=update_electrode_progress,
        )
        average_progress.empty()
        st.session_state["_electrode_pairwise_hep_cache"] = {
            "key": cache_key,
            "average_result": average_result,
            "channel_results": channel_results,
        }

    if average_result is None:
        st.warning("The selected contrast has insufficient channel-averaged patient data.")
        return
    if not channel_results:
        st.warning(
            "No selected electrode has valid traces in at least "
            f"{ELECTRODE_MIN_PATIENT_COVERAGE_FRACTION:.0%} of patients in both "
            "comparison cells."
        )
        return

    st.subheader("1. Average across selected electrodes")
    channels_used = average_result.get("channels_used", list(selected_channels))
    st.caption(
        f"Average uses {len(channels_used)} electrode(s) meeting the "
        f"≥{ELECTRODE_MIN_PATIENT_COVERAGE_FRACTION:.0%} coverage rule in both "
        f"comparison cells: {', '.join(channels_used)}"
    )
    st.plotly_chart(
        plot_waveform_contrast(
            average_result, label_a, label_b, zscore_subjects,
            ica_cleaned=apply_ica,
        ),
        use_container_width=False, theme=None,
        key="electrode_average_waveform_plot",
    )

    channel_rows = []
    for channel, result in channel_results.items():
        significant_clusters = [
            cluster for cluster in result["clusters"] if cluster["significant"]
        ]
        best_cluster = (
            min(significant_clusters, key=lambda cluster: cluster["p_value"])
            if significant_clusters else None
        )
        channel_rows.append({
            "Electrode": channel,
            "N A": len(result["patient_ids_a"]),
            "A total patients": result.get("coverage_denominator_a", np.nan),
            "A coverage": result.get("coverage_fraction_a", np.nan),
            "N B": len(result["patient_ids_b"]),
            "B total patients": result.get("coverage_denominator_b", np.nan),
            "B coverage": result.get("coverage_fraction_b", np.nan),
            "minimum cluster p": result["contrast_p"],
            "electrode q": result.get("q_value", np.nan),
            "cluster significant": result.get("cluster_significant", False),
            "FDR significant": result.get("fdr_significant", False),
            "significant clusters": len(significant_clusters),
            "signed peak T": best_cluster["peak_t"] if best_cluster else np.nan,
        })
    channel_summary = pd.DataFrame(channel_rows).sort_values(
        ["FDR significant", "minimum cluster p"], ascending=[False, True]
    )
    st.dataframe(channel_summary, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download per-electrode statistics",
        channel_summary.to_csv(index=False),
        "hep_per_electrode_statistics.csv",
        "text/csv",
    )

    st.subheader("2. Differences for every electrode")
    st.plotly_chart(
        plot_electrode_overview(
            channel_results, label_a, label_b, zscore_subjects
        ),
        use_container_width=True, theme=None,
        key="electrode_overview_plot",
    )
    st.caption(
        "Orange spans are channel-specific clusters with permutation "
        f"p < {PAIRWISE_CLUSTER_ALPHA:.2f}. Electrode q-values correct across all "
        "electrodes with valid traces in at least "
        f"{ELECTRODE_MIN_PATIENT_COVERAGE_FRACTION:.0%} of patients in both comparison cells. "
        "The statistics table reports each electrode's actual patient coverage."
    )

    st.subheader("3. Electrode significance: white-to-red topomap")
    import matplotlib.pyplot as plt
    red_topomap = make_red_significance_topomap(
        channel_results, label_a, label_b
    )
    st.pyplot(red_topomap, use_container_width=True)
    plt.close(red_topomap)
    st.caption(
        "Darker red means a smaller p-value; white means p≥0.01."
    )

    st.subheader("4. Peak significant time topomap")
    time_topomap = make_peak_significance_time_topomap(
        channel_results, label_a, label_b
    )
    st.pyplot(time_topomap, use_container_width=True)
    plt.close(time_topomap)
    st.caption(
        "Jet color shows the time of the strongest T-statistic inside each "
        "electrode's most significant cluster. White means no significant cluster."
    )

    st.subheader("5. Significant-channel scalp topomaps")
    topomap_figure, significant_channels = make_significant_channel_topomaps(
        channel_results, label_a, label_b
    )
    if significant_channels:
        st.pyplot(topomap_figure, use_container_width=True)
        import matplotlib.pyplot as plt
        plt.close(topomap_figure)
        st.success(
            "FDR-significant electrodes: " + ", ".join(significant_channels)
        )
        st.caption(
            "Only channels with cluster p < 0.01 and electrode-level FDR q < 0.05 "
            "contribute non-zero values and receive black electrode markers."
        )
    else:
        import matplotlib.pyplot as plt
        plt.close(topomap_figure)
        st.info(
            "No electrode survived both cluster p < 0.01 and electrode-level "
            "FDR q < 0.05, so there is no significant-channel topomap to display."
        )

    st.subheader("6. Detailed plot for one electrode")
    detail_channel = st.selectbox(
        "Electrode", list(channel_results), key="electrode_detail_channel"
    )
    st.plotly_chart(
        plot_waveform_contrast(
            channel_results[detail_channel],
            f"{label_a} — {detail_channel}",
            f"{label_b} — {detail_channel}",
            zscore_subjects,
            ica_cleaned=apply_ica,
        ),
        use_container_width=False, theme=None,
        key="electrode_detail_waveform_plot",
    )


def render_testing_figures_mode(
    grouped_df: pd.DataFrame,
    all_groups: Sequence[str],
    selected_stages: Sequence[str],
    selected_channels: Sequence[str],
    apply_ica: bool = False,
) -> None:
    """Experimental figure lab for comparing ways to show per-electrode differences."""
    st.header("Testing figures: per-electrode difference visualization lab")
    st.caption(
        "Choose one contrast, then compare several ways of showing which electrodes "
        "differ, when they differ, and whether the selected time window is significant."
    )
    if len(all_groups) < 2 and len(selected_stages) < 2:
        st.warning("Select at least two diagnosis groups or two sleep stages.")
        return

    settings_a, settings_b, settings_c = st.columns(3)
    with settings_a:
        n_permutations = st.slider(
            "Per-electrode permutations", 100, 1000, 200, 100,
            key="testing_figures_n_permutations",
            help=(
                "Used for the channel-wise cluster permutation test. Higher values "
                "are slower but give more stable p-values."
            ),
        )
    with settings_b:
        zscore_subjects = st.checkbox(
            "Z-score each patient", value=False, key="testing_figures_zscore"
        )
    with settings_c:
        time_window = st.slider(
            "Specific time window (s)",
            min_value=-0.30,
            max_value=0.40,
            value=(0.15, 0.30),
            step=0.01,
            key="testing_figures_time_window",
            help=(
                "The window-specific map averages each patient/electrode in this "
                "interval, tests A vs B, and FDR-corrects across electrodes."
            ),
        )

    contrast_options = []
    if len(all_groups) >= 2:
        contrast_options.append("Diagnoses within one sleep stage")
    if len(selected_stages) >= 2:
        contrast_options.append("Sleep stages within one diagnosis")
    contrast_kind = st.radio(
        "Contrast for testing figures",
        contrast_options,
        horizontal=True,
        key="testing_figures_contrast_kind",
    )

    if contrast_kind == "Diagnoses within one sleep stage":
        control_a, control_b, control_stage = st.columns(3)
        with control_a:
            group_a = st.selectbox(
                "Diagnosis A", all_groups, index=0, key="testing_figures_group_a"
            )
        with control_b:
            group_b = st.selectbox(
                "Diagnosis B", all_groups, index=1, key="testing_figures_group_b"
            )
        with control_stage:
            stage_a = st.selectbox(
                "Sleep stage", selected_stages, key="testing_figures_stage",
                format_func=lambda value: STAGE_LABELS.get(value, value),
            )
        stage_b = stage_a
        paired = False
        label_a, label_b = group_a, group_b
    else:
        control_group, control_a, control_b = st.columns(3)
        with control_group:
            group_a = st.selectbox(
                "Diagnosis", all_groups, index=0, key="testing_figures_same_group"
            )
        group_b = group_a
        with control_a:
            stage_a = st.selectbox(
                "Sleep stage A", selected_stages, index=0,
                key="testing_figures_stage_a",
                format_func=lambda value: STAGE_LABELS.get(value, value),
            )
        with control_b:
            stage_b = st.selectbox(
                "Sleep stage B", selected_stages, index=1,
                key="testing_figures_stage_b",
                format_func=lambda value: STAGE_LABELS.get(value, value),
            )
        paired = True
        label_a = f"{group_a} — {STAGE_LABELS.get(stage_a, stage_a)}"
        label_b = f"{group_b} — {STAGE_LABELS.get(stage_b, stage_b)}"

    if group_a == group_b and stage_a == stage_b:
        st.warning("Choose two different diagnoses or sleep stages.")
        return

    patient_signature = tuple(sorted(
        (
            str(row.diagnosis_group), str(row.stage), str(row.patient_id)
        )
        for row in grouped_df[["diagnosis_group", "stage", "patient_id"]]
        .drop_duplicates().itertuples(index=False)
    ))
    cache_key = (
        "testing_figures_electrodes_5pct_coverage_v3",
        group_a, stage_a, group_b, stage_b, tuple(selected_channels),
        bool(zscore_subjects), int(n_permutations), bool(apply_ica), paired,
        patient_signature,
    )
    cache_entry = st.session_state.get("_testing_figures_electrode_cache")
    if cache_entry and cache_entry.get("key") == cache_key:
        channel_results = cache_entry["channel_results"]
        st.success("Loaded testing-figure electrode statistics from this session.")
    else:
        # The complete no-diagnosis mode renders the electrode/topomap section
        # immediately before this figure lab. Reuse its identical channel tests
        # instead of repeating every permutation for the same stage contrast.
        electrode_entry = st.session_state.get("_electrode_pairwise_hep_cache")
        electrode_key = electrode_entry.get("key") if electrode_entry else None
        matching_electrode_result = bool(
            electrode_key
            and len(electrode_key) >= 10
            and electrode_key[1:9] == (
                group_a, stage_a, group_b, stage_b, tuple(selected_channels),
                zscore_subjects, n_permutations, bool(apply_ica),
            )
            and electrode_key[9] == patient_signature
        )
        if matching_electrode_result:
            channel_results = electrode_entry["channel_results"]
            st.success("Reused electrode statistics from the topomap analysis.")
        else:
            progress = st.progress(0.0, text="Testing electrodes for figure lab...")

            def update_progress(index: int, total: int, channel: str) -> None:
                progress.progress(
                    (index + 1) / max(total, 1),
                    text=f"Testing electrode {index + 1}/{total}: {channel}",
                )

            channel_results = prepare_electrode_contrasts(
                grouped_df, group_a, stage_a, group_b, stage_b,
                selected_channels, zscore_subjects, n_permutations, paired,
                progress_callback=update_progress,
            )
            progress.empty()
        st.session_state["_testing_figures_electrode_cache"] = {
            "key": cache_key,
            "channel_results": channel_results,
        }

    if not channel_results:
        st.warning(
            "No selected electrode has valid traces in at least "
            f"{ELECTRODE_MIN_PATIENT_COVERAGE_FRACTION:.0%} of patients in both "
            "comparison cells."
        )
        return

    summary = _channel_window_statistics(channel_results, time_window)
    if summary.empty:
        st.warning("No per-electrode statistics could be computed for this contrast.")
        return

    significant_window = int(summary["Window significant"].fillna(False).sum())
    significant_cluster = int(summary["FDR significant"].fillna(False).sum())
    metrics = st.columns(4)
    metrics[0].metric("Electrodes shown", len(summary))
    metrics[1].metric("Window q < 0.05", significant_window)
    metrics[2].metric("Cluster + electrode FDR", significant_cluster)
    metrics[3].metric(
        "Window",
        f"{time_window[0]:.2f}-{time_window[1]:.2f}s",
    )

    display_summary = summary.copy()
    for col in ("A coverage", "B coverage"):
        display_summary[col] = display_summary[col].map(
            lambda value: f"{value:.0%}" if pd.notna(value) else ""
        )
    st.dataframe(display_summary, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download testing-figure electrode statistics",
        summary.to_csv(index=False),
        "hep_testing_figures_electrode_statistics.csv",
        "text/csv",
    )

    tabs = st.tabs([
        "Every Electrode",
        "Brain Maps",
        "3D Brain Map",
        "Circle",
        "Window Graph",
    ])
    with tabs[0]:
        import matplotlib.pyplot as plt
        st.plotly_chart(
            plot_electrode_overview(channel_results, label_a, label_b, zscore_subjects),
            use_container_width=True,
            theme=None,
            key="testing_figures_every_electrode_plot",
        )
        red_topomap = make_red_significance_topomap(
            channel_results, label_a, label_b
        )
        st.pyplot(red_topomap, use_container_width=True)
        plt.close(red_topomap)
        st.caption(
            "Small multiples show each electrode's mean HEPs, A-B difference, "
            "and orange cluster-permutation significant time windows. The topomap "
            "uses darker red for smaller p-values and white for p≥0.01."
        )
        time_topomap = make_peak_significance_time_topomap(
            channel_results, label_a, label_b
        )
        st.pyplot(time_topomap, use_container_width=True)
        plt.close(time_topomap)
        st.caption(
            "Jet color shows the strongest significant time; white means no "
            "significant cluster."
        )
    with tabs[1]:
        topomap_fig = make_testing_window_topomaps(
            summary, label_a, label_b, time_window
        )
        st.pyplot(topomap_fig, use_container_width=True)
        import matplotlib.pyplot as plt
        plt.close(topomap_fig)
        st.caption(
            "Left: mean A-B in the selected time window. Right: signed -log10(q) "
            "for the selected-window test. Black rings mark significant electrodes."
        )
    with tabs[2]:
        color_metric = st.radio(
            "3D color metric",
            ["Window mean A-B", "Window t", "Peak |T| in window"],
            horizontal=True,
            key="testing_figures_3d_metric",
        )
        st.plotly_chart(
            plot_3d_electrode_brain_map(
                summary, label_a, label_b, time_window, color_metric
            ),
            use_container_width=True,
            theme=None,
            key="testing_figures_3d_brain_map",
        )
        st.caption(
            "Marker size follows selected-window q-value strength; larger markers "
            "indicate stronger window-level evidence."
        )
    with tabs[3]:
        st.plotly_chart(
            plot_circular_electrode_statistics(
                summary, label_a, label_b, time_window
            ),
            use_container_width=True,
            theme=None,
            key="testing_figures_circle_plot",
        )
        st.caption(
            "The circular layout preserves approximate scalp angle and makes it easy "
            "to scan electrodes with large effects or strong q-values."
        )
    with tabs[4]:
        network_fig, edge_summary = plot_significant_window_network(
            channel_results, summary, label_a, label_b
        )
        st.plotly_chart(
            network_fig,
            use_container_width=True,
            theme=None,
            key="testing_figures_window_network",
        )
        st.caption(
            "Edges connect electrodes whose significant cluster windows overlap in "
            "time. Larger nodes survived the channel-level cluster/FDR criterion."
        )
        if edge_summary.empty:
            st.info("No electrodes had overlapping significant cluster windows.")
        else:
            st.dataframe(edge_summary, use_container_width=True, hide_index=True)


def render_non_diagnosis_complete_mode(
    grouped_df: pd.DataFrame,
    selected_stages: Sequence[str],
    selected_channels: Sequence[str],
    apply_ica: bool = False,
) -> None:
    """Expose every waveform/electrode/statistical view for the no-diagnosis cohort."""
    st.header("No-diagnosis cohort: complete sleep-stage HEP analysis")
    st.caption(
        "All analyses below use only patients with no diagnosis recorded in the "
        "available diagnosis data. Sleep-stage waveform comparisons are paired by "
        "patient and tested with cluster-mass permutations; electrode tests include "
        "electrode-level FDR correction. Electrodes represented in less than 5% of "
        "either comparison cell are excluded from both individual-electrode figures "
        "and the across-electrode average."
    )

    stage_counts = (
        grouped_df.groupby("stage")["patient_id"]
        .nunique()
        .reindex(selected_stages, fill_value=0)
    )
    count_columns = st.columns(max(1, len(selected_stages)))
    for column, stage in zip(count_columns, selected_stages):
        column.metric(
            STAGE_LABELS.get(stage, stage),
            f"{int(stage_counts.get(stage, 0)):,} patients",
        )

    waveform_tab, electrode_tab, brain_map_tab = st.tabs([
        "All sleep-stage waveform statistics",
        "All electrodes + significant topomaps",
        "Brain maps + window statistics",
    ])
    with waveform_tab:
        render_pairwise_waveform_mode(
            grouped_df,
            ["No diagnosis"],
            selected_stages,
            selected_channels,
            apply_ica=apply_ica,
        )
    with electrode_tab:
        render_electrode_topomap_mode(
            grouped_df,
            ["No diagnosis"],
            selected_stages,
            selected_channels,
            apply_ica=apply_ica,
        )
    with brain_map_tab:
        render_testing_figures_mode(
            grouped_df,
            ["No diagnosis"],
            selected_stages,
            selected_channels,
            apply_ica=apply_ica,
        )


def plot_ai_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> go.Figure:
    """Normalized held-out confusion matrix with counts in hover text."""
    labels = np.arange(len(class_names))
    counts = confusion_matrix(y_true, y_pred, labels=labels)
    denominators = counts.sum(axis=1, keepdims=True)
    normalized = counts / np.where(denominators == 0, 1, denominators)
    text = np.empty(counts.shape, dtype=object)
    for row in range(counts.shape[0]):
        for column in range(counts.shape[1]):
            text[row, column] = f"{normalized[row, column]:.2f}<br>n={counts[row, column]}"
    fig = go.Figure(go.Heatmap(
        z=normalized, x=list(class_names), y=list(class_names),
        text=text, texttemplate="%{text}", colorscale="Blues",
        zmin=0, zmax=1, colorbar_title="Row proportion",
    ))
    fig.update_layout(
        xaxis_title="Predicted diagnosis", yaxis_title="True diagnosis",
        height=max(480, 65 * len(class_names) + 180),
        margin=dict(l=180, b=150, t=50, r=30),
    )
    fig.update_xaxes(tickangle=-35)
    return fig


def plot_ai_feature_importance(model_result: dict, title: str, top_n: int = 20) -> go.Figure:
    """Plot Random Forest importance for one feature representation."""
    importance = np.asarray(model_result["model"].feature_importances_, dtype=float)
    names = np.asarray(model_result["feature_names"], dtype=object)
    order = np.argsort(importance)[-min(top_n, len(importance)):]
    fig = go.Figure(go.Bar(
        x=importance[order], y=names[order], orientation="h",
        marker=dict(color=importance[order], colorscale="Viridis"),
    ))
    fig.update_layout(
        title=title, xaxis_title="Random Forest importance",
        yaxis_title="Feature", height=max(430, 25 * len(order) + 120),
        margin=dict(l=210, r=30, t=70, b=60),
    )
    return fig


def render_ai_classification_mode(
    selected_groups: Sequence[str],
    selected_stages: Sequence[str],
) -> None:
    """Train disease and exact-comorbidity classifiers from physiology and AE features."""
    st.header("AI disease and comorbidity-subgroup classification")
    st.caption(
        "Compares interpretable EEG/ECG physiology, AI-selected important features, "
        "autoencoder latent features, and a combined representation on held-out subjects."
    )
    with st.expander("Methodological safeguards", expanded=False):
        st.markdown(
            "- Labels come from the same BDSP I0006 and I0002 EHR diagnosis sources used "
            "by Diagnosis Group Comparison.\n"
            "- Disease classification keeps only subjects belonging to exactly one selected "
            "disease, avoiding ambiguous multi-label targets.\n"
            "- Train/test splitting is subject-level and stratified. Median imputation, "
            "scaling, feature selection, and autoencoder fitting use training subjects only.\n"
            "- Autoencoder features are unsupervised latent representations; the diagnosis "
            "labels are used only by the downstream classifier."
        )

    with st.spinner("Loading cached EEG/ECG physiological features…"):
        feature_df = load_ai_physiological_features(
            tuple(selected_groups), tuple(selected_stages)
        )
    if feature_df.empty:
        st.error(
            "No cached autoencoder/physiological features match the selected cohorts and stages."
        )
        return
    with st.spinner("Joining BDSP diagnoses to feature subjects…"):
        ehr_raw = load_ehr_data(tuple(feature_df.index.astype(str)))
        ehr_subjects = consolidate_ai_ehr(ehr_raw)
    common_subjects = feature_df.index.intersection(ehr_subjects.index)
    if len(common_subjects) < 12:
        st.error("Too few subjects have both physiological features and EHR diagnoses.")
        return
    feature_df = feature_df.loc[common_subjects]
    ehr_subjects = ehr_subjects.loc[common_subjects]

    category_counts: Counter = Counter()
    for categories in ehr_subjects["categories"]:
        category_counts.update(categories)
    available_categories = [
        category for category, count in
        sorted(category_counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= 5
    ]
    overview_columns = st.columns(4)
    overview_columns[0].metric("Feature subjects", len(feature_df))
    overview_columns[1].metric("Engineered features", feature_df.shape[1])
    overview_columns[2].metric("Diagnosis categories", len(available_categories))
    overview_columns[3].metric("Selected stages", len(selected_stages))

    target_kind = st.radio(
        "Classification target",
        ["Different diseases", "One disease and its exact comorbidity subgroups"],
        horizontal=True, key="ai_target_kind",
    )
    min_class_size = int(st.number_input(
        "Minimum subjects per class", 5, 100, 8, 1, key="ai_min_class_size"
    ))

    selected_diseases: List[str] = []
    index_disease: Optional[str] = None
    selected_target_labels: List[str] = []
    if target_kind == "Different diseases":
        labels_with_counts = [
            f"{category} (n={category_counts[category]})"
            for category in available_categories
        ]
        selected_display = st.multiselect(
            "Disease categories to classify",
            labels_with_counts,
            default=labels_with_counts[:min(4, len(labels_with_counts))],
            key="ai_selected_diseases",
        )
        selected_diseases = [value.rsplit(" (n=", 1)[0] for value in selected_display]
        labels = build_ai_target_labels(
            ehr_subjects, "Different diseases", selected_diseases
        )
        counts = labels.value_counts()
        retained = counts[counts >= min_class_size].index.tolist()
        labels = labels[labels.isin(retained)]
        selected_target_labels = retained
        st.caption(
            "Patients matching more than one selected disease are excluded from this "
            "multiclass task rather than assigned an arbitrary primary diagnosis."
        )
    else:
        if not available_categories:
            st.warning("No diagnosis category has enough subjects for subgroup analysis.")
            return
        index_disease = st.selectbox(
            "Index disease", available_categories,
            format_func=lambda category: f"{category} (n={category_counts[category]})",
            key="ai_index_disease",
        )
        labels = build_ai_target_labels(
            ehr_subjects, "One disease and its exact comorbidity subgroups", [],
            index_disease=index_disease,
        )
        subgroup_counts = labels.value_counts()
        available_subgroups = subgroup_counts[
            subgroup_counts >= min_class_size
        ].index.tolist()
        subgroup_display = [
            f"{label} (n={subgroup_counts[label]})" for label in available_subgroups
        ]
        selected_display = st.multiselect(
            "Exact subgroups to classify",
            subgroup_display,
            default=subgroup_display[:min(6, len(subgroup_display))],
            key="ai_selected_subgroups",
        )
        selected_target_labels = [
            value.rsplit(" (n=", 1)[0] for value in selected_display
        ]
        labels = labels[labels.isin(selected_target_labels)]

    if labels.nunique() < 2:
        st.warning(
            "At least two classes must meet the minimum size. Select more categories "
            "or lower the minimum subjects per class."
        )
        return
    class_distribution = labels.value_counts().rename_axis("Class").reset_index(name="N")
    st.dataframe(class_distribution, use_container_width=True, hide_index=True)

    modality_columns = st.columns(2)
    with modality_columns[0]:
        include_eeg = st.checkbox("EEG engineered features", True, key="ai_include_eeg")
    with modality_columns[1]:
        include_ecg = st.checkbox("ECG/HRV engineered features", True, key="ai_include_ecg")
    selected_feature_columns = [
        column for column in feature_df.columns
        if (include_eeg and "__eeg_" in column)
        or (include_ecg and "__ecg_" in column)
    ]
    if not selected_feature_columns:
        st.warning("Select at least one physiological feature modality.")
        return
    model_features = feature_df[selected_feature_columns]

    settings = st.columns(5)
    with settings[0]:
        test_fraction = st.slider("Test fraction", 0.20, 0.40, 0.25, 0.05, key="ai_test_fraction")
    with settings[1]:
        top_feature_count = int(st.number_input(
            "Top features", 2, 50, 15, 1, key="ai_top_features"
        ))
    with settings[2]:
        latent_dim = int(st.number_input(
            "AE dimensions", 2, 16, 6, 1, key="ai_latent_dim"
        ))
    with settings[3]:
        autoencoder_epochs = int(st.number_input(
            "AE epochs", 20, 300, 80, 10, key="ai_epochs"
        ))
    with settings[4]:
        n_estimators = int(st.number_input(
            "RF trees", 100, 1000, 300, 100, key="ai_trees"
        ))

    training_key = (
        "ai_diagnosis_v1", tuple(selected_groups), tuple(selected_stages), target_kind,
        tuple(sorted(selected_target_labels)), tuple(selected_feature_columns),
        test_fraction, top_feature_count, latent_dim, autoencoder_epochs, n_estimators,
    )
    cached_training = st.session_state.get("_ai_diagnosis_training_cache")
    retrain = st.button("Train / retrain AI classifiers", type="primary")
    if retrain:
        try:
            with st.spinner("Training engineered-feature and autoencoder classifiers…"):
                bundle = train_ai_feature_models(
                    model_features, labels, test_fraction, top_feature_count,
                    latent_dim, autoencoder_epochs, n_estimators,
                )
            st.session_state["_ai_diagnosis_training_cache"] = {
                "key": training_key, "bundle": bundle,
            }
            cached_training = st.session_state["_ai_diagnosis_training_cache"]
        except Exception as error:
            st.error(f"AI training failed: {error}")
            return
    if not cached_training or cached_training.get("key") != training_key:
        st.info("Configure the task, then click **Train / retrain AI classifiers**.")
        return
    bundle = cached_training["bundle"]

    result_tabs = st.tabs([
        "Accuracy comparison", "Confusion matrices",
        "Important physiological features", "Autoencoder feature plots",
    ])
    with result_tabs[0]:
        metric_rows = []
        for representation, result in bundle["models"].items():
            metric_rows.extend([
                {"Representation": representation, "Metric": "Accuracy", "Value": result["accuracy"]},
                {"Representation": representation, "Metric": "Balanced accuracy", "Value": result["balanced_accuracy"]},
                {"Representation": representation, "Metric": "Macro F1", "Value": result["macro_f1"]},
            ])
        metrics_df = pd.DataFrame(metric_rows)
        st.plotly_chart(
            px.bar(
                metrics_df, x="Representation", y="Value", color="Metric",
                barmode="group", range_y=[0, 1], text_auto=".3f",
                title="Held-out classification performance by feature representation",
            ),
            use_container_width=True, theme=None,
        )
        st.dataframe(
            metrics_df.pivot(index="Representation", columns="Metric", values="Value")
            .reset_index(),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            f"Subject-level split: {bundle['train_size']} training and "
            f"{bundle['test_size']} held-out test subjects."
        )

    with result_tabs[1]:
        confusion_tabs = st.tabs(list(bundle["models"]))
        for confusion_tab, (representation, result) in zip(
            confusion_tabs, bundle["models"].items()
        ):
            with confusion_tab:
                figure = plot_ai_confusion_matrix(
                    bundle["y_test"], result["prediction"], bundle["class_names"]
                )
                figure.update_layout(title=representation)
                st.plotly_chart(
                    figure, use_container_width=True, theme=None,
                    key=f"ai_confusion_{representation}",
                )

    with result_tabs[2]:
        importance_tabs = st.tabs(list(bundle["models"]))
        for importance_tab, (representation, result) in zip(
            importance_tabs, bundle["models"].items()
        ):
            with importance_tab:
                st.plotly_chart(
                    plot_ai_feature_importance(result, representation),
                    use_container_width=True, theme=None,
                    key=f"ai_importance_{representation}",
                )
        st.subheader("Distribution of an AI-selected physiological feature")
        selected_feature = st.selectbox(
            "Important feature", bundle["top_feature_names"],
            key="ai_feature_distribution_choice",
        )
        distribution = pd.DataFrame({
            "value": bundle["feature_frame"][selected_feature],
            "diagnosis": bundle["labels_text"],
        }).dropna()
        st.plotly_chart(
            px.violin(
                distribution, x="diagnosis", y="value", color="diagnosis",
                box=True, points="outliers",
                title=f"{selected_feature} by diagnosis class",
            ),
            use_container_width=True, theme=None,
        )

    with result_tabs[3]:
        latent = bundle["latent_all"]
        if latent.shape[1] > 2:
            embedding = PCA(n_components=2, random_state=42).fit_transform(latent)
        else:
            embedding = latent[:, :2]
        latent_plot = pd.DataFrame({
            "AE dimension 1": embedding[:, 0],
            "AE dimension 2": embedding[:, 1],
            "diagnosis": bundle["labels_text"].to_numpy(),
            "patient_id": bundle["feature_frame"].index.astype(str),
        })
        st.plotly_chart(
            px.scatter(
                latent_plot, x="AE dimension 1", y="AE dimension 2",
                color="diagnosis", hover_name="patient_id",
                title="Autoencoder latent projection by diagnosis",
            ),
            use_container_width=True, theme=None,
        )
        st.plotly_chart(
            plot_ai_feature_importance(
                bundle["models"]["Autoencoder latent features"],
                "Which autoencoder dimensions classify diagnosis?",
                top_n=len(bundle["latent_names"]),
            ),
            use_container_width=True, theme=None,
        )
        correlations = bundle["feature_latent_correlations"]
        correlation_strength = np.nan_to_num(
            np.nanmax(np.abs(correlations), axis=1), nan=0.0
        )
        strongest = np.argsort(correlation_strength)[-20:]
        correlation_figure = px.imshow(
            correlations[strongest],
            x=bundle["latent_names"],
            y=np.asarray(bundle["all_feature_names"], dtype=object)[strongest],
            color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
            zmin=-1, zmax=1, aspect="auto",
            title="Physiological meaning of autoencoder dimensions (Pearson r)",
        )
        correlation_figure.update_layout(
            height=650, margin=dict(l=220, r=30, t=70, b=70)
        )
        st.plotly_chart(correlation_figure, use_container_width=True, theme=None)


# =============================================================================
# Methodology tab
# =============================================================================

_METRIC_EXPLANATIONS = {
    "EEG T-wave amplitude (mean)": """
**Plain language:** the average EEG signal level (in µV), across the selected
EEG channels and across the time window you chose, computed from each
patient's own averaged heartbeat-evoked-potential (HEP) waveform.

**How it's computed** (`extract_metric`, metric `"EEG T-wave amplitude (mean)"`):
1. Take `hep_data` — the per-channel HEP waveform (already averaged over R-peak-locked
   epochs for that patient/stage; see the "HEP pipeline" section below).
2. Restrict to the channel rows in `ch_names` that match your **EEG channels**
   sidebar selection.
3. Restrict to the time samples where `t_window[0] <= times <= t_window[1]`
   (your **T-wave window start/end** sliders).
4. Convert volts to microvolts (`* 1e6`).
5. `np.nanmean(...)` over the selected channels × selected time samples.
""",
    "EEG T-wave amplitude (peak)": """
**Plain language:** the single largest deflection (positive or negative, in
µV) seen anywhere across the selected EEG channels within the T-wave time
window.

**How it's computed** (`extract_metric`, metric `"EEG T-wave amplitude (peak)"`):
Same channel/time selection and V→µV conversion as the mean metric above,
then `np.nanmax(np.abs(eeg_win))` — the largest absolute value in that
channel × time slice (so it captures whichever direction the deflection
points).
""",
    "ECG T-peak time": """
**Plain language:** how long (in ms) after the R-peak the ECG's T-wave
reaches its maximum, measured on the patient's averaged ECG heartbeat-evoked
waveform.

**How it's computed:** calls `hep_mod.find_ecg_t_peak_time(ecg_hep_data, times, t_window=t_window)`
in `6_hep_group_comparison.py`. That function:
1. Restricts the averaged ECG trace to the T-wave window (`t_window`,
   default `(0.15, 0.50)` s post R-peak, overridable by the sliders).
2. Because the ECG is already polarity-corrected earlier in the pipeline
   (see `fix_inverted_ecg` / the R-peak-based flip logic), it takes the
   **maximum** value in that window (`np.nanargmax`), not the largest
   absolute value.
3. Returns the *time* (in seconds) at which that maximum occurs; the
   dashboard multiplies by 1000 to report milliseconds.
""",
    "N_epochs kept": """
**Plain language:** an approximate count of usable heartbeats found in this
patient's ECG recording for this stage.

**How it's computed — and an important caveat:** `extract_metric` sets this
to `len(rpeaks)`, the number of R-peaks detected by
`detect_rpeaks_robust()` in `6_hep_group_comparison.py`. This is the peak
count that *feeds into* the HEP average, not the "kept after quality
filtering" count from the old dashboard. The actual epoch-level
keep/reject decision (flat/noisy-window and spectral-power-ratio
filtering) happens inside `compute_hep_avg()`'s `_valid_hep_window_mask`
step and that per-epoch keep count is **not** returned by
`get_group_individuals`, so it cannot be reconstructed here — this number
is a proxy, and will generally be **higher** than the true "kept" count
used to build the averaged waveform (see the code comment at the top of
`extract_metric`).
""",
}

_STATS_COLUMN_EXPLANATIONS = [
    ("N", "Number of patients in that group/stage with a non-missing metric value "
          "(`vals = grouped_df.loc[... , metric_col].dropna().values`; `N = len(vals)`)."),
    ("Mean", "Arithmetic mean of the metric across patients in the group: `np.nanmean(vals)`."),
    ("Std", "Sample standard deviation (ddof=1) across patients in the group: `np.nanstd(vals, ddof=1)`. "
            "Left blank (NaN) if the group has fewer than 2 patients."),
    ("Median", "Median metric value across patients in the group: `np.nanmedian(vals)`."),
    ("CI_lo / CI_hi", "A 95% bootstrap confidence interval around the **median** "
        "(`bootstrap_median_ci`, `ci=0.95`), computed by: resampling the group's values "
        "with replacement 1000 times (`n_boot=1000`, fixed random seed 42 for "
        "reproducibility), taking the median of each resample, then reading off the "
        "2.5th and 97.5th percentiles of that distribution of 1000 medians. "
        "Requires at least 2 patients, else NaN."),
    ("Effect_size", "Rank-biserial correlation `r`, a non-parametric effect size derived "
        "from the Mann-Whitney U statistic, comparing this group's values against the "
        "**reference group**'s values (`rank_biserial_r(vals, ref_vals)` = "
        "`2*U/(n1*n2) - 1`). Ranges from -1 to +1; 0 means the two groups' rank "
        "distributions fully overlap, ±1 means complete separation. Not computed for "
        "the reference group itself, or when either group has fewer than 2 patients."),
    ("p_value", "Two-sided Mann-Whitney U test p-value comparing this group's metric "
        "values against the **reference group**'s values "
        "(`scipy.stats.mannwhitneyu(vals, ref_vals, alternative=\"two-sided\")`). "
        "This tests whether one group's values tend to be systematically higher/lower "
        "than the reference group's, without assuming normality."),
    ("q_value", "The p-value after Benjamini-Hochberg false-discovery-rate (FDR) "
        "correction (`benjamini_hochberg`), applied **across all the finite p-values "
        "shown in the table** (i.e. across every group/stage row's comparison to its "
        "reference group in the current table — not per-stage). Controls the expected "
        "proportion of false positives among all the significant results reported, "
        "which is important because many pairwise tests are run at once."),
]


def _render_methodology_tab(metric: str, t_window: tuple) -> None:
    st.subheader("Methodology — what each number means and how it was computed")
    st.caption(
        "Grounded directly in this file's code and in 6_hep_group_comparison.py. "
        f"Currently selected metric: **{metric}**  |  T-wave window: "
        f"**{t_window[0]:.2f}–{t_window[1]:.2f} s** post R-peak."
    )

    st.markdown("### 1. The selected metric")
    for name, text in _METRIC_EXPLANATIONS.items():
        is_active = (name == metric)
        label = f"▶ **{name}**" + ("  ← currently selected" if is_active else "")
        with st.expander(label, expanded=is_active):
            st.markdown(text)
    st.caption(
        "\"Signal quality score\" (kept/total epoch ratio) was removed from the "
        "metric list entirely — `get_group_individuals` does not return a "
        "total-epochs-attempted count, so that ratio cannot be faithfully "
        "reproduced (see the `extract_metric` code comment)."
    )

    st.markdown("### 2. Where the HEP waveform comes from (the pipeline)")
    with st.expander("How hep_data / ecg_hep_data / times / rpeaks are built", expanded=False):
        st.markdown("""
This dashboard does not process any raw signals itself — it calls
`6_hep_group_comparison.get_group_individuals(group, stage, ...)`, which for
every patient in that group/stage runs (via `_process_patient_worker`,
parallelized with `ProcessPoolExecutor`):

1. **Load & channel prep** — unpickle the patient's MNE `raw` object, drop
   non-EEG/ECG channels.
2. **`process_file_data(raw, patient_id)`** — find the ECG channel (name
   contains `ecg`/`ekg`), correct polarity if inverted
   (`fix_inverted_ecg`), clean it (`clean_ecg_high_fidelity`), then detect
   R-peaks with `detect_rpeaks_robust()` (WFDB `xqrs_detect`, refined to
   local maxima and filtered for physiologically implausible RR intervals).
   The epoch window is a fixed constant in this function:
   `minmax = (-0.3, 0.4)` seconds relative to each R-peak.
3. **`process_and_invert_hep(...)`** — calls `compute_hep_avg()` (EEG) and
   `compute_ecg_hep_avg()` (ECG): for every detected R-peak, cut out a
   `[-0.3, +0.4]` s window from the continuous signal, stack all these
   per-heartbeat windows ("epochs"), drop epochs that are flat/too noisy or
   whose high/low spectral power ratio exceeds a threshold
   (`_valid_hep_window_mask`, threshold constant `MAX_SPECTRAL_POWER_RATIO` —
   see code), and average the surviving epochs into **one HEP waveform per
   patient/channel** — this averaged waveform is `hep_data`
   (EEG, shape channels × time) / `ecg_hep_data` (ECG, 1 channel × time),
   with `times` the shared time axis in seconds relative to the R-peak.
4. **Polarity/flip correction** — `flip_eeg_channels_around_r_peak` inspects
   a ±100 ms window around the R-peak on the averaged EEG waveform per
   channel; channels showing a clear negative valley there get inverted
   (multiplied by -1) and the HEP average is recomputed. The list of
   flipped channels is returned as `flipped_channels`.
5. The tuple `(patient_id, hep_data, times, ch_names, rpeaks, ecg_hep_data,
   ecg_ch_names, log_msg, flipped_channels)` is what this dashboard receives
   and what `extract_metric()` reads from — no raw EEG/ECG signal is ever
   held in this dashboard's own memory beyond that per-patient tuple.
""")

    st.markdown("### 3. Statistical Details table columns")
    with st.expander("What each column in the '📋 Statistical Details' tab means", expanded=False):
        st.markdown(
            "Computed per group, per stage, by `compute_group_stats()`, comparing "
            "every non-reference group against the **Reference group** you picked "
            "in the sidebar.\n\n"
            + "\n\n".join(f"**{col}** — {desc}" for col, desc in _STATS_COLUMN_EXPLANATIONS)
        )

    st.markdown("### 4. Kruskal-Wallis test (shown atop the Sleep Stage Comparison panels)")
    with st.expander("Kruskal-Wallis vs. the pairwise Mann-Whitney U above", expanded=False):
        st.markdown("""
Each subplot title in the **📊 Sleep Stage Comparison** tab includes a
Kruskal-Wallis result, e.g. `KW ** (p=0.003)`. This comes from
`scipy.stats.kruskal(*group_vals)` in `plot_stage_comparison()`, run once per
sleep stage across **all** diagnosis groups shown in that stage's panel
(only groups with ≥2 patients are included).

This is an **omnibus** test: it asks "do at least two of these groups differ
in their metric distribution within this sleep stage?" without saying which
pair(s) differ. It is distinct from the **p_value** column in the
'📋 Statistical Details' table, which is a **pairwise** Mann-Whitney U test
of each individual group against the reference group only. A significant
Kruskal-Wallis result does not by itself identify which group is different —
the pairwise table (with its q_value FDR correction) is what does that.
""")


# =============================================================================
# Main app
# =============================================================================

def main() -> None:
    st.set_page_config(page_title="HEP Diagnosis × Sleep Stage", layout="wide")
    st.title("🫀🧠 HEP Diagnosis × Sleep Stage Comparison")
    st.caption(
        "Compare Heartbeat-Evoked Potential T-wave modulation across sleep stages and disease groups"
    )

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Settings")

    available_groups = list_groups(BASE_PATH)
    default_groups = [g for g in DEFAULT_HEP_GROUPS if g in available_groups] or available_groups[:3]
    selected_groups = st.sidebar.multiselect("HEP groups", available_groups, default=default_groups)

    if not selected_groups:
        st.warning("Select at least one HEP group.")
        return

    non_diagnosis_mode = "Non-diagnosis patients — complete analysis"
    analysis_mode = st.sidebar.radio(
        "Analysis mode",
        [
            non_diagnosis_mode,
            "Summary metrics",
            "Pairwise HEP waveforms",
            "Electrode HEP + significant topomaps",
            "Testing figures",
            "AI disease classification",
        ],
        index=0,
        help=(
            "The default non-diagnosis mode combines all sleep-stage waveform tests, "
            "per-electrode analysis, significant scalp topomaps, brain maps, and "
            "window-level statistics."
        ),
    )
    _GROUPING_MODES = [
        "By Diagnosis Category",
        "By Specific Diagnosis",
        "Diagnosis Comorbidity Subgroups",
        "By Age Group",
        "By Sex",
        "Exclusive Diagnosis Category",
    ]
    if analysis_mode == non_diagnosis_mode:
        grouping_mode = "No diagnosis cohort"
        st.sidebar.caption("Grouping fixed to patients with no recorded diagnosis.")
    else:
        grouping_mode = st.sidebar.radio(
            "Grouping mode", _GROUPING_MODES,
            index=_GROUPING_MODES.index("Exclusive Diagnosis Category"),
        )
    min_patients = int(st.sidebar.number_input("Min patients per group", 2, 200, 150, 1))

    available_stages = list_stages(BASE_PATH, selected_groups)
    default_stages = [s for s in STAGE_ORDER if s in available_stages] or available_stages
    selected_stages = st.sidebar.multiselect(
        "Sleep stages", available_stages, default=default_stages,
        format_func=lambda s: STAGE_LABELS.get(s, s),
    )
    if not selected_stages:
        st.warning("Select at least one sleep stage.")
        return

    if analysis_mode == "AI disease classification":
        render_ai_classification_mode(selected_groups, selected_stages)
        return

    hep_mod = _load_hep_module()
    # Also enforce restoration when _load_hep_module itself is returned from
    # Streamlit's resource cache on a hot-reload.
    if getattr(hep_mod, "_original_st_pyplot", None) is not None:
        st.pyplot = hep_mod._original_st_pyplot
    if getattr(hep_mod, "_original_st_plotly_chart", None) is not None:
        st.plotly_chart = hep_mod._original_st_plotly_chart

    if analysis_mode == "Summary metrics":
        t_min = st.sidebar.slider("T-wave window start (s)", 0.0, 0.4, 0.15, 0.01)
        t_max = st.sidebar.slider("T-wave window end (s)", 0.1, 0.8, 0.5, 0.01)
        t_window = (t_min, t_max)
        metric = st.sidebar.selectbox("Metric", [
            "EEG T-wave amplitude (mean)",
            "EEG T-wave amplitude (peak)",
            "ECG T-peak time",
            "N_epochs kept",
            # ponytail: "Signal quality score" (n_epochs_kept / n_epochs_total)
            # dropped — get_group_individuals doesn't return a total-epochs-
            # attempted count, so the ratio can't be faithfully reproduced.
        ])
    else:
        t_window = (0.15, 0.5)
        metric = "EEG T-wave amplitude (mean)"

    # ── Load HEP data via get_group_individuals ─────────────────────────────
    _force_rebuild = st.sidebar.button(
        "🔄 Rebuild cache",
        help="Force reload all pickle files and overwrite each group/stage's individuals_cache*.pkl.",
    )
    _apply_ica = st.sidebar.checkbox(
        "Apply ICA ECG-artifact removal", value=False,
        help="Passed through to get_group_individuals(apply_ica=...).",
    )
    _only_gt10_eeg = st.sidebar.toggle(
        "Only patients with ≥10 standard 10-20 EEG channels",
        value=True,
        help=(
            "A patient must have at least 10 distinct channels from the exact "
            "24-channel 10-20 list shown below. Other EEG labels do not count. "
            "This cohort is stored in a separate disk cache."
        ),
    )
    if _only_gt10_eeg:
        st.sidebar.caption(
            "Active: ≥10 listed 10-20 EEG channels · separate `min10_1020eeg` cache"
        )
    if _force_rebuild:
        st.session_state.pop("_automatic_pairwise_hep_cache", None)
        st.session_state.pop("_automatic_pairwise_electrode_cache", None)
        st.session_state.pop("_electrode_pairwise_hep_cache", None)
        st.session_state.pop("_testing_figures_electrode_cache", None)

    loading_message = (
        "Loading HEP data for patients with ≥10 listed 10-20 EEG channels "
        "(`min10_1020eeg` cache)…"
        if _only_gt10_eeg
        else "Loading HEP data (get_group_individuals)…"
    )
    with st.spinner(loading_message):
        raw_df = load_patient_data(
            hep_mod, selected_groups, selected_stages,
            force_rebuild=_force_rebuild, apply_ica=_apply_ica,
            min_eeg_channels=10 if _only_gt10_eeg else None,
        )

    if _force_rebuild:
        st.sidebar.success("Cache rebuilt.")

    if raw_df.empty:
        st.error("No patient data found for selected groups/stages.")
        return

    channel_filter_text = (
        " with ≥10 listed 10-20 EEG channels" if _only_gt10_eeg else ""
    )
    st.success(
        f"Loaded {len(raw_df)} patient-stage records{channel_filter_text} "
        f"from {raw_df['patient_id'].nunique()} unique patients."
    )

    # Restrict analysis controls to canonical international 10-20 electrodes.
    # Both legacy temporal names (T3/T4/T5/T6) and their modern equivalents
    # (T7/T8/P7/P8) are accepted when present in the processed data.
    montage_1020_channel_order = [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
        "T3", "T7", "C3", "Cz", "C4", "T4", "T8",
        "T5", "P7", "P3", "Pz", "P4", "T6", "P8",
        "O1", "Oz", "O2",
    ]
    montage_1020_by_lower = {
        channel.lower(): channel for channel in montage_1020_channel_order
    }
    available_eeg_channels = sorted({
        montage_1020_by_lower[str(channel).lower()]
        for individual in raw_df["individual"]
        if individual is not None and len(individual) > 3 and individual[3] is not None
        for channel in individual[3]
        if str(channel).lower() in montage_1020_by_lower
    })
    preferred_rank = {
        channel.lower(): index
        for index, channel in enumerate(montage_1020_channel_order)
    }
    available_eeg_channels.sort(
        key=lambda channel: (preferred_rank.get(channel.lower(), len(preferred_rank)), channel.lower())
    )
    selected_channels = st.sidebar.multiselect(
        "EEG channels",
        available_eeg_channels,
        default=available_eeg_channels,
        help=(
            "Only canonical international 10-20 montage electrodes are shown. "
            "All available matching electrodes are selected by default."
        ),
    )
    if not selected_channels and "EEG" in metric:
        selected_channels = available_eeg_channels

    # Keep the individual tuple in the working frame so waveform mode uses
    # the exact same patient/stage records and diagnosis assignments.
    metric_rows = []
    progress = st.progress(0) if analysis_mode == "Summary metrics" else None
    for i, (_, row) in enumerate(raw_df.iterrows()):
        if progress is not None:
            progress.progress((i + 1) / len(raw_df))
        individual = row["individual"]
        if individual is None:
            continue
        val = (
            extract_metric(hep_mod, individual, metric, t_window, selected_channels)
            if analysis_mode == "Summary metrics" else np.nan
        )
        metric_rows.append({
            "group": row["group"],
            "stage": row["stage"],
            "patient_id": row["patient_id"],
            "metric_value": val,
            "individual": individual,
        })
    if progress is not None:
        progress.empty()
    metric_df_base = pd.DataFrame(metric_rows)
    if analysis_mode == "Summary metrics":
        metric_df_base = metric_df_base.dropna(subset=["metric_value"])

    if metric_df_base.empty:
        message = (
            "No valid metric values extracted. Check metric / channel settings."
            if analysis_mode == "Summary metrics"
            else "No valid HEP waveforms were loaded. Check channel settings."
        )
        st.error(message)
        return

    # Load EHR / clinical data
    harvard_pids = raw_df[
        raw_df["group"] == "Harvard_Electroencephalography"
    ]["patient_id"].unique().tolist()

    with st.spinner("Loading EHR diagnosis data..."):
        ehr_df = load_ehr_data(tuple(harvard_pids)) if harvard_pids else pd.DataFrame()

    clinical_df = load_cobrad_clinical()

    # ── Grouping ──────────────────────────────────────────────────────────────
    grouped_df = pd.DataFrame()
    comorbidity_overview_df = pd.DataFrame()
    comorbidity_index_group = None
    comorbidity_subgroup_order: List[str] = []

    if analysis_mode == non_diagnosis_mode:
        # Match the dashboard's established "No diagnosis" semantics: patients
        # with an empty diagnosis list, including cohorts without mapped EHR
        # diagnosis rows, form one cohort for paired sleep-stage comparisons.
        grouped_df = select_non_diagnosis_cohort(metric_df_base, ehr_df)
        if grouped_df["patient_id"].nunique() < min_patients:
            grouped_df = pd.DataFrame()

    elif grouping_mode == "By Diagnosis Category":
        if ehr_df.empty:
            st.warning("No EHR data available for Harvard group.")
        else:
            cat_counts: Counter = Counter()
            for cats in ehr_df["categories"]:
                cat_counts.update(cats)
            avail_cats = sorted(
                [c for c, n in cat_counts.items() if n >= min_patients],
                key=lambda c: -cat_counts[c],
            )
            default_cats = avail_cats[:min(4, len(avail_cats))]
            sel_cat_labels = st.sidebar.multiselect(
                "Select diagnosis categories",
                [f"{c} (n={cat_counts[c]})" for c in avail_cats],
                default=[f"{c} (n={cat_counts[c]})" for c in default_cats],
            )
            selected_cats = [s.split(" (n=")[0] for s in sel_cat_labels]
            merged = metric_df_base.merge(
                ehr_df[["patient_id", "categories"]], on="patient_id", how="left"
            )
            merged["categories"] = merged["categories"].apply(
                lambda x: x if isinstance(x, list) else []
            )
            rows = []
            for _, row in merged.iterrows():
                cats = row["categories"]
                matched = [c for c in selected_cats if c in cats]
                if matched:
                    for c in matched:
                        r = row.to_dict()
                        r["diagnosis_group"] = c
                        rows.append(r)
                else:
                    r = row.to_dict()
                    r["diagnosis_group"] = "No diagnosis"
                    rows.append(r)
            grouped_df = pd.DataFrame(rows)
            if not grouped_df.empty:
                counts = grouped_df.groupby("diagnosis_group")["patient_id"].nunique()
                valid = counts[counts >= min_patients].index
                grouped_df = grouped_df[grouped_df["diagnosis_group"].isin(valid)]

    elif grouping_mode == "By Specific Diagnosis":
        if ehr_df.empty:
            st.warning("No EHR data available for Harvard group.")
        else:
            all_dx = sorted({dx for dxs in ehr_df["dx_names"] for dx in dxs})
            search = st.sidebar.text_input("Search diagnosis names:")
            filtered_dx = (
                [d for d in all_dx if search.lower() in d.lower()] if search else all_dx[:80]
            )
            sel_dx = st.sidebar.multiselect(
                "Select diagnoses", filtered_dx,
                default=filtered_dx[:2] if filtered_dx else [],
            )
            merged = metric_df_base.merge(
                ehr_df[["patient_id", "dx_names"]], on="patient_id", how="left"
            )
            merged["dx_names"] = merged["dx_names"].apply(
                lambda x: x if isinstance(x, list) else []
            )
            rows = []
            for _, row in merged.iterrows():
                for dx in sel_dx:
                    if dx in row["dx_names"]:
                        r = row.to_dict()
                        r["diagnosis_group"] = dx[:60]
                        rows.append(r)
            grouped_df = pd.DataFrame(rows)
            if not grouped_df.empty:
                counts = grouped_df.groupby("diagnosis_group")["patient_id"].nunique()
                valid = counts[counts >= min_patients].index
                grouped_df = grouped_df[grouped_df["diagnosis_group"].isin(valid)]

    elif grouping_mode == "Diagnosis Comorbidity Subgroups":
        if ehr_df.empty:
            st.warning("No EHR data available for Harvard group.")
        else:
            cat_counts2: Counter = Counter()
            for cats in ehr_df["categories"]:
                cat_counts2.update(cats)
            avail2 = [
                c for c, n in sorted(cat_counts2.items(), key=lambda x: -x[1])
                if n >= min_patients
            ]
            idx_dx = st.sidebar.selectbox("Index diagnosis", avail2 or ["None"])
            max_subgroup_lines = int(st.sidebar.number_input(
                "Max subgroup lines",
                min_value=2,
                max_value=50,
                value=12,
                step=1,
                help=(
                    "Shows the largest exact comorbidity subgroups in the overview. "
                    "Lines can then be hidden or shown from the Plotly legend without "
                    "refreshing the page."
                ),
            ))
            merged = metric_df_base.merge(
                ehr_df[["patient_id", "categories"]], on="patient_id", how="left"
            )
            merged["categories"] = merged["categories"].apply(
                lambda x: x if isinstance(x, list) else []
            )
            index_rows = merged[
                merged["categories"].apply(lambda cats: idx_dx in cats)
            ].copy()
            comorbidity_index_group = f"{idx_dx} (all)"
            rows = []
            for _, row in merged.iterrows():
                cats = row["categories"]
                if idx_dx not in cats:
                    continue
                other = sorted(c for c in cats if c != idx_dx)
                n_c = len(other)
                if n_c == 0:
                    label = f"{idx_dx} only"
                else:
                    label = f"{idx_dx} + " + " + ".join(other)
                r = row.to_dict()
                r["diagnosis_group"] = label
                rows.append(r)
            grouped_df = pd.DataFrame(rows)
            if not grouped_df.empty:
                counts = grouped_df.groupby("diagnosis_group")["patient_id"].nunique()
                valid = counts[counts >= min_patients].index
                grouped_df = grouped_df[grouped_df["diagnosis_group"].isin(valid)]
                subgroup_counts = (
                    grouped_df.groupby("diagnosis_group")["patient_id"]
                    .nunique()
                    .sort_values(ascending=False)
                )
                comorbidity_subgroup_order = subgroup_counts.head(max_subgroup_lines).index.tolist()
                grouped_df = grouped_df[
                    grouped_df["diagnosis_group"].isin(comorbidity_subgroup_order)
                ]
                if not index_rows.empty:
                    index_rows["diagnosis_group"] = comorbidity_index_group
                    comorbidity_overview_df = pd.concat(
                        [
                            index_rows,
                            grouped_df[grouped_df["diagnosis_group"].isin(comorbidity_subgroup_order)],
                        ],
                        ignore_index=True,
                    )

    elif grouping_mode == "By Age Group":
        if clinical_df.empty or "age" not in clinical_df.columns:
            st.warning("No clinical data available for age grouping.")
        else:
            age_map = dict(zip(clinical_df["record_id"].astype(str), clinical_df["age"]))
            rows = []
            for _, row in metric_df_base.iterrows():
                age = age_map.get(str(row["patient_id"]), np.nan)
                if not np.isfinite(float(age)) if age is not None else True:
                    label = "Unknown"
                elif float(age) < 40:
                    label = "<40 years"
                elif float(age) < 60:
                    label = "40–60 years"
                elif float(age) < 75:
                    label = "60–75 years"
                else:
                    label = "≥75 years"
                r = row.to_dict()
                r["diagnosis_group"] = label
                rows.append(r)
            grouped_df = pd.DataFrame(rows)
            if not grouped_df.empty:
                counts = grouped_df.groupby("diagnosis_group")["patient_id"].nunique()
                valid = counts[counts >= min_patients].index
                grouped_df = grouped_df[grouped_df["diagnosis_group"].isin(valid)]

    elif grouping_mode == "By Sex":
        if clinical_df.empty or "sex" not in clinical_df.columns:
            st.warning("No clinical data available for sex grouping.")
        else:
            sex_map = dict(zip(clinical_df["record_id"].astype(str), clinical_df["sex"]))
            rows = []
            for _, row in metric_df_base.iterrows():
                s = sex_map.get(str(row["patient_id"]), None)
                label = "Male" if s == 1 else ("Female" if s == 0 else "Unknown")
                r = row.to_dict()
                r["diagnosis_group"] = label
                rows.append(r)
            grouped_df = pd.DataFrame(rows)
            if not grouped_df.empty:
                counts = grouped_df.groupby("diagnosis_group")["patient_id"].nunique()
                valid = counts[counts >= min_patients].index
                grouped_df = grouped_df[grouped_df["diagnosis_group"].isin(valid)]

    elif grouping_mode == "Exclusive Diagnosis Category":
        if ehr_df.empty:
            st.warning("No EHR data available for Harvard group.")
        else:
            dx_map = dict(zip(ehr_df["patient_id"], ehr_df["dx_names"]))
            cat_keys = list(_DIAG_CATEGORIES.keys())
            rows = []
            for _, row in metric_df_base.iterrows():
                dx_names = dx_map.get(row["patient_id"], [])
                assigned = next(
                    (cat for cat in cat_keys if any(_DIAG_CATEGORIES[cat](dx) for dx in dx_names)),
                    "No diagnosis",
                )
                r = row.to_dict()
                r["diagnosis_group"] = assigned
                rows.append(r)
            grouped_df = pd.DataFrame(rows)
            if not grouped_df.empty:
                counts = grouped_df.groupby("diagnosis_group")["patient_id"].nunique()
                valid = counts[counts >= min_patients].index
                grouped_df = grouped_df[grouped_df["diagnosis_group"].isin(valid)]

    if grouped_df.empty:
        st.warning(
            "No groups meet the minimum patient threshold. "
            "Try lowering **Min patients per group**."
        )
        return

    all_groups = sorted(grouped_df["diagnosis_group"].unique())

    st.markdown(
        f"**Groups:** {', '.join(all_groups)} | "
        f"**Stages:** {', '.join(STAGE_LABELS.get(s, s) for s in selected_stages)}"
    )

    if analysis_mode == non_diagnosis_mode:
        render_non_diagnosis_complete_mode(
            grouped_df,
            selected_stages,
            selected_channels,
            apply_ica=_apply_ica,
        )
        return

    if analysis_mode == "Pairwise HEP waveforms":
        render_pairwise_waveform_mode(
            grouped_df, all_groups, selected_stages, selected_channels,
            apply_ica=_apply_ica,
        )
        return

    if analysis_mode == "Electrode HEP + significant topomaps":
        render_electrode_topomap_mode(
            grouped_df, all_groups, selected_stages, selected_channels,
            apply_ica=_apply_ica,
        )
        return

    if analysis_mode == "Testing figures":
        render_testing_figures_mode(
            grouped_df, all_groups, selected_stages, selected_channels,
            apply_ica=_apply_ica,
        )
        return

    ref_group = st.sidebar.selectbox("Reference group", all_groups, index=0)

    if (
        analysis_mode == "Summary metrics"
        and grouping_mode == "Diagnosis Comorbidity Subgroups"
        and not comorbidity_overview_df.empty
        and comorbidity_index_group
        and comorbidity_subgroup_order
    ):
        st.subheader("Comorbidity Contribution Overview")
        st.caption(
            "The black line is every patient with the selected index diagnosis. "
            "Colored lines are the largest exact comorbidity subgroups. The upper "
            "panel reports each subgroup's delta from the all-index mean and a "
            "Mann-Whitney p/q value versus the remaining index-diagnosis patients "
            "for that sleep stage. Click legend items to hide/show subgroup lines "
            "without a Streamlit rerun."
        )
        fig_overview = plot_comorbidity_overview(
            comorbidity_overview_df,
            selected_stages,
            metric,
            comorbidity_index_group,
            comorbidity_subgroup_order,
        )
        st.plotly_chart(fig_overview, use_container_width=True, theme=None)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Sleep Stage Comparison",
        "📈 Cross-Stage Comparison",
        "🗺️ Group × Stage Heatmap",
        "📋 Statistical Details",
        "🔵 Patient Overview",
        "ℹ️ Methodology",
    ])

    with tab1:
        fig1 = plot_stage_comparison(grouped_df, selected_stages, metric, grouping_mode)
        st.plotly_chart(fig1, use_container_width=True, theme=None)

    with tab2:
        fig2 = plot_cross_stage(grouped_df, selected_stages, metric)
        st.plotly_chart(fig2, use_container_width=True, theme=None)

    with tab3:
        fig3 = plot_heatmap(grouped_df, selected_stages, ref_group)
        st.plotly_chart(fig3, use_container_width=True, theme=None)

    with tab4:
        st.subheader("Statistical Details")
        all_stat_rows = []
        for stage in selected_stages:
            stage_df = grouped_df[grouped_df["stage"] == stage].copy()
            if stage_df.empty:
                continue
            stat_df = compute_group_stats(stage_df, "metric_value", ref_group)
            stat_df.insert(0, "Stage", STAGE_LABELS.get(stage, stage))
            all_stat_rows.append(stat_df)
        if all_stat_rows:
            full_stat = pd.concat(all_stat_rows, ignore_index=True)
            full_stat["Mean"] = full_stat["Mean"].round(4)
            full_stat["Median"] = full_stat["Median"].round(4)
            full_stat["Effect_size"] = full_stat["Effect_size"].round(3)
            full_stat["p_value"] = full_stat["p_value"].apply(
                lambda p: format_p(p) if isinstance(p, float) and np.isfinite(p) else "NA"
            )
            full_stat["q_value"] = full_stat["q_value"].apply(
                lambda p: format_p(p) if isinstance(p, float) and np.isfinite(p) else "NA"
            )
            st.dataframe(full_stat, use_container_width=True)
            csv = full_stat.to_csv(index=False)
            st.download_button("⬇️ Download CSV", csv, "hep_stats.csv", "text/csv")
        else:
            st.info("No statistics to display.")

    with tab5:
        st.subheader("Patient Overview — Individual Data Points")
        stage_color_map = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(selected_stages)}
        fig5 = go.Figure()
        for si, stage in enumerate(selected_stages):
            sdf = grouped_df[grouped_df["stage"] == stage]
            for gi, group in enumerate(all_groups):
                gdf = sdf[sdf["diagnosis_group"] == group]
                if gdf.empty:
                    continue
                # ponytail: use go.Box with boxpoints="all" — plotly handles jitter natively
                fig5.add_trace(go.Box(
                    y=gdf["metric_value"].values,
                    x=[group] * len(gdf),
                    name=STAGE_LABELS.get(stage, stage),
                    legendgroup=stage,
                    showlegend=(gi == 0),
                    marker=dict(color=stage_color_map[stage], size=4, opacity=0.6),
                    line=dict(color=stage_color_map[stage]),
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=0,
                    text=gdf["patient_id"].astype(str),
                    hovertemplate="Patient: %{text}<br>Value: %{y:.4f}<extra></extra>",
                ))
        unit5 = _metric_unit(metric)
        fig5.update_layout(
            title="Individual Patient Values by Group and Stage",
            title_font_size=16,
            font_size=12,
            boxmode="group",
            xaxis_title="Diagnosis Group",
            yaxis_title=f"{metric} ({unit5})" if unit5 else metric,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_color="black",
            xaxis=dict(showgrid=True, gridcolor="#eee"),
            yaxis=dict(showgrid=True, gridcolor="#eee"),
            legend_title="Sleep Stage",
        )
        st.plotly_chart(fig5, use_container_width=True, theme=None)

        st.markdown("**Sample of underlying row data** (a few patients from a couple of groups)")
        sample_groups = all_groups[:2]
        sample_df = (
            grouped_df[grouped_df["diagnosis_group"].isin(sample_groups)]
            .groupby("diagnosis_group", group_keys=False)
            .head(5)
            .sort_values(["diagnosis_group", "patient_id"])
        )
        st.dataframe(sample_df, use_container_width=True)

    with tab6:
        _render_methodology_tab(metric, t_window)


if __name__ == "__main__":
    main()
