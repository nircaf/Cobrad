"""
COBRAD Diagnosis ML Classification Dashboard.

PhD-oriented ML classification dashboard:
- loads COBRAD clinical data and diagnosis labels;
- supports disease selection sorted by patient count;
- exclusive / non-exclusive patient filtering;
- trains multiple classifiers (LR, RF, XGBoost, SVM, NN+Autoencoder);
- shows confusion matrices, ROC curves, feature importance;
- visualises autoencoder latent space with group separation stats.

Run:
  streamlit run 15_diagnosis_ml_classification_dashboard.py
"""

from __future__ import annotations

import hashlib
import os
import re
import glob as _glob
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False


# =============================================================================
# Constants
# =============================================================================
BASE_PATH = Path("/storage/pblab_shared_data/Nir/Cobrad")
CLINICAL_FILE = BASE_PATH / "COBRAD_clinical_24022025.xlsx"
PARQUET_CACHE = BASE_PATH / "cache" / "14_autoencoder_clusters"
RANDOM_STATE = 42
DEFAULT_GROUPS: Tuple[str, ...] = (
    "The_Human_Sleep_Project",
    "Young_The_Human_Sleep_Project",
    "Harvard_Electroencephalography",
)

CLINICAL_NUMERIC_COLS = [
    "age", "mmse", "moca", "cdr_sum_of_boxes", "ApoE4",
]
CLINICAL_CAT_COLS = ["sex"]

# EHR data paths (Harvard Electroencephalography study)
_EHR_BASE = "/storage/pblab_shared_data/Nir/Cobrad/EDF_Format/Harvard_Electroencephalography/EHR"
_EHR_DIAG_BASE = os.path.join(
    _EHR_BASE, "data_Structured", "Parquet", "Parquet", "bdsp_i0006_diagnosis"
)
_EHR_I0002_ICD10_BASE = os.path.join(
    _EHR_BASE, "I0002-EHR", "I0002_structured", "icd10_codes_nax_2024_parquet"
)
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
# Helpers
# =============================================================================
def _hash_config(config: Dict[str, Any]) -> str:
    """Return a short stable hash for a config dict (for session caching)."""
    encoded = repr(sorted(config.items())).encode("utf-8")
    return hashlib.md5(encoded).hexdigest()[:10]


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


# =============================================================================
# EHR helpers (BDSP I0006 + I0002 diagnosis data)
# =============================================================================
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
# Data loading
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

    # numeric coercion
    for c in ["age", "mmse", "moca", "cdr_sum_of_boxes", "ApoE4",
              "Class_bvftd", "Class_LBD"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "sex" in df.columns:
        df["sex"] = _safe_encode_sex(df["sex"])

    # binarise class columns
    for c in ["Class_bvftd", "Class_LBD"]:
        if c in df.columns:
            df[c] = (df[c].fillna(0) > 0).astype(int)

    return df


@st.cache_data(show_spinner=False)
def load_icd9_diagnoses() -> pd.DataFrame:
    """Load and pivot the previous ICD9 sheet to per-patient binary columns."""
    if not CLINICAL_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_excel(CLINICAL_FILE, sheet_name="previous ICD9")
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # try to find columns
    col_id = None
    col_dx = None
    for c in df.columns:
        cl = str(c).strip().lower()
        if col_id is None and "record" in cl and "id" in cl:
            col_id = c
        if col_dx is None and ("diagnosis" in cl or "free_text" in cl or "free text" in cl):
            col_dx = c
    if col_id is None or col_dx is None:
        return pd.DataFrame()

    df = df[[col_id, col_dx]].dropna()
    df.columns = ["record_id", "diagnosis_free_text"]
    df["record_id"] = df["record_id"].astype(str).str.strip()
    df["diagnosis_free_text"] = (
        df["diagnosis_free_text"].astype(str).str.strip().str.title()
    )
    df = df[df["diagnosis_free_text"].str.len() > 0]

    # count diagnoses
    counts = df.groupby("diagnosis_free_text")["record_id"].nunique()
    valid_dx = counts[counts >= 2].index.tolist()
    if not valid_dx:
        return pd.DataFrame()

    df = df[df["diagnosis_free_text"].isin(valid_dx)]
    pivot = (
        df.assign(value=1)
          .pivot_table(
              index="record_id",
              columns="diagnosis_free_text",
              values="value",
              aggfunc="max",
              fill_value=0,
          )
          .astype(int)
          .reset_index()
    )
    pivot.columns.name = None
    return pivot


@st.cache_data(show_spinner=False)
def _get_parquet_groups() -> List[str]:
    """Return all group values available in the parquet cache."""
    if not PARQUET_CACHE.exists():
        return list(DEFAULT_GROUPS)
    files = sorted(PARQUET_CACHE.glob("*.parquet"))
    groups: set = set()
    for f in files[:8]:
        try:
            df = pd.read_parquet(f, columns=["group"])
            groups.update(df["group"].dropna().unique().tolist())
        except Exception:
            continue
    return sorted(groups) if groups else list(DEFAULT_GROUPS)


@st.cache_data(show_spinner=False)
def load_ehr_diagnosis_df(patient_ids: Tuple[str, ...]) -> Optional[pd.DataFrame]:
    """Map EEG patient IDs → BDSP EHR diagnoses. Returns DataFrame indexed by patient_id."""
    if not patient_ids:
        return None
    pid_to_bdsp: Dict[str, int] = {}
    for pid in patient_ids:
        bdsp = _pid_to_bdsp_ehr(pid)
        if bdsp is not None:
            pid_to_bdsp[pid] = bdsp
    if not pid_to_bdsp:
        return None
    ehr_map = _load_ehr_diagnosis_map(target_ids=set(pid_to_bdsp.values()))
    rows = []
    for pid in patient_ids:
        bdsp = pid_to_bdsp.get(pid)
        dx_names = ehr_map.get(bdsp, []) if bdsp is not None else []
        cats = _assign_categories(dx_names)
        rows.append({"patient_id": pid, "bdsp_id": bdsp, "dx_names": dx_names, "categories": cats})
    df = pd.DataFrame(rows)
    for cat in _DIAG_CATEGORIES:
        col = f"dx_cat_{cat.replace(' ', '_').replace('/', '_')}"
        df[col] = df["categories"].apply(lambda cats, c=cat: int(c in cats))
    return df


@st.cache_data(show_spinner=False)
def load_eeg_ecg_features(groups: Tuple[str, ...] = DEFAULT_GROUPS) -> Optional[pd.DataFrame]:
    """Load EEG/ECG features from the parquet cache and aggregate per patient."""
    if not PARQUET_CACHE.exists():
        return None
    files = sorted(PARQUET_CACHE.glob("*.parquet"))
    if not files:
        return None

    frames: List[pd.DataFrame] = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f))
        except Exception:
            continue
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)

    if "patient_id" not in df.columns:
        return None

    # filter to selected groups
    if groups and "group" in df.columns:
        df = df[df["group"].isin(groups)]
    if df.empty:
        return None

    # numeric feature columns
    feat_cols: List[str] = []
    for c in df.columns:
        cl = str(c)
        if cl.startswith("eeg_") and cl.endswith("_mean"):
            feat_cols.append(c)
        elif cl.startswith("ecg_"):
            feat_cols.append(c)
    if not feat_cols:
        return None

    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    keep_cols = ["patient_id"] + feat_cols
    if "group" in df.columns:
        keep_cols.append("group")
    if "stage" in df.columns:
        keep_cols.append("stage")
    df = df[keep_cols].copy()

    agg_dict = {c: "mean" for c in feat_cols}
    agg = (
        df.groupby("patient_id", dropna=False)
          .agg(agg_dict)
          .reset_index()
    )

    # try to map to COBRAD record_id format
    agg["record_id"] = agg["patient_id"].apply(_extract_cobrad_id)
    return agg


# =============================================================================
# Feature construction
# =============================================================================
def build_feature_matrix(
    clinical_df: pd.DataFrame,
    icd9_df: pd.DataFrame,
    eeg_df: Optional[pd.DataFrame],
    modalities: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Assemble the feature matrix and label option dict."""
    if clinical_df.empty:
        return pd.DataFrame(), {}

    base = clinical_df[["record_id"]].copy()

    cols_used: List[str] = []

    if "Clinical" in modalities:
        for c in CLINICAL_NUMERIC_COLS + CLINICAL_CAT_COLS:
            if c in clinical_df.columns:
                base[c] = clinical_df[c]
                cols_used.append(c)

    if eeg_df is not None and "record_id" in eeg_df.columns:
        eeg_cobrad = eeg_df.dropna(subset=["record_id"]).copy()
        if not eeg_cobrad.empty:
            eeg_band_cols = [c for c in eeg_cobrad.columns
                             if c.startswith("eeg_") and c.endswith("_mean")]
            ecg_cols = [c for c in eeg_cobrad.columns if c.startswith("ecg_")]

            if "EEG bands (mean)" in modalities and eeg_band_cols:
                base = base.merge(
                    eeg_cobrad[["record_id"] + eeg_band_cols],
                    on="record_id", how="left",
                )
                cols_used.extend(eeg_band_cols)
            if "ECG (HRV)" in modalities and ecg_cols:
                base = base.merge(
                    eeg_cobrad[["record_id"] + ecg_cols],
                    on="record_id", how="left",
                )
                cols_used.extend(ecg_cols)

    # drop columns with >40% missing
    if cols_used:
        miss = base[cols_used].isna().mean()
        keep_cols = [c for c in cols_used if miss.get(c, 0.0) <= 0.4]
    else:
        keep_cols = []

    X = base[["record_id"] + keep_cols].copy()

    # fillna with median
    for c in keep_cols:
        if X[c].dtype.kind in "biufc":
            med = X[c].median()
            X[c] = X[c].fillna(med if pd.notna(med) else 0.0)

    X = X.set_index("record_id")

    y_options: Dict[str, Any] = {}
    if "Class_bvftd" in clinical_df.columns:
        y_options["Class_bvftd"] = clinical_df.set_index("record_id")["Class_bvftd"]
    if "Class_LBD" in clinical_df.columns:
        y_options["Class_LBD"] = clinical_df.set_index("record_id")["Class_LBD"]
    y_options["ICD9_diagnoses"] = icd9_df

    return X, y_options


def build_labels(
    clinical_df: pd.DataFrame,
    icd9_df: pd.DataFrame,
    label_source: str,
    selected_classes: List[str],
    exclusive: bool,
    task_type: str,
) -> Tuple[pd.Series, List[str]]:
    """Return (y series indexed by record_id, class_name_list)."""
    if clinical_df.empty or not selected_classes:
        return pd.Series(dtype=int), []

    df = clinical_df.set_index("record_id")

    if label_source == "Class columns (bvFTD/LBD)":
        # collect each Class column
        available = [c for c in selected_classes if c in df.columns]
        if not available:
            return pd.Series(dtype=int), []

        # exclusivity: patient must have only one positive Class_ (across ALL Class_ columns)
        all_class_cols = [c for c in df.columns if c.startswith("Class_")]
        if exclusive and len(all_class_cols) > 1:
            row_sum = df[all_class_cols].sum(axis=1)
            mask_exclusive = (row_sum <= 1)
        else:
            mask_exclusive = pd.Series(True, index=df.index)

        if task_type.startswith("Binary"):
            # Binary: union of selected positives vs rest
            pos_mask = pd.Series(False, index=df.index)
            for c in available:
                pos_mask |= (df[c] > 0)
            y = pos_mask.astype(int)
            y = y[mask_exclusive]
            return y, ["Other", "+".join(available)]
        else:
            # Multi-class: 0 = none, 1..k = each selected class
            y = pd.Series(0, index=df.index, dtype=int)
            for i, c in enumerate(available, start=1):
                y[df[c] > 0] = i
            y = y[mask_exclusive]
            return y, ["None"] + available

    else:  # ICD9
        if icd9_df.empty:
            return pd.Series(dtype=int), []
        icd9 = icd9_df.set_index("record_id")
        # align: every clinical patient gets row, missing dx => 0
        icd9 = icd9.reindex(df.index, fill_value=0)
        available = [c for c in selected_classes if c in icd9.columns]
        if not available:
            return pd.Series(dtype=int), []

        if exclusive and len(available) > 1:
            row_sum = icd9[available].sum(axis=1)
            mask_exclusive = (row_sum <= 1)
        else:
            mask_exclusive = pd.Series(True, index=icd9.index)

        if task_type.startswith("Binary"):
            pos_mask = pd.Series(False, index=icd9.index)
            for c in available:
                pos_mask |= (icd9[c] > 0)
            y = pos_mask.astype(int)
            y = y[mask_exclusive]
            return y, ["Other", "+".join(available)]
        else:
            y = pd.Series(0, index=icd9.index, dtype=int)
            for i, c in enumerate(available, start=1):
                y[icd9[c] > 0] = i
            y = y[mask_exclusive]
            return y, ["None"] + available


def build_labels_ehr(
    ehr_df: Optional[pd.DataFrame],
    grouping_mode: str,
    selected_classes: List[str],
    exclusive: bool,
    task_type: str,
) -> Tuple[pd.Series, List[str]]:
    """Build labels from EHR diagnosis data indexed by patient_id."""
    if ehr_df is None or ehr_df.empty or not selected_classes:
        return pd.Series(dtype=int), []

    df = ehr_df.set_index("patient_id")

    if grouping_mode in ("By Diagnosis Category", "Exclusive Diagnosis Category"):
        if exclusive:
            mask_exclusive = df["categories"].apply(
                lambda cats: sum(1 for c in selected_classes if c in cats) <= 1
            )
        else:
            mask_exclusive = pd.Series(True, index=df.index)

        if task_type.startswith("Binary"):
            pos_mask = df["categories"].apply(lambda cats: any(c in cats for c in selected_classes))
            y = pos_mask.astype(int)
            y = y[mask_exclusive]
            class_names = ["Other", "+".join(selected_classes)]
        else:
            y = pd.Series(0, index=df.index, dtype=int)
            for i, c in enumerate(selected_classes, start=1):
                y[df["categories"].apply(lambda cats, _c=c: _c in cats)] = i
            y = y[mask_exclusive]
            class_names = ["None"] + selected_classes
    else:  # By Specific Diagnosis
        if exclusive:
            mask_exclusive = df["dx_names"].apply(
                lambda dxs: sum(1 for d in selected_classes if d in dxs) <= 1
            )
        else:
            mask_exclusive = pd.Series(True, index=df.index)

        if task_type.startswith("Binary"):
            pos_mask = df["dx_names"].apply(lambda dxs: any(d in dxs for d in selected_classes))
            y = pos_mask.astype(int)
            y = y[mask_exclusive]
            class_names = ["Other", "+".join(selected_classes[:2]) if len(selected_classes) > 1 else selected_classes[0]]
        else:
            y = pd.Series(0, index=df.index, dtype=int)
            for i, d in enumerate(selected_classes, start=1):
                y[df["dx_names"].apply(lambda dxs, _d=d: _d in dxs)] = i
            y = y[mask_exclusive]
            class_names = ["None"] + selected_classes

    return y, class_names


def build_feature_matrix_ehr(
    eeg_df: pd.DataFrame,
    modalities: List[str],
) -> pd.DataFrame:
    """Build feature matrix indexed by patient_id from EEG/ECG parquet features."""
    if eeg_df is None or eeg_df.empty:
        return pd.DataFrame()
    feat_cols: List[str] = []
    for c in eeg_df.columns:
        if "EEG bands (mean)" in modalities and c.startswith("eeg_") and c.endswith("_mean"):
            feat_cols.append(c)
        elif "ECG (HRV)" in modalities and c.startswith("ecg_"):
            feat_cols.append(c)
    if not feat_cols:
        return pd.DataFrame()
    X = eeg_df[["patient_id"] + feat_cols].copy().set_index("patient_id")
    for c in feat_cols:
        if X[c].dtype.kind in "biufc":
            med = X[c].median()
            X[c] = X[c].fillna(med if pd.notna(med) else 0.0)
    return X


# =============================================================================
# PyTorch autoencoder
# =============================================================================
if TORCH_AVAILABLE:
    class DenseAutoencoder(nn.Module):
        """Compact dense autoencoder for tabular features."""

        def __init__(self, n_features: int, latent_dim: int = 4) -> None:
            super().__init__()
            h1 = max(32, n_features // 2)
            h2 = max(16, latent_dim * 4)
            self.encoder = nn.Sequential(
                nn.Linear(n_features, h1), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(h1, h2), nn.ReLU(),
                nn.Linear(h2, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, h2), nn.ReLU(),
                nn.Linear(h2, h1), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(h1, n_features),
            )

        def forward(self, x):  # noqa: D401
            return self.decoder(self.encoder(x))

        def encode(self, x):
            return self.encoder(x)


def train_autoencoder_model(
    X_scaled: np.ndarray,
    latent_dim: int = 4,
    epochs: int = 100,
) -> Tuple[Any, np.ndarray]:
    """Train a dense autoencoder; return (model, latent_2d)."""
    if not TORCH_AVAILABLE:
        return None, np.zeros((len(X_scaled), 2))

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    n_features = X_scaled.shape[1]
    model = DenseAutoencoder(n_features=n_features, latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    Xt = torch.tensor(X_scaled, dtype=torch.float32)
    ds = TensorDataset(Xt, Xt)
    batch_size = min(32, max(4, len(Xt) // 4))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, _yb in loader:
            optimizer.zero_grad()
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        latent = model.encode(Xt).cpu().numpy()

    if latent.shape[1] > 2:
        latent_2d = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(latent)
    elif latent.shape[1] == 2:
        latent_2d = latent
    else:
        # pad to 2D
        latent_2d = np.column_stack([latent[:, 0], np.zeros(len(latent))])

    return model, latent_2d


# =============================================================================
# Classifier orchestration
# =============================================================================
def _make_classifier(name: str, multiclass: bool, balanced: bool):
    """Return a fresh classifier instance for the given algorithm name."""
    cw = "balanced" if balanced else None
    if name == "Logistic Regression":
        return LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE,
            class_weight=cw,
            multi_class="auto",
        )
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            class_weight=cw, n_jobs=1,
        )
    if name == "XGBoost" and XGBOOST_AVAILABLE:
        return XGBClassifier(
            n_estimators=100,
            use_label_encoder=False,
            eval_metric="mlogloss" if multiclass else "logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
        )
    if name == "SVM":
        return SVC(
            kernel="rbf", probability=True,
            random_state=RANDOM_STATE,
            class_weight=cw,
        )
    return None


def _nn_autoencoder_classifier(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, n_classes: int, epochs: int = 80,
) -> Tuple[np.ndarray, np.ndarray]:
    """Autoencoder pretraining + linear head classifier. Returns (preds, probs)."""
    if not TORCH_AVAILABLE:
        return np.zeros(len(X_test)), np.zeros((len(X_test), n_classes))

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    n_features = X_train.shape[1]
    latent_dim = max(2, min(8, n_features // 3))
    ae = DenseAutoencoder(n_features=n_features, latent_dim=latent_dim)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    Xt = torch.tensor(X_train, dtype=torch.float32)
    ds = TensorDataset(Xt, Xt)
    bs = min(16, max(4, len(Xt) // 4))
    loader = DataLoader(ds, batch_size=bs, shuffle=True)

    ae.train()
    for _ in range(epochs):
        for xb, _yb in loader:
            opt.zero_grad()
            recon = ae(xb)
            loss = crit(recon, xb)
            loss.backward()
            opt.step()

    head = nn.Linear(latent_dim, n_classes)
    head_opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()

    yt = torch.tensor(y_train, dtype=torch.long)
    ae.eval()
    with torch.no_grad():
        z_train = ae.encode(Xt)

    head.train()
    for _ in range(epochs):
        head_opt.zero_grad()
        logits = head(z_train)
        loss = ce(logits, yt)
        loss.backward()
        head_opt.step()

    head.eval()
    with torch.no_grad():
        Xtest = torch.tensor(X_test, dtype=torch.float32)
        z_test = ae.encode(Xtest)
        logits = head(z_test)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
    return preds, probs


def run_all_classifiers(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    selected_algos: List[str],
    cv_folds: int,
    multiclass: bool,
    balanced: bool,
    feature_names: List[str],
) -> Dict[str, Any]:
    """Train and evaluate each selected classifier; return results dict."""
    results: Dict[str, Any] = {}

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    n_classes = int(max(y_train.max(), y_test.max())) + 1

    cv_scoring = "f1_macro" if multiclass else "roc_auc"
    cv_folds_safe = max(2, min(cv_folds, int(np.bincount(y_train).min())))
    skf = StratifiedKFold(n_splits=cv_folds_safe, shuffle=True, random_state=RANDOM_STATE)

    for algo in selected_algos:
        if algo == "Neural Network + Autoencoder":
            if not TORCH_AVAILABLE:
                continue
            try:
                preds, probs = _nn_autoencoder_classifier(
                    X_train_s, y_train, X_test_s, n_classes=n_classes,
                )
                metrics = _compute_metrics(y_test, preds, probs, multiclass)
                results[algo] = {
                    "model": None,
                    "scaler": scaler,
                    "y_pred": preds,
                    "y_prob": probs,
                    "metrics": metrics,
                    "cv_scores": np.array([]),
                    "feature_names": feature_names,
                    "is_nn": True,
                }
            except Exception as e:
                results[algo] = {"error": str(e)}
            continue

        clf = _make_classifier(algo, multiclass=multiclass, balanced=balanced)
        if clf is None:
            continue

        try:
            clf.fit(X_train_s, y_train)
            y_pred = clf.predict(X_test_s)
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_test_s)
            elif hasattr(clf, "decision_function"):
                df_score = clf.decision_function(X_test_s)
                if df_score.ndim == 1:
                    y_prob = np.column_stack([1 - _sigmoid(df_score), _sigmoid(df_score)])
                else:
                    y_prob = _softmax(df_score)
            else:
                y_prob = np.zeros((len(y_test), n_classes))

            metrics = _compute_metrics(y_test, y_pred, y_prob, multiclass)

            # cross-validation
            try:
                cv_clf = _make_classifier(algo, multiclass=multiclass, balanced=balanced)
                cv_scores = cross_val_score(
                    cv_clf, X_train_s, y_train,
                    cv=skf, scoring=cv_scoring, n_jobs=1,
                )
            except Exception:
                cv_scores = np.array([])

            results[algo] = {
                "model": clf,
                "scaler": scaler,
                "y_pred": y_pred,
                "y_prob": y_prob,
                "metrics": metrics,
                "cv_scores": cv_scores,
                "feature_names": feature_names,
                "is_nn": False,
            }
        except Exception as e:
            results[algo] = {"error": str(e)}

    return results


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    multiclass: bool,
) -> Dict[str, float]:
    """Compute accuracy/precision/recall/f1/auc safely."""
    metrics: Dict[str, float] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    avg = "macro" if multiclass else "binary"
    try:
        metrics["precision"] = float(precision_score(
            y_true, y_pred, average=avg, zero_division=0,
        ))
    except Exception:
        metrics["precision"] = 0.0
    try:
        metrics["recall"] = float(recall_score(
            y_true, y_pred, average=avg, zero_division=0,
        ))
    except Exception:
        metrics["recall"] = 0.0
    try:
        metrics["f1"] = float(f1_score(
            y_true, y_pred, average=avg, zero_division=0,
        ))
    except Exception:
        metrics["f1"] = 0.0

    try:
        if multiclass:
            metrics["auc"] = float(roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro",
            ))
        else:
            if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
                metrics["auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                metrics["auc"] = float(roc_auc_score(y_true, y_prob.ravel()))
    except Exception:
        metrics["auc"] = float("nan")

    return metrics


# =============================================================================
# Group separation statistics (PERMANOVA / PERMDISP)
# =============================================================================
def group_separation_stats(
    latent: np.ndarray,
    group_labels: np.ndarray,
    n_perms: int = 299,
) -> Dict[str, float]:
    """PERMANOVA and PERMDISP on Euclidean distance matrix."""
    groups = np.unique(group_labels)
    n = len(latent)
    if len(groups) < 2 or n < 4:
        return {}

    D2 = cdist(latent, latent, metric="euclidean") ** 2
    idx_i, idx_j = np.triu_indices(n, k=1)
    d2_upper = D2[idx_i, idx_j]
    k = len(groups)

    def _perm_f(labels):
        same = labels[idx_i] == labels[idx_j]
        ss_w = d2_upper[same].sum()
        ss_t = d2_upper.sum()
        df_b, df_w = k - 1, n - k
        if df_w <= 0 or ss_w == 0:
            return 0.0
        return ((ss_t - ss_w) / df_b) / (ss_w / df_w)

    def _disp_f(labels):
        cents = {g: latent[labels == g].mean(axis=0) for g in np.unique(labels)}
        disp = np.array(
            [np.linalg.norm(latent[i] - cents[labels[i]]) for i in range(n)]
        )
        gm = disp.mean()
        uniq = np.unique(labels)
        ss_b = sum(
            np.sum(labels == g) * (disp[labels == g].mean() - gm) ** 2
            for g in uniq
        )
        ss_w = sum(
            np.sum((disp[labels == g] - disp[labels == g].mean()) ** 2)
            for g in uniq
        )
        df_b, df_w = len(uniq) - 1, n - len(uniq)
        if df_w <= 0 or ss_w == 0:
            return 0.0
        return (ss_b / df_b) / (ss_w / df_w)

    obs_p, obs_d = _perm_f(group_labels), _disp_f(group_labels)
    rng = np.random.default_rng(42)
    pp, dp = [], []
    for _ in range(n_perms):
        pl = rng.permutation(group_labels)
        pp.append(_perm_f(pl))
        dp.append(_disp_f(pl))
    return {
        "permanova_F": float(obs_p),
        "permanova_p": float((np.array(pp) >= obs_p).sum() + 1) / (n_perms + 1),
        "permdisp_F": float(obs_d),
        "permdisp_p": float((np.array(dp) >= obs_d).sum() + 1) / (n_perms + 1),
        "n_groups": int(k),
        "n_samples": int(n),
    }


# =============================================================================
# Plotting
# =============================================================================
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    normalized: bool = False,
) -> go.Figure:
    """Plotly confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    if normalized:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_n = cm.astype(float) / row_sums
        text = [[f"{cm_n[i, j]:.2f}" for j in range(cm.shape[1])]
                for i in range(cm.shape[0])]
        z = cm_n
    else:
        text = [[str(cm[i, j]) for j in range(cm.shape[1])]
                for i in range(cm.shape[0])]
        z = cm

    acc = float(accuracy_score(y_true, y_pred))
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=class_names, y=class_names,
        text=text, texttemplate="%{text}",
        colorscale="Blues",
        showscale=True,
    ))
    fig.update_layout(
        title=f"Confusion matrix (acc={acc:.3f})",
        xaxis_title="Predicted", yaxis_title="True",
        height=400,
    )
    return fig


def plot_roc_curves(
    results: Dict[str, Any],
    y_test: np.ndarray,
    multiclass: bool = False,
) -> go.Figure:
    """ROC curves for all algorithms."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Chance",
        showlegend=True,
    ))

    for algo, res in results.items():
        if "error" in res or "y_prob" not in res:
            continue
        y_prob = res["y_prob"]
        auc = res["metrics"].get("auc", float("nan"))
        try:
            if multiclass:
                # macro-average OvR
                n_classes = y_prob.shape[1]
                all_fpr = np.linspace(0, 1, 100)
                mean_tpr = np.zeros_like(all_fpr)
                count = 0
                for k in range(n_classes):
                    y_bin = (y_test == k).astype(int)
                    if y_bin.sum() == 0:
                        continue
                    fpr, tpr, _ = roc_curve(y_bin, y_prob[:, k])
                    mean_tpr += np.interp(all_fpr, fpr, tpr)
                    count += 1
                if count > 0:
                    mean_tpr /= count
                    fig.add_trace(go.Scatter(
                        x=all_fpr, y=mean_tpr, mode="lines",
                        name=f"{algo} (AUC={auc:.3f})",
                    ))
            else:
                if y_prob.ndim == 2:
                    score = y_prob[:, 1]
                else:
                    score = y_prob
                fpr, tpr, _ = roc_curve(y_test, score)
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"{algo} (AUC={auc:.3f})",
                ))
        except Exception:
            continue

    fig.update_layout(
        title="ROC curves",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        height=500,
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98),
    )
    return fig


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    model_name: str,
    top_n: int = 20,
) -> Optional[go.Figure]:
    """Horizontal bar chart of feature importance."""
    if model is None:
        return None
    importance: Optional[np.ndarray] = None
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coefs = np.asarray(model.coef_)
        if coefs.ndim == 1:
            importance = np.abs(coefs)
        else:
            importance = np.abs(coefs).mean(axis=0)
    if importance is None or len(importance) == 0:
        return None

    n = min(top_n, len(importance))
    order = np.argsort(importance)[-n:]
    feats = [feature_names[i] for i in order]
    vals = importance[order]

    fig = go.Figure(go.Bar(
        x=vals, y=feats, orientation="h",
        marker=dict(color=vals, colorscale="Viridis"),
    ))
    fig.update_layout(
        title=f"Feature importance — {model_name}",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=max(350, 22 * n + 100),
    )
    return fig


def plot_cv_comparison(results: Dict[str, Any]) -> go.Figure:
    """Bar chart of CV scores with error bars."""
    names, means, stds = [], [], []
    for algo, res in results.items():
        if "error" in res:
            continue
        cv = res.get("cv_scores")
        if cv is None or len(cv) == 0:
            continue
        names.append(algo)
        means.append(float(np.mean(cv)))
        stds.append(float(np.std(cv)))

    if not names:
        fig = go.Figure()
        fig.update_layout(title="No CV scores available", height=300)
        return fig

    fig = go.Figure(data=[go.Bar(
        x=names, y=means,
        error_y=dict(type="data", array=stds, visible=True),
        marker=dict(color=means, colorscale="Plasma"),
    )])
    fig.update_layout(
        title="Cross-validation scores (mean ± std)",
        yaxis_title="Score",
        height=420,
    )
    return fig


# =============================================================================
# Main Streamlit app
# =============================================================================
def main() -> None:
    st.set_page_config(
        page_title="COBRAD Diagnosis ML Classification",
        layout="wide",
        page_icon="🧠",
    )
    st.title("🧠 COBRAD Diagnosis ML Classification")
    st.caption(
        "AI/DL classification of dementia diagnoses from "
        "EEG, ECG, and clinical features"
    )

    # ----- Data source selector (must be before data loading) -----
    st.sidebar.header("🗂️ Data Source")
    available_groups = _get_parquet_groups()
    default_sel = [g for g in DEFAULT_GROUPS if g in available_groups]
    if not default_sel:
        default_sel = available_groups[:2] if len(available_groups) >= 2 else available_groups
    selected_groups = tuple(st.sidebar.multiselect(
        "EEG/ECG source groups",
        options=available_groups,
        default=default_sel,
        help="Which sleep study groups to use for EEG/ECG features. Filtered from parquet cache.",
    ))

    # ----- Load data -----
    with st.spinner("Loading EEG/ECG features..."):
        eeg_df = load_eeg_ecg_features(groups=selected_groups if selected_groups else DEFAULT_GROUPS)

    # Load EHR diagnosis data for the loaded patients
    ehr_df: Optional[pd.DataFrame] = None
    if eeg_df is not None and not eeg_df.empty and "patient_id" in eeg_df.columns:
        with st.spinner("Loading EHR diagnosis data..."):
            pids = tuple(sorted(eeg_df["patient_id"].dropna().unique().tolist()))
            ehr_df = load_ehr_diagnosis_df(pids)

    # Also load COBRAD clinical (optional, as additional feature source)
    clinical_df = load_cobrad_clinical()
    icd9_df = load_icd9_diagnoses()

    # ----- Sidebar: Diagnosis Grouping -----
    st.sidebar.header("📊 Diagnosis Grouping")

    grouping_mode = st.sidebar.radio(
        "Grouping mode",
        ["By Diagnosis Category", "By Specific Diagnosis"],
        index=0,
    )
    min_group_size = int(st.sidebar.number_input("Min patients per group", 2, 200, 3, 1))

    # Build diagnosis options from EHR data
    selected_classes: List[str] = []
    if ehr_df is not None and not ehr_df.empty:
        if grouping_mode == "By Diagnosis Category":
            cat_counts: Counter = Counter()
            for cats in ehr_df["categories"]:
                for c in cats:
                    cat_counts[c] += 1
            available_cats = sorted(
                [c for c, n in cat_counts.items() if n >= min_group_size],
                key=lambda c: -cat_counts[c],
            )
            default_cats = available_cats[:min(4, len(available_cats))]
            cat_options = [f"{c} (n={cat_counts[c]})" for c in available_cats]
            default_cat_opts = [f"{c} (n={cat_counts[c]})" for c in default_cats]
            sel_cat_strs = st.sidebar.multiselect(
                "Select diagnosis categories (sorted by N patients)",
                options=cat_options,
                default=default_cat_opts,
            )
            selected_classes = [s.split(" (n=")[0] for s in sel_cat_strs]

            # Show prevalence table in sidebar
            if cat_counts:
                st.sidebar.caption(
                    "Top diagnoses: " + ", ".join(
                        f"{c}={n}" for c, n in cat_counts.most_common(5)
                    )
                )
        else:  # By Specific Diagnosis
            all_dx: List[str] = sorted(set(
                dx for dxs in ehr_df["dx_names"] for dx in dxs
            ))
            search_term = st.sidebar.text_input("Search diagnosis names (filter):", key="dx15_search")
            filtered_dx = [d for d in all_dx if search_term.lower() in d.lower()] if search_term else all_dx[:80]
            selected_classes = st.sidebar.multiselect(
                "Select specific diagnoses",
                options=filtered_dx,
                default=[],
            )
    else:
        st.sidebar.warning("No EHR diagnosis data available for loaded patients.")

    exclusive = st.sidebar.checkbox(
        "Exclusive patients only",
        value=False,
        help="Keep only patients matched to exactly one of the selected groups (excludes comorbidities).",
    )

    task_type = st.sidebar.radio(
        "Classification task",
        ["Binary (disease vs rest)", "Multi-class"],
        index=0,
    )

    st.sidebar.header("🔬 Features")
    available_modalities: List[str] = []
    if eeg_df is not None and not eeg_df.empty:
        eeg_band_cols = [c for c in eeg_df.columns if c.startswith("eeg_") and c.endswith("_mean")]
        ecg_cols = [c for c in eeg_df.columns if c.startswith("ecg_")]
        if eeg_band_cols:
            available_modalities.append("EEG bands (mean)")
        if ecg_cols:
            available_modalities.append("ECG (HRV)")
    modalities = st.sidebar.multiselect(
        "Feature modalities",
        options=available_modalities,
        default=available_modalities,
    )

    st.sidebar.header("🤖 Algorithms")
    algo_options = ["Logistic Regression", "Random Forest", "SVM"]
    if XGBOOST_AVAILABLE:
        algo_options.append("XGBoost")
    if TORCH_AVAILABLE:
        algo_options.append("Neural Network + Autoencoder")
    selected_algos = st.sidebar.multiselect(
        "Algorithms",
        options=algo_options,
        default=algo_options[:3],
    )
    test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    cv_folds = st.sidebar.slider("CV folds", 2, 10, 5)
    seed = int(st.sidebar.number_input("Random seed", 0, 999, 42))

    run_btn = st.sidebar.button("▶ Run Classification", type="primary")

    # ----- Overview header -----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("EEG patients", f"{len(eeg_df) if eeg_df is not None else 0}")
    ehr_matched = int((ehr_df["bdsp_id"].notna()).sum()) if ehr_df is not None and not ehr_df.empty else 0
    c2.metric("EHR-matched", f"{ehr_matched}")
    if ehr_df is not None and not ehr_df.empty:
        ehr_with_dx = int((ehr_df["dx_names"].apply(len) > 0).sum())
        c3.metric("With diagnoses", f"{ehr_with_dx}")
    else:
        c3.metric("With diagnoses", "0")
    c4.metric("Selected groups", f"{len(selected_classes)}")

    if ehr_df is None or ehr_df.empty:
        st.info(
            "ℹ️ EHR diagnosis data could not be loaded. "
            "Check that the BDSP EHR parquet files exist at the expected path."
        )

    if not selected_classes:
        st.warning("Select at least one diagnosis group in the sidebar to proceed.")
        return
    if not modalities:
        st.warning("Select at least one feature modality.")
        return
    if not selected_algos:
        st.warning("Select at least one algorithm.")
        return

    # ----- Build features and labels -----
    X_full = build_feature_matrix_ehr(eeg_df, modalities)
    y, class_names = build_labels_ehr(
        ehr_df=ehr_df,
        grouping_mode=grouping_mode,
        selected_classes=selected_classes,
        exclusive=exclusive,
        task_type=task_type,
    )

    if X_full.empty or y.empty:
        st.error("Could not assemble features or labels. Check selections.")
        return

    # align X and y on patient_id
    common = X_full.index.intersection(y.index)
    X_full = X_full.loc[common]
    y = y.loc[common]

    # Drop rows with all-NaN features
    X_full = X_full.dropna(how="all")
    y = y.loc[X_full.index]
    # one more fillna pass with column median (in case modality-only NaNs)
    for c in X_full.columns:
        if X_full[c].dtype.kind in "biufc":
            med = X_full[c].median()
            X_full[c] = X_full[c].fillna(med if pd.notna(med) else 0.0)

    # ----- Always-on overview tab -----
    st.subheader("📈 Class distribution")
    vc = y.value_counts().sort_index()
    vc_df = pd.DataFrame({
        "Class": [class_names[i] if i < len(class_names) else str(i) for i in vc.index],
        "N patients": vc.values,
    })
    fig_dist = px.bar(
        vc_df, x="Class", y="N patients",
        color="Class", text="N patients",
        title="Patient counts per class",
    )
    fig_dist.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig_dist, use_container_width=True)

    n_per_class = vc.min() if len(vc) > 0 else 0
    if n_per_class < 10:
        st.warning(
            f"⚠️ Minimum class size = {int(n_per_class)} patients. "
            "Results may be unstable. Class-balanced weighting will be applied."
        )

    # ----- Decide problem type -----
    multiclass = (y.nunique() > 2)
    balanced = True

    # ----- Run / cache results -----
    config = {
        "label_source": grouping_mode,
        "selected_classes": tuple(sorted(selected_classes)),
        "exclusive": bool(exclusive),
        "task_type": task_type,
        "modalities": tuple(sorted(modalities)),
        "selected_algos": tuple(sorted(selected_algos)),
        "test_size": float(test_size),
        "cv_folds": int(cv_folds),
        "seed": int(seed),
        "n_samples": int(len(y)),
        "n_features": int(X_full.shape[1]),
    }
    cfg_hash = _hash_config(config)
    cache_key = f"results_{cfg_hash}"

    if run_btn or (cache_key in st.session_state):
        if cache_key not in st.session_state:
            if y.nunique() < 2:
                st.error("Need at least 2 distinct classes to train a classifier.")
                return

            # train/test split
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_full.values, y.values,
                    test_size=test_size,
                    random_state=seed,
                    stratify=y.values,
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_full.values, y.values,
                    test_size=test_size,
                    random_state=seed,
                )

            with st.spinner("Training classifiers..."):
                results = run_all_classifiers(
                    X_train, y_train, X_test, y_test,
                    selected_algos=selected_algos,
                    cv_folds=cv_folds,
                    multiclass=multiclass,
                    balanced=balanced,
                    feature_names=list(X_full.columns),
                )

            ae_data: Dict[str, Any] = {}
            if TORCH_AVAILABLE and X_full.shape[1] >= 2 and len(X_full) >= 4:
                with st.spinner("Training autoencoder for latent visualization..."):
                    sc = StandardScaler()
                    X_all_s = sc.fit_transform(X_full.values)
                    latent_dim = max(2, min(4, X_full.shape[1] // 3))
                    _ae_model, latent_2d = train_autoencoder_model(
                        X_all_s, latent_dim=latent_dim, epochs=120,
                    )
                    ae_data = {
                        "latent_2d": latent_2d,
                        "labels": y.values,
                        "class_names": class_names,
                    }

            st.session_state[cache_key] = {
                "results": results,
                "y_test": y_test,
                "X_test": X_test,
                "X_train": X_train,
                "y_train": y_train,
                "multiclass": multiclass,
                "class_names": class_names,
                "feature_names": list(X_full.columns),
                "ae_data": ae_data,
                "config": config,
            }

        bundle = st.session_state[cache_key]
        _render_results(bundle)
    else:
        st.info(
            "Configure options in the sidebar and click **▶ Run Classification** "
            "to train models."
        )

        # show data overview correlation regardless
        with st.expander("Feature preview", expanded=False):
            st.dataframe(X_full.head(20), use_container_width=True)


def _render_results(bundle: Dict[str, Any]) -> None:
    """Render the tabbed result view."""
    results = bundle["results"]
    y_test = bundle["y_test"]
    multiclass = bundle["multiclass"]
    class_names = bundle["class_names"]
    feature_names = bundle["feature_names"]
    ae_data = bundle.get("ae_data", {})

    tabs = st.tabs([
        "Overview",
        "Confusion Matrices",
        "ROC Curves",
        "Feature Importance",
        "Autoencoder Latent Space",
        "Cross-Validation",
    ])

    # ---------- Tab 1: Overview ----------
    with tabs[0]:
        st.subheader("Metric summary")
        rows = []
        for algo, res in results.items():
            if "error" in res:
                rows.append({"Algorithm": algo, "Error": res["error"]})
                continue
            m = res["metrics"]
            rows.append({
                "Algorithm": algo,
                "Accuracy": round(m["accuracy"], 3),
                "Precision": round(m["precision"], 3),
                "Recall": round(m["recall"], 3),
                "F1": round(m["f1"], 3),
                "AUC": round(m["auc"], 3) if not np.isnan(m["auc"]) else None,
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # correlation between features and label
        try:
            X_train = bundle["X_train"]
            y_train = bundle["y_train"]
            df_corr = pd.DataFrame(X_train, columns=feature_names)
            df_corr["__label__"] = y_train
            corrs = df_corr.corr(numeric_only=True)["__label__"].drop("__label__")
            corrs = corrs.reindex(corrs.abs().sort_values(ascending=False).index)
            top = corrs.head(15)
            fig_corr = px.imshow(
                top.values.reshape(-1, 1),
                labels=dict(color="Pearson r"),
                y=top.index,
                x=["label"],
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
            )
            fig_corr.update_layout(
                title="Top-15 feature ↔ label correlations",
                height=max(350, 22 * len(top) + 80),
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        except Exception as e:
            st.info(f"Feature-label correlation not available: {e}")

    # ---------- Tab 2: Confusion Matrices ----------
    with tabs[1]:
        st.subheader("Confusion matrices")
        algos_ok = [a for a, r in results.items() if "error" not in r]
        if not algos_ok:
            st.warning("No successful models to display.")
        else:
            ncols = min(2, len(algos_ok))
            cols = st.columns(ncols)
            for i, algo in enumerate(algos_ok):
                with cols[i % ncols]:
                    st.markdown(f"#### {algo}")
                    res = results[algo]
                    fig_cm = plot_confusion_matrix(
                        y_test, res["y_pred"], class_names,
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                    m = res["metrics"]
                    st.caption(
                        f"Accuracy={m['accuracy']:.3f} | "
                        f"Precision={m['precision']:.3f} | "
                        f"Recall={m['recall']:.3f} | "
                        f"F1={m['f1']:.3f} | "
                        f"AUC={m['auc']:.3f}"
                    )

    # ---------- Tab 3: ROC Curves ----------
    with tabs[2]:
        st.subheader("ROC curves")
        ok = {a: r for a, r in results.items() if "error" not in r}
        if ok:
            fig_roc = plot_roc_curves(ok, y_test, multiclass=multiclass)
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.warning("No models with probabilities available.")

    # ---------- Tab 4: Feature Importance ----------
    with tabs[3]:
        st.subheader("Feature importance")
        any_shown = False
        for algo, res in results.items():
            if "error" in res:
                continue
            model = res.get("model")
            fig_imp = plot_feature_importance(
                model, feature_names, model_name=algo, top_n=20,
            )
            if fig_imp is None:
                continue
            any_shown = True
            with st.expander(f"{algo}", expanded=False):
                st.plotly_chart(fig_imp, use_container_width=True)
        if not any_shown:
            st.info(
                "None of the selected models expose feature importances "
                "(or the NN/AE head was used)."
            )

    # ---------- Tab 5: Autoencoder Latent Space ----------
    with tabs[4]:
        st.subheader("Autoencoder latent space")
        if not TORCH_AVAILABLE:
            st.info("PyTorch not available — skip this tab.")
        elif not ae_data:
            st.info("Autoencoder not trained (insufficient samples or features).")
        else:
            latent_2d = ae_data["latent_2d"]
            labels = ae_data["labels"]
            names = ae_data["class_names"]
            label_names = [
                names[int(l)] if int(l) < len(names) else str(int(l))
                for l in labels
            ]
            df_plot = pd.DataFrame({
                "z1": latent_2d[:, 0],
                "z2": latent_2d[:, 1],
                "group": label_names,
            })
            # no patient_id available at this point (one row per sample), fall back to row count
            _group_counts = df_plot["group"].value_counts()
            df_plot["group"] = df_plot["group"].map(
                lambda g: f"{g} (n={_group_counts.get(g, 0)})"
            )
            fig_lat = px.scatter(
                df_plot, x="z1", y="z2", color="group",
                title="2D latent projection of autoencoder",
                height=500,
            )
            st.plotly_chart(fig_lat, use_container_width=True)

            stats_dict = group_separation_stats(latent_2d, labels)
            if stats_dict:
                cA, cB, cC, cD = st.columns(4)
                cA.metric("PERMANOVA F", f"{stats_dict['permanova_F']:.2f}")
                cB.metric("PERMANOVA p", f"{stats_dict['permanova_p']:.3f}")
                cC.metric("PERMDISP F", f"{stats_dict['permdisp_F']:.2f}")
                cD.metric("PERMDISP p", f"{stats_dict['permdisp_p']:.3f}")

                # latent boxplot per group, dim 1
                fig_box = px.box(
                    df_plot, x="group", y="z1", color="group",
                    title="Latent dimension 1 by group",
                )
                fig_box.update_layout(showlegend=False, height=380)
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("Not enough samples/groups for PERMANOVA/PERMDISP.")

    # ---------- Tab 6: Cross-Validation ----------
    with tabs[5]:
        st.subheader("Cross-validation")
        ok = {a: r for a, r in results.items() if "error" not in r}
        if ok:
            fig_cv = plot_cv_comparison(ok)
            st.plotly_chart(fig_cv, use_container_width=True)

            cv_rows = []
            for algo, res in ok.items():
                cv = res.get("cv_scores")
                if cv is None or len(cv) == 0:
                    continue
                row = {"Algorithm": algo}
                for i, s in enumerate(cv, start=1):
                    row[f"Fold {i}"] = round(float(s), 3)
                row["Mean"] = round(float(np.mean(cv)), 3)
                row["Std"] = round(float(np.std(cv)), 3)
                cv_rows.append(row)
            if cv_rows:
                st.dataframe(pd.DataFrame(cv_rows), use_container_width=True)
        else:
            st.warning("No CV results available.")


if __name__ == "__main__":
    main()
