"""HEP vs ICA vs CFAR phenotype-analysis dashboard.

Matched-cohort Streamlit dashboard for the 500-patient six-channel HSP sample
in ``paper utils/Paper CFA/hep_cleaning_compare.parquet``.  The same HEP is
represented three ways: uncleaned HEP, ECG-informed ICA, and population CFAR.

Run:
    streamlit run 24_hep_ica_cfar_comparison_dashboard.py
"""
from __future__ import annotations

import ast
import json
import pickle
import re
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from mne import create_info
from mne.channels import make_standard_montage
from mne.viz import plot_topomap
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    mean_absolute_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from statsmodels.stats.multitest import multipletests

ROOT = Path("/storage/pblab_shared_data2/Nir/Cobrad")
CFA_DIR = ROOT / "paper utils" / "Paper CFA"
FULL_HEP_FILE = CFA_DIR / "hep_ica_cfar_all.parquet"
HEP_FILE = FULL_HEP_FILE if FULL_HEP_FILE.exists() else CFA_DIR / "hep_cleaning_compare.parquet"
DEMO_FILE = CFA_DIR / "demographics_combined.parquet"
COBRAD_FILE = ROOT / "COBRAD_clinical_24022025.xlsx"
COBRAD_PICKLES = ROOT / "pickles_sleep_stage" / "EDF"
SOURCE_FILE = CFA_DIR / "edf_source_features.parquet"
SOURCE_LABELS = {
    "ANPHY-Sleep": "ANPHY-Sleep",
    "EDF": "EDF",
    "Berkeley_data": "Berkeley data",
    "The_Human_Sleep_Project": "Human Sleep Project",
    "Young_The_Human_Sleep_Project": "Young HSP",
    "Harvard_Electroencephalography": "Harvard EEG",
}
SITES = ["F3", "F4", "C3", "C4", "O1", "O2"]
METHODS = {"HEP": "hep_raw", "ICA": "hep_ica", "CFAR": "hep_cfar"}
METHOD_COLORS = {"HEP": "#D55E00", "ICA": "#0072B2", "CFAR": "#009E73"}
RANDOM_STATE = 42
DISEASE_GROUPS = {
    "Cardiometabolic / vascular": {
        "Heart Failure", "Coronary Artery Disease", "Atrial Fibrillation",
        "Hypertension", "Heart Transplant", "Stroke / Cerebrovascular", "Diabetes",
        "Obesity", "Kidney Disease", "Anemia",
    },
    "Neuropsychiatric / cognitive": {
        "Cognitive Impairment / Dementia", "Depression", "Anxiety",
    },
    "Sleep / respiratory": {"Obstructive Sleep Apnea", "COPD"},
}
# Ties go to the smaller clinically distinctive families first. Every patient
# receives at most one label; patients with none of these diagnoses are unlabeled.
DISEASE_GROUP_PRIORITY = [
    "Neuropsychiatric / cognitive", "Sleep / respiratory", "Cardiometabolic / vascular",
]
COGNITIVE_STAGES = ["W", "light_sleep", "N3", "R"]
COGNITIVE_REGIONS = {
    "Frontal": {"FP1", "FP2", "FPZ", "F3", "F4", "F7", "F8", "FZ"},
    "Central": {"C3", "C4", "CZ"},
    "Temporal": {"T3", "T4", "T5", "T6", "T7", "T8"},
    "Parietal": {"P3", "P4", "P7", "P8", "PZ"},
    "Occipital": {"O1", "O2", "OZ"},
}


def _diagnoses(value) -> list[str]:
    if isinstance(value, np.ndarray):
        return [str(x) for x in value]
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value]
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return [str(x) for x in parsed] if isinstance(parsed, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return []


def exclusive_disease_family(diagnoses: list[str]) -> str | None:
    scores = {name: len(set(diagnoses) & members) for name, members in DISEASE_GROUPS.items()}
    maximum = max(scores.values(), default=0)
    if maximum == 0:
        return None
    return next(name for name in DISEASE_GROUP_PRIORITY if scores[name] == maximum)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    hep = pd.read_parquet(HEP_FILE)
    if "error" in hep.columns:
        hep = hep[hep["error"].isna()].copy()
    hep = hep.dropna(subset=["hep_raw", "hep_ica", "hep_cfar"])
    demo = pd.read_parquet(DEMO_FILE)[
        ["patient_id", "age", "sex", "diagnosis_categories", "n_diagnoses"]
    ].drop_duplicates("patient_id")
    df = hep.merge(demo, on="patient_id", how="inner", validate="one_to_one")
    df["diagnoses"] = df["diagnosis_categories"].map(_diagnoses)
    df["disease_family"] = df["diagnoses"].map(exclusive_disease_family)
    df["sex"] = df["sex"].astype(str).str.title()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def feature_names() -> list[str]:
    edges = np.arange(0, 0.601, 0.05)
    return [f"{site}_{int(a*1000):03d}-{int(b*1000):03d}ms"
            for site in SITES for a, b in zip(edges[:-1], edges[1:])]


@st.cache_data(
    show_spinner=False,
    hash_funcs={pd.DataFrame: lambda frame: tuple(frame["patient_id"].astype(str))},
)
def method_features(df: pd.DataFrame, method: str) -> np.ndarray:
    """Return patients x 72 mean-amplitude features (6 sites x 12 bins)."""
    col = METHODS[method]
    rows = []
    edges = np.arange(0, 0.601, 0.05)
    for row in df.itertuples(index=False):
        tt = np.asarray(row.tt, dtype=float)
        hep = np.asarray(getattr(row, col), dtype=float).reshape(len(SITES), -1)
        rows.append(np.concatenate([
            [np.nanmean(hep[ch, (tt >= a) & (tt < b)]) for a, b in zip(edges[:-1], edges[1:])]
            for ch in range(len(SITES))
        ]))
    return np.asarray(rows, dtype=float)


@st.cache_data(
    show_spinner=False,
    hash_funcs={pd.DataFrame: lambda frame: tuple(frame["patient_id"].astype(str))},
)
def ecg_features(df: pd.DataFrame) -> np.ndarray:
    """Patient x 12 ECG-average time-bin features, 0--600 ms."""
    edges = np.arange(0, 0.601, 0.05)
    rows = []
    for row in df.itertuples(index=False):
        times = np.asarray(row.tt, dtype=float)
        ecg = np.asarray(row.ecg, dtype=float)
        rows.append([
            np.nanmean(ecg[mask]) if mask.any() else np.nan
            for a, b in zip(edges[:-1], edges[1:])
            for mask in [(times >= a) & (times < b)]
        ])
    return np.asarray(rows, dtype=float)


def decomposition_features(df: pd.DataFrame) -> dict[str, np.ndarray]:
    raw = method_features(df, "HEP")
    cfar = method_features(df, "CFAR")
    return {"Raw HEP": raw, "Cleaned brain (CFAR)": cfar,
            "Removed CFA": raw - cfar, "ECG": ecg_features(df)}


def channel_average_features(features: np.ndarray) -> np.ndarray:
    """Reduce six-channel HEP features to 12 bins for ECG-shape comparisons."""
    return features.reshape(len(features), len(SITES), 12).mean(axis=1)


def _cobrad_id(value: str) -> str | None:
    match = re.search(r"(\d{4}-\d{3}|\d{3}-\d{3})", str(value))
    if not match:
        return None
    text = match.group(1)
    return text if len(text.split("-")[0]) == 4 else "0" + text


def _cache_hep(individual) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Extract every usable HEP channel in µV from a cache tuple."""
    if individual is None or len(individual) < 4 or individual[1] is None:
        return None
    matrix = np.asarray(individual[1], dtype=float)
    times = np.asarray(individual[2], dtype=float)
    channels = [str(x).upper().split("-")[0] for x in individual[3]]
    if matrix.ndim != 2 or matrix.shape[0] != len(channels):
        return None
    # Pickle-cache HEPs are stored in volts.
    return matrix * 1e6, times[:matrix.shape[1]], channels


def _pattern(hep: np.ndarray) -> np.ndarray:
    centered = hep - hep.mean(axis=1, keepdims=True)
    u, _, _ = np.linalg.svd(centered, full_matrices=False)
    return u[:, 0]


def _mean_pattern(patterns: list[np.ndarray]) -> np.ndarray:
    P = np.asarray(patterns)
    P = P * np.sign(P @ P[0])[:, None]
    u = P.mean(axis=0)
    return u / np.linalg.norm(u)


def _bin_features(hep: np.ndarray, times: np.ndarray) -> np.ndarray:
    edges = np.arange(0, 0.601, 0.05)
    return np.concatenate([
        [np.nanmean(hep[ch, mask]) if mask.any() else np.nan
         for a, b in zip(edges[:-1], edges[1:])
         for mask in [(times >= a) & (times < b)]]
        for ch in range(len(SITES))
    ])


@st.cache_data(show_spinner=False)
def cognitive_feature_names() -> list[str]:
    edges = np.arange(0, 0.601, 0.05)
    return [f"{region}_{int(a*1000):03d}-{int(b*1000):03d}ms"
            for region in COGNITIVE_REGIONS for a, b in zip(edges[:-1], edges[1:])]


def _regional_features(hep: np.ndarray, times: np.ndarray, channels: list[str]) -> np.ndarray:
    """Fixed regional representation that tolerates rejected/missing electrodes."""
    edges = np.arange(0, 0.601, 0.05)
    values = []
    for members in COGNITIVE_REGIONS.values():
        indices = [idx for idx, channel in enumerate(channels) if channel in members]
        regional = np.nanmean(hep[indices], axis=0) if indices else None
        for a, b in zip(edges[:-1], edges[1:]):
            mask = (times >= a) & (times < b)
            values.append(np.nanmean(regional[mask]) if regional is not None and mask.any() else np.nan)
    return np.asarray(values)


@st.cache_data(show_spinner=False)
def load_cobrad_cognitive() -> pd.DataFrame:
    """All available COBRAD HEP, ICA, and patient-specific CFAR features."""
    clinical = pd.read_excel(COBRAD_FILE, sheet_name="clinical")
    clinical["cobrad_id"] = clinical["record_id"].map(_cobrad_id)
    clinical = clinical[["cobrad_id", "moca", "mmse", "age", "sex"]].drop_duplicates("cobrad_id")
    records = []
    for stage in COGNITIVE_STAGES:
        raw_path = COBRAD_PICKLES / stage / "individuals_cache_rpeak_shape_v1.pkl"
        ica_path = COBRAD_PICKLES / stage / "individuals_cache_ica_rpeak_shape_v1.pkl"
        if not raw_path.exists() or not ica_path.exists():
            continue
        raw_items = pickle.load(open(raw_path, "rb"))
        ica_items = pickle.load(open(ica_path, "rb"))
        raw_map = {_cobrad_id(x[0]): x for x in raw_items if _cobrad_id(x[0])}
        ica_map = {_cobrad_id(x[0]): x for x in ica_items if _cobrad_id(x[0])}
        for pid in sorted(raw_map):
            raw = _cache_hep(raw_map[pid])
            ica = _cache_hep(ica_map.get(pid)) if pid in ica_map else None
            if raw is None:
                continue
            # Patient-specific CFAR adapts to every participant's surviving
            # channels and reference instead of requiring the HSP six-site filter.
            u = _pattern(raw[0])
            cfar = raw[0] - np.outer(u, u @ raw[0])
            records.append({"cobrad_id": pid, "stage": stage,
                            "HEP": _regional_features(raw[0], raw[1], raw[2]),
                            "CFAR": _regional_features(cfar, raw[1], raw[2]),
                            "ICA": (_regional_features(ica[0], ica[1], ica[2])
                                    if ica is not None else None)})
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records).merge(clinical, on="cobrad_id", how="inner")
    return frame


def target_labels(df: pd.DataFrame, target_type: str, target_value: str | None) -> tuple[np.ndarray, np.ndarray, str]:
    if target_type == "Diagnosis":
        y = df["diagnoses"].map(lambda xs: int(target_value in xs)).to_numpy()
        valid = np.ones(len(df), dtype=bool)
        return y, valid, f"{target_value}: present vs absent"
    if target_type == "Disease group":
        valid = df["disease_family"].notna().to_numpy()
        y = (df["disease_family"] == target_value).astype(int).to_numpy()
        return y, valid, f"{target_value}: exclusive family vs other two families"
    if target_type == "Sex":
        valid = df["sex"].isin(["Female", "Male"]).to_numpy()
        y = (df["sex"] == "Male").astype(int).to_numpy()
        return y, valid, "Male vs Female"
    age = df["age"].to_numpy(float)
    valid = np.isfinite(age)
    median = float(np.nanmedian(age))
    return (age >= median).astype(int), valid, f"Older vs Younger (median {median:.1f} y)"


def bh(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    return multipletests(np.nan_to_num(p, nan=1.0), method="fdr_bh")[1]


@st.cache_data(show_spinner=False)
def univariate_stats(X: np.ndarray, y: np.ndarray, names: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(names):
        a, b = X[y == 1, j], X[y == 0, j]
        if len(a) < 3 or len(b) < 3:
            continue
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        # Rank-biserial correlation: positive means label 1 tends to be larger.
        rbc = 2 * u / (len(a) * len(b)) - 1
        rows.append({"feature": name, "median_label_1": np.nanmedian(a),
                     "median_label_0": np.nanmedian(b), "rank_biserial": rbc,
                     "p_value": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["q_value"] = bh(out["p_value"].tolist())
        out["fdr_significant"] = out["q_value"] < 0.05
        out = out.sort_values(["q_value", "p_value"])
    return out


def classifier(name: str):
    if name == "Logistic regression":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), LogisticRegression(
            max_iter=3000, class_weight="balanced", C=1.0, random_state=RANDOM_STATE))
    if name == "Random forest":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                             RandomForestClassifier(n_estimators=350, min_samples_leaf=3,
                                                    class_weight="balanced", n_jobs=-1,
                                                    random_state=RANDOM_STATE))
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), SVC(C=1.0, kernel="rbf", class_weight="balanced",
                                               probability=True, random_state=RANDOM_STATE))


@st.cache_data(show_spinner=False)
def classify(X: np.ndarray, y: np.ndarray, model_name: str, folds: int) -> tuple[dict, np.ndarray, np.ndarray]:
    counts = np.bincount(y.astype(int), minlength=2)
    n_splits = min(folds, int(counts.min()))
    if n_splits < 2:
        raise ValueError("At least two samples per class are required.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    model = classifier(model_name)
    prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=None)[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "ROC AUC": roc_auc_score(y, prob),
        "Balanced accuracy": balanced_accuracy_score(y, pred),
        "F1": f1_score(y, pred, zero_division=0),
        "N": len(y), "Positive N": int(y.sum()), "Negative N": int((y == 0).sum()),
    }
    return metrics, prob, confusion_matrix(y, pred, labels=[0, 1])


@st.cache_data(show_spinner=False)
def cluster_result(X: np.ndarray, y: np.ndarray, k: int) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    z = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(10, z.shape[1], z.shape[0] - 1), random_state=RANDOM_STATE)
    latent = pca.fit_transform(z)
    labels = KMeans(n_clusters=k, n_init=30, random_state=RANDOM_STATE).fit_predict(latent)
    plot = pd.DataFrame({"PC1": latent[:, 0], "PC2": latent[:, 1],
                         "cluster": labels.astype(str), "phenotype": y.astype(str)})
    table = pd.crosstab(labels, y)
    chi2, p_chi, _, _ = stats.chi2_contingency(table) if table.shape[1] > 1 else (np.nan, np.nan, None, None)
    summary = {
        "silhouette": silhouette_score(latent, labels),
        "adjusted_rand": adjusted_rand_score(y, labels),
        "normalized_mutual_info": normalized_mutual_info_score(y, labels),
        "chi_square_p": p_chi,
        "variance_pc1_pc2": float(pca.explained_variance_ratio_[:2].sum()),
    }
    return plot, summary, table


@st.cache_data(show_spinner=False)
def cluster_feature_profile(X: np.ndarray, labels: np.ndarray, cluster: int,
                            names: tuple[str, ...]) -> pd.DataFrame:
    """Feature differences for one cluster versus every other participant."""
    inside, outside = labels == cluster, labels != cluster
    rows = []
    for idx, name in enumerate(names):
        a, b = X[inside, idx], X[outside, idx]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 3 or len(b) < 3:
            continue
        pooled = np.sqrt(((len(a)-1)*np.var(a, ddof=1) + (len(b)-1)*np.var(b, ddof=1)) /
                         max(len(a)+len(b)-2, 1))
        effect = (np.mean(a)-np.mean(b))/pooled if pooled > 0 else 0.0
        _, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append({"feature": name, "cluster_mean": np.mean(a), "other_mean": np.mean(b),
                     "standardized_difference": effect, "p_value": p_value})
    out = pd.DataFrame(rows)
    if len(out):
        out["q_value"] = bh(out.p_value.tolist())
        out["fdr_significant"] = out.q_value < .05
        out = out.sort_values("standardized_difference", key=lambda s: s.abs(), ascending=False)
    return out


def cluster_clinical_profile(df: pd.DataFrame, labels: np.ndarray, cluster: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    inside = labels == cluster
    continuous_rows = []
    for column, label in [("age", "Age"), ("hr_bpm", "Heart rate"),
                          ("n_beats", "R peaks"), ("n_diagnoses", "Number of diagnoses")]:
        a = pd.to_numeric(df.loc[inside, column], errors="coerce").dropna()
        b = pd.to_numeric(df.loc[~inside, column], errors="coerce").dropna()
        if len(a) < 3 or len(b) < 3:
            continue
        _, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
        continuous_rows.append({"trait": label, "cluster_median": a.median(),
                                "other_median": b.median(), "difference": a.median()-b.median(),
                                "p_value": p_value})
    continuous = pd.DataFrame(continuous_rows)
    if len(continuous):
        continuous["q_value"] = bh(continuous.p_value.tolist())

    binary_traits = {"Male sex": (df.sex == "Male").to_numpy()}
    for group_name in DISEASE_GROUPS:
        binary_traits[f"[Group] {group_name}"] = (df.disease_family == group_name).to_numpy()
    for diagnosis_name in diagnosis_counts(df).index:
        binary_traits[diagnosis_name] = df.diagnoses.map(lambda ds, d=diagnosis_name: d in ds).to_numpy()
    binary_rows = []
    for trait, present in binary_traits.items():
        table = np.array([[np.sum(inside & present), np.sum(inside & ~present)],
                          [np.sum(~inside & present), np.sum(~inside & ~present)]])
        odds, p_value = stats.fisher_exact(table)
        binary_rows.append({"trait": trait, "cluster_n": int(table[0, 0]),
                            "cluster_percent": 100*table[0, 0]/max(table[0].sum(), 1),
                            "other_percent": 100*table[1, 0]/max(table[1].sum(), 1),
                            "odds_ratio": odds, "p_value": p_value})
    binary = pd.DataFrame(binary_rows)
    binary["q_value"] = bh(binary.p_value.tolist())
    binary["fdr_significant"] = binary.q_value < .05
    return continuous.sort_values("q_value"), binary.sort_values(["q_value", "p_value"])


def disease_group_cluster_enrichment(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """All cluster x disease-family enrichment tests with one global FDR."""
    rows = []
    labeled = df.disease_family.notna().to_numpy()
    for cluster in sorted(np.unique(labels)):
        inside = (labels == cluster) & labeled
        outside = (labels != cluster) & labeled
        for group_name in DISEASE_GROUPS:
            present = (df.disease_family == group_name).to_numpy()
            table = np.array([[np.sum(inside & present), np.sum(inside & ~present)],
                              [np.sum(outside & present), np.sum(outside & ~present)]])
            odds, p_value = stats.fisher_exact(table)
            rows.append({
                "cluster": int(cluster), "disease_group": group_name,
                "cluster_n": int(table[0, 0]), "cluster_total": int(table[0].sum()),
                "cluster_percent": 100*table[0, 0]/max(table[0].sum(), 1),
                "other_percent": 100*table[1, 0]/max(table[1].sum(), 1),
                "odds_ratio": odds, "p_value": p_value,
            })
    out = pd.DataFrame(rows)
    out["q_value"] = bh(out.p_value.tolist())
    out["fdr_significant"] = out.q_value < .05
    out["direction"] = np.where(out.odds_ratio >= 1, "Enriched", "Depleted")
    out["signed_log10_q"] = np.where(out.odds_ratio >= 1, 1, -1) * -np.log10(
        out.q_value.clip(lower=1e-300)
    )
    return out.sort_values(["q_value", "p_value"])


@st.cache_data(show_spinner=False)
def cognitive_regression(X: np.ndarray, score: np.ndarray, folds: int) -> tuple[dict, np.ndarray]:
    n_splits = min(folds, len(score))
    if n_splits < 3:
        raise ValueError("At least three cognitive-score observations are required.")
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                          StandardScaler(), Ridge(alpha=10.0))
    predicted = cross_val_predict(model, X, score, cv=cv)
    rho, p_value = stats.spearmanr(score, predicted)
    return {
        "N": len(score), "R²": r2_score(score, predicted),
        "MAE": mean_absolute_error(score, predicted),
        "Spearman rho": rho, "Spearman p": p_value,
    }, predicted


@st.cache_data(show_spinner=False)
def feature_score_correlations(X: np.ndarray, score: np.ndarray, names: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for idx, name in enumerate(names):
        valid = np.isfinite(X[:, idx]) & np.isfinite(score)
        if valid.sum() < 3 or np.nanstd(X[valid, idx]) == 0:
            rho, p_value = 0.0, 1.0
        else:
            rho, p_value = stats.spearmanr(X[valid, idx], score[valid])
        rows.append({"feature": name, "spearman_rho": rho, "p_value": p_value})
    out = pd.DataFrame(rows)
    out["q_value"] = bh(out.p_value.tolist())
    out["fdr_significant"] = out.q_value < .05
    return out.sort_values(["q_value", "p_value"])


def ecg_similarity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in df.itertuples(index=False):
        tt = np.asarray(row.tt, dtype=float)
        ecg = np.asarray(row.ecg, dtype=float)
        keep = np.abs(tt) > .05
        for method, column in METHODS.items():
            hep = np.asarray(getattr(row, column), dtype=float).reshape(6, -1)
            correlations = []
            for channel in hep:
                x, z = channel[keep], ecg[keep]
                valid = np.isfinite(x) & np.isfinite(z)
                if valid.sum() < 3 or np.std(x[valid]) == 0 or np.std(z[valid]) == 0:
                    continue
                correlations.append(abs(np.corrcoef(x[valid], z[valid])[0, 1]))
            rows.append({"patient_id": row.patient_id, "method": method,
                         "HEP–ECG |r|": float(np.mean(correlations)) if correlations else np.nan})
    return pd.DataFrame(rows)


def validity_targets(df: pd.DataFrame, selected_diagnosis: str | None) -> dict[str, np.ndarray]:
    cardiovascular = {"Heart Failure", "Coronary Artery Disease", "Atrial Fibrillation", "Hypertension"}
    targets = {
        "Sex (Male)": (df.sex == "Male").astype(int).to_numpy(),
        "Older age": (df.age >= df.age.median()).astype(int).to_numpy(),
        "High heart rate": (df.hr_bpm >= df.hr_bpm.median()).astype(int).to_numpy(),
        "Any cardiovascular diagnosis": df.diagnoses.map(lambda ds: int(bool(set(ds) & cardiovascular))).to_numpy(),
    }
    if selected_diagnosis:
        targets[f"Selected: {selected_diagnosis}"] = df.diagnoses.map(lambda ds: int(selected_diagnosis in ds)).to_numpy()
    return targets


def waveform_figure(df: pd.DataFrame, group: np.ndarray, site: str) -> go.Figure:
    ch = SITES.index(site)
    fig = make_subplots(rows=1, cols=3, subplot_titles=list(METHODS))
    labels = [(0, "Label 0", "#666666"), (1, "Label 1", "#CC3311")]
    # Recordings have several sampling rates. Interpolate only for display;
    # feature extraction above uses every recording's native time vector.
    common_tt = np.linspace(-0.2, 0.595, 160)
    for col_idx, method in enumerate(METHODS, 1):
        waves = np.stack([
            np.interp(common_tt, np.asarray(tt, dtype=float), np.asarray(v).reshape(6, -1)[ch])
            for v, tt in zip(df[METHODS[method]], df["tt"])
        ])
        tt = common_tt
        for label, label_name, color in labels:
            x = waves[group == label]
            if not len(x):
                continue
            mean = np.nanmean(x, axis=0)
            sem = stats.sem(x, axis=0, nan_policy="omit") if len(x) > 1 else np.zeros_like(mean)
            fig.add_trace(go.Scatter(x=tt, y=mean, mode="lines", name=label_name,
                                     legendgroup=label_name, showlegend=col_idx == 1,
                                     line=dict(color=color, width=2)), row=1, col=col_idx)
            fig.add_trace(go.Scatter(x=np.r_[tt, tt[::-1]],
                                     y=np.r_[mean-sem, (mean+sem)[::-1]], fill="toself",
                                     fillcolor=color, opacity=.12, line=dict(width=0),
                                     hoverinfo="skip", showlegend=False), row=1, col=col_idx)
        fig.add_vrect(x0=-.05, x1=.05, fillcolor="grey", opacity=.15, line_width=0,
                      row=1, col=col_idx)
        fig.add_vline(x=0, line_dash="dash", line_color="black", row=1, col=col_idx)
    fig.update_xaxes(title_text="Time from R-peak (s)")
    fig.update_yaxes(title_text="HEP amplitude (µV)", row=1, col=1)
    fig.update_layout(height=430, template="plotly_white", title=f"{site}: phenotype waveforms")
    return fig


def decomposition_waveform_figure(df: pd.DataFrame, site: str) -> go.Figure:
    """Grand-average ECG, raw HEP, cleaned HEP, and removed CFA."""
    channel = SITES.index(site)
    common_times = np.linspace(-.2, .595, 160)
    signals = {name: [] for name in ["Raw HEP", "Cleaned brain (CFAR)", "Removed CFA"]}
    ecgs = []
    for row in df.itertuples(index=False):
        times = np.asarray(row.tt, dtype=float)
        raw = np.asarray(row.hep_raw, dtype=float).reshape(6, -1)[channel]
        clean = np.asarray(row.hep_cfar, dtype=float).reshape(6, -1)[channel]
        signals["Raw HEP"].append(np.interp(common_times, times, raw))
        signals["Cleaned brain (CFAR)"].append(np.interp(common_times, times, clean))
        signals["Removed CFA"].append(np.interp(common_times, times, raw-clean))
        ecg = np.interp(common_times, times, np.asarray(row.ecg, dtype=float))
        scale = np.nanstd(ecg)
        ecgs.append((ecg-np.nanmean(ecg))/scale if scale > 0 else ecg*0)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=[f"EEG at {site}: raw = cleaned + removed", "ECG (standardized per patient)"],
                        vertical_spacing=.13)
    colors = {"Raw HEP": "#D55E00", "Cleaned brain (CFAR)": "#009E73", "Removed CFA": "#AA4499"}
    for name, values in signals.items():
        matrix = np.asarray(values)
        fig.add_trace(go.Scatter(x=common_times, y=np.nanmean(matrix, axis=0), mode="lines",
                                 name=name, line=dict(color=colors[name], width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=common_times, y=np.nanmean(ecgs, axis=0), mode="lines", name="ECG z-score",
                             line=dict(color="#333333", width=2.5)), row=2, col=1)
    for row_number in [1, 2]:
        fig.add_vrect(x0=-.05, x1=.05, fillcolor="grey", opacity=.15, line_width=0,
                      row=row_number, col=1)
        fig.add_vline(x=0, line_dash="dash", line_color="black", row=row_number, col=1)
    fig.update_yaxes(title_text="Amplitude (µV)", row=1, col=1)
    fig.update_yaxes(title_text="ECG (z)", row=2, col=1)
    fig.update_xaxes(title_text="Time from R-peak (s)", row=2, col=1)
    fig.update_layout(template="plotly_white", height=650,
                      title="Heart–brain decomposition across the cohort")
    return fig


def diagnosis_counts(df: pd.DataFrame) -> pd.Series:
    return pd.Series([d for ds in df["diagnoses"] for d in ds]).value_counts()


@st.cache_data(show_spinner=False)
def load_source_features() -> pd.DataFrame:
    """Per-recording HEP features for every EDF_Format folder (script 25)."""
    frame = pd.read_parquet(SOURCE_FILE)
    frame["source"] = frame["dataset"].map(SOURCE_LABELS).fillna(frame["dataset"])
    return frame


def source_feature_groups(frame: pd.DataFrame) -> dict[str, list[str]]:
    columns = frame.columns.tolist()
    waveform = [c for c in columns if c.endswith("ms") and not c.startswith("shape_")]
    shape = [c for c in columns if c.startswith("shape_")]
    heart = [c for c in columns if c in ("hr_bpm", "n_beats")]
    metadata = [c for c in columns if c in ("sfreq", "n_channels", "n_eeg_channels",
                                            "duration_s", "n_shared_sites")]
    return {"HEP waveform (µV)": waveform, "HEP shape (z-scored)": shape,
            "Heart rate": heart, "Recording metadata": metadata}


def balanced_sample(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Keep every folder comparable in size: at most `cap` recordings each."""
    parts = []
    for _, group in frame.groupby("source"):
        parts.append(group.sample(min(len(group), cap), random_state=RANDOM_STATE))
    return pd.concat(parts).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def classify_source(X: np.ndarray, labels: tuple[str, ...], model_name: str,
                    folds: int) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Out-of-fold multi-class folder prediction with chance baselines."""
    y = np.asarray(labels)
    classes = np.unique(y)
    counts = pd.Series(y).value_counts()
    if len(classes) < 2:
        raise ValueError("At least two source folders are required.")
    if counts.min() < 2:
        raise ValueError("Every selected folder needs at least two recordings.")
    n_splits = int(min(folds, counts.min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(classifier(model_name), X, y, cv=cv,
                              method="predict_proba", n_jobs=None)
    pred = classes[np.argmax(proba, axis=1)]
    shares = counts.reindex(classes).to_numpy(float) / len(y)
    metrics = {
        "Accuracy": float(np.mean(pred == y)),
        "Balanced accuracy": balanced_accuracy_score(y, pred),
        "Macro F1": f1_score(y, pred, average="macro", zero_division=0),
        "Weighted F1": f1_score(y, pred, average="weighted", zero_division=0),
        "Cohen's kappa": cohen_kappa_score(y, pred),
        "Chance accuracy": float(1 / len(classes)),
        "Majority-class accuracy": float(shares.max()),
        "N": len(y), "Folds": n_splits,
    }
    return metrics, y, pred, classes


@st.cache_data(show_spinner=False)
def source_feature_tests(X: np.ndarray, labels: tuple[str, ...],
                         names: tuple[str, ...]) -> pd.DataFrame:
    """Kruskal-Wallis test per feature: what separates the folders at all."""
    y = np.asarray(labels)
    groups = np.unique(y)
    rows = []
    for column, name in enumerate(names):
        samples = [X[y == group, column] for group in groups]
        samples = [s[np.isfinite(s)] for s in samples]
        if min(len(s) for s in samples) < 3:
            continue
        h, p_value = stats.kruskal(*samples)
        rows.append({"feature": name, "H": h, "epsilon_squared": (h - len(groups) + 1) / max(len(y) - len(groups), 1),
                     "p_value": p_value})
    out = pd.DataFrame(rows)
    if len(out):
        out["q_value"] = bh(out.p_value.tolist())
        out = out.sort_values("q_value")
    return out


SYSTOLE_WINDOW = (0.05, 0.35)
DIASTOLE_WINDOW = (0.35, 0.60)
PHASE_MIN_PATIENTS = 10


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.nanmean(values[mask])) if mask.any() else np.nan


def phase_names() -> tuple[str, ...]:
    return tuple(
        [f"{site} contraction" for site in SITES]
        + [f"{site} expansion" for site in SITES]
        + [f"{site} contraction−expansion" for site in SITES]
        + ["mean contraction", "mean expansion", "mean contraction−expansion"]
    )


@st.cache_data(
    show_spinner=False,
    hash_funcs={pd.DataFrame: lambda frame: tuple(frame["patient_id"].astype(str))},
)
def phase_features(df: pd.DataFrame, method: str, systole: tuple[float, float],
                   diastole: tuple[float, float], include_pre_r: bool
                   ) -> tuple[np.ndarray, tuple[str, ...]]:
    """Split every HEP into contraction (systole) and expansion (diastole)."""
    col = METHODS[method]
    rows = []
    for row in df.itertuples(index=False):
        times = np.asarray(row.tt, dtype=float)
        hep = np.asarray(getattr(row, col), dtype=float).reshape(len(SITES), -1)
        systole_mask = (times >= systole[0]) & (times < systole[1])
        diastole_mask = (times >= diastole[0]) & (times < diastole[1])
        if include_pre_r:
            # Late diastole sits before the next R-peak; the averaging window
            # starts at -200 ms, so it can be included when requested.
            diastole_mask = diastole_mask | ((times >= -0.2) & (times < 0.0))
        systolic = np.asarray([_masked_mean(hep[ch], systole_mask) for ch in range(len(SITES))])
        diastolic = np.asarray([_masked_mean(hep[ch], diastole_mask) for ch in range(len(SITES))])
        rows.append(np.concatenate([systolic, diastolic, systolic - diastolic,
                                    [np.nanmean(systolic), np.nanmean(diastolic),
                                     np.nanmean(systolic - diastolic)]]))
    return np.asarray(rows, dtype=float), phase_names()


def phase_label(feature: str) -> str:
    if "contraction−expansion" in feature:
        return "Contraction − expansion"
    return "Contraction" if "contraction" in feature else "Expansion"


@st.cache_data(
    show_spinner=False,
    hash_funcs={pd.DataFrame: lambda frame: tuple(frame["patient_id"].astype(str))},
)
def phase_diagnosis_findings(df: pd.DataFrame, methods: tuple[str, ...],
                             systole: tuple[float, float],
                             diastole: tuple[float, float], include_pre_r: bool,
                             min_patients: int) -> pd.DataFrame:
    """Diagnosis differences for every method, phase, and diagnosis at once."""
    diagnoses = diagnosis_counts(df)
    diagnoses = diagnoses[diagnoses >= min_patients]
    rows = []
    for method in methods:
        X, names = phase_features(df, method, systole, diastole, include_pre_r)
        for diagnosis in diagnoses.index:
            y_dx = np.asarray([int(diagnosis in item) for item in df["diagnoses"]])
            if y_dx.sum() < min_patients or (y_dx == 0).sum() < min_patients:
                continue
            table = univariate_stats(X, y_dx, names)
            for record in table.itertuples(index=False):
                rows.append({
                    "Method": method, "Diagnosis": diagnosis,
                    "Phase": phase_label(record.feature), "Feature": record.feature,
                    "Direction": ("higher in diagnosis" if record.rank_biserial > 0
                                  else "lower in diagnosis"),
                    "Median present": record.median_label_1,
                    "Median absent": record.median_label_0,
                    "Rank-biserial": record.rank_biserial, "p_value": record.p_value,
                    "N present": int(y_dx.sum()), "N absent": int((y_dx == 0).sum()),
                })
    out = pd.DataFrame(rows)
    if len(out):
        out["q_global"] = bh(out["p_value"].tolist())
        out["Significant"] = out.q_global < 0.05
        out = out.sort_values("q_global").reset_index(drop=True)
    return out


def phase_waveform_figure(df: pd.DataFrame, systole: tuple[float, float],
                          diastole: tuple[float, float],
                          include_pre_r: bool) -> go.Figure:
    """Cohort-average HEP per method with the contraction/expansion windows shaded."""
    common_times = np.linspace(-0.2, 0.595, 160)
    fig = make_subplots(rows=1, cols=3, subplot_titles=list(METHODS))
    for column, method in enumerate(METHODS, 1):
        waves = np.stack([
            np.interp(common_times, np.asarray(tt, dtype=float),
                      np.asarray(values, dtype=float).reshape(len(SITES), -1).mean(axis=0))
            for values, tt in zip(df[METHODS[method]], df["tt"])
        ])
        mean = np.nanmean(waves, axis=0)
        sem = stats.sem(waves, axis=0, nan_policy="omit") if len(waves) > 1 else np.zeros_like(mean)
        fig.add_trace(go.Scatter(x=common_times, y=mean, mode="lines", name=method,
                                 line=dict(color=METHOD_COLORS[method], width=2.5),
                                 showlegend=column == 1), row=1, col=column)
        fig.add_trace(go.Scatter(x=np.r_[common_times, common_times[::-1]],
                                 y=np.r_[mean - sem, (mean + sem)[::-1]], fill="toself",
                                 fillcolor=METHOD_COLORS[method], opacity=.12,
                                 line=dict(width=0), hoverinfo="skip",
                                 showlegend=False), row=1, col=column)
        fig.add_vrect(x0=systole[0], x1=systole[1], fillcolor="#CC3311", opacity=.15,
                      line_width=0, row=1, col=column)
        if include_pre_r:
            fig.add_vrect(x0=-0.2, x1=0.0, fillcolor="#0072B2", opacity=.12,
                          line_width=0, row=1, col=column)
        fig.add_vrect(x0=diastole[0], x1=diastole[1], fillcolor="#0072B2", opacity=.12,
                      line_width=0, row=1, col=column)
        fig.add_vline(x=0, line_dash="dash", line_color="black", row=1, col=column)
    fig.update_xaxes(title_text="Time from R-peak (s)")
    fig.update_yaxes(title_text="HEP amplitude (µV)", row=1, col=1)
    fig.update_layout(height=430, template="plotly_white",
                      title="Contraction (red) and expansion (blue) windows on the group-average HEP")
    return fig



DISEASE_FAMILY_MAP = {m: f for f, ms in DISEASE_GROUPS.items() for m in ms}

FAMILY_COLORS = {
    "Cardiometabolic / vascular": "#E69F00",
    "Neuropsychiatric / cognitive": "#56B4E9",
    "Sleep / respiratory": "#009E73",
    "Other": "#999999",
}

DIFF_CAT_COLORS = {
    "Artifact-driven (noise confound)": "#D55E00",
    "Preserved neuronal (authentic)": "#0072B2",
    "Unmasked neuronal (signal revealed)": "#009E73",
    "Equivocal / Low signal": "#7F7F7F",
}


@st.cache_data(show_spinner=False)
def pre_post_diagnosis_screen(
    features_raw: np.ndarray,
    features_cfar: np.ndarray,
    features_cfa: np.ndarray,
    diagnoses_tuples: tuple,
    eligible_diagnoses: tuple[str, ...],
    model_name: str,
    folds: int,
    delta_threshold: float = 0.02,
) -> pd.DataFrame:
    rows = []
    diag_sets = [set(ds) for ds in diagnoses_tuples]
    n_patients = len(diagnoses_tuples)
    for dx in eligible_diagnoses:
        y_dx = np.array([int(dx in ds) for ds in diag_sets], dtype=int)
        pos_n = int(y_dx.sum())
        neg_n = n_patients - pos_n
        if min(pos_n, neg_n) < 5:
            continue
        try:
            res_raw, _, _ = classify(features_raw, y_dx, model_name, folds)
            res_cfar, _, _ = classify(features_cfar, y_dx, model_name, folds)
            res_cfa, _, _ = classify(features_cfa, y_dx, model_name, folds)
        except Exception:
            continue

        raw_auc = float(res_raw["ROC AUC"])
        cfar_auc = float(res_cfar["ROC AUC"])
        cfa_auc = float(res_cfa["ROC AUC"])
        delta_auc = float(cfar_auc - raw_auc)

        if delta_auc <= -delta_threshold and cfa_auc >= 0.52:
            diff_cat = "Artifact-driven (noise confound)"
            interp = "Noise confound: difference in raw EEG collapses after CFAR; driven by volume-conducted cardiac artifact (body habitus / heart noise)."
        elif delta_auc >= delta_threshold:
            diff_cat = "Unmasked neuronal (signal revealed)"
            interp = "Signal unmasked: difference strengthens after CFAR; was obscured by cardiac noise in uncleaned EEG."
        elif cfar_auc >= 0.53:
            diff_cat = "Preserved neuronal (authentic)"
            interp = "Authentic interoception: difference survives intact after removing cardiac artifact."
        else:
            diff_cat = "Equivocal / Low signal"
            interp = "Near chance discrimination in both raw and cleaned HEP."

        fam = DISEASE_FAMILY_MAP.get(dx, "Other")
        rows.append({
            "Diagnosis": dx,
            "Disease Family": fam,
            "Positive N": pos_n,
            "Negative N": neg_n,
            "Prevalence (%)": round(100.0 * pos_n / n_patients, 1),
            "Raw AUC (Pre-cleaning)": raw_auc,
            "CFAR AUC (Post-cleaning)": cfar_auc,
            "Removed CFA AUC (Noise)": cfa_auc,
            "Δ AUC (CFAR − Raw)": delta_auc,
            "Raw Bal Acc": float(res_raw["Balanced accuracy"]),
            "CFAR Bal Acc": float(res_cfar["Balanced accuracy"]),
            "Removed CFA Bal Acc": float(res_cfa["Balanced accuracy"]),
            "Difference Category": diff_cat,
            "Scientific Interpretation": interp,
        })
    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.sort_values("Δ AUC (CFAR − Raw)", ascending=True)
    return df_out


@st.cache_data(show_spinner=False)
def disease_family_screen(
    features_raw: np.ndarray,
    features_cfar: np.ndarray,
    features_cfa: np.ndarray,
    family_labels: tuple,
    model_name: str,
    folds: int,
) -> pd.DataFrame:
    rows = []
    fam_arr = np.array(family_labels)
    valid_mask = pd.notna(fam_arr) & (fam_arr != "None")
    fam_valid = fam_arr[valid_mask]

    for fam_name in DISEASE_GROUPS:
        y_fam = (fam_valid == fam_name).astype(int)
        pos_n = int(y_fam.sum())
        neg_n = int((y_fam == 0).sum())
        if min(pos_n, neg_n) < 5:
            continue
        try:
            res_raw, _, _ = classify(features_raw[valid_mask], y_fam, model_name, folds)
            res_cfar, _, _ = classify(features_cfar[valid_mask], y_fam, model_name, folds)
            res_cfa, _, _ = classify(features_cfa[valid_mask], y_fam, model_name, folds)
        except Exception:
            continue

        raw_auc = float(res_raw["ROC AUC"])
        cfar_auc = float(res_cfar["ROC AUC"])
        cfa_auc = float(res_cfa["ROC AUC"])
        delta_auc = float(cfar_auc - raw_auc)

        if delta_auc <= -0.02 and cfa_auc >= 0.52:
            diff_cat = "Artifact-driven (noise confound)"
            interp = "Cardiac field artifact dominates this family; discrimination drops when noise is projected out."
        elif delta_auc >= 0.02:
            diff_cat = "Unmasked neuronal (signal revealed)"
            interp = "Interoceptive cortical signal unmasked once cardiac noise is eliminated."
        elif cfar_auc >= 0.53:
            diff_cat = "Preserved neuronal (authentic)"
            interp = "Robust neural interoceptive signature that survives artifact removal."
        else:
            diff_cat = "Equivocal / Low signal"
            interp = "Modest or chance-level discrimination in both states."

        rows.append({
            "Disease Family": fam_name,
            "Patients N": pos_n,
            "Other Families N": neg_n,
            "Raw AUC (Pre-cleaning)": raw_auc,
            "CFAR AUC (Post-cleaning)": cfar_auc,
            "Removed CFA AUC (Noise)": cfa_auc,
            "Δ AUC (CFAR − Raw)": delta_auc,
            "Raw Bal Acc": float(res_raw["Balanced accuracy"]),
            "CFAR Bal Acc": float(res_cfar["Balanced accuracy"]),
            "Removed CFA Bal Acc": float(res_cfa["Balanced accuracy"]),
            "Difference Category": diff_cat,
            "Scientific Interpretation": interp,
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def pairwise_family_screen(
    features_raw: np.ndarray,
    features_cfar: np.ndarray,
    features_cfa: np.ndarray,
    family_labels: tuple,
    model_name: str,
    folds: int,
) -> pd.DataFrame:
    rows = []
    fam_arr = np.array(family_labels)
    families = list(DISEASE_GROUPS.keys())

    for fam_a, fam_b in combinations(families, 2):
        pair_mask = (fam_arr == fam_a) | (fam_arr == fam_b)
        if pair_mask.sum() < 10:
            continue
        y_pair = (fam_arr[pair_mask] == fam_b).astype(int)
        if min(y_pair.sum(), (y_pair == 0).sum()) < 5:
            continue

        try:
            res_raw, _, _ = classify(features_raw[pair_mask], y_pair, model_name, folds)
            res_cfar, _, _ = classify(features_cfar[pair_mask], y_pair, model_name, folds)
            res_cfa, _, _ = classify(features_cfa[pair_mask], y_pair, model_name, folds)
        except Exception:
            continue

        raw_auc = float(res_raw["ROC AUC"])
        cfar_auc = float(res_cfar["ROC AUC"])
        cfa_auc = float(res_cfa["ROC AUC"])
        delta_auc = float(cfar_auc - raw_auc)

        rows.append({
            "Contrast": f"{fam_a} vs. {fam_b}",
            f"{fam_a} (N)": int((y_pair == 0).sum()),
            f"{fam_b} (N)": int(y_pair.sum()),
            "Raw AUC": raw_auc,
            "CFAR AUC": cfar_auc,
            "Removed CFA AUC": cfa_auc,
            "Δ AUC (CFAR − Raw)": delta_auc,
            "Interpretation": (
                "Difference was mainly cardiac noise (shrank after CFAR)"
                if delta_auc < -0.02 else
                "Difference emerged after cleaning (unmasked by CFAR)"
                if delta_auc > 0.02 else
                "Difference is preserved neural divergence"
                if cfar_auc >= 0.53 else "Groups barely differ"
            ),
        })
    return pd.DataFrame(rows)


def pre_post_contrast_waveforms(
    df: pd.DataFrame,
    y: np.ndarray,
    site: str,
    label_pos: str = "Positive",
    label_neg: str = "Negative",
) -> go.Figure:
    channel = SITES.index(site)
    common_tt = np.linspace(-0.2, 0.595, 160)

    raw_waves = np.stack([
        np.interp(common_tt, np.asarray(tt, dtype=float), np.asarray(v).reshape(6, -1)[channel])
        for v, tt in zip(df["hep_raw"], df["tt"])
    ])
    cfar_waves = np.stack([
        np.interp(common_tt, np.asarray(tt, dtype=float), np.asarray(v).reshape(6, -1)[channel])
        for v, tt in zip(df["hep_cfar"], df["tt"])
    ])
    cfa_waves = raw_waves - cfar_waves

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "1. Pre-cleaning (Raw HEP with noise)",
            "2. Post-cleaning (CFAR Cleaned Brain)",
            "3. Difference Δ (Group 1 − Group 0)",
        ],
        horizontal_spacing=0.08,
    )

    pos_mask = y == 1
    neg_mask = y == 0

    for col_idx, (waves, title_m) in enumerate([(raw_waves, "Raw"), (cfar_waves, "CFAR")], start=1):
        for mask, grp_label, col in [(pos_mask, label_pos, "#D55E00"), (neg_mask, label_neg, "#0072B2")]:
            sub_w = waves[mask]
            if len(sub_w) == 0:
                continue
            mean_w = np.nanmean(sub_w, axis=0)
            sem_w = stats.sem(sub_w, axis=0, nan_policy="omit") if len(sub_w) > 1 else np.zeros_like(mean_w)

            fig.add_trace(go.Scatter(
                x=common_tt, y=mean_w, mode="lines", name=f"{grp_label} ({title_m})",
                legendgroup=grp_label, showlegend=(col_idx == 1),
                line=dict(color=col, width=2.2),
            ), row=1, col=col_idx)
            fig.add_trace(go.Scatter(
                x=np.r_[common_tt, common_tt[::-1]],
                y=np.r_[mean_w - sem_w, (mean_w + sem_w)[::-1]],
                fill="toself", fillcolor=col, opacity=0.15, line=dict(width=0),
                hoverinfo="skip", showlegend=False,
            ), row=1, col=col_idx)

        fig.add_vrect(x0=-0.05, x1=0.05, fillcolor="grey", opacity=0.18, line_width=0, row=1, col=col_idx)
        fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=1, row=1, col=col_idx)
        fig.update_xaxes(title_text="Time from R-peak (s)", row=1, col=col_idx)

    diff_raw = np.nanmean(raw_waves[pos_mask], axis=0) - np.nanmean(raw_waves[neg_mask], axis=0)
    diff_cfar = np.nanmean(cfar_waves[pos_mask], axis=0) - np.nanmean(cfar_waves[neg_mask], axis=0)
    diff_cfa = np.nanmean(cfa_waves[pos_mask], axis=0) - np.nanmean(cfa_waves[neg_mask], axis=0)

    fig.add_trace(go.Scatter(
        x=common_tt, y=diff_raw, mode="lines", name="Raw Difference Δ",
        line=dict(color="#D55E00", width=2.5, dash="dash"),
    ), row=1, col=3)
    fig.add_trace(go.Scatter(
        x=common_tt, y=diff_cfar, mode="lines", name="CFAR Difference Δ",
        line=dict(color="#009E73", width=3),
    ), row=1, col=3)
    fig.add_trace(go.Scatter(
        x=common_tt, y=diff_cfa, mode="lines", name="Removed Noise Δ",
        line=dict(color="#AA4499", width=2, dash="dot"),
    ), row=1, col=3)

    fig.add_vrect(x0=-0.05, x1=0.05, fillcolor="grey", opacity=0.18, line_width=0, row=1, col=3)
    fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=1, row=1, col=3)
    fig.add_hline(y=0, line_dash="solid", line_color="#cccccc", line_width=1, row=1, col=3)
    fig.update_xaxes(title_text="Time from R-peak (s)", row=1, col=3)

    fig.update_yaxes(title_text="HEP Amplitude (µV)", row=1, col=1)
    fig.update_yaxes(title_text="Δ Amplitude (µV)", row=1, col=3)
    fig.update_layout(
        template="plotly_white",
        height=450,
        title=f"Electrode {site}: Pre vs. Post Cleaning Group HEP Waveforms and Δ Contrast",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    )
    return fig


def pre_post_channel_topography_bars(
    raw_feats: np.ndarray,
    cfar_feats: np.ndarray,
    y: np.ndarray,
    time_bins: tuple[int, int] = (3, 8),
) -> go.Figure:
    pos = y == 1
    neg = y == 0
    raw_diffs, cfar_diffs, cfa_diffs = [], [], []
    for ch_idx, site in enumerate(SITES):
        r_vals = raw_feats[:, ch_idx * 12 + time_bins[0] : ch_idx * 12 + time_bins[1]].mean(axis=1)
        c_vals = cfar_feats[:, ch_idx * 12 + time_bins[0] : ch_idx * 12 + time_bins[1]].mean(axis=1)
        r_diff = float(np.nanmean(r_vals[pos]) - np.nanmean(r_vals[neg]))
        c_diff = float(np.nanmean(c_vals[pos]) - np.nanmean(c_vals[neg]))
        raw_diffs.append(r_diff)
        cfar_diffs.append(c_diff)
        cfa_diffs.append(r_diff - c_diff)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=SITES, y=raw_diffs, name="Raw HEP Δ (Pre-cleaning)", marker_color="#D55E00"))
    fig.add_trace(go.Bar(x=SITES, y=cfar_diffs, name="CFAR HEP Δ (Post-cleaning)", marker_color="#009E73"))
    fig.add_trace(go.Bar(x=SITES, y=cfa_diffs, name="Removed Noise Δ (CFA)", marker_color="#AA4499"))
    fig.update_layout(
        barmode="group",
        template="plotly_white",
        title="Scalp Topography: Group Difference Amplitude across 6 Channels (150–400 ms window)",
        xaxis_title="EEG Channel",
        yaxis_title="Group Difference Δ (µV)",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    )
    return fig


STAGE_LABELS = {"R": "REM", "N3": "DEEP", "light_sleep": "LIGHT"}
STAGE_CONTRASTS = [
    ("R", "N3", "REM − DEEP"),
    ("R", "light_sleep", "REM − LIGHT"),
    ("N3", "light_sleep", "DEEP − LIGHT"),
]
STAGE_CONTRAST_NAMES = [label for _, _, label in STAGE_CONTRASTS]
STAGE_DATASETS = {
    "The_Human_Sleep_Project": "Human Sleep Project",
    "Young_The_Human_Sleep_Project": "Young HSP",
    "EDF": "COBRAD / EDF",
    "Berkeley_data": "Berkeley data",
}
# Six shared electrodes in a unit head (x right, y forward).
SITE_POSITIONS = {
    "F3": (-0.52, 0.48), "F4": (0.52, 0.48),
    "C3": (-0.58, 0.02), "C4": (0.58, 0.02),
    "O1": (-0.32, -0.68), "O2": (0.32, -0.68),
}
DELTA_BIN_RANGE = (0, 8)  # 0-400 ms after the R-peak


def _stage_patient_id(value, stage: str) -> str:
    text = str(value)
    marker = f"_{stage}_"
    return text.split(marker)[0] if marker in text else text


def _stage_hep(item, unit: float = 1.0) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Stage-cache HEP in µV, trimmed to the shared time axis."""
    if item is None or len(item) < 4 or item[1] is None:
        return None
    matrix = np.asarray(item[1], dtype=float)
    times = np.asarray(item[2], dtype=float)
    channels = [str(x).upper().split("-")[0] for x in item[3]]
    if matrix.ndim != 2 or matrix.shape[0] != len(channels) or not len(times):
        return None
    width = min(matrix.shape[1], len(times))
    return matrix[:, :width] * unit, times[:width], channels


def _site_features(hep: np.ndarray, times: np.ndarray,
                   channels: list[str]) -> np.ndarray:
    """Six-site HEP bins (72 values), NaN where an electrode is missing."""
    edges = np.arange(0, 0.601, 0.05)
    values = []
    for site in SITES:
        if site in channels:
            trace = hep[channels.index(site)]
            values.extend([np.nanmean(trace[(times >= a) & (times < b)])
                           if ((times >= a) & (times < b)).any() else np.nan
                           for a, b in zip(edges[:-1], edges[1:])])
        else:
            values.extend([np.nan] * 12)
    return np.asarray(values, dtype=float)


@st.cache_data(show_spinner="Loading per-stage HEP caches...")
def load_stage_features(datasets: tuple[str, ...]) -> pd.DataFrame:
    """One six-site HEP vector per patient, stage, and cleaning method."""
    records = []
    for dataset in datasets:
        for stage in COGNITIVE_STAGES:
            raw_path = COBRAD_PICKLES.parent / dataset / stage / "individuals_cache_rpeak_shape_v1.pkl"
            ica_path = COBRAD_PICKLES.parent / dataset / stage / "individuals_cache_ica_rpeak_shape_v1.pkl"
            if not raw_path.exists():
                continue
            raw_items = pickle.load(open(raw_path, "rb"))
            ica_map = {}
            if ica_path.exists():
                for item in pickle.load(open(ica_path, "rb")):
                    ica_map[str(item[0])] = item
                    ica_map.setdefault(_stage_patient_id(item[0], stage), item)
            # Every stage cache stores the HEP in volts.
            unit = 1e6
            for item in raw_items:
                raw = _stage_hep(item, unit)
                if raw is None:
                    continue
                patient = _stage_patient_id(item[0], stage)
                ica = ica_map.get(str(item[0])) or ica_map.get(patient)
                ica = _stage_hep(ica, unit) if ica is not None else None
                pattern = _pattern(raw[0])
                cfar = raw[0] - np.outer(pattern, pattern @ raw[0])
                hep_vector = _site_features(raw[0], raw[1], raw[2])
                ica_vector = _site_features(ica[0], ica[1], ica[2]) if ica is not None else None
                if ica_vector is not None:
                    # Several external-cohort ICA caches are degenerate (all
                    # zero, or blown up by failed decompositions); keep only a
                    # decomposition whose scale matches the uncleaned HEP.
                    raw_median = float(np.nanmedian(np.abs(hep_vector)))
                    ica_median = float(np.nanmedian(np.abs(ica_vector)))
                    ica_peak = float(np.nanmax(np.abs(ica_vector)))
                    if (not np.isfinite(ica_median) or not np.isfinite(ica_peak)
                            or not 0.1 * raw_median <= ica_median <= 10 * raw_median
                            or ica_peak > 500):
                        ica_vector = None
                records.append({
                    "patient": f"{dataset}:{patient}",
                    "dataset": STAGE_DATASETS.get(dataset, dataset),
                    "dataset_key": dataset, "stage": stage, "n_channels": len(raw[2]),
                    "HEP": hep_vector, "CFAR": _site_features(cfar, raw[1], raw[2]),
                    "ICA": ica_vector,
                })
    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def stage_delta_table(datasets: tuple[str, ...]) -> pd.DataFrame:
    """Per-patient HEP difference between sleep stages, for every method."""
    staged = load_stage_features(tuple(datasets))
    if staged.empty:
        return pd.DataFrame()
    rows = []
    for (dataset, pid), group in staged.groupby(["dataset", "patient"], sort=True):
        vectors = {stage: stage_rows for stage, stage_rows in group.groupby("stage")}
        for method in METHODS:
            per_method = {}
            for stage, stage_rows in vectors.items():
                value = stage_rows[method].iloc[0]
                if isinstance(value, (list, tuple, np.ndarray)):
                    per_method[stage] = np.asarray(value, dtype=float)
            for first, second, label in STAGE_CONTRASTS:
                if first not in per_method or second not in per_method:
                    continue
                delta = per_method[first] - per_method[second]
                if not np.isfinite(delta).any():
                    continue
                rows.append({
                    "cobrad_id": pid, "dataset": dataset, "method": method,
                    "contrast": label, "delta": delta, "n_stages": len(per_method),
                })
    return pd.DataFrame(rows)


def stage_delta_matrix(table: pd.DataFrame, method: str, contrast: str
                       ) -> tuple[np.ndarray, np.ndarray]:
    subset = table[(table.method == method) & (table.contrast == contrast)]
    return np.stack(subset["delta"].to_numpy()), subset["cobrad_id"].to_numpy()


def stage_delta_stats(table: pd.DataFrame, method: str, contrast: str,
                      names: list[str]) -> pd.DataFrame:
    """Is each patient-level stage delta different from zero?"""
    X, _ = stage_delta_matrix(table, method, contrast)
    rows = []
    for j, name in enumerate(names):
        values = X[:, j][np.isfinite(X[:, j])]
        if len(values) < 5 or np.allclose(values, 0):
            continue
        try:
            _, p = stats.wilcoxon(values, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            continue
        rows.append({
            "Feature": name, "Electrode": name.split("_")[0],
            "Window (ms)": name.split("_")[-1].replace("ms", ""),
            "N patients": int(len(values)),
            "Mean Δ (µV)": float(np.nanmean(values)),
            "Median Δ (µV)": float(np.nanmedian(values)),
            "Patients with Δ > 0": float(np.mean(values > 0)),
            "p_value": float(p),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["q_value"] = bh(out["p_value"].tolist())
        out["Significant"] = out["q_value"] < 0.05
        out = out.sort_values("q_value").reset_index(drop=True)
    return out


def stage_delta_method_comparison(table: pd.DataFrame, contrast: str,
                                  names: list[str]) -> pd.DataFrame:
    """Paired comparison of the same patient's delta under two cleaning methods."""
    wide = {}
    for method in METHODS:
        X, ids = stage_delta_matrix(table, method, contrast)
        wide[method] = pd.DataFrame(X, index=ids, columns=names)
    rows = []
    for first, second in combinations(METHODS, 2):
        common = wide[first].index.intersection(wide[second].index)
        if len(common) < 5:
            continue
        a_all = wide[first].loc[common].to_numpy()
        b_all = wide[second].loc[common].to_numpy()
        for j, name in enumerate(names):
            a, b = a_all[:, j], b_all[:, j]
            keep = np.isfinite(a) & np.isfinite(b)
            if keep.sum() < 5:
                continue
            difference = a[keep] - b[keep]
            if np.allclose(difference, 0):
                continue
            try:
                _, p = stats.wilcoxon(difference, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                continue
            rows.append({
                "Comparison": f"{first} vs {second}", "Feature": name,
                "Electrode": name.split("_")[0], "N patients": int(keep.sum()),
                "Mean Δ difference (µV)": float(np.nanmean(difference)),
                "Median Δ difference (µV)": float(np.nanmedian(difference)),
                "p_value": float(p),
            })
    out = pd.DataFrame(rows)
    if len(out):
        out["q_value"] = bh(out["p_value"].tolist())
        out["Significant"] = out["q_value"] < 0.05
        out = out.sort_values("q_value").reset_index(drop=True)
    return out


def site_window_delta(table: pd.DataFrame, method: str, contrast: str) -> dict[str, float]:
    """Mean stage delta per electrode in the 0-400 ms window."""
    X, _ = stage_delta_matrix(table, method, contrast)
    values = {}
    for index, site in enumerate(SITES):
        block = X[:, index * 12 + DELTA_BIN_RANGE[0]: index * 12 + DELTA_BIN_RANGE[1]]
        values[site] = float(np.nanmean(block)) if block.size else np.nan
    return values


import matplotlib.pyplot as plt

_EEG_MONTAGE = make_standard_montage("standard_1020")


def _site_info(sites: tuple[str, ...]) -> "mne.Info":
    """MNE info with 10-20 positions for the measured electrodes."""
    info = create_info(list(sites), 100.0, ch_types="eeg")
    info.set_montage(_EEG_MONTAGE, match_case=False, on_missing="raise")
    return info


def brain_map_figure(values: dict[str, float], title: str, scale: float,
                     n_patients: int | None = None) -> "matplotlib.figure.Figure":
    """MNE topomap of the six-site stage delta, with its own colour bar."""
    label = title if n_patients is None else f"{title}  (N={n_patients})"
    fig, ax = plt.subplots(figsize=(2.5, 2.35), dpi=150)
    usable = {site: float(value) for site, value in values.items() if np.isfinite(value)}
    if len(usable) < 4:
        ax.axis("off")
        ax.set_title(f"{label}\nnot enough electrodes", fontsize=7.5)
        return fig
    image, _ = plot_topomap(
        np.array(list(usable.values())), _site_info(tuple(usable)), axes=ax, show=False,
        vlim=(-scale, scale), cmap="RdBu_r", sensors=True, names=list(usable),
        contours=4, outlines="head", extrapolate="head",
        image_interp="cubic",
    )
    for text in ax.texts:
        text.set_fontsize(6)
    ax.set_title(label, fontsize=7.5, pad=2)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.048, pad=0.02)
    colorbar.set_label("Δ µV", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=6, length=2)
    return fig


def stage_delta_bars(table: pd.DataFrame, contrast: str) -> go.Figure:
    fig = go.Figure()
    for method in METHODS:
        values = site_window_delta(table, method, contrast)
        fig.add_trace(go.Bar(x=list(values), y=list(values.values()), name=method,
                             marker_color=METHOD_COLORS[method]))
    fig.add_hline(y=0, line_color="grey")
    fig.update_layout(barmode="group", template="plotly_white", height=360,
                      yaxis_title="Mean stage Δ (µV), 0-400 ms",
                      xaxis_title="Electrode",
                      title=f"{contrast} by cleaning method and electrode",
                      legend=dict(orientation="h", yanchor="bottom", y=-.25,
                                  xanchor="center", x=.5))
    return fig


