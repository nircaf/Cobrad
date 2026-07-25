"""
Sleep Stage EEG/ECG Autoencoder Cluster Dashboard.

PhD-oriented exploratory dashboard for Pickles sleep-stage data:
- extracts interpretable EEG and ECG features from fixed windows;
- trains a dense autoencoder on standardized feature vectors;
- clusters the latent representation with KMeans and, if installed, HDBSCAN;
- explains latent dimensions and clusters using physiological features.

Run:
  streamlit run 14_sleep_stage_autoencoder_cluster_dashboard.py
"""

from __future__ import annotations

import hashlib
import os
import pickle
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import signal, stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

try:
    import mne
    from mne.time_frequency import psd_array_welch
except Exception:
    mne = None
    psd_array_welch = None

# NeuroKit is useful for offline validation, but its ECG cleaning can be slow
# enough to make an exploratory dashboard feel frozen on large RawArray pickles.
try:
    import neurokit2 as nk  # noqa: F401
except Exception:
    nk = None

try:
    import hdbscan
except Exception:
    hdbscan = None


BASE_PATH = Path("/storage/pblab_shared_data2/Nir/Cobrad/pickles_sleep_stage")
EDF_FORMAT_DIR = Path("/storage/pblab_shared_data2/Nir/Cobrad/EDF_Format")
CACHE_DIR = Path("/storage/pblab_shared_data2/Nir/Cobrad/cache/14_autoencoder_clusters")
NON_PATIENT_PREFIXES = ("individuals_cache", "non_eeg_individuals_cache")
STAGE_ORDER = ["W", "light_sleep", "N3", "R", "N1", "N2"]
EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (12.0, 16.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}
RANDOM_STATE = 42


FEATURE_GLOSSARY = {
    "eeg_abs_delta": "Slow-wave activity. Higher values usually reflect deeper NREM sleep and cortical synchronization.",
    "eeg_abs_theta": "Theta activity. Often prominent in light sleep and drowsiness.",
    "eeg_abs_alpha": "Alpha rhythm. Wakefulness/relaxed occipital activity; persistent alpha in sleep may indicate arousal or altered sleep depth.",
    "eeg_abs_sigma": "Spindle-range activity. Clinically linked to N2 sleep spindles, thalamocortical integrity, and memory consolidation.",
    "eeg_abs_beta": "Fast cortical activity. Can rise with arousal, muscle artifact, medication effects, or lighter sleep.",
    "eeg_abs_gamma": "High-frequency activity. May capture cortical activation but is artifact-sensitive.",
    "eeg_rel_delta": "Fraction of EEG spectral power in slow-wave frequencies.",
    "eeg_rel_theta": "Fraction of EEG spectral power in theta frequencies.",
    "eeg_rel_alpha": "Fraction of EEG spectral power in alpha frequencies.",
    "eeg_rel_sigma": "Fraction of EEG spectral power in spindle frequencies.",
    "eeg_rel_beta": "Fraction of EEG spectral power in beta frequencies.",
    "eeg_rel_gamma": "Fraction of EEG spectral power in gamma frequencies.",
    "eeg_delta_theta_ratio": "Balance of deep slow-wave activity versus theta/light-sleep activity.",
    "eeg_theta_alpha_ratio": "Spectral slowing marker sometimes relevant in aging and cognitive impairment.",
    "eeg_spectral_entropy": "Flatness/complexity of the EEG spectrum; lower values suggest more frequency-concentrated activity.",
    "eeg_hjorth_activity": "Signal variance, a broad amplitude/power measure.",
    "eeg_hjorth_mobility": "Dominant-frequency proxy based on first derivative variance.",
    "eeg_hjorth_complexity": "Change in frequency content; higher values suggest more irregular waveform structure.",
    "eeg_line_length": "Time-domain signal complexity/amplitude proxy.",
    "ecg_mean_hr": "Average heart rate. Autonomic tone and sleep depth influence this strongly.",
    "ecg_sdnn": "Overall heart-rate variability. Reflects combined sympathetic and parasympathetic variability.",
    "ecg_rmssd": "Short-term vagal HRV. Often increases in stable NREM sleep and decreases with stress/arousal.",
    "ecg_pnn50": "Percentage of adjacent NN intervals differing by more than 50 ms; parasympathetic HRV marker.",
    "ecg_lf_power": "Low-frequency HRV power. Mixed sympathetic/parasympathetic and baroreflex contributions.",
    "ecg_hf_power": "High-frequency HRV power. Respiration-linked vagal modulation.",
    "ecg_lf_hf_ratio": "Autonomic balance proxy; interpret cautiously because it is not a pure sympathetic index.",
    "ecg_rpeak_count": "Number of detected R peaks; also a quality indicator.",
}
_DEMOGRAPHIC_CACHE: Dict[str, Tuple[pd.DataFrame, str]] = {}


@dataclass
class TrainingResult:
    latent: np.ndarray
    reconstruction_mse: np.ndarray
    train_loss: List[float]
    val_loss: List[float]
    model: Any


class DenseAutoencoder(nn.Module):
    def __init__(self, n_features: int, latent_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        h1 = max(hidden_dim, latent_dim * 2, 8)
        h2 = max(h1 // 2, latent_dim * 2, 4)
        self.encoder = nn.Sequential(
            nn.Linear(n_features, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def natural_stage_sort(stage: str) -> Tuple[int, str]:
    return (STAGE_ORDER.index(stage) if stage in STAGE_ORDER else 99, stage)


def list_groups(base_path: Path) -> List[str]:
    if not base_path.exists():
        return []
    return sorted([p.name for p in base_path.iterdir() if p.is_dir()])


def list_stages(base_path: Path, groups: Sequence[str]) -> List[str]:
    stages = set()
    for group in groups:
        gdir = base_path / group
        if gdir.is_dir():
            stages.update(p.name for p in gdir.iterdir() if p.is_dir())
    return sorted(stages, key=natural_stage_sort)


def list_patient_files(base_path: Path, group: str, stage: str, limit: Optional[int] = None) -> List[Path]:
    stage_dir = base_path / group / stage
    if not stage_dir.is_dir():
        return []
    files = sorted(
        p for p in stage_dir.glob("*.pkl")
        if not p.name.startswith(NON_PATIENT_PREFIXES)
    )
    return files[:limit] if limit else files


def patient_id_from_path(path: Path, stage: str) -> str:
    name = path.stem
    match = re.search(rf"_{re.escape(stage)}_\d+_\d+$", name)
    return name[:match.start()] if match else name


def _norm_col_name(col: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(col).strip().lower())


def _first_matching_col(columns: Iterable[Any], targets: Sequence[str]) -> Optional[str]:
    normalized_targets = {_norm_col_name(t) for t in targets}
    for col in columns:
        if _norm_col_name(col) in normalized_targets:
            return str(col)
    for col in columns:
        norm = _norm_col_name(col)
        if any(target in norm for target in normalized_targets):
            return str(col)
    return None


def _demographic_id_keys(value: Any) -> List[str]:
    if pd.isna(value):
        return []
    text = os.path.basename(str(value).strip()).replace(".pkl", "").replace(".edf", "")
    upper = text.upper()
    keys: List[str] = []

    def add(key: str) -> None:
        if key and key not in keys:
            keys.append(key)

    compact = re.sub(r"[^A-Z0-9]+", "", upper)
    add(compact)
    add(compact.lstrip("0"))
    bids = re.search(r"SUB[-_]?I\d+", upper)
    if bids:
        add(re.sub(r"[^A-Z0-9]+", "", bids.group(0)))
    site_patient = re.search(r"I\d{6,}", upper)
    if site_patient:
        add(site_patient.group(0))
    digit_groups = re.findall(r"\d+", upper)
    if digit_groups:
        joined = "".join(digit_groups)
        add(joined)
        add(joined.lstrip("0") or "0")
    return [k for k in keys if k]


def demographic_file_score(group: str, path: Path) -> int:
    name = path.name.lower()
    hints = {
        "young_the_human_sleep_project": ["psg_metadata"],
        "the_human_sleep_project": ["psg_metadata"],
        "edf": ["cobrad_clinical", "clinical"],
        "berkeley_data": ["00_df_all_demographics_young", "demographics"],
        "cap_sleep_database": ["gender-age"],
    }
    score = sum(10 for hint in hints.get(group.lower(), ["demographic", "clinical", "metadata"]) if hint in name)
    return score + (1 if name.endswith(".csv") else 0)


def load_group_demographics(group: str) -> Tuple[pd.DataFrame, str]:
    if group in _DEMOGRAPHIC_CACHE:
        return _DEMOGRAPHIC_CACHE[group]
    group_dir = EDF_FORMAT_DIR / group
    if not group_dir.exists():
        return pd.DataFrame(), ""
    search_dirs = [group_dir] + [p for p in group_dir.iterdir() if p.is_dir()]
    files = []
    for search_dir in search_dirs:
        files.extend(
            p for p in search_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xls"} and not p.name.startswith("~$")
        )
    files = sorted(files, key=lambda p: (-demographic_file_score(group, p), str(p)))
    best_df = pd.DataFrame()
    best_source = ""
    best_score = -1
    for path in files:
        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path, low_memory=False)
                source = path.name
            else:
                xls = pd.ExcelFile(path)
                sheet = next((s for s in xls.sheet_names if _norm_col_name(s) == "clinical"), xls.sheet_names[0])
                df = pd.read_excel(path, sheet_name=sheet)
                source = f"{path.name}:{sheet}"
        except Exception:
            continue
        cols = list(df.columns)
        score = demographic_file_score(group, path)
        score += 5 if _first_matching_col(cols, ["age", "ageatvisit", "age_at_visit"]) else 0
        score += 5 if _first_matching_col(cols, ["sex", "gender", "sexdsc"]) else 0
        score += 5 if _first_matching_col(cols, ["patient_id", "record_id", "subject", "bidsfolder"]) else 0
        if score > best_score:
            best_df, best_source, best_score = df, source, score
    _DEMOGRAPHIC_CACHE[group] = (best_df, best_source)
    return best_df, best_source


def build_demographic_lookup(df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], Dict[str, Optional[str]]]:
    if df.empty:
        return {}, {}
    id_cols = [
        col for col in df.columns
        if any(token in _norm_col_name(col) for token in ["patient", "subject", "subj", "recordid", "bidsfolder", "bdsppatientid"])
        or _norm_col_name(col) in {"id", "file", "filename"}
    ]
    age_col = _first_matching_col(df.columns, ["age", "ageatvisit", "age_at_visit"])
    sex_col = _first_matching_col(df.columns, ["sex", "gender", "sexdsc"])
    lookup: Dict[str, pd.Series] = {}
    for _, row in df.iterrows():
        for col in id_cols:
            for key in _demographic_id_keys(row.get(col)):
                lookup.setdefault(key, row)
    return lookup, {"age": age_col, "sex": sex_col}


def standardize_sex(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    lower = text.lower()
    if lower in {"1", "1.0", "m", "male"}:
        return "Male"
    if lower in {"2", "2.0", "f", "female"}:
        return "Female"
    return text


def add_demographics(rows: pd.DataFrame, groups: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if rows.empty:
        return rows, pd.DataFrame()
    sources = []
    demo_rows = []
    for group in groups:
        df, source = load_group_demographics(group)
        sources.append({"group": group, "source": source or "not found", "source_rows": len(df)})

        group_patients = rows.loc[rows["group"] == group, "patient_id"].drop_duplicates().tolist()
        wanted: Dict[str, str] = {}
        for patient_id in group_patients:
            for key in _demographic_id_keys(patient_id):
                wanted.setdefault(key, patient_id)

        cols = {
            "age": _first_matching_col(df.columns, ["age", "ageatvisit", "age_at_visit"]) if not df.empty else None,
            "sex": _first_matching_col(df.columns, ["sex", "gender", "sexdsc"]) if not df.empty else None,
        }
        id_cols = [
            col for col in df.columns
            if any(token in _norm_col_name(col) for token in ["patient", "subject", "subj", "recordid", "bidsfolder", "bdsppatientid"])
            or _norm_col_name(col) in {"id", "file", "filename"}
        ] if not df.empty else []
        matched: Dict[str, Dict[str, Any]] = {}
        if wanted and id_cols:
            id_text = df[id_cols].astype(str).apply(lambda col: col.str.upper())
            id_compact = id_text.apply(lambda col: col.str.replace(r"[^A-Z0-9]+", "", regex=True))
            for patient in group_patients:
                for key in _demographic_id_keys(patient):
                    key_re = re.escape(key)
                    mask = pd.Series(False, index=df.index)
                    for col in id_cols:
                        mask = mask | id_text[col].str.contains(key_re, regex=True, na=False)
                        mask = mask | id_compact[col].str.contains(key_re, regex=True, na=False)
                    if mask.any():
                        idx = mask[mask].index[0]
                        matched[patient] = {"row": df.loc[idx], "matched_key": key}
                        break

        for patient_id in group_patients:
            item = matched.get(patient_id)
            out = {
                "group": group,
                "patient_id": patient_id,
                "demographics_matched": item is not None,
                "matched_key": item["matched_key"] if item is not None else "",
                "demographics_source": source,
                "age": np.nan,
                "sex": "",
            }
            if item is not None:
                record = item["row"]
                age_col = cols.get("age")
                sex_col = cols.get("sex")
                out["age"] = pd.to_numeric(record.get(age_col), errors="coerce") if age_col else np.nan
                out["sex"] = standardize_sex(record.get(sex_col)) if sex_col else ""
            demo_rows.append(out)
    merged = rows.merge(pd.DataFrame(demo_rows), on=["group", "patient_id"], how="left")
    return merged, pd.DataFrame(sources)


def robust_float(value: Any) -> float:
    try:
        value = float(value)
    except Exception:
        return np.nan
    return value if np.isfinite(value) else np.nan


def spectral_entropy(psd: np.ndarray) -> float:
    psd = np.asarray(psd, dtype=float)
    total = np.nansum(psd)
    if total <= 0:
        return np.nan
    p = psd / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / np.log(len(psd))) if len(psd) > 1 else np.nan


def hjorth_parameters(x: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return np.nan, np.nan, np.nan
    dx = np.diff(x)
    ddx = np.diff(dx)
    var0 = np.var(x)
    var1 = np.var(dx)
    var2 = np.var(ddx)
    activity = var0
    mobility = np.sqrt(var1 / var0) if var0 > 0 else np.nan
    mobility_dx = np.sqrt(var2 / var1) if var1 > 0 else np.nan
    complexity = mobility_dx / mobility if mobility and np.isfinite(mobility) else np.nan
    return float(activity), float(mobility), float(complexity)


def amplitude_features(segment: np.ndarray, prefix: str) -> Dict[str, float]:
    x = np.asarray(segment, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {
            f"{prefix}_amplitude_mean": np.nan,
            f"{prefix}_amplitude_median": np.nan,
            f"{prefix}_amplitude_std": np.nan,
        }
    return {
        f"{prefix}_amplitude_mean": float(np.mean(x)),
        f"{prefix}_amplitude_median": float(np.median(x)),
        f"{prefix}_amplitude_std": float(np.std(x)),
    }


def summarize_feature_prefix(values: Dict[str, float], prefix: str) -> Dict[str, float]:
    rows = {k: v for k, v in values.items() if k.startswith(prefix)}
    return rows


def detect_channel_indices(raw: Any) -> Tuple[List[int], Optional[int]]:
    ch_types = raw.get_channel_types()
    eeg = [i for i, typ in enumerate(ch_types) if typ == "eeg"]
    ecg = next((i for i, typ in enumerate(ch_types) if typ == "ecg"), None)
    if ecg is None:
        ecg = next((i for i, ch in enumerate(raw.ch_names) if "ECG" in ch.upper() or "EKG" in ch.upper()), None)
    return eeg, ecg


def band_power_features(segment: np.ndarray, sfreq: float, ch_names: Sequence[str]) -> Dict[str, float]:
    features: Dict[str, float] = {}
    if segment.size == 0 or psd_array_welch is None:
        return features
    features.update(amplitude_features(segment, "eeg"))
    psds, freqs = psd_array_welch(
        segment,
        sfreq=sfreq,
        fmin=0.5,
        fmax=40.0,
        n_fft=min(int(sfreq * 4), segment.shape[1]),
        average="mean",
        verbose=False,
    )
    channel_band_values: Dict[str, List[float]] = {band: [] for band in EEG_BANDS}
    channel_rel_values: Dict[str, List[float]] = {band: [] for band in EEG_BANDS}
    entropies = []
    activities = []
    mobilities = []
    complexities = []
    line_lengths = []

    for ch_idx, ch_name in enumerate(ch_names):
        psd = psds[ch_idx]
        total = float(np.trapz(psd, freqs))
        for band, (lo, hi) in EEG_BANDS.items():
            mask = (freqs >= lo) & (freqs < hi)
            power = float(np.trapz(psd[mask], freqs[mask])) if np.any(mask) else np.nan
            features[f"eeg_{ch_name}_abs_{band}"] = power
            features[f"eeg_{ch_name}_rel_{band}"] = power / total if total > 0 else np.nan
            channel_band_values[band].append(power)
            channel_rel_values[band].append(power / total if total > 0 else np.nan)
        entropies.append(spectral_entropy(psd))
        activity, mobility, complexity = hjorth_parameters(segment[ch_idx])
        activities.append(activity)
        mobilities.append(mobility)
        complexities.append(complexity)
        line_lengths.append(float(np.mean(np.abs(np.diff(segment[ch_idx])))))

    for band in EEG_BANDS:
        features[f"eeg_abs_{band}_mean"] = float(np.nanmean(channel_band_values[band]))
        features[f"eeg_rel_{band}_mean"] = float(np.nanmean(channel_rel_values[band]))
    features["eeg_delta_theta_ratio_mean"] = features.get("eeg_abs_delta_mean", np.nan) / (features.get("eeg_abs_theta_mean", np.nan) + 1e-12)
    features["eeg_theta_alpha_ratio_mean"] = features.get("eeg_abs_theta_mean", np.nan) / (features.get("eeg_abs_alpha_mean", np.nan) + 1e-12)
    features["eeg_spectral_entropy_mean"] = float(np.nanmean(entropies))
    features["eeg_hjorth_activity_mean"] = float(np.nanmean(activities))
    features["eeg_hjorth_mobility_mean"] = float(np.nanmean(mobilities))
    features["eeg_hjorth_complexity_mean"] = float(np.nanmean(complexities))
    features["eeg_line_length_mean"] = float(np.nanmean(line_lengths))
    return features


def fallback_rpeaks(ecg: np.ndarray, sfreq: float) -> np.ndarray:
    ecg = np.asarray(ecg, dtype=float)
    if len(ecg) < sfreq * 5:
        return np.array([], dtype=int)
    filtered = signal.detrend(ecg)
    distance = max(int(0.35 * sfreq), 1)
    prominence = np.nanstd(filtered) * 0.5
    peaks, _ = signal.find_peaks(filtered, distance=distance, prominence=prominence)
    if len(peaks) < 3:
        peaks, _ = signal.find_peaks(-filtered, distance=distance, prominence=prominence)
    return peaks.astype(int)


def ecg_features(ecg: np.ndarray, sfreq: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    ecg = np.asarray(ecg, dtype=float)
    out.update(amplitude_features(ecg, "ecg"))
    if len(ecg) < sfreq * 10:
        return out
    rpeaks = fallback_rpeaks(ecg, sfreq)
    out["ecg_rpeak_count"] = float(len(rpeaks))
    if len(rpeaks) < 3:
        return out
    rr_ms = np.diff(rpeaks) / sfreq * 1000.0
    rr_ms = rr_ms[(rr_ms >= 300) & (rr_ms <= 2000)]
    if len(rr_ms) < 2:
        return out
    diff_rr = np.diff(rr_ms)
    out["ecg_mean_hr"] = 60000.0 / float(np.mean(rr_ms))
    out["ecg_sdnn"] = float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else np.nan
    out["ecg_rmssd"] = float(np.sqrt(np.mean(diff_rr ** 2))) if len(diff_rr) else np.nan
    out["ecg_pnn50"] = float(np.mean(np.abs(diff_rr) > 50.0) * 100.0) if len(diff_rr) else np.nan
    if len(rr_ms) >= 8:
        try:
            rr_s = rr_ms / 1000.0
            t = np.cumsum(np.r_[0, rr_s[:-1]])
            fs_interp = 4.0
            ti = np.arange(t[0], t[-1], 1.0 / fs_interp)
            if len(ti) > 8:
                rri = np.interp(ti, t, rr_s)
                freqs, pxx = signal.welch(signal.detrend(rri), fs=fs_interp, nperseg=min(256, len(rri)))
                lf = (freqs >= 0.04) & (freqs < 0.15)
                hf = (freqs >= 0.15) & (freqs < 0.40)
                out["ecg_lf_power"] = float(np.trapz(pxx[lf], freqs[lf])) if np.any(lf) else np.nan
                out["ecg_hf_power"] = float(np.trapz(pxx[hf], freqs[hf])) if np.any(hf) else np.nan
                out["ecg_lf_hf_ratio"] = out["ecg_lf_power"] / (out["ecg_hf_power"] + 1e-12)
        except Exception:
            pass
    return out


def extract_windows_from_file(
    file_path: Path,
    group: str,
    stage: str,
    window_s: float,
    step_s: float,
    max_windows: int,
    include_eeg: bool,
    include_ecg: bool,
) -> List[Dict[str, Any]]:
    with open(file_path, "rb") as f:
        raw = pickle.load(f)
    if mne is None:
        raise RuntimeError("mne is required to load and process the pickled Raw objects.")
    sfreq = float(raw.info["sfreq"])
    eeg_idx, ecg_idx = detect_channel_indices(raw)
    patient_id = patient_id_from_path(file_path, stage)
    win_n = int(round(window_s * sfreq))
    step_n = int(round(step_s * sfreq))
    if win_n <= 0 or raw.n_times < win_n:
        return []
    starts = np.arange(0, raw.n_times - win_n + 1, step_n)
    if len(starts) > max_windows:
        starts = np.linspace(starts[0], starts[-1], max_windows).astype(int)
    rows = []
    for window_idx, start in enumerate(starts):
        stop = int(start + win_n)
        row: Dict[str, Any] = {
            "group": group,
            "stage": stage,
            "patient_id": patient_id,
            "patient_stage_id": f"{patient_id}|{stage}",
            "file_path": str(file_path),
            "window_idx": int(window_idx),
            "window_start_s": float(start / sfreq),
            "window_length_s": float(window_s),
            "sfreq": sfreq,
            "n_eeg_channels": len(eeg_idx),
            "has_ecg": ecg_idx is not None,
        }
        if include_eeg and eeg_idx:
            data = raw.get_data(picks=eeg_idx, start=start, stop=stop)
            row.update(band_power_features(data, sfreq, [raw.ch_names[i] for i in eeg_idx]))
        if include_ecg and ecg_idx is not None:
            ecg = raw.get_data(picks=[ecg_idx], start=start, stop=stop)[0]
            row.update(ecg_features(ecg, sfreq))
        rows.append(row)
    return rows


def feature_cache_key(
    base_path: Path,
    groups: Sequence[str],
    stages: Sequence[str],
    window_s: float,
    step_s: float,
    max_files_per_stage: int,
    max_windows_per_file: int,
    include_eeg: bool,
    include_ecg: bool,
) -> str:
    payload = {
        "base": str(base_path),
        "groups": list(groups),
        "stages": list(stages),
        "window_s": window_s,
        "step_s": step_s,
        "max_files_per_stage": max_files_per_stage,
        "max_windows_per_file": max_windows_per_file,
        "include_eeg": include_eeg,
        "include_ecg": include_ecg,
        "version": 4,
    }
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()[:20]


@st.cache_data(show_spinner=False)
def load_or_extract_features(
    base_path: str,
    groups: Tuple[str, ...],
    stages: Tuple[str, ...],
    window_s: float,
    step_s: float,
    max_files_per_stage: int,
    max_windows_per_file: int,
    include_eeg: bool,
    include_ecg: bool,
    recompute: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(base_path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = feature_cache_key(base, groups, stages, window_s, step_s, max_files_per_stage, max_windows_per_file, include_eeg, include_ecg)
    cache_path = CACHE_DIR / f"features_{key}.parquet"
    log_path = CACHE_DIR / f"features_{key}_log.csv"
    if cache_path.exists() and not recompute:
        return pd.read_parquet(cache_path), pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    logs: List[Dict[str, Any]] = []
    for group in groups:
        for stage in stages:
            files = list_patient_files(base, group, stage, limit=max_files_per_stage or None)
            for file_path in files:
                try:
                    file_rows = extract_windows_from_file(
                        file_path=file_path,
                        group=group,
                        stage=stage,
                        window_s=window_s,
                        step_s=step_s,
                        max_windows=max_windows_per_file,
                        include_eeg=include_eeg,
                        include_ecg=include_ecg,
                    )
                    rows.extend(file_rows)
                    logs.append({"group": group, "stage": stage, "file": str(file_path), "status": "ok", "n_windows": len(file_rows), "error": ""})
                except Exception as exc:
                    logs.append({"group": group, "stage": stage, "file": str(file_path), "status": "error", "n_windows": 0, "error": str(exc)})
    df = pd.DataFrame(rows)
    log_df = pd.DataFrame(logs)
    if not df.empty:
        df.to_parquet(cache_path, index=False)
    log_df.to_csv(log_path, index=False)
    return df, log_df


def feature_columns(df: pd.DataFrame, modality: str) -> List[str]:
    exclude = {
        "group", "stage", "patient_id", "patient_stage_id", "file_path",
        "window_idx", "window_start_s", "window_length_s", "sfreq",
        "n_eeg_channels", "has_ecg", "age", "sex", "demographics_matched",
        "matched_key", "demographics_source",
    }
    numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if modality == "EEG only":
        return [c for c in numeric_cols if c.startswith("eeg_")]
    if modality == "ECG only":
        return [c for c in numeric_cols if c.startswith("ecg_")]
    return [c for c in numeric_cols if c.startswith(("eeg_", "ecg_"))]


def aggregate_patient_stage(df: pd.DataFrame, feat_cols: Sequence[str]) -> pd.DataFrame:
    meta = ["group", "stage", "patient_id", "patient_stage_id", "age", "sex", "demographics_matched"]
    agg = df.groupby(meta, dropna=False)[list(feat_cols)].agg(["mean", "std"]).reset_index()
    agg.columns = [c[0] if c[1] == "" else f"{c[0]}_{c[1]}" for c in agg.columns.to_flat_index()]
    agg["n_windows"] = df.groupby(meta, dropna=False).size().to_numpy()
    return agg


def add_amplitude_color_columns(df: pd.DataFrame, modality: str, analysis_unit: str) -> pd.DataFrame:
    out = df.copy()
    prefixes = ["ecg"] if modality == "ECG only" else ["eeg", "ecg"]
    suffix = "_mean" if analysis_unit == "Patient + sleep stage" else ""
    for label, base_metric in {
        "mean_amplitude": "amplitude_mean",
        "median_amplitude": "amplitude_median",
        "std_amplitude": "amplitude_std",
    }.items():
        for prefix in prefixes:
            source_col = f"{prefix}_{base_metric}{suffix}"
            if source_col in out.columns:
                out[label] = out[source_col]
                break
    return out


def prepare_matrix(df: pd.DataFrame, feat_cols: Sequence[str], missing_threshold: float) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler, List[str]]:
    usable = []
    for col in feat_cols:
        miss = df[col].isna().mean()
        if miss <= missing_threshold and df[col].nunique(dropna=True) > 1:
            usable.append(col)
    clean = df.copy()
    for col in usable:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
        clean[col] = clean[col].replace([np.inf, -np.inf], np.nan)
        clean[col] = clean[col].fillna(clean[col].median())
    scaler = StandardScaler()
    x = scaler.fit_transform(clean[usable].to_numpy(dtype=float))
    return clean, x, scaler, usable


def train_autoencoder(
    x: np.ndarray,
    latent_dim: int,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
) -> TrainingResult:
    if torch is None:
        raise RuntimeError("PyTorch is required for autoencoder training.")
    torch.manual_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = np.asarray(x, dtype=np.float32)
    idx = np.arange(len(x))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE) if len(x) >= 10 else (idx, idx)
    train_loader = DataLoader(TensorDataset(torch.tensor(x[train_idx])), batch_size=batch_size, shuffle=True)
    val_tensor = torch.tensor(x[val_idx], dtype=torch.float32).to(device)
    model = DenseAutoencoder(x.shape[1], latent_dim, hidden_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    train_loss: List[float] = []
    val_loss: List[float] = []
    best_state = None
    best_val = np.inf
    patience = 15
    stale = 0
    for _ in range(epochs):
        model.train()
        losses = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val = float(criterion(model(val_tensor), val_tensor).detach().cpu())
        train_loss.append(float(np.mean(losses)))
        val_loss.append(val)
        if val < best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32).to(device)
        reconstructed = model(xt).detach().cpu().numpy()
        latent = model.encoder(xt).detach().cpu().numpy()
    mse = np.mean((x - reconstructed) ** 2, axis=1)
    return TrainingResult(latent=latent, reconstruction_mse=mse, train_loss=train_loss, val_loss=val_loss, model=model)


def cluster_latent(latent: np.ndarray, method: str, k: int, min_cluster_size: int) -> np.ndarray:
    if method == "HDBSCAN":
        if hdbscan is None:
            raise RuntimeError("hdbscan is not installed. Install it or choose KMeans.")
        labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=max(2, min_cluster_size // 2)).fit_predict(latent)
    else:
        k = max(2, min(int(k), len(latent) - 1))
        labels = KMeans(n_clusters=k, n_init=30, random_state=RANDOM_STATE).fit_predict(latent)
    return labels.astype(int)


def cluster_metrics(latent: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    valid = labels >= 0
    unique = np.unique(labels[valid])
    rows = [{"metric": "n_clusters", "value": len(unique)}, {"metric": "n_noise", "value": int(np.sum(labels < 0))}]
    if len(unique) >= 2 and np.sum(valid) > len(unique):
        rows.extend([
            {"metric": "silhouette", "value": float(silhouette_score(latent[valid], labels[valid]))},
            {"metric": "davies_bouldin", "value": float(davies_bouldin_score(latent[valid], labels[valid]))},
            {"metric": "calinski_harabasz", "value": float(calinski_harabasz_score(latent[valid], labels[valid]))},
        ])
    return pd.DataFrame(rows)


def group_separation_stats(latent: np.ndarray, group_labels: np.ndarray, n_perms: int = 499) -> dict:
    """PERMANOVA and PERMDISP on Euclidean distance matrix of the latent space."""
    from scipy.spatial.distance import cdist

    groups = np.unique(group_labels)
    n = len(latent)
    if len(groups) < 2 or n < 4:
        return {}

    D = cdist(latent, latent, metric="euclidean")
    D2 = D ** 2
    idx_i, idx_j = np.triu_indices(n, k=1)
    d2_upper = D2[idx_i, idx_j]
    k = len(groups)

    def _permanova_f(labels: np.ndarray) -> float:
        same = labels[idx_i] == labels[idx_j]
        ss_within = d2_upper[same].sum()
        ss_total = d2_upper.sum()
        ss_between = ss_total - ss_within
        df_b, df_w = k - 1, n - k
        if df_w <= 0 or ss_within == 0.0:
            return 0.0
        return (ss_between / df_b) / (ss_within / df_w)

    def _permdisp_f(labels: np.ndarray) -> float:
        centroids = {g: latent[labels == g].mean(axis=0) for g in np.unique(labels)}
        disp = np.array([np.linalg.norm(latent[i] - centroids[labels[i]]) for i in range(n)])
        unique = np.unique(labels)
        grand = disp.mean()
        ss_b = sum(np.sum(labels == g) * (disp[labels == g].mean() - grand) ** 2 for g in unique)
        ss_w = sum(np.sum((disp[labels == g] - disp[labels == g].mean()) ** 2) for g in unique)
        df_b, df_w = len(unique) - 1, n - len(unique)
        if df_w <= 0 or ss_w == 0.0:
            return 0.0
        return (ss_b / df_b) / (ss_w / df_w)

    obs_perm = _permanova_f(group_labels)
    obs_disp = _permdisp_f(group_labels)
    rng = np.random.default_rng(42)
    perm_perm_fs = []
    perm_disp_fs = []
    for _ in range(n_perms):
        plab = rng.permutation(group_labels)
        perm_perm_fs.append(_permanova_f(plab))
        perm_disp_fs.append(_permdisp_f(plab))
    perm_p = (np.sum(np.array(perm_perm_fs) >= obs_perm) + 1) / (n_perms + 1)
    disp_p = (np.sum(np.array(perm_disp_fs) >= obs_disp) + 1) / (n_perms + 1)
    return {
        "permanova_F": float(obs_perm),
        "permanova_p": float(perm_p),
        "permdisp_F": float(obs_disp),
        "permdisp_p": float(disp_p),
        "n_groups": int(k),
        "n_samples": n,
    }


def embed_2d(latent: np.ndarray, method: str) -> np.ndarray:
    if latent.shape[1] == 1:
        return np.c_[latent[:, 0], np.zeros(len(latent))]
    if method == "t-SNE" and len(latent) >= 10:
        perplexity = min(30, max(5, (len(latent) - 1) // 3))
        return TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=RANDOM_STATE).fit_transform(latent)
    return PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(latent)


def eta_squared_kruskal(groups: List[np.ndarray], h: float) -> float:
    n = sum(len(g) for g in groups)
    k = len(groups)
    return float((h - k + 1) / (n - k)) if n > k else np.nan


def feature_cluster_tests(df: pd.DataFrame, feat_cols: Sequence[str], label_col: str = "cluster") -> pd.DataFrame:
    rows = []
    labels = [lab for lab in sorted(df[label_col].dropna().unique()) if int(lab) >= 0]
    if len(labels) < 2:
        return pd.DataFrame()
    for col in feat_cols:
        groups = [pd.to_numeric(df.loc[df[label_col] == lab, col], errors="coerce").dropna().to_numpy() for lab in labels]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            continue
        try:
            h, p = stats.kruskal(*groups)
        except Exception:
            continue
        medians = df.groupby(label_col)[col].median(numeric_only=True).to_dict()
        high_cluster = max(medians, key=lambda key: medians[key])
        low_cluster = min(medians, key=lambda key: medians[key])
        rows.append({
            "feature": col,
            "p": float(p),
            "h_stat": float(h),
            "eta_squared": eta_squared_kruskal(groups, float(h)),
            "highest_median_cluster": high_cluster,
            "lowest_median_cluster": low_cluster,
            "clinical_hint": clinical_hint_for_feature(col),
        })
    out = pd.DataFrame(rows).sort_values("p") if rows else pd.DataFrame()
    if not out.empty:
        out["q_fdr"] = multipletests(out["p"].to_numpy(), method="fdr_bh")[1]
    return out


def categorical_enrichment(df: pd.DataFrame, col: str, label_col: str = "cluster") -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame()
    sub = df[[label_col, col]].dropna()
    sub = sub[sub[label_col] >= 0]
    if sub.empty or sub[col].nunique() < 2 or sub[label_col].nunique() < 2:
        return pd.DataFrame()
    table = pd.crosstab(sub[label_col], sub[col])
    try:
        chi2, p, _, _ = stats.chi2_contingency(table)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame([{"variable": col, "chi2": chi2, "p": p, "n": int(table.to_numpy().sum())}])


def latent_feature_correlations(df: pd.DataFrame, feat_cols: Sequence[str], latent_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for latent_col in latent_cols:
        y = pd.to_numeric(df[latent_col], errors="coerce")
        for feat in feat_cols:
            x = pd.to_numeric(df[feat], errors="coerce")
            valid = x.notna() & y.notna()
            if valid.sum() < 8 or x[valid].nunique() < 2:
                continue
            rho, p = stats.spearmanr(x[valid], y[valid])
            rows.append({
                "latent_dimension": latent_col,
                "feature": feat,
                "spearman_rho": float(rho),
                "p": float(p),
                "abs_rho": abs(float(rho)),
                "clinical_hint": clinical_hint_for_feature(feat),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_fdr"] = multipletests(out["p"].to_numpy(), method="fdr_bh")[1]
        out = out.sort_values(["latent_dimension", "abs_rho"], ascending=[True, False])
    return out


def clinical_hint_for_feature(feature: str) -> str:
    normalized = feature
    normalized = re.sub(r"^eeg_[A-Za-z0-9]+_", "eeg_", normalized)
    normalized = normalized.replace("_mean", "").replace("_std", "")
    for key, text in FEATURE_GLOSSARY.items():
        if key in normalized or normalized.startswith(key):
            return text
    return "No specific clinical mapping was predefined; interpret through its source channel, modality, and cluster statistics."


def permutation_sensitivity(model: Any, x: np.ndarray, feat_cols: Sequence[str], repeats: int = 3) -> pd.DataFrame:
    if torch is None or model is None:
        return pd.DataFrame()
    device = next(model.parameters()).device
    base = torch.tensor(x.astype(np.float32), dtype=torch.float32).to(device)
    criterion = nn.MSELoss(reduction="none")
    model.eval()
    with torch.no_grad():
        base_loss = criterion(model(base), base).mean(dim=1).detach().cpu().numpy().mean()
    rows = []
    rng = np.random.default_rng(RANDOM_STATE)
    for j, feat in enumerate(feat_cols):
        deltas = []
        for _ in range(repeats):
            xp = x.copy()
            xp[:, j] = rng.permutation(xp[:, j])
            xt = torch.tensor(xp.astype(np.float32), dtype=torch.float32).to(device)
            with torch.no_grad():
                loss = criterion(model(xt), xt).mean(dim=1).detach().cpu().numpy().mean()
            deltas.append(float(loss - base_loss))
        rows.append({"feature": feat, "reconstruction_loss_increase": float(np.mean(deltas)), "clinical_hint": clinical_hint_for_feature(feat)})
    return pd.DataFrame(rows).sort_values("reconstruction_loss_increase", ascending=False)


def k_sweep(latent: np.ndarray, k_min: int, k_max: int) -> pd.DataFrame:
    rows = []
    for k in range(k_min, k_max + 1):
        if k >= len(latent):
            continue
        labels = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(latent)
        metrics = cluster_metrics(latent, labels)
        row = {"k": k}
        row.update(metrics.set_index("metric")["value"].to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def download_button(df: pd.DataFrame, label: str, filename: str) -> None:
    if df.empty:
        return
    st.download_button(label, data=df.to_csv(index=False), file_name=filename, mime="text/csv")


def render_stat_cards(metrics_df: pd.DataFrame, n_rows: int, n_features: int) -> None:
    cols = st.columns(5)
    cols[0].metric("Samples", f"{n_rows:,}")
    cols[1].metric("Features", f"{n_features:,}")
    for idx, metric in enumerate(["n_clusters", "silhouette", "davies_bouldin"]):
        value = metrics_df.loc[metrics_df["metric"] == metric, "value"]
        cols[idx + 2].metric(metric.replace("_", " ").title(), "NA" if value.empty else f"{value.iloc[0]:.3g}")


def main() -> None:
    st.set_page_config(page_title="Sleep Stage Autoencoder Clustering", layout="wide")
    st.title("Sleep Stage EEG/ECG Autoencoder Clustering")

    if mne is None:
        st.error("MNE is not available in this Python environment. Run with the project venv: ./venv/bin/streamlit run 14_sleep_stage_autoencoder_cluster_dashboard.py")
        st.stop()
    if torch is None:
        st.error("PyTorch is not available in this Python environment. It is required for autoencoder training.")
        st.stop()

    groups_all = list_groups(BASE_PATH)
    default_groups = [g for g in ["Young_The_Human_Sleep_Project", "The_Human_Sleep_Project", "EDF", "Berkeley_data"] if g in groups_all]
    if not default_groups:
        default_groups = groups_all[:2]
    stages_all = list_stages(BASE_PATH, groups_all)
    default_stages = [s for s in ["W", "light_sleep", "N3", "R"] if s in stages_all]

    st.sidebar.header("Data")
    selected_groups = st.sidebar.multiselect("Groups", groups_all, default=default_groups)
    selected_stages = st.sidebar.multiselect("Sleep stages", stages_all, default=default_stages)
    modality = st.sidebar.radio("Modality", ["EEG + ECG", "EEG only", "ECG only"], horizontal=False)
    analysis_unit = st.sidebar.radio("Analysis unit", ["Window", "Patient + sleep stage"], index=1)
    window_s = st.sidebar.slider("Window length, seconds", 10, 120, 30, step=5)
    step_s = st.sidebar.slider("Window step, seconds", 5, 120, 30, step=5)
    max_files = st.sidebar.number_input("Max files per group/stage", min_value=1, max_value=500, value=20, step=5)
    max_windows = st.sidebar.number_input("Max windows per file", min_value=1, max_value=200, value=20, step=5)
    recompute = st.sidebar.checkbox("Recompute feature cache", value=False)

    st.sidebar.header("Autoencoder")
    latent_dim = st.sidebar.slider("Latent dimensions", 2, 16, 4)
    hidden_dim = st.sidebar.slider("Hidden layer width", 16, 256, 64, step=16)
    dropout = st.sidebar.slider("Dropout", 0.0, 0.5, 0.10, step=0.05)
    epochs = st.sidebar.slider("Max epochs", 20, 300, 100, step=10)
    batch_size = st.sidebar.select_slider("Batch size", options=[16, 32, 64, 128, 256], value=64)

    st.sidebar.header("Clustering")
    cluster_method_options = ["KMeans"] + (["HDBSCAN"] if hdbscan is not None else [])
    cluster_method = st.sidebar.selectbox("Cluster method", cluster_method_options)
    k = st.sidebar.slider("KMeans k", 2, 12, 4)
    min_cluster_size = st.sidebar.slider("HDBSCAN min cluster size", 3, 50, 10)
    projection_method = st.sidebar.radio("2D projection", ["PCA", "t-SNE"], horizontal=True)

    if not selected_groups or not selected_stages:
        st.info("Select at least one group and one sleep stage.")
        st.stop()

    include_eeg = modality in {"EEG + ECG", "EEG only"}
    include_ecg = modality in {"EEG + ECG", "ECG only"}

    with st.spinner("Extracting or loading window-level physiological features..."):
        raw_features, extraction_log = load_or_extract_features(
            str(BASE_PATH),
            tuple(selected_groups),
            tuple(selected_stages),
            float(window_s),
            float(step_s),
            int(max_files),
            int(max_windows),
            include_eeg,
            include_ecg,
            bool(recompute),
        )
    if raw_features.empty:
        st.error("No usable feature rows were extracted. Check selected groups/stages and the extraction log.")
        st.dataframe(extraction_log)
        st.stop()

    raw_features, demo_sources = add_demographics(raw_features, selected_groups)
    base_feat_cols = feature_columns(raw_features, modality)
    if analysis_unit == "Patient + sleep stage":
        model_df = aggregate_patient_stage(raw_features, base_feat_cols)
        feat_cols = [c for c in model_df.columns if c.endswith("_mean") or c.endswith("_std")]
    else:
        model_df = raw_features.copy()
        feat_cols = base_feat_cols
    model_df = add_amplitude_color_columns(model_df, modality, analysis_unit)

    clean_df, x, scaler, usable_features = prepare_matrix(model_df, feat_cols, missing_threshold=0.35)
    if x.shape[0] < 8 or x.shape[1] < 2:
        st.error("Not enough samples or usable features after missing-value filtering.")
        st.write({"samples": x.shape[0], "features": x.shape[1]})
        st.stop()

    with st.spinner("Training autoencoder and clustering latent space..."):
        training = train_autoencoder(
            x,
            latent_dim=min(latent_dim, max(2, x.shape[1] - 1)),
            hidden_dim=hidden_dim,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            lr=1e-3,
        )
        labels = cluster_latent(training.latent, cluster_method, k, min_cluster_size)
        projection = embed_2d(training.latent, projection_method)

    latent_cols = [f"latent_{i + 1}" for i in range(training.latent.shape[1])]
    result_df = clean_df.copy()
    for i, col in enumerate(latent_cols):
        result_df[col] = training.latent[:, i]
    result_df["ae_mse"] = training.reconstruction_mse
    result_df["cluster"] = labels
    result_df["proj_x"] = projection[:, 0]
    result_df["proj_y"] = projection[:, 1]
    result_df["cluster_label"] = result_df["cluster"].map(lambda v: "noise" if v < 0 else f"cluster {int(v)}")

    metrics_df = cluster_metrics(training.latent, labels)
    render_stat_cards(metrics_df, len(result_df), len(usable_features))

    tab_map, tab_stats, tab_latent, tab_features, tab_quality, tab_exports = st.tabs(
        ["Map", "Cluster Statistics", "Latent Meaning", "Feature Explorer", "Quality", "Exports"]
    )

    hover_cols = [
        c for c in [
            "patient_id", "stage", "group", "age", "sex", "window_start_s",
            "mean_amplitude", "median_amplitude", "std_amplitude", "ae_mse",
        ]
        if c in result_df.columns
    ]
    with tab_map:
        color_options = [
            c for c in [
                "cluster_label", "group", "stage", "sex", "age", "ae_mse",
                "mean_amplitude", "median_amplitude", "std_amplitude",
            ]
            if c in result_df.columns
        ]
        color_by = st.selectbox("Color points by", color_options, index=0)
        symbol_by = st.selectbox("Symbol by", ["stage", "group", "sex", "cluster_label"], index=0)
        map_df = result_df
        if color_by == "group":
            if "patient_id" in result_df.columns:
                _map_group_n = result_df.groupby("group")["patient_id"].nunique()
            else:
                _map_group_n = result_df["group"].value_counts()
            map_df = result_df.copy()
            map_df["group"] = map_df["group"].map(lambda g: f"{g} (n={_map_group_n.get(g, 0)})")
        fig = px.scatter(
            map_df,
            x="proj_x",
            y="proj_y",
            color=color_by,
            symbol=symbol_by if symbol_by in result_df.columns else None,
            hover_data=hover_cols,
            title=f"{projection_method} projection of autoencoder latent space",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig.update_layout(height=680, legend_title_text=color_by)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("KMeans k sweep")
        sweep = k_sweep(training.latent, 2, min(12, len(result_df) - 1))
        if not sweep.empty:
            metric_to_plot = st.radio("Sweep metric", ["silhouette", "davies_bouldin", "calinski_harabasz"], horizontal=True)
            st.plotly_chart(px.line(sweep, x="k", y=metric_to_plot, markers=True), use_container_width=True)
            st.dataframe(sweep, use_container_width=True)

        st.subheader("Group Separation Statistics")
        if "group" not in result_df.columns or result_df["group"].nunique() < 2:
            st.info("Need ≥ 2 groups to compute group separation statistics.")
        else:
            n_perms_gs = st.number_input("Permutations", min_value=99, max_value=4999, value=499, step=100, key="gs_n_perms")
            with st.spinner("Computing PERMANOVA / PERMDISP..."):
                sep_stats = group_separation_stats(training.latent, result_df["group"].values, n_perms=int(n_perms_gs))
            if sep_stats:
                col_pa, col_pb, col_da, col_db = st.columns(4)
                col_pa.metric("PERMANOVA F", f"{sep_stats['permanova_F']:.3f}")
                col_pb.metric("PERMANOVA p", f"{sep_stats['permanova_p']:.4f}")
                col_da.metric("PERMDISP F", f"{sep_stats['permdisp_F']:.3f}")
                col_db.metric("PERMDISP p", f"{sep_stats['permdisp_p']:.4f}")
                st.caption(f"Groups: {sep_stats['n_groups']}  |  Samples: {sep_stats['n_samples']}  |  Permutations: {int(n_perms_gs)}")
                st.subheader("Latent Dimensions by Group")
                latent_dim_to_plot = st.selectbox("Latent dimension (boxplot)", latent_cols, key="group_sep_latent_sel")
                if "patient_id" in result_df.columns:
                    _group_n = result_df.groupby("group")["patient_id"].nunique()
                else:
                    _group_n = result_df["group"].value_counts()
                box_df = result_df.copy()
                box_df["group"] = box_df["group"].map(lambda g: f"{g} (n={_group_n.get(g, 0)})")
                fig_box = px.box(
                    box_df,
                    x="group",
                    y=latent_dim_to_plot,
                    color="group",
                    points="all",
                    title=f"{latent_dim_to_plot} distribution by group",
                    color_discrete_sequence=px.colors.qualitative.Safe,
                )
                fig_box.update_layout(height=480, showlegend=False, xaxis_title="Group", yaxis_title=latent_dim_to_plot)
                st.plotly_chart(fig_box, use_container_width=True)

    with tab_stats:
        st.subheader("Cluster-defining physiological features")
        tests = feature_cluster_tests(result_df, usable_features)
        if tests.empty:
            st.info("At least two non-noise clusters are needed for feature statistics.")
        else:
            sig_only = st.checkbox("Show FDR q < 0.05 only", value=False)
            shown = tests[tests["q_fdr"] < 0.05] if sig_only else tests
            st.dataframe(shown, use_container_width=True, height=420)
            top_features = shown.head(20)["feature"].tolist()
            if top_features:
                selected_feature = st.selectbox("Plot feature by cluster", top_features)
                fig = px.box(result_df, x="cluster_label", y=selected_feature, color="cluster_label", points="all")
                fig.update_layout(height=520, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Cluster enrichment against metadata")
        enrich_rows = []
        for col in ["group", "stage", "sex"]:
            enriched = categorical_enrichment(result_df, col)
            if not enriched.empty:
                enrich_rows.append(enriched)
        if "age" in result_df.columns and result_df["age"].notna().sum() > 5 and result_df["cluster"].nunique() > 1:
            age_tests = feature_cluster_tests(result_df.rename(columns={"age": "__age"}), ["__age"])
            if not age_tests.empty:
                age_row = age_tests.iloc[0].to_dict()
                enrich_rows.append(pd.DataFrame([{"variable": "age", "chi2": np.nan, "p": age_row["p"], "n": int(result_df["age"].notna().sum())}]))
        enrichment = pd.concat(enrich_rows, ignore_index=True) if enrich_rows else pd.DataFrame()
        st.dataframe(enrichment, use_container_width=True)

    with tab_latent:
        st.subheader("What each autoencoder dimension means")
        corr = latent_feature_correlations(result_df, usable_features, latent_cols)
        if corr.empty:
            st.info("Not enough variability for latent-feature correlations.")
        else:
            latent_choice = st.selectbox("Latent dimension", latent_cols)
            top_corr = corr[corr["latent_dimension"] == latent_choice].head(25)
            st.dataframe(top_corr, use_container_width=True, height=420)
            fig = px.bar(
                top_corr.sort_values("spearman_rho"),
                x="spearman_rho",
                y="feature",
                orientation="h",
                hover_data=["q_fdr", "clinical_hint"],
                title=f"Strongest physiological correlates of {latent_choice}",
            )
            fig.update_layout(height=650)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Permutation sensitivity")
        with st.spinner("Estimating feature importance by reconstruction-loss sensitivity..."):
            sensitivity = permutation_sensitivity(training.model, x, usable_features, repeats=2)
        st.dataframe(sensitivity.head(40), use_container_width=True, height=420)

        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(y=training.train_loss, mode="lines", name="train"))
        loss_fig.add_trace(go.Scatter(y=training.val_loss, mode="lines", name="validation"))
        loss_fig.update_layout(title="Autoencoder reconstruction loss", xaxis_title="Epoch", yaxis_title="MSE", height=360)
        st.plotly_chart(loss_fig, use_container_width=True)

    with tab_features:
        st.subheader("Feature glossary and raw distributions")
        feature_choice = st.selectbox("Feature", usable_features)
        st.info(clinical_hint_for_feature(feature_choice))
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(px.violin(result_df, x="stage", y=feature_choice, color="stage", box=True, points="all"), use_container_width=True)
        with col_b:
            st.plotly_chart(px.violin(result_df, x="group", y=feature_choice, color="group", box=True, points="all"), use_container_width=True)
        st.dataframe(
            pd.DataFrame(
                [{"feature_pattern": key, "clinical_or_physiological_explanation": value} for key, value in FEATURE_GLOSSARY.items()]
            ),
            use_container_width=True,
            height=420,
        )

    with tab_quality:
        st.subheader("Data coverage")
        st.write("Extraction log")
        st.dataframe(extraction_log, use_container_width=True, height=300)
        st.write("Demographic sources")
        st.dataframe(demo_sources, use_container_width=True)
        st.write("Missingness of retained candidate features")
        missing = model_df[feat_cols].isna().mean().sort_values(ascending=False).reset_index()
        missing.columns = ["feature", "missing_fraction"]
        st.dataframe(missing, use_container_width=True, height=320)
        if "demographics_matched" in result_df.columns:
            st.metric("Demographics matched", f"{100 * result_df['demographics_matched'].fillna(False).mean():.1f}%")

    with tab_exports:
        st.subheader("Export tables")
        download_button(result_df, "Download embeddings and clusters", "sleep_stage_autoencoder_embeddings.csv")
        download_button(metrics_df, "Download cluster metrics", "sleep_stage_cluster_metrics.csv")
        download_button(feature_cluster_tests(result_df, usable_features), "Download feature cluster tests", "sleep_stage_feature_cluster_tests.csv")
        download_button(latent_feature_correlations(result_df, usable_features, latent_cols), "Download latent explanations", "sleep_stage_latent_feature_correlations.csv")
        st.dataframe(result_df.head(200), use_container_width=True, height=500)


if __name__ == "__main__":
    main()
