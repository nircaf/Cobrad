"""
Sleep Stage HEP T-Wave Modulation Dashboard
PhD-grade analysis of heartbeat-evoked potential T-wave coupling across groups and sleep stages.
"""

import io
import hashlib
import json
import os
import pickle
import re
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mne
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import streamlit as st
from scipy import signal, stats
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.decomposition import FastICA


# ── Constants ────────────────────────────────────────────────────────────────

BASE_PATH = "/storage/pblab_shared_data2/Nir/Cobrad/pickles_sleep_stage"
DEFAULT_WINDOW = (-0.3, 0.5)
DEFAULT_T_WINDOW = (0.15, 0.5)
DEFAULT_EEG_T_RADIUS = 0.12
EEG_R_PEAK_FLIP_WINDOW = (-0.010, 0.100)
EEG_R_PEAK_FIT_MIN_HALF_WINDOW = 0.010
EEG_R_PEAK_FIT_MAX_HALF_WINDOW = 0.030
ICA_COMPONENTS_TO_REMOVE = 2
ICA_MAX_COMPONENTS = 20
ICA_MAX_FIT_SAMPLES = 60000
MAX_SPECTRAL_POWER_RATIO = 0.4
NON_PATIENT_CACHE_PREFIXES = ("individuals_cache", "non_eeg_individuals_cache")
PROCESSED_CACHE_VERSION = 3
PROCESSED_CACHE_DIRNAME = ".hep_twave_processed_cache"
EDF_FORMAT_DIRNAME = "EDF_Format"
_DEMOGRAPHIC_SOURCE_CACHE: Dict[str, Tuple[pd.DataFrame, str]] = {}
EEG_SIGNAL_MIN_FINITE_FRACTION = 0.95
EEG_SIGNAL_MIN_PTP_UV = 0.05
EEG_SIGNAL_MIN_STD_UV = 0.01
EEG_SIGNAL_MAX_ABS_UV = 500.0
EEG_SIGNAL_MAX_ROUGHNESS = 2.2
RAW_EEG_SIGNAL_MIN_PTP_UV = 0.10
RAW_EEG_SIGNAL_MIN_STD_UV = 0.01
RAW_EEG_SIGNAL_MAX_ABS_UV = 5000.0

# Colorblind-friendly Okabe-Ito palette for scientific figures
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00", "#000000"]
STAGE_ORDER = ["W", "light_sleep", "N3", "R"]
_TOPO_STD19 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
]
_TOPO_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

st.set_page_config(page_title="HEP T-Wave Modulation", layout="wide")

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ── Data class ───────────────────────────────────────────────────────────────

@dataclass
class PatientResult:
    group: str
    stage: str
    patient_id: str
    file_path: str
    sfreq: float
    n_rpeaks: int
    n_epochs_total: int
    n_epochs_kept: int
    ecg_channel: str
    eeg_channels: List[str]
    flipped_ecg: bool
    flipped_eeg_channels: List[str]
    times: np.ndarray
    ecg_average: np.ndarray
    eeg_average: np.ndarray
    ecg_t_peak_s: float
    quality_notes: List[str]
    flip_details: Dict[str, Dict[str, float]]
    spectral_power_ratios: Dict[str, float]
    ica_details: Optional[Dict[str, Any]] = None


# ── File discovery ────────────────────────────────────────────────────────────

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


def patient_id_from_path(path: str, stage: str) -> str:
    name = os.path.basename(path).replace(".pkl", "").replace(".edf", "")
    m = re.search(rf"_{re.escape(stage)}_\d+_\d+$", name)
    return name[:m.start()] if m else name


def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()


def list_patient_files(base_path: str, group: str, stage: str, limit: Optional[int] = None) -> List[str]:
    stage_dir = os.path.join(base_path, group, stage)
    if not os.path.isdir(stage_dir):
        return []
    files = sorted(
        os.path.join(stage_dir, f) for f in os.listdir(stage_dir)
        if f.endswith(".pkl") and not f.startswith(NON_PATIENT_CACHE_PREFIXES)
    )
    return files[:limit] if limit else files


def processed_cache_dir() -> str:
    return os.path.join(_script_dir(), PROCESSED_CACHE_DIRNAME)


def cache_mode_label(kwargs: Dict[str, Any]) -> str:
    return "ica" if kwargs.get("ica_ecg_clean", False) else "raw"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def source_file_fingerprint(base_path: str, groups: Sequence[str], stage: str,
                            test_run_limit: int) -> List[Dict[str, Any]]:
    limit = int(test_run_limit) if int(test_run_limit) > 0 else None
    rows: List[Dict[str, Any]] = []
    for group in groups:
        for file_path in list_patient_files(base_path, group, stage, limit=limit):
            try:
                stat = os.stat(file_path)
            except OSError:
                rows.append({"group": group, "path": file_path, "missing": True})
                continue
            try:
                rel_path = os.path.relpath(file_path, base_path)
            except ValueError:
                rel_path = file_path
            rows.append({
                "group": group,
                "path": rel_path,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            })
    return rows


def processed_cache_identity(kwargs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    base_path = str(kwargs["base_path"])
    groups = tuple(kwargs["groups"])
    stage = str(kwargs["stage"])
    test_run_limit = int(kwargs["test_run_limit"])
    payload = {
        "cache_version": PROCESSED_CACHE_VERSION,
        "mode": cache_mode_label(kwargs),
        "base_path": os.path.abspath(base_path),
        "params": _json_safe(kwargs),
        "source_files": source_file_fingerprint(base_path, groups, stage, test_run_limit),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def processed_cache_path(kwargs: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    key, metadata = processed_cache_identity(kwargs)
    stage = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(kwargs.get("stage", "stage")))
    mode = cache_mode_label(kwargs)
    filename = f"{stage}_{mode}_{key[:20]}.pkl"
    return os.path.join(processed_cache_dir(), filename), key, metadata


def load_processed_analysis_cache(kwargs: Dict[str, Any]) -> Optional[Tuple[List[PatientResult], pd.DataFrame, str]]:
    path, key, _ = processed_cache_path(kwargs)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None
    if payload.get("cache_version") != PROCESSED_CACHE_VERSION or payload.get("key") != key:
        return None
    results = payload.get("results")
    feature_df = payload.get("feature_df")
    if not isinstance(results, list) or not isinstance(feature_df, pd.DataFrame):
        return None
    return results, feature_df, path


def save_processed_analysis_cache(kwargs: Dict[str, Any], results: List[PatientResult],
                                  feature_df: pd.DataFrame) -> Optional[str]:
    path, key, metadata = processed_cache_path(kwargs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "cache_version": PROCESSED_CACHE_VERSION,
        "created_at": time.time(),
        "key": key,
        "metadata": metadata,
        "results": results,
        "feature_df": feature_df,
    }
    tmp_path = f"{path}.tmp-{os.getpid()}"
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        return None
    return path


def clear_processed_analysis_cache() -> int:
    cache_dir = processed_cache_dir()
    if not os.path.isdir(cache_dir):
        return 0
    removed = 0
    for name in os.listdir(cache_dir):
        if not name.endswith(".pkl"):
            continue
        try:
            os.remove(os.path.join(cache_dir, name))
            removed += 1
        except OSError:
            pass
    return removed


# ── Channel classification ────────────────────────────────────────────────────

def is_ecg_channel(ch: str) -> bool:
    c = ch.lower()
    return "ecg" in c or "ekg" in c


def is_eeg_channel(ch: str) -> bool:
    c = ch.strip()
    cl = c.lower()
    if is_ecg_channel(c):
        return False
    if cl in {"eog", "emg", "spo2", "beat", "status", "marker"}:
        return False
    if cl.startswith(("eog", "emg", "spo2", "beat", "trx", "dc", "stim", "trigger", "trig")):
        return False
    return (
        cl.startswith("eeg")
        or re.match(r"^[A-Za-z]{1,3}[0-9]+$", c) is not None
        or re.match(r"^[A-Za-z]{1,2}z$", c, re.IGNORECASE) is not None
    )


# ── Signal processing ─────────────────────────────────────────────────────────

def safe_filter(x: np.ndarray, sfreq: float, low: float, high: float, order: int = 2) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    nyq = 0.5 * sfreq
    high = min(high, nyq * 0.95)
    if low <= 0 or high <= low or len(x) < order * 12:
        return x.copy()
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x)


def clean_ecg(ecg: np.ndarray, sfreq: float, low_hz: float, high_hz: float,
               median_ms: float, clip_sd: float) -> np.ndarray:
    ecg = np.asarray(ecg, dtype=float)
    win = max(3, int(round((median_ms / 1000.0) * sfreq)) | 1)
    ecg = median_filter(ecg, size=win)
    ecg = safe_filter(ecg, sfreq, low_hz, high_hz, order=2)
    sd = np.nanstd(ecg)
    if np.isfinite(sd) and sd > 0 and clip_sd > 0:
        ecg = np.clip(ecg, np.nanmedian(ecg) - clip_sd * sd, np.nanmedian(ecg) + clip_sd * sd)
    return ecg


def detect_rpeaks(ecg_clean: np.ndarray, sfreq: float, qrs_low_hz: float, qrs_high_hz: float,
                   mad_multiplier: float, refractory_s: float, rr_min_s: float, rr_max_s: float) -> np.ndarray:
    x = safe_filter(ecg_clean, sfreq, qrs_low_hz, qrs_high_hz, order=2)
    energy = np.abs(x)
    smooth_win = max(5, int(round(0.08 * sfreq)) | 1)
    if len(energy) > smooth_win:
        energy = signal.savgol_filter(energy, smooth_win, polyorder=2)
    mad = np.nanmedian(np.abs(energy - np.nanmedian(energy))) + 1e-12
    height = np.nanmedian(energy) + mad_multiplier * mad
    distance = int(refractory_s * sfreq)
    peaks, _ = find_peaks(energy, height=height, distance=distance)
    if len(peaks) == 0:
        return peaks.astype(int)
    refine = int(0.05 * sfreq)
    refined = []
    for peak in peaks:
        lo, hi = max(0, peak - refine), min(len(ecg_clean), peak + refine + 1)
        local = ecg_clean[lo:hi]
        if local.size:
            refined.append(lo + int(np.nanargmax(np.abs(local))))
    rpeaks = np.unique(np.asarray(refined, dtype=int))
    if len(rpeaks) < 3:
        return rpeaks
    rr = np.diff(rpeaks) / sfreq
    keep = np.ones(len(rpeaks), dtype=bool)
    keep[1:] = (rr >= rr_min_s) & (rr <= rr_max_s)
    return rpeaks[keep]


def ecg_r_peak_points_up(ecg_clean: np.ndarray, rpeaks: np.ndarray, sfreq: float,
                           qrs_half_width_ms: float, min_template_beats: int,
                           flip_ratio: float) -> Tuple[np.ndarray, bool, Dict[str, float]]:
    half = int((qrs_half_width_ms / 1000.0) * sfreq)
    usable = rpeaks[(rpeaks > half) & (rpeaks < len(ecg_clean) - half)]
    if len(usable) < min_template_beats:
        return ecg_clean, False, {"reason": "too_few_rpeaks_for_ecg_flip"}
    beats = np.vstack([ecg_clean[p - half:p + half + 1] for p in usable])
    template = np.nanmedian(beats, axis=0)
    pos, neg = float(np.nanmax(template)), float(np.nanmin(template))
    flipped = abs(neg) > flip_ratio * abs(pos)
    details = {"qrs_template_max": pos, "qrs_template_min": neg,
                "n_template_beats": float(len(usable)), "flip_ratio": float(flip_ratio)}
    return (-ecg_clean if flipped else ecg_clean), bool(flipped), details


# ── Epoch building ────────────────────────────────────────────────────────────

def build_epochs(data: np.ndarray, rpeaks: np.ndarray, sfreq: float,
                  window: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre = int(round(abs(window[0]) * sfreq))
    post = int(round(window[1] * sfreq))
    offsets = np.arange(-pre, post + 1)
    valid = rpeaks[(rpeaks - pre >= 0) & (rpeaks + post < data.shape[-1])]
    if data.ndim == 1:
        epochs = np.asarray([data[p + offsets] for p in valid], dtype=float)
    else:
        epochs = np.asarray([data[:, p + offsets] for p in valid], dtype=float)
    return epochs, offsets / sfreq, valid


def robust_epoch_mask(ecg_epochs: np.ndarray, eeg_epochs: np.ndarray,
                       sfreq: Optional[float] = None,
                       max_bad_channel_fraction: float = 0.35,
                       artifact_mad_multiplier: float = 7.5,
                       ecg_low_ptp_percentile: float = 5.0,
                       max_spectral_power_ratio: float = MAX_SPECTRAL_POWER_RATIO) -> Tuple[np.ndarray, List[str]]:
    notes = []
    if len(ecg_epochs) == 0:
        return np.zeros(0, dtype=bool), ["no complete heartbeat windows"]

    ecg_ptp = np.ptp(ecg_epochs, axis=1)
    ecg_diff = np.nanstd(np.diff(ecg_epochs, axis=1), axis=1)
    eeg_ptp = np.ptp(eeg_epochs, axis=2) if eeg_epochs.size else np.empty((len(ecg_epochs), 0))
    eeg_std = np.nanstd(eeg_epochs, axis=2) if eeg_epochs.size else np.empty((len(ecg_epochs), 0))
    eeg_diff = np.nanstd(np.diff(eeg_epochs, axis=2), axis=2) if eeg_epochs.size else np.empty((len(ecg_epochs), 0))

    def robust_upper(values: np.ndarray, mult: float) -> float:
        med = np.nanmedian(values)
        mad = np.nanmedian(np.abs(values - med)) + 1e-12
        return float(med + mult * 1.4826 * mad)

    ecg_ok = np.isfinite(ecg_ptp) & np.isfinite(ecg_diff)
    ecg_ok &= ecg_ptp > np.nanpercentile(ecg_ptp, ecg_low_ptp_percentile)
    ecg_ok &= ecg_ptp < robust_upper(ecg_ptp, artifact_mad_multiplier)
    ecg_ok &= ecg_diff < robust_upper(ecg_diff, artifact_mad_multiplier)

    if eeg_ptp.size:
        ch_ptp_upper = (np.nanmedian(eeg_ptp, axis=0) + artifact_mad_multiplier * 1.4826 *
                        (np.nanmedian(np.abs(eeg_ptp - np.nanmedian(eeg_ptp, axis=0)), axis=0) + 1e-12))
        ch_diff_upper = (np.nanmedian(eeg_diff, axis=0) + artifact_mad_multiplier * 1.4826 *
                         (np.nanmedian(np.abs(eeg_diff - np.nanmedian(eeg_diff, axis=0)), axis=0) + 1e-12))
        bad_by_ch = (eeg_ptp > ch_ptp_upper) | (eeg_diff > ch_diff_upper) | ~np.isfinite(eeg_ptp) | ~np.isfinite(eeg_diff)
        flat_by_ch = (eeg_ptp * 1e6 < EEG_SIGNAL_MIN_PTP_UV) | (eeg_std * 1e6 < EEG_SIGNAL_MIN_STD_UV)
        bad_by_ch |= flat_by_ch
        if sfreq is not None:
            spectral_ratio = compute_epoch_spectral_power_ratios(eeg_epochs, float(sfreq))
            spectral_bad_by_ch = ~np.isfinite(spectral_ratio) | (spectral_ratio >= max_spectral_power_ratio)
            bad_by_ch |= spectral_bad_by_ch
            notes.append(
                f"dropped windows where too many EEG channels had spectral_power_ratio_hf_lf>={max_spectral_power_ratio}"
            )
        notes.append("dropped windows where too many EEG channels were flat/almost flat")
        eeg_ok = np.mean(bad_by_ch, axis=1) <= max_bad_channel_fraction
    else:
        eeg_ok = np.ones(len(ecg_epochs), dtype=bool)

    mask = ecg_ok & eeg_ok
    notes.append(f"kept {int(mask.sum())}/{len(mask)} windows after ECG/EEG noise rejection")
    return mask, notes


def peak_time(trace: np.ndarray, times: np.ndarray, window: Tuple[float, float],
               mode: str = "max_abs") -> Optional[float]:
    mask = (times >= window[0]) & (times <= window[1])
    if not np.any(mask):
        return None
    values, t = np.asarray(trace, dtype=float)[mask], times[mask]
    finite = np.isfinite(values)
    if not np.any(finite):
        return None
    values, t = values[finite], t[finite]
    if mode == "max":
        idx = int(np.nanargmax(values))
    elif mode == "min":
        idx = int(np.nanargmin(values))
    else:
        idx = int(np.nanargmax(np.abs(values)))
    return float(t[idx])


def score_negative_dip_multimethod(
    trace: np.ndarray,
    times: np.ndarray,
    center_time: float,
    pre_window: float,
    post_window: float,
    baseline_pre_window: float,
    swing_threshold: float,
    z_threshold: float,
    prominence_threshold: float,
    min_votes: int,
) -> Tuple[bool, Dict[str, float]]:
    """Vote-based check for whether a negative dip is deep enough to flip."""
    if trace is None or times is None or center_time is None:
        return False, {}

    times = np.asarray(times, dtype=float)
    trace = np.asarray(trace, dtype=float).squeeze()
    if trace.ndim != 1 or len(trace) != len(times):
        return False, {}

    target_mask = (times >= center_time - pre_window) & (times <= center_time + post_window)
    if not np.any(target_mask):
        return False, {}

    target_times = times[target_mask]
    target_values = trace[target_mask]
    finite = np.isfinite(target_values)
    if not np.any(finite):
        return False, {}

    target_times = target_times[finite]
    target_values = target_values[finite]
    if len(target_values) < 3:
        return False, {}

    dip_idx = int(np.nanargmin(target_values))
    max_idx = int(np.nanargmax(target_values))
    dip_amp = float(target_values[dip_idx])
    max_amp = float(target_values[max_idx])
    dip_time = float(target_times[dip_idx])

    left_values = target_values[:dip_idx + 1]
    shoulder_idx = int(np.nanargmax(left_values)) if len(left_values) else max_idx
    shoulder_peak = float(left_values[shoulder_idx]) if len(left_values) else max_amp
    shoulder_time = float(target_times[shoulder_idx]) if len(left_values) else float(target_times[max_idx])
    swing_pct = ((shoulder_peak - dip_amp) / (abs(shoulder_peak) + 1e-12) * 100.0) if shoulder_peak > 0 else 0.0
    swing_pass = bool(dip_amp < 0 and shoulder_peak > 0 and swing_pct >= swing_threshold)

    baseline_mask = (times >= center_time - baseline_pre_window) & (times < center_time - pre_window)
    baseline = trace[baseline_mask]
    baseline = baseline[np.isfinite(baseline)]
    if len(baseline) >= 5:
        baseline_mean = float(np.nanmean(baseline))
        baseline_std = float(np.nanstd(baseline, ddof=1))
    else:
        baseline_mean = float(np.nanmean(target_values))
        baseline_std = float(np.nanstd(target_values, ddof=1)) if len(target_values) > 1 else 0.0
    z_depth = (baseline_mean - dip_amp) / (baseline_std + 1e-12)
    z_pass = bool(dip_amp < baseline_mean - z_threshold * (baseline_std + 1e-12))

    global_range = float(np.nanmax(trace) - np.nanmin(trace))
    prominence = np.nan
    prominence_pct = 0.0
    if np.isfinite(global_range) and global_range > 0:
        troughs, props = find_peaks(-target_values, prominence=0)
        if len(troughs):
            nearest = int(np.argmin(np.abs(troughs - dip_idx)))
            prominence = float(props["prominences"][nearest])
        else:
            left_shoulder = float(np.nanmax(target_values[:dip_idx + 1])) if dip_idx > 0 else dip_amp
            right_shoulder = float(np.nanmax(target_values[dip_idx:])) if dip_idx < len(target_values) - 1 else dip_amp
            prominence = float(min(left_shoulder, right_shoulder) - dip_amp)
        prominence_pct = max(0.0, prominence / (global_range + 1e-12) * 100.0)
    prominence_pass = bool(dip_amp < 0 and prominence_pct >= prominence_threshold)

    vote_count = int(swing_pass) + int(z_pass) + int(prominence_pass)
    flip = bool(dip_amp < 0 and vote_count >= min_votes)
    return flip, {
        "center_time_s": float(center_time),
        "dip_time_s": dip_time,
        "dip_amp": dip_amp,
        "window_max_amp": max_amp,
        "shoulder_peak_amp": shoulder_peak,
        "shoulder_peak_time_s": shoulder_time,
        "swing_pct": float(swing_pct),
        "swing_threshold": float(swing_threshold),
        "swing_pass": bool(swing_pass),
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "z_depth": float(z_depth),
        "z_threshold": float(z_threshold),
        "z_pass": bool(z_pass),
        "prominence": float(prominence) if np.isfinite(prominence) else np.nan,
        "prominence_pct": float(prominence_pct),
        "prominence_threshold": float(prominence_threshold),
        "prominence_pass": bool(prominence_pass),
        "vote_count": int(vote_count),
        "min_votes": int(min_votes),
        "flip_reason": "multi_score_deep_dip" if flip else "multi_score_small_dip",
    }


# ── EEG polarity correction ───────────────────────────────────────────────────

def should_flip_eeg(avg_trace: np.ndarray, times: np.ndarray, ecg_t_peak: float,
                     swing_threshold: float, eeg_t_radius: float, eeg_t_post_s: float,
                     r_peak_flip_window_s: float, z_threshold: float = 3.0,
                     prominence_threshold: float = 25.0, min_votes: int = 2,
                     baseline_pre_window: float = 0.50,
                     r_peak_curvature_threshold: float = 0.0) -> Tuple[bool, Dict[str, float]]:
    """
    EEG polarity check based only on the ECG-aligned average around the R peak.

    The correctly oriented EEG R response should be a positive local deflection
    with a downward-facing quadratic fit between -10 ms and +100 ms from the ECG
    R peak at t=0. Negative/upward R shapes are inverted.
    """
    del (
        ecg_t_peak, swing_threshold, eeg_t_radius, eeg_t_post_s,
        r_peak_flip_window_s, z_threshold, prominence_threshold, min_votes,
        baseline_pre_window,
    )

    times = np.asarray(times, dtype=float)
    trace_uv = np.asarray(avg_trace, dtype=float).squeeze() * 1e6
    if trace_uv.ndim != 1 or len(trace_uv) != len(times):
        return False, {}

    finite = np.isfinite(trace_uv) & np.isfinite(times)
    r_window_start, r_window_end = EEG_R_PEAK_FLIP_WINDOW
    search_mask = finite & (times >= r_window_start) & (times <= r_window_end)
    if np.sum(search_mask) < 3:
        return False, {
            "method": "r_peak_parabola",
            "r_search_start_s": float(r_window_start),
            "r_search_end_s": float(r_window_end),
            "flip_reason": "too_few_r_window_samples",
        }

    search_indices = np.flatnonzero(search_mask)
    search_values = trace_uv[search_indices]
    local_baseline = float(np.nanmedian(search_values))
    centered_values = search_values - local_baseline
    positive_peaks, _ = find_peaks(centered_values)
    negative_peaks, _ = find_peaks(-centered_values)
    extrema = np.concatenate([positive_peaks, negative_peaks])
    if len(extrema):
        candidate_rel = int(extrema[int(np.nanargmax(np.abs(centered_values[extrema])))])
    else:
        candidate_rel = int(np.nanargmax(np.abs(centered_values)))

    candidate_idx = int(search_indices[candidate_rel])
    eeg_r_peak_time = float(times[candidate_idx])
    eeg_r_peak_amp = float(trace_uv[candidate_idx])
    eeg_r_peak_deflection = float(eeg_r_peak_amp - local_baseline)

    finite_times = np.sort(times[finite])
    if len(finite_times) > 1:
        dt = float(np.nanmedian(np.diff(finite_times)))
    else:
        dt = EEG_R_PEAK_FIT_MIN_HALF_WINDOW / 3.0
    fit_half_window = float(np.clip(
        3.0 * max(dt, 1e-6),
        EEG_R_PEAK_FIT_MIN_HALF_WINDOW,
        EEG_R_PEAK_FIT_MAX_HALF_WINDOW,
    ))
    fit_mask = finite & (times >= eeg_r_peak_time - fit_half_window) & (
        times <= eeg_r_peak_time + fit_half_window
    )
    if np.sum(fit_mask) < 5:
        fit_mask = search_mask
        fit_half_window = max(
            abs(eeg_r_peak_time - r_window_start),
            abs(r_window_end - eeg_r_peak_time),
        )
    if np.sum(fit_mask) < 3:
        return False, {
            "method": "r_peak_parabola",
            "r_search_start_s": float(r_window_start),
            "r_search_end_s": float(r_window_end),
            "eeg_r_peak_time_s": eeg_r_peak_time,
            "eeg_r_peak_amp_uv": eeg_r_peak_amp,
            "eeg_r_peak_deflection_uv": eeg_r_peak_deflection,
            "flip_reason": "too_few_fit_samples",
        }

    x_ms = (times[fit_mask] - eeg_r_peak_time) * 1000.0
    y_uv = trace_uv[fit_mask]
    if len(np.unique(x_ms)) < 3:
        return False, {
            "method": "r_peak_parabola",
            "r_search_start_s": float(r_window_start),
            "r_search_end_s": float(r_window_end),
            "eeg_r_peak_time_s": eeg_r_peak_time,
            "eeg_r_peak_amp_uv": eeg_r_peak_amp,
            "eeg_r_peak_deflection_uv": eeg_r_peak_deflection,
            "flip_reason": "too_few_unique_fit_times",
        }

    coef = np.polyfit(x_ms, y_uv, deg=2)
    curvature = float(coef[0])
    center_amp = float(np.polyval(coef, 0.0))
    left_amp = float(np.polyval(coef, float(np.min(x_ms))))
    right_amp = float(np.polyval(coef, float(np.max(x_ms))))
    edge_mean = (left_amp + right_amp) / 2.0
    center_minus_edges = center_amp - edge_mean

    curvature_threshold = float(r_peak_curvature_threshold)
    r_peak_positive = bool(center_minus_edges >= 0 and eeg_r_peak_deflection >= 0)
    downward_parabola = bool(curvature <= curvature_threshold)
    flip = bool((not r_peak_positive) or (not downward_parabola))

    if not r_peak_positive and not downward_parabola:
        flip_reason = "negative_upward_r_peak_parabola"
    elif not r_peak_positive:
        flip_reason = "negative_r_peak_deflection"
    elif not downward_parabola:
        flip_reason = "upward_r_peak_parabola"
    else:
        flip_reason = "positive_downward_or_flat_r_peak_parabola"

    return flip, {
        "method": "r_peak_parabola",
        "r_search_start_s": float(r_window_start),
        "r_search_end_s": float(r_window_end),
        "eeg_r_peak_time_s": eeg_r_peak_time,
        "eeg_r_peak_time_ms": eeg_r_peak_time * 1000.0,
        "eeg_r_peak_amp_uv": eeg_r_peak_amp,
        "eeg_r_peak_deflection_uv": eeg_r_peak_deflection,
        "r_window_baseline_uv": local_baseline,
        "r_peak_fit_half_window_s": float(fit_half_window),
        "parabola_curvature_uv_per_ms2": curvature,
        "parabola_center_amp_uv": center_amp,
        "parabola_left_edge_amp_uv": left_amp,
        "parabola_right_edge_amp_uv": right_amp,
        "parabola_center_minus_edges_uv": center_minus_edges,
        "parabola_faces": "up" if curvature > 0 else "down",
        "curvature_threshold_uv_per_ms2": curvature_threshold,
        "r_peak_positive": float(r_peak_positive),
        "r_peak_method_flip": float(flip),
        "final_flip": float(flip),
        "flip_reason": flip_reason,
    }


# ── Patient processing ────────────────────────────────────────────────────────

def load_raw(path: str) -> mne.io.BaseRaw:
    with open(path, "rb") as f:
        raw = pickle.load(f)
    if not hasattr(raw, "get_data"):
        raise TypeError(f"{path} is not an MNE Raw object")
    return raw


def compute_spectral_power_ratios(
    eeg_data: np.ndarray,
    eeg_channels: Sequence[str],
    sfreq: float,
    low_band: Tuple[float, float] = (1.0, 30.0),
    high_band: Tuple[float, float] = (30.0, 80.0),
) -> Dict[str, float]:
    """High-frequency power divided by low-frequency EEG power per channel."""
    ratios: Dict[str, float] = {}
    if eeg_data.size == 0:
        return ratios
    nperseg = int(min(max(256, sfreq * 4), eeg_data.shape[-1]))
    freqs, psd = signal.welch(eeg_data, fs=sfreq, axis=-1, nperseg=nperseg)
    low_mask = (freqs >= low_band[0]) & (freqs <= low_band[1])
    high_mask = (freqs >= high_band[0]) & (freqs <= min(high_band[1], sfreq / 2.0))
    for idx, ch in enumerate(eeg_channels):
        low_power = float(np.trapz(psd[idx, low_mask], freqs[low_mask])) if np.any(low_mask) else np.nan
        high_power = float(np.trapz(psd[idx, high_mask], freqs[high_mask])) if np.any(high_mask) else np.nan
        ratios[ch] = high_power / (low_power + 1e-18) if np.isfinite(low_power) and low_power > 0 else np.nan
    return ratios


def compute_epoch_spectral_power_ratios(
    eeg_epochs: np.ndarray,
    sfreq: float,
    low_band: Tuple[float, float] = (1.0, 30.0),
    high_band: Tuple[float, float] = (30.0, 80.0),
) -> np.ndarray:
    """High-frequency/low-frequency power ratio per epoch and EEG channel."""
    epochs = np.asarray(eeg_epochs, dtype=float)
    if epochs.ndim != 3 or epochs.size == 0:
        return np.empty((0, 0), dtype=float)
    n_times = epochs.shape[-1]
    if n_times < 4:
        return np.full(epochs.shape[:2], np.nan, dtype=float)

    nperseg = int(min(max(8, round(sfreq * 0.5)), n_times))
    freqs, psd = signal.welch(epochs, fs=sfreq, axis=-1, nperseg=nperseg)
    low_mask = (freqs >= low_band[0]) & (freqs <= min(low_band[1], sfreq / 2.0))
    high_mask = (freqs >= high_band[0]) & (freqs <= min(high_band[1], sfreq / 2.0))
    ratios = np.full(epochs.shape[:2], np.nan, dtype=float)
    if not np.any(low_mask) or not np.any(high_mask):
        return ratios
    low_power = np.trapz(psd[..., low_mask], freqs[low_mask], axis=-1)
    high_power = np.trapz(psd[..., high_mask], freqs[high_mask], axis=-1)
    valid_low = np.isfinite(low_power) & (low_power > 0)
    ratios[valid_low] = high_power[valid_low] / (low_power[valid_low] + 1e-18)
    return ratios


def assess_signal_quality(
    trace: np.ndarray,
    scale: float = 1.0,
    min_finite_fraction: float = EEG_SIGNAL_MIN_FINITE_FRACTION,
    min_ptp: Optional[float] = None,
    min_std: Optional[float] = None,
    max_abs: Optional[float] = None,
    max_roughness: Optional[float] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Return whether a trace is usable plus flat-line/noise diagnostics."""
    x = np.asarray(trace, dtype=float).ravel() * float(scale)
    total_n = int(len(x))
    finite = np.isfinite(x)
    finite_fraction = float(np.mean(finite)) if total_n else 0.0
    values = x[finite]
    info: Dict[str, Any] = {
        "quality_ok": False,
        "quality_reason": "",
        "finite_fraction": finite_fraction,
        "ptp": np.nan,
        "std": np.nan,
        "max_abs": np.nan,
        "roughness": np.nan,
    }
    if total_n == 0 or finite_fraction < min_finite_fraction or len(values) < 4:
        info["quality_reason"] = "too_few_finite_samples"
        return False, info

    ptp = float(np.nanmax(values) - np.nanmin(values))
    std = float(np.nanstd(values))
    max_abs_val = float(np.nanmax(np.abs(values)))
    diffs = np.diff(values)
    diff_std = float(np.nanstd(diffs)) if len(diffs) else np.nan
    roughness = diff_std / (std + 1e-12) if np.isfinite(diff_std) else np.nan
    info.update({"ptp": ptp, "std": std, "max_abs": max_abs_val, "roughness": roughness})

    reasons = []
    if min_ptp is not None and ptp < min_ptp:
        reasons.append("flat_low_ptp")
    if min_std is not None and std < min_std:
        reasons.append("flat_low_std")
    if max_abs is not None and max_abs_val > max_abs:
        reasons.append("excessive_amplitude")
    if max_roughness is not None and np.isfinite(roughness) and roughness > max_roughness:
        reasons.append("rough_high_frequency_noise")

    ok = not reasons
    info["quality_ok"] = ok
    info["quality_reason"] = "ok" if ok else ";".join(reasons)
    return ok, info


def assess_eeg_average_quality(trace: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
    ok, info = assess_signal_quality(
        trace,
        scale=1e6,
        min_ptp=EEG_SIGNAL_MIN_PTP_UV,
        min_std=EEG_SIGNAL_MIN_STD_UV,
        max_abs=EEG_SIGNAL_MAX_ABS_UV,
        max_roughness=EEG_SIGNAL_MAX_ROUGHNESS,
    )
    return ok, {f"eeg_signal_{k}": v for k, v in info.items()}


def assess_raw_eeg_quality(trace: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
    ok, info = assess_signal_quality(
        trace,
        scale=1e6,
        min_ptp=RAW_EEG_SIGNAL_MIN_PTP_UV,
        min_std=RAW_EEG_SIGNAL_MIN_STD_UV,
        max_abs=RAW_EEG_SIGNAL_MAX_ABS_UV,
        max_roughness=None,
    )
    return ok, info


def ica_clean_eeg_ecg_artifact(
    eeg_data: np.ndarray,
    ecg_clean: np.ndarray,
    rpeaks: np.ndarray,
    sfreq: float,
    window: Tuple[float, float],
    n_remove: int = ICA_COMPONENTS_TO_REMOVE,
    max_components: int = ICA_MAX_COMPONENTS,
    max_fit_samples: int = ICA_MAX_FIT_SAMPLES,
    random_state: int = 97,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Remove the most ECG-correlated ICA components from EEG data."""
    eeg_data = np.asarray(eeg_data, dtype=float)
    ecg_clean = np.asarray(ecg_clean, dtype=float).ravel()
    n_channels, n_samples = eeg_data.shape if eeg_data.ndim == 2 else (0, 0)
    details: Dict[str, Any] = {
        "ica_applied": False,
        "components_removed_count": 0,
        "component_summary": pd.DataFrame(),
        "component_times": None,
        "component_average_z": None,
    }
    if n_channels < 2 or n_samples < 10 or len(ecg_clean) != n_samples:
        details["ica_reason"] = "too_few_channels_or_samples"
        return eeg_data, details

    filled = eeg_data.copy()
    ch_mean = np.nanmean(filled, axis=1)
    ch_mean = np.where(np.isfinite(ch_mean), ch_mean, 0.0)
    for ch_idx in range(n_channels):
        bad = ~np.isfinite(filled[ch_idx])
        if np.any(bad):
            filled[ch_idx, bad] = ch_mean[ch_idx]
    ch_std = np.nanstd(filled, axis=1)
    ch_std = np.where(np.isfinite(ch_std) & (ch_std > 1e-18), ch_std, 1.0)
    x_scaled = ((filled - ch_mean[:, None]) / ch_std[:, None]).T

    n_fit = min(int(max_fit_samples), n_samples)
    if n_fit < 10:
        details["ica_reason"] = "too_few_fit_samples"
        return eeg_data, details
    fit_idx = np.linspace(0, n_samples - 1, n_fit, dtype=int) if n_fit < n_samples else np.arange(n_samples)
    x_fit = x_scaled[fit_idx]
    n_components = int(min(max_components, n_channels, max(2, len(fit_idx) - 1)))
    if n_components < 2:
        details["ica_reason"] = "too_few_ica_components"
        return eeg_data, details

    try:
        ica = FastICA(
            n_components=n_components,
            whiten="unit-variance",
            random_state=random_state,
            max_iter=500,
            tol=0.001,
        )
        sources_fit = ica.fit_transform(x_fit)
        sources_full = ica.transform(x_scaled)
    except Exception as exc:
        details["ica_reason"] = f"ica_failed: {exc}"
        return eeg_data, details

    ecg_fit = ecg_clean[fit_idx] - np.nanmean(ecg_clean[fit_idx])
    ecg_sd = np.nanstd(ecg_fit)
    component_vars = np.nanvar(sources_fit, axis=0)
    total_var = float(np.nansum(component_vars)) + 1e-12
    rows = []
    corrs = []
    for comp_idx in range(n_components):
        comp = sources_fit[:, comp_idx]
        comp_sd = np.nanstd(comp)
        if np.isfinite(ecg_sd) and ecg_sd > 0 and np.isfinite(comp_sd) and comp_sd > 0:
            corr = float(np.corrcoef(comp, ecg_fit)[0, 1])
        else:
            corr = np.nan
        corrs.append(corr)
        rows.append({
            "component": int(comp_idx),
            "ecg_corr": corr,
            "abs_ecg_corr": abs(corr) if np.isfinite(corr) else np.nan,
            "variance": float(component_vars[comp_idx]),
            "variance_ratio_pct": float(100.0 * component_vars[comp_idx] / total_var),
        })

    rank = np.argsort(np.nan_to_num(np.abs(corrs), nan=-np.inf))[::-1]
    removed = rank[:min(int(n_remove), len(rank))]
    sources_clean = sources_full.copy()
    sources_clean[:, removed] = 0.0
    clean_scaled = ica.inverse_transform(sources_clean)
    cleaned = (clean_scaled.T * ch_std[:, None]) + ch_mean[:, None]

    component_table = pd.DataFrame(rows)
    component_table["ecg_rank"] = component_table["component"].map(
        {int(comp): rank_idx + 1 for rank_idx, comp in enumerate(rank)}
    )
    component_table["removed"] = component_table["component"].isin([int(c) for c in removed])
    component_table = component_table.sort_values("ecg_rank").reset_index(drop=True)

    top = [int(c) for c in removed[:2]]
    comp_avg_z = None
    comp_times = None
    if top:
        comp_epochs, comp_times, _ = build_epochs(sources_full[:, top].T, rpeaks, sfreq, window)
        if comp_epochs.size:
            comp_avg = np.nanmedian(comp_epochs, axis=0)
            comp_avg_z = np.vstack([zscore_1d(row) for row in comp_avg])

    details.update({
        "ica_applied": True,
        "ica_reason": "ok",
        "components_removed": [int(c) for c in removed],
        "components_removed_count": int(len(removed)),
        "n_components": int(n_components),
        "n_fit_samples": int(len(fit_idx)),
        "component_summary": component_table,
        "component_times": comp_times,
        "component_average_z": comp_avg_z,
    })
    return cleaned, details


def process_patient_file(
    file_path: str, group: str, stage: str,
    window: Tuple[float, float], t_window: Tuple[float, float],
    swing_threshold: float, max_bad_epoch_channel_fraction: float,
    eeg_t_radius: float = DEFAULT_EEG_T_RADIUS, eeg_t_post_s: float = 0.03,
    r_peak_flip_window_s: float = 0.10, min_kept_epochs: int = 5,
    flip_z_threshold: float = 3.0, flip_prominence_threshold: float = 25.0,
    flip_min_votes: int = 2, flip_baseline_pre_window: float = 0.50,
    r_peak_curvature_threshold: float = 0.0,
    ecg_filter_low_hz: float = 0.5, ecg_filter_high_hz: float = 40.0,
    ecg_median_ms: float = 20.0, ecg_clip_sd: float = 6.0,
    qrs_filter_low_hz: float = 5.0, qrs_filter_high_hz: float = 25.0,
    rpeak_mad_multiplier: float = 4.0, rpeak_refractory_s: float = 0.4,
    rr_min_s: float = 0.4, rr_max_s: float = 2.2,
    ecg_flip_qrs_half_width_ms: float = 80.0, ecg_flip_min_beats: int = 5,
    ecg_flip_ratio: float = 0.9, artifact_mad_multiplier: float = 7.5,
    ecg_low_ptp_percentile: float = 5.0,
    ica_ecg_clean: bool = False, ica_components_to_remove: int = ICA_COMPONENTS_TO_REMOVE,
    ica_max_components: int = ICA_MAX_COMPONENTS, ica_max_fit_samples: int = ICA_MAX_FIT_SAMPLES,
) -> Optional[PatientResult]:
    raw = load_raw(file_path).copy().load_data(verbose=False)
    patient_id = patient_id_from_path(file_path, stage)
    sfreq = float(raw.info["sfreq"])
    ch_names = list(raw.ch_names)
    ecg_indices = [i for i, ch in enumerate(ch_names) if is_ecg_channel(ch)]
    eeg_indices = [i for i, ch in enumerate(ch_names) if is_eeg_channel(ch)]
    if not ecg_indices or not eeg_indices:
        return None

    ecg_channel = ch_names[ecg_indices[0]]
    data = raw.get_data()
    eeg_channels_all = [ch_names[i] for i in eeg_indices]
    eeg_data_all = data[eeg_indices].astype(float, copy=True)
    raw_quality_notes: List[str] = []
    keep_eeg = []
    dropped_eeg = []
    for idx, ch_name in enumerate(eeg_channels_all):
        ok, qinfo = assess_raw_eeg_quality(eeg_data_all[idx])
        if ok:
            keep_eeg.append(idx)
        else:
            dropped_eeg.append(f"{ch_name}:{qinfo.get('quality_reason', 'bad_signal')}")
    if not keep_eeg:
        return None
    if dropped_eeg:
        raw_quality_notes.append(
            "dropped flat/artifact EEG channel(s) before averaging: " + ", ".join(dropped_eeg[:12])
            + (f", +{len(dropped_eeg) - 12} more" if len(dropped_eeg) > 12 else "")
        )
    eeg_channels = [eeg_channels_all[i] for i in keep_eeg]

    ecg_clean = clean_ecg(data[ecg_indices[0]], sfreq, ecg_filter_low_hz, ecg_filter_high_hz,
                           ecg_median_ms, ecg_clip_sd)
    prelim_rpeaks = detect_rpeaks(ecg_clean, sfreq, qrs_filter_low_hz, qrs_filter_high_hz,
                                   rpeak_mad_multiplier, rpeak_refractory_s, rr_min_s, rr_max_s)
    ecg_clean, flipped_ecg, _ = ecg_r_peak_points_up(
        ecg_clean, prelim_rpeaks, sfreq, ecg_flip_qrs_half_width_ms, ecg_flip_min_beats, ecg_flip_ratio)
    rpeaks = detect_rpeaks(ecg_clean, sfreq, qrs_filter_low_hz, qrs_filter_high_hz,
                            rpeak_mad_multiplier, rpeak_refractory_s, rr_min_s, rr_max_s)
    if len(rpeaks) < min_kept_epochs:
        return None

    eeg_data = eeg_data_all[keep_eeg]
    ica_details: Optional[Dict[str, Any]] = None
    if ica_ecg_clean:
        eeg_data, ica_details = ica_clean_eeg_ecg_artifact(
            eeg_data,
            ecg_clean,
            rpeaks,
            sfreq,
            window,
            n_remove=ica_components_to_remove,
            max_components=ica_max_components,
            max_fit_samples=ica_max_fit_samples,
        )
    spectral_power_ratios = compute_spectral_power_ratios(eeg_data, eeg_channels, sfreq)
    ecg_epochs, times, valid_rpeaks = build_epochs(ecg_clean, rpeaks, sfreq, window)
    eeg_epochs, _, _ = build_epochs(eeg_data, rpeaks, sfreq, window)
    mask, quality_notes = robust_epoch_mask(
        ecg_epochs, eeg_epochs,
        sfreq=sfreq,
        max_bad_channel_fraction=max_bad_epoch_channel_fraction,
        artifact_mad_multiplier=artifact_mad_multiplier,
        ecg_low_ptp_percentile=ecg_low_ptp_percentile,
        max_spectral_power_ratio=MAX_SPECTRAL_POWER_RATIO,
    )
    quality_notes = raw_quality_notes + quality_notes
    if ica_details is not None:
        removed = ica_details.get("components_removed", [])
        quality_notes.append(
            f"ICA ECG cleanup {ica_details.get('ica_reason', 'unknown')}; removed components {removed}"
        )
    if int(mask.sum()) < min_kept_epochs:
        return None

    ecg_avg = np.nanmedian(ecg_epochs[mask], axis=0)
    eeg_avg = np.nanmedian(eeg_epochs[mask], axis=0)
    ecg_ok, ecg_quality = assess_signal_quality(
        ecg_avg,
        scale=1.0,
        min_ptp=1e-12,
        min_std=1e-13,
        max_abs=None,
        max_roughness=None,
    )
    quality_notes.append(
        f"ECG average quality: {ecg_quality.get('quality_reason', 'unknown')}"
    )
    if not ecg_ok:
        return None
    ecg_t = peak_time(ecg_avg, times, t_window, mode="max")
    if ecg_t is None:
        return None

    flipped_channels: List[str] = []
    flip_details: Dict[str, Dict[str, float]] = {}
    for idx, ch_name in enumerate(eeg_channels):
        flip, details = should_flip_eeg(
            eeg_avg[idx], times, ecg_t,
            swing_threshold=swing_threshold, eeg_t_radius=eeg_t_radius,
            eeg_t_post_s=eeg_t_post_s, r_peak_flip_window_s=r_peak_flip_window_s,
            z_threshold=flip_z_threshold,
            prominence_threshold=flip_prominence_threshold,
            min_votes=flip_min_votes,
            baseline_pre_window=flip_baseline_pre_window,
            r_peak_curvature_threshold=r_peak_curvature_threshold,
        )
        flip_details[ch_name] = details
        if flip:
            eeg_epochs[:, idx, :] *= -1.0
            flipped_channels.append(ch_name)

    if flipped_channels:
        eeg_avg = np.nanmedian(eeg_epochs[mask], axis=0)

    return PatientResult(
        group=group, stage=stage, patient_id=patient_id, file_path=file_path,
        sfreq=sfreq, n_rpeaks=int(len(rpeaks)), n_epochs_total=int(len(valid_rpeaks)),
        n_epochs_kept=int(mask.sum()), ecg_channel=ecg_channel,
        eeg_channels=eeg_channels, flipped_ecg=flipped_ecg,
        flipped_eeg_channels=flipped_channels, times=times,
        ecg_average=ecg_avg, eeg_average=eeg_avg, ecg_t_peak_s=float(ecg_t),
        quality_notes=quality_notes, flip_details=flip_details,
        spectral_power_ratios=spectral_power_ratios, ica_details=ica_details,
    )


# ── Feature extraction ────────────────────────────────────────────────────────

def distance_correlation_1d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Distance correlation for one-dimensional signals.

    Unlike Pearson correlation, distance correlation can capture non-linear
    dependence between ECG and EEG waveform shapes. Values are in [0, 1], where
    0 means no detected dependence and larger values indicate stronger coupling.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 4 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan

    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    a_centered = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    b_centered = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()

    dcov2 = np.mean(a_centered * b_centered)
    dvar_x = np.mean(a_centered * a_centered)
    dvar_y = np.mean(b_centered * b_centered)
    if dvar_x <= 0 or dvar_y <= 0:
        return np.nan
    return float(np.sqrt(max(dcov2, 0.0) / np.sqrt(dvar_x * dvar_y)))


def first_difference_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation between first differences of two signals."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 4:
        return np.nan
    dx = np.diff(x)
    dy = np.diff(y)
    if len(dx) < 3 or np.nanstd(dx) == 0 or np.nanstd(dy) == 0:
        return np.nan
    return float(np.corrcoef(dx, dy)[0, 1])


def patient_to_feature_rows(result: PatientResult, t_window: Tuple[float, float],
                              eeg_radius: float) -> List[Dict[str, object]]:
    rows = []
    eeg_window = (
        max(t_window[0], result.ecg_t_peak_s - eeg_radius),
        min(t_window[1], result.ecg_t_peak_s + eeg_radius),
    )
    for idx, channel in enumerate(result.eeg_channels):
        trace = result.eeg_average[idx]
        signal_ok, signal_quality = assess_eeg_average_quality(trace)
        ratio = result.spectral_power_ratios.get(channel, np.nan)
        spectral_ok = bool(np.isfinite(ratio) and ratio < MAX_SPECTRAL_POWER_RATIO)
        base_row = {
            "group": result.group,
            "stage": result.stage,
            "patient_id": result.patient_id,
            "channel": channel,
            "ecg_t_peak_ms": result.ecg_t_peak_s * 1000.0,
            "spectral_power_ratio_hf_lf": ratio,
            "signal_quality_rejected": not (signal_ok and spectral_ok),
            "signal_quality_reject_reason": (
                signal_quality.get("eeg_signal_quality_reason", "bad_signal")
                if not signal_ok else
                f"spectral_power_ratio_hf_lf>={MAX_SPECTRAL_POWER_RATIO}"
                if not spectral_ok else "ok"
            ),
            "n_epochs_kept": result.n_epochs_kept,
            "n_epochs_total": result.n_epochs_total,
            "flipped_eeg": channel in result.flipped_eeg_channels,
            "flipped_ecg": result.flipped_ecg,
            **signal_quality,
        }
        if not signal_ok or not spectral_ok:
            rows.append({
                **base_row,
                "eeg_t_peak_ms": np.nan,
                "eeg_positive_t_peak_ms": np.nan,
                "eeg_t_peak_amplitude_uv": np.nan,
                "ecg_eeg_distance_corr_epoch": np.nan,
                "ecg_eeg_distance_corr_twave": np.nan,
                "ecg_eeg_firstdiff_corr_epoch": np.nan,
                "ecg_eeg_firstdiff_corr_twave": np.nan,
                "distance_ms": np.nan,
                "signed_distance_ms": np.nan,
            })
            continue
        eeg_t_abs = peak_time(trace, result.times, eeg_window, mode="max_abs")
        eeg_t_pos = peak_time(trace, result.times, eeg_window, mode="max")
        if eeg_t_abs is None:
            rows.append({
                **base_row,
                "signal_quality_rejected": True,
                "signal_quality_reject_reason": "no_eeg_t_peak_in_window",
                "eeg_t_peak_ms": np.nan,
                "eeg_positive_t_peak_ms": np.nan,
                "eeg_t_peak_amplitude_uv": np.nan,
                "ecg_eeg_distance_corr_epoch": np.nan,
                "ecg_eeg_distance_corr_twave": np.nan,
                "ecg_eeg_firstdiff_corr_epoch": np.nan,
                "ecg_eeg_firstdiff_corr_twave": np.nan,
                "distance_ms": np.nan,
                "signed_distance_ms": np.nan,
            })
            continue
        # Amplitude at the detected EEG T-peak (µV)
        t_mask = (result.times >= eeg_window[0]) & (result.times <= eeg_window[1])
        eeg_amp_uv = float(trace[t_mask][int(np.nanargmax(np.abs(trace[t_mask])))] * 1e6) if np.any(t_mask) else np.nan
        ecg_trace = np.asarray(result.ecg_average, dtype=float)
        eeg_trace = np.asarray(trace, dtype=float)
        ecg_eeg_dcor_epoch = distance_correlation_1d(ecg_trace, eeg_trace)
        ecg_eeg_dcor_twave = distance_correlation_1d(ecg_trace[t_mask], eeg_trace[t_mask]) if np.any(t_mask) else np.nan
        ecg_eeg_firstdiff_corr_epoch = first_difference_corr(ecg_trace, eeg_trace)
        ecg_eeg_firstdiff_corr_twave = first_difference_corr(ecg_trace[t_mask], eeg_trace[t_mask]) if np.any(t_mask) else np.nan
        rows.append({
            **base_row,
            "signal_quality_rejected": False,
            "signal_quality_reject_reason": "ok",
            "eeg_t_peak_ms": eeg_t_abs * 1000.0,
            "eeg_positive_t_peak_ms": np.nan if eeg_t_pos is None else eeg_t_pos * 1000.0,
            "eeg_t_peak_amplitude_uv": eeg_amp_uv,
            "ecg_eeg_distance_corr_epoch": ecg_eeg_dcor_epoch,
            "ecg_eeg_distance_corr_twave": ecg_eeg_dcor_twave,
            "ecg_eeg_firstdiff_corr_epoch": ecg_eeg_firstdiff_corr_epoch,
            "ecg_eeg_firstdiff_corr_twave": ecg_eeg_firstdiff_corr_twave,
            "distance_ms": abs(eeg_t_abs - result.ecg_t_peak_s) * 1000.0,
            "signed_distance_ms": (eeg_t_abs - result.ecg_t_peak_s) * 1000.0,
        })
    return rows


# ── Statistics ────────────────────────────────────────────────────────────────

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


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled = ((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2)
    sp = np.sqrt(pooled) if pooled > 0 else np.nan
    return float((np.mean(a) - np.mean(b)) / sp) if np.isfinite(sp) else np.nan


def bootstrap_median_ci(data: np.ndarray, n_boot: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
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


def group_channel_stats(feature_df: pd.DataFrame, groups: Sequence[str],
                         metric: str = "distance_ms", boot_ci: bool = True) -> pd.DataFrame:
    if feature_df.empty or len(groups) < 2 or metric not in feature_df.columns:
        return pd.DataFrame()
    rows = []
    for channel, ch_df in feature_df.groupby("channel"):
        for g1, g2 in combinations(groups, 2):
            a = ch_df.loc[ch_df["group"] == g1, metric].dropna().to_numpy()
            b = ch_df.loc[ch_df["group"] == g2, metric].dropna().to_numpy()
            if len(a) < 2 or len(b) < 2:
                continue
            u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
            t_stat, p_welch = stats.ttest_ind(a, b, equal_var=False)
            ci_a = bootstrap_median_ci(a) if boot_ci else (np.nan, np.nan)
            ci_b = bootstrap_median_ci(b) if boot_ci else (np.nan, np.nan)
            rows.append({
                "channel": channel,
                "comparison": f"{g1} vs {g2}",
                "group_a": g1,
                "group_b": g2,
                "group_a_n": int(len(a)),
                "group_b_n": int(len(b)),
                "group_a_median": float(np.median(a)),
                "group_a_ci_lo": ci_a[0],
                "group_a_ci_hi": ci_a[1],
                "group_b_median": float(np.median(b)),
                "group_b_ci_lo": ci_b[0],
                "group_b_ci_hi": ci_b[1],
                f"{g1}_n": int(len(a)),
                f"{g2}_n": int(len(b)),
                f"{g1}_median": float(np.median(a)),
                f"{g1}_ci_lo": ci_a[0],
                f"{g1}_ci_hi": ci_a[1],
                f"{g2}_median": float(np.median(b)),
                f"{g2}_ci_lo": ci_b[0],
                f"{g2}_ci_hi": ci_b[1],
                "median_delta": float(np.median(a) - np.median(b)),
                "mann_whitney_u": float(u_stat),
                "p_value": float(p_val),
                "welch_t": float(t_stat),
                "p_welch": float(p_welch),
                "rank_biserial_r": rank_biserial_r(a, b),
                "cohens_d": cohens_d(a, b),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("p_value")
    df["p_fdr_bh"] = benjamini_hochberg(df["p_value"].to_numpy())
    df["p_bonferroni"] = np.minimum(df["p_value"].to_numpy() * len(df), 1.0)
    return df


def patient_median_stats(feature_df: pd.DataFrame, groups: Sequence[str],
                          metric: str = "distance_ms") -> Tuple[pd.DataFrame, pd.DataFrame]:
    patient_df = (
        feature_df
        .groupby(["group", "patient_id"], as_index=False)
        .agg(
            median_distance_ms=(metric, "median"),
            median_signed_distance_ms=("signed_distance_ms", "median"),
            n_channels=("channel", "nunique"),
        )
    )
    if len(groups) < 2 or patient_df.empty:
        return patient_df, pd.DataFrame()

    rows = []
    for g1, g2 in combinations(groups, 2):
        a = patient_df.loc[patient_df["group"] == g1, "median_distance_ms"].dropna().to_numpy()
        b = patient_df.loc[patient_df["group"] == g2, "median_distance_ms"].dropna().to_numpy()
        if len(a) < 2 or len(b) < 2:
            continue

        u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
        ci_a, ci_b = bootstrap_median_ci(a), bootstrap_median_ci(b)
        rows.append({
            "comparison": f"{g1} vs {g2}",
            "group_a": g1,
            "group_b": g2,
            "group_a_n": int(len(a)),
            "group_b_n": int(len(b)),
            "group_a_median": float(np.median(a)),
            "group_a_ci_lo": ci_a[0],
            "group_a_ci_hi": ci_a[1],
            "group_b_median": float(np.median(b)),
            "group_b_ci_lo": ci_b[0],
            "group_b_ci_hi": ci_b[1],
            f"{g1}_n": int(len(a)),
            f"{g2}_n": int(len(b)),
            f"{g1}_median": float(np.median(a)),
            f"{g1}_ci_lo": ci_a[0],
            f"{g1}_ci_hi": ci_a[1],
            f"{g2}_median": float(np.median(b)),
            f"{g2}_ci_lo": ci_b[0],
            f"{g2}_ci_hi": ci_b[1],
            "median_delta": float(np.median(a) - np.median(b)),
            "mann_whitney_u": float(u_stat),
            "p_value": float(p_val),
            "rank_biserial_r": rank_biserial_r(a, b),
            "cohens_d": cohens_d(a, b),
        })
    if not rows:
        return patient_df, pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("p_value")
    df["p_fdr_bh"] = benjamini_hochberg(df["p_value"].to_numpy())
    df["p_bonferroni"] = np.minimum(df["p_value"].to_numpy() * len(df), 1.0)
    return patient_df, df


def apply_spectral_quality_filter(
    feature_df: pd.DataFrame,
    max_ratio: float = MAX_SPECTRAL_POWER_RATIO,
) -> pd.DataFrame:
    if feature_df.empty or "spectral_power_ratio_hf_lf" not in feature_df.columns:
        return feature_df.copy()
    ratio = pd.to_numeric(feature_df["spectral_power_ratio_hf_lf"], errors="coerce")
    keep = ratio.notna() & (ratio < max_ratio)
    if "signal_quality_rejected" in feature_df.columns:
        keep &= ~feature_df["signal_quality_rejected"].fillna(True).astype(bool)
    if "eeg_signal_quality_ok" in feature_df.columns:
        keep &= feature_df["eeg_signal_quality_ok"].fillna(False).astype(bool)
    if "eeg_signal_max_abs" in feature_df.columns:
        max_abs = pd.to_numeric(feature_df["eeg_signal_max_abs"], errors="coerce")
        keep &= max_abs.notna() & (max_abs <= EEG_SIGNAL_MAX_ABS_UV)
    return feature_df.loc[keep].copy()


# ── Demographics ─────────────────────────────────────────────────────────────

def _norm_col_name(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _first_matching_col(columns: Sequence[Any], candidates: Sequence[str]) -> Optional[str]:
    norm_to_col = {_norm_col_name(col): col for col in columns}
    for cand in candidates:
        if _norm_col_name(cand) in norm_to_col:
            return norm_to_col[_norm_col_name(cand)]
    for col in columns:
        norm = _norm_col_name(col)
        if any(_norm_col_name(cand) in norm for cand in candidates):
            return str(col)
    return None


def _demographic_id_keys(value: Any) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    text = os.path.basename(text).replace(".pkl", "").replace(".edf", "")
    upper = text.upper()
    keys: List[str] = []

    def add_key(key: str) -> None:
        if key and key not in keys:
            keys.append(key)

    compact = re.sub(r"[^A-Z0-9]+", "", upper)
    add_key(compact)
    if compact.startswith("0"):
        add_key(compact.lstrip("0"))

    bids = re.search(r"SUB[-_]?I\d+", upper)
    if bids:
        add_key(re.sub(r"[^A-Z0-9]+", "", bids.group(0)))
    site_patient = re.search(r"I\d{6,}", upper)
    if site_patient:
        add_key(site_patient.group(0))

    digit_groups = re.findall(r"\d+", upper)
    if digit_groups:
        joined = "".join(digit_groups)
        if len(joined) >= 3:
            add_key(joined)
            add_key(joined.lstrip("0") or "0")
    if len(digit_groups) == 1 and re.search(r"[A-Z]", compact) and len(digit_groups[0]) >= 3:
        add_key(digit_groups[0].lstrip("0") or "0")
    return keys


def _demographic_file_score(group: str, path: str) -> int:
    name = os.path.basename(path).lower()
    group_l = group.lower()
    score = 0
    hints = {
        "young_the_human_sleep_project": ["psg_metadata"],
        "the_human_sleep_project": ["psg_metadata"],
        "edf": ["cobrad_clinical", "clinical"],
        "berkeley_data": ["00_df_all_demographics_young", "demographics"],
    }
    for hint in hints.get(group_l, ["demographic", "clinical", "metadata", "age", "gender", "sex"]):
        if hint in name:
            score += 10
    if name.endswith(".csv"):
        score += 1
    return score


def find_demographic_files_for_group(group: str) -> List[str]:
    group_dir = os.path.join(_script_dir(), EDF_FORMAT_DIRNAME, group)
    if not os.path.isdir(group_dir):
        return []
    files = []
    search_dirs = [group_dir]
    try:
        search_dirs.extend(
            os.path.join(group_dir, name)
            for name in os.listdir(group_dir)
            if os.path.isdir(os.path.join(group_dir, name))
        )
    except OSError:
        pass
    for root in search_dirs:
        try:
            names = os.listdir(root)
        except OSError:
            continue
        for name in names:
            if name.lower().endswith((".csv", ".xlsx", ".xls")) and not name.startswith("~$"):
                files.append(os.path.join(root, name))
    return sorted(files, key=lambda p: (-_demographic_file_score(group, p), p))


def _read_demographic_source(path: str) -> Tuple[pd.DataFrame, str]:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path), os.path.basename(path)

    excel = pd.ExcelFile(path)
    preferred = next((sheet for sheet in excel.sheet_names if _norm_col_name(sheet) == "clinical"), None)
    sheet = preferred or excel.sheet_names[0]
    return pd.read_excel(path, sheet_name=sheet), f"{os.path.basename(path)}:{sheet}"


def load_group_demographics(group: str) -> Tuple[pd.DataFrame, str]:
    if group in _DEMOGRAPHIC_SOURCE_CACHE:
        return _DEMOGRAPHIC_SOURCE_CACHE[group]
    files = find_demographic_files_for_group(group)
    if not files:
        return pd.DataFrame(), ""

    best_df = pd.DataFrame()
    best_source = ""
    best_score = -1
    for path in files:
        try:
            df, source = _read_demographic_source(path)
        except Exception:
            continue
        cols = list(df.columns)
        score = _demographic_file_score(group, path)
        if _first_matching_col(cols, ["age", "ageatvisit"]):
            score += 5
        if _first_matching_col(cols, ["sex", "gender", "sexdsc"]):
            score += 5
        if _first_matching_col(cols, ["patient_id", "record_id", "subj", "subject", "bidsfolder", "bdsppatientid"]):
            score += 5
        if score > best_score:
            best_df, best_source, best_score = df, source, score
    _DEMOGRAPHIC_SOURCE_CACHE[group] = (best_df, best_source)
    return best_df, best_source


def build_demographic_lookup(df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], Dict[str, Optional[str]]]:
    if df.empty:
        return {}, {}
    id_cols = [
        col for col in df.columns
        if _norm_col_name(col) in {
            "patientid", "patient", "subject", "subjectid", "subj", "recordid",
            "bidsfolder", "bdsppatientid", "file", "filename",
        }
        or any(token in _norm_col_name(col) for token in ["patient", "subject", "subj", "recordid", "bidsfolder"])
    ]
    age_col = _first_matching_col(df.columns, ["age", "ageatvisit"])
    sex_col = _first_matching_col(df.columns, ["sex", "gender", "sexdsc"])
    bmi_col = _first_matching_col(df.columns, ["bmi"])
    education_col = _first_matching_col(df.columns, ["education", "education_by_diploma"])
    handedness_col = _first_matching_col(df.columns, ["handedness"])

    lookup: Dict[str, pd.Series] = {}
    for _, row in df.iterrows():
        for col in id_cols:
            for key in _demographic_id_keys(row.get(col)):
                lookup.setdefault(key, row)
    cols = {
        "age": age_col,
        "sex": sex_col,
        "bmi": bmi_col,
        "education": education_col,
        "handedness": handedness_col,
    }
    return lookup, cols


def _standardize_sex(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    lower = text.lower()
    if lower in {"1", "1.0", "m", "male"}:
        return "Male"
    if lower in {"2", "2.0", "f", "female"}:
        return "Female"
    return text


def demographic_table_for_patients(patient_summary: pd.DataFrame, groups: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if patient_summary.empty:
        return pd.DataFrame(), pd.DataFrame()

    source_rows = []
    group_payload = {}
    for group in groups:
        df, source = load_group_demographics(group)
        lookup, cols = build_demographic_lookup(df)
        group_payload[group] = (lookup, cols, source)
        source_rows.append({
            "group": group,
            "source": source or "No .csv/.xlsx demographic file found",
            "source_rows": len(df),
            "matched_columns": ", ".join(f"{k}={v}" for k, v in cols.items() if v),
        })

    rows = []
    for _, patient in patient_summary[["group", "patient_id"]].drop_duplicates().iterrows():
        group = patient["group"]
        patient_id = patient["patient_id"]
        lookup, cols, source = group_payload.get(group, ({}, {}, ""))
        match = None
        matched_key = ""
        for key in _demographic_id_keys(patient_id):
            if key in lookup:
                match = lookup[key]
                matched_key = key
                break

        row = {
            "group": group,
            "patient_id": patient_id,
            "demographics_matched": match is not None,
            "matched_key": matched_key,
            "demographics_source": source,
        }
        if match is not None:
            age_col = cols.get("age")
            sex_col = cols.get("sex")
            bmi_col = cols.get("bmi")
            education_col = cols.get("education")
            handedness_col = cols.get("handedness")
            row["age"] = pd.to_numeric(match.get(age_col), errors="coerce") if age_col else np.nan
            row["sex"] = _standardize_sex(match.get(sex_col)) if sex_col else ""
            row["bmi"] = pd.to_numeric(match.get(bmi_col), errors="coerce") if bmi_col else np.nan
            row["education"] = match.get(education_col) if education_col else np.nan
            row["handedness"] = match.get(handedness_col) if handedness_col else ""
        else:
            row.update({"age": np.nan, "sex": "", "bmi": np.nan, "education": np.nan, "handedness": ""})
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(source_rows)


def demographic_summary_tables(demo_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if demo_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    age_rows = []
    for group, sub in demo_df.groupby("group"):
        ages = pd.to_numeric(sub["age"], errors="coerce").dropna()
        age_rows.append({
            "group": group,
            "n_patients": int(sub["patient_id"].nunique()),
            "n_with_demographics": int(sub["demographics_matched"].sum()),
            "n_age": int(len(ages)),
            "age_mean": float(ages.mean()) if len(ages) else np.nan,
            "age_sd": float(ages.std(ddof=1)) if len(ages) > 1 else np.nan,
            "age_median": float(ages.median()) if len(ages) else np.nan,
            "age_iqr": float(ages.quantile(0.75) - ages.quantile(0.25)) if len(ages) else np.nan,
            "age_min": float(ages.min()) if len(ages) else np.nan,
            "age_max": float(ages.max()) if len(ages) else np.nan,
        })
    sex = demo_df.copy()
    sex["sex"] = sex["sex"].replace("", np.nan).fillna("Unknown")
    sex_counts = (
        sex.groupby(["group", "sex"], as_index=False)
        .agg(n=("patient_id", "nunique"))
        .sort_values(["group", "sex"])
    )
    totals = sex.groupby("group")["patient_id"].nunique().to_dict()
    sex_counts["percent"] = sex_counts.apply(lambda r: 100.0 * r["n"] / totals.get(r["group"], np.nan), axis=1)
    return pd.DataFrame(age_rows).sort_values("group"), sex_counts


def mpl_demographic_overview(demo_df: pd.DataFrame, groups: Sequence[str],
                             stage: Optional[str] = None) -> Optional[Figure]:
    if demo_df.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), constrained_layout=True)

    age_data = [
        pd.to_numeric(demo_df.loc[demo_df["group"] == group, "age"], errors="coerce").dropna().to_numpy()
        for group in groups
    ]
    positions = np.arange(1, len(groups) + 1)
    valid_age = [(pos, group, vals) for pos, group, vals in zip(positions, groups, age_data) if len(vals)]
    if valid_age:
        bp = axes[0].boxplot(
            [vals for _, _, vals in valid_age],
            positions=[pos for pos, _, _ in valid_age],
            widths=0.58,
            patch_artist=True,
            showfliers=False,
        )
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(PALETTE[i % len(PALETTE)])
            patch.set_alpha(0.35)
        rng = np.random.default_rng(101)
        for i, (pos, _, vals) in enumerate(valid_age):
            axes[0].scatter(
                np.full(len(vals), pos) + rng.normal(0, 0.035, len(vals)),
                vals,
                color=PALETTE[i % len(PALETTE)],
                s=18,
                alpha=0.75,
                edgecolors="white",
                linewidths=0.25,
            )
    else:
        axes[0].text(0.5, 0.5, "No matched ages", ha="center", va="center", transform=axes[0].transAxes)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(groups, rotation=25, ha="right")
    axes[0].set_ylabel("Age")
    set_centered_title(axes[0], "Age Distribution", stage)
    scientific_axes(axes[0])

    sex_df = demo_df.copy()
    sex_df["sex"] = sex_df["sex"].replace("", np.nan).fillna("Unknown")
    sex_pivot = sex_df.pivot_table(index="group", columns="sex", values="patient_id", aggfunc="nunique", fill_value=0)
    sex_pivot = sex_pivot.reindex(groups).fillna(0)
    bottoms = np.zeros(len(sex_pivot))
    for idx, sex in enumerate(sex_pivot.columns):
        vals = sex_pivot[sex].to_numpy(dtype=float)
        axes[1].bar(
            np.arange(len(sex_pivot)),
            vals,
            bottom=bottoms,
            label=sex,
            color=PALETTE[idx % len(PALETTE)],
            edgecolor="black",
            linewidth=0.35,
        )
        bottoms += vals
    axes[1].set_xticks(np.arange(len(sex_pivot)))
    axes[1].set_xticklabels(sex_pivot.index, rotation=25, ha="right")
    axes[1].set_ylabel("Patients")
    set_centered_title(axes[1], "Sex Counts", stage)
    axes[1].legend(frameon=False)
    scientific_axes(axes[1])

    set_centered_suptitle(fig, "Demographic Overview", stage)
    return fig


def channel_group_n_text(feature_df: pd.DataFrame, groups: Sequence[str], channel: str) -> str:
    parts = []
    for group in groups:
        n = feature_df[(feature_df["group"] == group) & (feature_df["channel"] == channel)]["patient_id"].nunique()
        parts.append(f"{group}: N={n}")
    return "; ".join(parts)


def channel_group_n_dict(feature_df: pd.DataFrame, groups: Sequence[str], channel: str) -> Dict[str, int]:
    return {
        group: int(feature_df[(feature_df["group"] == group) & (feature_df["channel"] == channel)]["patient_id"].nunique())
        for group in groups
    }


def metric_label(metric: str) -> str:
    labels = {
        "distance_ms": "Distance (ms)",
        "signed_distance_ms": "Signed Distance (ms)",
        "eeg_t_peak_amplitude_uv": "T-Peak Amplitude (uV)",
        "ecg_eeg_distance_corr_twave": "Raw Correlation",
        "ecg_eeg_distance_corr_epoch": "Raw Correlation (Epoch)",
        "ecg_eeg_firstdiff_corr_twave": "First-Difference Correlation",
        "ecg_eeg_firstdiff_corr_epoch": "First-Difference Correlation (Epoch)",
        "peak_time_ms": "Peak Time (ms)",
        "peak_z": "Peak Mismatch (z)",
        "top_decile_z": "Top-Decile Mismatch (z)",
        "mean_percent": "Mean Mismatch (%)",
        "peak_percent": "Peak Mismatch (%)",
    }
    return labels.get(metric, metric.replace("_", " ").replace("ms", "(ms)").replace("uv", "(uV)"))


def stage_label(stage: Optional[str]) -> str:
    if not stage:
        return ""
    return str(stage).replace("_", " ").title()


def title_text(title: str, stage: Optional[str] = None, subtitle: Optional[str] = None) -> str:
    parts = [title]
    if stage:
        parts.append(f"Sleep Stage: {stage_label(stage)}")
    if subtitle:
        parts.append(subtitle)
    return "\n".join(parts)


def set_centered_title(ax, title: str, stage: Optional[str] = None,
                       subtitle: Optional[str] = None, **kwargs) -> None:
    defaults = {"fontweight": "bold", "pad": 10, "loc": "center"}
    defaults.update(kwargs)
    ax.set_title(title_text(title, stage, subtitle), multialignment="center", **defaults)


def set_centered_suptitle(fig: Figure, title: str, stage: Optional[str] = None,
                          subtitle: Optional[str] = None, **kwargs) -> None:
    defaults = {"x": 0.5, "ha": "center", "fontsize": 11, "fontweight": "bold"}
    defaults.update(kwargs)
    fig.suptitle(title_text(title, stage, subtitle), multialignment="center", **defaults)


def _canonical_topomap_channel(channel: str, montage_lookup: Dict[str, str]) -> Optional[str]:
    """Return the MNE standard_1020 channel name for a dashboard channel label."""
    cleaned = re.sub(r"(?i)^eeg\s+", "", str(channel).strip())
    cleaned = re.split(r"[-_/ ](?:ref|avg|a1|a2|m1|m2)\b", cleaned, flags=re.IGNORECASE)[0]
    compact = re.sub(r"[^A-Za-z0-9]", "", cleaned).upper()
    compact = _TOPO_ALIASES.get(compact, compact).upper()
    return montage_lookup.get(compact)


def _metric_topomap_limits(values: Sequence[float], metric: str, difference: bool = False) -> Optional[Tuple[float, float]]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    if difference or metric in {"signed_distance_ms", "eeg_t_peak_amplitude_uv"}:
        vmax = float(np.nanmax(np.abs(vals))) or 1.0
        return -vmax, vmax
    if "firstdiff_corr" in metric:
        return -1.0, 1.0
    if "distance_corr" in metric:
        return 0.0, 1.0
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if np.isclose(lo, hi):
        pad = abs(lo) * 0.1 + 1.0
        return lo - pad, hi + pad
    return lo, hi


def _render_topomap(ch_values: Dict[str, float], ax, title: str, cmap: str = "RdBu_r",
                    vlim: Optional[Tuple[float, float]] = None, colorbar_label: str = "",
                    pad_missing: bool = False, missing_fill: float = 0.0,
                    show_electrode_names: bool = False):
    """
    Plot a topomap from a {channel_name: scalar} dict using the standard 10-20 montage.

    Legacy temporal labels T3/T4/T5/T6 are mapped to their standard_1020 aliases
    when needed. If ``pad_missing`` is enabled, missing standard-19 electrodes and
    T3/T4/T5/T6 aliases are padded with ``missing_fill`` so p-value maps keep the
    same scalp geometry across analyses.
    """
    mont = mne.channels.make_standard_montage("standard_1020")
    montage_lookup = {c.upper(): c for c in mont.ch_names}

    values_by_montage: Dict[str, List[float]] = {}
    for ch, val in ch_values.items():
        if not np.isfinite(val):
            continue
        topo_ch = _canonical_topomap_channel(ch, montage_lookup)
        if topo_ch is None:
            continue
        values_by_montage.setdefault(topo_ch, []).append(float(val))

    if pad_missing:
        for base_ch in _TOPO_STD19:
            topo_ch = _canonical_topomap_channel(base_ch, montage_lookup)
            if topo_ch is not None:
                values_by_montage.setdefault(topo_ch, [float(missing_fill)])
        for old_ch, alias_ch in _TOPO_ALIASES.items():
            for candidate in (old_ch, alias_ch):
                topo_ch = _canonical_topomap_channel(candidate, montage_lookup)
                if topo_ch is not None:
                    values_by_montage.setdefault(topo_ch, [float(missing_fill)])

    if len(values_by_montage) < 3:
        ax.text(0.5, 0.5, "Too few channels\nfor topomap", ha="center", va="center", transform=ax.transAxes)
        set_centered_title(ax, title, fontweight="normal")
        ax.set_axis_off()
        return None

    plot_channels = []
    plot_values = []
    for ch in _TOPO_STD19:
        topo_ch = _canonical_topomap_channel(ch, montage_lookup)
        if topo_ch in values_by_montage and topo_ch not in plot_channels:
            plot_channels.append(topo_ch)
            plot_values.append(float(np.nanmedian(values_by_montage[topo_ch])))
    if pad_missing:
        for old_ch, alias_ch in _TOPO_ALIASES.items():
            for candidate in (old_ch, alias_ch):
                topo_ch = _canonical_topomap_channel(candidate, montage_lookup)
                if topo_ch in values_by_montage and topo_ch not in plot_channels:
                    plot_channels.append(topo_ch)
                    plot_values.append(float(np.nanmedian(values_by_montage[topo_ch])))
    if not pad_missing:
        for topo_ch, vals in values_by_montage.items():
            if topo_ch not in plot_channels:
                plot_channels.append(topo_ch)
                plot_values.append(float(np.nanmedian(vals)))

    info2 = mne.create_info(ch_names=plot_channels, sfreq=250.0, ch_types="eeg")
    info2.set_montage(mont, on_missing="ignore")
    valid = np.array([
        not np.any(np.isnan(ch["loc"][:3])) and np.any(ch["loc"][:3] != 0)
        for ch in info2["chs"]
    ])
    if not np.any(valid):
        ax.text(0.5, 0.5, "No valid montage\nlocations", ha="center", va="center", transform=ax.transAxes)
        set_centered_title(ax, title, fontweight="normal")
        ax.set_axis_off()
        return None

    data = np.asarray(plot_values, dtype=float)[valid]
    names = np.asarray(plot_channels, dtype=object)[valid].tolist() if show_electrode_names else None
    info2 = mne.pick_info(info2, np.where(valid)[0])
    if vlim is None:
        vmax = float(np.nanmax(np.abs(data))) or 1.0
        vlim = (-vmax, vmax)

    try:
        res = mne.viz.plot_topomap(
            data, info2, axes=ax, cmap=cmap, vlim=vlim, extrapolate="head",
            names=names, show=False
        )
    except TypeError:
        res = mne.viz.plot_topomap(
            data, info2, axes=ax, cmap=cmap, vmin=vlim[0], vmax=vlim[1],
            extrapolate="head", names=names, show=False
        )
    im = res[0] if isinstance(res, tuple) else res
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if colorbar_label:
        cb.set_label(colorbar_label)
    set_centered_title(ax, title)
    return im


def _render_pval_topomap(p_vals: Dict[str, float], ax_t, title: str):
    """Plot a p-value topomap into ax_t. Returns the AxesImage or None."""
    _mont = mne.channels.make_standard_montage("standard_1020")
    _mont_upper = [c.upper() for c in _mont.ch_names]
    _pch, _pd = [], []
    _real_channels = set()
    for _c, _p in p_vals.items():
        _cu = str(_c).upper()
        if _cu in _mont_upper and np.isfinite(_p):
            _mont_name = _mont.ch_names[_mont_upper.index(_cu)]
            _pch.append(_mont_name)
            _pd.append(float(_p))
            _real_channels.add(_mont_name.upper())
    _std19 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
              "C3", "Cz", "C4", "P3", "Pz", "P4", "O1", "O2"]
    _aliases = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
    for _bc in _std19:
        if not any(_bc.upper() == _x.upper() for _x in _pch):
            _pch.append(_bc)
            _pd.append(0.05)
    for _nn, _on in _aliases.items():
        if not any(_c2.upper() in [_nn.upper(), _on.upper()] for _c2 in _pch):
            _pch.append(_nn)
            _pd.append(0.05)
    if not _pch:
        return None
    _info2 = mne.create_info(ch_names=_pch, sfreq=250.0, ch_types="eeg")
    _info2.set_montage(_mont, on_missing="ignore")
    _valid2 = np.array([
        not np.any(np.isnan(_ch["loc"][:3])) and np.any(_ch["loc"][:3] != 0)
        for _ch in _info2["chs"]
    ])
    if not np.any(_valid2):
        return None
    _pch_v = [_pch[i] if _pch[i].upper() in _real_channels else "" for i in np.where(_valid2)[0]]
    _p_clipped = np.clip(np.array(_pd, dtype=float)[_valid2], 1e-4, 0.05)
    _da = -np.log10(_p_clipped)
    _vmin = -np.log10(0.05)
    _vmax = -np.log10(1e-4)
    _info2 = mne.pick_info(_info2, np.where(_valid2)[0])
    try:
        _res = mne.viz.plot_topomap(
            _da, _info2, axes=ax_t,
            cmap="Reds", names=_pch_v, vlim=(_vmin, _vmax), extrapolate="head", show=False
        )
    except TypeError:
        _res = mne.viz.plot_topomap(
            _da, _info2, axes=ax_t,
            cmap="Reds", names=_pch_v, vmin=_vmin, vmax=_vmax, extrapolate="head", show=False
        )
    _im = _res[0] if isinstance(_res, tuple) else _res
    _cb = ax_t.figure.colorbar(_im, ax=ax_t)
    _cb.set_ticks([_vmin, -np.log10(0.01), -np.log10(0.001), _vmax])
    _cb.set_ticklabels(["0.05", "0.01", "0.001", "0"])
    _cb.ax.invert_yaxis()
    _cb.set_label("P value", labelpad=8)
    set_centered_title(ax_t, title)
    return _im


def topomap_channel_summary(feature_df: pd.DataFrame, groups: Sequence[str], metric: str) -> pd.DataFrame:
    rows = []
    montage_lookup = {c.upper(): c for c in mne.channels.make_standard_montage("standard_1020").ch_names}
    channels = sorted(feature_df["channel"].dropna().unique())
    for channel in channels:
        topo_channel = _canonical_topomap_channel(channel, montage_lookup)
        row = {
            "channel": channel,
            "topomap_channel": topo_channel or "",
            "included_in_topomap": topo_channel is not None,
        }
        medians = {}
        for group in groups:
            sub = feature_df[(feature_df["group"] == group) & (feature_df["channel"] == channel)]
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
            med = float(vals.median()) if not vals.empty else np.nan
            medians[group] = med
            row[f"{group}_n_patients"] = int(sub["patient_id"].nunique())
            row[f"{group}_median_{metric}"] = med
        if len(groups) == 2:
            row[f"delta_{groups[0]}_minus_{groups[1]}"] = medians.get(groups[0], np.nan) - medians.get(groups[1], np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def mpl_group_topomaps(feature_df: pd.DataFrame, groups: Sequence[str], metric: str,
                       stage: Optional[str] = None) -> Tuple[Optional[Figure], pd.DataFrame]:
    if feature_df.empty or metric not in feature_df.columns or len(groups) < 1:
        return None, pd.DataFrame()

    summary = topomap_channel_summary(feature_df, groups, metric)
    if summary.empty:
        return None, summary

    metric_cols = [f"{g}_median_{metric}" for g in groups]
    group_values = {
        group: summary.set_index("channel")[f"{group}_median_{metric}"].dropna().to_dict()
        for group in groups
    }

    group_vlim = _metric_topomap_limits(summary[metric_cols].to_numpy().ravel(), metric, difference=False)
    group_cmap = "RdBu_r" if (metric.startswith("signed_") or "corr" in metric or "amplitude" in metric) else "viridis"
    label = metric_label(metric)

    n_maps = len(groups) + (1 if len(groups) == 2 else 0)
    fig, axes = plt.subplots(1, n_maps, figsize=(max(5.2, 3.4 * n_maps), 3.7), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, group in zip(axes[:len(groups)], groups):
        n_patients = int(feature_df[feature_df["group"] == group]["patient_id"].nunique())
        _render_topomap(
            group_values[group],
            ax,
            f"{group}\nmedian {label} (N={n_patients})",
            cmap=group_cmap,
            vlim=group_vlim,
            colorbar_label=label,
        )
    if len(groups) == 2:
        g1, g2 = groups
        diff_col = f"delta_{g1}_minus_{g2}"
        diff_values = summary.dropna(subset=[diff_col]).set_index("channel")[diff_col].to_dict()
        diff_vlim = _metric_topomap_limits(summary[diff_col].to_numpy(), metric, difference=True)
        _render_topomap(
            diff_values,
            axes[-1],
            f"Difference\n{g1} - {g2}",
            cmap="RdBu_r",
            vlim=diff_vlim,
            colorbar_label=f"Delta {label}",
        )
    set_centered_suptitle(fig, "Scalp Topography of Channel-Level Group Medians", stage)
    return fig, summary


def correlation_pvalue_topomap_table(raw_stats: pd.DataFrame, fd_stats: pd.DataFrame,
                                     p_col: str = "p_value") -> pd.DataFrame:
    raw_cols = ["channel"]
    fd_cols = ["channel"]
    for col in [p_col, "p_value", "p_fdr_bh", "p_bonferroni", "rank_biserial_r", "median_delta"]:
        if col in raw_stats.columns and col not in raw_cols:
            raw_cols.append(col)
        if col in fd_stats.columns and col not in fd_cols:
            fd_cols.append(col)

    raw = raw_stats[raw_cols].copy() if not raw_stats.empty else pd.DataFrame(columns=raw_cols)
    fd = fd_stats[fd_cols].copy() if not fd_stats.empty else pd.DataFrame(columns=fd_cols)
    raw = raw.rename(columns={c: f"raw_{c}" for c in raw.columns if c != "channel"})
    fd = fd.rename(columns={c: f"firstdiff_{c}" for c in fd.columns if c != "channel"})
    return raw.merge(fd, on="channel", how="outer").sort_values("channel")


def mpl_correlation_pvalue_topomaps(raw_stats: pd.DataFrame, fd_stats: pd.DataFrame,
                                    raw_metric: str, fd_metric: str,
                                    p_col: str = "p_value",
                                    stage: Optional[str] = None) -> Tuple[Optional[Figure], Optional[Figure], pd.DataFrame]:
    if raw_stats.empty and fd_stats.empty:
        return None, None, pd.DataFrame()

    table = correlation_pvalue_topomap_table(raw_stats, fd_stats, p_col=p_col)
    raw_col = f"raw_{p_col}"
    fd_col = f"firstdiff_{p_col}"
    values = []
    for col in [raw_col, fd_col]:
        if col in table:
            values.extend(pd.to_numeric(table[col], errors="coerce").dropna().tolist())
    if not values:
        return None, None, table

    raw_values = table.dropna(subset=[raw_col]).set_index("channel")[raw_col].to_dict() if raw_col in table else {}
    fd_values = table.dropna(subset=[fd_col]).set_index("channel")[fd_col].to_dict() if fd_col in table else {}
    raw_fig, raw_ax = plt.subplots(figsize=(5.2, 4.8), dpi=220, constrained_layout=True)
    fd_fig, fd_ax = plt.subplots(figsize=(5.2, 4.8), dpi=220, constrained_layout=True)
    _render_pval_topomap(
        raw_values,
        raw_ax,
        "Raw Correlation P Value",
    )
    _render_pval_topomap(
        fd_values,
        fd_ax,
        "First-Difference P Value",
    )
    set_centered_suptitle(raw_fig, "Group Difference P-Value Topomap", stage)
    set_centered_suptitle(fd_fig, "Group Difference P-Value Topomap", stage)
    return raw_fig, fd_fig, table


def mpl_correlation_delta_topomaps(raw_stats: pd.DataFrame, fd_stats: pd.DataFrame,
                                   groups: Sequence[str], raw_metric: str,
                                   fd_metric: str,
                                   stage: Optional[str] = None) -> Tuple[Optional[Figure], pd.DataFrame]:
    if raw_stats.empty and fd_stats.empty:
        return None, pd.DataFrame()

    table = correlation_pvalue_topomap_table(raw_stats, fd_stats, p_col="p_value")
    raw_col = "raw_median_delta"
    fd_col = "firstdiff_median_delta"
    values = []
    for col in [raw_col, fd_col]:
        if col in table:
            values.extend(pd.to_numeric(table[col], errors="coerce").dropna().tolist())
    if not values:
        return None, table

    vmax = float(np.nanmax(np.abs(values))) or 1.0
    vlim = (-vmax, vmax)
    group_text = f"{groups[0]} - {groups[1]}" if len(groups) == 2 else "group difference"
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.7), constrained_layout=True)
    raw_values = table.dropna(subset=[raw_col]).set_index("channel")[raw_col].to_dict() if raw_col in table else {}
    fd_values = table.dropna(subset=[fd_col]).set_index("channel")[fd_col].to_dict() if fd_col in table else {}
    _render_topomap(
        raw_values,
        axes[0],
        f"Raw correlation\nmedian delta",
        cmap="RdBu_r",
        vlim=vlim,
        colorbar_label=f"Delta {metric_label(raw_metric)}",
    )
    _render_topomap(
        fd_values,
        axes[1],
        f"First differences\nmedian delta",
        cmap="RdBu_r",
        vlim=vlim,
        colorbar_label=f"Delta {metric_label(fd_metric)}",
    )
    set_centered_suptitle(fig, "Correlation-Value Group Difference Topomaps", stage, group_text)
    return fig, table


# ── Cache + batch processing ──────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_analysis_cached(
    base_path: str, groups: Tuple[str, ...], stage: str, test_run_limit: int,
    window: Tuple[float, float], t_window: Tuple[float, float],
    swing_threshold: float, max_bad_epoch_channel_fraction: float,
    eeg_t_radius: float = DEFAULT_EEG_T_RADIUS, eeg_t_post_s: float = 0.03,
    r_peak_flip_window_s: float = 0.10, min_kept_epochs: int = 5,
    flip_z_threshold: float = 3.0, flip_prominence_threshold: float = 25.0,
    flip_min_votes: int = 2, flip_baseline_pre_window: float = 0.50,
    r_peak_curvature_threshold: float = 0.0,
    ecg_filter_low_hz: float = 0.5, ecg_filter_high_hz: float = 40.0,
    ecg_median_ms: float = 20.0, ecg_clip_sd: float = 6.0,
    qrs_filter_low_hz: float = 5.0, qrs_filter_high_hz: float = 25.0,
    rpeak_mad_multiplier: float = 4.0, rpeak_refractory_s: float = 0.4,
    rr_min_s: float = 0.4, rr_max_s: float = 2.2,
    ecg_flip_qrs_half_width_ms: float = 80.0, ecg_flip_min_beats: int = 5,
    ecg_flip_ratio: float = 0.9, artifact_mad_multiplier: float = 7.5,
    ecg_low_ptp_percentile: float = 5.0, cache_token: int = 0,
) -> Tuple[List[PatientResult], pd.DataFrame]:
    del cache_token
    results: List[PatientResult] = []
    rows: List[Dict] = []
    for group in groups:
        limit = test_run_limit if test_run_limit > 0 else None
        for file_path in list_patient_files(base_path, group, stage, limit=limit):
            try:
                result = process_patient_file(
                    file_path=file_path, group=group, stage=stage,
                    window=window, t_window=t_window,
                    swing_threshold=swing_threshold,
                    max_bad_epoch_channel_fraction=max_bad_epoch_channel_fraction,
                    eeg_t_radius=eeg_t_radius, eeg_t_post_s=eeg_t_post_s,
                    r_peak_flip_window_s=r_peak_flip_window_s,
                    min_kept_epochs=min_kept_epochs,
                    flip_z_threshold=flip_z_threshold,
                    flip_prominence_threshold=flip_prominence_threshold,
                    flip_min_votes=flip_min_votes,
                    flip_baseline_pre_window=flip_baseline_pre_window,
                    r_peak_curvature_threshold=r_peak_curvature_threshold,
                    ecg_filter_low_hz=ecg_filter_low_hz,
                    ecg_filter_high_hz=ecg_filter_high_hz,
                    ecg_median_ms=ecg_median_ms, ecg_clip_sd=ecg_clip_sd,
                    qrs_filter_low_hz=qrs_filter_low_hz,
                    qrs_filter_high_hz=qrs_filter_high_hz,
                    rpeak_mad_multiplier=rpeak_mad_multiplier,
                    rpeak_refractory_s=rpeak_refractory_s,
                    rr_min_s=rr_min_s, rr_max_s=rr_max_s,
                    ecg_flip_qrs_half_width_ms=ecg_flip_qrs_half_width_ms,
                    ecg_flip_min_beats=ecg_flip_min_beats,
                    ecg_flip_ratio=ecg_flip_ratio,
                    artifact_mad_multiplier=artifact_mad_multiplier,
                    ecg_low_ptp_percentile=ecg_low_ptp_percentile,
                )
            except Exception as exc:
                st.toast(f"Skipped {os.path.basename(file_path)}: {exc}", icon="!")
                result = None
            if result is None:
                continue
            results.append(result)
            rows.extend(patient_to_feature_rows(result, t_window=t_window, eeg_radius=eeg_t_radius))
    return results, pd.DataFrame(rows)


def run_analysis_with_progress(**kwargs) -> Tuple[List[PatientResult], pd.DataFrame]:
    """Run the same analysis with Streamlit progress feedback."""
    base_path = kwargs["base_path"]
    groups = tuple(kwargs["groups"])
    stage = kwargs["stage"]
    test_run_limit = int(kwargs["test_run_limit"])
    limit = test_run_limit if test_run_limit > 0 else None

    work_items = []
    for group in groups:
        for file_path in list_patient_files(base_path, group, stage, limit=limit):
            work_items.append((group, file_path))

    total = len(work_items)
    if total == 0:
        return [], pd.DataFrame()

    analysis_label = "ICA-cleaned ECG-aligned HEP analysis" if kwargs.get("ica_ecg_clean", False) else "ECG-aligned HEP analysis"
    progress = st.progress(0.0, text=f"Preparing {analysis_label} for {total} files...")
    status = st.empty()
    counts = st.empty()

    results: List[PatientResult] = []
    rows: List[Dict] = []
    skipped = 0

    for idx, (group, file_path) in enumerate(work_items, start=1):
        patient_label = patient_id_from_path(file_path, stage)
        progress.progress((idx - 1) / total, text=f"Processing {idx}/{total}: {group} | {patient_label}")
        status.write(f"Current file: `{os.path.basename(file_path)}`")

        try:
            result = process_patient_file(
                file_path=file_path,
                group=group,
                stage=stage,
                window=kwargs["window"],
                t_window=kwargs["t_window"],
                swing_threshold=kwargs["swing_threshold"],
                max_bad_epoch_channel_fraction=kwargs["max_bad_epoch_channel_fraction"],
                eeg_t_radius=kwargs.get("eeg_t_radius", DEFAULT_EEG_T_RADIUS),
                eeg_t_post_s=kwargs.get("eeg_t_post_s", 0.03),
                r_peak_flip_window_s=kwargs.get("r_peak_flip_window_s", 0.10),
                min_kept_epochs=kwargs.get("min_kept_epochs", 5),
                flip_z_threshold=kwargs.get("flip_z_threshold", 3.0),
                flip_prominence_threshold=kwargs.get("flip_prominence_threshold", 25.0),
                flip_min_votes=kwargs.get("flip_min_votes", 2),
                flip_baseline_pre_window=kwargs.get("flip_baseline_pre_window", 0.50),
                r_peak_curvature_threshold=kwargs.get("r_peak_curvature_threshold", 0.0),
                ecg_filter_low_hz=kwargs.get("ecg_filter_low_hz", 0.5),
                ecg_filter_high_hz=kwargs.get("ecg_filter_high_hz", 40.0),
                ecg_median_ms=kwargs.get("ecg_median_ms", 20.0),
                ecg_clip_sd=kwargs.get("ecg_clip_sd", 6.0),
                qrs_filter_low_hz=kwargs.get("qrs_filter_low_hz", 5.0),
                qrs_filter_high_hz=kwargs.get("qrs_filter_high_hz", 25.0),
                rpeak_mad_multiplier=kwargs.get("rpeak_mad_multiplier", 4.0),
                rpeak_refractory_s=kwargs.get("rpeak_refractory_s", 0.4),
                rr_min_s=kwargs.get("rr_min_s", 0.4),
                rr_max_s=kwargs.get("rr_max_s", 2.2),
                ecg_flip_qrs_half_width_ms=kwargs.get("ecg_flip_qrs_half_width_ms", 80.0),
                ecg_flip_min_beats=kwargs.get("ecg_flip_min_beats", 5),
                ecg_flip_ratio=kwargs.get("ecg_flip_ratio", 0.9),
                artifact_mad_multiplier=kwargs.get("artifact_mad_multiplier", 7.5),
                ecg_low_ptp_percentile=kwargs.get("ecg_low_ptp_percentile", 5.0),
                ica_ecg_clean=kwargs.get("ica_ecg_clean", False),
                ica_components_to_remove=kwargs.get("ica_components_to_remove", ICA_COMPONENTS_TO_REMOVE),
                ica_max_components=kwargs.get("ica_max_components", ICA_MAX_COMPONENTS),
                ica_max_fit_samples=kwargs.get("ica_max_fit_samples", ICA_MAX_FIT_SAMPLES),
            )
        except Exception as exc:
            st.toast(f"Skipped {os.path.basename(file_path)}: {exc}", icon="!")
            result = None

        if result is None:
            skipped += 1
        else:
            results.append(result)
            rows.extend(patient_to_feature_rows(
                result,
                t_window=kwargs["t_window"],
                eeg_radius=kwargs.get("eeg_t_radius", DEFAULT_EEG_T_RADIUS),
            ))

        counts.caption(
            f"Completed {idx}/{total} files | valid patients: {len(results)} | skipped: {skipped}"
        )
        progress.progress(idx / total, text=f"Processed {idx}/{total}: {group} | {patient_label}")

    progress.progress(1.0, text=f"Done: {len(results)} valid patients from {total} files")
    status.empty()
    return results, pd.DataFrame(rows)


def run_analysis_with_processed_cache(**kwargs) -> Tuple[List[PatientResult], pd.DataFrame]:
    """Load processed analysis results from disk when available, otherwise compute and cache them."""
    mode = cache_mode_label(kwargs)
    mode_text = "ICA-cleaned" if mode == "ica" else "raw"
    cached = load_processed_analysis_cache(kwargs)
    if cached is not None:
        results, feature_df, cache_path = cached
        st.caption(
            f"Loaded cached {mode_text} processed data "
            f"({len(results)} patients, {len(feature_df)} feature rows) from `{os.path.basename(cache_path)}`."
        )
        return results, feature_df

    results, feature_df = run_analysis_with_progress(**kwargs)
    cache_path = save_processed_analysis_cache(kwargs, results, feature_df)
    if cache_path:
        st.caption(
            f"Cached {mode_text} processed data "
            f"({len(results)} patients, {len(feature_df)} feature rows) to `{os.path.basename(cache_path)}`."
        )
    else:
        st.warning(f"Could not write the {mode_text} processed-data cache.")
    return results, feature_df


# ── Download helpers ──────────────────────────────────────────────────────────

def _df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buf.getvalue()


def _fig_to_static_bytes(fig: Figure, fmt: str) -> Optional[bytes]:
    try:
        buf = io.BytesIO()
        save_kwargs = {"format": fmt, "bbox_inches": "tight"}
        if fmt in {"png", "jpg", "jpeg", "tiff"}:
            save_kwargs["dpi"] = 300
        fig.savefig(buf, **save_kwargs)
        return buf.getvalue()
    except Exception:
        return None


def download_block(df: Optional[pd.DataFrame] = None, fig: Optional[Figure] = None,
                   stem: str = "data", label: str = "Download",
                   use_expander: bool = True) -> None:
    """Render a compact row of download buttons for a table and/or figure."""
    download_container = st.expander(f"Download {label}", expanded=False) if use_expander else st.container()
    with download_container:
        if not use_expander:
            st.markdown(f"**Download {label}**")
        table_formats = []
        fig_formats = []
        if df is not None and not df.empty:
            table_formats = st.multiselect(
                "Table formats",
                ["CSV", "Excel", "JSON", "Parquet"],
                default=["CSV", "Excel"],
                key=f"fmt_tbl_{stem}",
            )
        if fig is not None:
            fig_formats = st.multiselect(
                "Figure formats",
                ["PNG", "SVG", "PDF", "JPEG", "TIFF"],
                default=["PNG", "PDF"],
                key=f"fmt_fig_{stem}",
            )

        col_defs = []
        if df is not None and not df.empty:
            if "CSV" in table_formats:
                col_defs.append(("CSV", df.to_csv(index=False).encode(), f"{stem}.csv", "text/csv"))
            if "Excel" in table_formats:
                col_defs.append(("Excel", _df_to_excel_bytes(df), f"{stem}.xlsx",
                                 "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))
            if "JSON" in table_formats:
                col_defs.append(("JSON", df.to_json(orient="records", indent=2).encode(),
                                 f"{stem}.json", "application/json"))
            if "Parquet" in table_formats:
                try:
                    buf = io.BytesIO()
                    df.to_parquet(buf, index=False)
                    col_defs.append(("Parquet", buf.getvalue(), f"{stem}.parquet", "application/octet-stream"))
                except Exception as exc:
                    st.warning(f"Parquet export is unavailable in this environment: {exc}")
        if fig is not None:
            for fmt, mime in [
                ("PNG", "image/png"),
                ("SVG", "image/svg+xml"),
                ("PDF", "application/pdf"),
                ("JPEG", "image/jpeg"),
                ("TIFF", "image/tiff"),
            ]:
                if fmt in fig_formats:
                    export_fmt = "jpg" if fmt == "JPEG" else fmt.lower()
                    data = _fig_to_static_bytes(fig, export_fmt)
                    if data is None:
                        st.warning(f"{fmt} export failed for this figure.")
                    else:
                        suffix = "jpg" if fmt == "JPEG" else fmt.lower()
                        col_defs.append((fmt, data, f"{stem}.{suffix}", mime))

        if not col_defs:
            st.info("Nothing to download.")
            return

        cols = st.columns(len(col_defs))
        for col, (btn_label, data, fname, mime) in zip(cols, col_defs):
            col.download_button(btn_label, data=data, file_name=fname, mime=mime,
                                key=f"dl_{stem}_{btn_label.split()[0].lower()}")


# ── Matplotlib visualisations ─────────────────────────────────────────────────

def scientific_axes(ax, x_major: Optional[float] = None, y_grid: bool = True) -> None:
    ax.grid(True, axis="y" if y_grid else "both", color="#d1d5db", linewidth=0.55, alpha=0.75)
    ax.tick_params(direction="out", length=3.5, width=0.8)
    if x_major is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_major))


def add_time_markers(ax, ecg_t_ms: Optional[float] = None) -> None:
    ax.axvline(0, color="#B22222", linestyle="--", linewidth=1.0, label="R peak")
    if ecg_t_ms is not None:
        ax.axvline(ecg_t_ms, color="#006400", linestyle=":", linewidth=1.2, label="ECG T peak")


def mpl_patient_traces(result: PatientResult, channels: Sequence[str],
                       height_per_row: int = 160) -> Figure:
    selected = [ch for ch in channels if ch in result.eeg_channels]
    n_rows = max(1, 1 + len(selected))
    fig_h = max(2.4, (height_per_row / 100.0) * n_rows)
    fig, axes = plt.subplots(n_rows, 1, figsize=(7.2, fig_h), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    t_ms = result.times * 1000

    axes[0].plot(t_ms, result.ecg_average, color="#222222", linewidth=1.4)
    add_time_markers(axes[0], result.ecg_t_peak_s * 1000)
    axes[0].set_ylabel("ECG (a.u.)")
    set_centered_title(axes[0], f"ECG ({result.ecg_channel})", fontweight="normal", fontsize=9)
    scientific_axes(axes[0], x_major=100)

    for ax, ch in zip(axes[1:], selected):
        idx = result.eeg_channels.index(ch)
        suffix = " (flipped)" if ch in result.flipped_eeg_channels else ""
        spr = result.spectral_power_ratios.get(ch, np.nan)
        spr_text = f" | HF/LF={spr:.3f}" if np.isfinite(spr) else " | HF/LF=NA"
        ax.plot(t_ms, result.eeg_average[idx] * 1e6, color=PALETTE[(idx + 1) % len(PALETTE)], linewidth=1.2)
        add_time_markers(ax, result.ecg_t_peak_s * 1000)
        ax.set_ylabel("EEG (uV)")
        set_centered_title(ax, f"{ch}{suffix}{spr_text}", fontweight="normal", fontsize=9)
        scientific_axes(ax, x_major=100)

    axes[-1].set_xlabel("Time from R peak (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncol=2)
    set_centered_suptitle(
        fig,
        "Heartbeat-Evoked Morphology",
        result.stage,
        f"{result.patient_id} | {result.group}",
    )
    return fig


def _group_channel_matrix(
    results: Sequence[PatientResult],
    group: str,
    channel: str,
    max_spectral_power_ratio: float = 0.4,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, int]:
    traces, times = [], None
    total_available = 0
    for r in results:
        if r.group != group or channel not in r.eeg_channels:
            continue
        total_available += 1
        ratio = r.spectral_power_ratios.get(channel, np.nan)
        if not np.isfinite(ratio) or ratio >= max_spectral_power_ratio:
            continue
        trace = r.eeg_average[r.eeg_channels.index(channel)]
        trace_ok, _ = assess_eeg_average_quality(trace)
        if not trace_ok:
            continue
        traces.append(trace * 1e6)
        times = r.times
    if not traces or times is None:
        return None, None, 0, total_available
    return np.vstack(traces), times, len(traces), total_available


def patient_channel_signal_ok(
    result: PatientResult,
    channel: str,
    max_spectral_power_ratio: float = MAX_SPECTRAL_POWER_RATIO,
) -> bool:
    if channel not in result.eeg_channels:
        return False
    ratio = result.spectral_power_ratios.get(channel, np.nan)
    if not np.isfinite(ratio) or ratio >= max_spectral_power_ratio:
        return False
    trace = result.eeg_average[result.eeg_channels.index(channel)]
    trace_ok, _ = assess_eeg_average_quality(trace)
    return trace_ok


def mpl_group_overlay(results: Sequence[PatientResult], groups: Sequence[str], channel: str,
                      show_individual: bool = True, feature_df: Optional[pd.DataFrame] = None,
                      height: int = 500, max_spectral_power_ratio: float = 0.4,
                      stage: Optional[str] = None) -> Optional[Figure]:
    fig, ax = plt.subplots(figsize=(7.2, max(3.0, height / 120.0)), constrained_layout=True)
    plotted = False
    for g_idx, group in enumerate(groups):
        mat, times, n_used, n_available = _group_channel_matrix(
            results, group, channel, max_spectral_power_ratio=max_spectral_power_ratio
        )
        if mat is None or times is None:
            continue
        color = PALETTE[g_idx % len(PALETTE)]
        t_ms = times * 1000
        if show_individual:
            for row in mat:
                ax.plot(t_ms, row, color=color, alpha=0.16, linewidth=0.55)
        mean = np.nanmean(mat, axis=0)
        sem = stats.sem(mat, axis=0, nan_policy="omit") if mat.shape[0] > 1 else np.zeros_like(mean)
        ch_dist = pd.Series(dtype=float)
        if feature_df is not None and not feature_df.empty:
            ch_dist = feature_df[
                (feature_df["group"] == group)
                & (feature_df["channel"] == channel)
            ]["distance_ms"].dropna()
        note = f", median distance={np.median(ch_dist):.1f} ms" if len(ch_dist) else ""
        ax.fill_between(t_ms, mean - sem, mean + sem, color=color, alpha=0.22, linewidth=0)
        ax.plot(
            t_ms,
            mean,
            color=color,
            linewidth=1.9,
            label=f"{group} (N={n_used}/{n_available}, HF/LF<{max_spectral_power_ratio:g}{note})",
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    add_time_markers(ax)
    set_centered_title(ax, "Group HEP Comparison", stage, channel)
    ax.set_xlabel("Time from R peak (ms)")
    ax.set_ylabel("EEG amplitude (uV)")
    ax.legend(frameon=False, loc="best")
    scientific_axes(ax, x_major=100)
    return fig


def mpl_distribution(feature_df: pd.DataFrame, channel: str, groups: Sequence[str],
                     metric: str = "distance_ms", plot_type: str = "Box",
                     stats_df: Optional[pd.DataFrame] = None,
                     height: int = 420, stage: Optional[str] = None) -> Optional[Figure]:
    df = feature_df[feature_df["channel"] == channel]
    if df.empty:
        return None
    label = metric_label(metric)
    data = [df.loc[df["group"] == g, metric].dropna().to_numpy() for g in groups]
    fig, ax = plt.subplots(figsize=(5.6, max(3.2, height / 130.0)), constrained_layout=True)
    positions = np.arange(1, len(groups) + 1)
    if plot_type == "Violin":
        vp = ax.violinplot(data, positions=positions, showmeans=False, showmedians=True, widths=0.72)
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(PALETTE[i % len(PALETTE)])
            body.set_edgecolor("black")
            body.set_alpha(0.35)
    else:
        bp = ax.boxplot(data, positions=positions, widths=0.58, patch_artist=True, showfliers=False)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(PALETTE[i % len(PALETTE)])
            patch.set_alpha(0.35)
            patch.set_linewidth(0.9)
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(1.3)
    rng = np.random.default_rng(42)
    for pos, vals, color in zip(positions, data, PALETTE):
        if len(vals):
            jitter = rng.normal(0, 0.035, len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals, s=16, color=color, alpha=0.72,
                       edgecolors="white", linewidths=0.25, zorder=3)
    p_text = ""
    if stats_df is not None and not stats_df.empty and channel in stats_df["channel"].values:
        ch_stats = stats_df.loc[stats_df["channel"] == channel]
        if len(ch_stats) == 1:
            row = ch_stats.iloc[0]
            p_text = (
                f"{row.get('comparison', 'group comparison')}: "
                f"P={format_p(row['p_value'])}; FDR={format_p(row['p_fdr_bh'])}; "
                f"r={row.get('rank_biserial_r', np.nan):.2f}"
            )
        else:
            p_text = "Pairwise p-values below"
    ax.set_xticks(positions)
    # df has multiple rows per patient (per-epoch/per-channel), so use unique patient_id count
    n_patients_by_group = df.groupby("group")["patient_id"].nunique()
    ax.set_xticklabels([f"{group}\n(N={n_patients_by_group.get(group, 0)})" for group in groups])
    ax.set_ylabel(label)
    ax.set_xlabel("Group")
    set_centered_title(ax, f"{channel}: {label}", stage, p_text or None)
    scientific_axes(ax)
    return fig


def mpl_patient_median_box(patient_df: pd.DataFrame, groups: Sequence[str],
                           pairwise_stats: Optional[pd.DataFrame], plot_type: str = "Box",
                           height: int = 440, stage: Optional[str] = None) -> Optional[Figure]:
    if patient_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(5.8, max(3.2, height / 130.0)), constrained_layout=True)
    data = [patient_df.loc[patient_df["group"] == g, "median_distance_ms"].dropna().to_numpy() for g in groups]
    positions = np.arange(1, len(groups) + 1)
    if plot_type == "Violin":
        vp = ax.violinplot(data, positions=positions, showmeans=False, showmedians=True, widths=0.72)
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(PALETTE[i % len(PALETTE)])
            body.set_edgecolor("black")
            body.set_alpha(0.35)
    else:
        bp = ax.boxplot(data, positions=positions, widths=0.58, patch_artist=True, showfliers=False)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(PALETTE[i % len(PALETTE)])
            patch.set_alpha(0.35)
    rng = np.random.default_rng(43)
    for pos, vals, color in zip(positions, data, PALETTE):
        ax.scatter(np.full(len(vals), pos) + rng.normal(0, 0.035, len(vals)), vals,
                   color=color, s=18, alpha=0.75, edgecolors="white", linewidths=0.25)
    p_text = ""
    if pairwise_stats is not None and not pairwise_stats.empty:
        if len(pairwise_stats) == 1:
            row = pairwise_stats.iloc[0]
            p_text = (
                f"{row.get('comparison', 'group comparison')}: "
                f"P={format_p(row['p_value'])}; r={row.get('rank_biserial_r', np.nan):.2f}; "
                f"d={row.get('cohens_d', np.nan):.2f}"
            )
        else:
            p_text = "Pairwise p-values below"
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{group}\n(N={len(vals)})" for group, vals in zip(groups, data)])
    ax.set_xlabel("Group")
    ax.set_ylabel("Median ECG-to-EEG T-peak distance (ms)")
    set_centered_title(ax, "Patient-Level Modulation", stage, p_text or None)
    scientific_axes(ax)
    return fig


def mpl_example_distance_traces(
    results: Sequence[PatientResult],
    feature_df: pd.DataFrame,
    groups: Sequence[str],
    channel: str,
    stage: Optional[str] = None,
) -> Optional[Figure]:
    rows = feature_df[feature_df["channel"] == channel].copy()
    if rows.empty:
        return None

    selected = []
    for group in groups:
        group_rows = rows[rows["group"] == group].sort_values("distance_ms")
        if group_rows.empty:
            continue
        example_row = group_rows.iloc[len(group_rows) // 2]
        patient_id = example_row["patient_id"]
        result = next((r for r in results if r.group == group and r.patient_id == patient_id), None)
        if result is not None and channel in result.eeg_channels:
            selected.append((group, result, example_row))

    if not selected:
        return None

    fig, axes = plt.subplots(
        len(selected), 1, figsize=(7.2, max(3.0, 2.8 * len(selected))),
        sharex=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    for ax, (group, result, row) in zip(axes, selected):
        idx = result.eeg_channels.index(channel)
        t_ms = result.times * 1000.0
        eeg_uv = result.eeg_average[idx] * 1e6
        ecg_t_ms = float(row["ecg_t_peak_ms"])
        eeg_t_ms = float(row["eeg_t_peak_ms"])
        distance_ms = float(row["distance_ms"])
        dcor_t = float(row.get("ecg_eeg_distance_corr_twave", np.nan))
        color = PALETTE[groups.index(group) % len(PALETTE)]

        ax.plot(t_ms, eeg_uv, color=color, linewidth=1.4, label=f"{group} EEG")
        ax.axvline(0, color="#B22222", linestyle="--", linewidth=0.9, label="R peak")
        ax.axvline(ecg_t_ms, color="#006400", linestyle=":", linewidth=1.3, label="ECG T peak")
        ax.axvline(eeg_t_ms, color="#4B0082", linestyle="-.", linewidth=1.2, label="EEG T peak")
        ax.scatter([ecg_t_ms], [np.interp(ecg_t_ms, t_ms, eeg_uv)], color="#006400", s=24, zorder=4)
        ax.scatter([eeg_t_ms], [np.interp(eeg_t_ms, t_ms, eeg_uv)], color="#4B0082", s=24, zorder=4)
        ax.set_ylabel("EEG (uV)")
        set_centered_title(
            ax,
            f"{group} Example",
            subtitle=f"{result.patient_id} | {channel} | distance={distance_ms:.1f} ms | nonlinear dCor={dcor_t:.3f}",
            fontweight="normal",
            fontsize=9,
        )
        scientific_axes(ax, x_major=100)

    axes[-1].set_xlabel("Time from R peak (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncol=2)
    set_centered_suptitle(fig, "Example T-Peak Distance Traces", stage)
    return fig


def mpl_pvalue_bar(stats_df: pd.DataFrame, max_channels: int = 40,
                   correction: str = "p_fdr_bh", height: Optional[int] = None,
                   stage: Optional[str] = None) -> Optional[Figure]:
    if stats_df.empty:
        return None
    df = stats_df.sort_values("p_value").head(max_channels).copy()
    y = np.arange(len(df))
    x = -np.log10(np.clip(df["p_value"].to_numpy(dtype=float), 1e-300, 1.0))
    sig = df[correction].to_numpy() < 0.05 if correction in df else df["p_value"].to_numpy() < 0.05
    colors = ["#B22222" if s else "#9CA3AF" for s in sig]
    fig, ax = plt.subplots(figsize=(6.8, max(3.5, 0.22 * len(df) + 1.4)), constrained_layout=True)
    ax.barh(y, x, color=colors, edgecolor="black", linewidth=0.35)
    ax.axvline(-np.log10(0.05), color="black", linestyle="--", linewidth=0.9, label="p=0.05")
    ax.set_yticks(y)
    labels = (
        df["channel"].astype(str) + "\n" + df["comparison"].astype(str)
        if "comparison" in df.columns else df["channel"].astype(str)
    )
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("-log10(P value)")
    ax.set_ylabel("EEG channel")
    set_centered_title(ax, "Channel-Level Significance", stage, correction)
    ax.legend(frameon=False)
    scientific_axes(ax, y_grid=False)
    return fig


def mpl_effect_forest(stats_df: pd.DataFrame, groups: Sequence[str],
                      max_channels: int = 30, height: Optional[int] = None,
                      stage: Optional[str] = None) -> Optional[Figure]:
    if stats_df.empty or "rank_biserial_r" not in stats_df.columns:
        return None
    df = stats_df.sort_values("p_value").head(max_channels).copy()
    se_vals = []
    for _, row in df.iterrows():
        g1 = row.get("group_a", groups[0] if groups else "")
        g2 = row.get("group_b", groups[1] if len(groups) > 1 else "")
        n1 = row.get("group_a_n", row.get(f"{g1}_n", np.nan))
        n2 = row.get("group_b_n", row.get(f"{g2}_n", np.nan))
        se_vals.append(np.sqrt((n1 + n2 + 1) / (3 * n1 * n2)) if np.isfinite(n1) and np.isfinite(n2) and n1 > 0 and n2 > 0 else np.nan)
    r = df["rank_biserial_r"].to_numpy()
    se = np.asarray(se_vals)
    lo = r - 1.96 * se
    hi = r + 1.96 * se
    y = np.arange(len(df))
    sig = df["p_fdr_bh"].to_numpy() < 0.05 if "p_fdr_bh" in df else np.zeros(len(df), dtype=bool)
    fig, ax = plt.subplots(figsize=(6.8, max(3.5, 0.24 * len(df) + 1.5)), constrained_layout=True)
    for yi, ri, l, h, s in zip(y, r, lo, hi, sig):
        color = "#B22222" if s else "#4B5563"
        ax.plot([l, h], [yi, yi], color=color, linewidth=1.2)
        ax.scatter([ri], [yi], color=color, s=24, marker="D", zorder=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    labels = (
        df["channel"].astype(str) + "\n" + df["comparison"].astype(str)
        if "comparison" in df.columns else df["channel"].astype(str)
    )
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Rank-biserial r (positive = higher in group A)")
    ax.set_ylabel("EEG channel")
    set_centered_title(ax, "Effect Sizes With Approximate 95% CI", stage)
    scientific_axes(ax, y_grid=False)
    return fig


def mpl_heatmap(feature_df: pd.DataFrame, groups: Sequence[str], metric: str = "distance_ms",
                colorscale: str = "viridis", height: Optional[int] = None,
                stage: Optional[str] = None) -> Optional[Figure]:
    if feature_df.empty:
        return None
    heat = feature_df[feature_df["group"].isin(groups)].pivot_table(
        index="channel", columns="group", values=metric, aggfunc="median"
    ).sort_index()
    if heat.empty:
        return None
    fig, ax = plt.subplots(figsize=(5.4, max(3.6, 0.22 * len(heat) + 1.4)), constrained_layout=True)
    im = ax.imshow(heat.to_numpy(), aspect="auto", cmap=colorscale)
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="white" if val > np.nanmedian(heat.to_numpy()) else "black", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    label = metric_label(metric)
    cbar.set_label(label)
    ax.set_xlabel("Group")
    ax.set_ylabel("EEG channel")
    set_centered_title(ax, f"Median {label}", stage, "By Channel and Group")
    return fig


def mpl_correlation_scatter(feature_df: pd.DataFrame, groups: Sequence[str],
                            channel: str, height: int = 450,
                            stage: Optional[str] = None) -> Optional[Figure]:
    df = feature_df[(feature_df["channel"] == channel) & (feature_df["group"].isin(groups))].dropna(
        subset=["ecg_t_peak_ms", "eeg_t_peak_ms"]
    )
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(5.8, max(3.4, height / 130.0)), constrained_layout=True)
    for g_idx, group in enumerate(groups):
        sub = df[df["group"] == group]
        # rows may repeat per patient (e.g. multiple windows), so count unique patients
        n_sub_patients = sub["patient_id"].nunique() if "patient_id" in sub.columns else len(sub)
        ax.scatter(sub["ecg_t_peak_ms"], sub["eeg_t_peak_ms"], s=24, alpha=0.75,
                   color=PALETTE[g_idx % len(PALETTE)], label=f"{group} (n={n_sub_patients})",
                   edgecolors="white", linewidths=0.25)
        if len(sub) >= 3:
            coef = np.polyfit(sub["ecg_t_peak_ms"], sub["eeg_t_peak_ms"], deg=1)
            xs = np.linspace(sub["ecg_t_peak_ms"].min(), sub["ecg_t_peak_ms"].max(), 100)
            ax.plot(xs, np.polyval(coef, xs), color=PALETTE[g_idx % len(PALETTE)], linewidth=1.2)
    ax.set_xlabel("ECG T-peak latency (ms)")
    ax.set_ylabel("EEG T-peak latency (ms)")
    set_centered_title(ax, "T-Peak Latency Relationship", stage, channel)
    ax.legend(frameon=False)
    scientific_axes(ax)
    return fig


def mpl_raw_corr_patient_example(result: PatientResult, channel: str) -> Optional[Figure]:
    if channel not in result.eeg_channels:
        return None
    idx = result.eeg_channels.index(channel)
    t_ms = result.times * 1000.0
    ecg = np.asarray(result.ecg_average, dtype=float)
    eeg = np.asarray(result.eeg_average[idx], dtype=float) * 1e6
    dcor_epoch = distance_correlation_1d(ecg, eeg)

    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    ax.plot(t_ms, ecg / (np.nanstd(ecg) + 1e-12), color="#222222", linewidth=1.2, label="ECG z")
    ax.plot(t_ms, (eeg - np.nanmean(eeg)) / (np.nanstd(eeg) + 1e-12), color=PALETTE[0], linewidth=1.2, label=f"{channel} EEG z")
    add_time_markers(ax)
    ax.set_xlabel("Time from R peak (ms)")
    ax.set_ylabel("Z-score")
    set_centered_title(
        ax,
        "Raw Waveform Nonlinear Dependence",
        result.stage,
        f"{result.group} | {result.patient_id} | {channel} | dCor={dcor_epoch:.3f}",
    )
    ax.legend(frameon=False)
    scientific_axes(ax, x_major=100)
    return fig


def mpl_firstdiff_patient_example(result: PatientResult, channel: str) -> Optional[Figure]:
    if channel not in result.eeg_channels:
        return None
    idx = result.eeg_channels.index(channel)
    t_ms = result.times * 1000.0
    ecg = np.asarray(result.ecg_average, dtype=float)
    eeg = np.asarray(result.eeg_average[idx], dtype=float) * 1e6
    dt_ms = t_ms[1:]
    decg = np.diff(ecg)
    deeg = np.diff(eeg)
    corr_epoch = first_difference_corr(ecg, eeg)

    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    ax.plot(dt_ms, decg / (np.nanstd(decg) + 1e-12), color="#222222", linewidth=1.1, label="Delta ECG z")
    ax.plot(dt_ms, (deeg - np.nanmean(deeg)) / (np.nanstd(deeg) + 1e-12), color=PALETTE[0], linewidth=1.1, label=f"Delta {channel} EEG z")
    add_time_markers(ax)
    ax.set_xlabel("Time from R peak (ms)")
    ax.set_ylabel("First difference (z)")
    set_centered_title(
        ax,
        "First-Difference Correlation",
        result.stage,
        f"{result.group} | {result.patient_id} | {channel} | r={corr_epoch:.3f}",
    )
    ax.legend(frameon=False)
    scientific_axes(ax, x_major=100)
    return fig


def mpl_raw_corr_group_average(results: Sequence[PatientResult], groups: Sequence[str], channel: str,
                               max_spectral_power_ratio: float = MAX_SPECTRAL_POWER_RATIO,
                               stage: Optional[str] = None) -> Optional[Figure]:
    fig, ax = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
    plotted = False
    for g_idx, group in enumerate(groups):
        mat, times, n_used, n_available = _group_channel_matrix(
            results, group, channel, max_spectral_power_ratio=max_spectral_power_ratio
        )
        if mat is None or times is None:
            continue
        ecg_traces = []
        for r in results:
            if r.group == group and patient_channel_signal_ok(r, channel, max_spectral_power_ratio):
                ecg_traces.append(r.ecg_average)
        if not ecg_traces:
            continue
        color = PALETTE[g_idx % len(PALETTE)]
        t_ms = times * 1000
        eeg_mean = np.nanmean(mat, axis=0)
        ecg_mean = np.nanmean(np.vstack(ecg_traces), axis=0)
        dcor = distance_correlation_1d(ecg_mean, eeg_mean)
        ax.plot(t_ms, ecg_mean / (np.nanstd(ecg_mean) + 1e-12), color=color, linestyle="--", linewidth=1.0, alpha=0.85)
        ax.plot(t_ms, (eeg_mean - np.nanmean(eeg_mean)) / (np.nanstd(eeg_mean) + 1e-12), color=color, linewidth=1.7,
                label=f"{group} EEG mean (N={n_used}/{n_available}, dCor={dcor:.3f})")
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    add_time_markers(ax)
    scientific_axes(ax, x_major=100)
    ax.legend(frameon=False)
    ax.set_ylabel("Mean waveform (z)")
    ax.set_xlabel("Time from R peak (ms)")
    set_centered_title(ax, "Raw Group-Average Waveforms", stage, channel)
    return fig


def mpl_firstdiff_group_average(results: Sequence[PatientResult], groups: Sequence[str], channel: str,
                                max_spectral_power_ratio: float = MAX_SPECTRAL_POWER_RATIO,
                                stage: Optional[str] = None) -> Optional[Figure]:
    fig, ax = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
    plotted = False
    for g_idx, group in enumerate(groups):
        mat, times, n_used, n_available = _group_channel_matrix(
            results, group, channel, max_spectral_power_ratio=max_spectral_power_ratio
        )
        if mat is None or times is None:
            continue
        ecg_traces = []
        for r in results:
            if r.group == group and patient_channel_signal_ok(r, channel, max_spectral_power_ratio):
                ecg_traces.append(r.ecg_average)
        if not ecg_traces:
            continue
        color = PALETTE[g_idx % len(PALETTE)]
        t_ms = times * 1000
        eeg_mean = np.nanmean(mat, axis=0)
        ecg_mean = np.nanmean(np.vstack(ecg_traces), axis=0)
        dx = np.diff(ecg_mean)
        dy = np.diff(eeg_mean)
        corr = first_difference_corr(ecg_mean, eeg_mean)
        ax.plot(t_ms[1:], dx / (np.nanstd(dx) + 1e-12), color=color, linestyle="--", linewidth=1.0, alpha=0.85)
        ax.plot(t_ms[1:], (dy - np.nanmean(dy)) / (np.nanstd(dy) + 1e-12), color=color, linewidth=1.7,
                label=f"{group} Delta EEG (N={n_used}/{n_available}, r={corr:.3f})")
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    add_time_markers(ax)
    scientific_axes(ax, x_major=100)
    ax.legend(frameon=False)
    ax.set_ylabel("First difference (z)")
    ax.set_xlabel("Time from R peak (ms)")
    set_centered_title(ax, "Group-Average First Differences", stage, channel)
    return fig


def ica_component_summary_table(results: Sequence[PatientResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        details = result.ica_details or {}
        table = details.get("component_summary")
        if not isinstance(table, pd.DataFrame) or table.empty:
            rows.append({
                "group": result.group,
                "patient_id": result.patient_id,
                "ica_applied": bool(details.get("ica_applied", False)),
                "ica_reason": details.get("ica_reason", "missing"),
            })
            continue
        top = table.head(2).copy()
        for _, row in top.iterrows():
            rows.append({
                "group": result.group,
                "patient_id": result.patient_id,
                "ica_applied": bool(details.get("ica_applied", False)),
                "ica_reason": details.get("ica_reason", ""),
                "ecg_rank": int(row.get("ecg_rank", 0)),
                "component": int(row.get("component", -1)),
                "removed": bool(row.get("removed", False)),
                "ecg_corr": float(row.get("ecg_corr", np.nan)),
                "abs_ecg_corr": float(row.get("abs_ecg_corr", np.nan)),
                "variance_ratio_pct": float(row.get("variance_ratio_pct", np.nan)),
            })
    return pd.DataFrame(rows)


def mpl_ica_component_averages(result: PatientResult) -> Optional[Figure]:
    details = result.ica_details or {}
    comp_times = details.get("component_times")
    comp_avg_z = details.get("component_average_z")
    summary = details.get("component_summary")
    if comp_times is None or comp_avg_z is None or not isinstance(summary, pd.DataFrame):
        return None
    comp_times = np.asarray(comp_times, dtype=float)
    comp_avg_z = np.asarray(comp_avg_z, dtype=float)
    if comp_avg_z.ndim != 2 or comp_avg_z.shape[1] != len(comp_times):
        return None

    top = summary.head(comp_avg_z.shape[0]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.3), constrained_layout=True)
    for idx, row in top.iterrows():
        if idx >= comp_avg_z.shape[0]:
            break
        label = (
            f"IC {int(row['component'])}: "
            f"var={row['variance_ratio_pct']:.1f}%, rECG={row['ecg_corr']:.2f}"
        )
        ax.plot(comp_times * 1000.0, comp_avg_z[idx], linewidth=1.4,
                color=PALETTE[idx % len(PALETTE)], label=label)
    add_time_markers(ax)
    ax.set_xlabel("Time from R peak (ms)")
    ax.set_ylabel("ICA component average (z)")
    set_centered_title(
        ax,
        "Top ECG-Correlated ICA Components",
        result.stage,
        f"{result.group} | {result.patient_id}",
    )
    ax.legend(frameon=False)
    scientific_axes(ax, x_major=100)
    return fig


def mpl_raw_clean_patient_side_by_side(
    raw_result: PatientResult,
    clean_result: PatientResult,
    channel: str,
    mode: str = "raw",
) -> Optional[Figure]:
    if channel not in raw_result.eeg_channels or channel not in clean_result.eeg_channels:
        return None
    if not patient_channel_signal_ok(raw_result, channel) or not patient_channel_signal_ok(clean_result, channel):
        return None
    raw_idx = raw_result.eeg_channels.index(channel)
    clean_idx = clean_result.eeg_channels.index(channel)
    panels = [
        ("Raw EEG", raw_result, raw_idx),
        ("ICA-clean EEG", clean_result, clean_idx),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.4), sharey=True, constrained_layout=True)
    for ax, (title, result, idx) in zip(axes, panels):
        t_ms = np.asarray(result.times, dtype=float) * 1000.0
        ecg = np.asarray(result.ecg_average, dtype=float)
        eeg = np.asarray(result.eeg_average[idx], dtype=float) * 1e6
        if mode == "firstdiff":
            t_plot = t_ms[1:]
            ecg_plot = np.diff(ecg)
            eeg_plot = np.diff(eeg)
            ylabel = "First difference (z)"
        else:
            t_plot = t_ms
            ecg_plot = ecg
            eeg_plot = eeg
            ylabel = "Waveform (z)"
        ax.plot(t_plot, zscore_1d(ecg_plot), color="#222222", linestyle="--", linewidth=1.0, label="ECG z")
        ax.plot(t_plot, zscore_1d(eeg_plot), color=PALETTE[0], linewidth=1.35, label=f"{channel} EEG z")
        add_time_markers(ax)
        ax.set_title(title, fontweight="bold", loc="center")
        ax.set_xlabel("Time from R peak (ms)")
        scientific_axes(ax, x_major=100)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(frameon=False)
    set_centered_suptitle(
        fig,
        "Sample Patient: Raw vs ICA-Clean",
        raw_result.stage,
        f"{raw_result.group} | {raw_result.patient_id} | {channel}",
    )
    return fig


def mpl_raw_clean_group_average_side_by_side(
    raw_results: Sequence[PatientResult],
    clean_results: Sequence[PatientResult],
    groups: Sequence[str],
    channel: str,
    mode: str = "raw",
    max_spectral_power_ratio: float = MAX_SPECTRAL_POWER_RATIO,
    stage: Optional[str] = None,
) -> Optional[Figure]:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), sharey=True, constrained_layout=True)
    plotted_any = False
    for ax, title, source_results in [
        (axes[0], "Raw EEG", raw_results),
        (axes[1], "ICA-clean EEG", clean_results),
    ]:
        plotted = False
        for g_idx, group in enumerate(groups):
            mat, times, n_used, n_available = _group_channel_matrix(
                source_results, group, channel, max_spectral_power_ratio=max_spectral_power_ratio
            )
            if mat is None or times is None:
                continue
            ecg_traces = [
                r.ecg_average for r in source_results
                if r.group == group
                and patient_channel_signal_ok(r, channel, max_spectral_power_ratio)
            ]
            if not ecg_traces:
                continue
            eeg_mean = np.nanmean(mat, axis=0)
            ecg_mean = np.nanmean(np.vstack(ecg_traces), axis=0)
            t_ms = times * 1000.0
            if mode == "firstdiff":
                t_plot = t_ms[1:]
                ecg_plot = np.diff(ecg_mean)
                eeg_plot = np.diff(eeg_mean)
            else:
                t_plot = t_ms
                ecg_plot = ecg_mean
                eeg_plot = eeg_mean
            color = PALETTE[g_idx % len(PALETTE)]
            ax.plot(t_plot, zscore_1d(ecg_plot), color=color, linestyle="--", linewidth=1.0, alpha=0.8)
            ax.plot(
                t_plot,
                zscore_1d(eeg_plot),
                color=color,
                linewidth=1.55,
                label=f"{group} (N={n_used}/{n_available})",
            )
            plotted = True
        if plotted:
            plotted_any = True
            add_time_markers(ax)
            ax.set_title(title, fontweight="bold", loc="center")
            ax.set_xlabel("Time from R peak (ms)")
            ax.legend(frameon=False)
            scientific_axes(ax, x_major=100)
        else:
            ax.text(0.5, 0.5, "No retained traces", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
    if not plotted_any:
        plt.close(fig)
        return None
    axes[0].set_ylabel("First difference (z)" if mode == "firstdiff" else "Waveform (z)")
    set_centered_suptitle(
        fig,
        "Group Average: Raw vs ICA-Clean",
        stage,
        f"{channel} | {'First differences' if mode == 'firstdiff' else 'Raw waveforms'}",
    )
    return fig


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - np.nanmean(x)) / sd


def time_resolved_score(result: PatientResult, channel: str, mode: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if channel not in result.eeg_channels:
        return None
    idx = result.eeg_channels.index(channel)
    ecg = np.asarray(result.ecg_average, dtype=float)
    eeg = np.asarray(result.eeg_average[idx], dtype=float) * 1e6
    times_ms = np.asarray(result.times, dtype=float) * 1000.0
    if mode == "firstdiff":
        if len(ecg) < 2 or len(eeg) < 2:
            return None
        ecg_z = zscore_1d(np.diff(ecg))
        eeg_z = zscore_1d(np.diff(eeg))
        times_ms = times_ms[1:]
    else:
        ecg_z = zscore_1d(ecg)
        eeg_z = zscore_1d(eeg)
    mismatch = np.abs(eeg_z - ecg_z)
    mismatch_z = zscore_1d(mismatch)
    max_mismatch = np.nanmax(mismatch) if np.any(np.isfinite(mismatch)) else np.nan
    mismatch_pct = 100.0 * mismatch / (max_mismatch + 1e-12) if np.isfinite(max_mismatch) and max_mismatch > 0 else np.zeros_like(mismatch)
    return times_ms, mismatch_z, mismatch_pct


def compute_time_resolved_difference_tables(
    results: Sequence[PatientResult],
    feature_df: pd.DataFrame,
    groups: Sequence[str],
    channels: Sequence[str],
    mode: str,
    time_bin_ms: float = 5.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    valid = feature_df[
        feature_df["group"].isin(groups) & feature_df["channel"].isin(channels)
    ][["group", "patient_id", "channel"]].drop_duplicates()
    valid_keys = set(map(tuple, valid.to_numpy()))
    summary_rows, series_rows = [], []
    for result in results:
        if result.group not in groups:
            continue
        for channel in channels:
            if (result.group, result.patient_id, channel) not in valid_keys:
                continue
            scored = time_resolved_score(result, channel, mode)
            if scored is None:
                continue
            times_ms, mismatch_z, mismatch_pct = scored
            finite = np.isfinite(times_ms) & np.isfinite(mismatch_z) & np.isfinite(mismatch_pct)
            if not np.any(finite):
                continue
            times_ms = times_ms[finite]
            mismatch_z = mismatch_z[finite]
            mismatch_pct = mismatch_pct[finite]
            peak_idx = int(np.nanargmax(mismatch_z))
            q90 = np.nanpercentile(mismatch_z, 90)
            top_mask = mismatch_z >= q90
            summary_rows.append({
                "group": result.group,
                "patient_id": result.patient_id,
                "channel": channel,
                "mode": mode,
                "peak_time_ms": float(times_ms[peak_idx]),
                "peak_z": float(mismatch_z[peak_idx]),
                "top_decile_z": float(np.nanmean(mismatch_z[top_mask])) if np.any(top_mask) else np.nan,
                "mean_percent": float(np.nanmean(mismatch_pct)),
                "peak_percent": float(mismatch_pct[peak_idx]),
            })
            binned_t = np.round(times_ms / time_bin_ms) * time_bin_ms
            for t_val, z_val, pct_val in zip(binned_t, mismatch_z, mismatch_pct):
                series_rows.append({
                    "group": result.group,
                    "patient_id": result.patient_id,
                    "channel": channel,
                    "mode": mode,
                    "time_ms": float(t_val),
                    "mismatch_z": float(z_val),
                    "mismatch_percent": float(pct_val),
                })
    return pd.DataFrame(summary_rows), pd.DataFrame(series_rows)


def time_resolved_peak_ranking(summary_df: pd.DataFrame, groups: Sequence[str], bin_ms: float = 25.0) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    df = summary_df.copy()
    df["peak_time_bin_ms"] = np.round(df["peak_time_ms"] / bin_ms) * bin_ms
    ranked = (
        df.groupby(["mode", "group", "channel", "peak_time_bin_ms"], as_index=False)
        .agg(
            n_patients=("patient_id", "nunique"),
            median_peak_z=("peak_z", "median"),
            mean_top_decile_z=("top_decile_z", "mean"),
            mean_percent=("mean_percent", "mean"),
        )
        .sort_values(["mode", "group", "n_patients", "mean_top_decile_z"], ascending=[True, True, False, False])
    )
    return ranked


def mpl_time_resolved_average(series_df: pd.DataFrame, groups: Sequence[str], channel: str,
                              mode: str, stage: Optional[str] = None) -> Optional[Figure]:
    df = series_df[(series_df["channel"] == channel) & (series_df["group"].isin(groups))]
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    plotted = False
    for g_idx, group in enumerate(groups):
        sub = df[df["group"] == group]
        if sub.empty:
            continue
        patient_time = sub.pivot_table(
            index="patient_id", columns="time_ms", values="mismatch_z", aggfunc="mean"
        ).sort_index(axis=1)
        if patient_time.empty:
            continue
        xs = patient_time.columns.to_numpy(dtype=float)
        mat = patient_time.to_numpy(dtype=float)
        mean = np.nanmean(mat, axis=0)
        sem = stats.sem(mat, axis=0, nan_policy="omit") if mat.shape[0] > 1 else np.zeros_like(mean)
        color = PALETTE[g_idx % len(PALETTE)]
        ax.plot(xs, mean, color=color, linewidth=1.8, label=f"{group} (N={mat.shape[0]})")
        ax.fill_between(xs, mean - sem, mean + sem, color=color, alpha=0.22, linewidth=0)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    add_time_markers(ax)
    ax.axhline(0, color="black", linewidth=0.7, alpha=0.65)
    ax.set_xlabel("Time from R peak (ms)")
    ax.set_ylabel("Mismatch score (z)")
    ax.legend(frameon=False)
    mode_title = "Raw Time-Resolved Difference" if mode == "raw" else "First-Difference Time-Resolved Difference"
    set_centered_title(ax, mode_title, stage, channel)
    scientific_axes(ax, x_major=100)
    return fig


def mpl_time_resolved_patient_examples(series_df: pd.DataFrame, summary_df: pd.DataFrame,
                                       groups: Sequence[str], channel: str, mode: str,
                                       stage: Optional[str] = None) -> Optional[Figure]:
    df = series_df[(series_df["channel"] == channel) & (series_df["group"].isin(groups))]
    summary = summary_df[(summary_df["channel"] == channel) & (summary_df["group"].isin(groups))]
    if df.empty or summary.empty:
        return None

    selected = []
    for group in groups:
        sub_summary = summary[summary["group"] == group].dropna(subset=["top_decile_z"]).sort_values("top_decile_z")
        if sub_summary.empty:
            continue
        row = sub_summary.iloc[len(sub_summary) // 2]
        patient_id = row["patient_id"]
        sub_series = df[(df["group"] == group) & (df["patient_id"] == patient_id)].sort_values("time_ms")
        if not sub_series.empty:
            selected.append((group, patient_id, row, sub_series))

    if not selected:
        return None

    fig, axes = plt.subplots(
        len(selected), 1, figsize=(7.4, max(3.0, 2.8 * len(selected))),
        sharex=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    all_scores = df["mismatch_z"].to_numpy(dtype=float)
    finite_scores = all_scores[np.isfinite(all_scores)]
    vmin = float(np.nanpercentile(finite_scores, 5)) if finite_scores.size else -2.0
    vmax = float(np.nanpercentile(finite_scores, 95)) if finite_scores.size else 2.0
    if np.isclose(vmin, vmax):
        vmin, vmax = vmin - 1.0, vmax + 1.0

    for ax, (group, patient_id, row, sub_series) in zip(axes, selected):
        xs = sub_series["time_ms"].to_numpy(dtype=float)
        ys = sub_series["mismatch_z"].to_numpy(dtype=float)
        pct = sub_series["mismatch_percent"].to_numpy(dtype=float)
        high_cut = np.nanpercentile(ys, 90)
        low_cut = np.nanpercentile(ys, 10)
        colors = plt.cm.Reds(np.clip((ys - vmin) / (vmax - vmin + 1e-12), 0, 1))
        widths = np.diff(np.unique(xs))
        width = float(np.nanmedian(widths)) if len(widths) else 5.0
        ax.bar(xs, ys, width=width * 0.9, color=colors, edgecolor="none", alpha=0.88)
        ax.scatter(xs[ys >= high_cut], ys[ys >= high_cut], color="#7F0000", s=22, label="High-difference bins", zorder=3)
        ax.scatter(xs[ys <= low_cut], ys[ys <= low_cut], color="#2563EB", s=22, label="Low-difference bins", zorder=3)
        ax.axhline(0, color="black", linewidth=0.7)
        add_time_markers(ax)
        ax.set_ylabel("Mismatch (z)")
        subtitle = (
            f"{group} | {patient_id} | peak={row['peak_time_ms']:.1f} ms | "
            f"peak={row['peak_z']:.2f} z | mean={np.nanmean(pct):.1f}%"
        )
        set_centered_title(ax, "Representative Patient Time Bins", subtitle=subtitle,
                           fontweight="normal", fontsize=9)
        scientific_axes(ax, x_major=100)
    axes[-1].set_xlabel("Time from R peak (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False, ncol=2)
    mode_title = "Raw Patient Time-Bin Differences" if mode == "raw" else "First-Difference Patient Time-Bin Differences"
    set_centered_suptitle(fig, mode_title, stage, channel)
    return fig


def mpl_time_resolved_group_bins(series_df: pd.DataFrame, groups: Sequence[str], channel: str,
                                 mode: str, stage: Optional[str] = None) -> Optional[Figure]:
    df = series_df[(series_df["channel"] == channel) & (series_df["group"].isin(groups))]
    if df.empty:
        return None
    selected = []
    all_group_means = []
    for group in groups:
        sub = df[df["group"] == group]
        if sub.empty:
            continue
        patient_time = sub.pivot_table(
            index="patient_id", columns="time_ms", values="mismatch_z", aggfunc="mean"
        ).sort_index(axis=1)
        if patient_time.empty:
            continue
        xs = patient_time.columns.to_numpy(dtype=float)
        mat = patient_time.to_numpy(dtype=float)
        mean = np.nanmean(mat, axis=0)
        sem = stats.sem(mat, axis=0, nan_policy="omit") if mat.shape[0] > 1 else np.zeros_like(mean)
        selected.append((group, xs, mean, sem, mat.shape[0]))
        all_group_means.extend(mean[np.isfinite(mean)].tolist())

    if not selected:
        return None

    finite_scores = np.asarray(all_group_means, dtype=float)
    vmin = float(np.nanpercentile(finite_scores, 5)) if finite_scores.size else -2.0
    vmax = float(np.nanpercentile(finite_scores, 95)) if finite_scores.size else 2.0
    if np.isclose(vmin, vmax):
        vmin, vmax = vmin - 1.0, vmax + 1.0

    fig, axes = plt.subplots(
        len(selected), 1, figsize=(7.4, max(3.0, 2.7 * len(selected))),
        sharex=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for ax, (group, xs, mean, sem, n_patients) in zip(axes, selected):
        high_cut = np.nanpercentile(mean, 90)
        low_cut = np.nanpercentile(mean, 10)
        widths = np.diff(np.unique(xs))
        width = float(np.nanmedian(widths)) if len(widths) else 5.0
        colors = plt.cm.Reds(np.clip((mean - vmin) / (vmax - vmin + 1e-12), 0, 1))
        ax.bar(xs, mean, width=width * 0.9, color=colors, edgecolor="none", alpha=0.9)
        ax.errorbar(xs, mean, yerr=sem, fmt="none", ecolor="#374151", elinewidth=0.6, alpha=0.55, capsize=0)
        ax.scatter(xs[mean >= high_cut], mean[mean >= high_cut], color="#7F0000", s=24,
                   label="High-difference group bins", zorder=3)
        ax.scatter(xs[mean <= low_cut], mean[mean <= low_cut], color="#2563EB", s=24,
                   label="Low-difference group bins", zorder=3)
        ax.axhline(0, color="black", linewidth=0.7)
        add_time_markers(ax)
        ax.set_ylabel("Mean mismatch (z)")
        set_centered_title(
            ax,
            "Group Time-Bin Differences",
            subtitle=f"{group} | N={n_patients}",
            fontweight="normal",
            fontsize=9,
        )
        scientific_axes(ax, x_major=100)
    axes[-1].set_xlabel("Time from R peak (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False, ncol=2)
    mode_title = "Raw Group Time-Bin Differences" if mode == "raw" else "First-Difference Group Time-Bin Differences"
    set_centered_suptitle(fig, mode_title, stage, channel)
    return fig


def mpl_quality_summary(results: List[PatientResult], stage: Optional[str] = None) -> Figure:
    rows = [{
        "patient_id": r.patient_id,
        "group": r.group,
        "pct_kept": 100 * r.n_epochs_kept / r.n_epochs_total if r.n_epochs_total else 0,
        "n_rpeaks": r.n_rpeaks,
        "n_eeg_flipped": len(r.flipped_eeg_channels),
    } for r in results]
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    if df.empty:
        ax.text(0.5, 0.5, "No quality data", ha="center", va="center")
        return fig
    for g_idx, group in enumerate(sorted(df["group"].unique())):
        sub = df[df["group"] == group]
        ax.scatter(sub["patient_id"], sub["pct_kept"], s=np.clip(sub["n_rpeaks"] / 10, 14, 90),
                   color=PALETTE[g_idx % len(PALETTE)], alpha=0.75, label=group,
                   edgecolors="black", linewidths=0.25)
    ax.set_xlabel("Patient")
    ax.set_ylabel("Epochs retained (%)")
    set_centered_title(ax, "Epoch Retention by Patient", stage)
    ax.tick_params(axis="x", rotation=60)
    ax.legend(frameon=False)
    scientific_axes(ax)
    return fig


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("Sleep Stage HEP T-Wave Modulation")
    st.caption(
        "ECG-aligned EEG windows (−0.3 s to +0.5 s). Per-patient ECG polarity correction, "
        "EEG polarity correction from the averaged R-window HEP shape, noisy-window rejection. "
        "Mann-Whitney U with FDR-BH, Bonferroni, rank-biserial r, Cohen's d, and bootstrap CIs."
    )

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Data selection")
        base_path = st.text_input("Pickle base folder", value=BASE_PATH)
        groups_available = list_groups(base_path)
        default_groups = [g for g in ["Berkeley_data", "EDF"] if g in groups_available] or groups_available[:2]
        selected_groups = st.multiselect("Groups", groups_available, default=default_groups)
        stages = list_stages(base_path, selected_groups or groups_available)
        if not stages:
            selected_stage = None
            st.error("No sleep-stage folders were found under the selected base path/groups.")
        else:
            requested_stage = os.environ.get("STREAMLIT_DEFAULT_SLEEP_STAGE", "").strip()
            if requested_stage in stages:
                default_stage_idx = stages.index(requested_stage)
            else:
                default_stage_idx = stages.index("light_sleep") if "light_sleep" in stages else 0
            selected_stage = st.selectbox("Sleep stage", stages, index=default_stage_idx)

        st.divider()
        st.header("Processing")
        test_run_limit = st.number_input("Files per group (0 = all)", 0, 500, 0, 5)
        swing_threshold = 50.0
        bad_fraction = st.slider("Max noisy-channel fraction per epoch", 0.0, 1.0, 0.35, 0.05)

        with st.expander("Advanced analysis settings", expanded=False):
            st.markdown("Alignment windows")
            c1, c2 = st.columns(2)
            win_pre = c1.number_input("Epoch start before R (s)", 0.05, 2.0, abs(DEFAULT_WINDOW[0]), 0.05)
            win_post = c2.number_input("Epoch end after R (s)", 0.05, 2.0, DEFAULT_WINDOW[1], 0.05)
            c1, c2 = st.columns(2)
            t_start = c1.number_input("ECG T-window start (s)", 0.02, 1.5, DEFAULT_T_WINDOW[0], 0.01)
            t_end = c2.number_input("ECG T-window end (s)", 0.03, 1.5, DEFAULT_T_WINDOW[1], 0.01)
            c1, c2 = st.columns(2)
            eeg_t_radius = c1.number_input("EEG T pre-radius (s)", 0.01, 0.5, DEFAULT_EEG_T_RADIUS, 0.01)
            eeg_t_post_s = c2.number_input("EEG T post-radius (s)", 0.0, 0.5, 0.03, 0.01)
            r_peak_flip_window_s = EEG_R_PEAK_FLIP_WINDOW[1]

            st.markdown("EEG flip rule: R-peak parabola direction")
            c1, c2 = st.columns(2)
            r_peak_curvature_threshold = c1.number_input(
                "Flip if R curvature is above",
                min_value=0.0,
                max_value=0.001,
                value=0.0,
                step=0.000005,
                format="%.6f",
                help="Quadratic coefficient after fitting EEG amplitude in uV against time in ms around the detected EEG R peak. Positive means upward-facing; above threshold is flipped.",
            )
            c2.caption(
                "EEG flip uses the averaged HEP trace only in the fixed -10 ms to +100 ms "
                "window around the ECG R peak. Correct orientation is a positive, downward-facing EEG R parabola."
            )
            flip_z_threshold = 3.0
            flip_prominence_threshold = 25.0
            flip_min_votes = 2
            flip_baseline_pre_window = 0.50

            st.markdown("ECG and R-peak detection")
            c1, c2, c3, c4 = st.columns(4)
            ecg_filter_low_hz = c1.number_input("ECG filter low Hz", 0.01, 20.0, 0.5, 0.1)
            ecg_filter_high_hz = c2.number_input("ECG filter high Hz", 5.0, 120.0, 40.0, 1.0)
            ecg_median_ms = c3.number_input("ECG median filter ms", 0.0, 200.0, 20.0, 5.0)
            ecg_clip_sd = c4.number_input("ECG clip SD (0 off)", 0.0, 20.0, 6.0, 0.5)
            c1, c2, c3, c4 = st.columns(4)
            qrs_filter_low_hz = c1.number_input("QRS filter low Hz", 0.5, 30.0, 5.0, 0.5)
            qrs_filter_high_hz = c2.number_input("QRS filter high Hz", 5.0, 80.0, 25.0, 1.0)
            rpeak_mad_multiplier = c3.number_input("R threshold MAD", 1.0, 12.0, 4.0, 0.25)
            rpeak_refractory_s = c4.number_input("R refractory (s)", 0.2, 1.2, 0.4, 0.05)
            c1, c2, c3 = st.columns(3)
            rr_min_s = c1.number_input("RR min (s)", 0.2, 1.5, 0.4, 0.05)
            rr_max_s = c2.number_input("RR max (s)", 0.5, 4.0, 2.2, 0.1)
            min_kept_epochs = c3.number_input("Minimum kept epochs", 3, 200, 5, 1)

            st.markdown("Polarity and artifact rejection")
            c1, c2, c3, c4 = st.columns(4)
            ecg_flip_qrs_half_width_ms = c1.number_input("ECG flip QRS half-width ms", 20.0, 250.0, 80.0, 5.0)
            ecg_flip_min_beats = c2.number_input("ECG flip min beats", 3, 100, 5, 1)
            ecg_flip_ratio = c3.number_input("ECG flip neg/pos ratio", 0.1, 2.0, 0.9, 0.05)
            artifact_mad_multiplier = c4.number_input("Artifact MAD multiplier", 2.0, 20.0, 7.5, 0.5)
            ecg_low_ptp_percentile = st.slider("Reject ECG windows below PTP percentile", 0.0, 25.0, 5.0, 1.0)

        st.divider()
        st.header("Statistics")
        boot_ci = st.checkbox("Bootstrap 95% CIs (slower)", value=True)
        stat_metric = st.selectbox("Primary distance metric",
                                    [
                                        "distance_ms",
                                        "signed_distance_ms",
                                        "eeg_t_peak_amplitude_uv",
                                        "ecg_eeg_distance_corr_twave",
                                        "ecg_eeg_distance_corr_epoch",
                                        "ecg_eeg_firstdiff_corr_twave",
                                        "ecg_eeg_firstdiff_corr_epoch",
                                    ],
                                    format_func=metric_label)

        st.divider()
        recompute = st.button("🔄 Recompute (clear cache)")

    if len(selected_groups) < 1:
        st.warning("Select at least one group in the sidebar.")
        return
    if not selected_stage:
        st.warning("No sleep-stage folders found.")
        return
    if win_pre <= 0 or win_post <= 0 or t_start >= t_end:
        st.error("Invalid epoch/T-wave windows. Epoch sides must be positive and T-window start must be before T-window end.")
        return
    if not (-win_pre <= t_start <= win_post and -win_pre <= t_end <= win_post):
        st.error("The ECG T-wave search window must be inside the R-aligned epoch window.")
        return
    if ecg_filter_low_hz >= ecg_filter_high_hz or qrs_filter_low_hz >= qrs_filter_high_hz:
        st.error("Filter low cutoff must be lower than high cutoff.")
        return
    if rr_min_s >= rr_max_s:
        st.error("RR min must be smaller than RR max.")
        return

    if recompute and hasattr(run_analysis_cached, "clear"):
        run_analysis_cached.clear()
    if recompute:
        removed = clear_processed_analysis_cache()
        st.session_state.pop("ica_ecg_cleanup_payload", None)
        st.sidebar.caption(f"Cleared {removed} processed-data cache file(s).")

    analysis_kwargs = dict(
        base_path=base_path,
        groups=tuple(selected_groups),
        stage=selected_stage,
        test_run_limit=int(test_run_limit),
        window=(-float(win_pre), float(win_post)),
        t_window=(float(t_start), float(t_end)),
        swing_threshold=float(swing_threshold),
        max_bad_epoch_channel_fraction=float(bad_fraction),
        eeg_t_radius=float(eeg_t_radius),
        eeg_t_post_s=float(eeg_t_post_s),
        r_peak_flip_window_s=float(r_peak_flip_window_s),
        min_kept_epochs=int(min_kept_epochs),
        flip_z_threshold=float(flip_z_threshold),
        flip_prominence_threshold=float(flip_prominence_threshold),
        flip_min_votes=int(flip_min_votes),
        flip_baseline_pre_window=float(flip_baseline_pre_window),
        r_peak_curvature_threshold=float(r_peak_curvature_threshold),
        ecg_filter_low_hz=float(ecg_filter_low_hz),
        ecg_filter_high_hz=float(ecg_filter_high_hz),
        ecg_median_ms=float(ecg_median_ms),
        ecg_clip_sd=float(ecg_clip_sd),
        qrs_filter_low_hz=float(qrs_filter_low_hz),
        qrs_filter_high_hz=float(qrs_filter_high_hz),
        rpeak_mad_multiplier=float(rpeak_mad_multiplier),
        rpeak_refractory_s=float(rpeak_refractory_s),
        rr_min_s=float(rr_min_s),
        rr_max_s=float(rr_max_s),
        ecg_flip_qrs_half_width_ms=float(ecg_flip_qrs_half_width_ms),
        ecg_flip_min_beats=int(ecg_flip_min_beats),
        ecg_flip_ratio=float(ecg_flip_ratio),
        artifact_mad_multiplier=float(artifact_mad_multiplier),
        ecg_low_ptp_percentile=float(ecg_low_ptp_percentile),
    )

    results, feature_df = run_analysis_with_processed_cache(**analysis_kwargs)

    if not results or feature_df.empty:
        st.error("No valid patient data produced. Check folders, ECG channels, and stage selection.")
        return

    feature_df_all = feature_df.copy()
    feature_df = apply_spectral_quality_filter(feature_df_all, MAX_SPECTRAL_POWER_RATIO)
    if feature_df.empty:
        st.error(
            f"No patient-channel rows remained after flat-line/noise rejection and "
            f"spectral_power_ratio_hf_lf < {MAX_SPECTRAL_POWER_RATIO}."
        )
        return
    retained_patient_keys = set(
        feature_df[["group", "patient_id"]].drop_duplicates().itertuples(index=False, name=None)
    )
    retained_channel_keys = set(
        feature_df[["group", "patient_id", "channel"]].drop_duplicates().itertuples(index=False, name=None)
    )
    retained_results = [
        r for r in results
        if (r.group, r.patient_id) in retained_patient_keys
    ]

    # Pre-compute derived tables
    patient_summary_rows = []
    for r in retained_results:
        retained_chs = sorted(feature_df.loc[
            (feature_df["group"] == r.group) & (feature_df["patient_id"] == r.patient_id),
            "channel"
        ].unique())
        patient_summary_rows.append({
            "group": r.group, "patient_id": r.patient_id,
            "n_rpeaks": r.n_rpeaks, "epochs_kept": r.n_epochs_kept,
            "epochs_total": r.n_epochs_total,
            "pct_kept": f"{100 * r.n_epochs_kept / r.n_epochs_total:.0f}%" if r.n_epochs_total else "0%",
            "ecg_t_peak_ms": f"{r.ecg_t_peak_s * 1000:.1f}",
            "ecg_flipped": r.flipped_ecg,
            "n_retained_eeg_channels": len(retained_chs),
            "n_retained_eeg_flipped": sum(ch in r.flipped_eeg_channels for ch in retained_chs),
        })
    patient_summary = pd.DataFrame(patient_summary_rows)

    with st.spinner("Computing statistics…"):
        stats_df = group_channel_stats(feature_df, selected_groups, metric=stat_metric, boot_ci=bool(boot_ci))
        patient_median_df, patient_pairwise_stats = patient_median_stats(feature_df, selected_groups, metric=stat_metric)

    common_channels = sorted(
        set.intersection(*[set(feature_df.loc[feature_df["group"] == g, "channel"]) for g in selected_groups])
    ) if selected_groups else sorted(feature_df["channel"].unique())
    if not common_channels:
        st.error(
            f"No common EEG channels remained after flat-line/noise rejection and "
            f"spectral_power_ratio_hf_lf < {MAX_SPECTRAL_POWER_RATIO}."
        )
        return

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tabs = st.tabs(["Overview", "EEG Traces", "Per-Channel Stats",
                     "Patient-Level Stats", "First-Difference Corr",
                     "Time-Resolved Diff", "Heatmap & Correlation", "Quality"])

    # ── Tab 0: Overview ───────────────────────────────────────────────────────
    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        c1.metric("Retained patients", len(patient_summary))
        c2.metric("Channels tested", feature_df["channel"].nunique())
        c3.metric("Feature rows", len(feature_df))
        group_metric_cols = st.columns(min(len(selected_groups), 4))
        for idx, group in enumerate(selected_groups):
            group_metric_cols[idx % len(group_metric_cols)].metric(
                f"{group} patients",
                int((patient_summary["group"] == group).sum()),
            )
        st.caption(
            f"All statistics and group comparisons below use only patient-channel rows with "
            f"non-flat averaged EEG traces, bounded amplitude/roughness, and "
            f"`spectral_power_ratio_hf_lf < {MAX_SPECTRAL_POWER_RATIO}` "
            f"({len(feature_df)}/{len(feature_df_all)} rows retained)."
        )

        if not patient_pairwise_stats.empty:
            best = patient_pairwise_stats.iloc[0]
            st.info(
                f"**Patient-median modulation** - strongest pairwise difference: "
                f"{best['comparison']} | P={format_p(best['p_value'])} | "
                f"FDR={format_p(best['p_fdr_bh'])} | "
                f"rank-biserial r={best.get('rank_biserial_r', np.nan):.3f} | "
                f"Cohen's d={best.get('cohens_d', np.nan):.2f}"
            )

        st.subheader("Retained patient summary")
        st.caption(
            f"This table includes only patients with at least one retained EEG channel after "
            f"flat-line/noise rejection and spectral_power_ratio_hf_lf < {MAX_SPECTRAL_POWER_RATIO}."
        )
        st.dataframe(patient_summary, use_container_width=True, hide_index=True)
        download_block(df=patient_summary, stem=f"{selected_stage}_patient_summary",
                       label="Download patient summary")

        st.subheader("Demographic Analysis")
        with st.spinner("Matching demographic files to retained patients..."):
            demographics_df, demographic_sources_df = demographic_table_for_patients(patient_summary, selected_groups)
            age_summary_df, sex_counts_df = demographic_summary_tables(demographics_df)
        if demographics_df.empty:
            st.warning("No retained patients were available for demographic matching.")
        else:
            matched = int(demographics_df["demographics_matched"].sum())
            total = int(len(demographics_df))
            st.caption(
                f"Matched demographics for {matched}/{total} retained patients. "
                f"Files are loaded from `{EDF_FORMAT_DIRNAME}/<group>/` and matched using normalized patient IDs."
            )
            demo_fig = mpl_demographic_overview(demographics_df, selected_groups, stage=selected_stage)
            if demo_fig:
                st.pyplot(demo_fig, clear_figure=False)
            c1, c2 = st.columns(2)
            c1.markdown("Age summary")
            c1.dataframe(age_summary_df, use_container_width=True, hide_index=True)
            c2.markdown("Sex counts")
            c2.dataframe(sex_counts_df, use_container_width=True, hide_index=True)
            with st.expander("Demographic sources and matched patient table", expanded=False):
                st.markdown("Sources")
                st.dataframe(demographic_sources_df, use_container_width=True, hide_index=True)
                st.markdown("Matched patients")
                st.dataframe(demographics_df.sort_values(["group", "patient_id"]), use_container_width=True, hide_index=True)
            download_block(
                df=demographics_df,
                fig=demo_fig,
                stem=f"{selected_stage}_demographics",
                label="Download demographics",
            )

        settings_df = pd.DataFrame([{
            "stage": selected_stage,
            "groups": ", ".join(selected_groups),
            "epoch_window_s": f"{-float(win_pre):.3f} to {float(win_post):.3f}",
            "ecg_t_window_s": f"{float(t_start):.3f} to {float(t_end):.3f}",
            "primary_metric": stat_metric,
            "eeg_flip_rule": "R-window quadratic curvature; negative/upward R parabolas are flipped",
            "eeg_r_flip_window_ms": (
                f"{EEG_R_PEAK_FLIP_WINDOW[0] * 1000.0:.0f} to "
                f"{EEG_R_PEAK_FLIP_WINDOW[1] * 1000.0:.0f}"
            ),
            "r_peak_curvature_threshold_uv_per_ms2": r_peak_curvature_threshold,
            "max_noisy_channel_fraction": bad_fraction,
            "eeg_avg_min_ptp_uv": EEG_SIGNAL_MIN_PTP_UV,
            "eeg_avg_min_std_uv": EEG_SIGNAL_MIN_STD_UV,
            "eeg_avg_max_abs_uv": EEG_SIGNAL_MAX_ABS_UV,
            "eeg_avg_max_roughness": EEG_SIGNAL_MAX_ROUGHNESS,
            "max_spectral_power_ratio_hf_lf": MAX_SPECTRAL_POWER_RATIO,
            "min_kept_epochs": min_kept_epochs,
            "ecg_filter_hz": f"{ecg_filter_low_hz}-{ecg_filter_high_hz}",
            "qrs_filter_hz": f"{qrs_filter_low_hz}-{qrs_filter_high_hz}",
            "rpeak_mad_multiplier": rpeak_mad_multiplier,
            "rr_bounds_s": f"{rr_min_s}-{rr_max_s}",
        }])
        st.subheader("Analysis settings")
        st.dataframe(settings_df, use_container_width=True, hide_index=True)
        download_block(df=settings_df, stem=f"{selected_stage}_analysis_settings",
                       label="Download analysis settings")

    # ── Tab 1: EEG Traces ─────────────────────────────────────────────────────
    with tabs[1]:
        trace_mode = st.radio("View", ["Per patient", "All patients", "Group comparison"], horizontal=True,
                               key="trace_mode")

        with st.expander("Plot options", expanded=False):
            c1, c2 = st.columns(2)
            show_individual = c1.checkbox("Show individual traces", value=True, key="show_ind")
            height_per_row = c2.slider("Row height (px)", 100, 300, 150, 10, key="row_h")

        if trace_mode == "Per patient":
            patient_labels = [f"{r.group} — {r.patient_id}" for r in retained_results]
            sel_label = st.selectbox("Patient", patient_labels, key="pt_sel")
            result = retained_results[patient_labels.index(sel_label)]
            retained_patient_channels = sorted(feature_df.loc[
                (feature_df["group"] == result.group)
                & (feature_df["patient_id"] == result.patient_id),
                "channel"
            ].unique())
            ch_default = retained_patient_channels
            patient_chs = st.multiselect("EEG channels", retained_patient_channels,
                                          default=ch_default, key="pt_chs")
            fig = mpl_patient_traces(result, patient_chs, height_per_row=height_per_row)
            st.pyplot(fig, clear_figure=False)
            st.caption("Figure note: ECG and selected EEG channels are median heartbeat-locked averages. Red line marks the R peak; green line marks the ECG T-wave peak used for modulation distance.")

            rows = feature_df[(feature_df["group"] == result.group) &
                               (feature_df["patient_id"] == result.patient_id)]
            st.dataframe(rows.sort_values("distance_ms"), use_container_width=True, hide_index=True)
            download_block(df=rows, fig=fig, stem=f"{result.patient_id}_traces",
                           label="Download patient data")

        elif trace_mode == "All patients":
            st.subheader("All Patients EEG Channel View")
            all_patient_groups = st.multiselect(
                "Groups to show",
                selected_groups,
                default=selected_groups,
                key="all_pt_groups",
            )
            all_patient_channels = st.multiselect(
                "EEG channels for each patient figure",
                common_channels,
                default=common_channels,
                key="all_pt_channels",
            )
            max_patients = st.number_input(
                "Maximum patients to render (0 = all)",
                min_value=0,
                max_value=max(1, len(retained_results)),
                value=0,
                step=5,
                key="all_pt_max",
            )

            selected_results = [
                r for r in retained_results
                if r.group in all_patient_groups
                and any((r.group, r.patient_id, ch) in retained_channel_keys for ch in all_patient_channels)
            ]
            if max_patients:
                selected_results = selected_results[: int(max_patients)]

            st.caption(
                f"Rendering {len(selected_results)} patient figure(s). Each figure uses the selected EEG channel list where available for that patient."
            )

            for result in selected_results:
                patient_channels = [
                    ch for ch in all_patient_channels
                    if (result.group, result.patient_id, ch) in retained_channel_keys
                ]
                if not patient_channels:
                    continue
                with st.expander(f"{result.group} - {result.patient_id}", expanded=False):
                    fig = mpl_patient_traces(result, patient_channels, height_per_row=height_per_row)
                    st.pyplot(fig, clear_figure=False)
                    rows = feature_df[
                        (feature_df["group"] == result.group)
                        & (feature_df["patient_id"] == result.patient_id)
                        & (feature_df["channel"].isin(patient_channels))
                    ]
                    st.dataframe(rows.sort_values("distance_ms"), use_container_width=True, hide_index=True)
                    download_block(
                        df=rows,
                        fig=fig,
                        stem=f"{result.group}_{result.patient_id}_all_patient_traces",
                        label="Download this patient figure/data",
                        use_expander=False,
                    )

        else:  # Group comparison
            ch_sel = st.selectbox("Channel", common_channels, key="grp_ch")
            fig = mpl_group_overlay(retained_results, selected_groups, ch_sel,
                                        show_individual=show_individual,
                                        feature_df=feature_df, height=height_per_row * 4,
                                        stage=selected_stage)
            st.pyplot(fig, clear_figure=False)
            n_text = []
            for grp in selected_groups:
                _, _, n_used, n_available = _group_channel_matrix(
                    retained_results, grp, ch_sel, max_spectral_power_ratio=MAX_SPECTRAL_POWER_RATIO
                )
                n_text.append(f"{grp}: N={n_used}/{n_available}")
            st.caption(
                f"Figure note: Group averages include only EEG signals passing flat-line/noise checks and spectral_power_ratio_hf_lf < {MAX_SPECTRAL_POWER_RATIO}. "
                f"Used for {ch_sel}: " + "; ".join(n_text) + ". "
                "Thick traces are group means and shaded bands are SEM across included patients. Faint lines are included individual HEP traces when enabled."
            )

            # Also show individual groups side-by-side below
            cols = st.columns(len(selected_groups))
            for col_i, grp in enumerate(selected_groups):
                single_fig = mpl_group_overlay([r for r in retained_results if r.group == grp],
                                                   [grp], ch_sel,
                                                   show_individual=show_individual,
                                                   feature_df=feature_df, height=300,
                                                   stage=selected_stage)
                if single_fig:
                    cols[col_i].pyplot(single_fig, clear_figure=False)

            download_block(fig=fig, stem=f"{selected_stage}_{ch_sel}_group_comparison",
                           label="Download figure")

    # ── Tab 2: Per-Channel Statistics ────────────────────────────────────────
    with tabs[2]:
        with st.expander("Plot options", expanded=False):
            c1, c2, c3 = st.columns(3)
            plot_type = c1.radio("Plot type", ["Box", "Violin"], horizontal=True, key="pch_type")
            n_top = c2.slider("Top N channels (significance plot)", 10, 60, 30, 5, key="n_top")
            correction = c3.selectbox("Correction for colouring",
                                       ["p_fdr_bh", "p_bonferroni", "p_value"], key="corr_sel")

        st.subheader("Channel-level group comparison")
        st.caption(
            "Two-sided Mann-Whitney U test. Effect size: rank-biserial r (matched to U) and "
            "Cohen's d (for reference). FDR-BH and Bonferroni multiple-comparison corrections. "
            "Bootstrap 95% CIs on group medians."
        )

        if stats_df.empty:
            st.warning("No per-channel statistics computed.")
        else:
            # Significance bar chart
            sig_fig = mpl_pvalue_bar(stats_df, max_channels=n_top, correction=correction,
                                     stage=selected_stage)
            if sig_fig:
                st.pyplot(sig_fig, clear_figure=False)
                st.caption("Figure note: Bars show -log10 uncorrected P values by channel and pairwise group comparison. Red bars pass the selected correction threshold at alpha=0.05.")

            # Forest plot of effect sizes
            forest_fig = mpl_effect_forest(stats_df, selected_groups, max_channels=n_top,
                                           stage=selected_stage)
            if forest_fig:
                st.pyplot(forest_fig, clear_figure=False)
                st.caption("Figure note: Rank-biserial r is positive when group A has larger values than group B in that comparison. Intervals are normal approximations and should be interpreted as descriptive.")

            # Channel-level distance distribution for selected channel
            st.subheader("Per-channel distance distribution")
            ch_box = st.selectbox("Channel", common_channels, key="stats_ch")
            box_metric = st.selectbox("Metric", [
                                                   "distance_ms",
                                                   "signed_distance_ms",
                                                   "eeg_t_peak_amplitude_uv",
                                                   "ecg_eeg_distance_corr_twave",
                                                   "ecg_eeg_distance_corr_epoch",
                                                   "ecg_eeg_firstdiff_corr_twave",
                                                   "ecg_eeg_firstdiff_corr_epoch",
                                                   ],
                                       format_func=metric_label, key="box_met")
            dist_fig = mpl_distribution(feature_df, ch_box, selected_groups,
                                            metric=box_metric, plot_type=plot_type,
                                            stats_df=stats_df, stage=selected_stage)
            if dist_fig:
                st.pyplot(dist_fig, clear_figure=False)
                st.caption(
                    f"Figure note: Only rows with spectral_power_ratio_hf_lf < {MAX_SPECTRAL_POWER_RATIO} are included. "
                    f"Used for {ch_box}: {channel_group_n_text(feature_df, selected_groups, ch_box)}. "
                    "Each point is one patient-channel measurement; boxes/violins summarize the selected metric distribution by group."
                )
                ch_pairwise = stats_df[stats_df["channel"] == ch_box].copy()
                if not ch_pairwise.empty:
                    st.markdown("Pairwise p-values")
                    st.dataframe(ch_pairwise, use_container_width=True, hide_index=True)

            st.subheader("Topographic maps")
            topo_metric = st.selectbox(
                "Topomap metric",
                [
                    "distance_ms",
                    "signed_distance_ms",
                    "eeg_t_peak_amplitude_uv",
                    "ecg_eeg_distance_corr_twave",
                    "ecg_eeg_distance_corr_epoch",
                    "ecg_eeg_firstdiff_corr_twave",
                    "ecg_eeg_firstdiff_corr_epoch",
                ],
                index=[
                    "distance_ms",
                    "signed_distance_ms",
                    "eeg_t_peak_amplitude_uv",
                    "ecg_eeg_distance_corr_twave",
                    "ecg_eeg_distance_corr_epoch",
                    "ecg_eeg_firstdiff_corr_twave",
                    "ecg_eeg_firstdiff_corr_epoch",
                ].index(stat_metric) if stat_metric in [
                    "distance_ms",
                    "signed_distance_ms",
                    "eeg_t_peak_amplitude_uv",
                    "ecg_eeg_distance_corr_twave",
                    "ecg_eeg_distance_corr_epoch",
                    "ecg_eeg_firstdiff_corr_twave",
                    "ecg_eeg_firstdiff_corr_epoch",
                ] else 0,
                format_func=metric_label,
                key="topo_metric",
            )
            topo_fig, topo_table = mpl_group_topomaps(feature_df, selected_groups, topo_metric,
                                                      stage=selected_stage)
            if topo_fig:
                st.pyplot(topo_fig, clear_figure=False)
                st.caption(
                    f"Figure note: Topomaps use channel medians after spectral_power_ratio_hf_lf < {MAX_SPECTRAL_POWER_RATIO}. "
                    "Maps show each selected group separately; when exactly two groups are selected, a difference map is also shown. "
                    "Electrode positions use the standard 10-20 montage, with T3/T4/T5/T6 mapped to T7/T8/P7/P8."
                )
                st.dataframe(topo_table, use_container_width=True, hide_index=True)
                download_block(
                    df=topo_table,
                    fig=topo_fig,
                    stem=f"{selected_stage}_topomaps_{topo_metric}",
                    label="Download topomap figure/data",
                )
            else:
                st.warning("Topomap could not be rendered because too few valid 10-20 EEG channels were available.")

            st.subheader("Full channel table")
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            download_block(df=stats_df, fig=sig_fig,
                           stem=f"{selected_stage}_channel_stats",
                           label="Download channel statistics")

    # ── Tab 3: Patient-Level Statistics ──────────────────────────────────────
    with tabs[3]:
        with st.expander("Plot options", expanded=False):
            pt_plot_type = st.radio("Plot type", ["Box", "Violin"], horizontal=True, key="pt_plot_t")

        pt_fig = mpl_patient_median_box(patient_median_df, selected_groups, patient_pairwise_stats,
                                            plot_type=pt_plot_type, stage=selected_stage)
        if pt_fig:
            st.pyplot(pt_fig, clear_figure=False)
            pt_n_text = "; ".join(
                f"{grp}: N={patient_median_df[patient_median_df['group'] == grp]['patient_id'].nunique()}"
                for grp in selected_groups
            )
            st.caption(
                f"Figure note: Only channels with spectral_power_ratio_hf_lf < {MAX_SPECTRAL_POWER_RATIO} contribute to each patient median. "
                f"Used patients: {pt_n_text}. Each point is one patient after median aggregation across retained EEG channels."
            )

        if not patient_pairwise_stats.empty:
            st.markdown("Pairwise patient-level p-values")
            st.dataframe(patient_pairwise_stats, use_container_width=True, hide_index=True)

        st.subheader("Patient-median table")
        st.dataframe(patient_median_df.sort_values(["group", "patient_id"]),
                     use_container_width=True, hide_index=True)
        download_block(df=patient_median_df, fig=pt_fig,
                       stem=f"{selected_stage}_patient_median_stats",
                       label="Download patient statistics")

        # All feature rows
        st.subheader("All feature rows (patient × channel)")
        flt_group = st.multiselect("Filter by group", selected_groups, default=selected_groups,
                                    key="feat_grp_flt")
        flt_ch = st.multiselect("Filter by channel", common_channels,
                                 default=common_channels[:6], key="feat_ch_flt")
        flt_df = feature_df[feature_df["group"].isin(flt_group) & feature_df["channel"].isin(flt_ch)]
        st.dataframe(flt_df.sort_values(["group", "patient_id", "channel"]),
                     use_container_width=True, hide_index=True)
        download_block(df=flt_df, stem=f"{selected_stage}_feature_rows",
                       label="Download feature rows")

    # ── Tab 4: First-Difference Correlation ──────────────────────────────────
    with tabs[4]:
        st.subheader("Correlation of First Differences")
        st.caption(
            "This analysis correlates Delta ECG and Delta EEG, where Delta x_t = x_t - x_{t-1}. "
            "It measures whether the two waveforms change together over time, while reducing sensitivity to absolute signal level."
        )
        st.latex(r"\Delta x_t = x_t - x_{t-1}, \qquad \Delta y_t = y_t - y_{t-1}")
        st.latex(r"r_{\Delta x,\Delta y} = \frac{\sum_t(\Delta x_t-\overline{\Delta x})(\Delta y_t-\overline{\Delta y})}{\sqrt{\sum_t(\Delta x_t-\overline{\Delta x})^2}\sqrt{\sum_t(\Delta y_t-\overline{\Delta y})^2}}")
        st.caption(
            "Math meaning: instead of asking whether ECG and EEG have the same absolute level, "
            "first-difference correlation asks whether their moment-to-moment changes rise and fall together."
        )
        with st.expander("Correlation algorithm and calculation", expanded=True):
            st.markdown(
                """
                **Raw waveform correlation algorithm:** distance correlation (`dCor`).

                Distance correlation was chosen for the raw ECG/EEG average waveforms because the ECG morphology is not expected to be linearly related to EEG amplitude. Unlike Pearson correlation, distance correlation can detect nonlinear dependence and is zero only when the two signals are statistically independent.

                **First-difference correlation algorithm:** Pearson correlation on first differences.

                First differences convert each signal into moment-to-moment change before correlation, so the statistic asks whether ECG and EEG slopes rise/fall together rather than whether their absolute amplitudes match.
                """
            )
            st.latex(r"\mathrm{dCor}(X,Y)=\frac{\mathrm{dCov}(X,Y)}{\sqrt{\mathrm{dVar}(X)\mathrm{dVar}(Y)}}")
            st.latex(r"\Delta x_t=x_t-x_{t-1},\quad \Delta y_t=y_t-y_{t-1},\quad r=\mathrm{corr}_{Pearson}(\Delta x,\Delta y)")
            st.caption(
                "Both algorithms are calculated per patient and per EEG channel on the heartbeat-locked average. "
                "The dashboard stores values for the full epoch window and for the selected ECG T-wave window."
            )
        fd_channel = st.selectbox("Channel", common_channels, key="fd_channel")
        fd_metric = st.selectbox(
            "First-difference metric",
            ["ecg_eeg_firstdiff_corr_twave", "ecg_eeg_firstdiff_corr_epoch"],
            format_func=metric_label,
            key="fd_metric",
        )
        raw_metric = st.selectbox(
            "Raw nonlinear correlation metric",
            ["ecg_eeg_distance_corr_twave", "ecg_eeg_distance_corr_epoch"],
            format_func=metric_label,
            key="raw_corr_metric",
        )

        st.markdown("### ICA ECG Artifact Cleanup")
        st.caption(
            "ICA cleanup is calculated when this button is pressed, then reused from the processed-data "
            "cache on later reruns. The cleaned run removes the two ICA components with the strongest "
            "ECG correlation, then recomputes the heartbeat averages, correlations, box plots, and topomaps."
        )
        ica_kwargs = {
            **analysis_kwargs,
            "ica_ecg_clean": True,
            "ica_components_to_remove": ICA_COMPONENTS_TO_REMOVE,
            "ica_max_components": ICA_MAX_COMPONENTS,
            "ica_max_fit_samples": ICA_MAX_FIT_SAMPLES,
        }
        ica_key, _ = processed_cache_identity(ica_kwargs)

        ica_payload = st.session_state.get("ica_ecg_cleanup_payload")
        if not ica_payload or ica_payload.get("key") != ica_key:
            cached_ica = load_processed_analysis_cache(ica_kwargs)
            if cached_ica is not None:
                cached_results, cached_feature_df, cached_path = cached_ica
                st.session_state["ica_ecg_cleanup_payload"] = {
                    "key": ica_key,
                    "results": cached_results,
                    "feature_df_all": cached_feature_df,
                }
                ica_payload = st.session_state["ica_ecg_cleanup_payload"]
                st.caption(
                    f"Loaded cached ICA-cleaned processed data from `{os.path.basename(cached_path)}`."
                )

        if st.button("Run / refresh ICA ECG-artifact cleanup", type="primary", key="run_ica_ecg_cleanup"):
            with st.spinner("Loading cached ICA cleanup or recomputing HEP features..."):
                ica_results_run, ica_feature_df_run = run_analysis_with_processed_cache(**ica_kwargs)
            st.session_state["ica_ecg_cleanup_payload"] = {
                "key": ica_key,
                "results": ica_results_run,
                "feature_df_all": ica_feature_df_run,
            }

        ica_payload = st.session_state.get("ica_ecg_cleanup_payload")
        if not ica_payload or ica_payload.get("key") != ica_key:
            st.info("Press the ICA cleanup button to calculate and cache cleaned-data comparisons.")
        else:
            ica_results = ica_payload["results"]
            ica_feature_df_all = ica_payload["feature_df_all"]
            ica_feature_df = apply_spectral_quality_filter(ica_feature_df_all, MAX_SPECTRAL_POWER_RATIO)
            if not ica_results or ica_feature_df.empty:
                st.warning("ICA cleanup did not produce retained patient-channel rows after spectral filtering.")
            else:
                ica_component_df = ica_component_summary_table(ica_results)
                applied_count = sum(bool((r.ica_details or {}).get("ica_applied", False)) for r in ica_results)
                c1, c2, c3 = st.columns(3)
                c1.metric("ICA-clean patients", len(ica_results))
                c2.metric("ICA applied", applied_count)
                c3.metric("Clean feature rows", len(ica_feature_df))

                st.markdown("#### Top 2 ECG-Correlated ICA Components")
                st.dataframe(ica_component_df, use_container_width=True, hide_index=True)
                download_block(
                    df=ica_component_df,
                    stem=f"{selected_stage}_ica_top2_components",
                    label="Download ICA component summary",
                )

                ica_channels = sorted(set(common_channels) & set(ica_feature_df["channel"].unique()))
                if not ica_channels:
                    st.warning("No common channels were retained in both raw and ICA-clean outputs.")
                else:
                    compare_channel = fd_channel if fd_channel in ica_channels else ica_channels[0]
                    if fd_channel not in ica_channels:
                        compare_channel = st.selectbox(
                            "ICA comparison channel",
                            ica_channels,
                            key="ica_compare_channel",
                        )

                    raw_sample_keys = set(
                        feature_df.loc[feature_df["channel"] == compare_channel, ["group", "patient_id"]]
                        .drop_duplicates()
                        .itertuples(index=False, name=None)
                    )
                    clean_sample_keys = set(
                        ica_feature_df.loc[ica_feature_df["channel"] == compare_channel, ["group", "patient_id"]]
                        .drop_duplicates()
                        .itertuples(index=False, name=None)
                    )
                    sample_keys = sorted(raw_sample_keys & clean_sample_keys)
                    if not sample_keys:
                        st.warning(f"No matched raw/ICA patient samples for {compare_channel}.")
                    else:
                        sample_labels = [f"{group} - {patient_id}" for group, patient_id in sample_keys]
                        sample_label = st.selectbox(
                            "Sample patient",
                            sample_labels,
                            key="ica_sample_patient",
                        )
                        sample_group, sample_patient = sample_keys[sample_labels.index(sample_label)]
                        raw_sample = next(
                            (r for r in results if r.group == sample_group and r.patient_id == sample_patient),
                            None,
                        )
                        clean_sample = next(
                            (r for r in ica_results if r.group == sample_group and r.patient_id == sample_patient),
                            None,
                        )
                        if raw_sample is not None and clean_sample is not None:
                            comp_fig = mpl_ica_component_averages(clean_sample)
                            if comp_fig:
                                st.pyplot(comp_fig, clear_figure=False)

                            sample_mode = st.radio(
                                "Sample comparison view",
                                ["raw", "firstdiff"],
                                horizontal=True,
                                format_func=lambda x: "Raw waveform" if x == "raw" else "First differences",
                                key="ica_sample_mode",
                            )
                            sample_fig = mpl_raw_clean_patient_side_by_side(
                                raw_sample, clean_sample, compare_channel, mode=sample_mode
                            )
                            if sample_fig:
                                st.pyplot(sample_fig, clear_figure=False)

                    st.markdown("#### Group Average: Raw vs ICA-Clean")
                    group_raw_fig = mpl_raw_clean_group_average_side_by_side(
                        results, ica_results, selected_groups, compare_channel,
                        mode="raw", stage=selected_stage,
                    )
                    if group_raw_fig:
                        st.pyplot(group_raw_fig, clear_figure=False)
                    group_fd_fig = mpl_raw_clean_group_average_side_by_side(
                        results, ica_results, selected_groups, compare_channel,
                        mode="firstdiff", stage=selected_stage,
                    )
                    if group_fd_fig:
                        st.pyplot(group_fd_fig, clear_figure=False)

                    st.markdown("#### Box Plots: Raw vs ICA-Clean")
                    raw_stats_compare = group_channel_stats(
                        feature_df, selected_groups, metric=raw_metric, boot_ci=bool(boot_ci)
                    )
                    clean_raw_stats = group_channel_stats(
                        ica_feature_df, selected_groups, metric=raw_metric, boot_ci=bool(boot_ci)
                    )
                    raw_fd_stats_compare = group_channel_stats(
                        feature_df, selected_groups, metric=fd_metric, boot_ci=bool(boot_ci)
                    )
                    clean_fd_stats = group_channel_stats(
                        ica_feature_df, selected_groups, metric=fd_metric, boot_ci=bool(boot_ci)
                    )
                    c1, c2 = st.columns(2)
                    raw_before_box = mpl_distribution(
                        feature_df, compare_channel, selected_groups,
                        metric=raw_metric, plot_type="Box", stats_df=raw_stats_compare,
                        stage=selected_stage,
                    )
                    raw_after_box = mpl_distribution(
                        ica_feature_df, compare_channel, selected_groups,
                        metric=raw_metric, plot_type="Box", stats_df=clean_raw_stats,
                        stage=selected_stage,
                    )
                    if raw_before_box:
                        c1.pyplot(raw_before_box, clear_figure=False)
                        c1.caption("Raw EEG")
                    if raw_after_box:
                        c2.pyplot(raw_after_box, clear_figure=False)
                        c2.caption("ICA-clean EEG")

                    c1, c2 = st.columns(2)
                    fd_before_box = mpl_distribution(
                        feature_df, compare_channel, selected_groups,
                        metric=fd_metric, plot_type="Box", stats_df=raw_fd_stats_compare,
                        stage=selected_stage,
                    )
                    fd_after_box = mpl_distribution(
                        ica_feature_df, compare_channel, selected_groups,
                        metric=fd_metric, plot_type="Box", stats_df=clean_fd_stats,
                        stage=selected_stage,
                    )
                    if fd_before_box:
                        c1.pyplot(fd_before_box, clear_figure=False)
                        c1.caption("Raw EEG")
                    if fd_after_box:
                        c2.pyplot(fd_after_box, clear_figure=False)
                        c2.caption("ICA-clean EEG")
                    raw_before_stats = raw_stats_compare[raw_stats_compare["channel"] == compare_channel].copy()
                    raw_after_stats = clean_raw_stats[clean_raw_stats["channel"] == compare_channel].copy()
                    fd_before_stats = raw_fd_stats_compare[raw_fd_stats_compare["channel"] == compare_channel].copy()
                    fd_after_stats = clean_fd_stats[clean_fd_stats["channel"] == compare_channel].copy()
                    if not raw_before_stats.empty or not raw_after_stats.empty:
                        st.markdown("Pairwise raw-correlation p-values: raw vs ICA-clean")
                        c1, c2 = st.columns(2)
                        if not raw_before_stats.empty:
                            c1.dataframe(raw_before_stats, use_container_width=True, hide_index=True)
                        if not raw_after_stats.empty:
                            c2.dataframe(raw_after_stats, use_container_width=True, hide_index=True)
                    if not fd_before_stats.empty or not fd_after_stats.empty:
                        st.markdown("Pairwise first-difference p-values: raw vs ICA-clean")
                        c1, c2 = st.columns(2)
                        if not fd_before_stats.empty:
                            c1.dataframe(fd_before_stats, use_container_width=True, hide_index=True)
                        if not fd_after_stats.empty:
                            c2.dataframe(fd_after_stats, use_container_width=True, hide_index=True)

                    st.markdown("#### Brain Maps: Raw vs ICA-Clean")
                    topo_metric_ica = st.selectbox(
                        "ICA brain-map metric",
                        [fd_metric, raw_metric],
                        format_func=metric_label,
                        key="ica_topo_metric",
                    )
                    raw_topo_fig, _ = mpl_group_topomaps(
                        feature_df, selected_groups, topo_metric_ica, stage=selected_stage
                    )
                    clean_topo_fig, clean_topo_table = mpl_group_topomaps(
                        ica_feature_df, selected_groups, topo_metric_ica, stage=selected_stage
                    )
                    c1, c2 = st.columns(2)
                    if raw_topo_fig:
                        c1.pyplot(raw_topo_fig, clear_figure=False)
                        c1.caption("Raw EEG")
                    if clean_topo_fig:
                        c2.pyplot(clean_topo_fig, clear_figure=False)
                        c2.caption("ICA-clean EEG")
                    if clean_topo_fig:
                        download_block(
                            df=clean_topo_table,
                            fig=clean_topo_fig,
                            stem=f"{selected_stage}_ica_clean_topomap_{topo_metric_ica}",
                            label="Download ICA-clean brain map",
                        )
                    download_block(
                        df=ica_feature_df,
                        stem=f"{selected_stage}_ica_clean_feature_rows",
                        label="Download ICA-clean feature rows",
                    )

        shared_example_rows = {}
        for group in selected_groups:
            rows = feature_df[
                (feature_df["group"] == group)
                & (feature_df["channel"] == fd_channel)
            ].dropna(subset=[raw_metric, fd_metric]).sort_values(raw_metric)
            if not rows.empty:
                shared_example_rows[group] = rows.iloc[len(rows) // 2]

        st.markdown("#### Patient Examples: Raw Data Correlation")
        fd_cols = st.columns(len(selected_groups))
        for col, group in zip(fd_cols, selected_groups):
            example_row = shared_example_rows.get(group)
            if example_row is None:
                col.warning(f"No retained rows for {group} / {fd_channel}.")
                continue
            result = next((r for r in results if r.group == group and r.patient_id == example_row["patient_id"]), None)
            if result is None:
                continue
            fig = mpl_raw_corr_patient_example(result, fd_channel)
            if fig:
                col.pyplot(fig, clear_figure=False)
                col.caption(
                    f"{group}: {result.patient_id}; {metric_label(raw_metric)}={example_row[raw_metric]:.3f}; "
                    f"{metric_label(fd_metric)}={example_row[fd_metric]:.3f}"
                )

        st.markdown("#### Patient Examples: First Differences")
        fd_cols = st.columns(len(selected_groups))
        for col, group in zip(fd_cols, selected_groups):
            example_row = shared_example_rows.get(group)
            if example_row is None:
                col.warning(f"No retained rows for {group} / {fd_channel}.")
                continue
            result = next((r for r in results if r.group == group and r.patient_id == example_row["patient_id"]), None)
            if result is None:
                continue
            fig = mpl_firstdiff_patient_example(result, fd_channel)
            if fig:
                col.pyplot(fig, clear_figure=False)
                col.caption(
                    f"{group}: {result.patient_id}; {metric_label(raw_metric)}={example_row[raw_metric]:.3f}; "
                    f"{metric_label(fd_metric)}={example_row[fd_metric]:.3f}"
                )

        st.markdown("#### Group Average Raw Data")
        raw_group_fig = mpl_raw_corr_group_average(retained_results, selected_groups, fd_channel,
                                                   stage=selected_stage)
        if raw_group_fig:
            st.pyplot(raw_group_fig, clear_figure=False)
            st.caption(
                f"Only channels with spectral_power_ratio_hf_lf < {MAX_SPECTRAL_POWER_RATIO} are included. "
                f"Used for {fd_channel}: {channel_group_n_text(feature_df, selected_groups, fd_channel)}. "
                "Only the R peak is marked."
            )

        st.markdown("#### Group Average First Differences")
        fd_group_fig = mpl_firstdiff_group_average(retained_results, selected_groups, fd_channel,
                                                   stage=selected_stage)
        if fd_group_fig:
            st.pyplot(fd_group_fig, clear_figure=False)
            st.caption(
                f"Only channels with spectral_power_ratio_hf_lf < {MAX_SPECTRAL_POWER_RATIO} are included. "
                f"Used for {fd_channel}: {channel_group_n_text(feature_df, selected_groups, fd_channel)}. "
                "Only the R peak is marked."
            )

        st.markdown("#### Group Difference: Raw Data Correlation")
        raw_stats = group_channel_stats(feature_df, selected_groups, metric=raw_metric, boot_ci=bool(boot_ci))
        raw_box = mpl_distribution(feature_df, fd_channel, selected_groups, metric=raw_metric,
                                   plot_type="Box", stats_df=raw_stats, stage=selected_stage)
        if raw_box:
            st.pyplot(raw_box, clear_figure=False)
            st.caption(
                f"Box plot compares raw nonlinear correlation by group for {fd_channel}. "
                f"Used: {channel_group_n_text(feature_df, selected_groups, fd_channel)}."
            )
            raw_ch_stats = raw_stats[raw_stats["channel"] == fd_channel].copy()
            if not raw_ch_stats.empty:
                st.markdown("Pairwise raw-correlation p-values")
                st.dataframe(raw_ch_stats, use_container_width=True, hide_index=True)

        st.markdown("#### Group Difference: First-Difference Correlation")
        fd_stats = group_channel_stats(feature_df, selected_groups, metric=fd_metric, boot_ci=bool(boot_ci))
        fd_box = mpl_distribution(feature_df, fd_channel, selected_groups, metric=fd_metric,
                                  plot_type="Box", stats_df=fd_stats, stage=selected_stage)
        if fd_box:
            st.pyplot(fd_box, clear_figure=False)
            st.caption(
                f"Box plot compares first-difference correlation by group for {fd_channel}. "
                f"Used: {channel_group_n_text(feature_df, selected_groups, fd_channel)}."
            )
            fd_ch_stats = fd_stats[fd_stats["channel"] == fd_channel].copy()
            if not fd_ch_stats.empty:
                st.markdown("Pairwise first-difference p-values")
                st.dataframe(fd_ch_stats, use_container_width=True, hide_index=True)

        topomap_raw_stats = raw_stats
        topomap_fd_stats = fd_stats
        selected_pair_label = None
        if len(selected_groups) > 2 and ("comparison" in raw_stats.columns or "comparison" in fd_stats.columns):
            pair_labels = sorted(set(raw_stats.get("comparison", pd.Series(dtype=str)).dropna()) |
                                 set(fd_stats.get("comparison", pd.Series(dtype=str)).dropna()))
            if pair_labels:
                selected_pair_label = st.selectbox(
                    "Pair for correlation p-value/difference topomaps",
                    pair_labels,
                    key="corr_topomap_pair",
                )
                if "comparison" in raw_stats.columns:
                    topomap_raw_stats = raw_stats[raw_stats["comparison"] == selected_pair_label].copy()
                if "comparison" in fd_stats.columns:
                    topomap_fd_stats = fd_stats[fd_stats["comparison"] == selected_pair_label].copy()

        st.markdown("#### Group Difference P-Value Topomaps")
        p_topo_col = st.selectbox(
            "P-value correction shown on topomap",
            ["p_value", "p_fdr_bh", "p_bonferroni"],
            format_func=lambda x: {
                "p_value": "Uncorrected P value",
                "p_fdr_bh": "FDR-BH corrected P value",
                "p_bonferroni": "Bonferroni corrected P value",
            }.get(x, x),
            key="corr_p_topomap_col",
        )
        raw_p_topo_fig, fd_p_topo_fig, p_topo_table = mpl_correlation_pvalue_topomaps(
            topomap_raw_stats, topomap_fd_stats, raw_metric, fd_metric, p_col=p_topo_col,
            stage=selected_stage
        )
        if raw_p_topo_fig or fd_p_topo_fig:
            if raw_p_topo_fig:
                st.pyplot(raw_p_topo_fig, clear_figure=False)
            if fd_p_topo_fig:
                st.pyplot(fd_p_topo_fig, clear_figure=False)
            st.caption(
                "Figure note: These topomaps use the same per-channel tests as the raw-correlation "
                "and first-difference group-difference sections. Color uses -log10(p): p=0.05 is white and "
                "smaller p-values move toward red; the colorbar is labeled in P values. Only rows passing spectral_power_ratio_hf_lf < "
                f"{MAX_SPECTRAL_POWER_RATIO} enter the tests."
            )
            st.dataframe(p_topo_table, use_container_width=True, hide_index=True)
            download_block(
                df=p_topo_table,
                fig=raw_p_topo_fig,
                stem=f"{selected_stage}_raw_correlation_pvalue_topomap_{p_topo_col}",
                label="Download raw-correlation p-value topomap",
            )
            if fd_p_topo_fig:
                download_block(
                    fig=fd_p_topo_fig,
                    stem=f"{selected_stage}_firstdiff_correlation_pvalue_topomap_{p_topo_col}",
                    label="Download first-difference p-value topomap",
                )
        else:
            st.warning("P-value topomaps could not be rendered because too few valid 10-20 EEG channels were available.")

        st.markdown("#### Group Difference Correlation-Value Topomaps")
        delta_topo_fig, delta_topo_table = mpl_correlation_delta_topomaps(
            topomap_raw_stats, topomap_fd_stats, selected_groups, raw_metric, fd_metric,
            stage=selected_stage
        )
        if delta_topo_fig:
            st.pyplot(delta_topo_fig, clear_figure=False)
            group_text = selected_pair_label.replace(" vs ", " minus ") if selected_pair_label else "the selected pair"
            st.caption(
                f"Figure note: These maps show the channel-wise median correlation difference "
                f"({group_text}) from the same Mann-Whitney input table. Red means higher values in group A; "
                f"blue means higher values in group B; white is no median difference."
            )
            download_block(
                df=delta_topo_table,
                fig=delta_topo_fig,
                stem=f"{selected_stage}_correlation_delta_topomaps",
                label="Download correlation-value difference topomaps",
            )
        else:
            st.warning("Correlation-value difference topomaps could not be rendered because too few valid 10-20 EEG channels were available.")

        st.markdown("Raw data correlation statistics")
        st.dataframe(raw_stats, use_container_width=True, hide_index=True)
        download_block(df=raw_stats, fig=raw_box, stem=f"{selected_stage}_{fd_channel}_raw_corr",
                       label="Download raw correlation")
        st.markdown("First-difference correlation statistics")
        st.dataframe(fd_stats, use_container_width=True, hide_index=True)
        download_block(df=fd_stats, fig=fd_box, stem=f"{selected_stage}_{fd_channel}_firstdiff_corr",
                       label="Download first-difference correlation")

    # ── Tab 5: Time-Resolved Differences ─────────────────────────────────────
    with tabs[5]:
        st.subheader("Time-Resolved Coupling Differences")
        st.caption(
            "For each patient and EEG channel, the averaged ECG and EEG traces are normalized and compared over time. "
            "Raw mode compares waveform mismatch; first-difference mode compares slope mismatch. Larger scores mark time points "
            "where heart-brain coupling diverges most strongly for that patient/channel."
        )
        with st.expander("Time-resolved options", expanded=False):
            c1, c2, c3 = st.columns(3)
            tr_channels = c1.multiselect(
                "Channels",
                common_channels,
                default=common_channels,
                key="tr_channels",
            )
            tr_modes = c2.multiselect(
                "Modes",
                ["raw", "firstdiff"],
                default=["raw", "firstdiff"],
                format_func=lambda x: "Raw waveform" if x == "raw" else "First difference",
                key="tr_modes",
            )
            tr_bin_ms = c3.slider("Peak-time ranking bin (ms)", 5.0, 100.0, 25.0, 5.0, key="tr_bin")
            c1, c2 = st.columns(2)
            tr_box_metric = c1.selectbox(
                "Box plot metric",
                ["top_decile_z", "peak_z", "mean_percent", "peak_time_ms"],
                format_func=lambda x: {
                    "top_decile_z": "Top-decile mismatch (z)",
                    "peak_z": "Peak mismatch (z)",
                    "mean_percent": "Mean mismatch (%)",
                    "peak_time_ms": "Peak time (ms)",
                }.get(x, x),
                key="tr_box_metric",
            )
            tr_topo_metric = c2.selectbox(
                "Topomap metric",
                ["top_decile_z", "peak_z", "mean_percent", "peak_time_ms"],
                format_func=lambda x: {
                    "top_decile_z": "Top-decile mismatch (z)",
                    "peak_z": "Peak mismatch (z)",
                    "mean_percent": "Mean mismatch (%)",
                    "peak_time_ms": "Peak time (ms)",
                }.get(x, x),
                key="tr_topo_metric",
            )

        if not tr_channels or not tr_modes:
            st.warning("Select at least one channel and one mode.")
        else:
            for tr_mode in tr_modes:
                mode_name = "Raw Waveform" if tr_mode == "raw" else "First Difference"
                st.markdown(f"#### {mode_name}")
                tr_summary, tr_series = compute_time_resolved_difference_tables(
                    results,
                    feature_df,
                    selected_groups,
                    tr_channels,
                    mode=tr_mode,
                    time_bin_ms=5.0,
                )
                if tr_summary.empty:
                    st.warning(f"No time-resolved rows were available for {mode_name}.")
                    continue

                tr_channel = st.selectbox(
                    f"Channel for {mode_name} time course and box plot",
                    sorted(tr_summary["channel"].unique()),
                    key=f"tr_channel_{tr_mode}",
                )
                avg_fig = mpl_time_resolved_average(
                    tr_series, selected_groups, tr_channel, tr_mode, stage=selected_stage
                )
                if avg_fig:
                    st.pyplot(avg_fig, clear_figure=False)
                    st.caption(
                        "Figure note: The trace is the group mean of patient-level mismatch z-scores at each time bin; "
                        "shaded bands are SEM. Higher values indicate times with stronger ECG-EEG divergence."
                    )

                example_fig = mpl_time_resolved_patient_examples(
                    tr_series, tr_summary, selected_groups, tr_channel, tr_mode, stage=selected_stage
                )
                if example_fig:
                    st.pyplot(example_fig, clear_figure=False)
                    st.caption(
                        "Figure note: Each panel shows one representative patient from that group for the selected channel. "
                        "Redder bins are more different; blue markers indicate low-difference bins and dark-red markers indicate high-difference bins."
                    )

                group_bins_fig = mpl_time_resolved_group_bins(
                    tr_series, selected_groups, tr_channel, tr_mode, stage=selected_stage
                )
                if group_bins_fig:
                    st.pyplot(group_bins_fig, clear_figure=False)
                    st.caption(
                        "Figure note: Each panel shows the group-average mismatch for the selected channel at each time bin. "
                        "Error bars are SEM across retained patients; blue markers identify low-difference bins and dark-red markers identify high-difference bins."
                    )

                tr_stats = group_channel_stats(tr_summary, selected_groups, metric=tr_box_metric, boot_ci=False)
                tr_box = mpl_distribution(
                    tr_summary,
                    tr_channel,
                    selected_groups,
                    metric=tr_box_metric,
                    plot_type="Box",
                    stats_df=tr_stats,
                    stage=selected_stage,
                )
                if tr_box:
                    st.pyplot(tr_box, clear_figure=False)
                    st.caption(
                        f"Box plot uses one row per retained patient-channel average. Used for {tr_channel}: "
                        f"{channel_group_n_text(tr_summary, selected_groups, tr_channel)}."
                    )
                    tr_ch_stats = tr_stats[tr_stats["channel"] == tr_channel].copy()
                    if not tr_ch_stats.empty:
                        st.markdown("Pairwise time-resolved p-values")
                        st.dataframe(tr_ch_stats, use_container_width=True, hide_index=True)

                ranking_df = time_resolved_peak_ranking(tr_summary, selected_groups, bin_ms=float(tr_bin_ms))
                st.markdown("Peak-time ranking")
                st.dataframe(ranking_df.head(80), use_container_width=True, hide_index=True)

                topo_fig, topo_table = mpl_group_topomaps(
                    tr_summary,
                    selected_groups,
                    tr_topo_metric,
                    stage=selected_stage,
                )
                if topo_fig:
                    st.pyplot(topo_fig, clear_figure=False)
                    st.caption(
                        "Figure note: Topomaps show channel-wise group medians and the group difference for the selected "
                        "time-resolved mismatch metric."
                    )

                download_block(
                    df=tr_summary,
                    fig=avg_fig,
                    stem=f"{selected_stage}_{tr_mode}_time_resolved_summary",
                    label=f"Download {mode_name.lower()} time-resolved summary",
                )
                if example_fig:
                    download_block(
                        fig=example_fig,
                        stem=f"{selected_stage}_{tr_mode}_{tr_channel}_patient_time_bin_examples",
                        label=f"Download {mode_name.lower()} patient time-bin examples",
                    )
                if group_bins_fig:
                    download_block(
                        fig=group_bins_fig,
                        stem=f"{selected_stage}_{tr_mode}_{tr_channel}_group_time_bin_differences",
                        label=f"Download {mode_name.lower()} group time-bin differences",
                    )
                download_block(
                    df=ranking_df,
                    fig=topo_fig,
                    stem=f"{selected_stage}_{tr_mode}_time_resolved_ranking_topomap",
                    label=f"Download {mode_name.lower()} ranking/topomap",
                )

    # ── Tab 6: Heatmap & Correlation ──────────────────────────────────────────
    with tabs[6]:
        with st.expander("Heatmap options", expanded=False):
            c1, c2, c3 = st.columns(3)
            heat_metric = c1.selectbox("Metric",
                                        [
                                            "distance_ms",
                                            "signed_distance_ms",
                                            "eeg_t_peak_amplitude_uv",
                                            "ecg_eeg_distance_corr_twave",
                                            "ecg_eeg_distance_corr_epoch",
                                            "ecg_eeg_firstdiff_corr_twave",
                                            "ecg_eeg_firstdiff_corr_epoch",
                                        ],
                                        format_func=metric_label, key="heat_met")
            colorscale = c2.selectbox("Colorscale",
                                       ["viridis", "plasma", "RdBu_r", "Reds", "Blues", "cividis"],
                                       key="heat_cs")
            heat_h = c3.slider("Height (px)", 300, 1200, 500, 50, key="heat_h")

        heat_fig = mpl_heatmap(feature_df, selected_groups, metric=heat_metric,
                                   colorscale=colorscale, height=heat_h,
                                   stage=selected_stage)
        if heat_fig:
            st.pyplot(heat_fig, clear_figure=False)
            st.caption(
                f"Figure note: Cell values are group medians after flat-line/noise checks and spectral_power_ratio_hf_lf < {MAX_SPECTRAL_POWER_RATIO}. "
                "Rows are EEG channels and columns are groups; cell annotations show medians."
            )
            download_block(fig=heat_fig, stem=f"{selected_stage}_heatmap_{heat_metric}",
                           label="Download heatmap")

        st.divider()
        st.subheader("ECG T-peak vs EEG T-peak correlation")
        corr_ch = st.selectbox("Channel", common_channels, key="corr_ch")
        corr_fig = mpl_correlation_scatter(feature_df, selected_groups, corr_ch,
                                           stage=selected_stage)
        if corr_fig:
            st.pyplot(corr_fig, clear_figure=False)
            st.caption(
                f"Figure note: Only rows passing flat-line/noise checks and spectral_power_ratio_hf_lf < {MAX_SPECTRAL_POWER_RATIO} are included. "
                f"Used for {corr_ch}: {channel_group_n_text(feature_df, selected_groups, corr_ch)}. "
                "Patient-channel latencies are plotted to inspect whether EEG T-wave timing follows ECG T-wave timing within each group."
            )
            download_block(fig=corr_fig, stem=f"{selected_stage}_{corr_ch}_correlation",
                           label="Download correlation figure")

    # ── Tab 7: Quality ────────────────────────────────────────────────────────
    with tabs[7]:
        st.subheader("Epoch retention and polarity corrections")
        qual_fig = mpl_quality_summary(retained_results, stage=selected_stage)
        st.pyplot(qual_fig, clear_figure=False)
        st.caption(
            "Figure note: Epoch retention summarizes retained patients only after complete-window, artifact, "
            "flat-line/noise, "
            f"and spectral_power_ratio_hf_lf < {MAX_SPECTRAL_POWER_RATIO} filtering. "
            "Low-retention patients should be inspected before final inference."
        )

        if "signal_quality_rejected" in feature_df_all.columns:
            st.subheader("Rejected patient-channel signals")
            reject_df = feature_df_all[feature_df_all["signal_quality_rejected"].fillna(False)].copy()
            if reject_df.empty:
                st.caption("No patient-channel rows were rejected by the flat-line/noise quality gate.")
            else:
                reject_summary = (
                    reject_df.groupby(["group", "signal_quality_reject_reason"], dropna=False)
                    .size()
                    .reset_index(name="n_patient_channels")
                    .sort_values(["group", "n_patient_channels"], ascending=[True, False])
                )
                st.dataframe(reject_summary, use_container_width=True, hide_index=True)
                reject_cols = [
                    c for c in [
                        "group", "patient_id", "channel", "signal_quality_reject_reason",
                        "spectral_power_ratio_hf_lf", "eeg_signal_ptp", "eeg_signal_std",
                        "eeg_signal_max_abs", "eeg_signal_roughness",
                    ]
                    if c in reject_df.columns
                ]
                st.dataframe(reject_df[reject_cols], use_container_width=True, hide_index=True)

        # Per-channel flip detail table
        flip_rows = []
        for r in retained_results:
            for ch in r.eeg_channels:
                if (r.group, r.patient_id, ch) not in retained_channel_keys:
                    continue
                details = r.flip_details.get(ch, {})
                flip_rows.append({
                    "group": r.group, "patient_id": r.patient_id,
                    "stage": r.stage, "channel": ch,
                    "flipped": ch in r.flipped_eeg_channels,
                    **{k: (f"{v:.4f}" if isinstance(v, float) else v)
                       for k, v in details.items()},
                })
        flip_df = pd.DataFrame(flip_rows)

        # Summary counts
        if not flip_df.empty and "flipped" in flip_df.columns:
            flip_rate = flip_df.groupby("group")["flipped"].mean().reset_index()
            flip_rate.columns = ["group", "fraction_channels_flipped"]
            st.dataframe(flip_rate, use_container_width=True, hide_index=True)

        st.caption(
            "ECG polarity: the median QRS template must have an upward R-peak; if not, the whole "
            "ECG trace is negated. EEG channels are flipped only from the averaged HEP trace in the "
            "fixed -10 ms to +100 ms ECG R-window; the detected EEG R shape must be a positive, "
            "downward-facing parabola. This table shows only retained EEG channels "
            f"passing flat-line/noise checks and spectral_power_ratio_hf_lf < {MAX_SPECTRAL_POWER_RATIO}."
        )
        st.dataframe(flip_df, use_container_width=True, hide_index=True)
        download_block(df=flip_df, stem=f"{selected_stage}_flip_details",
                       label="Download flip details")


if __name__ == "__main__":
    main()
