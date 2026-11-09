#!/usr/bin/env python3
"""Per-recording features for the EDF_Format source-folder comparison.

Builds the cache behind dashboard 24's "EDF source folders" tab: for every
dataset folder under ``EDF_Format`` (ANPHY-Sleep, EDF, Berkeley_data,
The_Human_Sleep_Project, Young_The_Human_Sleep_Project,
Harvard_Electroencephalography) it extracts one row per patient-recording:

* HEP waveform features on the six sites shared by every dataset
  (F3/F4/C3/C4/O1/O2), in 50 ms bins from 0 to 400 ms after the R-peak,
  both in µV and z-scored within the recording (shape only),
* heart-rate summary (bpm, R-peak count),
* recording metadata (sampling rate, channel counts, duration when known).

Already-processed datasets reuse the per-stage
``individuals_cache_rpeak_shape_v1.pkl`` files (no EDF re-reading). ANPHY-Sleep
has no cache yet, so its EDFs are read directly: the recording's own ECG is
used for R-peak detection and the HEP is computed on a quality-controlled
10-minute window, exactly like the rest of the HEP pipeline.

Writes ``paper utils/Paper CFA/edf_source_features.parquet``.

Run:
    venv/bin/python 25_build_edf_source_cache.py
    venv/bin/python 25_build_edf_source_cache.py --datasets ANPHY-Sleep EDF
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
CFA_DIR = os.path.join(ROOT, "paper utils", "Paper CFA")
PICKLE_ROOT = os.path.join(ROOT, "pickles_sleep_stage")
EDF_ROOT = os.path.join(ROOT, "EDF_Format")
OUTPUT = os.path.join(CFA_DIR, "edf_source_features.parquet")
SITES = ["F3", "F4", "C3", "C4", "O1", "O2"]
STAGES = ["W", "light_sleep", "N3", "R"]
CACHE_NAME = "individuals_cache_rpeak_shape_v1.pkl"
BIN_EDGES = np.arange(0.0, 0.401, 0.05)
HEP_TMIN, HEP_TMAX = -0.3, 0.4
ANPHY_WINDOW_MIN = 10
DATASETS = [
    "ANPHY-Sleep",
    "EDF",
    "Berkeley_data",
    "The_Human_Sleep_Project",
    "Young_The_Human_Sleep_Project",
    "Harvard_Electroencephalography",
]

# Reuse the repository's own I/O, quality control, and R-peak conventions.
sys.path.insert(0, CFA_DIR)
from ica_ecg_component_variance import (  # noqa: E402
    _is_1020_channel,
    channel_names,
    epoch_around_r_peaks,
    select_quality_window,
)


def bin_names() -> list[str]:
    return [f"{int(round(a * 1000)):03d}-{int(round(b * 1000)):03d}ms"
            for a, b in zip(BIN_EDGES[:-1], BIN_EDGES[1:])]


def feature_names() -> list[str]:
    names = [f"{site}_{bin_name}" for site in SITES for bin_name in bin_names()]
    names += [f"mean_{bin_name}" for bin_name in bin_names()]
    names += [f"shape_{site}_{bin_name}" for site in SITES for bin_name in bin_names()]
    names += ["hr_bpm", "n_beats"]
    return names


def _site_index(channels: list[str]) -> dict[str, int]:
    """First channel index per shared 10-20 site (case-insensitive)."""
    index: dict[str, int] = {}
    for position, channel in enumerate(channels):
        name = str(channel).upper().strip()
        if name in SITES and name not in index:
            index[name] = position
    return index


def _waveform_features(hep: np.ndarray, times: np.ndarray, channels: list[str]) -> dict:
    hep = np.asarray(hep, dtype=float)
    times = np.asarray(times, dtype=float)
    sites = _site_index(channels)

    values = np.full((len(SITES), len(BIN_EDGES) - 1), np.nan)
    for row, site in enumerate(SITES):
        if site not in sites:
            continue
        channel = hep[sites[site]]
        for column, (a, b) in enumerate(zip(BIN_EDGES[:-1], BIN_EDGES[1:])):
            mask = (times >= a) & (times < b)
            if mask.any():
                values[row, column] = np.nanmean(channel[mask])

    features = {f"{site}_{name}": values[row, column]
                for row, site in enumerate(SITES)
                for column, name in enumerate(bin_names())}
    channel_mean = np.nanmean(values, axis=0) if np.isfinite(values).any() else np.full(values.shape[1], np.nan)
    for column, name in enumerate(bin_names()):
        features[f"mean_{name}"] = channel_mean[column]

    # Shape: remove the recording's overall amplitude so folders are compared on
    # waveform morphology rather than on hardware gain and referencing.
    rms = np.sqrt(np.nanmean(values ** 2)) if np.isfinite(values).any() else np.nan
    shape = values / rms if np.isfinite(rms) and rms > 0 else np.full_like(values, np.nan)
    for row, site in enumerate(SITES):
        for column, name in enumerate(bin_names()):
            features[f"shape_{site}_{name}"] = shape[row, column]
    return features, int(np.isfinite(values).any(axis=1).sum())


def _recording_duration(recording: str) -> float:
    """Cache ids end with ``<stage>_<window seconds>_<step seconds>``."""
    tokens = recording.split("_")
    if len(tokens) >= 2 and tokens[-2].isdigit():
        return float(tokens[-2])
    return np.nan


def _row_from_cache_item(item, dataset: str, stage: str) -> dict | None:
    if item is None or len(item) < 5:
        return None
    # Per-stage caches store HEP in volts; the dashboard works in µV.
    hep = np.asarray(item[1], dtype=float) * 1e6
    times = np.asarray(item[2], dtype=float)
    channels = [str(x) for x in item[3]]
    if hep.ndim != 2 or hep.shape[0] != len(channels) or not len(times):
        return None
    recording = str(item[0])
    patient = recording.split(f"_{stage}_")[0] if f"_{stage}_" in recording else recording
    features, n_sites = _waveform_features(hep, times, channels)
    if not n_sites:
        return None
    beats = np.asarray(item[4]).ravel()
    duration = _recording_duration(recording)
    n_beats = int(np.isfinite(beats).sum())
    eeg_channels = [c for c in channels if _is_1020_channel(c)]
    row = {
        "dataset": dataset, "recording_id": recording, "patient_id": patient,
        "stage": stage, "sfreq": np.nan, "n_channels": len(channels),
        "n_eeg_channels": len(eeg_channels), "duration_s": duration,
        "n_shared_sites": n_sites,
        "hr_bpm": 60 * n_beats / duration if np.isfinite(duration) and duration > 0 else np.nan,
        "n_beats": n_beats,
    }
    row.update(features)
    return row


def collect_from_cache(dataset: str) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for stage in STAGES:
        path = os.path.join(PICKLE_ROOT, dataset, stage, CACHE_NAME)
        if not os.path.exists(path):
            continue
        with open(path, "rb") as handle:
            items = pickle.load(handle)
        for item in items:
            row = _row_from_cache_item(item, dataset, stage)
            if row is None or row["patient_id"] in seen:
                continue
            seen.add(row["patient_id"])
            rows.append(row)
        del items
    return rows


def _anphy_job(path: str) -> dict | None:
    import mne

    try:
        raw = mne.io.read_raw_edf(path, preload=False, encoding="latin1", verbose="ERROR")
        eeg, ecg = channel_names(raw)
        duration = raw.n_times / float(raw.info["sfreq"])
        segment, qc, start, _ = select_quality_window(
            raw, eeg, ecg, duration, ANPHY_WINDOW_MIN, 1, path, 3
        )
        sfreq = float(segment.info["sfreq"])
        segment.filter(1.0, min(100.0, sfreq / 2 - 0.5), verbose="ERROR")
        data = segment.get_data() * 1e6
        peaks = np.asarray(qc["r_peak_samples"])
        if len(peaks) < 20:
            return None
        times = np.arange(int(round(HEP_TMIN * sfreq)), int(round(HEP_TMAX * sfreq))) / sfreq
        order = list(range(len(eeg)))
        hep = epoch_around_r_peaks(data[order], peaks, sfreq, HEP_TMIN, HEP_TMAX).mean(axis=0)
        features, n_sites = _waveform_features(hep, times, [ch for ch in eeg])
        if not n_sites:
            return None
        row = {
            "dataset": "ANPHY-Sleep", "recording_id": os.path.basename(path),
            "patient_id": os.path.basename(os.path.dirname(path)),
            "stage": "unknown", "sfreq": sfreq, "n_channels": len(raw.ch_names),
            "n_eeg_channels": len(eeg), "duration_s": duration,
            "n_shared_sites": n_sites, "window_start_s": start,
            "hr_bpm": 60 * len(peaks) / (ANPHY_WINDOW_MIN * 60),
            "n_beats": int(len(peaks)),
        }
        row.update(features)
        return row
    except Exception as error:  # noqa: BLE001 -- unreadable recordings are reported and skipped
        print(f"[ANPHY] {os.path.basename(os.path.dirname(path))}: {type(error).__name__}: {error}",
              flush=True)
        return None


def collect_from_anphy(workers: int) -> list[dict]:
    from concurrent.futures import ProcessPoolExecutor

    folder = os.path.join(EDF_ROOT, "ANPHY-Sleep")
    if not os.path.isdir(folder):
        return []
    paths = []
    for directory, _, files in os.walk(folder):
        paths += [os.path.join(directory, name) for name in sorted(files)
                  if name.lower().endswith(".edf") and not name.startswith("._")]
    paths.sort()
    if not paths:
        return []
    rows: list[dict] = []
    if workers <= 1:
        for path in paths:
            row = _anphy_job(path)
            if row is not None:
                rows.append(row)
    else:
        with ProcessPoolExecutor(workers) as executor:
            for row in executor.map(_anphy_job, paths, chunksize=1):
                if row is not None:
                    rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        choices=DATASETS, help="folders to include")
    parser.add_argument("--workers", type=int, default=8, help="ANPHY EDF workers")
    args = parser.parse_args()

    rows: list[dict] = []
    for dataset in args.datasets:
        if dataset == "ANPHY-Sleep":
            collected = collect_from_anphy(args.workers)
        else:
            collected = collect_from_cache(dataset)
        print(f"{dataset}: {len(collected)} recordings", flush=True)
        rows += collected

    if not rows:
        print("No recordings were collected.", file=sys.stderr)
        return 1

    frame = pd.DataFrame(rows)
    for column in feature_names() + ["n_shared_sites"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(["dataset", "patient_id"]).reset_index(drop=True)
    tmp = OUTPUT + ".tmp.parquet"
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, OUTPUT)
    print(f"Wrote {OUTPUT}: {frame.shape[0]} recordings x {frame.shape[1]} columns",
          flush=True)
    print(frame.groupby("dataset").agg(recordings=("patient_id", "size"),
                                       stages=("stage", "nunique"),
                                       shared_sites=("n_shared_sites", "mean")).to_string(),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
