#!/usr/bin/env python3
"""Entropy companion to make_non_locked_control.py, for Figure 5 panels c/d:
spectral entropy of the pre-ICA, post-ICA, and non-heartbeat-locked evoked
average, per channel.

Unlike variance (already cached per-EDF by ica_ecg_component_variance.py),
entropy needs the actual evoked *waveform*, which that batch does not save
-- only the scalar variance is persisted. This script therefore re-fits ICA
itself on a subsample (same window-selection approach as the main batch:
patient/recording/window taken from ica_combined.parquet so it scores the
same data), computing all three evoked waveforms in one pass so the
comparison is apples-to-apples with Figure 5's variance panels.

Spectral entropy = Shannon entropy of the (Welch) power spectral density of
the evoked waveform, normalised to [0, 1] by log2(n_freq_bins): near 1 means
a flat, noise-like spectrum; near 0 means power concentrated in a few
frequencies (a more "structured" waveform).

Run on a subsample (ICA fitting is the expensive step -- not the full corpus):
  venv/bin/python "Paper CFA/make_entropy_control.py" --limit 500
"""
from __future__ import annotations

import argparse, sys, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
from tqdm.auto import tqdm

from ica_ecg_component_variance import HERE, atomic_parquet, channel_names, epoch_around_r_peaks, quality

COLUMNS = ["patient_id", "recording_id", "edf_path", "eeg_channel",
           "entropy_pre_ica", "entropy_post_ica", "entropy_non_locked"]
ERROR_COLUMNS = ["patient_id", "recording_id", "edf_path", "error_type", "error", "traceback"]


def spectral_entropy(x: np.ndarray, sfreq: float) -> float:
    """Normalised (0-1) Shannon entropy of the Welch PSD."""
    nperseg = min(len(x), 128)
    if nperseg < 8:
        return np.nan
    _, psd = welch(x, fs=sfreq, nperseg=nperseg)
    psd = psd[psd > 0]
    if psd.size < 2:
        return np.nan
    p = psd / psd.sum()
    h = -np.sum(p * np.log2(p))
    return float(h / np.log2(p.size))


def process(job):
    patient_id, recording_id, path_text, start_s, duration_s, low, high = job
    path = Path(path_text)
    try:
        raw = mne.io.read_raw_edf(path, preload=False, encoding="latin1", verbose="ERROR")
        eeg, ecg = channel_names(raw)
        raw.set_channel_types({ch: "eeg" for ch in eeg}, on_unit_change="ignore", verbose="ERROR")
        raw.set_channel_types({ecg: "ecg"}, on_unit_change="ignore", verbose="ERROR")
        segment = raw.copy().pick(eeg + [ecg]).crop(start_s, start_s + duration_s, include_tmax=False)
        segment.load_data(verbose="ERROR")
        sfreq = float(segment.info["sfreq"])
        upper = min(high, sfreq / 2 - .5)
        if upper <= low:
            raise ValueError(f"sampling rate {sfreq} too low")
        segment.filter(low, upper, picks=eeg + [ecg], verbose="ERROR")

        passed, qc = quality(segment, eeg, ecg)
        r_peaks = qc["r_peak_samples"]
        if len(r_peaks) < 20:
            raise ValueError("too few R-peaks for evoked estimate")

        try:
            import picard  # noqa: F401
            method, params = "picard", {"ortho": False, "extended": True}
        except ImportError:
            method, params = "infomax", {"extended": True}
        count = min(15, len(eeg) - 1)
        ica = mne.preprocessing.ICA(n_components=count, method=method, fit_params=params,
                                     random_state=42, max_iter=500)
        ica.fit(segment, picks=eeg, verbose="ERROR")
        bad, scores = ica.find_bads_ecg(segment, ch_name=ecg, method="correlation", verbose="ERROR")
        scores = np.asarray(scores, float)
        if not bad and scores.size:
            bad = [int(np.nanargmax(np.abs(scores)))]
        cleaned = segment.copy()
        ica.apply(cleaned, exclude=bad, verbose="ERROR")

        eeg_pre = segment.get_data(picks=eeg) * 1e6
        eeg_post = cleaned.get_data(picks=eeg) * 1e6
        n_samples = eeg_pre.shape[1]

        pre_epochs = epoch_around_r_peaks(eeg_pre, r_peaks, sfreq, tmin=-0.3, tmax=0.4)
        post_epochs = epoch_around_r_peaks(eeg_post, r_peaks, sfreq, tmin=-0.3, tmax=0.4)
        if pre_epochs.shape[0] < 20:
            raise ValueError("too few valid R-peak epochs")
        n_epochs = pre_epochs.shape[0]

        pre_margin = int(round(0.3 * sfreq))
        post_margin = int(round(0.4 * sfreq))
        rng = np.random.default_rng(abs(hash((patient_id, recording_id, "entropy"))) % (2**32))
        valid_lo, valid_hi = pre_margin, n_samples - post_margin
        if valid_hi <= valid_lo:
            raise ValueError("window too short")
        pseudo_peaks = rng.integers(valid_lo, valid_hi, size=int(n_epochs))
        non_locked_epochs = epoch_around_r_peaks(eeg_pre, pseudo_peaks, sfreq, tmin=-0.3, tmax=0.4)
        if non_locked_epochs.shape[0] < 20:
            raise ValueError("too few valid pseudo-epochs")

        evoked_pre = pre_epochs.mean(axis=0)
        evoked_post = post_epochs.mean(axis=0)
        evoked_nl = non_locked_epochs.mean(axis=0)

        rows = []
        for i, ch in enumerate(eeg):
            rows.append({
                "patient_id": patient_id, "recording_id": recording_id, "edf_path": str(path),
                "eeg_channel": ch,
                "entropy_pre_ica": spectral_entropy(evoked_pre[i], sfreq),
                "entropy_post_ica": spectral_entropy(evoked_post[i], sfreq),
                "entropy_non_locked": spectral_entropy(evoked_nl[i], sfreq),
            })
        return rows, None
    except Exception as exc:
        return [], {"patient_id": patient_id, "recording_id": recording_id, "edf_path": str(path),
                     "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "entropy_control.parquet")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=500, help="Subsample size, not the full corpus")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--l-freq", type=float, default=1)
    parser.add_argument("--h-freq", type=float, default=100)
    args = parser.parse_args()

    ica = pd.read_parquet(HERE / "ica_combined.parquet")
    recordings = ica.drop_duplicates(subset=["patient_id", "recording_id"])[
        ["patient_id", "recording_id", "edf_path", "window_start_s", "window_duration_s"]
    ]
    if len(recordings) > args.limit:
        recordings = recordings.sample(n=args.limit, random_state=args.random_seed)
    print(f"Subsampled {len(recordings):,} / {ica.recording_id.nunique():,} recordings; "
          f"{args.workers} workers (ICA re-fit per recording -- slower than the variance-only control)")

    jobs = [
        (r.patient_id, r.recording_id, r.edf_path, r.window_start_s, r.window_duration_s, args.l_freq, args.h_freq)
        for r in recordings.itertuples()
    ]
    all_rows, all_errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, job): job[1] for job in jobs}
        bar = tqdm(as_completed(futures), total=len(futures), unit="rec", desc="entropy control")
        for future in bar:
            rows, error = future.result()
            if error:
                all_errors.append(error)
            else:
                all_rows.extend(rows)
            bar.set_postfix(rows=len(all_rows), errors=len(all_errors), refresh=False)

    result = pd.DataFrame(all_rows, columns=COLUMNS)
    errors = pd.DataFrame(all_errors, columns=ERROR_COLUMNS)
    atomic_parquet(result, args.output)
    atomic_parquet(errors, args.output.with_name(args.output.stem + ".errors.parquet"))
    print(f"Wrote {len(result):,} rows ({result.patient_id.nunique() if len(result) else 0} patients) "
          f"to {args.output}; {len(errors):,} errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
