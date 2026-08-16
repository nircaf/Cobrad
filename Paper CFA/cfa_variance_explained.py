#!/usr/bin/env python3
"""Direct cardiac-field-artifact (CFA) variance explained, per EEG channel.

Complements ica_ecg_component_variance.py's *ICA-based* estimate (how much
of a channel's HEP evoked variance sits in the component ICA flags as
ECG-related) with a model-free number: R^2 of each EEG channel's own
R-peak-locked HEP evoked average regressed on the ECG's own evoked average
(same epoch window as the HEP analysis, see HEP_TMIN/HEP_TMAX in
ica_ecg_component_variance.py) -- i.e. is the averaged heartbeat-evoked EEG
deflection just a copy of the averaged ECG waveform (cardiac field
artifact), quantified without depending on ICA being correct. Reports this
both over the full epoch and restricted to outside the +/-50ms QRS window
this repo's stats already exclude as unavoidably CFA-dominated.

Uses select_quality_window() from ica_ecg_component_variance.py so, with
matching --window-minutes/--random-seed/--max-window-attempts (the
defaults), this scores the *same* window the ICA run analyzed -- deliberately
kept in its own cache so this doesn't force the (already many-hours-deep)
ICA batch to recompute.

Run:
  python Paper1/cfa_variance_explained.py [--limit N]
  bash Paper1/run_cfa_variance_explained.sh
"""
from __future__ import annotations

import argparse, hashlib, sys, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import mne
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ica_ecg_component_variance import (
    HERE, HEP_TMIN, QRS_EXCLUDE_SEC, atomic_parquet, channel_names,
    discover, epoch_around_r_peaks, identifiers, select_quality_window,
)

COLUMNS = ["patient_id", "recording_id", "edf_path", "eeg_channel", "n_eeg_channels", "sfreq_hz", "recording_duration_s", "window_start_s", "window_duration_s", "window_attempt", "qc_good_eeg_fraction", "qc_ecg_beats", "qc_ecg_bpm", "qc_plausible_rr_fraction", "samples_used", "n_hep_epochs", "cfa_r2_full_epoch", "cfa_r2_excl_qrs", "ecg_channel"]
ERROR_COLUMNS = ["patient_id", "recording_id", "edf_path", "error_type", "error", "traceback"]
CACHE_VERSION = 3  # bumped: cfa_r2 now measured on the R-peak-locked HEP evoked average, not a whole-window correlation


def process(job):
    path_text, minutes, seed, max_attempts, low, high = job
    path = Path(path_text)
    patient, recording = identifiers(path)
    try:
        raw = mne.io.read_raw_edf(path, preload=False, encoding="latin1", verbose="ERROR")
        eeg, ecg = channel_names(raw)
        raw.set_channel_types({ch: "eeg" for ch in eeg}, on_unit_change="ignore", verbose="ERROR")
        raw.set_channel_types({ecg: "ecg"}, on_unit_change="ignore", verbose="ERROR")
        duration = raw.n_times / float(raw.info["sfreq"])
        selected, qc, start, attempt = select_quality_window(raw, eeg, ecg, duration, minutes, seed, path, max_attempts)

        sfreq = float(selected.info["sfreq"])
        upper = min(high, sfreq / 2 - .5)
        if upper <= low:
            raise ValueError(f"sampling rate {sfreq} Hz is too low")
        # Band-pass both EEG and ECG the same way ICA's input is filtered, so
        # this R^2 reflects the artifact within the analysis band, not
        # incidental agreement outside it.
        selected.filter(low, upper, picks=eeg + [ecg], verbose="ERROR")
        eeg_data = selected.get_data(picks=eeg) * 1e6
        ecg_signal = selected.get_data(picks=[ecg])[0] * 1e6
        n_samples = eeg_data.shape[1]

        eeg_epochs = epoch_around_r_peaks(eeg_data, qc["r_peak_samples"], sfreq)  # (n_epochs, n_channels, n_epoch_samples)
        ecg_epochs = epoch_around_r_peaks(ecg_signal, qc["r_peak_samples"], sfreq)  # (n_epochs, n_epoch_samples)
        n_hep_epochs = eeg_epochs.shape[0]
        if n_hep_epochs < 20:
            raise ValueError(f"only {n_hep_epochs} R-peak epochs fit in this window; too few for a HEP evoked estimate")
        evoked_eeg = eeg_epochs.mean(axis=0)  # (n_channels, n_epoch_samples)
        evoked_ecg = ecg_epochs.mean(axis=0)  # (n_epoch_samples,)

        pre = int(round(-HEP_TMIN * sfreq))
        epoch_times = (np.arange(evoked_ecg.shape[0]) - pre) / sfreq
        outside_qrs = np.abs(epoch_times) > QRS_EXCLUDE_SEC

        rows = []
        for channel_index, channel in enumerate(eeg):
            # ponytail: zero-lag correlation only (CFA volume conduction is
            # near-instantaneous); add a small (~50ms) per-channel lag search
            # if conduction-delay underestimation becomes a concern.
            r_full = np.corrcoef(evoked_eeg[channel_index], evoked_ecg)[0, 1]
            r_excl = np.corrcoef(evoked_eeg[channel_index, outside_qrs], evoked_ecg[outside_qrs])[0, 1]
            cfa_r2_full = float(r_full ** 2) if np.isfinite(r_full) else np.nan
            cfa_r2_excl = float(r_excl ** 2) if np.isfinite(r_excl) else np.nan
            rows.append({"patient_id": patient, "recording_id": recording, "edf_path": str(path), "eeg_channel": channel, "n_eeg_channels": len(eeg), "sfreq_hz": sfreq, "recording_duration_s": duration, "window_start_s": start, "window_duration_s": n_samples / sfreq, "window_attempt": attempt, "qc_good_eeg_fraction": qc["good_eeg_fraction"], "qc_ecg_beats": qc["ecg_beats"], "qc_ecg_bpm": qc["ecg_bpm"], "qc_plausible_rr_fraction": qc["plausible_rr_fraction"], "samples_used": n_samples, "n_hep_epochs": n_hep_epochs, "cfa_r2_full_epoch": cfa_r2_full, "cfa_r2_excl_qrs": cfa_r2_excl, "ecg_channel": ecg})
        return rows, None
    except Exception as exc:
        return [], {"patient_id": patient, "recording_id": recording, "edf_path": str(path), "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}


def cache_paths(path: Path, args) -> tuple[Path, Path]:
    """Return parameter-specific success/error cache paths for one EDF."""
    identity = "|".join(map(str, (
        CACHE_VERSION, path.absolute(), args.window_minutes, args.random_seed,
        args.max_window_attempts, args.l_freq, args.h_freq,
    )))
    key = hashlib.sha256(identity.encode()).hexdigest()
    return args.cache_dir / "results" / f"{key}.parquet", args.cache_dir / "errors" / f"{key}.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edf-root", type=Path, default=HERE.parent / "EDF_Format")
    parser.add_argument("--output", type=Path, default=HERE / "cfa_variance_explained.parquet")
    parser.add_argument("--cache-dir", type=Path, default=HERE / "cfa_variance_explained_cache")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--window-minutes", type=float, default=10)
    parser.add_argument("--max-window-attempts", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--l-freq", type=float, default=1)
    parser.add_argument("--h-freq", type=float, default=100)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-errors", action="store_true", help="Retry EDFs already stored in the error cache")
    args = parser.parse_args()
    paths = discover(args.edf_root, args.limit)
    if not paths:
        parser.error(f"no EDF files found under {args.edf_root}")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    success_caches, error_caches = [], []
    for path in paths:
        success_cache, error_cache = cache_paths(path, args)
        if success_cache.exists():
            success_caches.append(success_cache)
            continue
        if error_cache.exists() and not args.retry_errors:
            error_caches.append(error_cache)
            continue
        job = (str(path), args.window_minutes, args.random_seed, args.max_window_attempts, args.l_freq, args.h_freq)
        jobs.append((job, success_cache, error_cache))
    cached = len(success_caches) + len(error_caches)
    print(
        f"Found {len(paths)} EDF files; cached={cached}; remaining={len(jobs)}; "
        f"{args.window_minutes:g}-minute QC windows; {args.workers} workers",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process, job): (job[0], success_cache, error_cache)
            for job, success_cache, error_cache in jobs
        }
        bar = tqdm(as_completed(futures), total=len(futures), unit="EDF", desc="CFA variance", dynamic_ncols=True)
        for future in bar:
            result, error = future.result()
            edf_path, success_cache, error_cache = futures[future]
            if error:
                atomic_parquet(pd.DataFrame([error], columns=ERROR_COLUMNS), error_cache)
                error_caches.append(error_cache)
                tqdm.write(f"ERROR {edf_path}: {error['error']}")
            else:
                atomic_parquet(pd.DataFrame(result, columns=COLUMNS), success_cache)
                success_caches.append(success_cache)
                error_cache.unlink(missing_ok=True)
            bar.set_postfix(cached=len(success_caches), errors=len(error_caches), refresh=False)

    result_frames = [pd.read_parquet(path) for path in success_caches]
    error_frames = [pd.read_parquet(path) for path in error_caches]
    result_frame = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame(columns=COLUMNS)
    error_frame = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame(columns=ERROR_COLUMNS)
    atomic_parquet(result_frame, args.output)
    error_output = args.output.with_name(args.output.stem + ".errors.parquet")
    atomic_parquet(error_frame, error_output)
    print(f"Wrote {len(result_frame):,} rows to {args.output}; {len(error_frame):,} errors to {error_output}")
    return int(bool(len(error_frame) and not len(result_frame)))


if __name__ == "__main__":
    sys.exit(main())
