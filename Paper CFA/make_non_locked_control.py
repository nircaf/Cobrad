#!/usr/bin/env python3
"""Non-heartbeat-locked control for Figure 6: re-epoch the exact same
quality-controlled windows already selected by ica_ecg_component_variance.py
(same edf_path/window_start_s/window_duration_s, read straight from
ica_combined.parquet -- no quality-window re-search needed), but around
uniformly-random pseudo-event times instead of true R-peaks, using the same
epoch count as that recording's real HEP analysis (n_hep_epochs).

If the post-ICA "HEP" variance in Figure 6 were driven purely by finite-epoch
averaging noise (i.e. no real R-peak-locked structure survives at all), it
would look the same as this non-locked control. This is a noise floor, not
an artifact estimate -- unlike CFA, it should NOT differ by channel or by
pre/post-ICA in any structured way; on any given recording it is close to
zero mean, with spread coming from finite-epoch-count sampling variance.

Run on a subsample (representative, not the full corpus -- this is a
control curve, not the main analysis):
  venv/bin/python "Paper CFA/make_non_locked_control.py" --limit 1000
"""
from __future__ import annotations

import argparse, sys, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ica_ecg_component_variance import HERE, atomic_parquet, channel_names, epoch_around_r_peaks

COLUMNS = ["patient_id", "recording_id", "edf_path", "eeg_channel", "n_epochs", "channel_hep_variance_non_locked"]
ERROR_COLUMNS = ["patient_id", "recording_id", "edf_path", "error_type", "error", "traceback"]


def process(job):
    patient_id, recording_id, path_text, start_s, duration_s, sfreq_expected, n_epochs, low, high = job
    path = Path(path_text)
    try:
        raw = mne.io.read_raw_edf(path, preload=False, encoding="latin1", verbose="ERROR")
        eeg, ecg = channel_names(raw)
        raw.set_channel_types({ch: "eeg" for ch in eeg}, on_unit_change="ignore", verbose="ERROR")
        segment = raw.copy().pick(eeg).crop(start_s, start_s + duration_s, include_tmax=False)
        segment.load_data(verbose="ERROR")
        sfreq = float(segment.info["sfreq"])
        upper = min(high, sfreq / 2 - .5)
        if upper <= low:
            raise ValueError(f"sampling rate {sfreq} too low")
        segment.filter(low, upper, picks=eeg, verbose="ERROR")
        eeg_data = segment.get_data(picks=eeg) * 1e6
        n_samples = eeg_data.shape[1]

        pre = int(round(0.3 * sfreq))
        post = int(round(0.4 * sfreq))
        rng = np.random.default_rng(abs(hash((patient_id, recording_id))) % (2**32))
        valid_lo, valid_hi = pre, n_samples - post
        if valid_hi <= valid_lo or n_epochs < 20:
            raise ValueError("window too short for requested epoch count")
        pseudo_peaks = rng.integers(valid_lo, valid_hi, size=int(n_epochs))

        eeg_epochs = epoch_around_r_peaks(eeg_data, pseudo_peaks, sfreq, tmin=-0.3, tmax=0.4)
        if eeg_epochs.shape[0] < 20:
            raise ValueError("too few valid pseudo-epochs")
        evoked = eeg_epochs.mean(axis=0)
        var_per_channel = np.var(evoked, axis=1)

        rows = [{"patient_id": patient_id, "recording_id": recording_id, "edf_path": str(path),
                 "eeg_channel": ch, "n_epochs": int(eeg_epochs.shape[0]),
                 "channel_hep_variance_non_locked": float(var_per_channel[i])}
                for i, ch in enumerate(eeg)]
        return rows, None
    except Exception as exc:
        return [], {"patient_id": patient_id, "recording_id": recording_id, "edf_path": str(path),
                     "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "non_locked_control.parquet")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=1000, help="Subsample size, not the full corpus")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--l-freq", type=float, default=1)
    parser.add_argument("--h-freq", type=float, default=100)
    args = parser.parse_args()

    ica = pd.read_parquet(HERE / "ica_combined.parquet")
    recordings = ica.drop_duplicates(subset=["patient_id", "recording_id"])[
        ["patient_id", "recording_id", "edf_path", "window_start_s", "window_duration_s", "sfreq_hz", "n_hep_epochs"]
    ]
    rng = np.random.default_rng(args.random_seed)
    if len(recordings) > args.limit:
        recordings = recordings.sample(n=args.limit, random_state=args.random_seed)
    print(f"Subsampled {len(recordings):,} / {ica.recording_id.nunique():,} recordings "
          f"already processed by the ICA batch; {args.workers} workers")

    jobs = [
        (r.patient_id, r.recording_id, r.edf_path, r.window_start_s, r.window_duration_s,
         r.sfreq_hz, r.n_hep_epochs, args.l_freq, args.h_freq)
        for r in recordings.itertuples()
    ]
    all_rows, all_errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, job): job[1] for job in jobs}
        bar = tqdm(as_completed(futures), total=len(futures), unit="rec", desc="non-locked control")
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
