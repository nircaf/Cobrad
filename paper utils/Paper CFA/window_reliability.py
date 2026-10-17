#!/usr/bin/env python3
"""Test-retest reliability of CFA R² vs. window length -- the test for the optimal duration.

cfa_variance_explained_{N}min.parquet draws ONE random window per length, so
length and location are confounded and there is no within-patient repeat to
measure reliability against. Here each recording is cut into 5-min blocks; each
block gets the same QC as select_quality_window() (quality()) and stores the
SUM of its R-peak-locked EEG/ECG epochs. A window of L minutes is then L/5
contiguous QC-passing blocks, and its HEP evoked average is exactly
sum(block sums) / sum(block counts) -- so every length is scored on the same
night, nested, and with as many non-overlapping repeats as the recording allows.

Per (patient, length, window) we store the channel-mean CFA R² (outside QRS),
same definition as cfa_variance_explained.py. make_window_reliability_figure.py
turns this into ICC / within-patient SD vs. length.

Run:
  venv/bin/python "paper utils/Paper CFA/window_reliability.py" \
      --edf-root /storage/pblab_shared_data2/Nir/Cobrad/EDF_Format \
      --patient-list "paper utils/Paper CFA/figure5_7230_patients.txt" --workers 48
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
    HERE, HEP_TMAX, HEP_TMIN, QRS_EXCLUDE_SEC, atomic_parquet, channel_names,
    discover, epoch_around_r_peaks, identifiers, quality,
)

BLOCK_MIN = 5
LENGTHS = [5, 10, 15, 20, 30, 45, 60, 75, 90]
COLUMNS = ["patient_id", "recording_id", "edf_path", "window_minutes", "window_index", "start_block",
           "n_good_blocks", "n_blocks", "n_eeg_channels", "n_hep_epochs", "cfa_r2_excl_qrs"]
ERROR_COLUMNS = ["patient_id", "recording_id", "edf_path", "error_type", "error", "traceback"]
CACHE_VERSION = 1


def r2_excl_qrs(evoked_eeg, evoked_ecg, outside):
    x = evoked_eeg[:, outside] - evoked_eeg[:, outside].mean(1, keepdims=True)
    y = evoked_ecg[outside] - evoked_ecg[outside].mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (x @ y) / np.sqrt((x ** 2).sum(1) * (y ** 2).sum())
    return float(np.nanmean(r ** 2))


def process(job):
    path_text, low, high = job
    path = Path(path_text)
    patient, recording = identifiers(path)
    try:
        raw = mne.io.read_raw_edf(path, preload=False, encoding="latin1", verbose="ERROR")
        eeg, ecg = channel_names(raw)
        raw.pick(eeg + [ecg]).load_data(verbose="ERROR")
        raw.set_channel_types({ch: "eeg" for ch in eeg} | {ecg: "ecg"}, on_unit_change="ignore", verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        upper = min(high, sfreq / 2 - .5)
        if upper <= low:
            raise ValueError(f"sampling rate {sfreq} Hz is too low")
        filtered = raw.copy().filter(low, upper, picks=eeg + [ecg], verbose="ERROR")
        eeg_data = filtered.get_data(picks=eeg) * 1e6
        ecg_data = filtered.get_data(picks=[ecg])[0] * 1e6
        del filtered

        block = int(BLOCK_MIN * 60 * sfreq)
        n_blocks = raw.n_times // block
        pre = int(round(-HEP_TMIN * sfreq))
        sums_eeg, sums_ecg, counts, good = [], [], [], []
        for b in range(n_blocks):
            a = b * block
            passed, qc = quality(mne.io.RawArray(raw._data[:, a:a + block], raw.info, verbose="ERROR"), eeg, ecg)  # view, no copy
            e = epoch_around_r_peaks(eeg_data[:, a:a + block], qc["r_peak_samples"], sfreq)
            c = epoch_around_r_peaks(ecg_data[a:a + block], qc["r_peak_samples"], sfreq)
            good.append(passed and len(c) > 0)
            sums_eeg.append(e.sum(0) if len(e) else 0)
            sums_ecg.append(c.sum(0) if len(c) else 0)
            counts.append(len(c))
        n_t = pre + int(round(HEP_TMAX * sfreq))
        outside = np.abs((np.arange(n_t) - pre) / sfreq) > QRS_EXCLUDE_SEC

        rows = []
        for minutes in LENGTHS:
            k, i, w = minutes // BLOCK_MIN, 0, 0
            while i + k <= n_blocks:  # greedy non-overlapping runs of k contiguous good blocks
                if all(good[i:i + k]):
                    n = sum(counts[i:i + k])
                    if n >= 20:
                        r2 = r2_excl_qrs(sum(sums_eeg[i:i + k]) / n, sum(sums_ecg[i:i + k]) / n, outside)
                        rows.append({"patient_id": patient, "recording_id": recording, "edf_path": str(path),
                                     "window_minutes": minutes, "window_index": w, "start_block": i,
                                     "n_good_blocks": int(sum(good)), "n_blocks": int(n_blocks),
                                     "n_eeg_channels": len(eeg), "n_hep_epochs": n, "cfa_r2_excl_qrs": r2})
                        w += 1
                    i += k
                else:
                    i += 1
        if not rows:
            raise ValueError(f"no QC-passing {BLOCK_MIN}-min block ({n_blocks} blocks)")
        return rows, None
    except Exception as exc:
        return [], {"patient_id": patient, "recording_id": recording, "edf_path": str(path), "error_type": type(exc).__name__,
                    "error": str(exc), "traceback": traceback.format_exc()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edf-root", type=Path, default=HERE.parent.parent / "EDF_Format")
    parser.add_argument("--output", type=Path, default=HERE / "window_reliability.parquet")
    parser.add_argument("--cache-dir", type=Path, default=HERE / "window_reliability_cache")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--l-freq", type=float, default=1)
    parser.add_argument("--h-freq", type=float, default=100)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--patient-list", type=Path)
    args = parser.parse_args()

    paths = discover(args.edf_root, None)
    if args.patient_list:
        wanted = set(args.patient_list.read_text().split())
        paths = [p for p in paths if identifiers(p)[0] in wanted]
    one = {}
    for p in sorted(paths, key=lambda p: str(p).lower()):  # same one-EDF-per-patient rule as cfa_variance_explained.py
        one.setdefault(identifiers(p)[0], p)
    paths = list(one.values())[:args.limit]
    if not paths:
        parser.error(f"no EDF files found under {args.edf_root}")

    jobs, done = [], []
    for p in paths:
        key = hashlib.sha256(f"{CACHE_VERSION}|{p.absolute()}|{args.l_freq}|{args.h_freq}".encode()).hexdigest()
        ok, err = args.cache_dir / "results" / f"{key}.parquet", args.cache_dir / "errors" / f"{key}.parquet"
        if ok.exists() or err.exists():
            done.append(ok if ok.exists() else err)
        else:
            jobs.append(((str(p), args.l_freq, args.h_freq), ok, err))
    print(f"{len(paths):,} patients; cached={len(done):,}; remaining={len(jobs):,}; {args.workers} workers", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, job): (ok, err) for job, ok, err in jobs}
        for future in tqdm(as_completed(futures), total=len(futures), unit="EDF", desc="Window reliability", dynamic_ncols=True):
            rows, error = future.result()
            ok, err = futures[future]
            if error:
                atomic_parquet(pd.DataFrame([error], columns=ERROR_COLUMNS), err)
                tqdm.write(f"ERROR {error['edf_path']}: {error['error']}")
            else:
                atomic_parquet(pd.DataFrame(rows, columns=COLUMNS), ok)
            done.append(err if error else ok)

    frames = [pd.read_parquet(p) for p in done if p.parent.name == "results" and p.exists()]
    errors = [pd.read_parquet(p) for p in done if p.parent.name == "errors" and p.exists()]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    atomic_parquet(out, args.output)
    atomic_parquet(pd.concat(errors, ignore_index=True) if errors else pd.DataFrame(columns=ERROR_COLUMNS),
                   args.output.with_name(args.output.stem + ".errors.parquet"))
    print(f"Wrote {len(out):,} windows from {out.patient_id.nunique():,} patients to {args.output}; {len(errors):,} errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
