#!/usr/bin/env python3
"""Sensitivity check for cfa_variance_explained.py's two arbitrary choices:
the 10-minute window length, and picking the window location without regard
to sleep stage.

On a subsample of patients (not the full corpus -- this is a diagnostic run,
not the main analysis), for each patient:
  1. Stage the whole recording once with YASA (30 s epochs; same
     prep_for_yasa/staging approach as HEP_parquet_generation.py).
  2. For each of Wake / Light (N1+N2) / N3 / REM, find contiguous stretches
     of that stage (bridging short gaps, same as the project's HEP
     extraction pipeline).
  3. For each window length in --window-minutes-grid, draw up to
     --draws-per-cell reproducible random windows from that stage's
     stretches (when long enough) and compute the same model-free CFA R^2
     as cfa_variance_explained.py.

Output columns let you pivot mean/SD of cfa_r2_excl_qrs by stage and by
window length, and compare variance *within* a (patient, stage, length) cell
across draws (location effect) to variance *across* window lengths (length
effect).

Run:
  venv/bin/python "Paper CFA/window_stage_sensitivity.py" --limit 30 --workers 4
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
    discover, epoch_around_r_peaks, identifiers, quality,
)

REPO = HERE.parent
sys.path.insert(0, str(REPO))
from HEP_parquet_generation import find_stage_epochs, prep_for_yasa  # noqa: E402

EPOCH_SEC = 30
STAGE_TARGETS = {"W": ["W"], "light_sleep": ["N1", "N2"], "N3": ["N3"], "R": ["R"]}
BRIDGE_GAP_EPOCHS = 4  # bridge gaps up to 2 min, matching HEP_parquet_generation.py's stage extraction
DEFAULT_WINDOW_GRID = [2, 5, 10, 20]
COLUMNS = ["patient_id", "recording_id", "edf_path", "stage", "window_minutes", "draw",
           "run_available_minutes", "start_s", "n_eeg_channels", "sfreq_hz",
           "qc_good_eeg_fraction", "qc_ecg_beats", "qc_plausible_rr_fraction",
           "n_hep_epochs", "cfa_r2_full_mean", "cfa_r2_excl_qrs_mean"]
ERROR_COLUMNS = ["patient_id", "recording_id", "edf_path", "error_type", "error", "traceback"]


def contiguous_runs(epoch_indices: np.ndarray) -> list[tuple[int, int]]:
    """[(start_epoch, n_epochs), ...] for consecutive-integer runs in a sorted index array."""
    if epoch_indices.size == 0:
        return []
    epoch_indices = np.sort(epoch_indices)
    breaks = np.where(np.diff(epoch_indices) != 1)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(epoch_indices) - 1]))
    return [(int(epoch_indices[s]), int(e - s + 1)) for s, e in zip(starts, ends)]


def cfa_r2_for_segment(segment, eeg, ecg, low, high) -> dict | None:
    sfreq = float(segment.info["sfreq"])
    upper = min(high, sfreq / 2 - .5)
    if upper <= low:
        return None
    segment.filter(low, upper, picks=eeg + [ecg], verbose="ERROR")
    eeg_data = segment.get_data(picks=eeg) * 1e6
    ecg_signal = segment.get_data(picks=[ecg])[0] * 1e6

    good_ok, qc = quality(segment, eeg, ecg)
    if not (qc["good_eeg_fraction"] >= .50 and qc["plausible_rr_fraction"] >= .50 and qc["ecg_beats"] >= 10):
        return None

    eeg_epochs = epoch_around_r_peaks(eeg_data, qc["r_peak_samples"], sfreq)
    ecg_epochs = epoch_around_r_peaks(ecg_signal, qc["r_peak_samples"], sfreq)
    n_hep_epochs = eeg_epochs.shape[0]
    if n_hep_epochs < 20:
        return None
    evoked_eeg = eeg_epochs.mean(axis=0)
    evoked_ecg = ecg_epochs.mean(axis=0)
    pre = int(round(-HEP_TMIN * sfreq))
    epoch_times = (np.arange(evoked_ecg.shape[0]) - pre) / sfreq
    outside_qrs = np.abs(epoch_times) > QRS_EXCLUDE_SEC

    r2_full, r2_excl = [], []
    for ci in range(len(eeg)):
        r_full = np.corrcoef(evoked_eeg[ci], evoked_ecg)[0, 1]
        r_excl = np.corrcoef(evoked_eeg[ci, outside_qrs], evoked_ecg[outside_qrs])[0, 1]
        if np.isfinite(r_full):
            r2_full.append(r_full ** 2)
        if np.isfinite(r_excl):
            r2_excl.append(r_excl ** 2)
    if not r2_full or not r2_excl:
        return None
    return {
        "n_hep_epochs": n_hep_epochs, "cfa_r2_full_mean": float(np.mean(r2_full)),
        "cfa_r2_excl_qrs_mean": float(np.mean(r2_excl)),
        "qc_good_eeg_fraction": qc["good_eeg_fraction"], "qc_ecg_beats": qc["ecg_beats"],
        "qc_plausible_rr_fraction": qc["plausible_rr_fraction"],
    }


def process(job):
    path_text, window_grid, draws_per_cell, seed, low, high = job
    path = Path(path_text)
    patient, recording = identifiers(path)
    try:
        raw = mne.io.read_raw_edf(path, preload=False, encoding="latin1", verbose="ERROR")
        eeg, ecg = channel_names(raw)
        raw.set_channel_types({ch: "eeg" for ch in eeg}, on_unit_change="ignore", verbose="ERROR")
        raw.set_channel_types({ecg: "ecg"}, on_unit_change="ignore", verbose="ERROR")
        raw.load_data(verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        duration = raw.n_times / sfreq

        stage_electrode = next((c for c in ("Fz", "C4", "C3") if c in eeg), eeg[0])
        raw_stage = prep_for_yasa(raw.copy().pick(eeg + [ecg]), stage_electrode)
        import yasa
        preds = yasa.SleepStaging(raw_stage, eeg_name=stage_electrode).predict()
        preds = np.asarray(preds)

        rows = []
        rng_master = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{seed}:{path}".encode()).digest()[:8], "little"))
        for stage_name, targets in STAGE_TARGETS.items():
            idx = find_stage_epochs(preds.copy(), targets, bridge_gap=BRIDGE_GAP_EPOCHS)
            runs = contiguous_runs(idx)
            if not runs:
                continue
            for minutes in window_grid:
                need_epochs = int(np.ceil(minutes * 60 / EPOCH_SEC))
                candidate_runs = [r for r in runs if r[1] >= need_epochs]
                run_available_minutes = max((r[1] * EPOCH_SEC / 60 for r in runs), default=0.0)
                if not candidate_runs:
                    continue
                for draw in range(draws_per_cell):
                    run_start_epoch, run_len_epochs = candidate_runs[rng_master.integers(len(candidate_runs))]
                    slack_epochs = run_len_epochs - need_epochs
                    offset_epochs = int(rng_master.integers(0, slack_epochs + 1))
                    start_s = (run_start_epoch + offset_epochs) * EPOCH_SEC
                    end_s = start_s + minutes * 60
                    if end_s > duration:
                        continue
                    segment = raw.copy().pick(eeg + [ecg]).crop(start_s, end_s, include_tmax=False)
                    result = cfa_r2_for_segment(segment, eeg, ecg, low, high)
                    del segment
                    if result is None:
                        continue
                    rows.append({
                        "patient_id": patient, "recording_id": recording, "edf_path": str(path),
                        "stage": stage_name, "window_minutes": minutes, "draw": draw,
                        "run_available_minutes": run_available_minutes, "start_s": start_s,
                        "n_eeg_channels": len(eeg), "sfreq_hz": sfreq, **result,
                    })
        return rows, None
    except Exception as exc:
        return [], {"patient_id": patient, "recording_id": recording, "edf_path": str(path),
                     "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edf-root", type=Path, default=REPO / "EDF_Format")
    parser.add_argument("--output", type=Path, default=HERE / "window_stage_sensitivity.parquet")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=30, help="Subsample size, not the full corpus")
    parser.add_argument("--window-minutes-grid", type=int, nargs="+", default=DEFAULT_WINDOW_GRID)
    parser.add_argument("--draws-per-cell", type=int, default=2, help="Independent random locations per (patient, stage, length)")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--l-freq", type=float, default=1)
    parser.add_argument("--h-freq", type=float, default=100)
    args = parser.parse_args()

    all_paths = discover(args.edf_root, None)
    rng = np.random.default_rng(args.random_seed)
    paths = list(rng.choice(all_paths, size=min(args.limit, len(all_paths)), replace=False))
    print(f"Subsampled {len(paths)} / {len(all_paths)} EDFs; grid={args.window_minutes_grid} min; "
          f"{args.draws_per_cell} draws/cell; {args.workers} workers", flush=True)

    jobs = [(str(p), args.window_minutes_grid, args.draws_per_cell, args.random_seed, args.l_freq, args.h_freq) for p in paths]
    all_rows, all_errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, job): job[0] for job in jobs}
        bar = tqdm(as_completed(futures), total=len(futures), unit="EDF", desc="window/stage sensitivity")
        for future in bar:
            rows, error = future.result()
            if error:
                all_errors.append(error)
                tqdm.write(f"ERROR {futures[future]}: {error['error']}")
            else:
                all_rows.extend(rows)
            bar.set_postfix(rows=len(all_rows), errors=len(all_errors), refresh=False)

    result = pd.DataFrame(all_rows, columns=COLUMNS)
    errors = pd.DataFrame(all_errors, columns=ERROR_COLUMNS)
    atomic_parquet(result, args.output)
    atomic_parquet(errors, args.output.with_name(args.output.stem + ".errors.parquet"))
    print(f"Wrote {len(result):,} rows ({result.patient_id.nunique() if len(result) else 0} patients) to {args.output}; {len(errors):,} errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
