#!/usr/bin/env python3
"""Cardiac-phase-of-arousal dataset builder.

For every cortical-arousal onset (rising edge of annotations/arousal) in a
capped subset of Harvard I0003 PSG .h5 sessions, compute the cardiac phase

    phi = 2*pi * (t - R_i) / (R_{i+1} - R_i),    R_i <= t < R_{i+1}

and attach sleep stage, subject, age and sex.

  source venv/bin/activate && python3 "paper utils/Paper Cardiac Phase Arousal/build_dataset.py"

Resumable: one parquet per session under cache/; re-runs skip cached sessions,
so N_SESSIONS can be raised later without redoing work.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import neurokit2 as nk
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BIDS_ROOT = HERE.parent.parent / "EDF_Format" / "Harvard_Electroencephalography" / "bids" / "I0003"
DEMO_PARQUET = HERE.parent / "Paper CFA" / "demographics_combined.parquet"
CACHE = HERE / "cache"
EVENTS_PARQUET = HERE / "events.parquet"
SESSIONS_PARQUET = HERE / "sessions.parquet"
LOG = HERE / "build_dataset.log"

N_SESSIONS = 1500             # cap; raise and re-run to extend (cache makes it cheap)
PLAUSIBLE_RR = (0.3, 2.0)     # s; bracketing RR outside this -> event dropped
RR_STABILITY_TOL = 0.20       # bracketing RR must be within 20% of local median RR
RR_LOCAL_WINDOW = 10          # beats each side for the local median
N_PHASE_BINS = 12

# annotations/stage code -> label. Derived by matching per-code epoch counts against
# the human-readable sleep_stage labels in the sibling *_sleepannotations.csv across
# 10 sessions (counts matched exactly, +-1 epoch; N3 and N4 both map to code 2).
STAGE_MAP = {0: "N1", 1: "N2", 2: "N3", 3: "REM", 4: "W", 9: "UNSCORED"}


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def r_peak_times(ecg: np.ndarray, sfreq: float) -> np.ndarray:
    """neurokit2 clean + R-peak detection -> peak times in seconds."""
    ecg = np.nan_to_num(ecg.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    clean = nk.ecg_clean(ecg, sampling_rate=sfreq)
    _, info = nk.ecg_peaks(clean, sampling_rate=sfreq)
    return np.asarray(info["ECG_R_Peaks"], dtype=float) / sfreq


def local_median_rr(rr: np.ndarray, w: int = RR_LOCAL_WINDOW) -> np.ndarray:
    """Median RR over a +-w-beat neighbourhood, per interval."""
    return pd.Series(rr).rolling(2 * w + 1, center=True, min_periods=3).median().to_numpy()


def process_session(h5_path: Path) -> pd.DataFrame | None:
    with h5py.File(h5_path, "r") as f:
        sfreq = float(f.attrs["sampling_rate"])
        age_years = float(f.attrs["AgeinDays"]) / 365.25
        ecg = f["signals/ecg"][:, 0]
        arousal = f["annotations/arousal"][:, 0]
        stage = f["annotations/stage"][:, 0]

    onset_idx = np.flatnonzero((arousal[1:] == 1) & (arousal[:-1] == 0)) + 1
    if onset_idx.size == 0:
        return None
    r_t = r_peak_times(ecg, sfreq)
    if r_t.size < 100:
        return None

    rr = np.diff(r_t)
    med = local_median_rr(rr)
    ok_rr = (
        (rr >= PLAUSIBLE_RR[0]) & (rr <= PLAUSIBLE_RR[1])
        & np.isfinite(med) & (np.abs(rr - med) <= RR_STABILITY_TOL * med)
    )

    t_on = onset_idx / sfreq
    # bracketing interval index i such that r_t[i] <= t < r_t[i+1]
    i = np.searchsorted(r_t, t_on, side="right") - 1
    inside = (i >= 0) & (i < rr.size)
    i, t_on, onset_idx = i[inside], t_on[inside], onset_idx[inside]
    if i.size == 0:
        return None
    keep = ok_rr[i]
    i, t_on, onset_idx = i[keep], t_on[keep], onset_idx[keep]
    if i.size == 0:
        return None

    phi = 2 * np.pi * (t_on - r_t[i]) / rr[i]
    sub, ses = h5_path.parts[-4], h5_path.parts[-3]
    return pd.DataFrame({
        "subject": sub,
        "session": ses,
        "patient_id": sub.replace("sub-", ""),
        "t_onset": t_on,
        "phi": phi,
        "rr": rr[i],
        "stage": [STAGE_MAP.get(int(c), "UNKNOWN") for c in stage[onset_idx]],
        "age_years": age_years,
        "sfreq": sfreq,
        "n_rr_valid": int(ok_rr.sum()),
        "n_rr_total": int(rr.size),
        "recording_hours": len(arousal) / sfreq / 3600.0,
    })


def main() -> None:
    CACHE.mkdir(exist_ok=True)
    sessions = sorted(BIDS_ROOT.glob("sub-*/ses-*/eeg/*_task-PSG_eeg.h5"))[:N_SESSIONS]
    log(f"=== build_dataset: {len(sessions)} sessions targeted (of "
        f"{len(list(BIDS_ROOT.glob('sub-*/ses-*/eeg/*_task-PSG_eeg.h5')))} available) ===")

    skipped = {}
    for n, p in enumerate(sessions, 1):
        key = f"{p.parts[-4]}_{p.parts[-3]}"
        out = CACHE / f"{key}.parquet"
        if out.exists():
            continue
        t0 = time.time()
        try:
            df = process_session(p)
        except Exception as e:                                  # noqa: BLE001
            skipped[key] = f"error: {type(e).__name__}: {e}"
            log(f"[{n}/{len(sessions)}] {key} SKIPPED {skipped[key]}")
            continue
        if df is None or len(df) == 0:
            skipped[key] = "no usable arousal onsets / too few R-peaks"
            log(f"[{n}/{len(sessions)}] {key} SKIPPED {skipped[key]}")
            continue
        df.to_parquet(out, index=False)
        log(f"[{n}/{len(sessions)}] {key} {len(df)} events, "
            f"{df['n_rr_valid'].iloc[0]}/{df['n_rr_total'].iloc[0]} usable RR "
            f"({time.time() - t0:.1f}s)")

    parts = [pd.read_parquet(f) for f in sorted(CACHE.glob("*.parquet"))]
    if not parts:
        log("no cached sessions; nothing to write")
        sys.exit(1)
    ev = pd.concat(parts, ignore_index=True)

    demo = pd.read_parquet(DEMO_PARQUET)[["patient_id", "sex"]].drop_duplicates("patient_id")
    demo["patient_id"] = demo["patient_id"].astype(str)
    ev = ev.merge(demo, on="patient_id", how="left")

    ev.to_parquet(EVENTS_PARQUET, index=False)
    ses = (ev.groupby(["subject", "session"])
             .agg(n_events=("phi", "size"), n_rr_valid=("n_rr_valid", "first"),
                  n_rr_total=("n_rr_total", "first"), age_years=("age_years", "first"),
                  recording_hours=("recording_hours", "first"), sex=("sex", "first"))
             .reset_index())
    ses.to_parquet(SESSIONS_PARQUET, index=False)

    log(f"WROTE {EVENTS_PARQUET.name}: {len(ev):,} events, {ses['subject'].nunique():,} subjects, "
        f"{len(ses):,} sessions; sex known for {ev['sex'].notna().sum():,} events")
    log("stage counts: " + str(ev["stage"].value_counts().to_dict()))
    if skipped:
        log(f"skipped {len(skipped)} sessions this run: "
            + str(pd.Series(list(skipped.values())).value_counts().to_dict()))


if __name__ == "__main__":
    main()
