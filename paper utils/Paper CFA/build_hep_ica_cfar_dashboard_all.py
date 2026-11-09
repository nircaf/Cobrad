#!/usr/bin/env python3
"""Build all-patient raw HEP / ICA / CFAR waveforms for dashboard 24.

Resumable: seeds from hep_cleaning_compare.parquet, checkpoints to
hep_ica_cfar_all.parquet, and skips completed patient IDs on every rerun.
ICA is fitted once per patient (unlike the synthetic-injection benchmark).
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import make_cleaning_demo as cd
from ica_ecg_component_variance import epoch_around_r_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, "hep_ica_cfar_all.parquet")
SEED_FILE = os.path.join(HERE, "hep_cleaning_compare.parquet")
TMIN, TMAX = -0.2, 0.6
U_POP = np.array([
    json.load(open(os.path.join(HERE, "paper_stats.json")))["cleaning_demo"]["u_pop"][site]
    for site in cd.SITES
])


def hep(signal, peaks, sfreq):
    return epoch_around_r_peaks(signal, peaks, sfreq, TMIN, TMAX).mean(0)


def process(job):
    rec = cd.load(job)
    if rec is None:
        return None
    patient_id, eeg, ecg, peaks, sfreq = rec
    try:
        ica = cd.ica_clean(eeg, ecg, sfreq)
    except Exception as exc:  # keep an auditable failed row
        return {"patient_id": patient_id, "error": f"ICA: {type(exc).__name__}: {exc}"[:500]}
    cfar = cd.project_out(eeg, U_POP)
    times = np.arange(int(round(TMIN * sfreq)), int(round(TMAX * sfreq))) / sfreq
    return {
        "patient_id": patient_id,
        "sfreq": float(sfreq),
        "hr_bpm": float(60 * sfreq / np.median(np.diff(peaks))),
        "n_beats": int(len(peaks)),
        "tt": times.astype(np.float32).tolist(),
        "ecg": hep(ecg[None], peaks, sfreq)[0].astype(np.float32).tolist(),
        "hep_raw": hep(eeg, peaks, sfreq).astype(np.float32).ravel().tolist(),
        "hep_ica": hep(ica, peaks, sfreq).astype(np.float32).ravel().tolist(),
        "hep_cfar": hep(cfar, peaks, sfreq).astype(np.float32).ravel().tolist(),
        "error": None,
    }


def write_checkpoint(rows):
    frame = pd.DataFrame(rows).drop_duplicates("patient_id", keep="last")
    tmp = OUTPUT + ".tmp.parquet"
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, OUTPUT)
    ok = frame.hep_ica.notna().sum() if "hep_ica" in frame else 0
    print(f"checkpoint: {len(frame):,} attempted, {ok:,} complete", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    args = parser.parse_args()

    rows = []
    if os.path.exists(OUTPUT):
        rows = pd.read_parquet(OUTPUT).to_dict("records")
    elif os.path.exists(SEED_FILE):
        seed = pd.read_parquet(SEED_FILE)
        keep = ["patient_id", "sfreq", "hr_bpm", "n_beats", "tt", "ecg",
                "hep_raw", "hep_ica", "hep_cfar"]
        seed = seed[keep].copy()
        seed["error"] = None
        rows = seed.to_dict("records")

    completed = {str(row["patient_id"]) for row in rows}
    cfa = pd.read_parquet(
        os.path.join(HERE, "cfa_combined.parquet"),
        columns=["patient_id", "edf_path", "window_start_s"],
    ).drop_duplicates("patient_id")
    cfa = cfa[cfa.patient_id.astype(str).str.startswith("I0")]
    jobs = [(r.patient_id, r.edf_path, float(r.window_start_s)) for r in cfa.itertuples()
            if str(r.patient_id) not in completed]
    print(f"seeded {len(rows):,}; remaining {len(jobs):,}; workers {args.workers}", flush=True)
    if not jobs:
        write_checkpoint(rows)
        return

    with ProcessPoolExecutor(args.workers) as executor:
        futures = [executor.submit(process, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result is not None:
                rows.append(result)
            if index % args.checkpoint_every == 0:
                write_checkpoint(rows)
    write_checkpoint(rows)


if __name__ == "__main__":
    main()
