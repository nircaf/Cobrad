#!/usr/bin/env python3
"""Per-patient PSD (dB) for Figure S4 b,c: raw EEG (continuous 10-min), HEP (R-peak-locked
average of raw EEG) and patient-specific CFAR (continuous EEG after projecting out the
patient's own ECG-derived spatial pattern). 6-site montage patients only (make_cleaning_demo.load).

Run: venv/bin/python "paper utils/Paper CFA/make_psd_raw_hep_cfar.py" [--limit 300]
"""
import argparse, os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.signal import welch

import make_cleaning_demo as cd
from ica_ecg_component_variance import epoch_around_r_peaks

HERE = os.path.dirname(os.path.abspath(__file__))


def db(x, sfreq, nperseg):
    f, p = welch(x, sfreq, nperseg=nperseg)
    return f, 10 * np.log10(np.clip(p, 1e-12, None))


def job(j):
    r = cd.load(j)
    if r is None:
        return []
    pid, eeg, ecg, peaks, sfreq = r
    hep = epoch_around_r_peaks(eeg, peaks, sfreq).mean(0)
    clean = cd.project_out(eeg, cd.cfa_pattern(eeg, peaks, sfreq))
    ica = cd.ica_clean(eeg, ecg, sfreq)
    rows = []
    for i, site in enumerate(cd.SITES):
        for cond, x, nper in [("raw", eeg[i], int(4 * sfreq)), ("hep", hep[i], min(hep.shape[1], 128)),
                              ("cfar", clean[i], int(4 * sfreq)), ("ica", ica[i], int(4 * sfreq))]:
            f, p = db(x, sfreq, nper)
            rows += [(pid, site, cond, a, b) for a, b in zip(f, p)]
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    a = ap.parse_args()
    d = pd.read_parquet(os.path.join(HERE, "cfa_combined.parquet"),
                        columns=["patient_id", "edf_path", "window_start_s"]).drop_duplicates("patient_id")
    d = d.sample(n=min(a.limit * 2, len(d)), random_state=42)  # oversample: non-6-site montages are skipped
    jobs = [(r.patient_id, r.edf_path, float(r.window_start_s)) for r in d.itertuples()]
    out, done = [], set()
    with ProcessPoolExecutor(cd.N_WORKERS) as ex:
        for rows in ex.map(job, jobs):
            if rows and len(done) < a.limit:
                done.add(rows[0][0])
                out += rows
    df = pd.DataFrame(out, columns=["patient_id", "site", "condition", "freq_hz", "power_db"])
    df.to_parquet(os.path.join(HERE, "psd_raw_hep_cfar.parquet"))
    print(len(df), df.patient_id.nunique(), "patients")
