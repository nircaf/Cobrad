#!/usr/bin/env python3
"""HEP after cleaning: patient-specific CFAR (ps-CFAR) vs. ECG-informed ICA vs. cohort CFAR vs. raw.

Per patient (main-analysis window, 6-site montage), HEP = R-peak-locked average, -0.2..0.6 s (the
window HEP studies analyse). Saved per method: the 6-channel HEP, the ECG average, and a synthetic
neural HEP's recovery (Gaussian, 300 ms, frontal > occipital, as make_cleaning_demo) when each method
is re-fitted on the injected data. Group statistics are in make_hep_cleaning_figure.py.

Writes hep_cleaning_compare.parquet. Run:
  venv/bin/python "paper utils/Paper CFA/hep_cleaning_compare.py" --limit 500 --workers 48
"""
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import make_cleaning_demo as cd
from ica_ecg_component_variance import epoch_around_r_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
TMIN, TMAX = -0.2, 0.6
U_POP = np.array([json.load(open(os.path.join(HERE, "paper_stats.json")))["cleaning_demo"]["u_pop"][s] for s in cd.SITES])


def hep(x, peaks, sf):
    return epoch_around_r_peaks(x, peaks, sf, TMIN, TMAX).mean(0)


def clean(m, eeg, ecg, peaks, sf):
    if m == "raw":
        return eeg
    if m == "ica":
        return cd.ica_clean(eeg, ecg, sf)
    if m == "cfar":
        return cd.project_out(eeg, U_POP)
    return cd.project_out(eeg, cd.cfa_pattern(eeg, peaks, sf))  # own


def job(j):
    rec = cd.load(j)
    if rec is None:
        return None
    pid, eeg, ecg, peaks, sf = rec
    tt = np.arange(int(round(TMIN * sf)), int(round(TMAX * sf))) / sf
    wave = -cd.INJECT_UV * np.exp(-0.5 * ((tt - cd.INJECT_LAT) / cd.INJECT_SD) ** 2)
    target = cd.INJECT_W[:, None] * wave
    inj = np.zeros_like(eeg)
    a0 = int(round(-TMIN * sf))
    for p in peaks:
        if p - a0 >= 0 and p - a0 + len(tt) <= eeg.shape[1]:
            inj[:, p - a0:p - a0 + len(tt)] += target
    out = {"patient_id": pid, "sfreq": sf, "hr_bpm": 60 * sf / np.median(np.diff(peaks)), "n_beats": len(peaks),
           "tt": tt.tolist(), "ecg": hep(ecg[None], peaks, sf)[0].tolist()}
    for m in ["raw", "ica", "cfar", "own"]:
        h = hep(clean(m, eeg, ecg, peaks, sf), peaks, sf)
        rec_inj = hep(clean(m, eeg + inj, ecg, peaks, sf), peaks, sf) - h
        out[f"hep_{m}"] = h.ravel().tolist()
        out[f"inj_kept_{m}"] = float((rec_inj * target).sum() / (target * target).sum())
        out[f"inj_r_{m}"] = float(np.corrcoef(rec_inj.ravel(), target.ravel())[0, 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    cfa = pd.read_parquet(os.path.join(HERE, "cfa_combined.parquet"), columns=["patient_id", "edf_path", "window_start_s"])
    cfa = cfa[cfa.patient_id.str.startswith("I0")].drop_duplicates("patient_id").sample(frac=1, random_state=cd.SEED + 7)
    jobs = [(r.patient_id, r.edf_path, float(r.window_start_s)) for r in cfa.itertuples()]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(job, jobs[:a.limit * 3]):
            if r:
                rows.append(r)
                if len(rows) % 50 == 0:
                    print(len(rows), flush=True)
            if len(rows) >= a.limit:
                break
    pd.DataFrame(rows).to_parquet(os.path.join(HERE, "hep_cleaning_compare.parquet"))
    print(f"n = {len(rows)}")


if __name__ == "__main__":
    main()
