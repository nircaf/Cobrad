#!/usr/bin/env python3
"""Per-patient HEP features for every six-channel HSP patient (cleaning_demo.parquet), raw vs. CFAR vs. ps-CFAR,
plus the ECG average's own features (what the heart alone predicts).

HEP = R-peak-locked average, -0.2..0.6 s; features = mean in 50-ms bins from 0 to 600 ms, per channel (12 x 6 = 72).
CFAR uses the whole-cohort pattern u_pop (each patient is ~1/6,700 of its training data; leakage negligible).
Writes hep_features_all.parquet. Run:
  venv/bin/python "paper utils/Paper CFA/hep_features_all.py" --workers 64
"""
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import make_cleaning_demo as cd
from ica_ecg_component_variance import epoch_around_r_peaks

HERE = cd.HERE
U_POP = np.array([json.load(open(os.path.join(HERE, "paper_stats.json")))["cleaning_demo"]["u_pop"][s] for s in cd.SITES])
EDGES = np.arange(0, 0.601, 0.05)


def feats(ev, tt):
    return np.stack([ev[..., (tt >= a) & (tt < b)].mean(-1) for a, b in zip(EDGES[:-1], EDGES[1:])], -1).ravel()


def job(j):
    rec = cd.load(j)
    if rec is None:
        return None
    pid, eeg, ecg, peaks, sf = rec
    tt = np.arange(int(round(-0.2 * sf)), int(round(0.6 * sf))) / sf
    out = {"patient_id": pid, "hr_bpm": float(60 * sf / np.median(np.diff(peaks)))}
    for m, x in [("raw", eeg), ("cfar", cd.project_out(eeg, U_POP)), ("own", cd.project_out(eeg, cd.cfa_pattern(eeg, peaks, sf)))]:
        out[f"f_{m}"] = feats(epoch_around_r_peaks(x, peaks, sf, -0.2, 0.6).mean(0), tt).astype(np.float32).tolist()
    out["f_ecg"] = feats(epoch_around_r_peaks(ecg[None], peaks, sf, -0.2, 0.6).mean(0), tt).astype(np.float32).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()
    ids = set(pd.read_parquet(os.path.join(HERE, "cleaning_demo.parquet")).patient_id)
    c = pd.read_parquet(os.path.join(HERE, "cfa_combined.parquet"), columns=["patient_id", "edf_path", "window_start_s"]).drop_duplicates("patient_id")
    c = c[c.patient_id.isin(ids)]
    jobs = [(r.patient_id, r.edf_path, float(r.window_start_s)) for r in c.itertuples()]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(job, jobs, chunksize=4), 1):
            if r:
                rows.append(r)
            if k % 500 == 0:
                print(k, len(rows), flush=True)
    pd.DataFrame(rows).to_parquet(os.path.join(HERE, "hep_features_all.parquet"))
    print("n =", len(rows))


if __name__ == "__main__":
    main()
