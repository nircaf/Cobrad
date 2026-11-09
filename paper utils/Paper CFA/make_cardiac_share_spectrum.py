#!/usr/bin/env python3
"""How much of the EEG power is heart, per frequency band: continuous EEG vs. HEP, before and after CFAR.

"Heart noise" = the ECG-shaped part of each channel: the patient's ECG average scaled per channel
(least squares outside +-50 ms QRS, the same part CFA R² measures). For the HEP (R-peak-locked
average) that part is beta * ECG average; for the continuous EEG it is that scaled ECG average
overlap-added at every beat. Per band, share = P_heart / (P_heart + P_rest), with P from the
periodogram (HEP, 0.7 s, Hann) or Welch (continuous, 4 s). CFAR = population pattern from
paper_stats.json. Main-analysis windows (make_cleaning_demo.load), 6-site montage patients.

Run: venv/bin/python "paper utils/Paper CFA/make_cardiac_share_spectrum.py" [--limit 200 --workers 12]
"""
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.signal import periodogram, welch

import make_cleaning_demo as cd
from ica_ecg_component_variance import HEP_TMIN, QRS_EXCLUDE_SEC, epoch_around_r_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
BANDS = {"delta 1-4": (1, 4), "theta 4-8": (4, 8), "alpha 8-12": (8, 12), "sigma 12-16": (12, 16),
         "beta 16-30": (16, 30), "gamma 30-40": (30, 40)}
U_POP = np.array([json.load(open(os.path.join(HERE, "paper_stats.json")))["cleaning_demo"]["u_pop"][s] for s in cd.SITES])


def shares(f, p_heart, p_rest):
    return {b: float(p_heart[:, (f >= lo) & (f < hi)].sum() / (p_heart[:, (f >= lo) & (f < hi)].sum()
                     + p_rest[:, (f >= lo) & (f < hi)].sum())) for b, (lo, hi) in BANDS.items()}


def split(eeg, ecg, peaks, sf):
    """(HEP heart part, HEP rest, continuous heart part, continuous rest) for one signal."""
    tt = cd.epoch_times(sf)
    keep = np.abs(tt) > QRS_EXCLUDE_SEC
    hep = epoch_around_r_peaks(eeg, peaks, sf).mean(0)
    e = epoch_around_r_peaks(ecg[None], peaks, sf).mean(0)[0]
    e = e - e[keep].mean()
    h0 = hep - hep[:, keep].mean(1, keepdims=True)
    beta = h0[:, keep] @ e[keep] / (e[keep] @ e[keep])
    heart_hep = np.outer(beta, e)
    card = np.zeros_like(eeg)
    a0 = int(round(-HEP_TMIN * sf))
    for p in peaks:
        a = p - a0
        if a >= 0 and a + len(e) <= eeg.shape[1]:
            card[:, a:a + len(e)] += heart_hep
    return heart_hep, h0 - heart_hep, card, eeg - card


def job(j):
    r = cd.load(j)
    if r is None:
        return None
    pid, eeg, ecg, peaks, sf = r
    out = {"patient_id": pid}
    for cond, x in [("raw", eeg), ("cfar", cd.project_out(eeg, U_POP))]:
        hh, hr, ch, cr = split(x, ecg, peaks, sf)
        f, ph = periodogram(hh, sf, window="hann", nfft=int(sf), axis=1)
        _, pr = periodogram(hr, sf, window="hann", nfft=int(sf), axis=1)
        out.update({f"hep_{cond}|{b}": v for b, v in shares(f, ph, pr).items()})
        out[f"hep_{cond}|all"] = float((hh ** 2).sum() / ((hh ** 2).sum() + (hr ** 2).sum()))
        f, ph = welch(ch, sf, nperseg=int(4 * sf), axis=1)
        _, pr = welch(cr, sf, nperseg=int(4 * sf), axis=1)
        out.update({f"eeg_{cond}|{b}": v for b, v in shares(f, ph, pr).items()})
        out[f"eeg_{cond}|all"] = float(ch.var(1).sum() / (ch.var(1).sum() + cr.var(1).sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    cfa = pd.read_parquet(os.path.join(HERE, "cfa_combined.parquet"), columns=["patient_id", "edf_path", "window_start_s"])
    cfa = cfa[cfa.patient_id.str.startswith("I0")].drop_duplicates("patient_id").sample(frac=1, random_state=cd.SEED)
    jobs = [(r.patient_id, r.edf_path, float(r.window_start_s)) for r in cfa.itertuples()]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:  # walk the shuffled list until --limit 6-site patients succeed
        for r in ex.map(job, jobs[:a.limit * 3], chunksize=2):
            if r:
                rows.append(r)
            if len(rows) >= a.limit:
                break
    d = pd.DataFrame(rows)
    d.to_parquet(os.path.join(HERE, "cardiac_share_spectrum.parquet"))
    print(f"n = {len(d)}")
    print((d.drop(columns="patient_id").median() * 100).round(1).to_string())


if __name__ == "__main__":
    main()
