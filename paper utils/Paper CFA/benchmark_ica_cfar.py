#!/usr/bin/env python3
"""Benchmark CFA cleaning pipelines, incl. ICA and ICA+CFAR combinations, on held-out patients.

Same loading/scoring as make_cleaning_demo.py. Patterns are learned on a training half and
applied to the held-out half. Pipelines:
  raw            no cleaning
  ica            ECG-informed ICA (Section 2.4)
  ica_iclabel    standard ECG-free ICA: components labelled "heart beat" by ICLabel removed
  cfar           CFAR on raw EEG, filter trained on raw EEG (ECG-free)
  ica_cfar_raw   ICA, then the raw-trained CFAR filter
  ica_cfar_ica   ICA, then a CFAR filter trained on post-ICA EEG
  cfar_ica       CFAR, then ICA fitted on the CFAR output
  ecgreg         per-channel regression on the ECG at short lags (+-LAG_MS); uses ECG
  ica_ecgreg     ICA, then ecgreg
  cfar_ecgreg    CFAR, then ecgreg
  cfar_ecgreg_dir  "CFAR+ECG regression": only the ECG-explained part of the CFAR direction is removed
                 (spatial filter picks the cardiac direction, short-lag ECG regression its cardiac part)
  own            patient-specific CFAR: projection of the patient's own ECG-locked pattern (uses ECG)
Scores (ECG used only here): CFA R^2 (shape, as in the paper), CFA power (R^2 x evoked variance
outside QRS, uV^2: how much ECG-shaped signal is left, which R^2 alone cannot show), synthetic
HEP retained, median 1-40 Hz PSD change vs raw.

Run: venv/bin/python "paper utils/Paper CFA/benchmark_ica_cfar.py" [--only name1,name2]
     --only scores just those pipelines and merges them into the existing parquet (same split).
     --all  every eligible patient, CFAR filter cross-fitted over cd.N_FOLDS folds, only the Figure 4d
            pipelines; writes benchmark_all.parquet (two passes, so EDFs are not held in memory).
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
from scipy.signal import welch

import make_cleaning_demo as cd
from ica_ecg_component_variance import HEP_TMIN, QRS_EXCLUDE_SEC, epoch_around_r_peaks

N_LOAD, LAG_MS = 1000, 20


ica_clean = cd.ica_clean


def ecg_regress(eeg, ecg, sf):
    """Remove the part of each channel that is a short-lag linear function of the ECG."""
    L = int(round(LAG_MS / 1000 * sf))
    X = np.stack([np.roll(ecg, k) for k in range(-L, L + 1)] + [np.ones_like(ecg)], 1)
    beta, *_ = np.linalg.lstsq(X, eeg.T, rcond=None)
    return eeg - (X @ beta).T


def score(x, ecg, peaks, sf):
    r2, ev, _ = cd.cfa_r2(x, ecg, peaks, sf)
    keep = np.abs(cd.epoch_times(sf)) > QRS_EXCLUDE_SEC
    return r2.mean(), float(np.mean(r2 * ev[:, keep].var(1)))


def fit_job(job, with_ica=True):
    rec = cd.load(job)
    if rec is None:
        return None
    pid, raw, ecg, peaks, sf = rec
    u_ica = cd.cfa_pattern(cd.ica_clean(raw, ecg, sf), peaks, sf) if with_ica else None
    return rec, cd.cfa_pattern(raw, peaks, sf), u_ica


def cfar_ecg_regress(x, u, ecg, sf):
    """Remove from x only the ECG-explained part of its CFAR-direction source u.x."""
    s = u @ x
    return x - np.outer(u, s - ecg_regress(s[None], ecg, sf)[0])


def run(args, extra=None):
    rec, u_raw, u_ica, only = args
    pid, raw, ecg, peaks, sf = rec
    tt = cd.epoch_times(sf)
    wave = -cd.INJECT_UV * np.exp(-0.5 * ((tt - cd.INJECT_LAT) / cd.INJECT_SD) ** 2)
    inj = np.zeros_like(raw)
    for p in peaks:
        a = p - int(round(-HEP_TMIN * sf))
        if a >= 0 and a + len(tt) <= raw.shape[1]:
            inj[:, a:a + len(tt)] += cd.INJECT_W[:, None] * wave
    target = cd.INJECT_W[:, None] * wave
    f, p0 = welch(raw, sf, nperseg=int(4 * sf), axis=1)
    band = (f >= 1) & (f <= 40)
    P = cd.project_out
    pipes = {
        "raw": lambda x: x,
        "ica": lambda x: ica_clean(x, ecg, sf),
        "ica_iclabel": lambda x: cd.ica_iclabel(x, sf),
        "cfar": lambda x: P(x, u_raw),
        "ica_cfar_raw": lambda x: P(ica_clean(x, ecg, sf), u_raw),
        "ica_cfar_ica": lambda x: P(ica_clean(x, ecg, sf), u_ica),
        "cfar_ica": lambda x: ica_clean(P(x, u_raw), ecg, sf),
        "ecgreg": lambda x: ecg_regress(x, ecg, sf),
        "ica_ecgreg": lambda x: ecg_regress(ica_clean(x, ecg, sf), ecg, sf),
        "cfar_ecgreg": lambda x: ecg_regress(P(x, u_raw), ecg, sf),
        "cfar_ecgreg_dir": lambda x: cfar_ecg_regress(x, u_raw, ecg, sf),
        "own": lambda x: P(x, cd.cfa_pattern(raw, peaks, sf)),
    }
    if only:
        pipes = {k: v for k, v in pipes.items() if k in only}
    pipes.update(extra or {})  # e.g. learned cleaners from dl_cfar.py
    out = {"patient_id": pid}
    for name, fn in pipes.items():
        try:
            c, ci = fn(raw), fn(raw + inj)
            out[f"r2_{name}"], out[f"pow_{name}"] = score(c, ecg, peaks, sf)
            rec_inj = epoch_around_r_peaks(ci - c, peaks, sf).mean(0)
            out[f"hep_{name}"] = float(np.sum(rec_inj * target) / np.sum(target * target))
            _, p1 = welch(c, sf, nperseg=int(4 * sf), axis=1)
            out[f"psd_{name}"] = float(np.median(10 * np.log10(p1[:, band] / p0[:, band])))
        except Exception:  # noqa: BLE001 -- e.g. ICA failure on one patient
            pass
    return out


def main(only=None):
    cfa = pd.read_parquet(os.path.join(cd.HERE, "cfa_combined.parquet"),
                          columns=["patient_id", "edf_path", "window_start_s"]).drop_duplicates("patient_id")
    cfa = cfa[cfa.patient_id.str.startswith("I0") & (cfa.patient_id != cd.EXAMPLE)].sample(N_LOAD, random_state=cd.SEED)
    with ProcessPoolExecutor(cd.N_WORKERS) as ex:
        fitted = [r for r in ex.map(partial(fit_job, with_ica=not only), [(r.patient_id, r.edf_path, float(r.window_start_s))
                                              for r in cfa.itertuples()], chunksize=4) if r]
    half = len(fitted) // 2
    train, test = fitted[:half], fitted[half:]
    u_raw = cd.mean_pattern([r[1] for r in train])
    u_ica = None if only else cd.mean_pattern([r[2] for r in train])
    print("u_raw", np.round(u_raw, 2))
    with ProcessPoolExecutor(cd.N_WORKERS) as ex:
        res = pd.DataFrame(list(ex.map(run, [(r[0], u_raw, u_ica, only) for r in test])))
    out = os.path.join(cd.HERE, "benchmark_ica_cfar.parquet")
    if only:  # same sample/split as the full run, so merge new columns onto it
        old = pd.read_parquet(out)
        assert set(res.patient_id) == set(old.patient_id), "test split differs from the saved run"
        res = old.drop(columns=[c for c in res.columns if c != "patient_id" and c in old], errors="ignore").merge(res, on="patient_id")
    res.to_parquet(out)
    names = [c[3:] for c in res.columns if c.startswith("r2_")]
    print(f"n test = {len(res)}")
    print(pd.DataFrame({
        "mean_r2": [res[f"r2_{n}"].mean() for n in names],
        "median_cfa_pow_uV2": [res[f"pow_{n}"].median() for n in names],
        "pct_pow_vs_raw": [(res[f"pow_{n}"] / res.pow_raw).median() * 100 for n in names],
        "hep_retained": [res[f"hep_{n}"].median() for n in names],
        "psd_db": [res[f"psd_{n}"].median() for n in names],
    }, index=names).round(3).to_string())


FIG4D = ["raw", "ica", "ecgreg", "cfar", "cfar_ica", "own"]


def pattern_job(job):
    rec = cd.load(job)
    return None if rec is None else (job, cd.cfa_pattern(rec[1], rec[3], rec[4]))


def run_job(args):
    job, u, only = args
    rec = cd.load(job)
    return None if rec is None else run((rec, u, None, only))


def main_all(only=None):
    from sklearn.model_selection import KFold
    cfa = pd.read_parquet(os.path.join(cd.HERE, "cfa_combined.parquet"),
                          columns=["patient_id", "edf_path", "window_start_s"]).drop_duplicates("patient_id")
    cfa = cfa[cfa.patient_id.str.startswith("I0")]
    with ProcessPoolExecutor(cd.N_WORKERS) as ex:
        fitted = [r for r in ex.map(pattern_job, [(r.patient_id, r.edf_path, float(r.window_start_s))
                                                  for r in cfa.itertuples()], chunksize=8) if r]
    P = np.array([f[1] for f in fitted])
    u_of = {}
    for tr, te in KFold(cd.N_FOLDS, shuffle=True, random_state=cd.SEED).split(P):
        u = cd.mean_pattern(P[tr])
        for i in te:
            u_of[i] = u
    with ProcessPoolExecutor(cd.N_WORKERS) as ex:
        rows = [r for r in ex.map(run_job, [(f[0], u_of[i], only or FIG4D) for i, f in enumerate(fitted)], chunksize=4) if r]
    res = pd.DataFrame(rows)
    out = os.path.join(cd.HERE, "benchmark_all.parquet")
    if only:  # same patients and folds as the saved run: merge the new pipelines onto it
        old = pd.read_parquet(out)
        res = old.drop(columns=[c for c in res.columns if c != "patient_id" and c in old]).merge(res, on="patient_id", how="left")
    res.to_parquet(out)
    print(f"n = {len(res)}")
    names = [c[4:] for c in res.columns if c.startswith("pow_")]
    print(pd.DataFrame({"pct_pow_vs_raw": [(res[f"pow_{n}"] / res.pow_raw).median() * 100 for n in names],
                        "hep_retained": [res[f"hep_{n}"].median() for n in names],
                        "psd_db": [res[f"psd_{n}"].median() for n in names]}, index=names).round(3).to_string())


if __name__ == "__main__":
    if "--all" in sys.argv:
        main_all(sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None)
        sys.exit()
    main(sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None)
