#!/usr/bin/env python3
"""Benchmark ECG-free CFA cleaning variants on held-out patients.

Same data, split and scoring as make_cleaning_demo.py (ECG used only to score).
Variants (no ECG at cleaning time):
  A pop1     project out the population CFA pattern (current method)
  B pop2     project out the top-2 population CFA subspace
  C pica     per-patient periodic component (heart-rate lag estimated from EEG),
             picked by similarity to the population pattern, projected out
  D hybrid   population filter -> cardiac source; beats detected on it; only the
             beat-locked template of that source is subtracted (spares
             non-cardiac activity in that spatial dimension)
  E selfcal  beats from D -> patient's own pattern from the EEG -> projected out
Reference: own   patient's own pattern from true ECG R-peaks (upper bound)

Writes benchmark_cleaning.parquet and prints a summary table.
Run: venv/bin/python "paper utils/Paper CFA/benchmark_cleaning.py"
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

import make_cleaning_demo as cd
from ica_ecg_component_variance import HEP_TMIN, epoch_around_r_peaks

HERE = cd.HERE
TPL_PRE, TPL_POST = 0.25, 0.60


def acf(x):
    x = x - x.mean()
    f = np.fft.rfft(x, 2 * len(x))
    a = np.fft.irfft(f * np.conj(f))[: len(x)]
    return a / a[0]


def heart_lag(s, sf):
    """Dominant heartbeat period (samples) from the envelope of a cardiac source."""
    env = np.abs(sosfiltfilt(butter(3, [5, 25], btype="bandpass", fs=sf, output="sos"), s))
    a = acf(env)
    lo, hi = int(0.33 * sf), int(1.7 * sf)
    return lo + int(np.argmax(a[lo:hi]))


def detect(s, sf, lag):
    """Beats on a (spatially filtered) cardiac source, refined by a matched template."""
    x = sosfiltfilt(butter(3, [5, 25], btype="bandpass", fs=sf, output="sos"), s)
    env = np.abs(x)
    mad = np.median(np.abs(env - np.median(env)))
    pk, _ = find_peaks(env, distance=int(0.6 * lag), prominence=3 * mad)
    if len(pk) < 20:
        return pk
    # matched-filter refinement on the broadband source
    tpl = epoch_around_r_peaks(s, pk, sf, -0.1, 0.3).mean(0)
    mf = np.convolve(s - s.mean(), (tpl - tpl.mean())[::-1], "same")
    mad = np.median(np.abs(mf - np.median(mf)))
    pk2, _ = find_peaks(mf, distance=int(0.6 * lag), prominence=2 * mad)
    return pk2


def template_subtract(s, peaks, sf):
    pre, post = int(TPL_PRE * sf), int(TPL_POST * sf)
    tpl = epoch_around_r_peaks(s, peaks, sf, -TPL_PRE, TPL_POST).mean(0)
    out = np.zeros_like(s)
    bounds = np.r_[0, (peaks[1:] + peaks[:-1]) // 2, len(s)]
    for i, p in enumerate(peaks):
        a, b = max(p - pre, bounds[i]), min(p + post, bounds[i + 1])
        if b > a:
            out[a:b] = tpl[a - (p - pre): b - (p - pre)]
    return out  # cardiac part of s


def pica_pattern(eeg, sf, lag, u_pop):
    """Periodic component analysis: filters maximising lag-T autocorrelation."""
    x = eeg - eeg.mean(1, keepdims=True)
    C0 = x @ x.T / x.shape[1]
    CT = x[:, :-lag] @ x[:, lag:].T / (x.shape[1] - lag)
    CT = (CT + CT.T) / 2
    vals, W = np.linalg.eig(np.linalg.solve(C0, CT))
    W = np.real(W[:, np.argsort(-np.real(vals))[:3]])
    A = C0 @ W  # forward patterns
    A = A / np.linalg.norm(A, axis=0)
    k = int(np.argmax(np.abs(u_pop @ A)))
    return A[:, k]


def run(args):
    rec, u1, U2, idx = args
    pid, eeg, ecg, peaks, sf = rec
    tt = cd.epoch_times(sf)
    out = {"patient_id": pid}
    wave = -cd.INJECT_UV * np.exp(-0.5 * ((tt - cd.INJECT_LAT) / cd.INJECT_SD) ** 2)
    inj = np.zeros_like(eeg)
    for p in peaks:
        a = p - int(round(-HEP_TMIN * sf))
        if a >= 0 and a + len(tt) <= eeg.shape[1]:
            inj[:, a:a + len(tt)] += cd.INJECT_W[:, None] * wave
    target = cd.INJECT_W[:, None] * wave
    f, p0 = welch(eeg, sf, nperseg=int(4 * sf), axis=1)
    band = (f >= 1) & (f <= 40)

    def hybrid(x, u):
        s = u @ x
        lag = heart_lag(s, sf)
        pk = detect(s, sf, lag)
        return x - np.outer(u, template_subtract(s, pk, sf)), pk, lag

    def selfcal(x, u):
        _, pk, _ = hybrid(x, u)
        return cd.project_out(x, cd.cfa_pattern(x, pk, sf)) if len(pk) >= 20 else x

    methods = {
        "pop1": lambda x: cd.project_out(x, u1),
        "pop2": lambda x: x - U2 @ (U2.T @ x),
        "pica": lambda x: cd.project_out(x, pica_pattern(x, sf, heart_lag(u1 @ x, sf), u1)),
        "hybrid": lambda x: hybrid(x, u1)[0],
        "selfcal": lambda x: selfcal(x, u1),
        "own": lambda x: cd.project_out(x, cd.cfa_pattern(eeg, peaks, sf)),
    }
    out["r2_pre"] = cd.cfa_r2(eeg, ecg, peaks, sf)[0].mean()
    _, pk, _ = hybrid(eeg, u1)
    out["beat_sens"] = cd_match(peaks, pk, sf)
    for name, fn in methods.items():
        try:
            c = fn(eeg)
            ci = fn(eeg + inj)
            out[f"r2_{name}"] = cd.cfa_r2(c, ecg, peaks, sf)[0].mean()
            rec_inj = epoch_around_r_peaks(ci - c, peaks, sf).mean(0)
            out[f"hep_{name}"] = float(np.sum(rec_inj * target) / np.sum(target * target))
            _, p1 = welch(c, sf, nperseg=int(4 * sf), axis=1)
            out[f"psd_{name}"] = float(np.median(10 * np.log10(p1[:, band] / p0[:, band])))
        except Exception:  # noqa: BLE001
            out[f"r2_{name}"] = np.nan
    return out


def cd_match(true_peaks, found, sf, tol=0.05):
    if len(found) < 2:
        return 0.0
    idx = np.clip(np.searchsorted(found, true_peaks), 1, len(found) - 1)
    near = np.where(np.abs(found[idx] - true_peaks) < np.abs(found[idx - 1] - true_peaks), found[idx], found[idx - 1])
    off = np.median(near - true_peaks)
    return float(np.mean(np.abs(near - true_peaks - off) / sf <= tol))


def main():
    cfa = pd.read_parquet(os.path.join(HERE, "cfa_combined.parquet"),
                          columns=["patient_id", "edf_path", "window_start_s"]).drop_duplicates("patient_id")
    cfa = cfa[cfa.patient_id.str.startswith("I0")]
    cfa["bdsp_patient_id"] = pd.to_numeric(cfa.patient_id.str.extract(r"I\d{4}(\d{9})", expand=False))
    meta = cfa.merge(pd.read_parquet(os.path.join(HERE, "bmi_combined.parquet")), on="bdsp_patient_id").merge(
        pd.read_parquet(os.path.join(HERE, "demographics_combined.parquet"))[["patient_id", "sex"]], on="patient_id")
    meta = meta[meta.sex.isin(["Male", "Female"])]
    sample = meta[meta.patient_id != cd.EXAMPLE].sample(cd.N_LOAD, random_state=cd.SEED)
    with ProcessPoolExecutor(48) as ex:
        loaded = [r for r in ex.map(cd.load, [(r.patient_id, r.edf_path, float(r.window_start_s))
                                              for r in sample.itertuples()]) if r]
    train, test = loaded[: cd.N_TRAIN], loaded[cd.N_TRAIN:]
    u1 = cd.mean_pattern([cd.cfa_pattern(r[1], r[3], r[4]) for r in train])
    C = np.zeros((6, 6))
    for r in train:
        ev = epoch_around_r_peaks(r[1], r[3], r[4]).mean(0)
        ev = ev - ev.mean(1, keepdims=True)
        C += ev @ ev.T / np.sum(ev ** 2)
    U2 = np.linalg.eigh(C)[1][:, ::-1][:, :2]
    with ProcessPoolExecutor(48) as ex:
        rows = list(ex.map(run, [(r, u1, U2, i) for i, r in enumerate(test)]))
    res = pd.DataFrame(rows)
    res.to_parquet(os.path.join(HERE, "benchmark_cleaning.parquet"))
    names = ["pop1", "pop2", "pica", "hybrid", "selfcal", "own"]
    summary = pd.DataFrame({
        "mean_r2": [res[f"r2_{n}"].mean() for n in names],
        "median_r2": [res[f"r2_{n}"].median() for n in names],
        "pct_pts_improved_vs_pop1": [(res[f"r2_{n}"] < res["r2_pop1"]).mean() * 100 for n in names],
        "hep_retained_median": [res[f"hep_{n}"].median() for n in names],
        "psd_db_median": [res[f"psd_{n}"].median() for n in names],
    }, index=names)
    print(f"n test = {len(res)}; r2_pre mean = {res.r2_pre.mean():.3f}; "
          f"beat detection on cardiac source, median sensitivity = {res.beat_sens.median():.2f}")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
