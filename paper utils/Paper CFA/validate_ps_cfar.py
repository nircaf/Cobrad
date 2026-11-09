#!/usr/bin/env python3
"""Validation of patient-specific CFAR (ps-CFAR): does it remove the heart, and only the heart?

Every patient's main-analysis window is split in half. The patient's own ECG-derived pattern is
learned on half A and every score is computed on half B (held out), so nothing is graded on the
data it was fitted to.

  1  cross-half: gain (neural kept / cardiac left, as Figure 4e) on B with the pattern from A vs. from B
  2a ECG reconstruction: lagged linear map EEG -> ECG fitted on A, R² on B, before vs. after cleaning
  2b EEG-ECG coherence spectrum on B, before vs. after
  3a fake R-peaks: heartbeat-locked variance on B for real vs. random event times, before vs. after
  3b pattern learned from random event times instead of R-peaks (should remove nothing cardiac)
  4  injected artifact: known pattern x patient's ECG template added to A and B; fraction removed on B
  5  brain features on B: band power change, spindle density (YASA), before vs. after
  6a beat-to-beat: per-beat CFA amplitude vs. QRS amplitude correlation, before vs. after
  6b premature beats (RR < 80% of median): cardiac power left on premature vs. normal beats

Writes validate_ps_cfar.parquet. Run:
  venv/bin/python "paper utils/Paper CFA/validate_ps_cfar.py" [--limit 300 --workers 48]
"""
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.signal import coherence, welch

import make_cleaning_demo as cd
from ica_ecg_component_variance import QRS_EXCLUDE_SEC, epoch_around_r_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
U_POP = np.array([json.load(open(os.path.join(HERE, "paper_stats.json")))["cleaning_demo"]["u_pop"][s] for s in cd.SITES])
F_GRID = np.arange(0.5, 40.01, 0.25)
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "sigma": (12, 16), "beta": (16, 30)}
LAGS = range(-8, 9, 2)  # samples; ponytail: fixed +-8 samples, fine at 200-256 Hz
INJECT_REL = 0.3  # injected artifact RMS relative to EEG RMS


def hep_kept(u):
    """Synthetic neural HEP surviving projection (exact for a projection, as in make_cleaning_demo)."""
    w = cd.INJECT_W
    return 1 - (w @ u) ** 2 / (w @ w)


def lagged(eeg):
    return np.vstack([np.roll(eeg, k, axis=1) for k in LAGS] + [np.ones((1, eeg.shape[1]))])


def ecg_recon_r2(eeg_a, ecg_a, eeg_b, ecg_b):
    w, *_ = np.linalg.lstsq(lagged(eeg_a).T, ecg_a, rcond=None)
    pred = lagged(eeg_b).T @ w
    return float(1 - np.var(ecg_b - pred) / np.var(ecg_b))


def overlay(template, peaks, n, a0):
    out = np.zeros((template.shape[0], n))
    for p in peaks:
        a = p - a0
        if a >= 0 and a + template.shape[1] <= n:
            out[:, a:a + template.shape[1]] += template
    return out


def job(j):
    rec = cd.load(j)
    if rec is None:
        return None
    pid, eeg, ecg, peaks, sf = rec
    tt = cd.epoch_times(sf)
    keep = np.abs(tt) > QRS_EXCLUDE_SEC
    a0 = int(round(-tt[0] * sf))
    h = eeg.shape[1] // 2
    A, B, eA, eB = eeg[:, :h], eeg[:, h:], ecg[:h], ecg[h:]
    pA, pB = peaks[peaks < h], peaks[peaks >= h] - h
    if len(pA) < 50 or len(pB) < 50:
        return None
    rng = np.random.default_rng(abs(hash(pid)) % 2**32)
    fake = lambda n, k: np.sort(rng.integers(a0 + 1, n - len(tt) - 1, k))

    u_a, u_b = cd.cfa_pattern(A, pA, sf), cd.cfa_pattern(B, pB, sf)
    u_fake = cd.cfa_pattern(A, fake(h, len(pA)), sf)
    pow_raw = cd.cfa_power(B, eB, pB, sf)
    out = {"patient_id": pid, "sfreq": sf, "hr_bpm": 60 * sf / np.median(np.diff(peaks)),
           "r2_raw": float(cd.cfa_r2(B, eB, pB, sf)[0].mean()), "pow_raw": pow_raw}
    # 1 + 3b: gain on held-out half B
    for k, u in {"in": u_b, "cross": u_a, "pop": U_POP, "fake": u_fake}.items():
        p = cd.cfa_power(cd.project_out(B, u), eB, pB, sf)
        out[f"pow_{k}"], out[f"hep_{k}"], out[f"gain_{k}"] = p, hep_kept(u), hep_kept(u) * pow_raw / p
    out["cos_ab"] = float(abs(u_a @ u_b))
    cB = cd.project_out(B, u_a)
    # 2a
    out["ecg_r2_raw"] = ecg_recon_r2(A, eA, B, eB)
    out["ecg_r2_clean"] = ecg_recon_r2(cd.project_out(A, u_a), eA, cB, eB)
    # 2b: channel-mean magnitude-squared coherence with the ECG
    for k, x in {"raw": B, "clean": cB}.items():
        f, c = coherence(x, eB[None], sf, nperseg=int(8 * sf), axis=1)
        out[f"coh_{k}"] = np.interp(F_GRID, f, c.mean(0)).tolist()
    # 3a: heartbeat-locked variance outside QRS, real vs. fake events
    fk = fake(B.shape[1], len(pB))
    hv = lambda x, pk: float(epoch_around_r_peaks(x, pk, sf).mean(0)[:, keep].var(1).mean())
    for k, x in {"raw": B, "clean": cB}.items():
        out[f"hv_real_{k}"], out[f"hv_fake_{k}"] = hv(x, pB), hv(x, fk)
    # 4: inject U_POP x own ECG template; learn on injected A, score on injected B
    ecg_t = epoch_around_r_peaks(ecg[None], peaks, sf).mean(0)
    ecg_t = ecg_t - ecg_t.mean()
    artA, artB = (np.outer(U_POP, overlay(ecg_t, pk, n, a0)[0]) for pk, n in [(pA, h), (pB, B.shape[1])])
    s = INJECT_REL * B.std() / artB.std()
    u_inj = cd.cfa_pattern(A + s * artA, pA, sf)
    out["inj_left"] = float((cd.project_out(s * artB, u_inj) ** 2).sum() / ((s * artB) ** 2).sum())
    out["inj_brain_kept"] = float((cd.project_out(B, u_inj) ** 2).sum() / (B ** 2).sum())
    out["inj_cos"] = float(abs(u_inj @ U_POP))
    # 5: band power and spindles
    f, pr = welch(B, sf, nperseg=int(4 * sf), axis=1)
    _, pc = welch(cB, sf, nperseg=int(4 * sf), axis=1)
    for b, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        out[f"bp_{b}_db"] = float(np.median(10 * np.log10(pc[:, m].sum(1) / pr[:, m].sum(1))))
    try:
        import yasa
        for k, x in {"raw": B, "clean": cB}.items():
            sp = yasa.spindles_detect(x, sf, ch_names=cd.SITES, verbose="ERROR")
            out[f"spindles_{k}"] = 0 if sp is None else len(sp.summary())
    except Exception:  # noqa: BLE001 -- spindle detection is optional
        pass
    # 6a: per-beat amplitude along the raw cardiac template (learned on A) vs. QRS amplitude
    T = epoch_around_r_peaks(A, pA, sf).mean(0)[:, keep]
    qrs = np.abs(eB[pB]) if len(pB) else np.array([])
    for k, x in {"raw": B, "clean": cB}.items():
        ep = epoch_around_r_peaks(x, pB, sf)
        amp = np.einsum("bct,ct->b", ep[:, :, keep], T) / (T * T).sum()
        q = qrs[:len(amp)] if len(amp) <= len(qrs) else qrs
        out[f"beat_r_{k}"] = float(np.corrcoef(amp[:len(q)], q[:len(amp)])[0, 1])
    # 6b: premature beats
    rr = np.diff(pB)
    prem = pB[1:][rr < 0.8 * np.median(rr)]
    norm = pB[1:][rr >= 0.8 * np.median(rr)]
    out["n_premature"] = int(len(prem))
    if len(prem) >= 20:
        out["prem_left"] = cd.cfa_power(cB, eB, prem, sf) / cd.cfa_power(B, eB, prem, sf)
        out["norm_left"] = cd.cfa_power(cB, eB, norm, sf) / cd.cfa_power(B, eB, norm, sf)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    cfa = pd.read_parquet(os.path.join(HERE, "cfa_combined.parquet"), columns=["patient_id", "edf_path", "window_start_s"])
    cfa = cfa[cfa.patient_id.str.startswith("I0")].drop_duplicates("patient_id").sample(frac=1, random_state=cd.SEED)
    jobs = [(r.patient_id, r.edf_path, float(r.window_start_s)) for r in cfa.itertuples()]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(job, jobs[:a.limit * 3], chunksize=1):
            if r:
                rows.append(r)
                if len(rows) % 25 == 0:
                    print(len(rows), flush=True)
            if len(rows) >= a.limit:
                break
    d = pd.DataFrame(rows)
    d.to_parquet(os.path.join(HERE, "validate_ps_cfar.parquet"))
    print(f"n = {len(d)}")
    print(d.drop(columns=["patient_id", "coh_raw", "coh_clean"]).median().round(3).to_string())


if __name__ == "__main__":
    main()
