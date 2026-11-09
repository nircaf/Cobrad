#!/usr/bin/env python3
"""Independent clinical cohort (Rabin Medical Center, Beilinson; COBRAD, IDs 0345-xxx): patient-specific
CFAR (ps-CFAR) vs. ECG-informed ICA, on each recording's own montage (bipolar double-banana or referential).

Same main-analysis window as cfa_combined.parquet (45 min). R-peaks: detect_r_peaks + an R-height filter
(robust_peaks); recordings without a plausible rhythm are excluded. Per patient:
  pow_*   ECG-explained EEG variance (CFA R² x heartbeat-evoked variance outside QRS, channel mean; as Figure 4d)
  hep_*   synthetic neural HEP kept (2 µV Gaussian at 300 ms, added before cleaning; each method re-fitted
          on the injected data, so ps-CFAR may absorb part of it)
  gain_*  hep kept / (pow / pow_raw), as Figure 4e
  r2_*    patient-mean CFA R²
Writes beilinson_cfar_vs_ica.parquet (+ per-patient HEPs for the figure).

Run: venv/bin/python "paper utils/Paper CFA/beilinson_cfar_vs_ica.py" --workers 39
"""
import argparse, os
from concurrent.futures import ProcessPoolExecutor

import mne
import numpy as np
import pandas as pd

import make_cleaning_demo as cd
from scipy.signal import butter, sosfiltfilt

from ica_ecg_component_variance import channel_names, detect_r_peaks, epoch_around_r_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHT = {"F": 1.0, "C": 0.7, "T": 0.5, "P": 0.3, "O": 0.3}  # synthetic HEP: frontal > central > posterior


def robust_peaks(ecg, sf):
    """detect_r_peaks, then drop peaks under half the typical R height: some Beilinson leads have T waves the
    MAD-prominence detector also counts (0345-115: 146 bpm -> ~75). Returns (peaks, ok): ok = plausible rhythm."""
    pk = detect_r_peaks(ecg, sf)
    if len(pk) < 100:
        return pk, False
    env = np.abs(sosfiltfilt(butter(3, [5, min(25.0, sf / 2 - 0.5)], btype="bandpass", fs=sf, output="sos"), ecg - np.median(ecg)))
    h = env[pk]
    pk = pk[h >= 0.5 * np.median(h[h >= np.median(h)])]
    rr = np.diff(pk) / sf
    return pk, bool(35 <= 60 / np.median(rr) <= 120 and rr.std() / rr.mean() < 0.25)


def load(job):
    pid, path, start, dur = job
    raw = mne.io.read_raw_edf(path, preload=False, encoding="latin1", verbose="ERROR")
    eeg_names, ecg_name = channel_names(raw)
    seg = raw.copy().pick(eeg_names + [ecg_name]).crop(start, start + dur, include_tmax=False)
    seg.load_data(verbose="ERROR")
    sf = float(seg.info["sfreq"])
    seg.filter(1.0, min(100.0, sf / 2 - 0.5), verbose="ERROR")
    data = seg.get_data() * 1e6
    names = [n.replace("EEG ", "").strip() for n in eeg_names]
    peaks, ok = robust_peaks(data[-1], sf)
    return pid, names, data[:-1], data[-1], peaks, sf, ok


def ica_clean(eeg, ecg, sf):
    """ECG-informed ICA (Section 2.4) on any montage."""
    n = eeg.shape[0]
    info = mne.create_info([f"e{i}" for i in range(n)] + ["ECG"], sf, ["eeg"] * n + ["ecg"])
    raw = mne.io.RawArray(np.vstack([eeg, ecg[None]]) * 1e-6, info, verbose="ERROR")
    ica = mne.preprocessing.ICA(n_components=n - 1, method="picard", fit_params={"ortho": False, "extended": True},
                                random_state=42, max_iter=500)
    ica.fit(raw, picks="eeg", verbose="ERROR")
    bad, scores = ica.find_bads_ecg(raw, ch_name="ECG", method="correlation", verbose="ERROR")
    bad = bad or [int(np.nanargmax(np.abs(scores)))]
    return ica.apply(raw, exclude=bad, verbose="ERROR").get_data(picks="eeg") * 1e6


def job(j):
    try:
        pid, names, eeg, ecg, peaks, sf, ok = load(j)
    except Exception as exc:  # noqa: BLE001 -- unreadable EDFs skipped
        return {"patient_id": j[0], "error": str(exc)[:200]}
    if not ok:
        return {"patient_id": pid, "error": "ECG quality (implausible rhythm after R-peak cleaning)"}
    tt = cd.epoch_times(sf)
    w = np.array([WEIGHT.get(n.lstrip("Fp")[:1] if n.startswith("Fp") else n[:1], 0.5) for n in names])
    w[[n.startswith("Fp") for n in names]] = 1.0
    wave = -cd.INJECT_UV * np.exp(-0.5 * ((tt - cd.INJECT_LAT) / cd.INJECT_SD) ** 2)
    target = w[:, None] * wave
    inj = np.zeros_like(eeg)
    a0 = int(round(-tt[0] * sf))
    for p in peaks:
        if p - a0 >= 0 and p - a0 + len(tt) <= eeg.shape[1]:
            inj[:, p - a0:p - a0 + len(tt)] += target
    methods = {"raw": lambda x: x, "ica": lambda x: ica_clean(x, ecg, sf),
               "own": lambda x: cd.project_out(x, cd.cfa_pattern(x, peaks, sf))}
    out = {"patient_id": pid, "montage": "bipolar" if any("-" in n for n in names) else "referential",
           "n_channels": len(names), "channels": names, "sfreq": sf, "n_beats": len(peaks)}
    hep = lambda x: epoch_around_r_peaks(x, peaks, sf).mean(0)
    for m, f in methods.items():
        x = f(eeg)
        r2, ev, ecg_ev = cd.cfa_r2(x, ecg, peaks, sf)
        out[f"r2_{m}"], out[f"pow_{m}"] = float(r2.mean()), cd.cfa_power(x, ecg, peaks, sf)
        rec = hep(f(eeg + inj)) - ev
        out[f"hep_{m}"] = float((rec * target).sum() / (target * target).sum())
        out[f"ev_{m}"] = ev.ravel().astype(np.float32).tolist()
        out[f"r2ch_{m}"] = r2.tolist()
        out["ecg_ev"] = ecg_ev.astype(np.float32).tolist()
    for m in ["ica", "own"]:
        out[f"gain_{m}"] = out[f"hep_{m}"] * out["pow_raw"] / out[f"pow_{m}"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=39)
    a = ap.parse_args()
    c = pd.read_parquet(os.path.join(HERE, "cfa_combined.parquet"), columns=["patient_id", "edf_path", "window_start_s", "window_duration_s"])
    c = c[c.patient_id.str.startswith("0345")].drop_duplicates("patient_id")
    jobs = [(r.patient_id, r.edf_path, float(r.window_start_s), float(r.window_duration_s)) for r in c.itertuples()]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(job, jobs))
    d = pd.DataFrame(rows)
    d.to_parquet(os.path.join(HERE, "beilinson_cfar_vs_ica.parquet"))
    ok = d[d.get("error").isna()] if "error" in d else d
    print(f"n = {len(ok)} / {len(d)}")
    print(ok[["r2_raw", "r2_ica", "r2_own", "hep_ica", "hep_own", "gain_ica", "gain_own"]].median().round(3).to_string())
    print((ok[["pow_ica", "pow_own"]].div(ok.pow_raw, axis=0) * 100).median().round(1).to_string())


if __name__ == "__main__":
    main()
