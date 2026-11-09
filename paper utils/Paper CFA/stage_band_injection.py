#!/usr/bin/env python3
"""Ground-truth check for stage_band_cleaning.py: inject a known heartbeat artifact, see who recovers the bands.

Pairs a LOW-CFA recipient A (bottom quartile of per-patient CFA R^2, cfa_combined.parquet) with a HIGH-CFA
donor B (top quartile, same sampling rate). B's own R-peak-locked EEG average (its real 6-channel CFA template)
is overlap-added onto A's whole night at B's R-peak times, and B's ECG stands in as the recording's ECG.
The whole-night average template is ~10x weaker than B's real artifact (beat-to-beat/position variability averages
out, and 1-40 Hz drops QRS energy), so it is scaled until its share of A's EEG variance equals B's own measured
cardiac share (cleaning_demo.parquet frac_pre, paper method; capped at MAX_FRAC).
Each cleaning method (raw = contaminated, ica, cfar, own; stage_band_cleaning.clean_all) is applied to the
contaminated night; "truth" is A's original night. Per-stage spectra/features are computed exactly as in
stage_band_cleaning.py (A's YASA and expert staging), so the dashboard can score band-power error vs truth.

Caveats: A's own small CFA stays in "truth"; the injected artifact is one fixed template (real CFA varies
beat to beat), so this tests spatial/temporal removal, not tracking of a changing artifact.

Run: venv/bin/python "paper utils/Paper CFA/stage_band_injection.py" --pairs 150 --workers 24
"""
import argparse
import os
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import make_cleaning_demo as cd
import stage_band_cleaning as sb
from ica_ecg_component_variance import HEP_TMIN, detect_r_peaks, epoch_around_r_peaks

CACHE = os.path.join(cd.HERE, "stage_band_injection_cache")
MAX_FRAC = 0.2  # a few donors have frac_pre up to 0.7 (artifact-dominated windows)


def pairs(n, seed=cd.SEED):
    c = pd.read_parquet(os.path.join(cd.HERE, "cfa_combined.parquet"),
                        columns=["patient_id", "edf_path", "sfreq_hz", "n_eeg_channels", "cfa_r2_excl_qrs"])
    c = c[c.patient_id.str.startswith("I0") & (c.n_eeg_channels == 6)]
    p = c.groupby("patient_id").agg(edf_path=("edf_path", "first"), sf=("sfreq_hz", "first"), r2=("cfa_r2_excl_qrs", "mean"))
    frac = pd.read_parquet(os.path.join(cd.HERE, "cleaning_demo.parquet"), columns=["patient_id", "frac_pre"])
    p = p.join(frac.set_index("patient_id").frac_pre)
    lo, hi = p[p.r2 <= p.r2.quantile(0.25)], p[(p.r2 >= p.r2.quantile(0.75)) & p.frac_pre.notna()]
    rng = np.random.default_rng(seed)
    out = []
    for a in lo.sample(frac=1, random_state=seed).itertuples():
        donors = hi[hi.sf == a.sf]
        if len(donors):
            b = donors.iloc[rng.integers(len(donors))]
            out.append((a.Index, a.edf_path, b.name, b.edf_path, a.r2, b.r2, min(b.frac_pre, MAX_FRAC)))
        if len(out) == n:
            break
    return out


def process(job):
    a_id, a_path, b_id, b_path, a_r2, b_r2, target = job
    out = os.path.join(CACHE, f"{a_id}__{b_id}.parquet")
    if os.path.exists(out):
        return a_id, "cached"
    try:
        A, B = sb.load_night(a_path), sb.load_night(b_path, stage=False)
        if isinstance(A, str) or isinstance(B, str):
            return a_id, f"skip {A if isinstance(A, str) else B}"
        eeg_a, _, sf, stages = A
        eeg_b, ecg_b, sf_b, _ = B
        if sf != sf_b:
            return a_id, "sf_mismatch"
        L = min(eeg_a.shape[1], eeg_b.shape[1])
        peaks = detect_r_peaks(ecg_b[:L], sf)
        if len(peaks) < 100:
            return a_id, "few_peaks"
        ev = epoch_around_r_peaks(eeg_b[:, :L], peaks, sf).mean(0)
        ev -= ev.mean(1, keepdims=True)
        art, a0 = np.zeros((6, L)), int(round(-HEP_TMIN * sf))
        for p in peaks:
            if p - a0 >= 0 and p - a0 + ev.shape[1] <= L:
                art[:, p - a0:p - a0 + ev.shape[1]] += ev
        truth = eeg_a[:, :L]
        art *= np.sqrt(target / np.mean(art.var(1) / truth.var(1)))
        sigs = {**sb.clean_all(truth + art, ecg_b[:L], peaks, sf), "truth": truth}
        expert = sb.expert_stages(a_path)
        stagings = {"yasa": stages, **({"expert": expert} if expert is not None else {})}
        rows = sb.stage_rows(sigs, stagings, truth, sf, patient_id=a_id, donor_id=b_id, recipient_r2=a_r2,
                             donor_r2=b_r2, target_pct_var=100 * target, artifact_pct_var=float(100 * np.mean(art.var(1) / truth.var(1))))
        pd.DataFrame(rows).to_parquet(out)
        return a_id, f"ok {len(rows)}"
    except Exception as exc:  # noqa: BLE001
        return a_id, f"error {type(exc).__name__}: {exc} | {traceback.format_exc(limit=1).splitlines()[-1]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=150)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    jobs = pairs(args.pairs)
    print(f"{len(jobs)} pairs, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (pid, status) in enumerate(ex.map(process, jobs), 1):
            print(f"[{i}/{len(jobs)}] {pid} {status}", flush=True)


if __name__ == "__main__":
    main()
