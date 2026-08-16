#!/usr/bin/env python3
"""Figure 4: how much of the EEG is heart data, measured two ways beyond R²
(§2.2) -- lag-resolved cross-correlation and mutual information between each
channel's HEP evoked average and the patient's own ECG evoked average,
computed pre-ICA, post-ICA (ECG-flagged component removed), and against a
non-heartbeat-locked control (same pre-ICA EEG, epoched around random
pseudo-events instead of true R-peaks, correlated against the same real ECG
evoked average) -- the chance-level floor for an EEG waveform carrying no
genuine R-peak-locked structure at all. If ICA cleaning is working, pre/post
should both sit above this floor and post should sit below pre.

Re-fits ICA per recording (same as make_entropy_control.py) since the
post-ICA EEG waveform, and the paired ECG evoked average, are not persisted
by the main ICA batch -- only summary variance is.

Produces three tables:
  crosscorr_curve.parquet    -- lag-resolved r, core 4-electrode channels only
                                 (patient, channel, condition, lag_ms, r)
  crosscorr_mi_summary.parquet -- one row per patient/channel: peak |r| (and
                                 its lag) and mutual information, pre vs post
  psd_curve.parquet          -- Welch PSD (dB) of the evoked waveform, core-4
                                 electrodes + ECG, for pre-ICA/post-ICA/
                                 non-locked/ecg conditions

Run on a subsample (ICA re-fit is the expensive step -- not the full corpus):
  venv/bin/python "Paper CFA/make_crosscorr_mi.py" --limit 500
"""
from __future__ import annotations

import argparse, sys, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.feature_selection import mutual_info_regression
from tqdm.auto import tqdm

from ica_ecg_component_variance import HERE, atomic_parquet, channel_names, epoch_around_r_peaks, quality

CORE_FOUR = ["F3", "F4", "C3", "C4"]
# The 6 well-covered canonical sites (§2.7): PSD is reported per-electrode for
# all 6, while cross-correlation/MI's group-overall average uses CORE_FOUR.
PSD_SITES = ["F3", "F4", "C3", "C4", "O1", "O2"]
MAX_LAG_MS = 100
LAG_STEP_MS = 5
LAGS_MS = list(range(-MAX_LAG_MS, MAX_LAG_MS + 1, LAG_STEP_MS))

CURVE_COLUMNS = ["patient_id", "recording_id", "eeg_channel", "condition", "lag_ms", "r"]
SUMMARY_COLUMNS = ["patient_id", "recording_id", "eeg_channel",
                    "peak_r_pre", "peak_lag_ms_pre", "peak_r_post", "peak_lag_ms_post",
                    "peak_r_non_locked", "peak_lag_ms_non_locked",
                    "mi_pre", "mi_post", "mi_non_locked"]
# PSD rows are restricted to the core-4 electrodes + ECG (not all channels)
# to keep output size manageable -- this is a supplementary spectral-shape
# comparison, not a per-electrode statistical test.
PSD_COLUMNS = ["patient_id", "recording_id", "eeg_channel", "condition", "freq_hz", "power_db"]
ERROR_COLUMNS = ["patient_id", "recording_id", "edf_path", "error_type", "error", "traceback"]


def psd_rows(evoked: np.ndarray, sfreq: float) -> list[tuple[float, float]]:
    """Welch PSD of a single evoked waveform, in dB (10*log10 power)."""
    nperseg = min(len(evoked), 128)
    if nperseg < 8:
        return []
    freqs, power = welch(evoked, fs=sfreq, nperseg=nperseg)
    power_db = 10 * np.log10(np.clip(power, 1e-12, None))
    return list(zip(freqs.tolist(), power_db.tolist()))


def shifted_corr(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    """corr(x[t], y[t+lag]): positive lag means y leads x."""
    n = len(x)
    if lag >= 0:
        a, b = (x[: n - lag] if lag > 0 else x), y[lag:]
    else:
        a, b = x[-lag:], y[: n + lag]
    if len(a) < 10 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def process(job):
    patient_id, recording_id, path_text, start_s, duration_s, low, high = job
    path = Path(path_text)
    try:
        raw = mne.io.read_raw_edf(path, preload=False, encoding="latin1", verbose="ERROR")
        eeg, ecg = channel_names(raw)
        raw.set_channel_types({ch: "eeg" for ch in eeg}, on_unit_change="ignore", verbose="ERROR")
        raw.set_channel_types({ecg: "ecg"}, on_unit_change="ignore", verbose="ERROR")
        segment = raw.copy().pick(eeg + [ecg]).crop(start_s, start_s + duration_s, include_tmax=False)
        segment.load_data(verbose="ERROR")
        sfreq = float(segment.info["sfreq"])
        upper = min(high, sfreq / 2 - .5)
        if upper <= low:
            raise ValueError(f"sampling rate {sfreq} too low")
        segment.filter(low, upper, picks=eeg + [ecg], verbose="ERROR")

        passed, qc = quality(segment, eeg, ecg)
        r_peaks = qc["r_peak_samples"]
        if len(r_peaks) < 20:
            raise ValueError("too few R-peaks for evoked estimate")

        try:
            import picard  # noqa: F401
            method, params = "picard", {"ortho": False, "extended": True}
        except ImportError:
            method, params = "infomax", {"extended": True}
        count = min(15, len(eeg) - 1)
        ica = mne.preprocessing.ICA(n_components=count, method=method, fit_params=params,
                                     random_state=42, max_iter=500)
        ica.fit(segment, picks=eeg, verbose="ERROR")
        bad, scores = ica.find_bads_ecg(segment, ch_name=ecg, method="correlation", verbose="ERROR")
        scores = np.asarray(scores, float)
        if not bad and scores.size:
            bad = [int(np.nanargmax(np.abs(scores)))]
        cleaned = segment.copy()
        ica.apply(cleaned, exclude=bad, verbose="ERROR")

        eeg_pre = segment.get_data(picks=eeg) * 1e6
        eeg_post = cleaned.get_data(picks=eeg) * 1e6
        ecg_signal = segment.get_data(picks=[ecg])[0] * 1e6

        pre_epochs = epoch_around_r_peaks(eeg_pre, r_peaks, sfreq, tmin=-0.3, tmax=0.4)
        post_epochs = epoch_around_r_peaks(eeg_post, r_peaks, sfreq, tmin=-0.3, tmax=0.4)
        ecg_epochs = epoch_around_r_peaks(ecg_signal, r_peaks, sfreq, tmin=-0.3, tmax=0.4)
        if pre_epochs.shape[0] < 20:
            raise ValueError("too few valid R-peak epochs")

        evoked_pre = pre_epochs.mean(axis=0)   # (n_channels, n_samples)
        evoked_post = post_epochs.mean(axis=0)
        evoked_ecg = ecg_epochs.mean(axis=0)   # (n_samples,)

        # Non-heartbeat-locked control: same pre-ICA EEG, epoched around
        # uniformly-random pseudo-events (same count as real R-peak epochs)
        # instead of true R-peaks, then correlated/MI'd against the SAME real
        # (R-peak-locked) ECG evoked average -- the chance-level EEG-ECG
        # relationship expected if the EEG evoked waveform carried no genuine
        # R-peak-locked structure at all.
        n_samples_seg = eeg_pre.shape[1]
        pre_margin = int(round(0.3 * sfreq))
        post_margin = int(round(0.4 * sfreq))
        rng = np.random.default_rng(abs(hash((patient_id, recording_id, "crosscorr_nl"))) % (2**32))
        valid_lo, valid_hi = pre_margin, n_samples_seg - post_margin
        if valid_hi <= valid_lo:
            raise ValueError("window too short for pseudo-event epoching")
        pseudo_peaks = rng.integers(valid_lo, valid_hi, size=pre_epochs.shape[0])
        nl_epochs = epoch_around_r_peaks(eeg_pre, pseudo_peaks, sfreq, tmin=-0.3, tmax=0.4)
        if nl_epochs.shape[0] < 20:
            raise ValueError("too few valid pseudo-epochs")
        evoked_nl = nl_epochs.mean(axis=0)

        psd_out = []
        psd_canon = {c.split("-")[0].upper() for c in PSD_SITES}
        for i, ch in enumerate(eeg):
            if ch.split("-")[0].upper() not in psd_canon:
                continue
            for condition, waveform in [("pre_ica", evoked_pre[i]), ("post_ica", evoked_post[i]),
                                         ("non_locked", evoked_nl[i])]:
                for freq_hz, power_db in psd_rows(waveform, sfreq):
                    psd_out.append({"patient_id": patient_id, "recording_id": recording_id,
                                     "eeg_channel": ch, "condition": condition,
                                     "freq_hz": freq_hz, "power_db": power_db})
        for freq_hz, power_db in psd_rows(evoked_ecg, sfreq):
            psd_out.append({"patient_id": patient_id, "recording_id": recording_id,
                             "eeg_channel": "ECG", "condition": "ecg",
                             "freq_hz": freq_hz, "power_db": power_db})

        curve_rows, summary_rows = [], []
        for i, ch in enumerate(eeg):
            by_lag = {
                "pre_ica": {lag_ms: shifted_corr(evoked_pre[i], evoked_ecg, int(round(lag_ms / 1000 * sfreq)))
                            for lag_ms in LAGS_MS},
                "post_ica": {lag_ms: shifted_corr(evoked_post[i], evoked_ecg, int(round(lag_ms / 1000 * sfreq)))
                             for lag_ms in LAGS_MS},
                "non_locked": {lag_ms: shifted_corr(evoked_nl[i], evoked_ecg, int(round(lag_ms / 1000 * sfreq)))
                               for lag_ms in LAGS_MS},
            }
            # Channel-level filtering (core-4-electrode vs. all) happens downstream
            # in the plotting script via channel_utils.canonicalize -- raw labels
            # here are montage-specific (e.g. "F3-M2"), not the canonical "F3".
            peaks = {}
            for condition, r_by_lag in by_lag.items():
                for lag_ms, r in r_by_lag.items():
                    curve_rows.append({"patient_id": patient_id, "recording_id": recording_id,
                                        "eeg_channel": ch, "condition": condition, "lag_ms": lag_ms, "r": r})
                vals = np.array(list(r_by_lag.values()), float)
                lags = np.array(list(r_by_lag.keys()))
                peak_idx = np.nanargmax(np.abs(vals)) if np.isfinite(vals).any() else None
                peaks[condition] = (float(vals[peak_idx]), float(lags[peak_idx])) if peak_idx is not None else (np.nan, np.nan)

            mi_pre = float(mutual_info_regression(evoked_ecg.reshape(-1, 1), evoked_pre[i],
                                                    n_neighbors=3, random_state=42)[0])
            mi_post = float(mutual_info_regression(evoked_ecg.reshape(-1, 1), evoked_post[i],
                                                     n_neighbors=3, random_state=42)[0])
            mi_nl = float(mutual_info_regression(evoked_ecg.reshape(-1, 1), evoked_nl[i],
                                                   n_neighbors=3, random_state=42)[0])

            summary_rows.append({
                "patient_id": patient_id, "recording_id": recording_id, "eeg_channel": ch,
                "peak_r_pre": peaks["pre_ica"][0], "peak_lag_ms_pre": peaks["pre_ica"][1],
                "peak_r_post": peaks["post_ica"][0], "peak_lag_ms_post": peaks["post_ica"][1],
                "peak_r_non_locked": peaks["non_locked"][0], "peak_lag_ms_non_locked": peaks["non_locked"][1],
                "mi_pre": mi_pre, "mi_post": mi_post, "mi_non_locked": mi_nl,
            })
        return curve_rows, summary_rows, psd_out, None
    except Exception as exc:
        return [], [], [], {"patient_id": patient_id, "recording_id": recording_id, "edf_path": str(path),
                             "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve-output", type=Path, default=HERE / "crosscorr_curve.parquet")
    parser.add_argument("--summary-output", type=Path, default=HERE / "crosscorr_mi_summary.parquet")
    parser.add_argument("--psd-output", type=Path, default=HERE / "psd_curve.parquet")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=500, help="Subsample size, not the full corpus")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--l-freq", type=float, default=1)
    parser.add_argument("--h-freq", type=float, default=100)
    args = parser.parse_args()

    ica = pd.read_parquet(HERE / "ica_combined.parquet")
    recordings = ica.drop_duplicates(subset=["patient_id", "recording_id"])[
        ["patient_id", "recording_id", "edf_path", "window_start_s", "window_duration_s"]
    ]
    if len(recordings) > args.limit:
        recordings = recordings.sample(n=args.limit, random_state=args.random_seed)
    print(f"Subsampled {len(recordings):,} / {ica.recording_id.nunique():,} recordings; "
          f"{args.workers} workers (ICA re-fit per recording)")

    jobs = [
        (r.patient_id, r.recording_id, r.edf_path, r.window_start_s, r.window_duration_s, args.l_freq, args.h_freq)
        for r in recordings.itertuples()
    ]
    all_curve, all_summary, all_psd, all_errors = [], [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, job): job[1] for job in jobs}
        bar = tqdm(as_completed(futures), total=len(futures), unit="rec", desc="crosscorr/MI")
        for future in bar:
            curve_rows, summary_rows, psd_out, error = future.result()
            if error:
                all_errors.append(error)
            else:
                all_curve.extend(curve_rows)
                all_summary.extend(summary_rows)
                all_psd.extend(psd_out)
            bar.set_postfix(rows=len(all_summary), errors=len(all_errors), refresh=False)

    curve = pd.DataFrame(all_curve, columns=CURVE_COLUMNS)
    summary = pd.DataFrame(all_summary, columns=SUMMARY_COLUMNS)
    psd = pd.DataFrame(all_psd, columns=PSD_COLUMNS)
    errors = pd.DataFrame(all_errors, columns=ERROR_COLUMNS)
    atomic_parquet(curve, args.curve_output)
    atomic_parquet(summary, args.summary_output)
    atomic_parquet(psd, args.psd_output)
    atomic_parquet(errors, args.summary_output.with_name(args.summary_output.stem + ".errors.parquet"))
    print(f"Wrote {len(curve):,} curve rows to {args.curve_output}")
    print(f"Wrote {len(summary):,} summary rows ({summary.patient_id.nunique() if len(summary) else 0} patients) "
          f"to {args.summary_output}; {len(errors):,} errors")
    print(f"Wrote {len(psd):,} PSD rows to {args.psd_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
