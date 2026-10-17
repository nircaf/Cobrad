#!/usr/bin/env python3
"""Can the ECG (or at least its R-peaks / HR / HRV) be recovered from EEG alone,
via the cardiac field artifact (CFA)?

Per recording, on the same QC window as the other CFA scripts
(select_quality_window), the real ECG channel gives ground-truth R-peaks via
neurokit2 -- NOT the repo's detect_r_peaks, which also fires on T-waves
(~2x beats vs MNE's find_ecg_events on most recordings checked).
EEG-derived candidates are every EEG channel plus every ICA source (EEG
only, the ECG is never shown to ICA); beats via detect_eeg_beats. Two ways
to pick one candidate:

  blind  -- most periodic candidate (envelope autocorrelation at 40-180 bpm):
            no ECG used, i.e. what you'd get on an EEG-only recording.
  trained -- ECG used on the first half only: multichannel R-locked CFA
            template + matched filter, scored on the held-out second half.
  oracle -- candidate with best F1 against the real ECG (optimistic: picked on test data).

Scored with AAMI-EC57-style beat matching (+/-75 ms), plus HR and RMSSD
error. Chance level: the blind beats circularly shifted by a random offset
(same rate and regularity, timing destroyed). Separately, ECG *waveform* reconstruction: least-squares
ECG ~ EEG channels fit on the first half of the window, R^2 on the second.

Run:
  python ecg_from_eeg.py --per-dataset 500 --workers 64   (resumable, per-EDF cache)
  python ecg_from_eeg.py --plot-only
"""
from __future__ import annotations

import argparse, hashlib, sys, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve, find_peaks
from tqdm.auto import tqdm

from ica_ecg_component_variance import (
    HERE, atomic_parquet, channel_names, discover, epoch_around_r_peaks, identifiers, select_quality_window,
)

TOL_S = 0.075  # AAMI EC57 beat-match window (150 ms total)
SNIPPET_S = 10
CACHE_VERSION = 1


def match(true, pred, sfreq):
    """Sensitivity, PPV, F1 and median signed lag (s) of pred peaks vs true peaks within +/-TOL_S."""
    if len(true) == 0 or len(pred) == 0:
        return 0.0, 0.0, 0.0, np.nan
    tol = TOL_S * sfreq
    def nearest(a, b):  # signed distance from each a to its nearest b
        i = np.clip(np.searchsorted(b, a), 1, len(b) - 1) if len(b) > 1 else np.zeros(len(a), int)
        left, right = b[np.maximum(i - 1, 0)], b[i]
        return np.where(np.abs(a - left) <= np.abs(a - right), a - left, a - right)
    d_true = nearest(true, pred)
    sens = float(np.mean(np.abs(d_true) <= tol))
    ppv = float(np.mean(np.abs(nearest(pred, true)) <= tol))
    f1 = 2 * sens * ppv / (sens + ppv) if sens + ppv else 0.0
    hit = np.abs(d_true) <= tol
    lag = float(np.median(-d_true[hit]) / sfreq) if hit.any() else np.nan
    return sens, ppv, f1, lag


def hr_rmssd(peaks, sfreq):
    """Mean HR (bpm) and RMSSD (ms) from peaks, using only physiologically plausible RR (0.33-2 s)."""
    rr = np.diff(peaks) / sfreq
    rr = rr[(rr >= .33) & (rr <= 2)]
    if len(rr) < 3:
        return np.nan, np.nan
    return float(60 / rr.mean()), float(np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000)


def detect_eeg_beats(sig, sfreq):
    """Beats in an EEG trace, no ECG involved. An ECG-style MAD threshold
    fires on every EEG wiggle (HR pinned at the refractory limit), so the beat
    period comes first: highest local max of the autocorrelation of the
    smoothed, 2 s-detrended 5-25 Hz envelope at 0.33-1.5 s lags (40-180 bpm;
    plain argmax just finds the lower edge of a decaying curve). Peaks are
    then spaced >= 0.6 period and kept if >= half the median peak height.
    Returns (peaks, periodicity); periodicity = normalised autocorrelation at
    that period = the blind selection score."""
    band = mne.filter.filter_data(sig, sfreq, 5, min(25, sfreq / 2 - .5), verbose="ERROR")
    k = max(1, int(.05 * sfreq))
    env = np.convolve(np.abs(band), np.ones(k) / k, mode="same")
    w = int(2 * sfreq)
    x = env - np.convolve(env, np.ones(w) / w, mode="same")
    ac = np.fft.irfft(np.abs(np.fft.rfft(x, 2 * len(x))) ** 2)[:len(x)]
    lo, hi = int(.33 * sfreq), int(1.5 * sfreq)
    local, _ = find_peaks(ac[lo:hi])
    if not len(local):
        return np.array([], dtype=int), -np.inf
    period = lo + int(local[np.argmax(ac[lo + local])])
    peaks, props = find_peaks(env, distance=max(1, int(.6 * period)), height=0)
    peaks = peaks[props["peak_heights"] >= .5 * np.median(props["peak_heights"])]
    return peaks, float(ac[period] / ac[0])


def process(job):
    path_text, minutes, seed, max_attempts = job
    path = Path(path_text)
    patient, recording = identifiers(path)
    try:
        raw = mne.io.read_raw_edf(path, preload=False, encoding="latin1", verbose="ERROR")
        eeg, ecg = channel_names(raw)
        raw.set_channel_types({ch: "eeg" for ch in eeg}, on_unit_change="ignore", verbose="ERROR")
        raw.set_channel_types({ecg: "ecg"}, on_unit_change="ignore", verbose="ERROR")
        duration = raw.n_times / float(raw.info["sfreq"])
        selected, qc, _, _ = select_quality_window(raw, eeg, ecg, duration, minutes, seed, path, max_attempts)
        sfreq = float(selected.info["sfreq"])
        ecg_sig = selected.get_data(picks=[ecg])[0]
        true = np.asarray(nk.ecg_peaks(nk.ecg_clean(ecg_sig, sampling_rate=sfreq), sampling_rate=sfreq)[1]["ECG_R_Peaks"])
        n = len(ecg_sig)
        win_s = n / sfreq

        selected.filter(1, min(45, sfreq / 2 - .5), picks=eeg, verbose="ERROR")
        eeg_data = selected.get_data(picks=eeg)
        ica = mne.preprocessing.ICA(n_components=min(15, len(eeg) - 1), method="fastica", max_iter="auto", random_state=0, verbose="ERROR")
        ica.fit(selected, picks=eeg, verbose="ERROR")
        sources = ica.get_sources(selected).get_data()

        cands = [(f"ch:{c}", eeg_data[i]) for i, c in enumerate(eeg)] + [(f"ica:{i}", s) for i, s in enumerate(sources)]
        scored = []
        for name, sig in cands:
            peaks, periodicity = detect_eeg_beats(sig, sfreq)
            scored.append((name, sig, peaks, periodicity, match(true, peaks, sfreq)))
        blind = max(scored, key=lambda s: s[3])
        oracle = max(scored, key=lambda s: s[4][2])

        true_hr, true_rmssd = hr_rmssd(true, sfreq)
        row = {"patient_id": patient, "recording_id": recording, "edf_path": str(path), "n_eeg_channels": len(eeg),
               "sfreq_hz": sfreq, "window_s": win_s, "n_true_beats": len(true), "true_hr_bpm": true_hr, "true_rmssd_ms": true_rmssd}
        for tag, (name, sig, peaks, _, (sens, ppv, f1, lag)) in (("blind", blind), ("oracle", oracle)):
            hr, rmssd = hr_rmssd(peaks, sfreq)
            row |= {f"{tag}_source": name, f"{tag}_sens": sens, f"{tag}_ppv": ppv, f"{tag}_f1": f1, f"{tag}_lag_s": lag,
                    f"{tag}_hr_bpm": hr, f"{tag}_rmssd_ms": rmssd}
        offset = np.random.default_rng(0).integers(int(5 * sfreq), n - int(5 * sfreq))
        row["chance_f1"] = match(true, np.sort((blind[2] + offset) % n), sfreq)[2]

        # Trained (ECG used only on the first half): multichannel CFA template
        # (R-locked EEG average, +/-100 ms) -> summed matched filter -> peaks,
        # threshold picked for best F1 on the train half; scored on the test half.
        half = n // 2
        train_true, test_true = true[true < half], true[true >= half] - half
        tmpl = epoch_around_r_peaks(eeg_data[:, :half], train_true, sfreq, -.1, .1).mean(0)
        mf = sum(fftconvolve(eeg_data[c], tmpl[c][::-1], mode="same") for c in range(len(eeg)))
        cand, props = find_peaks(mf, distance=int(.33 * sfreq), height=0)
        heights = props["peak_heights"]
        def at(thr, lo, hi, ref):
            keep = cand[(heights >= thr) & (cand >= lo) & (cand < hi)] - lo
            return keep, match(ref, keep, sfreq)
        thr = max(np.quantile(heights, np.linspace(0, .95, 40)), key=lambda q: at(q, 0, half, train_true)[1][2])
        tpeaks, (sens, ppv, f1, lag) = at(thr, half, n, test_true)
        hr, rmssd = hr_rmssd(tpeaks, sfreq)
        true_hr2, true_rmssd2 = hr_rmssd(test_true, sfreq)
        row |= {"trained_sens": sens, "trained_ppv": ppv, "trained_f1": f1, "trained_lag_s": lag, "trained_hr_bpm": hr,
                "trained_rmssd_ms": rmssd, "test_true_hr_bpm": true_hr2, "test_true_rmssd_ms": true_rmssd2}

        # ponytail: zero-lag linear map; add time-lagged EEG copies (FIR) if R^2 looks capped by conduction delay.
        X = np.vstack([eeg_data, np.ones(n)]).T
        ecg_f = mne.filter.filter_data(ecg_sig, sfreq, 1, min(45, sfreq / 2 - .5), verbose="ERROR")
        w, *_ = np.linalg.lstsq(X[:half], ecg_f[:half], rcond=None)
        resid = ecg_f[half:] - X[half:] @ w
        row["recon_r2_test"] = float(1 - resid.var() / ecg_f[half:].var())

        s0 = int(n // 2)
        s1 = s0 + int(SNIPPET_S * sfreq)
        z = lambda x: (x - x.mean()) / (x.std() + 1e-12)
        row |= {"snip_ecg": z(ecg_f[s0:s1]).tolist(), "snip_blind": z(blind[1][s0:s1]).tolist(),
                "snip_recon": z((X[s0:s1] @ w)).tolist(),
                "snip_true_peaks": ((true[(true >= s0) & (true < s1)] - s0) / sfreq).tolist(),
                "snip_blind_peaks": ((blind[2][(blind[2] >= s0) & (blind[2] < s1)] - s0) / sfreq).tolist()}
        return row, None
    except Exception as exc:
        return None, {"patient_id": patient, "edf_path": str(path), "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}


def plot(df: pd.DataFrame, out: Path):
    fig = plt.figure(figsize=(15, 12.5))
    gs = fig.add_gridspec(3, 3)
    ax = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])
    ex = df.iloc[(df["blind_f1"] - df["blind_f1"].median()).abs().argmin()]  # median, not cherry-picked
    t = np.arange(len(ex["snip_ecg"])) / ex["sfreq_hz"]
    a = ax[0, 0]
    a.plot(t, ex["snip_ecg"] + 6, "k", lw=.8, label="true ECG")
    a.plot(t, ex["snip_blind"], color="C0", lw=.8, label=f"EEG-derived ({ex['blind_source']})")
    a.plot(t, np.asarray(ex["snip_recon"]) - 6, color="C2", lw=.8, label="linear EEG→ECG reconstruction")
    for p in ex["snip_true_peaks"]:
        a.axvline(p, color="k", alpha=.15, lw=.6)
    a.plot(ex["snip_blind_peaks"], np.full(len(ex["snip_blind_peaks"]), 3), "v", color="C3", ms=5, label="EEG-detected R-peaks")
    a.set(title=f"Median-F1 recording ({ex['patient_id']}), blind F1={ex['blind_f1']:.2f}", xlabel="s", yticks=[])
    a.legend(fontsize=7, loc="lower right")

    a = ax[0, 1]
    bins = np.linspace(0, 1, 21)
    for tag, c in (("chance", "0.6"), ("blind", "C0"), ("trained", "C3"), ("oracle", "C1")):
        a.hist(df[f"{tag}_f1"], bins, alpha=.5, color=c, label=f"{tag}: median {df[f'{tag}_f1'].median():.2f}, ≥0.9 in {np.mean(df[f'{tag}_f1'] >= .9):.0%}")
    a.set(title=f"R-peak F1 (±{TOL_S*1000:.0f} ms), n={len(df)}", xlabel="F1", ylabel="recordings")
    a.legend(fontsize=7)

    a = ax[0, 2]
    a.boxplot([df["blind_sens"], df["blind_ppv"], df["trained_sens"], df["trained_ppv"]], tick_labels=["blind\nsens", "blind\nPPV", "trained\nsens", "trained\nPPV"])
    a.set(title="Sensitivity / PPV (trained: held-out half)", ylim=(0, 1.02))

    for a, key, unit, lim in ((ax[1, 0], "hr_bpm", "bpm", None), (ax[1, 1], "rmssd_ms", "ms", 200)):
        stats, hi = [], lim or 0
        for tag, ref, c in (("blind", "true_", "C0"), ("trained", "test_true_", "C3")):
            d = df[[ref + key, f"{tag}_" + key]].dropna()
            err = (d.iloc[:, 1] - d.iloc[:, 0]).abs()
            r = np.corrcoef(d.iloc[:, 0], d.iloc[:, 1])[0, 1]
            a.scatter(d.iloc[:, 0], d.iloc[:, 1], s=10, alpha=.6, color=c, label=f"{tag}: r={r:.2f}, median AE={err.median():.1f}")
            hi = hi if lim else max(hi, float(np.nanmax(d.values)))
        a.plot([0, hi], [0, hi], "k--", lw=.8)
        a.set(xlim=(0, hi), ylim=(0, hi), xlabel=f"ECG {key.split('_')[0].upper()} ({unit})", ylabel=f"EEG-derived ({unit})",
              title=f"{key.split('_')[0].upper()} from EEG vs ECG")
        a.legend(fontsize=8)

    a = ax[1, 2]
    a.hist(df["recon_r2_test"].clip(-1, 1), np.linspace(-1, 1, 41), color="C2")
    a.set(title=f"ECG waveform recon, held-out R²: median {df['recon_r2_test'].median():.2f}", xlabel="R² (clipped at -1)", ylabel="recordings")
    a = fig.add_subplot(gs[2, :])
    df = df.assign(dataset=df["edf_path"].str.split("EDF_Format/").str[1].str.split("/").str[0])
    sets = df["dataset"].value_counts().index
    tags = (("chance", "0.6"), ("blind", "C0"), ("trained", "C3"), ("oracle", "C1"))
    for k, (tag, c) in enumerate(tags):
        b = a.boxplot([df.loc[df["dataset"] == d, f"{tag}_f1"] for d in sets], positions=np.arange(len(sets)) + (k - 1.5) * .2,
                      widths=.17, patch_artist=True, showfliers=False, manage_ticks=False)
        for box in b["boxes"]:
            box.set(facecolor=c, alpha=.7)
        a.plot([], [], "s", color=c, label=tag)
    a.set_xticks(range(len(sets)), [f"{d}\nn={(df['dataset'] == d).sum()}" for d in sets], fontsize=8)
    a.set(ylabel="R-peak F1", ylim=(0, 1.02), title="R-peak F1 per dataset")
    a.legend(fontsize=8, ncol=4, loc="upper right")
    fig.suptitle("Recovering ECG from EEG alone via the cardiac field artifact", fontsize=13)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix("." + ext), dpi=150)
    print(f"Saved {out.with_suffix('.png')}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--edf-root", type=Path, default=HERE.parent.parent / "EDF_Format")
    p.add_argument("--output", type=Path, default=HERE / "ecg_from_eeg.parquet")
    p.add_argument("--figure", type=Path, default=HERE / "figures" / "ecg_from_eeg")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--window-minutes", type=float, default=10)
    p.add_argument("--max-window-attempts", type=int, default=10)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--limit", type=int, help="max EDFs scanned before one-per-patient dedup")
    p.add_argument("--per-dataset", type=int, help="random (seeded) cap of patients per top-level dataset folder, so no dataset dominates")
    p.add_argument("--cache-dir", type=Path, default=HERE / "ecg_from_eeg_cache")
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()

    if not args.plot_only:
        (args.cache_dir / "ok").mkdir(parents=True, exist_ok=True)
        (args.cache_dir / "err").mkdir(parents=True, exist_ok=True)
        # The walk over ~47k EDFs on shared storage takes >10 min; list it once.
        listing = args.cache_dir / f"edf_list_{args.limit}.txt"
        if not listing.exists():
            listing.write_text("\n".join(map(str, discover(args.edf_root, args.limit))))
        one_per_patient = {}
        for path in sorted(map(Path, listing.read_text().splitlines()), key=lambda q: str(q).lower()):
            one_per_patient.setdefault(identifiers(path)[0], path)
        by_set = {}
        for path in one_per_patient.values():
            by_set.setdefault(path.relative_to(args.edf_root.absolute()).parts[0], []).append(path)
        rng = np.random.default_rng(args.random_seed)
        paths = [q for d in sorted(by_set) for q in (list(rng.choice(by_set[d], args.per_dataset, replace=False))
                 if args.per_dataset and args.per_dataset < len(by_set[d]) else by_set[d])]
        print({d: min(len(v), args.per_dataset or len(v)) for d, v in sorted(by_set.items())}, flush=True)

        def key(q):
            ident = f"{CACHE_VERSION}|{q}|{args.window_minutes}|{args.random_seed}|{args.max_window_attempts}"
            return hashlib.sha256(ident.encode()).hexdigest() + ".parquet"
        done = [q for q in paths if (args.cache_dir / "ok" / key(q)).exists() or (not args.retry_errors and (args.cache_dir / "err" / key(q)).exists())]
        done_set = set(done)
        todo = [q for q in paths if q not in done_set]
        print(f"{len(paths)} patients; cached={len(done)}; remaining={len(todo)}", flush=True)
        with ProcessPoolExecutor(args.workers) as pool:
            futs = {pool.submit(process, (str(q), args.window_minutes, args.random_seed, args.max_window_attempts)): q for q in todo}
            for fut in tqdm(as_completed(futs), total=len(futs), unit="EDF"):
                row, err = fut.result()
                atomic_parquet(pd.DataFrame([row or err]), args.cache_dir / ("ok" if row else "err") / key(futs[fut]))
        for kind, out in (("ok", args.output), ("err", args.output.with_name(args.output.stem + ".errors.parquet"))):
            files = [args.cache_dir / kind / key(q) for q in paths if (args.cache_dir / kind / key(q)).exists()]
            atomic_parquet(pd.concat(map(pd.read_parquet, files), ignore_index=True) if files else pd.DataFrame(), out)
            print(f"{kind}: {len(files)}")

    df = pd.read_parquet(args.output)
    cols = ["chance_f1", "blind_f1", "trained_f1", "trained_sens", "trained_ppv", "oracle_f1", "recon_r2_test"]
    print(df[cols].describe().round(3).to_string())
    print(f"blind source type: {df['blind_source'].str.split(':').str[0].value_counts().to_dict()}")
    plot(df, args.figure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
