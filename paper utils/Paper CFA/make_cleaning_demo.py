#!/usr/bin/env python3
"""Proof of concept: ECG-free CFA cleaning with a cohort-trained spatial filter.

Single-beat CFA is too weak to detect heartbeats reliably from EEG alone (in
the high-CFA example patient, even a matched filter built from the patient's
own ECG-locked template found only ~76% of beats). CFA, however, has a nearly
fixed scalp pattern across patients. So the cleaning does not look for beats:

  train  For patients WITH ECG (standard F3/F4/C3/C4/O1/O2 mastoid montage),
         take the dominant spatial pattern of the R-peak-locked EEG average
         (first SVD component) and average the sign-aligned patterns into one
         population CFA pattern.
  apply  For a held-out recording, project that pattern out of every EEG
         sample (signal-space projection). The ECG is never used.
  score  The held-out ECG then measures the result: CFA R^2 (outside QRS)
         before vs. after, vs. a pseudo-event chance level, and vs. an upper
         bound that uses the patient's own ECG-derived pattern. A synthetic
         neural HEP is injected to measure how much brain signal survives,
         and Welch PSD measures the cost to ongoing EEG.

BMI/sex-specific patterns are also learned, to test whether conditioning on
them helps (it did not in the prototype; reported in the stats).

Writes cleaning_demo.parquet, figures/fig5_cleaning_demo.png and
paper_stats.json["cleaning_demo"].

Run: venv/bin/python "paper utils/Paper CFA/make_cleaning_demo.py"
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch

from channel_utils import canonicalize
from ica_ecg_component_variance import (
    HEP_TMAX, HEP_TMIN, QRS_EXCLUDE_SEC, channel_names, epoch_around_r_peaks, quality,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00", "#000000"]
SITES = ["F3", "F4", "C3", "C4", "O1", "O2"]
EXAMPLE = "I0002150024023"  # Patient B of Figure 1 (high BMI, high CFA); always held out
N_LOAD, N_TRAIN, SEED = 900, 500, 1
INJECT_LAT, INJECT_SD, INJECT_UV = 0.30, 0.05, 2.0  # synthetic neural HEP (Gaussian)
INJECT_W = np.array([1.0, 1.0, 0.7, 0.7, 0.3, 0.3])  # frontal > central > occipital


def epoch_times(sfreq):
    return np.arange(-int(round(-HEP_TMIN * sfreq)), int(round(HEP_TMAX * sfreq))) / sfreq


def load(job):
    """10-min QC'd window (as in cfa_combined), standard 6-site montage only."""
    pid, path, start = job
    try:
        raw = mne.io.read_raw_edf(path, preload=False, encoding="latin1", verbose="ERROR")
        eeg_names, ecg_name = channel_names(raw)
        canon = canonicalize(pd.Series(eeg_names)).tolist()
        if len(eeg_names) != 6 or sorted(canon) != sorted(SITES):
            return None
        seg = raw.copy().pick(eeg_names + [ecg_name]).crop(start, start + 600, include_tmax=False)
        seg.load_data(verbose="ERROR")
        _, qc = quality(seg, eeg_names, ecg_name)
        sfreq = float(seg.info["sfreq"])
        seg.filter(1.0, min(100.0, sfreq / 2 - 0.5), verbose="ERROR")
        data = seg.get_data() * 1e6
        order = [canon.index(s) for s in SITES]
        peaks = np.asarray(qc["r_peak_samples"])
        return (pid, data[:-1][order], data[-1], peaks, sfreq) if len(peaks) >= 20 else None
    except Exception:  # noqa: BLE001 -- unreadable EDFs are skipped
        return None


def cfa_pattern(eeg, peaks, sfreq):
    """Dominant spatial pattern of the R-peak-locked EEG average (needs ECG peaks)."""
    ev = epoch_around_r_peaks(eeg, peaks, sfreq).mean(0)
    u, _, _ = np.linalg.svd(ev - ev.mean(1, keepdims=True), full_matrices=False)
    return u[:, 0]


def project_out(eeg, u):
    return eeg - np.outer(u, u @ eeg)


def cfa_r2(eeg, ecg, peaks, sfreq):
    ev = epoch_around_r_peaks(eeg, peaks, sfreq).mean(0)
    ecg_ev = epoch_around_r_peaks(ecg, peaks, sfreq).mean(0)
    keep = np.abs(epoch_times(sfreq)) > QRS_EXCLUDE_SEC
    return np.array([np.corrcoef(c[keep], ecg_ev[keep])[0, 1] ** 2 for c in ev]), ev, ecg_ev


def mean_pattern(patterns):
    P = np.array(patterns)
    P = P * np.sign(P @ P[0])[:, None]
    u = P.mean(0)
    return u / np.linalg.norm(u)


def bmi_group(bmi, sex):
    return sex, 0 if bmi < 27 else 1 if bmi < 35 else 2


def evaluate(rec, u_pop, u_grp, seed):
    pid, eeg, ecg, peaks, sfreq = rec
    tt = epoch_times(sfreq)
    clean = project_out(eeg, u_pop)
    r2_pre, ev_pre, ecg_ev = cfa_r2(eeg, ecg, peaks, sfreq)
    r2_clean, ev_clean, _ = cfa_r2(clean, ecg, peaks, sfreq)
    r2_own, ev_own, _ = cfa_r2(project_out(eeg, cfa_pattern(eeg, peaks, sfreq)), ecg, peaks, sfreq)
    r2_grp, _, _ = cfa_r2(project_out(eeg, u_grp), ecg, peaks, sfreq)
    rng = np.random.default_rng(seed)
    pseudo = np.sort(rng.integers(int(0.3 * sfreq), eeg.shape[1] - int(0.4 * sfreq), len(peaks)))
    keep = np.abs(tt) > QRS_EXCLUDE_SEC
    ev_null = epoch_around_r_peaks(eeg, pseudo, sfreq).mean(0)
    r2_null = np.array([np.corrcoef(c[keep], ecg_ev[keep])[0, 1] ** 2 for c in ev_null])
    # synthetic neural HEP survives projection by 1 - (w.u)^2/|w|^2 exactly; measure it empirically
    wave = -INJECT_UV * np.exp(-0.5 * ((tt - INJECT_LAT) / INJECT_SD) ** 2)
    inj = np.zeros_like(eeg)
    for p in peaks:
        a = p - int(round(-HEP_TMIN * sfreq))
        if a >= 0 and a + len(tt) <= eeg.shape[1]:
            inj[:, a:a + len(tt)] += INJECT_W[:, None] * wave
    rec_inj = epoch_around_r_peaks(project_out(eeg + inj, u_pop) - clean, peaks, sfreq).mean(0)
    target = INJECT_W[:, None] * wave
    f, p_pre = welch(eeg, sfreq, nperseg=int(4 * sfreq), axis=1)
    _, p_clean = welch(clean, sfreq, nperseg=int(4 * sfreq), axis=1)
    band = (f >= 1) & (f <= 40)
    return {
        "patient_id": pid, "r2_pre": r2_pre.mean(), "r2_clean": r2_clean.mean(), "r2_own": r2_own.mean(),
        "r2_grp": r2_grp.mean(), "r2_null": r2_null.mean(),
        "hep_retained": float(np.sum(rec_inj * target) / np.sum(target * target)),
        "psd_change_db": float(np.median(10 * np.log10(p_clean[:, band] / p_pre[:, band]))),
    }, {"eeg": eeg, "clean": clean, "ecg": ecg, "sfreq": sfreq, "peaks": peaks, "tt": tt,
        "ev_pre": ev_pre, "ev_clean": ev_clean, "ev_own": ev_own, "r2_pre": r2_pre, "r2_clean": r2_clean}


def topo(ax, values, cmap="RdBu_r"):
    info = mne.create_info(SITES, 100.0, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"))
    lim = float(np.max(np.abs(values)))
    im, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap=cmap, vlim=(-lim, lim),
                                 contours=0, sensors=False, names=SITES, extrapolate="head")
    for t in ax.texts:
        t.set_fontsize(7)
    return im


def main():
    cfa = pd.read_parquet(os.path.join(HERE, "cfa_combined.parquet"),
                          columns=["patient_id", "edf_path", "window_start_s"]).drop_duplicates("patient_id")
    cfa = cfa[cfa.patient_id.str.startswith("I0")]
    cfa["bdsp_patient_id"] = pd.to_numeric(cfa.patient_id.str.extract(r"I\d{4}(\d{9})", expand=False))
    meta = cfa.merge(pd.read_parquet(os.path.join(HERE, "bmi_combined.parquet")), on="bdsp_patient_id").merge(
        pd.read_parquet(os.path.join(HERE, "demographics_combined.parquet"))[["patient_id", "sex"]], on="patient_id")
    meta = meta[meta.sex.isin(["Male", "Female"])]
    sample = meta[meta.patient_id != EXAMPLE].sample(N_LOAD, random_state=SEED)
    sample = pd.concat([meta[meta.patient_id == EXAMPLE], sample])
    with ProcessPoolExecutor(48) as ex:
        loaded = [r for r in ex.map(load, [(r.patient_id, r.edf_path, float(r.window_start_s))
                                           for r in sample.itertuples()]) if r]
    example = next(r for r in loaded if r[0] == EXAMPLE)
    others = [r for r in loaded if r[0] != EXAMPLE]
    train, test = others[:N_TRAIN], others[N_TRAIN:]
    info = meta.set_index("patient_id")
    patterns = [cfa_pattern(r[1], r[3], r[4]) for r in train]
    u_pop = mean_pattern(patterns)
    groups = {}
    for r, u in zip(train, patterns):
        groups.setdefault(bmi_group(info.loc[r[0], "bmi"], info.loc[r[0], "sex"]), []).append(u)
    u_grp = {k: mean_pattern(v) for k, v in groups.items()}
    consistency = float(np.median(np.abs(np.array(patterns) @ u_pop)))

    rows = []
    for i, r in enumerate(test):
        row, _ = evaluate(r, u_pop, u_grp[bmi_group(info.loc[r[0], "bmi"], info.loc[r[0], "sex"])], i)
        rows.append(row)
    res = pd.DataFrame(rows).merge(meta[["patient_id", "bmi", "sex"]], on="patient_id")
    res.to_parquet(os.path.join(HERE, "cleaning_demo.parquet"))
    _, ex_data = evaluate(example, u_pop, u_grp[bmi_group(info.loc[EXAMPLE, "bmi"], info.loc[EXAMPLE, "sex"])], 0)

    with open(os.path.join(HERE, "paper_stats.json")) as f:
        S = json.load(f)
    S["cleaning_demo"] = {
        "n_train": len(train), "n_test": int(len(res)), "pattern_consistency_median_cos": consistency,
        "group_pattern_min_cos": float(min(abs(v @ u_pop) for v in u_grp.values())),
        "u_pop": dict(zip(SITES, map(float, u_pop))),
        "r2_pre_mean": float(res.r2_pre.mean()), "r2_clean_mean": float(res.r2_clean.mean()),
        "r2_own_mean": float(res.r2_own.mean()), "r2_grp_mean": float(res.r2_grp.mean()),
        "r2_null_mean": float(res.r2_null.mean()),
        "pct_reduction_mean": float((1 - res.r2_clean.mean() / res.r2_pre.mean()) * 100),
        "pct_patients_reduced": float((res.r2_clean < res.r2_pre).mean() * 100),
        "p_pre_vs_clean": float(stats.wilcoxon(res.r2_pre, res.r2_clean).pvalue),
        "p_grp_vs_pop": float(stats.wilcoxon(res.r2_grp, res.r2_clean).pvalue),
        "pct_patients_grp_better": float((res.r2_grp < res.r2_clean).mean() * 100),
        "hep_retained_median": float(res.hep_retained.median()),
        "psd_change_db_median": float(res.psd_change_db.median()),
        "r_bmi_pre": float(stats.pearsonr(res.bmi, res.r2_pre)[0]),
        "p_bmi_pre": float(stats.pearsonr(res.bmi, res.r2_pre)[1]),
        "r_bmi_clean": float(stats.pearsonr(res.bmi, res.r2_clean)[0]),
        "p_bmi_clean": float(stats.pearsonr(res.bmi, res.r2_clean)[1]),
        "example_r2_pre": float(ex_data["r2_pre"].mean()), "example_r2_clean": float(ex_data["r2_clean"].mean()),
    }
    with open(os.path.join(HERE, "paper_stats.json"), "w") as f:
        json.dump(S, f, indent=2)
    print(json.dumps(S["cleaning_demo"], indent=1))
    plot(res, ex_data, u_pop)


def plot(res, ex, u_pop):
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 300, "savefig.dpi": 300})
    fig = plt.figure(figsize=(11, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.42, wspace=0.34)
    sf = ex["sfreq"]

    # a: continuous EEG before vs after, with the (unused) ECG for reference
    ax = fig.add_subplot(gs[0, :2])
    sl = slice(int(120 * sf), int(126 * sf))
    t = np.arange(sl.stop - sl.start) / sf
    spacing = 2.4 * np.percentile(np.abs(ex["eeg"][:, sl]), 98)
    for k, name in enumerate(SITES):
        off = -k * spacing
        ax.plot(t, ex["eeg"][k, sl] + off, color=PALETTE[1], lw=0.8, label="Before" if k == 0 else None)
        ax.plot(t, ex["clean"][k, sl] + off, color="black", lw=0.8, label="After (no ECG used)" if k == 0 else None)
        ax.text(-0.12, off, name, ha="right", va="center", fontsize=8)
    ecg = ex["ecg"][sl]
    ax.plot(t, (ecg - ecg.mean()) / np.ptp(ecg) * spacing - len(SITES) * spacing, color=PALETTE[0], lw=0.8)
    ax.text(-0.12, -len(SITES) * spacing, "ECG*", ha="right", va="center", fontsize=8, color=PALETTE[0])
    for p in ex["peaks"]:
        if sl.start <= p < sl.stop:
            ax.axvline((p - sl.start) / sf, color="0.75", lw=0.5, ls=":", zorder=0)
    ax.plot([t[-1] + 0.1] * 2, [0, -20], color="black", lw=1.2)
    ax.text(t[-1] + 0.15, -10, "20 µV", va="center", fontsize=7)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Time (s)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=2, bbox_to_anchor=(1, 1.1))
    ax.set_title("a", loc="left", fontsize=10, fontweight="bold")

    # b: learned population CFA pattern
    ax = fig.add_subplot(gs[0, 2])
    im = topo(ax, u_pop)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("Pattern weight (a.u.)", fontsize=7.5)
    ax.set_title("b", loc="left", fontsize=10, fontweight="bold")

    # c: heartbeat-locked average, channel where the filter removes most CFA
    ax = fig.add_subplot(gs[1, 0])
    i = int(np.argmax(ex["r2_pre"] - ex["r2_clean"]))
    tt = ex["tt"] * 1000
    ax.axvspan(-QRS_EXCLUDE_SEC * 1000, QRS_EXCLUDE_SEC * 1000, color="0.92", zorder=0)
    ax.plot(tt, ex["ev_pre"][i], color=PALETTE[1], lw=1.5, label=f"Before (R² = {ex['r2_pre'][i]:.2f})")
    ax.plot(tt, ex["ev_clean"][i], color="black", lw=1.5, label=f"After (R² = {ex['r2_clean'][i]:.2f})")
    ax.set_xlabel("Time from R-peak (ms)")
    ax.set_ylabel(f"{SITES[i]} heartbeat-locked average (µV)")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("c", loc="left", fontsize=10, fontweight="bold")

    # d: CFA R^2 across held-out patients
    ax = fig.add_subplot(gs[1, 1])
    cols = [("r2_pre", "Before", PALETTE[1]), ("r2_clean", "After\n(no ECG)", "black"),
            ("r2_own", "ECG-based\nbound", PALETTE[2]), ("r2_null", "Chance", "0.55")]
    for r in res.itertuples():
        ax.plot([0, 1], [r.r2_pre, r.r2_clean], color="0.8", lw=0.3, zorder=0)
    for j, (c, _, color) in enumerate(cols):
        ax.boxplot(res[c], positions=[j], widths=0.55, showfliers=False, patch_artist=True,
                   boxprops={"facecolor": color, "alpha": 0.35}, medianprops={"color": color, "lw": 1.8})
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[1] for c in cols], fontsize=7.5)
    ax.set_ylabel("Patient-mean CFA R² (outside QRS)")
    ax.set_title("d", loc="left", fontsize=10, fontweight="bold")

    # e: before vs after across BMI
    ax = fig.add_subplot(gs[1, 2])
    edges = [15, 25, 30, 35, 40, 60]
    mid, pre, post = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        g = res[(res.bmi >= lo) & (res.bmi < hi)]
        if len(g) >= 10:
            mid.append((lo + hi) / 2)
            pre.append((g.r2_pre.mean(), g.r2_pre.sem()))
            post.append((g.r2_clean.mean(), g.r2_clean.sem()))
    for vals, color, label in [(pre, PALETTE[1], "Before"), (post, "black", "After (no ECG)")]:
        m_, s_ = np.array(vals).T
        ax.errorbar(mid, m_, yerr=s_, color=color, marker="o", ms=4, capsize=2, lw=1.4, label=label)
    ax.set_xlabel("BMI (kg/m²)")
    ax.set_ylabel("Mean CFA R² (± SEM)")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("e", loc="left", fontsize=10, fontweight="bold")

    fig.savefig(os.path.join(FIG_DIR, "fig5_cleaning_demo.png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "fig5_cleaning_demo.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
