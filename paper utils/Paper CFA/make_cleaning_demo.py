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

All eligible patients are used with K-fold cross-fitting: each patient is
cleaned by patterns learned only from the other folds. Besides the single
population pattern, a covariate model (ridge regression on BMI, age, sex and
diagnosis flags, weighted by each training patient's CFA strength) predicts a
patient-specific pattern from the participant's data, again with no ECG.
Projection removes a direction regardless of amplitude, so covariates can only
help through how they change the pattern's direction, not its size.
Besides heartbeat-locked R^2, the cardiac share of the continuous (non-locked)
EEG variance is scored: the ECG-locked template is overlap-added at every beat
and its variance divided by the total EEG variance.

CFAR is trained on raw EEG. For comparison, ECG-informed ICA (same as Section 2.4) is scored alone
and combined with CFAR in both orders (ICA->CFAR, CFAR->ICA; benchmark_ica_cfar.py found CFAR->ICA
best). Besides shape-based R^2, CFA power (R^2 x evoked variance outside QRS, uV^2) measures how
much ECG-shaped signal is left, which R^2 alone cannot show.

Writes cleaning_demo.parquet, figures/fig5_cleaning_demo.png and
paper_stats.json["cleaning_demo"].

Run: venv/bin/python "paper utils/Paper CFA/make_cleaning_demo.py" [--plot-only]
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import mne
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from channel_utils import canonicalize
from ica_ecg_component_variance import (
    HEP_TMAX, HEP_TMIN, QRS_EXCLUDE_SEC, channel_names, epoch_around_r_peaks, quality,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00", "#000000"]
SITES = ["F3", "F4", "C3", "C4", "O1", "O2"]
# Figure 4a example: chosen by scratch search over the 60 patients with most CFA removed, for the
# 5-s single-channel window where cleaning most reduces variance and the removed part is beat-locked
EXAMPLE, EXAMPLE_SITE, EXAMPLE_T0 = "I0002150406452", "C4", 443.0
# Figure 4b example: chosen by scratch search over benchmark patients with high CFA power and <10% left
# after CFAR, for the clearest ECG-shaped HEP (P-QRS-T) that CFAR flattens
HEP_EXAMPLE, HEP_SITE = "I0002150007970", "F4"
N_FOLDS, SEED, N_WORKERS = 5, 1, 64
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


def ica_clean(eeg, ecg, sfreq):
    """ECG-informed ICA as in ica_ecg_component_variance.py (Section 2.4), on arrays (uV)."""
    info = mne.create_info(SITES + ["ECG"], sfreq, ["eeg"] * len(SITES) + ["ecg"])
    raw = mne.io.RawArray(np.vstack([eeg, ecg[None]]) * 1e-6, info, verbose="ERROR")
    ica = mne.preprocessing.ICA(n_components=len(SITES) - 1, method="picard",
                                fit_params={"ortho": False, "extended": True}, random_state=42, max_iter=500)
    ica.fit(raw, picks="eeg", verbose="ERROR")
    bad, scores = ica.find_bads_ecg(raw, ch_name="ECG", method="correlation", verbose="ERROR")
    bad = bad or [int(np.nanargmax(np.abs(scores)))]
    return ica.apply(raw, exclude=bad, verbose="ERROR").get_data(picks="eeg") * 1e6


def ica_iclabel(eeg, sfreq, return_n=False):
    """Standard ECG-free ICA: same ICA as ica_clean, components labelled "heart beat" by ICLabel removed.
    ponytail: data kept on its mastoid reference (ICLabel warns; it was trained on average reference), so
    scores stay comparable with raw; no heart component found -> nothing removed."""
    from mne_icalabel import label_components
    info = mne.create_info(SITES, sfreq, "eeg")
    raw = mne.io.RawArray(eeg * 1e-6, info, verbose="ERROR")
    raw.set_montage("standard_1020", verbose="ERROR")
    ica = mne.preprocessing.ICA(n_components=len(SITES) - 1, method="picard",
                                fit_params={"ortho": False, "extended": True}, random_state=42, max_iter=500)
    ica.fit(raw, verbose="ERROR")
    labels = label_components(raw, ica, method="iclabel")["labels"]
    bad = [i for i, lab in enumerate(labels) if lab == "heart beat"]
    out = ica.apply(raw, exclude=bad, verbose="ERROR").get_data() * 1e6
    return (out, len(bad)) if return_n else out


def cfa_power(eeg, ecg, peaks, sfreq):
    """ECG-shaped evoked power outside QRS (uV^2): R^2 x evoked variance, channel mean."""
    r2, ev, _ = cfa_r2(eeg, ecg, peaks, sfreq)
    keep = np.abs(epoch_times(sfreq)) > QRS_EXCLUDE_SEC
    return float(np.mean(r2 * ev[:, keep].var(1)))


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


def cardiac_fraction(eeg, peaks, sfreq):
    """Share of continuous EEG variance carried by the ECG-locked template, overlap-added at every beat."""
    ev = epoch_around_r_peaks(eeg, peaks, sfreq).mean(0)
    card = np.zeros_like(eeg)
    a0, n = int(round(-HEP_TMIN * sfreq)), ev.shape[1]
    for p in peaks:
        a = p - a0
        if a >= 0 and a + n <= eeg.shape[1]:
            card[:, a:a + n] += ev
    return float(np.mean(card.var(1) / eeg.var(1)))


def fit_job(job):
    """Pass 1: own ECG-derived pattern and CFA strength (training signal only)."""
    rec = load(job)
    if rec is None:
        return None
    pid, eeg, ecg, peaks, sfreq = rec
    return pid, cfa_pattern(eeg, peaks, sfreq), cfa_r2(eeg, ecg, peaks, sfreq)[0].mean()


def eval_job(job):
    """Pass 2: clean with patterns learned from the other folds; score with the held-out ECG."""
    *load_args, u_pop, u_cov, seed = job
    rec = load(tuple(load_args))
    return None if rec is None else evaluate(rec, u_pop, u_cov, seed)[0]


def covariates(meta):
    meta = meta.assign(diagnosis_categories=[list(v) if v is not None else [] for v in meta.diagnosis_categories])
    dx = sorted({d for v in meta.diagnosis_categories for d in v})
    # BMI exists for only ~40% of patients (EHR); impute + flag so everyone can train the filter
    X = pd.DataFrame({"bmi": meta.bmi.fillna(meta.bmi.median()), "bmi_missing": meta.bmi.isna().astype(float),
                      "age": meta.age.fillna(meta.age.median()),
                      "male": (meta.sex == "Male").astype(float),
                      "log_n_dx": np.log1p(meta.n_diagnoses.fillna(0))}, index=meta.patient_id)
    for d in dx:
        X[d] = [float(d in v) for v in meta.diagnosis_categories]
    return X


def evaluate(rec, u_pop, u_cov, seed):
    pid, eeg, ecg, peaks, sfreq = rec
    tt = epoch_times(sfreq)
    clean = project_out(eeg, u_pop)
    ica = ica_clean(eeg, ecg, sfreq)
    combos = {"ica": ica, "ica_cfar": project_out(ica, u_pop), "cfar_ica": ica_clean(clean, ecg, sfreq)}
    own = project_out(eeg, cfa_pattern(eeg, peaks, sfreq))
    r2_pre, ev_pre, ecg_ev = cfa_r2(eeg, ecg, peaks, sfreq)
    r2_clean, ev_clean, _ = cfa_r2(clean, ecg, peaks, sfreq)
    r2_own, ev_own, _ = cfa_r2(own, ecg, peaks, sfreq)
    cov = project_out(eeg, u_cov)
    r2_cov, _, _ = cfa_r2(cov, ecg, peaks, sfreq)
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
        "patient_id": pid, **{f"r2_{k}": cfa_r2(x, ecg, peaks, sfreq)[0].mean() for k, x in combos.items()},
        **{f"pow_{k}": cfa_power(x, ecg, peaks, sfreq) for k, x in
           {"pre": eeg, "clean": clean, "own": own, **combos}.items()},
        "r2_pre": r2_pre.mean(), "r2_clean": r2_clean.mean(), "r2_own": r2_own.mean(),
        "r2_cov": r2_cov.mean(), "r2_null": r2_null.mean(),
        "frac_pre": cardiac_fraction(eeg, peaks, sfreq), "frac_clean": cardiac_fraction(clean, peaks, sfreq),
        "frac_cov": cardiac_fraction(cov, peaks, sfreq),
        "hep_retained": float(np.sum(rec_inj * target) / np.sum(target * target)),
        "psd_change_db": float(np.median(10 * np.log10(p_clean[:, band] / p_pre[:, band]))),
    }, {"eeg": eeg, "clean": clean, "ecg": ecg, "sfreq": sfreq, "peaks": peaks, "tt": tt,
        "ev_pre": ev_pre, "ev_clean": ev_clean, "ev_own": ev_own, "r2_pre": r2_pre, "r2_clean": r2_clean}


# Figure 4d pipelines from benchmark_ica_cfar.py: (column, label, uses ECG, color)
BENCH = [("ica_iclabel", "ICA", False, PALETTE[5]), ("ica", "ECG-informed ICA", True, PALETTE[5]), ("ecgreg", "ECG regression", True, PALETTE[3]),
         ("cfar", "CFAR", False, "black"), ("ica_cfar_raw", "ICA+CFAR", True, PALETTE[6]),
         ("cfar_ica", "CFAR+ICA", True, PALETTE[0]), ("cfar_ecgreg_dir", "CFAR+ECG regression", True, PALETTE[6]),
         ("own", "Patient-specific CFAR", True, PALETTE[2])]


def bench_summary(n_boot=2000):
    """Median CFA power left (% of raw) and synthetic HEP kept per pipeline, 95% bootstrap CIs."""
    # all patients (benchmark_ica_cfar.py --all) when available, else the 438-patient benchmark
    f = os.path.join(HERE, "benchmark_all.parquet")
    b = pd.read_parquet(f if os.path.exists(f) else os.path.join(HERE, "benchmark_ica_cfar.parquet"))
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(b), (n_boot, len(b)))
    out = {"n": int(len(b))}
    for k, *_ in BENCH:
        if f"pow_{k}" not in b:
            continue
        for name, v in [("pow_pct", (b[f"pow_{k}"] / b.pow_raw * 100).to_numpy()), ("hep", b[f"hep_{k}"].to_numpy())]:
            boot = np.nanmedian(v[idx], axis=1)
            out[f"{k}_{name}"] = float(np.nanmedian(v))
            out[f"{k}_{name}_ci"] = [float(x) for x in np.percentile(boot, [2.5, 97.5])]
    if "pow_ica_iclabel" in b:  # ICLabel found no heart component -> output identical to raw
        v = b.dropna(subset=["pow_ica_iclabel"])
        out["ica_iclabel_found_pct"] = float((~np.isclose(v.pow_ica_iclabel, v.pow_raw, rtol=1e-9)).mean() * 100)
    return out


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
    meta = cfa.merge(pd.read_parquet(os.path.join(HERE, "bmi_combined.parquet")), on="bdsp_patient_id", how="left").merge(
        pd.read_parquet(os.path.join(HERE, "demographics_combined.parquet"))[["patient_id", "sex", "age", "n_diagnoses", "diagnosis_categories"]], on="patient_id")
    meta = meta[meta.sex.isin(["Male", "Female"])].drop_duplicates("patient_id")
    jobs = {r.patient_id: (r.patient_id, r.edf_path, float(r.window_start_s)) for r in meta.itertuples()}
    with ProcessPoolExecutor(N_WORKERS) as ex:
        fitted = [r for r in ex.map(fit_job, jobs.values(), chunksize=8) if r]
    pids = [r[0] for r in fitted]
    P = np.array([r[1] for r in fitted])
    P = P * np.sign(P @ mean_pattern(P))[:, None]  # SVD sign is arbitrary; align to one reference
    w = np.array([r[2] for r in fitted])  # weight by CFA strength: strong-CFA patients give reliable patterns
    X = covariates(meta.set_index("patient_id", drop=False).loc[pids]).to_numpy()
    u_pop_all = (w @ P) / np.linalg.norm(w @ P)
    consistency = float(np.median(np.abs(P @ u_pop_all)))

    # ponytail: linear ridge covariates -> pattern; swap in a nonlinear model if pattern_cos_cov barely beats pop
    u_pop_of, u_cov_of, cos_pop, cos_cov = {}, {}, [], []
    for tr, te in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(P):
        u_pop = (w[tr] @ P[tr]) / np.linalg.norm(w[tr] @ P[tr])
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 12)))
        model.fit(X[tr], P[tr], ridgecv__sample_weight=w[tr])
        pred = model.predict(X[te])
        pred /= np.linalg.norm(pred, axis=1, keepdims=True)
        for i, u in zip(te, pred):
            u_pop_of[pids[i]], u_cov_of[pids[i]] = u_pop, u
            cos_pop.append(abs(P[i] @ u_pop))
            cos_cov.append(abs(P[i] @ u))
    with ProcessPoolExecutor(N_WORKERS) as ex:
        rows = [r for r in ex.map(eval_job, [(*jobs[p], u_pop_of[p], u_cov_of[p], i) for i, p in enumerate(pids)],
                                  chunksize=8) if r]
    res = pd.DataFrame(rows).merge(meta[["patient_id", "bmi", "sex", "age"]], on="patient_id")
    res.to_parquet(os.path.join(HERE, "cleaning_demo.parquet"))
    _, ex_data = evaluate(load(jobs[EXAMPLE]), u_pop_of[EXAMPLE], u_cov_of[EXAMPLE], 0)
    _, hx = evaluate(load(jobs[HEP_EXAMPLE]), u_pop_of[HEP_EXAMPLE], u_cov_of[HEP_EXAMPLE], 0)

    rb = res.dropna(subset=["bmi"])  # BMI-known subset for BMI statistics and Figure 4e
    with open(os.path.join(HERE, "paper_stats.json")) as f:
        S = json.load(f)
    S["cleaning_demo"] = {
        "n_folds": N_FOLDS, "n_train": int(round(len(pids) * (N_FOLDS - 1) / N_FOLDS)), "n_test": int(len(res)),
        "n_eligible": len(jobs), "n_test_with_bmi": int(len(rb)), "pattern_consistency_median_cos": consistency,
        "pattern_cos_pop_median": float(np.median(cos_pop)), "pattern_cos_cov_median": float(np.median(cos_cov)),
        "u_pop": dict(zip(SITES, map(float, u_pop_all))),
        **{f"r2_{k}_mean": float(res[f"r2_{k}"].mean()) for k in ["ica", "ica_cfar", "cfar_ica"]},
        **{f"pow_{k}_median": float(res[f"pow_{k}"].median()) for k in ["pre", "clean", "own", "ica", "ica_cfar", "cfar_ica"]},
        **{f"pow_{k}_pct_of_raw_median": float((res[f"pow_{k}"] / res.pow_pre).median() * 100)
           for k in ["clean", "own", "ica", "ica_cfar", "cfar_ica"]},
        "p_pow_cfar_vs_ica": float(stats.wilcoxon(res.pow_clean, res.pow_ica).pvalue),
        "p_pow_cfar_ica_vs_cfar": float(stats.wilcoxon(res.pow_cfar_ica, res.pow_clean).pvalue),
        "r2_pre_mean": float(res.r2_pre.mean()), "r2_clean_mean": float(res.r2_clean.mean()),
        "r2_own_mean": float(res.r2_own.mean()), "r2_cov_mean": float(res.r2_cov.mean()),
        "r2_null_mean": float(res.r2_null.mean()),
        "pct_reduction_mean": float((1 - res.r2_clean.mean() / res.r2_pre.mean()) * 100),
        "pct_patients_reduced": float((res.r2_clean < res.r2_pre).mean() * 100),
        "p_pre_vs_clean": float(stats.wilcoxon(res.r2_pre, res.r2_clean).pvalue),
        "p_cov_vs_pop": float(stats.wilcoxon(res.r2_cov, res.r2_clean).pvalue),
        "pct_patients_cov_better": float((res.r2_cov < res.r2_clean).mean() * 100),
        "frac_pre_mean": float(res.frac_pre.mean()), "frac_clean_mean": float(res.frac_clean.mean()),
        "frac_cov_mean": float(res.frac_cov.mean()),
        "p_frac_pre_vs_clean": float(stats.wilcoxon(res.frac_pre, res.frac_clean).pvalue),
        "frac_pre_median": float(res.frac_pre.median()), "frac_clean_median": float(res.frac_clean.median()),
        "frac_drop_median_pct": float((1 - res.frac_clean / res.frac_pre).median() * 100),
        "pct_patients_frac_reduced": float((res.frac_clean < res.frac_pre).mean() * 100),
        "obese_n": int((res.bmi >= 30).sum()),
        "lean_n": int((res.bmi < 25).sum()),
        "lean_pct_reduction": float((1 - res[res.bmi < 25].r2_clean.mean() / res[res.bmi < 25].r2_pre.mean()) * 100),
        "obese_removed": float((res.r2_pre - res.r2_clean)[res.bmi >= 30].mean()),
        "lean_removed": float((res.r2_pre - res.r2_clean)[res.bmi < 25].mean()),
        "p_removed_obese_vs_lean": float(stats.mannwhitneyu((res.r2_pre - res.r2_clean)[res.bmi >= 30],
                                                            (res.r2_pre - res.r2_clean)[res.bmi < 25]).pvalue),
        "obese_pct_reduction": float((1 - res[res.bmi >= 30].r2_clean.mean() / res[res.bmi >= 30].r2_pre.mean()) * 100),
        "obese_p": float(stats.wilcoxon(res[res.bmi >= 30].r2_pre, res[res.bmi >= 30].r2_clean).pvalue),
        "hep_retained_median": float(res.hep_retained.median()),
        "psd_change_db_median": float(res.psd_change_db.median()),
        "r_bmi_pre": float(stats.pearsonr(rb.bmi, rb.r2_pre)[0]),
        "p_bmi_pre": float(stats.pearsonr(rb.bmi, rb.r2_pre)[1]),
        "r_bmi_clean": float(stats.pearsonr(rb.bmi, rb.r2_clean)[0]),
        "p_bmi_clean": float(stats.pearsonr(rb.bmi, rb.r2_clean)[1]),
        "example_u_pop": list(map(float, u_pop_of[EXAMPLE])),
        "example_r2_pre": float(ex_data["r2_pre"].mean()), "example_r2_clean": float(ex_data["r2_clean"].mean()),
    }
    S["cleaning_benchmark"] = bench_summary()
    S["cleaning_demo"].update(example_site=EXAMPLE_SITE, hep_site=HEP_SITE)
    with open(os.path.join(HERE, "paper_stats.json"), "w") as f:
        json.dump(S, f, indent=2)
    print(json.dumps(S["cleaning_demo"], indent=1))
    plot(res, ex_data, u_pop_all, hx)


def plot(res, ex, u_pop, hx):
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 300, "savefig.dpi": 300})
    fig = plt.figure(figsize=(13, 7))
    rows = fig.add_gridspec(2, 1, hspace=0.45)
    top = rows[0].subgridspec(1, 3, width_ratios=[1.6, 1, 0.9], wspace=0.35)
    bot = rows[1].subgridspec(1, 2, width_ratios=[1.6, 1], wspace=0.3)
    sf = ex["sfreq"]
    i = int(np.argmax(ex["r2_pre"] - ex["r2_clean"]))  # channel where cleaning removes most CFA

    # a: example: 5 s of one channel, dirty vs cleaned; lines mark R-peaks (ECG used only for display)
    ax = fig.add_subplot(top[0])
    i = SITES.index(EXAMPLE_SITE)
    sl = slice(int(EXAMPLE_T0 * sf), int((EXAMPLE_T0 + 5) * sf))
    t = np.arange(sl.stop - sl.start) / sf
    for p in ex["peaks"]:
        if sl.start <= p < sl.stop:
            ax.axvline((p - sl.start) / sf, color="0.75", lw=0.8, ls="--", zorder=0)
    ax.plot([], [], color="0.75", lw=0.8, ls="--", label="R-peak")
    ax.plot(t, ex["eeg"][i, sl], color=PALETTE[1], lw=1.1, label="Raw EEG")
    ax.plot(t, ex["clean"][i, sl], color="black", lw=1.1, label="After CFAR (no ECG)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{EXAMPLE_SITE} (µV)")
    ax.set_xlim(0, 5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.18))
    ax.set_title("a", loc="left", fontsize=10, fontweight="bold")
    pt = res.set_index("patient_id").loc[EXAMPLE]
    # R² is scale-free (shape match), so report patient-mean R² plus the amplitude drop shown here
    rms = np.sqrt(np.mean(ex["clean"][i, sl] ** 2) / np.mean(ex["eeg"][i, sl] ** 2))
    ax.set_title(f"Example: {pt.sex.lower()}, {pt.age:.0f} y" + (f", BMI {pt.bmi:.1f}" if pd.notna(pt.bmi) else "") + "; R² "
                 f"{ex['r2_pre'].mean():.2f} → {ex['r2_clean'].mean():.2f}\n{EXAMPLE_SITE} amplitude "
                 f"−{(1 - rms) * 100:.0f}%", fontsize=9)

    # b: HEP (R-peak-locked average) of a second high-CFA patient, before vs. after CFAR
    ax = fig.add_subplot(top[1])
    j = SITES.index(HEP_SITE)
    ms = hx["tt"] * 1000
    ax.axvspan(-QRS_EXCLUDE_SEC * 1000, QRS_EXCLUDE_SEC * 1000, color="0.92", lw=0, zorder=0)
    ax.axvline(0, color="0.75", lw=0.8, ls="--", zorder=0)
    ax.axhline(0, color="0.75", lw=0.6, zorder=0)
    ax.plot(ms, hx["ev_pre"][j], color=PALETTE[1], lw=1.4,
            label=f"Raw EEG, R² {hx['r2_pre'][j]:.2f}")
    ax.plot(ms, hx["ev_clean"][j], color="black", lw=1.4,
            label=f"After CFAR, R² {hx['r2_clean'][j]:.2f}")
    ax.set_xlim(ms[0], ms[-1])
    ax.set_xlabel("Time from R-peak (ms)")
    ax.set_ylabel(f"{HEP_SITE} HEP (µV)")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.set_title("b", loc="left", fontsize=10, fontweight="bold")
    hp = res.set_index("patient_id").loc[HEP_EXAMPLE]
    bmi = f", BMI {hp.bmi:.1f}" if pd.notna(hp.bmi) else ""
    ax.set_title(f"HEP: {hp.sex.lower()}, {hp.age:.0f} y{bmi}\n({len(hx['peaks']):,} beats; "
                 f"grey: QRS excluded)", fontsize=9)

    # c: the filter: the one cohort-learned scalp pattern subtracted from every sample
    ax = fig.add_subplot(top[2])
    im = topo(ax, u_pop)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("Weight (a.u.)", fontsize=7.5)
    ax.set_title("c", loc="left", fontsize=10, fontweight="bold")
    ax.set_title("CFAR filter", fontsize=9)

    # d: cleaning trade-off as two side-by-side bar charts (ICA+CFAR, CFAR+ECG regression kept in stats, not plotted),
    # pipelines sorted by CFA left, most to least
    from matplotlib.patches import Patch
    B = bench_summary()
    rows = sorted([b for b in BENCH if b[0] not in ("ica", "ica_cfar_raw", "cfar_ecgreg_dir")], key=lambda b: -B[f"{b[0]}_pow_pct"])
    x = np.arange(len(rows))
    dd = bot[0].subgridspec(1, 2, wspace=0.3)
    for col, (metric, ylabel, scale) in enumerate([("pow_pct", "ECG-explained EEG variance left\n(% of raw EEG; lower = better)", 1),
                                                   ("hep", "Neural signal preserved\n(%; higher = better)", 100)]):
        ax = fig.add_subplot(dd[col])
        for xi, (k, label, ecg, color) in zip(x, rows):
            if f"{k}_{metric}" not in B:
                continue
            v = B[f"{k}_{metric}"] * scale
            err = np.abs(np.array(B[f"{k}_{metric}_ci"]) * scale - v)[:, None]
            ax.bar(xi, v, yerr=err, width=0.7, color=color, alpha=0.85, hatch="///" if ecg else None,
                   edgecolor="white" if ecg else color, lw=0, error_kw={"lw": 1, "capsize": 2})
            ax.text(xi, v + err[1, 0], f"{v:.0f}%", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels([b[1].replace("-", "-\n").replace(" ", "\n") for b in rows], fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_ylim(0, 115 if metric == "hep" else max(B[f"{b[0]}_pow_pct_ci"][1] for b in rows) * 1.2)
        if col == 0:
            ax.set_title("d", loc="left", fontsize=10, fontweight="bold")
            ax.legend(handles=[Patch(facecolor="0.35", label="No ECG"),
                               Patch(facecolor="0.35", hatch="///", edgecolor="white", lw=0, label="Uses ECG")],
                      frameon=False, fontsize=7.5, loc="upper center", ncol=2, bbox_to_anchor=(1.15, -0.25))
    fig.text(0.5 * (dd[0].get_position(fig).x0 + dd[1].get_position(fig).x1), dd[0].get_position(fig).y1 + 0.012,
             f"Cleaning pipelines (n = {B['n']:,} patients; median, 95% CI)", ha="center", fontsize=9)

    # e: who needs cleaning: CFA by BMI, all patients, before (solid) and after (dashed)
    ax = fig.add_subplot(bot[1])
    edges = [15, 25, 30, 35, 40, 60]
    for col, ls, color, tag in [("r2_pre", "-", PALETTE[1], "Raw EEG"), ("r2_clean", "--", "black", "After CFAR")]:
        mid, m_, s_ = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            b_ = res[(res.bmi >= lo) & (res.bmi < hi)][col]
            if len(b_) >= 10:
                mid.append((lo + hi) / 2); m_.append(b_.mean()); s_.append(b_.sem())
        ax.errorbar(mid, m_, yerr=s_, color=color, ls=ls, marker="o", ms=4, capsize=2, lw=1.4, label=tag)
    ax.set_xlabel("BMI (kg/m²)")
    ax.set_ylabel("Mean CFA R² (± SEM)")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.set_title("e", loc="left", fontsize=10, fontweight="bold")
    ax.set_title(f"CFA by BMI\n(n = {res.bmi.notna().sum():,})", fontsize=9)

    fig.savefig(os.path.join(FIG_DIR, "fig5_cleaning_demo.png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "fig5_cleaning_demo.pdf"), bbox_inches="tight")
    plt.close(fig)


def replot():
    """Redraw the figure from saved results (no 30-min rerun)."""
    res = pd.read_parquet(os.path.join(HERE, "cleaning_demo.parquet"))
    with open(os.path.join(HERE, "paper_stats.json")) as f:
        CD = json.load(f)["cleaning_demo"]
    u_pop = np.array([CD["u_pop"][s] for s in SITES])
    cfa = pd.read_parquet(os.path.join(HERE, "cfa_combined.parquet"), columns=["patient_id", "edf_path", "window_start_s"])
    r = cfa[cfa.patient_id == EXAMPLE].iloc[0]
    # ponytail: population pattern, not the example's fold pattern (cos ~0.999); full run uses the fold one
    _, ex = evaluate(load((EXAMPLE, r.edf_path, float(r.window_start_s))), u_pop, u_pop, 0)
    r = cfa[cfa.patient_id == HEP_EXAMPLE].iloc[0]
    _, hx = evaluate(load((HEP_EXAMPLE, r.edf_path, float(r.window_start_s))), u_pop, u_pop, 0)
    with open(os.path.join(HERE, "paper_stats.json")) as f:
        S = json.load(f)
    S["cleaning_demo"].update(example_r2_pre=float(ex["r2_pre"].mean()), example_r2_clean=float(ex["r2_clean"].mean()))
    S["cleaning_demo"].pop("example_u_pop", None)
    S["cleaning_benchmark"] = bench_summary()
    S["cleaning_demo"].update(example_site=EXAMPLE_SITE, hep_site=HEP_SITE)
    with open(os.path.join(HERE, "paper_stats.json"), "w") as f:
        json.dump(S, f, indent=2)
    plot(res, ex, u_pop, hx)


if __name__ == "__main__":
    import sys
    replot() if "--plot-only" in sys.argv else main()
