#!/usr/bin/env python3
"""Main figures for the v3 manuscript (NeuroImage). Each population figure opens with an individual example.

  Figure 1  fig1_v3_cfa_cfar.png        CFA in one typical patient (19 electrodes, Beilinson referential),
                                         CFA across the cohort, and how CFAR removes it
  Figure 2  fig5_cleaning_demo.png      (make_cleaning_demo.py) how much CFAR cleans
  Figure 3  fig3_v3_beilinson.png       independent cohort: ps-CFAR vs. ICA (make_beilinson_figure.py)
  Figure 4  fig4_v3_demographics.png    examples (high-BMI man, low-BMI woman), then CFA vs. age, sex, BMI
                                         before and after CFAR
  Figure 5  fig5_v3_diagnosis.png       examples (obesity, heart transplant), then CFA by diagnosis before
                                         and after CFAR
  Figure 6  fig6_v3_duration.png        one patient's HEP at 5/10/20/60 min, then cohort duration effects

Stats go to paper_stats.json["v3"]. Run: venv/bin/python "paper utils/Paper CFA/make_v3_figures.py" [--fig 1 4 5 6]
"""
import argparse, json, math, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats
from statsmodels.stats.multitest import multipletests

import make_cleaning_demo as cd
from channel_utils import CHANNEL_ORDER, canonicalize
from ica_ecg_component_variance import QRS_EXCLUDE_SEC, channel_names, epoch_around_r_peaks, quality

HERE = cd.HERE
FIG_DIR = cd.FIG_DIR
P = cd.PALETTE
RAW_C, CFAR_C, OWN_C = P[1], "black", P[2]
SEX_C = {"Female": P[3], "Male": P[0]}
SITES19 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
STATS = json.load(open(os.path.join(HERE, "paper_stats.json")))
U_POP = np.array([STATS["cleaning_demo"]["u_pop"][s] for s in cd.SITES])
EX19 = "0345-105"  # Beilinson referential recording closest to the cohort-average CFA
SEX_EXAMPLES = [("Man, high BMI", "I0002150028912"), ("Woman, low BMI", "I0002150035125")]
MONTAGE = mne.channels.make_standard_montage("standard_1020")
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
YLAB = "Patient-mean CFA R²"


def letter(ax, t, title=None, x=-0.13):
    ax.text(x, 1.12, t.upper(), transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
    if title:
        ax.set_title(title, fontsize=8.5)


def topo(ax, site_values, vlim, cmap="Reds", names=True):
    sites = [c for c in CHANNEL_ORDER if c in site_values]
    info = mne.create_info(sites, 100.0, "eeg")
    info.set_montage(MONTAGE)
    im, _ = mne.viz.plot_topomap(np.array([site_values[c] for c in sites]), info, axes=ax, show=False, cmap=cmap,
                                 vlim=vlim, contours=0, sensors=not names, names=sites if names else None,
                                 extrapolate="head")
    for t in ax.texts:
        t.set_fontsize(5.5)
    return im


def load_window(path, start, dur):
    """QC'd window, any montage: canonical site names, EEG, ECG (µV), R-peaks, sfreq."""
    raw = mne.io.read_raw_edf(path, preload=False, encoding="latin1", verbose="ERROR")
    eeg_names, ecg_name = channel_names(raw)
    seg = raw.copy().pick(eeg_names + [ecg_name]).crop(start, start + dur, include_tmax=False)
    seg.load_data(verbose="ERROR")
    _, qc = quality(seg, eeg_names, ecg_name)
    sf = float(seg.info["sfreq"])
    seg.filter(1.0, min(100.0, sf / 2 - 0.5), verbose="ERROR")
    data = seg.get_data() * 1e6
    canon = canonicalize(pd.Series([n.replace("EEG ", "").strip() for n in eeg_names]))  # Beilinson labels carry "EEG "
    keep = [i for i, c in enumerate(canon) if isinstance(c, str)]
    peaks = np.asarray(qc["r_peak_samples"])
    if "0345-" in path:  # Beilinson leads: T waves get counted as beats without the R-height filter
        from beilinson_cfar_vs_ica import robust_peaks
        peaks = robust_peaks(data[-1], sf)[0]
    return [canon[i] for i in keep], data[keep], data[-1], peaks, sf


def window_of(pid, fname="cfa_combined.parquet"):
    w = pd.read_parquet(os.path.join(HERE, fname), columns=["patient_id", "edf_path", "window_start_s", "window_duration_s"])
    w = w[w.patient_id == pid].iloc[0]
    return w.edf_path, float(w.window_start_s), float(w.window_duration_s)


def hep_panel(ax, tt, traces, ecg_ev, ylab=None, note=None):
    """HEP lines [(y, color, label, ls)] over a grey QRS band, ECG average (scaled) on a hidden twin axis."""
    tw = ax.twinx()
    tw.axvspan(-QRS_EXCLUDE_SEC * 1e3, QRS_EXCLUDE_SEC * 1e3, color="0.92", lw=0, zorder=0)
    tw.plot(tt * 1e3, ecg_ev, color="0.65", lw=0.7, ls=":")
    tw.set_yticks([])
    tw.spines[["top", "right"]].set_visible(False)
    for y, c, lab, ls in traces:
        ax.plot(tt * 1e3, y, color=c, lw=1.1, ls=ls, label=lab)
    ax.set_zorder(tw.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_xlim(tt[0] * 1e3, tt[-1] * 1e3)
    ax.set_xlabel("Time from R-peak (ms)")
    if ylab:
        ax.set_ylabel(ylab)
    if note:
        ax.text(0.02, 0.97, note, transform=ax.transAxes, va="top", fontsize=7)


# =============================================================================
# Figure 1
# =============================================================================
FIG1_EX, FIG1_T0 = "I0002150007970", 1560.0  # chosen by search: heart samples stand out along u on both sides, brain cloud ~isotropic


def synthetic_beats(sf=250, sec=5.0, hr=72, seed=3):
    """Illustration only: PQRST ECG, and EEG = 1/f brain noise + a small heartbeat-shaped field."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sec * sf)) / sf
    rr = 60 / hr
    beats = np.arange(0.35, sec, rr) + rng.normal(0, 0.02, len(np.arange(0.35, sec, rr)))
    g = lambda c, w, a: a * np.exp(-0.5 * ((t[:, None] - (beats + c)) / w) ** 2).sum(1)
    ecg = g(-0.16, 0.025, 0.12) + g(-0.03, 0.008, -0.15) + g(0, 0.012, 1.0) + g(0.03, 0.01, -0.25) + g(0.25, 0.045, 0.3)
    f = np.fft.rfftfreq(len(t), 1 / sf)
    spec = rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))
    spec[1:] /= f[1:] ** 0.8  # pink (1/f) background
    spec[0] = 0
    brain = np.fft.irfft(spec, len(t))
    brain = brain / brain.std() + 0.4 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6))  # plus some alpha
    brain = brain / brain.std()
    eeg = brain + 1.6 * (g(0, 0.014, 1.0) + g(0.25, 0.05, 0.35) + g(0.03, 0.012, -0.3))
    return t, ecg, eeg, (beats * sf).astype(int), sf


def fig1():
    """CFAR only: how EEG + ECG become a HEP (illustration), the algorithm, and what it does to real EEG."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig = plt.figure(figsize=(11, 12.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.85, 1.25, 1.05], hspace=0.32)
    g0 = gs[0].subgridspec(1, 2, width_ratios=[1.7, 1], wspace=0.22)
    g2 = gs[2].subgridspec(1, 2, width_ratios=[1.6, 1], wspace=0.25)
    HEART, BRAIN, GREEN = "#b2182b", P[0], OWN_C

    # A: illustration -- ECG, EEG, R-peaks, one window per beat
    t, ecg_s, eeg_s, bt, sfs = synthetic_beats()
    ax = fig.add_subplot(g0[0])
    pre, post = 0.3, 0.4
    for k, b_ in enumerate(bt):
        ax.axvspan(max(0, t[b_] - pre), min(t[-1], t[b_] + post), color=["#e8f1f8", "#fbeee6"][k % 2], lw=0, zorder=0)
        ax.axvline(t[b_], color=HEART, lw=0.7, ls="--", zorder=1)
    ax.plot(t, ecg_s * 2.2 + 4.2, color=HEART, lw=1.1)
    ax.plot(t, eeg_s * 0.55, color=BRAIN, lw=0.9)
    ax.text(-0.02, 4.4, "ECG", color=HEART, fontweight="bold", ha="right", va="center", transform=ax.get_yaxis_transform())
    ax.text(-0.02, 0, "EEG", color=BRAIN, fontweight="bold", ha="right", va="center", transform=ax.get_yaxis_transform())
    b0 = bt[1]
    for lab, dt, yy in [("P", -0.16, 4.75), ("QRS", 0, 6.6), ("T", 0.25, 5.15)]:
        ax.text(t[b0] + dt, yy, lab, color=HEART, ha="center", fontsize=7.5)
    ax.annotate("", xy=(t[bt[2]] + post, -2.4), xytext=(t[bt[2]] - pre, -2.4), arrowprops=dict(arrowstyle="<->", color="0.3", lw=0.8))
    ax.text(t[bt[2]] + 0.05, -2.95, "one window: −300 to +400 ms", ha="center", fontsize=7, color="0.3")
    ax.set(xlim=(0, t[-1]), ylim=(-3.3, 7.2), xlabel="Time (s)")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    letter(ax, "a", "Cut the EEG at every heartbeat (R-peak from the ECG)", x=-0.06)

    # B: windows stacked, then averaged = HEP
    ax = fig.add_subplot(g0[1])
    t2, ecg2, eeg2, bt2, _ = synthetic_beats(sec=300, seed=5)
    n0, n1 = int(pre * sfs), int(post * sfs)
    ok = [b_ for b_ in bt2 if b_ - n0 >= 0 and b_ + n1 < len(t2)]
    ep = np.array([eeg2[b_ - n0:b_ + n1] for b_ in ok]) * 0.55
    ee = np.array([ecg2[b_ - n0:b_ + n1] for b_ in ok])
    te = (np.arange(-n0, n1) / sfs) * 1e3
    for k in range(6):
        ax.plot(te, ep[k] + 9 - 1.6 * k, color=BRAIN, lw=0.6, alpha=0.7)
    ax.text(te[-1] + 15, 9 - 1.6 * 2.5, f"single\nbeats", fontsize=7, color="0.35", va="center")
    ax.annotate("", xy=(0, -0.3), xytext=(0, 0.9), arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1))
    ax.text(15, 0.3, f"average of {len(ep)} beats", fontsize=7, color="0.35", va="center")
    ax.axvspan(-50, 50, ymin=0, ymax=0.3, color="0.9", lw=0, zorder=0)
    ax.plot(te, ee.mean(0) * 2 - 2.6, color=HEART, lw=1, ls=":")
    ax.plot(te, ep.mean(0) * 1.6 - 2.6, color=BRAIN, lw=2)
    ax.text(te[-1] + 15, -2.2, "HEP", color=BRAIN, fontweight="bold", fontsize=8)
    ax.text(te[-1] + 15, -3.9, "ECG avg.", color=HEART, fontsize=7)
    ax.set(xlim=(te[0], te[-1]), ylim=(-4.3, 10.5), xlabel="Time from R-peak (ms)")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    letter(ax, "b", "Stack and average: the HEP copies the ECG", x=-0.06)

    # C: the algorithm, two columns
    ax = fig.add_subplot(gs[1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    TR, INF = "#fdf0e6", "#eaf3fa"
    for x0, w, col, ec, title in [(0.0, 0.53, TR, "#d8894a", "TRAIN once, on patients WITH an ECG"),
                                  (0.58, 0.42, INF, "#5b9bd5", "APPLY to any recording, NO ECG")]:
        ax.add_patch(FancyBboxPatch((x0, 0.0), w, 0.98, boxstyle="round,pad=0.005,rounding_size=0.02", fc=col, ec=ec, lw=1.2))
        ax.text(x0 + w / 2, 0.94, title, ha="center", va="center", fontsize=9, fontweight="bold", color=ec)
    pid, x, ecg6, pk6, sf6 = cd.load((FIG1_EX, *window_of(FIG1_EX)[:2]))
    hep6 = epoch_around_r_peaks(x, pk6, sf6).mean(0)
    clean = cd.project_out(x, U_POP)
    rows = [0.72, 0.5, 0.28, 0.06]
    hgt = 0.17

    def step(x0, y0, num, text):
        ax.add_patch(plt.Circle((x0 + 0.012, y0 + hgt / 2), 0.016, color="0.25", transform=ax.transData, clip_on=False))
        ax.text(x0 + 0.012, y0 + hgt / 2, str(num), color="white", ha="center", va="center", fontsize=8, fontweight="bold")
        ax.text(x0 + 0.215, y0 + hgt / 2, text, va="center", fontsize=7.8)

    def icon(x0, y0, w=0.16):
        return ax.inset_axes([x0, y0, w, hgt], transform=ax.transData)

    s1 = int(FIG1_T0 * sf6)
    tr = [
        ("EEG + ECG recorded together", lambda a_: [a_.plot(ecg6[s1:s1 + int(3 * sf6)] / np.ptp(ecg6[s1:s1 + int(3 * sf6)]) * 2 + 1.6, color=HEART, lw=0.7),
                                                    a_.plot(x[cd.SITES.index("F4"), s1:s1 + int(3 * sf6)] / 60, color=BRAIN, lw=0.6)]),
        ("Windows at each R-peak, averaged\n→ the patient's HEP", None),
        ("HEP across electrodes (channels × time);\nSVD → its scalp pattern uᵢ", None),
        ("Average uᵢ over patients, weighted by\nCFA R² → one filter u", None),
    ]
    for k, (txt, draw) in enumerate(tr):
        step(0.02, rows[k], k + 1, txt)
        ia = icon(0.05, rows[k])
        if k == 0:
            draw(ia)
        elif k == 1:
            epx = epoch_around_r_peaks(x[cd.SITES.index("F4")], pk6, sf6)
            for row in epx[:25]:
                ia.plot(row, color="0.75", lw=0.4)
            ia.plot(epx.mean(0), color=BRAIN, lw=1.5)
            ia.set_ylim(np.percentile(epx, 2), np.percentile(epx, 98))
        elif k == 2:
            ia.remove()
            ia = ax.inset_axes([0.085, rows[k], 0.125, hgt], transform=ax.transData)
            ia.imshow(hep6, aspect="auto", cmap="RdBu_r", vmin=-np.abs(hep6).max(), vmax=np.abs(hep6).max())
            ia.set_yticks(range(6), cd.SITES, fontsize=5)
            ia.tick_params(length=0, labelbottom=False)
        else:
            ia.remove()
            ia = ax.inset_axes([0.07, rows[k], 0.12, hgt], transform=ax.transData)
            topo(ia, dict(zip(cd.SITES, U_POP)), (-np.abs(U_POP).max(), np.abs(U_POP).max()), cmap="RdBu_r")
        if k != 2 and k != 3:
            ia.axis("off")
        if k < 3:
            ax.add_patch(FancyArrowPatch((0.13, rows[k] - 0.005), (0.13, rows[k + 1] + hgt + 0.005), arrowstyle="-|>", mutation_scale=9, color="0.35"))
    inf_rows = [0.68, 0.38, 0.08]
    sl = slice(s1, s1 + int(3 * sf6))
    for k, (txt) in enumerate(["New EEG (any patient, any time)", "Remove u from every sample:\nx ← x − u (uᵀx)", "Clean EEG"]):
        step(0.6, inf_rows[k], k + 1, txt)
        ia = icon(0.63, inf_rows[k])
        if k == 1:
            ia.axis("off")
            ia.text(0.5, 0.5, "(I − uuᵀ)", ha="center", va="center", fontsize=12, fontweight="bold", color="#2166ac",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#2166ac"))
        else:
            src = x if k == 0 else clean
            for j, sname in enumerate(["F4", "C4", "O1"]):
                ia.plot(src[cd.SITES.index(sname), sl] / 80 - j, color=BRAIN if k == 0 else GREEN, lw=0.5)
            ia.axis("off")
        if k < 2:
            ax.add_patch(FancyArrowPatch((0.71, inf_rows[k] - 0.005), (0.71, inf_rows[k + 1] + hgt + 0.005), arrowstyle="-|>", mutation_scale=9, color="0.35"))
    # u travels below the steps and up the gap between the columns, into step 2 of APPLY
    yb, xg, ye = 0.025, 0.555, inf_rows[1] + hgt / 2
    ax.plot([0.13, 0.13, xg, xg], [rows[3] + 0.005, yb, yb, ye], color=HEART, lw=1.6, solid_joinstyle="round")
    ax.add_patch(FancyArrowPatch((xg, ye), (0.615, ye), arrowstyle="-|>", mutation_scale=12, color=HEART, lw=1.6))
    ax.text(xg, (yb + ye) / 2, "filter u", color=HEART, fontweight="bold", fontsize=9, rotation=90, ha="center", va="center",
            bbox=dict(fc="white", ec="none", pad=1))
    letter(ax, "c", "The CFAR algorithm", x=-0.01)

    # D: real EEG before/after CFAR, ECG with P/QRS/T
    ax = fig.add_subplot(g2[0])
    s0, n = int(FIG1_T0 * sf6), int(4 * sf6)
    tt = np.arange(n) / sf6
    beats = pk6[(pk6 >= s0) & (pk6 < s0 + n)] - s0
    for b_ in beats:
        ax.axvspan(b_ / sf6 - QRS_EXCLUDE_SEC, b_ / sf6 + QRS_EXCLUDE_SEC, color="#f4c7c7", lw=0, zorder=0)
        ax.axvspan(b_ / sf6 + 0.2, b_ / sf6 + 0.38, color="#fbe3e3", lw=0, zorder=0)
    sig = {sn: x[cd.SITES.index(sn), s0:s0 + n] for sn in ["F4", "O1"]}
    gap = 1.3 * max(np.ptp(v) for v in sig.values())
    off = 0
    for sn, col in [("F4", P[0]), ("O1", P[1])]:
        i_ = cd.SITES.index(sn)
        base = np.median(x[i_, s0:s0 + n])
        ax.plot(tt, x[i_, s0:s0 + n] - base + off, color=col, lw=0.8, alpha=0.45)
        ax.plot(tt, clean[i_, s0:s0 + n] - np.median(clean[i_, s0:s0 + n]) + off, color=GREEN, lw=1.0)
        ax.text(-0.02, off, sn, color=col, ha="right", va="center", fontweight="bold", transform=ax.get_yaxis_transform())
        off -= gap
    e = ecg6[s0:s0 + n]
    e = (e - np.median(e)) / np.ptp(e) * gap * 0.8 + off
    ax.plot(tt, e, color=HEART, lw=0.9)
    ax.text(-0.02, off, "ECG", color=HEART, ha="right", va="center", fontweight="bold", transform=ax.get_yaxis_transform())
    if len(beats):
        b_ = beats[0] / sf6
        top = e.max()
        for lab, dt in [("QRS", 0), ("T", 0.29)]:
            ax.text(b_ + dt, top + 0.08 * gap, lab, color=HEART, ha="center", fontsize=7)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 4)
    ax.set_xlabel("Time (s)")
    ax.legend(handles=[plt.Line2D([], [], color="0.5", alpha=0.5, label="Raw EEG"), plt.Line2D([], [], color=GREEN, label="CFAR"),
                       Patch(color="#f4c7c7", label="QRS"), Patch(color="#fbe3e3", label="T wave")],
              frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    letter(ax, "d", "EEG = brain + heart; CFAR removes the heartbeat, no ECG needed", x=-0.07)

    # E: each sample one dot in the F4-O1 plane
    ax = fig.add_subplot(g2[1])
    i4, i1 = cd.SITES.index("F4"), cd.SITES.index("O1")
    seg = slice(s0, s0 + int(60 * sf6))
    X = x[[i4, i1], seg]
    X = X - X.mean(1, keepdims=True)
    during = np.zeros(X.shape[1], bool)
    for b_ in pk6[(pk6 >= seg.start) & (pk6 < seg.stop)] - seg.start:
        during[max(0, b_ - int(0.025 * sf6)):b_ + int(0.025 * sf6)] = True  # the QRS peak itself
    u2 = U_POP[[i4, i1]] / np.linalg.norm(U_POP[[i4, i1]])
    X2 = X - np.outer(u2, u2 @ X)
    rng = np.random.default_rng(0)
    bsub = rng.choice(np.flatnonzero(~during), 2500, replace=False)
    hsub = np.flatnonzero(during)[::2]
    lim = np.percentile(np.abs(X[:, during]), 98) * 1.1
    inview = hsub[(np.abs(X[:, hsub]) < 0.9 * lim).all(0)]
    far = inview[np.argsort(-np.abs(u2 @ X[:, inview]))][::max(1, len(inview) // 40)][:14]
    for j in far:  # projection lines, heart sample -> its CFAR value
        ax.plot([X[0, j], X2[0, j]], [X[1, j], X2[1, j]], color="0.6", lw=0.5, zorder=1)
    ax.scatter(*X[:, bsub], s=3, color="#9ecae1", lw=0, label="Between beats: brain", zorder=2)
    ax.scatter(*X[:, hsub], s=6, color="#d6604d", lw=0, label="During QRS: brain + heart", zorder=3)
    w2 = np.array([-u2[1], u2[0]])
    ax.plot(*np.outer(w2, [-lim, lim]), color=GREEN, ls="--", lw=1.2, zorder=2)
    ax.scatter(*X2[:, np.r_[bsub[::3], hsub]], s=4, color=GREEN, lw=0, label="CFAR: kept part ⊥ u", zorder=4)
    ax.annotate("", xy=u2 * lim * 0.92, xytext=-u2 * lim * 0.92, arrowprops=dict(arrowstyle="<->", color=HEART, lw=1.6), zorder=5)
    ax.text(*(u2 * lim * 0.7 + w2 * lim * 0.12), "CFA direction u", color=HEART, fontsize=7.5, fontweight="bold", ha="center")
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="F4 (µV)", ylabel="O1 (µV)")
    ax.set_aspect("equal")
    ax.axhline(0, color="0.88", lw=0.5, zorder=0)
    ax.axvline(0, color="0.88", lw=0.5, zorder=0)
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), markerscale=3)
    letter(ax, "e", "Each sample is a dot; CFAR projects u out", x=-0.22)
    STATS_V3["fig1_cfar_example"] = {"patient_id": FIG1_EX, "t0": FIG1_T0}
    save(fig, "fig1_v3_cfar")


def fig2():
    fig = plt.figure(figsize=(11, 7.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.05], hspace=0.45)
    g0 = gs[0].subgridspec(1, 3, width_ratios=[0.9, 1.4, 1], wspace=0.35)
    g1 = gs[1].subgridspec(1, 3, width_ratios=[0.8, 1, 1.1], wspace=0.45)

    # a-c: one typical patient, 19 electrodes
    sites, eeg, ecg, peaks, sf = load_window(*window_of(EX19))
    idx = [sites.index(s) for s in SITES19 if s in sites]
    sites, eeg = [sites[i] for i in idx], eeg[idx]
    r2, ev, ecg_ev = cd.cfa_r2(eeg, ecg, peaks, sf)
    meta = pd.read_parquet(os.path.join(HERE, "demographics_combined.parquet"))
    m = meta[meta.patient_id == EX19]
    who = f"{m.sex.iat[0].lower()}, {m.age.iat[0]:.0f} y" if len(m) and pd.notna(m.age.iat[0]) else "Beilinson patient"
    ax = fig.add_subplot(g0[0])
    im = topo(ax, dict(zip(sites, r2)), (0, 1))
    cb = fig.colorbar(im, ax=ax, shrink=0.65, pad=0.02)
    cb.set_label("CFA R²", fontsize=7)
    letter(ax, "a", f"One patient, {len(sites)} electrodes\nmean R² {r2.mean():.2f} (cohort median {STATS_V3['cohort_median_r2']:.2f})")
    si = int(np.argmax(r2))
    t0 = int(60 * sf)
    n = int(5 * sf)
    ax = fig.add_subplot(g0[1])
    t = np.arange(n) / sf
    for p in peaks[(peaks >= t0) & (peaks < t0 + n)]:
        ax.axvspan((p - t0) / sf - QRS_EXCLUDE_SEC, (p - t0) / sf + QRS_EXCLUDE_SEC, color="0.92", lw=0)
    ax.plot(t, eeg[si, t0:t0 + n], color=P[0], lw=0.8, label=f"{sites[si]} EEG")
    e = ecg[t0:t0 + n]
    ax.plot(t, (e - e.mean()) / e.std() * eeg[si, t0:t0 + n].std() - 3 * eeg[si, t0:t0 + n].std(), color="0.5", lw=0.7, label="ECG (scaled)")
    ax.set_xlim(0, 5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Time (s)")
    ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=2)
    letter(ax, "b", f"{sites[si]}: EEG and ECG, 5 s", x=-0.05)
    ax = fig.add_subplot(g0[2])
    hep_panel(ax, cd.epoch_times(sf), [(ev[si], P[0], "HEP", "-")], ecg_ev, f"{sites[si]} HEP (µV)",
              f"R² {r2[si]:.2f}\n{len(peaks):,} beats")
    letter(ax, "c", "HEP vs. ECG average (dotted)")
    STATS_V3["fig1_example"] = {"patient_id": EX19, "n_sites": len(sites), "mean_r2": float(r2.mean()), "site": sites[si],
                                "site_r2": float(r2[si]), "n_beats": int(len(peaks)), "who": who}

    # d-f: cohort
    cfa = pd.read_parquet(os.path.join(HERE, "cfa_combined.parquet"), columns=["patient_id", "eeg_channel", "cfa_r2_excl_qrs", "cfa_r2_full_epoch"])
    ax = fig.add_subplot(g1[0])
    data = [cfa.cfa_r2_full_epoch.dropna().values, cfa.cfa_r2_excl_qrs.dropna().values]
    parts = ax.violinplot(data, showmedians=True, widths=0.8)
    for pc, c in zip(parts["bodies"], [P[1], P[0]]):
        pc.set(facecolor=c, alpha=0.6, edgecolor="black", linewidth=0.5)
    for k in ("cmedians", "cbars", "cmins", "cmaxes"):
        parts[k].set(color="black", linewidth=0.7)
    ax.set_xticks([1, 2], ["Full epoch", "Outside\nQRS"])
    ax.set_ylabel("Channel CFA R²")
    letter(ax, "d", f"All channels (n = {len(data[1]):,})", x=-0.3)
    cfa["canon"] = canonicalize(cfa.eeg_channel)
    site = cfa.dropna(subset=["canon"])
    sn = site.groupby("canon").patient_id.nunique()
    site = site[site.canon.isin(sn[sn >= 100].index)]
    per = site.groupby("canon").cfa_r2_excl_qrs.agg(["mean", "median"])
    ax = fig.add_subplot(g1[1])
    im = topo(ax, per["mean"].to_dict(), (0, float(per["mean"].max())))
    cb = fig.colorbar(im, ax=ax, shrink=0.65, pad=0.02)
    cb.set_label("Mean CFA R²", fontsize=7)
    letter(ax, "e", f"Mean CFA by electrode ({cfa.patient_id.nunique():,} patients)")
    ax = fig.add_subplot(g1[2])
    order = per["median"].sort_values().index.tolist()
    box = [site.loc[site.canon == c, "cfa_r2_excl_qrs"].dropna().values for c in order]
    bp = ax.boxplot(box, tick_labels=order, vert=False, patch_artist=True, widths=0.65, showfliers=False, whis=0,
                    medianprops={"color": P[6]})
    for line in bp["whiskers"] + bp["caps"]:
        line.set_visible(False)
    mm = per.loc[order, "mean"].values
    for patch, v in zip(bp["boxes"], (mm - mm.min()) / (mm.max() - mm.min())):
        patch.set_facecolor(plt.get_cmap("Reds")(0.25 + 0.65 * v))
    ax.set_xlabel("CFA R² (IQR, median)")
    ax.tick_params(axis="y", labelsize=6.5)
    letter(ax, "f", "CFA by electrode", x=-0.2)

    save(fig, "fig2_v3_cfa_cohort")


# =============================================================================
# Figure 4: demographics, before and after CFAR
# =============================================================================
def ci_mean(v):
    v = np.asarray(v)
    return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v))


def fig4():
    fig = plt.figure(figsize=(11, 8.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.5)
    top = gs[0].subgridspec(1, 4, wspace=0.45, width_ratios=[0.85, 1, 0.85, 1])
    bot = gs[1].subgridspec(1, 3, wspace=0.35)
    demo = pd.read_parquet(os.path.join(HERE, "demographics_combined.parquet"))
    bmi = pd.read_parquet(os.path.join(HERE, "bmi_combined.parquet"))
    recs = {pid: load_window(*window_of(pid, "cfa_variance_explained_60min.parquet")) for _, pid in SEX_EXAMPLES}
    common = [s for s in recs[SEX_EXAMPLES[0][1]][0] if s in recs[SEX_EXAMPLES[1][1]][0]]
    out = []
    for k, (lab, pid) in enumerate(SEX_EXAMPLES):
        sites, eeg, ecg, peaks, sf = recs[pid]
        r2, ev, ecg_ev = cd.cfa_r2(eeg, ecg, peaks, sf)
        own = cd.project_out(eeg, cd.cfa_pattern(eeg, peaks, sf))
        r2o, evo, _ = cd.cfa_r2(own, ecg, peaks, sf)
        p = demo[demo.patient_id == pid].iloc[0]
        b = float(bmi.loc[bmi.bdsp_patient_id == p.bdsp_patient_id, "bmi"].iat[0])
        ax = fig.add_subplot(top[2 * k])
        im = topo(ax, dict(zip(sites, r2)), (0, 1))
        letter(ax, "ab"[k], f"{lab}\n{p.sex}, {p.age:.0f} y, BMI {b:.1f}; R² {r2.mean():.2f}", x=-0.1)
        s = "O2" if "O2" in common else common[0]
        si = sites.index(s)
        ax = fig.add_subplot(top[2 * k + 1])
        hep_panel(ax, cd.epoch_times(sf), [(ev[si], RAW_C, f"Raw (R² {r2[si]:.2f})", "-"),
                                           (evo[si], OWN_C, f"ps-CFAR (R² {r2o[si]:.2f})", "-")], ecg_ev, f"{s} HEP (µV)")
        ax.legend(frameon=False, fontsize=6.5, loc="lower right")
        ax.set_title(f"{s} HEP, {len(peaks):,} beats", fontsize=8.5)
        out.append({"label": lab, "patient_id": pid, "sex": str(p.sex), "age": float(p.age), "bmi": b, "mean_r2": float(r2.mean()),
                    "mean_r2_own": float(r2o.mean()), "site": s, "site_r2": float(r2[si]), "site_r2_own": float(r2o[si])})
    cb = fig.colorbar(im, cax=fig.add_axes([0.08, 0.56, 0.008, 0.12]))
    cb.set_label("R²", fontsize=7)
    cb.ax.yaxis.set_label_position("left")
    cb.ax.yaxis.set_ticks_position("left")
    STATS_V3["fig4_examples"] = out

    d = pd.read_parquet(os.path.join(HERE, "cleaning_demo.parquet"))
    S4 = {"n": int(len(d))}
    # c: age
    ax = fig.add_subplot(bot[0])
    a = d.dropna(subset=["age"])
    for col, c, lab in [("r2_pre", RAW_C, "Raw"), ("r2_clean", CFAR_C, "CFAR")]:
        ax.scatter(a.age, a[col], s=3, alpha=0.06, color=c, lw=0, rasterized=True)
        r, pv = stats.pearsonr(a.age, a[col])
        k_, b_ = np.polyfit(a.age, a[col], 1)
        xs = np.linspace(a.age.min(), a.age.max(), 50)
        ax.plot(xs, k_ * xs + b_, color=c, lw=1.8, label=f"{lab}: r = {r:.2f}")
        S4[f"age_r_{col}"], S4[f"age_p_{col}"] = float(r), float(pv)
    ax.set(xlabel="Age (years)", ylabel=YLAB)
    ax.legend(frameon=False, fontsize=7)
    letter(ax, "c", f"CFA vs. age (n = {len(a):,})")
    # d: sex
    ax = fig.add_subplot(bot[1])
    for j, (col, lab) in enumerate([("r2_pre", "Raw"), ("r2_clean", "CFAR")]):
        g = [d.loc[d.sex == s, col].values for s in SEX_C]
        bp = ax.boxplot(g, positions=[j * 2.6, j * 2.6 + 1], widths=0.75, patch_artist=True, showfliers=False,
                        medianprops={"color": "black"})
        for patch, c in zip(bp["boxes"], SEX_C.values()):
            patch.set(facecolor=c, alpha=0.7 if j == 0 else 0.35, lw=0)
        pv = stats.mannwhitneyu(*g).pvalue
        diff = np.median(g[1]) - np.median(g[0])
        ax.text(j * 2.6 + 0.5, 1.0, f"Δ {diff:+.3f}\np = {pv:.1g}", ha="center", fontsize=7)
        S4[f"sex_p_{col}"], S4[f"sex_median_diff_{col}"] = float(pv), float(diff)
        S4[f"sex_median_{col}"] = {s: float(np.median(v)) for s, v in zip(SEX_C, g)}
    ax.set_xticks([0, 1, 2.6, 3.6], ["F", "M", "F", "M"])
    ax.text(0.5, -0.17, "Raw", transform=ax.get_xaxis_transform(), ha="center")
    ax.text(3.1, -0.17, "CFAR", transform=ax.get_xaxis_transform(), ha="center")
    ax.set_ylabel(YLAB)
    ax.set_ylim(0, 1.12)
    letter(ax, "d", f"CFA by sex (F {(d.sex == 'Female').sum():,}, M {(d.sex == 'Male').sum():,})")
    # e: BMI by sex
    ax = fig.add_subplot(bot[2])
    bb = d.dropna(subset=["bmi"])
    for sex, c in SEX_C.items():
        g = bb[bb.sex == sex]
        for col, ls in [("r2_pre", "-"), ("r2_clean", "--")]:
            r, pv = stats.pearsonr(g.bmi, g[col])
            k_, b_ = np.polyfit(g.bmi, g[col], 1)
            xs = np.linspace(g.bmi.quantile(0.01), g.bmi.quantile(0.99), 50)
            ax.plot(xs, k_ * xs + b_, color=c, ls=ls, lw=1.6, label=f"{sex}, {'raw' if ls == '-' else 'CFAR'}: r = {r:.2f}")
            S4[f"bmi_r_{sex}_{col}"], S4[f"bmi_slope10_{sex}_{col}"] = float(r), float(k_ * 10)
            S4[f"bmi_n_{sex}"] = int(len(g))
    for col in ["r2_pre", "r2_clean"]:
        r, pv = stats.pearsonr(bb.bmi, bb[col])
        S4[f"bmi_r_all_{col}"], S4[f"bmi_p_all_{col}"] = float(r), float(pv)
    ax.set(xlabel="BMI (kg/m²)", ylabel=YLAB)
    ax.legend(frameon=False, fontsize=6.5)
    letter(ax, "e", f"CFA vs. BMI by sex (n = {len(bb):,})")
    STATS_V3["fig4"] = S4
    save(fig, "fig4_v3_demographics")


# =============================================================================
# Figure 5: diagnoses, before and after CFAR
# =============================================================================
def fig5():
    d = pd.read_parquet(os.path.join(HERE, "cleaning_demo.parquet"))
    demo = pd.read_parquet(os.path.join(HERE, "demographics_combined.parquet"))[["patient_id", "diagnosis_categories", "n_diagnoses"]]
    d = d.merge(demo, on="patient_id", how="left")
    cats = pd.read_parquet(os.path.join(HERE, "fig2_dx_forest.parquet")).category.tolist()
    has = {c: d.diagnosis_categories.apply(lambda v: v is not None and c in list(v)).values for c in cats}
    none = (d.n_diagnoses.fillna(0) == 0).values
    rows = []
    for c in cats:
        for col in ["r2_pre", "r2_clean"]:
            m_, h_ = ci_mean(d.loc[has[c], col])
            rows.append({"category": c, "col": col, "mean": m_, "ci": h_, "n": int(has[c].sum())})
    F = pd.DataFrame(rows)
    order = F[F.col == "r2_pre"].sort_values("mean").category.tolist()
    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.75, 1.3], hspace=0.38)
    top = gs[0].subgridspec(1, 2, wspace=0.3)
    bot = gs[1].subgridspec(1, 2, wspace=0.55, width_ratios=[1, 1.1])
    # a-b: example from the highest- and lowest-CFA diagnosis, closest to its group mean
    out = []
    for k, c in enumerate([order[-1], order[0]]):
        g = d[has[c]]
        pid = g.iloc[(g.r2_pre - g.r2_pre.mean()).abs().argsort().iloc[0]].patient_id
        rec = cd.load((pid, *window_of(pid)[:2]))
        _, x, ecg, peaks, sf = rec
        r2, ev, ecg_ev = cd.cfa_r2(x, ecg, peaks, sf)
        r2c, evc, _ = cd.cfa_r2(cd.project_out(x, U_POP), ecg, peaks, sf)
        si = int(np.argmax(r2))
        row = d[d.patient_id == pid].iloc[0]
        ax = fig.add_subplot(top[k])
        hep_panel(ax, cd.epoch_times(sf), [(ev[si], RAW_C, f"Raw (R² {r2[si]:.2f})", "-"),
                                           (evc[si], CFAR_C, f"CFAR (R² {r2c[si]:.2f})", "-")], ecg_ev,
                  f"{cd.SITES[si]} HEP (µV)")
        ax.legend(frameon=False, fontsize=7, loc="lower right")
        letter(ax, "ab"[k], f"{c} (group mean R² {g.r2_pre.mean():.2f}): {row.sex.lower()}, {row.age:.0f} y; patient R² {r2.mean():.2f} → {r2c.mean():.2f}", x=-0.1)
        out.append({"category": c, "patient_id": pid, "mean_r2": float(r2.mean()), "mean_r2_cfar": float(r2c.mean())})
    STATS_V3["fig5_examples"] = out
    # c: forest raw vs. CFAR
    ax = fig.add_subplot(bot[0])
    y = np.arange(len(order))
    for col, c, dy, lab in [("r2_pre", RAW_C, 0.15, "Raw"), ("r2_clean", CFAR_C, -0.15, "CFAR")]:
        f = F[F.col == col].set_index("category").loc[order]
        ax.errorbar(f["mean"], y + dy, xerr=f["ci"], fmt="o", ms=4, color=c, capsize=2, lw=1, label=lab)
        ax.axvline(d.loc[none, col].mean(), color=c, ls=":", lw=0.8)
    n_ = F[F.col == "r2_pre"].set_index("category").loc[order, "n"]
    ax.set_yticks(y, [f"{c} ({n:,})" for c, n in zip(order, n_)], fontsize=7)
    ax.set_xlabel(f"{YLAB} (mean, 95% CI); dotted: no diagnosis")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    letter(ax, "c", "CFA by diagnosis", x=-0.6)
    # d: pairwise differences, raw (upper triangle) vs. CFAR (lower)
    ax = fig.add_subplot(bot[1])
    lab = order[::-1]  # display order: top row = highest CFA
    k = len(lab)
    Q = np.full((k, k), np.nan)
    S5 = {}
    # upper triangle (row i < col j) = raw, lower triangle = CFAR; built in display order, no flipping
    for col, tri in [("r2_pre", np.triu_indices(k, 1)), ("r2_clean", np.tril_indices(k, -1))]:
        ps = [stats.mannwhitneyu(d.loc[has[lab[i]], col], d.loc[has[lab[j]], col]).pvalue for i, j in zip(*tri)]
        q = multipletests(ps, method="fdr_bh")[1]
        Q[tri] = q
        S5[f"n_sig_pairs_{col}"] = int((q < 0.05).sum())
        S5["n_pairs"] = len(ps)
        S5[f"obesity_vs_none_p_{col}"] = float(stats.mannwhitneyu(d.loc[has["Obesity"], col], d.loc[none, col]).pvalue) if "Obesity" in has else None
    im = ax.imshow(Q, cmap="Reds_r", vmin=0, vmax=0.5)
    ax.set_xticks(range(k), lab, rotation=90, fontsize=6.5)
    ax.set_yticks(range(k), lab, fontsize=6.5)
    ax.plot([-0.5, k - 0.5], [-0.5, k - 0.5], color="black", lw=1.2)
    box = dict(boxstyle="round,pad=0.35", lw=1.2)
    ax.text(0.66, 0.80, f"RAW\n{S5['n_sig_pairs_r2_pre']}/{S5['n_pairs']} pairs differ", transform=ax.transAxes, ha="center",
            va="center", fontsize=9, fontweight="bold", color=RAW_C, bbox={**box, "fc": "white", "ec": RAW_C})
    ax.text(0.30, 0.28, f"CFAR\n{S5['n_sig_pairs_r2_clean']}/{S5['n_pairs']} pairs differ", transform=ax.transAxes, ha="center",
            va="center", fontsize=9, fontweight="bold", color=CFAR_C, bbox={**box, "fc": "white", "ec": CFAR_C})
    cb = fig.colorbar(im, ax=ax, shrink=0.6, extend="max")
    cb.set_label("FDR-corrected p", fontsize=7)
    letter(ax, "d", "Pairwise differences: raw (upper) vs. CFAR (lower)", x=-0.05)
    S5.update({f"{r.category}|{r.col}": float(r["mean"]) for _, r in F.iterrows()})
    S5["none_mean_raw"], S5["none_mean_cfar"] = float(d.loc[none, "r2_pre"].mean()), float(d.loc[none, "r2_clean"].mean())
    S5["n_none"] = int(none.sum())
    S5["order"] = order
    STATS_V3["fig5"] = S5
    save(fig, "fig5_v3_diagnosis")


# =============================================================================
# Figure 6: segment duration
# =============================================================================
def fig6():
    dose = pd.DataFrame(STATS["dose_response"]["lengths"])
    dstrat = pd.DataFrame(STATS["dose_response"]["stratified_by_length"])
    mins = [5, 10, 20, 60]
    files = {5: "cfa_variance_explained_5min.parquet", 10: "cfa_combined_10min.parquet",
             20: "cfa_variance_explained_20min.parquet", 60: "cfa_variance_explained_60min.parquet"}
    pm = pd.DataFrame({m: pd.read_parquet(os.path.join(HERE, f), columns=["patient_id", "cfa_r2_excl_qrs"])
                       .groupby("patient_id").cfa_r2_excl_qrs.mean() for m, f in files.items()}).dropna()
    pm = pm[pm.index.str.startswith("I0")]
    target = np.array([dict(zip(dose.window_minutes, dose["mean"]))[m] for m in mins])
    cand = ((pm - target) ** 2).sum(1).nsmallest(16).index.tolist()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(16) as ex:  # nested windows (first m min of one 60-min window): rising and closest to cohort
        nested = dict(zip(cand, ex.map(nested_r2, cand)))
    ok = {k: v for k, v in nested.items() if v is not None and np.all(np.diff(v) > 0)}
    pid = min(ok or nested, key=lambda k: ((np.array((ok or nested)[k]) - target) ** 2).sum())
    sites, eeg, ecg, peaks, sf = load_window(*window_of(pid, "cfa_variance_explained_60min.parquet"))
    r2_60, _, _ = cd.cfa_r2(eeg, ecg, peaks, sf)
    n5 = int(5 * 60 * sf)
    r2_5, _, _ = cd.cfa_r2(eeg[:, :n5], ecg[:n5], peaks[peaks < n5 - int(0.5 * sf)], sf)
    si = int(np.argmax(r2_60 - r2_5))  # the electrode where averaging more beats reveals the most CFA
    fig = plt.figure(figsize=(11, 9.6))
    gs = fig.add_gridspec(3, 1, hspace=0.45, height_ratios=[0.75, 1, 1])
    maps = gs[0].subgridspec(1, 4, wspace=0.3)
    top = gs[1].subgridspec(1, 4, wspace=0.3)
    bot = gs[2].subgridspec(1, 3, wspace=0.35)
    ex = []
    axes = []
    for k, m in enumerate(mins):
        nsamp = int(m * 60 * sf)
        pk = peaks[peaks < nsamp - int(0.5 * sf)]
        r2, ev, ecg_ev = cd.cfa_r2(eeg[:, :nsamp], ecg[:nsamp], pk, sf)
        tax = fig.add_subplot(maps[k])  # scalp map of R² from the same first m minutes
        im = topo(tax, dict(zip(sites, r2)), (0, 1))
        letter(tax, "a" if k == 0 else "", f"First {m} min: mean R² {r2.mean():.2f}", x=-0.25)
        ax = fig.add_subplot(top[k], sharey=axes[0] if axes else None)
        axes.append(ax)
        hep_panel(ax, cd.epoch_times(sf), [(ev[si], P[0], "HEP", "-")], ecg_ev,
                  f"{sites[si]} HEP (µV)" if k == 0 else None, f"{sites[si]} R² {r2[si]:.2f}\n{len(pk):,} beats")
        if k:
            ax.tick_params(labelleft=False)
        ex.append({"minutes": m, "site_r2": float(r2[si]), "mean_r2": float(r2.mean()), "n_beats": int(len(pk))})
    pos = tax.get_position()  # own axis beside the last scalp map, so the maps keep equal size
    cb = fig.colorbar(im, cax=fig.add_axes([pos.x1 + 0.01, pos.y0 + 0.1 * pos.height, 0.01, 0.8 * pos.height]))
    cb.set_label("CFA R²", fontsize=7)
    STATS_V3["fig6_example"] = {"patient_id": pid, "site": sites[si], "steps": ex}

    def sat(t, A, k, c):
        return c + (A - c) * t / (t + k)
    from scipy.optimize import curve_fit
    ax = fig.add_subplot(bot[0])
    ax.errorbar(dose.window_minutes, dose["mean"], yerr=1.96 * dose["sem"], marker="o", ms=4, capsize=3, color=P[0], lw=1.6)
    p, _ = curve_fit(sat, dose.window_minutes, dose["mean"], p0=(0.45, 10, 0.2), sigma=dose["sem"], absolute_sigma=True)
    xs = np.linspace(1, 92, 100)
    ax.plot(xs, sat(xs, *p), color="0.6", ls="--", lw=1)
    ax.set_ylabel("Mean CFA R² (± 95% CI)")
    letter(ax, "b", f"Cohort (n = {STATS['dose_response']['n_common_patients']:,})")
    ax = fig.add_subplot(bot[1])
    for col, sex in [("male_mean", "Male"), ("female_mean", "Female")]:
        ax.plot(dstrat.window_minutes, dstrat[col], marker="o", ms=4, color=SEX_C[sex], lw=1.6,
                label=f"{sex} (n = {int(dstrat[f'n_{sex.lower()}'].iloc[0]):,})")
    ax.set_ylabel("Mean CFA R²")
    ax.legend(frameon=False, fontsize=7)
    letter(ax, "c", "By sex")
    DX = dx_by_length()
    ax = fig.add_subplot(bot[2])
    for key, lab, c in [("zero", "Undiagnosed", P[2]), ("one", "1 diagnosis", P[1]), ("two", "2+ diagnoses", P[3])]:
        ax.plot(dstrat.window_minutes, DX[key][0], marker="s", ms=4, color=c, lw=1.6, label=f"{lab} (n = {DX[key][1]:,})")
    ax.set_ylabel("Mean CFA R²")
    ax.legend(frameon=False, fontsize=7)
    letter(ax, "d", "By number of diagnoses")
    for a in fig.axes[-3:]:
        a.set_xlabel("Segment duration (min)")
        a.set_xticks([5, 10, 20, 30, 45, 60, 75, 90])
        a.set_xlim(2, 93)
    save(fig, "fig6_v3_duration")


def nested_r2(pid, mins=(5, 10, 20, 60)):
    try:
        sites, eeg, ecg, peaks, sf = load_window(*window_of(pid, "cfa_variance_explained_60min.parquet"))
    except Exception:  # noqa: BLE001
        return None
    out = []
    for m in mins:
        n = int(m * 60 * sf)
        out.append(float(cd.cfa_r2(eeg[:, :n], ecg[:n], peaks[peaks < n - int(0.5 * sf)], sf)[0].mean()))
    return out


def dx_by_length():
    """As make_fig3.dx_by_length: mean CFA R² of patients with 0, 1, >=2 diagnoses per length (common cohort)."""
    import glob, re
    files = {10: os.path.join(HERE, "cfa_combined_10min.parquet")}
    for f in glob.glob(os.path.join(HERE, "cfa_variance_explained_*min.parquet")):
        files[int(re.search(r"_(\d+)min\.parquet$", f).group(1))] = f
    demo = pd.read_parquet(os.path.join(HERE, "demographics_combined.parquet"))[["patient_id", "age", "n_diagnoses"]]
    dfs = {m: pd.read_parquet(f, columns=["patient_id", "cfa_r2_excl_qrs"]) for m, f in files.items()}
    common = set.intersection(*[set(v.patient_id) for v in dfs.values()])
    out = {"zero": ([], 0), "one": ([], 0), "two": ([], 0)}
    for m in sorted(dfs):
        pt = dfs[m][dfs[m].patient_id.isin(common)].groupby("patient_id", as_index=False).mean().merge(demo, on="patient_id").dropna(subset=["age"])
        nd = pt.n_diagnoses.fillna(0)
        for key, mask in [("zero", nd == 0), ("one", nd == 1), ("two", nd >= 2)]:
            g = pt.loc[mask, "cfa_r2_excl_qrs"]
            out[key][0].append(g.mean())
            out[key] = (out[key][0], len(g))
    return out


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


STATS_V3 = dict(STATS.get("v3", {}))
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", nargs="*", type=int, default=[1, 2, 4, 5, 6])
    a = ap.parse_args()
    c = pd.read_parquet(os.path.join(HERE, "cfa_combined.parquet"), columns=["patient_id", "cfa_r2_excl_qrs"])
    STATS_V3["cohort_median_r2"] = float(c.groupby("patient_id").cfa_r2_excl_qrs.mean().median())
    for f in a.fig:
        {1: fig1, 2: fig2, 4: fig4, 5: fig5, 6: fig6}[f]()
    S = json.load(open(os.path.join(HERE, "paper_stats.json")))
    S["v3"] = STATS_V3
    json.dump(S, open(os.path.join(HERE, "paper_stats.json"), "w"), indent=2, default=float)
