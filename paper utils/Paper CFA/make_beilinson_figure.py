#!/usr/bin/env python3
"""Figure 4 (v3): independent clinical cohort (Rabin Medical Center, Beilinson). What patient-specific CFAR
(ps-CFAR) reveals that ICA does not.

  A  example: HEP at the channel where ICA leaves the most ECG shape; raw vs. ICA vs. ps-CFAR (display 30 Hz)
  B  where the cardiac artifact remains: ECG-explained HEP power over the heartbeat (median over patients)
  C  channels free of CFA (R² < 0.1) per patient: raw -> ICA -> ps-CFAR
  D  scalp maps of mean CFA R², referential recordings (19 electrodes): raw, ICA, ps-CFAR
Brain-signal cost (synthetic HEP kept) and net gain go to figS_beilinson_cost.png.
Input: beilinson_cfar_vs_ica.parquet. Writes figures/fig3_v3_beilinson.png and beilinson_summary.json.
"""
import json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, filtfilt

import make_cleaning_demo as cd
from channel_utils import canonicalize
from ica_ecg_component_variance import QRS_EXCLUDE_SEC
from make_v3_figures import letter, topo

HERE, P = cd.HERE, cd.PALETTE
M = ["raw", "ica", "own"]
C = {"raw": P[1], "ica": P[5], "own": P[2]}
LAB = {"raw": "Raw", "ica": "ICA", "own": "ps-CFAR"}  # HEP-derived values throughout
THR = 0.1  # channel counted as CFA-free below this R²
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})

d = pd.read_parquet(os.path.join(HERE, "beilinson_cfar_vs_ica.parquet"))
n_all = len(d)
excluded = d[d["error"].notna()]
d = d[d["error"].isna()].reset_index(drop=True)
for m in ["ica", "own"]:
    d[f"left_{m}"] = d[f"pow_{m}"] / d.pow_raw * 100
S = {"n_total": int(n_all), "n": int(len(d)), "n_excluded": int(len(excluded)),
     "n_bipolar": int((d.montage == "bipolar").sum()), "n_referential": int((d.montage == "referential").sum())}
for k in ["r2_raw", "r2_ica", "r2_own", "left_ica", "left_own", "hep_ica", "hep_own", "gain_ica", "gain_own"]:
    S[f"{k}_median"] = float(d[k].median())
S["p_left"] = float(stats.wilcoxon(d.left_own, d.left_ica).pvalue)
S["p_hep"] = float(stats.wilcoxon(d.hep_own, d.hep_ica).pvalue)
S["p_gain"] = float(stats.wilcoxon(d.gain_own, d.gain_ica).pvalue)
S["pct_own_cleaner"] = float((d.left_own < d.left_ica).mean() * 100)
S["pct_own_higher_gain"] = float((d.gain_own > d.gain_ica).mean() * 100)


def hep(row, m):
    return np.asarray(row[f"ev_{m}"]).reshape(int(row.n_channels), len(row.ecg_ev))


def ecg_part(h, e, keep):
    """Per-channel LS fit of the HEP on the ECG average outside QRS -> ECG-explained part (ch, t)."""
    e0 = e - e[keep].mean()
    h0 = h - h[:, keep].mean(1, keepdims=True)
    return np.outer(h0[:, keep] @ e0[keep] / (e0[keep] @ e0[keep]), e0)


fig = plt.figure(figsize=(11, 7.4))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.45)
top = gs[0].subgridspec(1, 2, width_ratios=[1.25, 1], wspace=0.3)
bot = gs[1].subgridspec(1, 4, width_ratios=[1, 0.8, 0.8, 0.8], wspace=0.3)

# A: example -- channel where ICA leaves most ECG shape and ps-CFAR removes it, readable (little HF noise)
best = None
for i, row in d.iterrows():
    r = {m: np.asarray(row[f"r2ch_{m}"]) for m in M}
    ok = r["raw"] > 0.5
    if not ok.any():
        continue
    ch = int(np.argmax(np.where(ok, r["ica"] - r["own"], -1)))
    h = hep(row, "own")[ch]
    b, a = butter(4, 30, fs=row.sfreq)
    rough = np.std(h - filtfilt(b, a, h)) / np.std(h)
    score = (r["ica"][ch] - r["own"][ch]) * (1 - rough)
    if best is None or score > best[0]:
        best = (score, i, ch)
_, i, ch = best
ex = d.loc[i]
sf = ex.sfreq
tt = cd.epoch_times(sf)[:len(ex.ecg_ev)] * 1e3
b, a = butter(4, 30, fs=sf)
ax = fig.add_subplot(top[0])
tw = ax.twinx()
tw.axvspan(-QRS_EXCLUDE_SEC * 1e3, QRS_EXCLUDE_SEC * 1e3, color="0.92", lw=0)
tw.plot(tt, ex.ecg_ev, color="0.6", lw=0.8, ls=":")
tw.set_yticks([])
tw.spines[["top", "right"]].set_visible(False)
for m, lw, ls in [("ica", 2.2, "-"), ("own", 1.8, "-"), ("raw", 1.1, "--")]:  # raw dashed on top: ICA can coincide with it
    ax.plot(tt, filtfilt(b, a, hep(ex, m)[ch]), color=C[m], lw=lw, ls=ls, label=f"{LAB[m]}: R² {ex[f'r2ch_{m}'][ch]:.2f}")
ax.set_zorder(tw.get_zorder() + 1)
ax.patch.set_visible(False)
ax.set(xlim=(tt[0], tt[-1]), xlabel="Time from R-peak (ms)", ylabel=f"{ex.channels[ch]} HEP (µV)")
ax.legend(frameon=False, fontsize=7, loc="upper left")
letter(ax, "a", f"One patient ({ex.montage}): ICA keeps the ECG shape, ps-CFAR removes it", x=-0.1)
S["example"] = {"patient_id": ex.patient_id, "montage": ex.montage, "channel": ex.channels[ch],
                **{f"r2_{m}": float(ex[f"r2ch_{m}"][ch]) for m in M}}

# B: ECG-explained HEP power over time, median across patients (each patient: mean over channels)
ax = fig.add_subplot(top[1])
T = min(len(v) for v in d.ecg_ev)
keep = np.abs(cd.epoch_times(d.sfreq.iloc[0])[:T]) > QRS_EXCLUDE_SEC
curves = {m: [] for m in M}
for _, row in d.iterrows():
    e = np.asarray(row.ecg_ev)[:T]
    for m in M:
        curves[m].append((ecg_part(hep(row, m)[:, :T], e, keep) ** 2).mean(0))
t_ms = cd.epoch_times(d.sfreq.iloc[0])[:T] * 1e3
for m in M:
    ax.plot(t_ms, np.median(curves[m], 0), color=C[m], lw=1.6, label=LAB[m])
ax.axvspan(-QRS_EXCLUDE_SEC * 1e3, QRS_EXCLUDE_SEC * 1e3, color="0.92", lw=0, zorder=0)
ax.set_yscale("log")
ax.set(xlim=(t_ms[0], t_ms[-1]), xlabel="Time from R-peak (ms)", ylabel="ECG-explained HEP power\n(µV², median over patients)")
ax.legend(frameon=False, fontsize=7)
win = (t_ms > 150) & (t_ms < 400)
S["twave_power"] = {m: float(np.median(np.array(curves[m])[:, win].mean(1))) for m in M}
letter(ax, "b", "Where the heart artifact remains", x=-0.2)

# C: CFA-free channels per patient
ax = fig.add_subplot(bot[0])
frac = {m: np.array([(np.asarray(r) < THR).mean() * 100 for r in d[f"r2ch_{m}"]]) for m in M}
for j in range(len(d)):
    ax.plot(range(3), [frac[m][j] for m in M], color="0.8", lw=0.6, zorder=0)
for k, m in enumerate(M):
    ax.scatter(np.full(len(d), k), frac[m], s=12, color=C[m], zorder=2)
    ax.plot([k - 0.25, k + 0.25], [np.median(frac[m])] * 2, color="black", lw=2, zorder=3)
    ax.text(k, 104, f"{np.median(frac[m]):.0f}%", ha="center", fontsize=7.5)
    S[f"cfa_free_pct_{m}"] = float(np.median(frac[m]))
S["p_free_own_vs_ica"] = float(stats.wilcoxon(frac["own"], frac["ica"]).pvalue)
ax.set_xticks(range(3), ["Raw", "ICA", "ps-CFAR"])
ax.set(ylim=(-3, 112), ylabel=f"Channels free of CFA (R² < {THR}), %")
letter(ax, "c", f"Usable channels (n = {len(d)})", x=-0.3)

# D: scalp maps, referential recordings
ref = d[d.montage == "referential"]
for k, m in enumerate(M):
    vals = {}
    for _, row in ref.iterrows():
        names = canonicalize(pd.Series(list(row.channels))).tolist()
        for nm, v in zip(names, row[f"r2ch_{m}"]):
            if isinstance(nm, str):
                vals.setdefault(nm, []).append(v)
    ax = fig.add_subplot(bot[k + 1])
    im = topo(ax, {s_: float(np.mean(v)) for s_, v in vals.items()}, (0, 0.6))
    ax.set_title(LAB[m], fontsize=8.5, color=C[m], fontweight="bold")
    S[f"ref_mean_r2_{m}"] = float(np.mean([np.mean(v) for v in vals.values()]))
    if k == 0:
        letter(ax, "d", "", x=-0.1)
        ax.text(1.65, 1.22, f"Mean CFA R², 19-electrode recordings (n = {len(ref)})", transform=ax.transAxes, ha="center", fontsize=8.5)
pos = ax.get_position()
cb = fig.colorbar(im, cax=fig.add_axes([pos.x1 + 0.01, pos.y0 + 0.15 * pos.height, 0.008, 0.7 * pos.height]))
cb.set_label("CFA R²", fontsize=7)
fig.savefig(os.path.join(cd.FIG_DIR, "fig3_v3_beilinson.png"), bbox_inches="tight")
fig.savefig(os.path.join(cd.FIG_DIR, "fig3_v3_beilinson.pdf"), bbox_inches="tight")
plt.close(fig)

# Supplementary: the cost (brain signal kept) and net gain, paired per patient
fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.2))
for ax, (key, ylab, log, scale) in zip(axes, [("left", "ECG-explained HEP variance left (%)", False, 1),
                                              ("hep", "Synthetic neural HEP kept (%)", False, 100),
                                              ("gain", "Net gain vs. raw", True, 1)]):
    a_, b_ = d[f"{key}_ica"] * scale, d[f"{key}_own"] * scale
    for x0, x1 in zip(a_, b_):
        ax.plot([0, 1], [x0, x1], color="0.8", lw=0.5)
    for j, (v, m) in enumerate([(a_, "ica"), (b_, "own")]):
        ax.scatter(np.full(len(v), j), v, s=10, color=C[m])
        ax.plot([j - 0.2, j + 0.2], [v.median()] * 2, color="black", lw=2)
    if log:
        ax.set_yscale("log")
        ax.axhline(1, color="0.5", ls="--", lw=0.8)
    ax.set_xticks([0, 1], ["ICA", "ps-CFAR"])
    ax.set_ylabel(ylab)
fig.tight_layout()
fig.savefig(os.path.join(cd.FIG_DIR, "figS_beilinson_cost.png"), bbox_inches="tight")
json.dump(S, open(os.path.join(HERE, "beilinson_summary.json"), "w"), indent=1, default=float)
print({k: v for k, v in S.items() if k.startswith(("cfa_free", "ref_mean", "p_free"))})
print(S["example"], S["twave_power"])
