#!/usr/bin/env python3
"""More views of the sleep-stage HEP result (stage_hep_amp.parquet: HEP per expert stage, raw vs. CFAR).

  stage_diff.png      stage-minus-wake difference waves, raw vs. CFAR, with times where the paired difference is
                      significant (Wilcoxon per sample, FDR) marked
  stage_topo.png      scalp maps of HEP amplitude (200-400 ms, RMS) per stage, raw vs. CFAR
  stage_slopes.png    per patient: N3/wake and REM/wake amplitude ratio, raw -> CFAR
  stage_ecg_r2.png    how ECG-like the HEP is in each stage (R² with the stage's ECG average, outside QRS)
  stage_decode.png    decoding the stage from one patient's HEP (leave-patients-out), raw vs. CFAR: accuracy + confusion
Writes figures/uses/*.png and adds "stage_views" to clinical_uses.json.
"""
import json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import make_cleaning_demo as cd
from make_v3_figures import topo

HERE = cd.HERE
OUT = os.path.join(cd.FIG_DIR, "uses")
STG = ["W", "N1", "N2", "N3", "R"]
SC = dict(zip(STG, plt.cm.viridis(np.linspace(0, 0.9, 5))))
TITLE = {"raw": "HEP", "cfar": "CFAR"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})

s = pd.read_parquet(os.path.join(HERE, "stage_hep_amp.parquet"))
sf = s.sfreq.mode()[0]
s = s[s.sfreq == sf].reset_index(drop=True)
tt = np.arange(int(round(-0.2 * sf)), int(round(0.6 * sf))) / sf
T = len(tt)
H = {m: np.stack([np.asarray(h).reshape(6, T) for h in s[f"hep_{m}"]]) for m in ["raw", "cfar"]}
E = np.stack(s.ecg.values)
keep_qrs = np.abs(tt) > 0.05
w24 = (tt >= 0.2) & (tt < 0.4)
R = {}

# ---- 1. difference waves (stage - wake), channel mean, paired over patients
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
wake = s.stage == "W"
diff_stats = {}
for ax, m in zip(axes, ["raw", "cfar"]):
    cm = H[m].mean(1)
    for st in STG[1:]:
        pts = sorted(set(s.patient_id[wake]) & set(s.patient_id[s.stage == st]))
        a = np.stack([cm[(s.patient_id == p) & (s.stage == st)][0] for p in pts])
        b = np.stack([cm[(s.patient_id == p) & wake][0] for p in pts])
        dlt = a - b
        ax.plot(tt * 1e3, dlt.mean(0), color=SC[st], lw=1.4, label=f"{st} − W (n = {len(pts)})")
        pv = np.array([stats.wilcoxon(dlt[:, i]).pvalue for i in range(T)])
        sig = multipletests(pv, method="fdr_bh")[0] & (tt > 0.05)
        y0 = -0.21 - 0.012 * STG.index(st)
        ax.scatter(tt[sig] * 1e3, np.full(sig.sum(), y0), s=2, color=SC[st], marker="s")
        diff_stats[f"{m}|{st}"] = {"n": len(pts), "ms_significant": float(sig.sum() / sf * 1e3)}
    ax.axhline(0, color="0.5", lw=0.7)
    ax.axvspan(-50, 50, color="0.92", lw=0, zorder=0)
    ax.set(xlabel="Time from R-peak (ms)", title=TITLE[m])
axes[0].set_ylabel("Stage − wake HEP difference (µV)\n(bars: FDR-significant samples)")
axes[1].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "stage_diff.png"), bbox_inches="tight")
plt.close(fig)
R["diff"] = diff_stats

# ---- 2. scalp maps of HEP amplitude 200-400 ms per stage
fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
amp = {m: np.sqrt((H[m][:, :, w24] ** 2).mean(2)) for m in ["raw", "cfar"]}  # (rows, 6)
for r_, m in enumerate(["raw", "cfar"]):
    vmax = np.percentile([np.median(amp[m][s.stage == st], 0) for st in STG], 100)
    for c_, st in enumerate(STG):
        im = topo(axes[r_, c_], dict(zip(cd.SITES, np.median(amp[m][s.stage == st], 0))), (0, vmax), cmap="viridis")
        axes[r_, c_].set_title(st if r_ == 0 else "", fontsize=9)
    axes[r_, 0].text(-0.35, 0.5, TITLE[m], transform=axes[r_, 0].transAxes, rotation=90, va="center", ha="center", fontsize=8.5)
    cb = fig.colorbar(im, ax=axes[r_, :], shrink=0.8, pad=0.01)
    cb.set_label("HEP RMS 200–400 ms (µV)", fontsize=7)
fig.savefig(os.path.join(OUT, "stage_topo.png"), bbox_inches="tight")
plt.close(fig)

# ---- 3. per-patient slopegraphs
s["amp_raw"] = np.sqrt((H["raw"][:, :, w24] ** 2).mean((1, 2)))
s["amp_cfar"] = np.sqrt((H["cfar"][:, :, w24] ** 2).mean((1, 2)))
wide = s.pivot_table(index="patient_id", columns="stage", values=["amp_raw", "amp_cfar"])
fig, axes = plt.subplots(1, 2, figsize=(7, 3.4), sharey=True)
slopes = {}
for ax, st in zip(axes, ["N3", "R"]):
    rr = (wide["amp_raw"][st] / wide["amp_raw"]["W"]).dropna()
    rc = (wide["amp_cfar"][st] / wide["amp_cfar"]["W"]).reindex(rr.index)
    ok = rc.notna()
    rr, rc = rr[ok], rc[ok]
    up = rc > rr
    for a, b, u in zip(rr, rc, up):
        ax.plot([0, 1], [a, b], color="#1f8a6a" if u else "0.75", lw=0.6, alpha=0.8)
    ax.plot([0, 1], [rr.median(), rc.median()], color="black", lw=2.5)
    ax.axhline(1, color="0.5", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.set_xticks([0, 1], ["HEP", "CFAR"])
    p = stats.wilcoxon(rc, rr).pvalue
    ax.set_title(f"{st} / wake: larger after CFAR in {up.mean()*100:.0f}%\n(n = {len(rr)}, Wilcoxon p = {p:.1g})", fontsize=8.5)
    slopes[st] = {"n": int(len(rr)), "pct_up": float(up.mean() * 100), "p": float(p),
                  "median_raw": float(rr.median()), "median_cfar": float(rc.median())}
axes[0].set_ylabel("HEP amplitude 200–400 ms, stage ÷ wake (log)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "stage_slopes.png"), bbox_inches="tight")
plt.close(fig)
R["slopes"] = slopes

# ---- 4. how ECG-like the HEP is per stage
def r2(h, e):
    return np.mean([np.corrcoef(c[keep_qrs], e[keep_qrs])[0, 1] ** 2 for c in h])


for m in ["raw", "cfar"]:
    s[f"r2_{m}"] = [r2(H[m][i], E[i]) for i in range(len(s))]
fig, ax = plt.subplots(figsize=(5, 3.2))
r2s = {}
for k, (m, c) in enumerate([("raw", cd.PALETTE[1]), ("cfar", "0.15")]):
    med = [s.loc[s.stage == st, f"r2_{m}"].median() for st in STG]
    ax.plot(np.arange(5) + (k - 0.5) * 0.12, med, "o-", color=c, label=TITLE[m])
    r2s[m] = dict(zip(STG, map(float, med)))
ax.set_xticks(range(5), STG)
ax.set(ylabel="HEP–ECG R² (outside QRS; median)", ylim=(0, None))
ax.legend(frameon=False, fontsize=7)
ax.set_title("The raw HEP is half heart in every stage", fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "stage_ecg_r2.png"), bbox_inches="tight")
plt.close(fig)
R["ecg_r2"] = r2s

# ---- 5. decode the stage from one patient's HEP (grouped CV by patient)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
bins = [(a, a + 0.05) for a in np.arange(0.05, 0.6, 0.05)]


def feats(h):  # 6 channels x 11 bins of 50 ms, each HEP z-scored within patient-stage for scale invariance
    return np.concatenate([[h[:, (tt >= a) & (tt < b)].mean(1) for a, b in bins]]).T.ravel()


y = s.stage.map({k: i for i, k in enumerate(STG)}).values
dec = {}
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), gridspec_kw={"width_ratios": [0.8, 1, 1]})
for k, m in enumerate(["raw", "cfar"]):
    X = np.stack([feats(h) for h in H[m]])
    # remove each patient's mean HEP so the classifier must use within-patient stage differences
    X = X - pd.DataFrame(X).groupby(s.patient_id.values).transform("mean").values
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
    pred = cross_val_predict(clf, X, y, cv=GroupKFold(5), groups=s.patient_id.values)
    acc = balanced_accuracy_score(y, pred)
    dec[m] = float(acc)
    cmx = confusion_matrix(y, pred, normalize="true")
    ax = axes[k + 1]
    ax.imshow(cmx, cmap="Greens", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cmx[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if cmx[i, j] > 0.5 else "black")
    ax.set_xticks(range(5), STG)
    ax.set_yticks(range(5), STG)
    ax.set(xlabel="Predicted", ylabel="True", title=f"{TITLE[m]}: {acc*100:.0f}% (chance 20%)")
ax = axes[0]
ax.bar([0, 1], [dec["raw"] * 100, dec["cfar"] * 100], color=[cd.PALETTE[1], "0.15"])
ax.axhline(20, color="0.5", ls="--", lw=0.8)
ax.set_xticks([0, 1], ["HEP", "CFAR"])
ax.set(ylabel="Balanced accuracy (%)", title="Sleep stage from the HEP")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "stage_decode.png"), bbox_inches="tight")
plt.close(fig)
R["decode"] = dec

U = json.load(open(os.path.join(HERE, "clinical_uses.json")))
U["stage_views"] = R
json.dump(U, open(os.path.join(HERE, "clinical_uses.json"), "w"), indent=1, default=float)
print(json.dumps({k: R[k] for k in ["slopes", "ecg_r2", "decode"]}, indent=1))
print({k: v["ms_significant"] for k, v in R["diff"].items()})
