#!/usr/bin/env python3
"""What clean EEG can show that raw (or ICA) EEG cannot: four analyses, one PNG each, plus clinical_uses.json.

  1 uses_dx_classifier.png  HSP: predict each diagnosis from HEP features (raw, CFAR, ps-CFAR) and from the ECG
                            average alone; cross-validated AUC. Heart-driven signal should vanish after cleaning.
  2 uses_stage.png          HSP: HEP amplitude by expert-scored sleep stage, raw vs. CFAR.
  3 uses_bmi.png            HSP: HEP amplitude vs. BMI, heart rate and ECG amplitude, per cleaning method.
  4 uses_cobrad.png         Beilinson / COBRAD: HEP amplitude vs. MoCA, MMSE, CDR per method.
Each part runs only if its input exists (hep_features_all.parquet, stage_hep_amp.parquet, hep_cleaning_compare.parquet,
beilinson_cfar_vs_ica.parquet). Run: venv/bin/python "paper utils/Paper CFA/make_clinical_uses.py"
"""
import json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import make_cleaning_demo as cd

HERE, P = cd.HERE, cd.PALETTE
OUT = os.path.join(cd.FIG_DIR, "uses")
os.makedirs(OUT, exist_ok=True)
C = {"raw": P[1], "ica": P[5], "cfar": "0.15", "own": P[2], "ecg": "#b2182b"}
LAB = {"raw": "HEP", "ica": "ICA", "cfar": "CFAR", "own": "ps-CFAR", "ecg": "ECG alone"}  # all values are HEPs; HEP = uncleaned
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})
R = json.load(open(os.path.join(HERE, "clinical_uses.json"))) if os.path.exists(os.path.join(HERE, "clinical_uses.json")) else {}
pth = lambda f: os.path.join(HERE, f)


def bmi_table():
    b = pd.read_parquet(pth("bmi_combined.parquet")).drop_duplicates("bdsp_patient_id")
    return b


def with_demo(d):
    d = d.copy()
    d["bdsp_patient_id"] = pd.to_numeric(d.patient_id.str.extract(r"I\d{4}(\d{9})", expand=False))
    demo = pd.read_parquet(pth("demographics_combined.parquet"))[["patient_id", "sex", "age", "diagnosis_categories", "n_diagnoses"]]
    return d.merge(demo.drop_duplicates("patient_id"), on="patient_id", how="left").merge(bmi_table(), on="bdsp_patient_id", how="left")


# ---------------------------------------------------------------- 1: diagnosis classifier
if os.path.exists(pth("hep_features_all.parquet")):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    d = with_demo(pd.read_parquet(pth("hep_features_all.parquet")))
    d = d[d.diagnosis_categories.notna()]
    cats = pd.read_parquet(pth("fig2_dx_forest.parquet")).category.tolist()
    X = {m: np.vstack(d[f"f_{m}"].values) for m in ["raw", "cfar", "own", "ecg"]}
    res = []
    for c in cats:
        y = d.diagnosis_categories.apply(lambda v: c in list(v)).values.astype(int)
        if y.sum() < 100 or (1 - y).sum() < 100:
            continue
        row = {"category": c, "n": int(y.sum())}
        for m, Xm in X.items():
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
            p = cross_val_predict(clf, Xm, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
            row[f"auc_{m}"] = float(roc_auc_score(y, p))
        res.append(row)
    T = pd.DataFrame(res).sort_values("auc_ecg", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(T) + 1.2))
    y = np.arange(len(T))
    for m, mk in [("ecg", "s"), ("raw", "o"), ("cfar", "o"), ("own", "D")]:
        ax.scatter(T[f"auc_{m}"], y, color=C[m], marker=mk, s=28, label=LAB[m], zorder=3)
    for k, (_, r) in enumerate(T.iterrows()):
        ax.plot([r.auc_cfar, r.auc_raw], [k, k], color="0.75", lw=1.2, zorder=1)
    ax.axvline(0.5, color="0.5", ls="--", lw=0.8)
    ax.set_yticks(y, [f"{c} ({n:,})" for c, n in zip(T.category, T.n)])
    ax.invert_yaxis()
    ax.set_xlabel("Cross-validated AUC (5-fold), predicting the diagnosis from the HEP")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "uses_dx_classifier.png"), bbox_inches="tight")
    plt.close(fig)
    R["dx_classifier"] = {"n": int(len(d)), "rows": T.round(3).to_dict("records")}

# ---------------------------------------------------------------- 2: sleep-stage HEP
if os.path.exists(pth("stage_hep_amp.parquet")):
    s = pd.read_parquet(pth("stage_hep_amp.parquet"))
    sf = s.sfreq.mode()[0]
    s = s[s.sfreq == sf]
    tt = np.arange(int(round(-0.2 * sf)), int(round(0.6 * sf))) / sf
    win = (tt >= 0.2) & (tt < 0.6)
    T_ = len(tt)
    for m in ["raw", "cfar"]:
        s[f"amp_{m}"] = [np.sqrt((np.asarray(h).reshape(6, T_)[:, win] ** 2).mean()) for h in s[f"hep_{m}"]]
    STG = ["W", "N1", "N2", "N3", "R"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), gridspec_kw={"width_ratios": [1.2, 1.2, 1.2, 1]})
    mean_hep = {}
    for ax, m in zip(axes[:2], ["raw", "cfar"]):
        for st, col in zip(STG, plt.cm.viridis(np.linspace(0, 0.9, 5))):
            g = s[s.stage == st]
            mean_hep[(m, st)] = np.vstack([np.asarray(h).reshape(6, T_).mean(0) for h in g[f"hep_{m}"]]).mean(0)
            ax.plot(tt * 1e3, mean_hep[(m, st)], color=col, lw=1.3, label=f"{st} (n = {len(g)})")
        ax.axvspan(-50, 50, color="0.92", lw=0, zorder=0)
        ax.set(xlabel="Time from R-peak (ms)", ylabel="Channel mean (µV)", title=LAB[m])
    ax = axes[2]  # the delta: what CFAR removed, per stage
    for st, col in zip(STG, plt.cm.viridis(np.linspace(0, 0.9, 5))):
        ax.plot(tt * 1e3, mean_hep[("raw", st)] - mean_hep[("cfar", st)], color=col, lw=1.3)
    ax.axvspan(-50, 50, color="0.92", lw=0, zorder=0)
    ax.set(xlabel="Time from R-peak (ms)", ylabel="Channel mean (µV)", title="Removed (HEP − CFAR)")
    axes[2].sharey(axes[0])
    axes[0].legend(frameon=False, fontsize=6.5)
    ax = axes[3]
    wide = s.pivot_table(index="patient_id", columns="stage", values=["amp_raw", "amp_cfar"])
    out = {}
    for m, off in [("raw", -0.15), ("cfar", 0.15)]:
        rel = wide[f"amp_{m}"][STG].div(wide[f"amp_{m}"]["W"], axis=0).dropna()
        med = rel.median()
        ax.plot(np.arange(5) + off, med.values, "o-", color=C[m], label=LAB[m])
        fr = stats.friedmanchisquare(*[rel[c] for c in STG])
        out[m] = {"median_rel_to_wake": med.round(3).to_dict(), "friedman_p": float(fr.pvalue), "n": int(len(rel))}
    ax.axhline(1, color="0.5", ls="--", lw=0.8)
    ax.set_xticks(range(5), STG)
    ax.set(ylabel="HEP amplitude, 200–600 ms\n(relative to wake; median)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "uses_stage.png"), bbox_inches="tight")
    plt.close(fig)
    R["stage"] = out

# ---------------------------------------------------------------- 3: HEP amplitude vs body and heart
if os.path.exists(pth("hep_cleaning_compare.parquet")):
    d = with_demo(pd.read_parquet(pth("hep_cleaning_compare.parquet")))
    d = d[d.sfreq == d.sfreq.mode()[0]].reset_index(drop=True)
    tt = np.asarray(d.tt.iloc[0])
    win = (tt >= 0.2) & (tt < 0.6)
    T_ = len(tt)
    for m in ["raw", "ica", "cfar", "own"]:
        d[f"amp_{m}"] = [np.sqrt((np.asarray(h).reshape(6, T_)[:, win] ** 2).mean()) for h in d[f"hep_{m}"]]
    d["ecg_amp"] = [np.sqrt((np.asarray(e)[win] ** 2).mean()) for e in d.ecg]
    covs = [("bmi", "BMI"), ("hr_bpm", "Heart rate"), ("ecg_amp", "ECG amplitude (200–600 ms)")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    out = {}
    for ax, (cv, lab) in zip(axes, covs):
        for k, m in enumerate(["raw", "ica", "cfar", "own"]):
            g = d[[cv, f"amp_{m}"]].dropna()
            r, p = stats.spearmanr(g[cv], g[f"amp_{m}"])
            lo, hi = np.percentile([stats.spearmanr(*g.sample(len(g), replace=True, random_state=b_).T.values)[0] for b_ in range(300)], [2.5, 97.5])
            ax.bar(k, r, color=C[m], yerr=[[r - lo], [hi - r]], capsize=2)
            ax.text(k, (hi + 0.02) if r >= 0 else (lo - 0.02), f"{r:.2f}", ha="center", va="bottom" if r >= 0 else "top", fontsize=7)
            out[f"{cv}|{m}"] = {"r": float(r), "p": float(p), "n": int(len(g))}
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xticks(range(4), ["Raw", "ICA", "CFAR", "ps-CFAR"], rotation=20)
        ax.set_title(f"HEP amplitude vs. {lab}", fontsize=8.5)
        ax.set_ylabel("Spearman r (95% CI)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "uses_bmi.png"), bbox_inches="tight")
    plt.close(fig)
    R["body"] = out

# ---------------------------------------------------------------- 4: COBRAD cognition
if os.path.exists(pth("beilinson_cfar_vs_ica.parquet")):
    b = pd.read_parquet(pth("beilinson_cfar_vs_ica.parquet"))
    b = b[b.error.isna()].copy()
    clin = pd.read_excel(os.path.join(os.path.dirname(os.path.dirname(HERE)), "EDF_Format", "EDF", "COBRAD_clinical_24022025.xlsx"))
    clin["key"] = clin.record_id.str.upper()
    b["key"] = b.patient_id.str.lstrip("0").str.upper()
    b = b.merge(clin[["key", "mmse", "moca", "cdr_sum_of_boxes", "age"]], on="key", how="inner")
    tt = cd.epoch_times(256.0)
    for m in ["raw", "ica", "own"]:
        amps = []
        for _, r in b.iterrows():
            H = np.asarray(r[f"ev_{m}"]).reshape(int(r.n_channels), len(r.ecg_ev))
            t_ = cd.epoch_times(r.sfreq)[:H.shape[1]]
            amps.append(np.sqrt((H[:, (t_ >= 0.2) & (t_ < 0.4)] ** 2).mean()))
        b[f"amp_{m}"] = amps
    b["ecg_amp"] = [np.sqrt((np.asarray(e)[(cd.epoch_times(r)[:len(e)] >= 0.2) & (cd.epoch_times(r)[:len(e)] < 0.4)] ** 2).mean())
                    for e, r in zip(b.ecg_ev, b.sfreq)]
    scores = [("moca", "MoCA"), ("mmse", "MMSE"), ("cdr_sum_of_boxes", "CDR-SB")]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3), gridspec_kw={"width_ratios": [1, 1, 1, 1.2]})
    out = {"n": int(len(b))}
    for ax, (sc, lab) in zip(axes[:3], scores):
        for k, m in enumerate(["raw", "ica", "own"]):
            g = b[[sc, f"amp_{m}"]].dropna()
            r, p = stats.spearmanr(g[sc], g[f"amp_{m}"])
            ax.bar(k, r, color=C[m])
            ax.text(k, (max(r, 0) + 0.03) if r >= 0 else r - 0.08, f"{r:.2f}\np={p:.2f}", ha="center", fontsize=6.5)
            out[f"{sc}|{m}"] = {"r": float(r), "p": float(p), "n": int(len(g))}
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_ylim(-0.75, 0.75)
        ax.set_xticks(range(3), ["Raw", "ICA", "ps-CFAR"])
        ax.set_title(f"HEP 200–400 ms vs. {lab}", fontsize=8.5)
        ax.set_ylabel("Spearman r")
    ax = axes[3]
    for m in ["raw", "own"]:
        ax.scatter(b.moca, b[f"amp_{m}"], color=C[m], s=14, label=LAB[m])
    ax.set(xlabel="MoCA", ylabel="HEP amplitude, 200–400 ms (µV RMS)")
    ax.legend(frameon=False, fontsize=7)
    r_ecg = stats.spearmanr(b.moca, b.ecg_amp, nan_policy="omit")
    out["moca|ecg"] = {"r": float(r_ecg[0]), "p": float(r_ecg[1])}
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "uses_cobrad.png"), bbox_inches="tight")
    plt.close(fig)
    R["cobrad"] = out

json.dump(R, open(pth("clinical_uses.json"), "w"), indent=1, default=float)
print(json.dumps({k: (v if k != "dx_classifier" else v["rows"][:5]) for k, v in R.items()}, indent=1)[:4000])
