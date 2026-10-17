#!/usr/bin/env python3
"""Fig S9: test-retest reliability of CFA R² vs. window length -> optimal duration.

Input: window_reliability.parquet (window_reliability.py): per patient, every
non-overlapping L-min window of contiguous QC-passing 5-min blocks, same night.

  a  ICC(1) across repeat windows of the same patient (95% bootstrap CI)
  b  within- vs between-patient SD of R²
  c  bias: nested mean R² vs. length (patients with a window at every length)
  d  retention: patients with >=1 / >=2 windows of that length
  e  example patients: spread of repeat windows shrinks with length
  f  decision: shortest length whose ICC lower CI clears 0.90 (else 0.75)

Run: venv/bin/python "paper utils/Paper CFA/make_window_reliability_figure.py" [parquet]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00", "#000000"]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300, "savefig.dpi": 300})
N_LIST = sum(1 for line in open(os.path.join(HERE, "figure5_7230_patients.txt")) if line.strip())
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "window_reliability.parquet")
df = pd.read_parquet(src, columns=["patient_id", "window_minutes", "window_index", "cfa_r2_excl_qrs"]).dropna()
L = sorted(df.window_minutes.unique())


def icc1(g):
    """One-way random ICC(1) for unbalanced groups; g = (n_i, mean_i, ss_within_i) arrays."""
    n, m, ssw = g
    N, a = n.sum(), len(n)
    grand = (n * m).sum() / N
    msb = (n * (m - grand) ** 2).sum() / (a - 1)
    msw = ssw.sum() / (N - a)
    k0 = (N - (n ** 2).sum() / N) / (a - 1)
    return (msb - msw) / (msb + (k0 - 1) * msw), np.sqrt(msw), np.sqrt(max((msb - msw) / k0, 0))


rng = np.random.default_rng(0)
rows = []
for minutes in L:
    s = df[df.window_minutes == minutes].groupby("patient_id").cfa_r2_excl_qrs
    st = pd.DataFrame({"n": s.size(), "m": s.mean(), "ssw": s.var(ddof=0) * s.size()})
    rep = st[st.n >= 2]
    row = {"window_minutes": int(minutes), "n_ge1": len(st), "n_ge2": len(rep)}
    if len(rep) >= 10:
        g = (rep.n.values, rep.m.values, rep.ssw.values)
        row["icc"], row["sd_within"], row["sd_between"] = icc1(g)
        boots = [icc1(tuple(x[idx] for x in g))[0] for idx in rng.integers(0, len(rep), (200, len(rep)))]
        row["icc_lo"], row["icc_hi"] = np.percentile(boots, [2.5, 97.5])
    rows.append(row)
T = pd.DataFrame(rows).set_index("window_minutes")
T["ret_ge1"], T["ret_ge2"] = 100 * T.n_ge1 / N_LIST, 100 * T.n_ge2 / N_LIST
print(T.round(3).to_string())

valid = T.dropna(subset=["icc"])
best, thr = None, None
for t in (0.90, 0.75):
    ok = valid.index[valid.icc_lo >= t]
    if len(ok):
        best, thr = int(ok.min()), t
        break
rule = f"ICC lower CI ≥ {thr}"
if best is None:  # no threshold cleared: shortest length within 0.02 of the best ICC
    best, rule = int(valid.index[valid.icc >= valid.icc.max() - 0.02].min()), "ICC within 0.02 of max"
icc5 = valid.icc.iloc[0]
k = np.arange(1, 19)
sb = k * icc5 / (1 + (k - 1) * icc5)  # Spearman-Brown: mean of k separate L[0]-min windows
print(f"Optimal single window: {best} min ({rule})")
print("Mean of k spread 5-min windows, predicted ICC:", dict(zip((k * L[0]).tolist(), sb.round(3))))

fig, axes = plt.subplots(2, 3, figsize=(11, 7.4))
for ax, letter in zip(axes.flat, "abcdef"):
    ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontsize=10, fontweight="bold")

ax = axes[0, 0]
ax.fill_between(valid.index, valid.icc_lo, valid.icc_hi, color=PALETTE[0], alpha=0.2, lw=0)
ax.plot(valid.index, valid.icc, "o-", color=PALETTE[0], ms=4, lw=1.6)
for t, ls in ((0.9, "--"), (0.75, ":")):
    ax.axhline(t, color="grey", ls=ls, lw=1)
    ax.text(valid.index.min(), t, f"{t}", va="bottom", fontsize=7, color="grey")
ax.set_ylabel("ICC(1) across repeat windows")
ax.set_title("Test-retest reliability (n with ≥2 windows)", fontsize=9)

ax = axes[0, 1]
ax.plot(valid.index, valid.sd_within, "o-", color=PALETTE[1], ms=4, lw=1.6, label="Within patient (window-to-window)")
ax.plot(valid.index, valid.sd_between, "s-", color=PALETTE[2], ms=4, lw=1.6, label="Between patients")
ax.set_ylabel("SD of CFA R²")
ax.set_ylim(bottom=0)
ax.legend(frameon=False, fontsize=7.5)
ax.set_title("Noise vs. signal", fontsize=9)

# bias: same patients, nested lengths -- every patient with >=1 window at each length up to the longest with n >= 200
ax = axes[0, 2]
pm = df.groupby(["patient_id", "window_minutes"]).cfa_r2_excl_qrs.mean().unstack()
# longest prefix of lengths that still keeps >= 25% of patients complete
cols_c = [m for j, m in enumerate(L) if pm[L[:j + 1]].notna().all(1).mean() >= 0.25]
coh = pm.loc[pm[cols_c].notna().all(1), cols_c]
sem = coh.std() / np.sqrt(len(coh))
ax.errorbar(coh.columns, coh.mean(), yerr=1.96 * sem, marker="o", ms=4, capsize=3, color=PALETTE[3], lw=1.6)
ax.set_ylabel("Mean CFA R² (± 95% CI)")
ax.set_title(f"Length bias, same patients & night (n = {len(coh):,})", fontsize=9)

ax = axes[1, 0]
ax.plot(T.index, T.ret_ge1, "o-", color=PALETTE[6], ms=4, lw=1.6, label="≥1 window")
ax.plot(T.index, T.ret_ge2, "s--", color=PALETTE[6], ms=4, lw=1.4, alpha=0.7, label="≥2 windows")
ax.set_ylim(0, 105)
ax.set_ylabel(f"Patients (% of {N_LIST:,})")
ax.legend(frameon=False, fontsize=7.5)
ax.set_title("Retention (contiguous QC-passing data)", fontsize=9)

# examples: patients with the most 5-min windows that also reach the longest reliable length
ax = axes[1, 1]
cnt = df.groupby(["patient_id", "window_minutes"]).size().unstack(fill_value=0)
reach = [m for m in L if (cnt[m] >= 2).sum() >= 3]
long_ok = cnt[cnt[reach[-1]] >= 2] if reach else cnt
spread5 = df[df.window_minutes == L[0]].groupby("patient_id").cfa_r2_excl_qrs.std().reindex(long_ok.index)
ex = list(spread5.sort_values().index[[0, len(spread5) // 2, -1]]) if len(spread5) >= 3 else list(long_ok.index[:3])
for j, (pid, c) in enumerate(zip(ex, [PALETTE[0], PALETTE[1], PALETTE[2]])):
    d = df[df.patient_id == pid]
    off = (j - 1) * 0.08
    xs = np.log(d.window_minutes.values) + off + rng.uniform(-0.03, 0.03, len(d))
    # ponytail: no per-length mean line -- some patients are bimodal across the night and a mean hides that
    ax.scatter(np.exp(xs), d.cfa_r2_excl_qrs, s=6, alpha=0.5, color=c, edgecolor="none", label=str(pid))
ax.set_xscale("log")
ax.set_ylabel("CFA R² of each repeat window")
ax.legend(frameon=False, fontsize=7, markerscale=2.5)
ax.set_title("Example patients: repeat windows", fontsize=9)

ax = axes[1, 2]
ax.plot(valid.index, valid.icc, "o-", color=PALETTE[0], ms=4, lw=1.6, label="One contiguous window (observed)")
ax.plot(k * L[0], sb, "--", color=PALETTE[1], lw=1.6, label=f"Mean of k separate {L[0]}-min windows (Spearman-Brown)")
ax.plot(best, valid.icc[best], "o", ms=9, mfc="none", mec=PALETTE[1], mew=1.5)
ax.axhline(0.9, color="grey", ls="--", lw=1)
ax.set_ylabel("ICC(1)")
ax.set_ylim(top=1)
ax.legend(frameon=False, fontsize=7, loc="lower right")
ax.set_title(f"Decision: single window = {best} min", fontsize=9)

for ax in axes.flat:
    ax.set_xlabel("Window length (min)")
    ax.set_xscale("log")
    ax.set_xticks(L)
    ax.set_xticklabels([str(int(m)) for m in L], fontsize=7)
    ax.minorticks_off()
axes[1, 2].set_xlabel("Total minutes analysed")
fig.tight_layout(h_pad=2.2, w_pad=2.0)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, "figures", f"figS9_window_reliability.{ext}"), bbox_inches="tight")
print("Figure written to figures/figS9_window_reliability.png")

if len(sys.argv) == 1:  # only the real run updates paper_stats.json
    with open(os.path.join(HERE, "paper_stats.json")) as f:
        S = json.load(f)
    S["window_reliability"] = {"optimal_minutes": best, "rule": rule,
                               "spearman_brown_5min": dict(zip((k * L[0]).tolist(), sb.tolist())), "examples": [str(p) for p in ex],
                               "bias_cohort_n": len(coh), "bias_cohort_lengths": [int(m) for m in cols_c], "table": T.reset_index().to_dict(orient="records")}
    with open(os.path.join(HERE, "paper_stats.json"), "w") as f:
        json.dump(S, f, indent=2, default=float)
