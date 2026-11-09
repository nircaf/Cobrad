#!/usr/bin/env python3
"""Figure S6: how CFAR works -- data in, processing layers, training, and patient-level confusion matrix.

(a) Pipeline schematic with real data from the Figure 4a example patient: 6-channel EEG + ECG in,
    training layers (R-peak epoching -> evoked average -> SVD pattern -> R^2-weighted population
    average; ECG needed), inference layer (signal-space projection I - u u^T; no ECG), clean EEG out.
(b) Training curve: median |cosine| between a filter learned from n training patients and held-out
    patients' own patterns (20 random draws per n).
(c) The learned projection layer as a 6x6 weight matrix.
(d) Confusion matrix: patients with CFA detected (R^2 above the 95th percentile of the
    non-heartbeat-locked null, r2_null) in raw EEG vs after CFAR (cleaning_demo.parquet).

Figure S7 is the same layout for patient-specific CFAR (filter = the patient's own u_i, r2_own):
(b) becomes the histogram of |cos(u_i, u)| and (c) shows the example patient's I - u_i u_i^T.

Per-patient patterns are cached in cfar_patterns.parquet (one EDF pass, same jobs as make_cleaning_demo.py).
Run: venv/bin/python "paper utils/Paper CFA/make_cfar_pipeline_figure.py"
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import make_cleaning_demo as cd

HERE, SITES, PAL = cd.HERE, cd.SITES, cd.PALETTE
CACHE = os.path.join(HERE, "cfar_patterns.parquet")
N_WORKERS = 32


def jobs():
    """Same eligible set as make_cleaning_demo.main(): I0 patients with Male/Female sex."""
    cfa = pd.read_parquet(os.path.join(HERE, "cfa_combined.parquet"),
                          columns=["patient_id", "edf_path", "window_start_s"]).drop_duplicates("patient_id")
    cfa = cfa[cfa.patient_id.str.startswith("I0")]
    dem = pd.read_parquet(os.path.join(HERE, "demographics_combined.parquet"))[["patient_id", "sex"]]
    cfa = cfa.merge(dem[dem.sex.isin(["Male", "Female"])], on="patient_id").drop_duplicates("patient_id")
    return {r.patient_id: (r.patient_id, r.edf_path, float(r.window_start_s)) for r in cfa.itertuples()}


def patterns():
    if not os.path.exists(CACHE):
        with ProcessPoolExecutor(N_WORKERS) as ex:
            fitted = [r for r in ex.map(cd.fit_job, jobs().values(), chunksize=8) if r]
        pd.DataFrame({"patient_id": [r[0] for r in fitted], "w": [r[2] for r in fitted],
                      **{s: [r[1][k] for r in fitted] for k, s in enumerate(SITES)}}).to_parquet(CACHE)
    df = pd.read_parquet(CACHE)
    P = df[SITES].to_numpy()
    P = P * np.sign(P @ cd.mean_pattern(P))[:, None]  # same sign alignment as make_cleaning_demo.main()
    return df.patient_id.to_numpy(), P, df.w.to_numpy()


def learning_curve(P, w, n_reps=20, seed=cd.SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(P))
    te, tr = idx[: len(P) // 5], idx[len(P) // 5:]
    ns = [n for n in [1, 3, 10, 30, 100, 300, 1000, 3000] if n < len(tr)] + [len(tr)]
    out = []
    for n in ns:
        meds = []
        for _ in range(n_reps if n < len(tr) else 1):
            s = rng.choice(tr, n, replace=False)
            u = w[s] @ P[s]
            meds.append(np.median(np.abs(P[te] @ (u / np.linalg.norm(u)))))
        out.append((n, np.median(meds), *np.percentile(meds, [2.5, 97.5])))
    return np.array(out)


def box(ax, xy, w, h, text, fc, ec="0.3", fs=7.5):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec=ec, lw=0.9))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, a, b, color="0.3", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=color, lw=1, ls=ls))


def traces(ax, x, sf, color, title, ecg=None):
    t = np.arange(x.shape[1]) / sf
    off = 4 * np.percentile(np.abs(x), 99)
    for k in range(len(x)):
        ax.plot(t, x[k] - k * off, color=color, lw=0.6)
        ax.text(-0.05, -k * off, SITES[k], ha="right", va="center", fontsize=6, transform=ax.get_yaxis_transform())
    if ecg is not None:
        e = (ecg - ecg.mean()) / np.ptp(ecg) * off * 1.5
        ax.plot(t, e - len(x) * off, color=PAL[3], lw=0.6)
        ax.text(-0.05, -len(x) * off, "ECG", ha="right", va="center", fontsize=6, color=PAL[3],
                transform=ax.get_yaxis_transform())
    ax.set_xlim(0, t[-1])
    ax.set_yticks([])
    ax.set_xticks([0, round(t[-1])])
    ax.tick_params(labelsize=6)
    ax.set_xlabel("s", fontsize=6, labelpad=0)
    for s in ["left", "top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=8)


def plot(P, w, curve, res, u_pop, ex, u_ex=None):
    """u_ex given -> Figure S7, patient-specific CFAR (filter = the patient's own u_i); else Figure S6."""
    own = u_ex is not None
    u = u_ex if own else u_pop
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 300, "savefig.dpi": 300})
    fig = plt.figure(figsize=(13, 8.2))
    rows = fig.add_gridspec(2, 1, height_ratios=[1.15, 1], hspace=0.25)
    bot = rows[1].subgridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.45)

    # a: schematic on an axes in figure-fraction-like [0,1] coords, real-data insets placed over it
    ax = fig.add_subplot(rows[0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("a", loc="left", fontsize=10, fontweight="bold")
    ax.text(0.5, 1.0, "Patient-specific CFAR: fit and apply on each patient's own EEG + ECG" if own else
            "CFAR pipeline: train once with ECG, apply to any EEG without ECG", ha="center",
            fontsize=9, transform=ax.transAxes)
    ax.add_patch(FancyBboxPatch((0.17, 0.53), 0.62, 0.42, boxstyle="round,pad=0.005", fc="#FFF4E8",
                                ec=PAL[1], lw=0.8, ls="--"))
    ax.text(0.175, 0.93, "FIT (ECG required; this patient's own recording, no training cohort)" if own else
            f"TRAINING (ECG used; {cd.N_FOLDS}-fold cross-fitting, "
            f"~{int(len(P) * (cd.N_FOLDS - 1) / cd.N_FOLDS):,} patients per fold)",
            fontsize=7.5, color=PAL[1], va="top", fontweight="bold")
    ax.add_patch(FancyBboxPatch((0.17, 0.03), 0.62, 0.42, boxstyle="round,pad=0.005", fc="#EAF3FA",
                                ec=PAL[0], lw=0.8, ls="--"))
    ax.text(0.175, 0.07, "APPLY (same recording the filter was fitted on)" if own else
            "INFERENCE (no ECG; held-out patient)", fontsize=7.5, color=PAL[0], fontweight="bold")
    layers = [(f"1. R-peak\nepoching\n({cd.HEP_TMIN * 1000:+.0f}…{cd.HEP_TMAX * 1000:+.0f} ms)", 0.19), ("2. Evoked\naverage\n(6 × T)", 0.315),
              ("3. SVD:\n1st spatial\nvector uᵢ", 0.44),
              ("4. Filter = uᵢ\n(no pooling\nacross patients)" if own else "4. Sign-align,\nR²-weighted\nmean → u", 0.565)]
    for text, x0 in layers:
        box(ax, (x0, 0.62), 0.1, 0.2, text, "white", ec=PAL[1])
    for (_, x0), (_, x1) in zip(layers[:-1], layers[1:]):
        arrow(ax, (x0 + 0.1, 0.72), (x1, 0.72), PAL[1])
    box(ax, (0.36, 0.13), 0.22, 0.2, "5. Projection layer\nx_clean = (I − uᵢ uᵢᵀ) x\napplied to every sample" if own
        else "5. Projection layer\nx_clean = (I − u uᵀ) x\napplied to every sample", "white", ec=PAL[0], fs=8)
    arrow(ax, (0.665, 0.62), (0.58, 0.33), "0.3", "--")  # learned filter feeds the layer
    ax.text(0.675, 0.47, "patient's filter uᵢ" if own else "learned filter u", fontsize=7, ha="left", va="center")
    arrow(ax, (0.155, 0.62), (0.19, 0.72), PAL[1])
    arrow(ax, (0.155, 0.38), (0.36, 0.23), PAL[0])
    arrow(ax, (0.58, 0.23), (0.83, 0.38), PAL[0])

    sf = ex["sfreq"]
    sl = slice(int(cd.EXAMPLE_T0 * sf), int((cd.EXAMPLE_T0 + 4) * sf))
    pos = ax.get_position()
    def inset(x0, y0, w_, h_):  # coordinates inside panel a -> figure
        return fig.add_axes([pos.x0 + x0 * pos.width, pos.y0 + y0 * pos.height, w_ * pos.width, h_ * pos.height])
    traces(inset(0.03, 0.08, 0.12, 0.8), ex["eeg"][:, sl], sf, PAL[1], "Data in: raw EEG + ECG", ex["ecg"][sl])
    traces(inset(0.86, 0.08, 0.12, 0.8), ex["own" if own else "clean"][:, sl], sf, "black",
           "Data out: after patient-specific CFAR" if own else "Data out: after CFAR")
    ia = inset(0.69, 0.56, 0.08, 0.36)
    cd.topo(ia, u)
    ia.set_title("uᵢ (this patient)" if own else "u (population)", fontsize=7)

    ax = fig.add_subplot(bot[0])
    ax.text(-0.18, 1.12, "b", transform=ax.transAxes, fontsize=10, fontweight="bold")
    if own:  # b: no training curve for a per-patient filter; show how far each u_i sits from the population u
        cos = np.abs(P @ u_pop)
        ax.hist(cos, bins=np.linspace(0, 1, 51), color=PAL[0], alpha=0.8)
        ax.axvline(np.median(cos), color="0.3", ls="--", lw=0.8)
        ax.text(np.median(cos), ax.get_ylim()[1] * 0.97, f"median {np.median(cos):.2f} ", fontsize=7,
                ha="right", va="top")
        ax.set_xlabel("|cos(uᵢ, population u)|")
        ax.set_ylabel("Patients")
        ax.set_title(f"Patient filter vs. population filter\n(n = {len(P):,} patients)", fontsize=8.5)
    else:  # b: training curve
        ax.fill_between(curve[:, 0], curve[:, 2], curve[:, 3], color=PAL[0], alpha=0.2, lw=0)
        ax.plot(curve[:, 0], curve[:, 1], "o-", color=PAL[0], ms=4, lw=1.4)
        ax.set_xscale("log")
        ax.set_xlabel("Training patients (n)")
        ax.set_ylabel("Median |cos(u, held-out uᵢ)|")
        ax.set_title(f"Training curve (held-out n = {len(P) // 5:,};\n20 draws per n, 95% range)", fontsize=8.5)
        ax.axhline(curve[-1, 1], color="0.6", ls="--", lw=0.8)
        ax.text(curve[0, 0], curve[-1, 1] + 0.005, f"plateau {curve[-1, 1]:.2f}", fontsize=7, va="bottom")

    # c: the projection layer's weights
    ax = fig.add_subplot(bot[1])
    W = np.eye(len(SITES)) - np.outer(u, u)
    im = ax.imshow(W, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(SITES)):
        for j in range(len(SITES)):
            ax.text(j, i, f"{W[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(W[i, j]) > 0.6 else "black")
    ax.set_xticks(range(len(SITES)), SITES, fontsize=7)
    ax.set_yticks(range(len(SITES)), SITES, fontsize=7)
    ax.set_xlabel("Input channel")
    ax.set_ylabel("Output channel")
    fig.colorbar(im, ax=ax, shrink=0.75, pad=0.03).ax.tick_params(labelsize=6)
    ax.text(-0.3, 1.12, "c", transform=ax.transAxes, fontsize=10, fontweight="bold")
    ax.set_title("Projection layer weights I − uᵢ uᵢᵀ\n(example patient)" if own else
                 "Projection layer weights I − u uᵀ", fontsize=8.5)

    # d: confusion matrix, CFA detected before vs after
    ax = fig.add_subplot(bot[2])
    thr = float(np.percentile(res.r2_null, 95))
    pre, post = res.r2_pre > thr, res["r2_own" if own else "r2_clean"] > thr
    M = np.array([[(pre & post).sum(), (pre & ~post).sum()], [(~pre & post).sum(), (~pre & ~post).sum()]])
    ax.imshow(M / M.sum(), cmap="Blues", vmin=0, vmax=M.max() / M.sum() * 1.2)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{M[i, j]:,}\n({M[i, j] / M.sum() * 100:.0f}%)", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] / M.max() > 0.6 else "black")
    ax.set_xticks([0, 1], ["CFA", "No CFA"])
    ax.set_yticks([0, 1], ["CFA", "No CFA"])
    ax.set_xlabel("After patient-specific CFAR" if own else "After CFAR")
    ax.set_ylabel("Raw EEG")
    ax.spines[:].set_visible(False)
    ax.text(-0.3, 1.12, "d", transform=ax.transAxes, fontsize=10, fontweight="bold")
    ax.set_title(f"Patients with CFA detected\n(R² > {thr:.2f}, null 95th pct; n = {M.sum():,})", fontsize=8.5)

    name = "figS11_cfar_own_pipeline" if own else "figS10_cfar_pipeline"
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(cd.FIG_DIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    out = {"n_patterns": int(len(P)), "null_thr": thr, "conf": M.tolist(),
           "pct_detected_pre": float(pre.mean() * 100), "pct_detected_post": float(post.mean() * 100),
           "pct_detected_cleared": float((pre & ~post).sum() / pre.sum() * 100)}
    if own:
        return {**out, "cos_pop_median": float(np.median(np.abs(P @ u_pop))),
                "cos_pop_iqr": np.percentile(np.abs(P @ u_pop), [25, 75]).tolist()}
    return {**out, "plateau_cos": float(curve[-1, 1]), "cos_at_100": float(curve[curve[:, 0] == 100, 1][0])}


def main():
    pids, P, w = patterns()
    curve = learning_curve(P, w)
    res = pd.read_parquet(os.path.join(HERE, "cleaning_demo.parquet"))
    with open(os.path.join(HERE, "paper_stats.json")) as f:
        S = json.load(f)
    u_pop = np.array([S["cleaning_demo"]["u_pop"][s] for s in SITES])
    _, ex = cd.evaluate(cd.load(jobs()[cd.EXAMPLE]), u_pop, u_pop, 0)
    S["cfar_pipeline"] = plot(P, w, curve, res, u_pop, ex)
    S["cfar_own_pipeline"] = plot(P, w, curve, res, u_pop, ex, u_ex=P[list(pids).index(cd.EXAMPLE)])
    with open(os.path.join(HERE, "paper_stats.json"), "w") as f:
        json.dump(S, f, indent=2)
    print(json.dumps({k: S[k] for k in ["cfar_pipeline", "cfar_own_pipeline"]}, indent=1))


if __name__ == "__main__":
    main()
