#!/usr/bin/env python3
"""Stats + figures for the cortical-cardiac arousal latency pilot.
Reads events_dt.csv (from run_pipeline.py), computes per-group summary
stats (n, median, IQR, sign-test p vs 0), Kruskal-Wallis across groups
with pairwise Mann-Whitney/Bonferroni post-hoc, and saves figures + a
paper_stats.json consumed by make_pdf.py.

  source venv/bin/activate && python3 "paper utils/cortical_cardiac_arousal_latency/analyze_stats.py"
"""
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, wilcoxon
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

STAGE_ORDER = ["N1", "N2", "N3", "R"]
STAGE_LABEL = {"N1": "N1", "N2": "N2", "N3": "N3", "R": "REM"}


def sign_test_p(x):
    """Two-sided sign test of median != 0 (nonparametric, no symmetry
    assumption -- used alongside Wilcoxon which assumes a symmetric
    distribution around 0)."""
    x = np.asarray(x)
    x = x[x != 0]
    if len(x) == 0:
        return np.nan
    n_pos = int((x > 0).sum())
    return binomtest(n_pos, len(x), 0.5).pvalue


def group_summary(df, group_col):
    rows = []
    for g, sub in df.groupby(group_col):
        dt = sub["dt"].dropna().values
        if len(dt) == 0:
            continue
        q1, med, q3 = np.percentile(dt, [25, 50, 75])
        p_sign = sign_test_p(dt)
        try:
            p_wil = wilcoxon(dt).pvalue if len(dt) >= 2 and np.any(dt != 0) else np.nan
        except ValueError:
            p_wil = np.nan
        rows.append(dict(group=g, n=len(dt), median=med, q1=q1, q3=q3,
                          p_sign=p_sign, p_wilcoxon=p_wil,
                          pct_heart_leads=100 * (dt < 0).mean()))
    return pd.DataFrame(rows)


def kruskal_and_posthoc(df, group_col, groups):
    samples = {g: df.loc[df[group_col] == g, "dt"].dropna().values for g in groups}
    samples = {g: v for g, v in samples.items() if len(v) >= 3}
    if len(samples) < 2:
        return dict(h=np.nan, p=np.nan, n_groups=len(samples)), []
    h, p = kruskal(*samples.values())
    pairs = list(itertools.combinations(samples.keys(), 2))
    posthoc = []
    for a, b in pairs:
        _, p_raw = mannwhitneyu(samples[a], samples[b], alternative="two-sided")
        posthoc.append(dict(a=a, b=b, p_raw=p_raw,
                             p_bonf=min(1.0, p_raw * len(pairs))))
    return dict(h=h, p=p, n_groups=len(samples)), posthoc


def box_figure(df, group_col, order, labels, title, out_path):
    groups = [g for g in order if g in df[group_col].unique()]
    data = [df.loc[df[group_col] == g, "dt"].dropna().values for g in groups]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    crowded = len(groups) > 5
    sep = " " if crowded else "\n"
    tick_labels = [f"{labels.get(g, g)}{sep}(n={len(d)})" for g, d in zip(groups, data)]
    bp = ax.boxplot(data, labels=tick_labels, showmeans=True,
                     patch_artist=True)
    # ponytail: long event-type names collide horizontally; rotate once there are many
    if crowded:
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right",
                 rotation_mode="anchor", fontsize=8)
    for patch in bp["boxes"]:
        patch.set_facecolor("#8faadc")
        patch.set_alpha(0.7)
    for i, d in enumerate(data):
        jitter = (np.random.RandomState(0).rand(len(d)) - 0.5) * 0.25
        ax.scatter(np.full(len(d), i + 1) + jitter, d, s=10, color="#31456a",
                   alpha=0.5, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("$\\Delta t = t_{HR} - t_{EEG}$ (s)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    df = pd.read_csv(HERE / "events_dt.csv")
    usable = df[df["flag"] == "ok"].copy()

    n_candidate = len(df)
    n_usable = len(usable)
    n_subjects = df["subject"].nunique()
    flag_counts = df["flag"].value_counts().to_dict()
    caisr_agree_rate = usable["caisr_arousal_agree"].dropna().mean() if \
        "caisr_arousal_agree" in usable else np.nan

    stage_summary = group_summary(usable, "stage")
    stage_summary["group_label"] = stage_summary["group"].map(STAGE_LABEL)
    type_summary = group_summary(usable, "event_type")

    stage_kw, stage_posthoc = kruskal_and_posthoc(usable, "stage", STAGE_ORDER)
    type_groups = sorted(usable["event_type"].unique())
    type_kw, type_posthoc = kruskal_and_posthoc(usable, "event_type", type_groups)

    box_figure(usable, "stage", STAGE_ORDER, STAGE_LABEL,
               "Cortical-cardiac arousal latency by sleep stage",
               FIG_DIR / "dt_by_stage.png")
    box_figure(usable, "event_type", type_groups, {g: g.replace("_", " ") for g in type_groups},
               "Cortical-cardiac arousal latency by event type",
               FIG_DIR / "dt_by_event_type.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(usable["dt"].dropna(), bins=30, color="#8faadc", edgecolor="#31456a")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("$\\Delta t = t_{HR} - t_{EEG}$ (s)")
    ax.set_ylabel("count")
    ax.set_title(f"All usable events pooled (n={n_usable})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dt_all_events.png", dpi=200)
    plt.close(fig)

    stats = dict(
        n_candidate_events=int(n_candidate),
        n_usable_events=int(n_usable),
        n_subjects=int(n_subjects),
        flag_counts={k: int(v) for k, v in flag_counts.items()},
        caisr_arousal_agree_rate=None if pd.isna(caisr_agree_rate) else float(caisr_agree_rate),
        pooled_median_dt=float(usable["dt"].median()) if n_usable else None,
        pooled_q1=float(usable["dt"].quantile(0.25)) if n_usable else None,
        pooled_q3=float(usable["dt"].quantile(0.75)) if n_usable else None,
        pooled_sign_p=float(sign_test_p(usable["dt"].dropna().values)) if n_usable else None,
        stage_summary=stage_summary.to_dict(orient="records"),
        type_summary=type_summary.to_dict(orient="records"),
        stage_kruskal=stage_kw,
        stage_posthoc=stage_posthoc,
        type_kruskal=type_kw,
        type_posthoc=type_posthoc,
        event_type_counts={k: int(v) for k, v in df["event_type"].value_counts().items()},
        stage_counts={k: int(v) for k, v in df["stage"].value_counts().items()},
    )
    with open(HERE / "paper_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    print(json.dumps({k: v for k, v in stats.items()
                       if k in ("n_candidate_events", "n_usable_events", "n_subjects",
                                 "flag_counts", "pooled_median_dt", "pooled_sign_p",
                                 "caisr_arousal_agree_rate")}, indent=2))


if __name__ == "__main__":
    main()
