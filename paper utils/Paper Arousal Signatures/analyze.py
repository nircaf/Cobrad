"""Stats + figure for the arousal-cause cardiocortical comparison, from events_features.parquet."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

HERE = Path(__file__).resolve().parent
FEATURES = [
    "delta_hr", "hr_latency_s",
    "eeg_delta_mean", "eeg_alpha_mean", "eeg_beta_mean",
    "eeg_entropy_delta",
    "eeg_delta_spread", "eeg_alpha_spread", "eeg_beta_spread",
]
CLASSES = ["spontaneous", "RERA", "obstructive apnea", "central apnea", "PLM"]


def run(csv_path: Path = HERE / "events_features.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["cause"].isin(CLASSES)]

    rows = []
    for feat in FEATURES:
        groups = [df.loc[df["cause"] == c, feat].dropna() for c in CLASSES if (df["cause"] == c).any()]
        present = [c for c in CLASSES if (df["cause"] == c).any()]
        if len(groups) < 2 or any(len(g) < 2 for g in groups):
            rows.append({"feature": feat, "n_classes": len(groups), "kruskal_p": float("nan")})
            continue
        h, p = kruskal(*groups)
        rows.append({"feature": feat, "n_classes": len(groups), "kruskal_p": p})
        for a, b in combinations(present, 2):
            ga = df.loc[df["cause"] == a, feat].dropna()
            gb = df.loc[df["cause"] == b, feat].dropna()
            if len(ga) < 2 or len(gb) < 2:
                continue
            _, pw = mannwhitneyu(ga, gb)
            rows.append({"feature": feat, "pair": f"{a} vs {b}", "mannwhitney_p": pw})

    results = pd.DataFrame(rows)
    results.to_csv(HERE / "stats_results.csv", index=False)

    n = len(FEATURES)
    fig, axs = plt.subplots(3, 3, figsize=(13, 10))
    for ax, feat in zip(axs.flat, FEATURES):
        data = [df.loc[df["cause"] == c, feat].dropna() for c in CLASSES if (df["cause"] == c).any()]
        labels = [c for c in CLASSES if (df["cause"] == c).any()]
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(feat, fontsize=9)
        ax.tick_params(axis="x", rotation=40, labelsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "feature_comparison.png", dpi=200)
    plt.close(fig)

    print(df["cause"].value_counts())
    print(results)
    return results


if __name__ == "__main__":
    (HERE / "figures").mkdir(exist_ok=True)
    run()
