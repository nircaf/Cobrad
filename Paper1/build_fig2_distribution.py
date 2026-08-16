"""
Figure 2 -- HEP population distribution.

Distribution of channel-averaged HEP amplitude (AMP_WINDOW = 0.15-0.5 s,
same metric as Paper1/build_diagnosis_alignment_analysis.py) across the
population, split by Neurological / Non-neurological / Unknown (primary
analysis: non-exclusive category membership -- a patient with >=1
neurological diagnosis category counts as Neurological regardless of
comorbid non-neurological categories; Non-neurological = >=1 non-neuro
category and zero neuro categories; Unknown = zero categories, i.e. the
undiagnosed reference cohort from mod16.select_non_diagnosis_cohort),
faceted by sleep stage. A sensitivity panel restricts to patients with
exactly one diagnosis category (n_categories == 1).

Reads the cached long-format dataset from build_hep_diagnosis_dataset.py
(Paper1/hep_diagnosis_long_df.pkl) -- no raw data reload.

Run: source venv/bin/activate && python3 Paper1/build_fig2_distribution.py
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

REPO = "/storage/pblab_shared_data2/Nir/Cobrad"
OUT_DIR = os.path.join(REPO, "Paper1")
FIG_DIR = os.path.join(OUT_DIR, "figures")
LONG_DF_PKL = os.path.join(OUT_DIR, "hep_diagnosis_long_df.pkl")
FIG2_JSON = os.path.join(OUT_DIR, "fig2_distribution_results.json")
FIG_PDF = os.path.join(FIG_DIR, "fig2_distribution.pdf")

STAGES = ["light_sleep", "N3", "R"]
STAGE_LABELS = {"light_sleep": "Light Sleep", "N3": "N3 (SWS)", "R": "REM"}
GROUP_ORDER = ["Unknown", "Non-neurological", "Neurological"]
GROUP_COLOR = {"Unknown": "#000000", "Non-neurological": "#0072B2", "Neurological": "#D55E00"}


def assign_neuro_group(row):
    if row["is_neuro"]:
        return "Neurological"
    if row["is_nonneuro"]:
        return "Non-neurological"
    return "Unknown"


def kw_bh(groups_by_stage):
    """Kruskal-Wallis per stage across the 3 broad groups, BH-adjusted across stages."""
    rows = []
    for stage, samples in groups_by_stage.items():
        samples = [s for s in samples if len(s) >= 2]
        if len(samples) >= 2:
            h, p = stats.kruskal(*samples)
        else:
            h, p = np.nan, np.nan
        rows.append({"stage": stage, "H_stat": h, "p_value": p, "n_groups": len(samples)})
    table = pd.DataFrame(rows)
    finite = table["p_value"].notna() & np.isfinite(table["p_value"])
    table["q_value"] = np.nan
    if finite.any():
        p = table.loc[finite, "p_value"].to_numpy(dtype=float)
        order = np.argsort(p)
        ranked = p[order]
        n = len(ranked)
        adj = ranked * n / (np.arange(n) + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        out = np.empty_like(adj)
        out[order] = np.clip(adj, 0, 1)
        table.loc[finite, "q_value"] = out
    return table


def main():
    with open(LONG_DF_PKL, "rb") as f:
        cached = pickle.load(f)
    avg_df = cached["avg_df"].copy()
    amp_window = cached["amp_window"]

    avg_df["neuro_group"] = avg_df.apply(assign_neuro_group, axis=1)
    avg_df["neuro_group"] = pd.Categorical(avg_df["neuro_group"], categories=GROUP_ORDER, ordered=True)

    print("N (patient-stage rows) per neuro_group x stage:")
    print(avg_df.groupby(["neuro_group", "stage"], observed=True)["patient_id"].nunique().unstack())

    # Primary analysis: Kruskal-Wallis across the 3 broad groups, per stage
    groups_by_stage = {}
    for stage in STAGES:
        groups_by_stage[stage] = [
            avg_df.loc[(avg_df["stage"] == stage) & (avg_df["neuro_group"] == g), "hep_amplitude_uv"].dropna().to_numpy()
            for g in GROUP_ORDER
        ]
    kw_table = kw_bh(groups_by_stage)
    print("\nKruskal-Wallis (Neurological vs Non-neurological vs Unknown), per stage:")
    print(kw_table.to_string())

    # Sensitivity: exactly one diagnosis category
    single_df = avg_df[(avg_df["n_categories"] <= 1)].copy()
    single_df["neuro_group"] = single_df.apply(assign_neuro_group, axis=1)
    single_df["neuro_group"] = pd.Categorical(single_df["neuro_group"], categories=GROUP_ORDER, ordered=True)
    groups_by_stage_single = {}
    for stage in STAGES:
        groups_by_stage_single[stage] = [
            single_df.loc[(single_df["stage"] == stage) & (single_df["neuro_group"] == g), "hep_amplitude_uv"].dropna().to_numpy()
            for g in GROUP_ORDER
        ]
    kw_table_single = kw_bh(groups_by_stage_single)
    print("\nSensitivity (patients with exactly 1 diagnosis category): Kruskal-Wallis per stage:")
    print(kw_table_single.to_string())
    n_single = single_df.groupby(["neuro_group", "stage"], observed=True)["patient_id"].nunique().unstack()
    print(n_single)

    # ---- Summary stats table for caption ----
    summary_rows = []
    for stage in STAGES:
        for g in GROUP_ORDER:
            vals = avg_df.loc[(avg_df["stage"] == stage) & (avg_df["neuro_group"] == g), "hep_amplitude_uv"].dropna()
            summary_rows.append({
                "stage": stage, "group": g, "n": int(len(vals)),
                "median": float(vals.median()) if len(vals) else np.nan,
                "iqr_lo": float(vals.quantile(0.25)) if len(vals) else np.nan,
                "iqr_hi": float(vals.quantile(0.75)) if len(vals) else np.nan,
            })
    summary_df = pd.DataFrame(summary_rows)

    results = {
        "amp_window": list(amp_window),
        "n_by_group_stage": {
            f"{g}|{s}": int(avg_df.loc[(avg_df['neuro_group'] == g) & (avg_df['stage'] == s), 'patient_id'].nunique())
            for g in GROUP_ORDER for s in STAGES
        },
        "kw_table": kw_table.to_dict(orient="records"),
        "kw_table_single_category": kw_table_single.to_dict(orient="records"),
        "n_single_category_by_group_stage": {
            f"{g}|{s}": int(single_df.loc[(single_df['neuro_group'] == g) & (single_df['stage'] == s), 'patient_id'].nunique())
            for g in GROUP_ORDER for s in STAGES
        },
        "summary_table": summary_df.to_dict(orient="records"),
    }
    with open(FIG2_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {FIG2_JSON}")

    # ---- Render figure: 1x3 violin/raincloud panels, one per stage ----
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.6), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, wspace=0.12, bottom=0.16, top=0.85)
    panel_letters = "ABC"
    for i, (ax, stage) in enumerate(zip(axes, STAGES)):
        data_by_group = [
            avg_df.loc[(avg_df["stage"] == stage) & (avg_df["neuro_group"] == g), "hep_amplitude_uv"].dropna().to_numpy()
            for g in GROUP_ORDER
        ]
        positions = np.arange(len(GROUP_ORDER))
        parts = ax.violinplot(data_by_group, positions=positions, showmedians=False, showextrema=False, widths=0.8)
        for pc, g in zip(parts["bodies"], GROUP_ORDER):
            pc.set_facecolor(GROUP_COLOR[g])
            pc.set_alpha(0.35)
            pc.set_edgecolor(GROUP_COLOR[g])
        bp = ax.boxplot(data_by_group, positions=positions, widths=0.12, showfliers=False,
                         patch_artist=True, medianprops=dict(color="black", linewidth=1.5))
        for patch, g in zip(bp["boxes"], GROUP_ORDER):
            patch.set_facecolor(GROUP_COLOR[g])
            patch.set_alpha(0.7)
        ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
        ax.set_xticks(positions)
        ax.set_xticklabels([g.replace("Non-neurological", "Non-\nneuro.") for g in GROUP_ORDER], fontsize=8.5)
        ax.set_title(STAGE_LABELS[stage], fontsize=10.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(-6, 8)
        krow = kw_table[kw_table["stage"] == stage].iloc[0]
        p_txt = f"p={krow['p_value']:.3f}" if np.isfinite(krow["p_value"]) and krow["p_value"] >= 0.001 else \
                (f"p={krow['p_value']:.1e}" if np.isfinite(krow["p_value"]) else "p=n/a")
        ax.text(0.5, 0.97, f"Kruskal-Wallis {p_txt}", transform=ax.transAxes, ha="center", va="top", fontsize=8)
        ax.text(-0.15, 1.12, panel_letters[i], transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="top", ha="left")
    axes[0].set_ylabel("Channel-averaged HEP amplitude\n(µV, 0.15-0.5 s window)", fontsize=9.5)
    fig.suptitle("Population distribution of HEP amplitude by diagnosis category and sleep stage", fontsize=11)
    fig.savefig(FIG_PDF, format="pdf")
    fig.savefig(FIG_PDF.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)
    print(f"Saved {FIG_PDF}")


if __name__ == "__main__":
    main()
