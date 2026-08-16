"""
Brain-Body Synchrony / Dysrhythmia Dashboard.

Tests the hypotheses of:
  Young A, et al. (2026). "I sync, therefore I am: brain-body synchrony in
  typical and disordered consciousness." Neuroscience of Consciousness,
  2026(1), niag028. https://doi.org/10.1093/nc/niag028

against this repo's existing HEP + sleep-stage + diagnosis data. Reuses
6_hep_group_comparison.get_group_individuals (via its individuals_cache*.pkl,
built by 17_generate_hep_cache.py) and 16_diagnosis_sleep_stage_comparison_
dashboard.py's diagnosis/loading machinery. Does not recompute or regenerate
any cache.

Run:
  streamlit run 23_brain_body_synchrony_dysrhythmia_dashboard.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import List, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# =============================================================================
# Import 16_diagnosis_sleep_stage_comparison_dashboard.py as a module. It in
# turn knows how to load 6_hep_group_comparison.py the same way (transiently
# no-oping st.set_page_config during import, since 16 must be the dashboard
# that actually calls it). Reused directly here: load_patient_data,
# load_ehr_data, select_non_diagnosis_cohort, _patient_hep_trace,
# benjamini_hochberg, STAGE_ORDER/STAGE_LABELS, BASE_PATH,
# MONTAGE_1020_CHANNEL_ORDER, PALETTE, _canonical_patient_id, list_groups,
# list_stages, format_p, pval_to_stars. Neither module executes any
# Streamlit widget code at import time (only inside main(), guarded by
# `if __name__ == "__main__"`), so importing them here is side-effect free.
# =============================================================================
_MOD16_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "16_diagnosis_sleep_stage_comparison_dashboard.py")


@st.cache_resource(show_spinner="Loading diagnosis/HEP dashboard module...")
def _load_mod16():
    spec = importlib.util.spec_from_file_location("diag_dashboard_mod16", _MOD16_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["diag_dashboard_mod16"] = mod
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# Constants
# =============================================================================
DIAG_GROUPS = [
    "Atrial Fibrillation",
    "Heart Failure",
    "Stroke / Cerebrovascular",
    "Cognitive Impairment / Dementia",
]
REF_GROUP = "Susp. Epilepsy"  # identical no-diagnosis reference cohort used in 16 (select_non_diagnosis_cohort)
# Canonical post-R-peak T-wave / HEP analysis window, matching 16's default
# t_window for every non-"Summary metrics" mode (16_diagnosis_sleep_stage_
# comparison_dashboard.py main(), ~line 9001) and the window fed into
# _channel_window_statistics elsewhere in that file.
AMP_WINDOW = (0.15, 0.5)
PAPER_CITATION = (
    "Young A, et al. (2026). “I sync, therefore I am: brain-body synchrony "
    "in typical and disordered consciousness.” *Neuroscience of "
    "Consciousness*, 2026(1), niag028."
)
PAPER_DOI = "https://doi.org/10.1093/nc/niag028"


# =============================================================================
# Statistics — Mann-Whitney U with rank-biserial effect size r = Z / sqrt(N),
# via the normal approximation to U (mu_U, sigma_U), matching the math panel.
# =============================================================================

def mannwhitney_with_effect(a: np.ndarray, b: np.ndarray, alternative: str = "two-sided") -> dict:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float)
    b = b[np.isfinite(b)]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return {"U": np.nan, "p": np.nan, "z": np.nan, "r": np.nan, "n1": n1, "n2": n2}
    try:
        u, p = stats.mannwhitneyu(a, b, alternative=alternative)
    except Exception:
        return {"U": np.nan, "p": np.nan, "z": np.nan, "r": np.nan, "n1": n1, "n2": n2}
    mu_u = n1 * n2 / 2.0
    sigma_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (u - mu_u) / sigma_u if sigma_u > 0 else np.nan
    r = z / np.sqrt(n1 + n2) if np.isfinite(z) else np.nan
    return {"U": float(u), "p": float(p), "z": float(z), "r": float(r), "n1": n1, "n2": n2}


# =============================================================================
# Data assembly
# =============================================================================

def build_grouped_df(
    mod16, hep_mod, hep_groups: Sequence[str], stages: Sequence[str], min_eeg_channels: int,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """
    raw_df: one row per (group, stage, patient, individual), via 16's
    load_patient_data (which itself delegates entirely to
    6_hep_group_comparison.get_group_individuals — no cache regeneration).

    grouped_df: raw_df rows duplicated into one row per matching diagnosis
    category (a patient with two of the four target diagnoses appears in both
    groups — same non-exclusive "By Diagnosis Category" convention 16 uses),
    plus the reference cohort via 16's select_non_diagnosis_cohort.
    """
    raw_df = mod16.load_patient_data(
        hep_mod, hep_groups, stages, min_eeg_channels=min_eeg_channels,
    )
    if raw_df.empty:
        return raw_df, pd.DataFrame()

    patient_ids = tuple(sorted(raw_df["patient_id"].unique()))
    ehr_df = mod16.load_ehr_data(patient_ids)
    ref_rows = mod16.select_non_diagnosis_cohort(raw_df, ehr_df)

    categories_by_patient = (
        dict(zip(ehr_df["patient_id"], ehr_df["categories"])) if not ehr_df.empty else {}
    )
    frames = [ref_rows]
    for diag_group in DIAG_GROUPS:
        matched_ids = {
            pid for pid, cats in categories_by_patient.items()
            if isinstance(cats, list) and diag_group in cats
        }
        subset = raw_df[raw_df["patient_id"].isin(matched_ids)].copy()
        subset["diagnosis_group"] = diag_group
        frames.append(subset)
    grouped_df = pd.concat(frames, ignore_index=True)
    return raw_df, grouped_df


def compute_amplitudes(
    grouped_df: pd.DataFrame, mod16, selected_channels: Sequence[str], zscore_subjects: bool,
) -> pd.DataFrame:
    """One HEP amplitude scalar per (diagnosis_group, stage, patient): the mean
    of the channel-averaged HEP trace (16's _patient_hep_trace) inside AMP_WINDOW."""
    rows = []
    for _, row in grouped_df.iterrows():
        trace_result = mod16._patient_hep_trace(row["individual"], selected_channels, zscore_subjects)
        if trace_result is None:
            continue
        trace, times = trace_result
        window_mask = (times >= AMP_WINDOW[0]) & (times <= AMP_WINDOW[1])
        if window_mask.sum() < 2:
            continue
        rows.append({
            "diagnosis_group": row["diagnosis_group"],
            "stage": row["stage"],
            "patient_id": mod16._canonical_patient_id(row["patient_id"]),
            "amplitude": float(np.nanmean(trace[window_mask])),
        })
    amp_df = pd.DataFrame(rows)
    if not amp_df.empty:
        amp_df = amp_df.drop_duplicates(subset=["diagnosis_group", "stage", "patient_id"])
    return amp_df


def add_zscores(amp_df: pd.DataFrame, ref_group: str) -> pd.DataFrame:
    """H2: z_i = (A_i - mean(A_ref)) / std(A_ref), computed within each stage
    from the reference group's own distribution."""
    amp_df = amp_df.copy()
    amp_df["z_vs_ref"] = np.nan
    for stage, stage_df in amp_df.groupby("stage"):
        ref_vals = stage_df.loc[stage_df["diagnosis_group"] == ref_group, "amplitude"].dropna().to_numpy()
        if len(ref_vals) < 2:
            continue
        mu, sd = float(np.nanmean(ref_vals)), float(np.nanstd(ref_vals, ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            continue
        stage_mask = amp_df["stage"] == stage
        amp_df.loc[stage_mask, "z_vs_ref"] = (amp_df.loc[stage_mask, "amplitude"] - mu) / sd
    amp_df["abs_z_vs_ref"] = amp_df["z_vs_ref"].abs()
    return amp_df


def compute_results_table(amp_df: pd.DataFrame, diag_groups: Sequence[str], stages: Sequence[str], ref_group: str, bh_func) -> pd.DataFrame:
    """Per group x stage: signed Mann-Whitney (H1) and |z|-deviation Mann-Whitney (H2), side by side."""
    rows = []
    for group in diag_groups:
        for stage in stages:
            group_stage = amp_df[(amp_df["diagnosis_group"] == group) & (amp_df["stage"] == stage)]
            ref_stage = amp_df[(amp_df["diagnosis_group"] == ref_group) & (amp_df["stage"] == stage)]
            signed = mannwhitney_with_effect(
                group_stage["amplitude"].to_numpy(), ref_stage["amplitude"].to_numpy(), alternative="two-sided",
            )
            absz = mannwhitney_with_effect(
                group_stage["abs_z_vs_ref"].to_numpy(), ref_stage["abs_z_vs_ref"].to_numpy(), alternative="greater",
            )
            rows.append({
                "Group": group, "Stage": stage,
                "N_group": signed["n1"], "N_ref": signed["n2"],
                "Signed_p": signed["p"], "Signed_r": signed["r"],
                "Signed_mean_diff": (
                    float(np.nanmean(group_stage["amplitude"])) - float(np.nanmean(ref_stage["amplitude"]))
                    if signed["n1"] and signed["n2"] else np.nan
                ),
                "AbsZ_p": absz["p"], "AbsZ_r": absz["r"],
            })
    table = pd.DataFrame(rows)
    table["Signed_q"] = np.nan
    table["AbsZ_q"] = np.nan
    signed_finite = table["Signed_p"].notna() & np.isfinite(table["Signed_p"])
    absz_finite = table["AbsZ_p"].notna() & np.isfinite(table["AbsZ_p"])
    # BH-FDR over the FULL family of group x stage x hypothesis tests together.
    all_p, all_idx = [], []
    for col, mask in (("Signed_p", signed_finite), ("AbsZ_p", absz_finite)):
        for i in table.index[mask]:
            all_p.append(table.loc[i, col])
            all_idx.append((col, i))
    if all_p:
        q_all = bh_func(np.array(all_p, dtype=float))
        for (col, i), q in zip(all_idx, q_all):
            table.loc[i, "Signed_q" if col == "Signed_p" else "AbsZ_q"] = q
    return table


def compute_h3_table(amp_df: pd.DataFrame, diag_groups: Sequence[str], stages: Sequence[str], bh_func) -> pd.DataFrame:
    """H3: per diagnosis group, Kruskal-Wallis on |z| deviation across the
    selected sleep stages — is the atypicality of coupling state-dependent?"""
    rows = []
    for group in diag_groups:
        stage_vals = [
            amp_df.loc[(amp_df["diagnosis_group"] == group) & (amp_df["stage"] == stage), "abs_z_vs_ref"].dropna().to_numpy()
            for stage in stages
        ]
        stage_vals = [v for v in stage_vals if len(v) >= 2]
        if len(stage_vals) >= 2:
            try:
                h_stat, p = stats.kruskal(*stage_vals)
            except Exception:
                h_stat, p = np.nan, np.nan
        else:
            h_stat, p = np.nan, np.nan
        rows.append({"Group": group, "H_stat": h_stat, "p_value": p, "N_stages": len(stage_vals)})
    table = pd.DataFrame(rows)
    table["q_value"] = np.nan
    finite = table["p_value"].notna() & np.isfinite(table["p_value"])
    if finite.any():
        table.loc[finite, "q_value"] = bh_func(table.loc[finite, "p_value"].to_numpy(dtype=float))
    return table


# =============================================================================
# Plotting
# =============================================================================

def plot_group_distributions(amp_df: pd.DataFrame, group: str, stages: Sequence[str], ref_group: str, mod16) -> go.Figure:
    """Violin distributions of HEP amplitude, group vs reference, per stage."""
    fig = go.Figure()
    palette = mod16.PALETTE
    for si, stage in enumerate(stages):
        stage_label = mod16.STAGE_LABELS.get(stage, stage)
        for gi, (label, name) in enumerate([(group, group), (ref_group, "Reference")]):
            vals = amp_df.loc[
                (amp_df["diagnosis_group"] == label) & (amp_df["stage"] == stage), "amplitude"
            ].dropna().to_numpy()
            fig.add_trace(go.Violin(
                x=[f"{stage_label}<br>{name} (N={len(vals)})"] * len(vals),
                y=vals,
                name=name,
                legendgroup=name,
                showlegend=(si == 0),
                marker_color=palette[gi % len(palette)],
                box_visible=True,
                meanline_visible=True,
            ))
    fig.update_layout(
        title=f"HEP T-wave amplitude: {group} vs {ref_group}",
        yaxis_title="Amplitude (µV, mean in analysis window)",
        height=450, plot_bgcolor="white", paper_bgcolor="white", font_color="black",
    )
    return fig


def plot_absz_heatmap(results_table: pd.DataFrame, diag_groups: Sequence[str], stages: Sequence[str], mod16) -> go.Figure:
    """Group x Stage heatmap of the H2 |z|-deviation effect size r (H3 visualization)."""
    stage_labels = [mod16.STAGE_LABELS.get(s, s) for s in stages]
    z_vals, text_vals = [], []
    for group in diag_groups:
        row_r, row_t = [], []
        for stage in stages:
            match = results_table[(results_table["Group"] == group) & (results_table["Stage"] == stage)]
            r = float(match["AbsZ_r"].iloc[0]) if not match.empty else np.nan
            q = float(match["AbsZ_q"].iloc[0]) if not match.empty else np.nan
            row_r.append(r)
            row_t.append(f"{r:.2f}<br>q={q:.3f}" if np.isfinite(r) and np.isfinite(q) else "NA")
        z_vals.append(row_r)
        text_vals.append(row_t)
    fig = go.Figure(go.Heatmap(
        z=z_vals, x=stage_labels, y=diag_groups, text=text_vals, texttemplate="%{text}",
        colorscale="Reds", zmin=0, colorbar_title="r (|z| effect)", hoverongaps=False,
    ))
    fig.update_layout(
        title="H3: |z|-deviation effect size by group × stage",
        xaxis_title="Sleep stage", yaxis_title="Diagnosis group",
        height=max(300, 70 * len(diag_groups) + 120),
    )
    return fig


# =============================================================================
# Conclusions — computed programmatically from p/q-values and effect signs.
# =============================================================================

def build_conclusions(results_table: pd.DataFrame, h3_table: pd.DataFrame, diag_groups: Sequence[str], alpha: float = 0.05) -> List[str]:
    lines = []
    for group in diag_groups:
        group_rows = results_table[results_table["Group"] == group]
        signed_sig = group_rows[group_rows["Signed_q"] < alpha]
        absz_sig = group_rows[group_rows["AbsZ_q"] < alpha]
        h3_row = h3_table[h3_table["Group"] == group]
        h3_sig = bool((h3_row["q_value"] < alpha).any()) if not h3_row.empty else False

        if not absz_sig.empty and signed_sig.empty:
            direction = "mixed/bidirectional" if (group_rows["Signed_mean_diff"] > 0).any() and (group_rows["Signed_mean_diff"] < 0).any() else "atypical (magnitude only)"
            verdict = (
                f"**{group}**: supports H1+H2 (atypical, bidirectional coupling) — "
                f"|z|-deviation from reference is significant in "
                f"{len(absz_sig)}/{len(group_rows)} stage(s) (q<{alpha}), "
                f"while the naive signed comparison is null in every stage. "
                f"Direction across stages: {direction}."
            )
        elif not signed_sig.empty:
            mean_diffs = signed_sig["Signed_mean_diff"].to_numpy()
            if np.all(mean_diffs > 0) or np.all(mean_diffs < 0):
                sign_word = "increase" if mean_diffs[0] > 0 else "decrease"
                verdict = (
                    f"**{group}**: consistent unidirectional {sign_word} vs reference "
                    f"in {len(signed_sig)}/{len(group_rows)} stage(s) (signed q<{alpha}) — "
                    f"a simple deficit/excess pattern, not the paper's bidirectional claim."
                )
            else:
                verdict = (
                    f"**{group}**: signed effect significant but sign varies by stage "
                    f"({len(signed_sig)}/{len(group_rows)} stage(s), q<{alpha}) — "
                    f"consistent with H1+H2 bidirectionality."
                )
        else:
            verdict = f"**{group}**: no significant coupling effect (signed or |z|) at q<{alpha} in any tested stage."

        verdict += (
            f" H3 (state-dependence): {'supported' if h3_sig else 'not supported'} "
            f"(Kruskal-Wallis on |z| across stages, q<{alpha}: {h3_sig})."
        )
        lines.append(verdict)
    return lines


def build_interpretation(results_table: pd.DataFrame, h3_table: pd.DataFrame, diag_groups: Sequence[str], alpha: float = 0.05) -> List[str]:
    """Plain-language narrative per group: what the numbers mean and how they
    map back onto the paper's dysrhythmia / optimal-window / state-dependence
    claims. Built entirely from the computed tables, not hardcoded."""
    blocks = []
    for group in diag_groups:
        rows = results_table[results_table["Group"] == group]
        if rows.empty:
            continue
        n_group = int(rows["N_group"].max())
        n_ref = int(rows["N_ref"].max())
        signed_sig_rows = rows[rows["Signed_q"] < alpha]
        absz_sig_rows = rows[rows["AbsZ_q"] < alpha]
        best_absz = rows.loc[rows["AbsZ_p"].idxmin()] if rows["AbsZ_p"].notna().any() else None
        h3_row = h3_table[h3_table["Group"] == group]
        h3_sig = bool((h3_row["q_value"] < alpha).any()) if not h3_row.empty else False
        h3_p = float(h3_row["p_value"].iloc[0]) if not h3_row.empty and h3_row["p_value"].notna().any() else np.nan

        text = [f"**{group}** (N={n_group} patients vs N={n_ref} reference):"]

        if not absz_sig_rows.empty and signed_sig_rows.empty:
            stage_list = ", ".join(absz_sig_rows["Stage"].astype(str).tolist())
            text.append(
                f"the *magnitude* of coupling abnormality (|z|-deviation from the reference cohort) is "
                f"significantly elevated in {len(absz_sig_rows)}/{len(rows)} stage(s) ({stage_list}, BH-q<{alpha}), "
                f"while the plain group-mean-vs-reference comparison shows **no** significant difference anywhere. "
                f"This is the exact signature the paper predicts for dysrhythmia-linked conditions: individual "
                f"patients drift away from the healthy 'optimal window' of brain-body synchrony in **both** "
                f"directions, so a naive mean comparison averages the over- and under-coupled patients back "
                f"toward the reference mean and looks null, even though something is clearly abnormal at the "
                f"individual level."
            )
        elif not signed_sig_rows.empty:
            mean_diffs = signed_sig_rows["Signed_mean_diff"].to_numpy()
            if np.all(mean_diffs > 0) or np.all(mean_diffs < 0):
                text.append(
                    f"amplitude shifts consistently {'up' if mean_diffs[0] > 0 else 'down'} vs the reference cohort "
                    f"in {len(signed_sig_rows)}/{len(rows)} stage(s) (BH-q<{alpha}). This looks like a plain, "
                    f"one-directional deficit/excess in HEP amplitude rather than the paper's bidirectional "
                    f"'optimal window' pattern — i.e. this group's data are more consistent with a classic "
                    f"group-difference story than with the paper's dysrhythmia framing."
                )
            else:
                text.append(
                    f"amplitude differs significantly from reference in {len(signed_sig_rows)}/{len(rows)} stage(s) "
                    f"(BH-q<{alpha}), but the *direction* of the shift is not consistent across stages — some "
                    f"stages show higher amplitude, others lower. That sign-flipping pattern is itself consistent "
                    f"with the paper's bidirectional/optimal-window claim, not a simple deficit."
                )
        else:
            text.append(
                "no significant difference from the reference cohort, signed or in absolute deviation, in any "
                f"tested stage (BH-q<{alpha}). With this sample size, the data neither confirm nor rule out the "
                "paper's hypothesis for this group — read this as inconclusive, not as evidence against it."
            )

        if best_absz is not None:
            text.append(
                f"Largest single-stage |z|-deviation effect: r={best_absz['AbsZ_r']:.2f} "
                f"(stage={best_absz['Stage']}, p={best_absz['AbsZ_p']:.4f})."
            )

        if h3_sig:
            text.append(
                f"The |z|-deviation also varies significantly **across sleep stages** for this group "
                f"(Kruskal-Wallis p={h3_p:.4f}, BH-q<{alpha}), matching the paper's claim that brain-body "
                "coupling is state-dependent rather than a fixed trait of the diagnosis."
            )
        else:
            text.append(
                "No significant across-stage variation in |z|-deviation was detected for this group, so the "
                "state-dependence claim (H3) is not supported here — coupling abnormality, if present, looks "
                "similar across light sleep, N3 and REM in this cohort."
            )

        blocks.append(" ".join(text))
    return blocks


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    st.set_page_config(page_title="Brain-Body Synchrony / Dysrhythmia", layout="wide")
    st.title("\U0001FAC0\U0001F9E0 Brain-Body Synchrony & Dysrhythmia Dashboard")

    st.markdown(f"""
**Paper:** {PAPER_CITATION} [{PAPER_DOI}]({PAPER_DOI})

Consciousness reflects a recurrent brain-body dialogue: cardiac, respiratory
and gastric rhythms modulate neural activity supporting interoception,
emotion and selfhood. The paper's core claim is that this brain-body coupling
is disturbed in disease **bidirectionally** — atypically increased *or*
decreased — rather than simply reduced, with an "optimal window" of synchrony
beyond which either direction is adverse. This dashboard tests three
sub-hypotheses against this repo's cached Heartbeat-Evoked Potential (HEP)
data:
- **H1** — diagnosis groups tied to cardiac/autonomic dysrhythmia or
  disrupted brain-body integration show HEP amplitude that differs from a
  no-diagnosis reference cohort (direction not assumed).
- **H2** — the *absolute* deviation |z| of a patient's HEP amplitude from the
  reference distribution is elevated in these groups even when the *signed*
  mean difference is null (bidirectional effects cancel in a naive test).
- **H3** — the coupling difference is state-dependent: it is tested and
  compared separately across light sleep, N3 and REM.
""")

    with st.expander("Math panel"):
        st.latex(r"z_i = \frac{A_i - \overline{A_{\text{ref}}}}{\mathrm{SD}(A_{\text{ref}})}")
        st.caption("H2: patient i's HEP amplitude expressed relative to the reference group's mean/SD, computed within stage.")
        st.latex(r"U = \sum \text{rank-based statistic (Mann–Whitney)}")
        st.caption("Non-parametric two-sample test comparing a diagnosis group to the reference cohort, on signed amplitude (H1) or on |z| (H2, one-sided).")
        st.latex(r"r = \frac{Z}{\sqrt{N}}, \quad Z = \frac{U - n_1 n_2/2}{\sqrt{n_1 n_2 (n_1+n_2+1)/12}}")
        st.caption("Rank-biserial effect size from the normal approximation to U, N = n1 + n2.")

    mod16 = _load_mod16()
    hep_mod = mod16._load_hep_module()

    st.sidebar.header("Settings")
    selected_groups = st.sidebar.multiselect(
        "Diagnosis groups", DIAG_GROUPS, default=DIAG_GROUPS,
    )
    if not selected_groups:
        st.warning("Select at least one diagnosis group.")
        return

    available_stages = mod16.list_stages(mod16.BASE_PATH, mod16.list_groups(mod16.BASE_PATH))
    default_stages = [s for s in mod16.STAGE_ORDER if s in available_stages] or available_stages
    selected_stages = st.sidebar.multiselect(
        "Sleep stages", [s for s in mod16.STAGE_ORDER if s in available_stages],
        default=default_stages, format_func=lambda s: mod16.STAGE_LABELS.get(s, s),
    )
    if not selected_stages:
        st.warning("Select at least one sleep stage.")
        return

    zscore_subjects = st.sidebar.toggle(
        "Z-score each patient's trace before averaging (H2)", value=False,
        help="Passed to 16's _patient_hep_trace(zscore_subjects=...): normalizes each patient's own waveform before the window mean is taken, ahead of the separate H2 z-score-vs-reference test.",
    )
    only_gt10_eeg = st.sidebar.toggle("Only patients with ≥10 standard 10-20 EEG channels", value=True)

    hep_source_groups = [g for g in mod16.DEFAULT_HEP_GROUPS if g in mod16.list_groups(mod16.BASE_PATH)] or mod16.list_groups(mod16.BASE_PATH)

    with st.spinner("Loading HEP data (reusing 6_hep_group_comparison's individuals_cache*.pkl)..."):
        raw_df, grouped_df = build_grouped_df(
            mod16, hep_mod, hep_source_groups, selected_stages,
            min_eeg_channels=10 if only_gt10_eeg else None,
        )
    if raw_df.empty or grouped_df.empty:
        st.error("No patient data found. Check that individuals_cache*.pkl exist under BASE_PATH.")
        return

    montage_by_lower = {c.lower(): c for c in mod16.MONTAGE_1020_CHANNEL_ORDER}
    available_channels = sorted({
        montage_by_lower[str(ch).lower()]
        for individual in raw_df["individual"]
        if individual is not None and len(individual) > 3 and individual[3] is not None
        for ch in individual[3]
        if str(ch).lower() in montage_by_lower
    }, key=lambda c: mod16.MONTAGE_1020_CHANNEL_ORDER.index(c))
    selected_channels = st.sidebar.multiselect(
        "EEG channels", available_channels, default=available_channels,
    )
    if not selected_channels:
        st.warning("Select at least one EEG channel.")
        return

    amp_df = compute_amplitudes(grouped_df, mod16, selected_channels, zscore_subjects)
    if amp_df.empty:
        st.error("No valid HEP traces for the selected channels/stages.")
        return
    amp_df = add_zscores(amp_df, REF_GROUP)

    st.header("HEP amplitude distributions")
    for group in selected_groups:
        st.plotly_chart(
            plot_group_distributions(amp_df, group, selected_stages, REF_GROUP, mod16),
            use_container_width=True, theme=None,
        )

    st.header("Results: signed comparison (H1) vs |z|-deviation comparison (H2)")
    results_table = compute_results_table(amp_df, selected_groups, selected_stages, REF_GROUP, mod16.benjamini_hochberg)
    display_table = results_table.copy()
    display_table["Stage"] = display_table["Stage"].map(lambda s: mod16.STAGE_LABELS.get(s, s))
    for col in ("Signed_p", "Signed_q", "AbsZ_p", "AbsZ_q"):
        display_table[col] = display_table[col].map(mod16.format_p)
    st.dataframe(display_table, use_container_width=True)
    st.caption(
        "Signed_* = two-sided Mann-Whitney on raw amplitude vs reference (naive H1 test, cannot "
        "distinguish a null effect from bidirectional effects cancelling). AbsZ_* = one-sided "
        "Mann-Whitney (alternative=greater) on |z_vs_ref| vs the reference's own |z| (H2). "
        "q-values are BH-FDR corrected across the full group × stage × hypothesis family."
    )

    st.header("H3: state-dependence heatmap")
    st.plotly_chart(plot_absz_heatmap(results_table, selected_groups, selected_stages, mod16), use_container_width=True, theme=None)
    h3_table = compute_h3_table(amp_df, selected_groups, selected_stages, mod16.benjamini_hochberg)
    h3_display = h3_table.copy()
    h3_display["p_value"] = h3_display["p_value"].map(mod16.format_p)
    h3_display["q_value"] = h3_display["q_value"].map(mod16.format_p)
    st.dataframe(h3_display, use_container_width=True)
    st.caption("Kruskal-Wallis on |z_vs_ref| across the selected sleep stages, per diagnosis group.")

    st.header("Conclusions")
    for line in build_conclusions(results_table, h3_table, selected_groups):
        st.markdown(line)

    st.header("Interpretation: what this means and how it maps to the paper")
    st.markdown(
        "Young et al. 2026 argue that clinical conditions disturb brain-body coupling "
        "**bidirectionally** (atypically higher *or* lower, not just reduced) and that this "
        "coupling sits in an 'optimal window' beyond which either direction is adverse, with "
        "the effect modulated by arousal/sleep state. The paragraphs below translate the "
        "statistics above into plain language for each diagnosis group, tying every claim back "
        "to a specific number computed on this repo's data."
    )
    for block in build_interpretation(results_table, h3_table, selected_groups):
        st.markdown(block)
        st.divider()


if __name__ == "__main__":
    main()
