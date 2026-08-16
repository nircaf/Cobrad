"""
Re-export the 3 age-median-split figures as clean 2-panel plots (trace + T-stat),
mirroring dash16._plot_delta_summary_and_tstat's layout exactly, so that
add_topomap_inset_to_waveform's hardcoded inset/legend positions (tuned for a
2-row figure) land correctly.

Root cause of the overlap: the dashboard's own _render_gender_age_tab combines
plot_waveform_contrast (3-panel: mean HEP / mean-difference / T-stat) with
add_topomap_inset_to_waveform, whose inset/legend y-coordinates assume a
2-panel figure (that combination is exercised nowhere else in the codebase
with a topomap overlay, so the mismatch was latent). Rather than hand-patch
coordinates or touch the shared dashboard module, this script builds the
missing 2-panel independent-samples equivalent of _plot_delta_summary_and_tstat
locally, using the same dash16 building blocks (_pointwise_t_components,
PALETTE, HEP_ARTIFACT_EXCLUDE_S, _dilate_bool_mask) — no statistics reimplemented.

Run with the project venv: source venv/bin/activate && python3 Paper1/fix_agesplit_figures.py
"""
import os
import sys
import types
import pickle
import importlib.util

import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO = "/storage/pblab_shared_data2/Nir/Cobrad"
OUT_DIR = os.path.join(REPO, "Paper1")
FIG_DIR = os.path.join(OUT_DIR, "figures")
N_PERM = 200
STAGES = ["light_sleep", "N3", "R"]


def _make_mock_streamlit():
    class _NoOp:
        def __init__(self, *a, **kw): pass
        def __call__(self, *a, **kw): return _NoOp()
        def __getattr__(self, name): return _NoOp()

    def _cache_data(func=None, **kwargs):
        return func if func is not None else (lambda f: f)

    st = types.ModuleType("streamlit")
    st.cache_data = _cache_data
    st.cache_resource = _cache_data
    st.session_state = {}
    st.sidebar = _NoOp()
    st.__enter__ = lambda s: s
    st.__exit__ = lambda s, *a: False
    for name in ["warning", "error", "info", "write", "title", "header", "subheader",
                 "markdown", "text", "pyplot", "spinner", "set_page_config", "stop",
                 "experimental_rerun", "rerun"]:
        setattr(st, name, (lambda *a, **kw: None))
    st.empty = lambda *a, **kw: _NoOp()
    st.progress = lambda *a, **kw: _NoOp()
    st.columns = lambda *a, **kw: [_NoOp() for _ in range(a[0] if a else 2)]
    st.tabs = lambda labels: [_NoOp() for _ in labels]
    st.expander = lambda *a, **kw: _NoOp()
    components = types.ModuleType("streamlit.components")
    components.v1 = _NoOp()
    st.components = components
    return st


def load_dash16_module():
    dash_path = os.path.join(REPO, "16_diagnosis_sleep_stage_comparison_dashboard.py")
    spec = importlib.util.spec_from_file_location("dash16_module", dash_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dash16_module"] = mod
    spec.loader.exec_module(mod)
    return mod


def plot_two_group_summary_and_tstat(dash16, result, label_a, label_b, stage_label):
    """2-panel (trace, T-stat) figure for an independent-samples contrast,
    structurally identical to dash16._plot_delta_summary_and_tstat's layout
    (same make_subplots call, same margin/legend geometry) so that
    add_topomap_inset_to_waveform's fixed inset position is correct."""
    a, b, times = result["matrix_a"], result["matrix_b"], result["times"]
    n_a, n_b = a.shape[0], b.shape[0]
    tc = dash16._pointwise_t_components(a, b, paired=False)
    mean_a, mean_b, sem_a, sem_b = tc["mean_a"], tc["mean_b"], tc["sem_a"], tc["sem_b"]
    t_stat = np.asarray(result["t_stat"], dtype=float)
    t_threshold = float(result["t_threshold"])
    clusters = result["clusters"]

    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

    for mean, sem, label_text, color in (
        (mean_a, sem_a, label_a, dash16.PALETTE[0]),
        (mean_b, sem_b, label_b, dash16.PALETTE[1]),
    ):
        rgba = dash16._hex_to_rgba(color, 0.18)
        figure.add_trace(go.Scatter(
            x=np.r_[times, times[::-1]], y=np.r_[mean - sem, (mean + sem)[::-1]],
            fill="toself", fillcolor=rgba, line=dict(width=0),
            hoverinfo="skip", showlegend=False,
        ), row=1, col=1)
        figure.add_trace(go.Scatter(
            x=times, y=mean, mode="lines", name=f"{label_text} mean (N={a.shape[0] if label_text == label_a else b.shape[0]})",
            line=dict(color=color, width=3),
            hovertemplate="t=%{x:.3f}s  y=%{y:.4f}<extra></extra>",
        ), row=1, col=1)

    figure.add_hline(y=0, line_color="#333333", line_dash="dot", line_width=1.5, row=1, col=1)
    figure.add_vline(x=0, line_color="#333333", line_dash="dash", line_width=1.5,
                      annotation_text="R-peak", annotation_position="top right", row=1, col=1)

    artifact_mask = (times >= dash16.HEP_ARTIFACT_EXCLUDE_S[0]) & (times <= dash16.HEP_ARTIFACT_EXCLUDE_S[1])
    artifact_mask_disp = dash16._dilate_bool_mask(artifact_mask)
    base_mask_disp = dash16._dilate_bool_mask(~artifact_mask)
    figure.add_trace(go.Scatter(
        x=times, y=np.where(base_mask_disp, t_stat, np.nan), mode="lines",
        name="Observed T statistic", line=dict(color="#6F42C1", width=2),
        hovertemplate="t=%{x:.3f}s  T=%{y:.3f}<extra></extra>", connectgaps=False,
    ), row=2, col=1)
    figure.add_trace(go.Scatter(
        x=times, y=np.where(artifact_mask_disp, t_stat, np.nan), mode="lines",
        name="QRS artifact window (excluded)", line=dict(color="#999999", width=2),
        hovertemplate="t=%{x:.3f}s  T=%{y:.3f}<extra></extra>", connectgaps=False,
    ), row=2, col=1)
    superthreshold_mask_disp = dash16._dilate_bool_mask((np.abs(t_stat) > t_threshold) & ~artifact_mask)
    figure.add_trace(go.Scatter(
        x=times, y=np.where(superthreshold_mask_disp, t_stat, np.nan), mode="lines",
        name="T exceeds cluster threshold", line=dict(color="#C0392B", width=3.5),
        connectgaps=False,
    ), row=2, col=1)
    figure.add_hline(y=t_threshold, line_color="#C0392B", line_dash="dash", line_width=1.5,
                      annotation_text=f"+T threshold = {t_threshold:.2f}",
                      annotation_position="top right", row=2, col=1)
    figure.add_hline(y=-t_threshold, line_color="#C0392B", line_dash="dash", line_width=1.5,
                      annotation_text=f"-T threshold = {-t_threshold:.2f}",
                      annotation_position="bottom right", row=2, col=1)
    for cluster in (c for c in clusters if c["significant"]):
        figure.add_vrect(x0=cluster["start"], x1=cluster["end"], fillcolor="#F0A202",
                          opacity=0.22, line_width=0, row=2, col=1)
    figure.add_vline(x=0, line_color="#333333", line_dash="dash", line_width=1.5,
                      annotation_text="R-peak", annotation_position="top right", row=2, col=1)
    figure.add_hline(y=0, line_color="#555555", opacity=0.4, row=2, col=1)

    figure.update_layout(
        title=dict(
            text=f"{label_a} (N={n_a}) vs {label_b} (N={n_b}): {stage_label}",
            font=dict(size=15, color="#111111"),
        ),
        height=760, hovermode="x unified",
        legend=dict(title="Trace", bgcolor="rgba(255,255,255,0.92)", bordercolor="#aaaaaa",
                    borderwidth=1, font=dict(color="#111111", size=12),
                    orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        showlegend=True, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="#111111"), margin=dict(l=70, r=170, t=90, b=60),
    )
    figure.update_xaxes(
        title=dict(text="Time from R-peak (s)", font=dict(color="#111111")),
        showgrid=True, gridcolor="#cccccc", gridwidth=1,
        zeroline=True, zerolinecolor="#888888", zerolinewidth=1,
        linecolor="#333333", linewidth=1.5, mirror=True,
        tickfont=dict(color="#111111"), row=2, col=1,
    )
    figure.update_yaxes(
        title=dict(text="HEP amplitude (µV)", font=dict(color="#111111")),
        showgrid=True, gridcolor="#cccccc", gridwidth=1,
        linecolor="#333333", linewidth=1.5, mirror=True,
        tickfont=dict(color="#111111"), row=1, col=1,
    )
    figure.update_yaxes(
        title=dict(text="T statistic", font=dict(color="#111111")),
        showgrid=True, gridcolor="#cccccc", gridwidth=1,
        linecolor="#333333", linewidth=1.5, mirror=True,
        tickfont=dict(color="#111111"), row=2, col=1,
    )
    figure.update_annotations(font=dict(color="#111111"))
    return figure


def main():
    mock_st = _make_mock_streamlit()
    sys.modules["streamlit"] = mock_st
    for submod in ["streamlit.components", "streamlit.components.v1"]:
        sys.modules.setdefault(submod, types.ModuleType(submod))
    dash16 = load_dash16_module()

    with open(os.path.join(OUT_DIR, "stage_delta_age_results.pkl"), "rb") as f:
        R = pickle.load(f)

    # We need the raw `grouped` df + `selected_channels` again to rebuild
    # prepare_waveform_contrast/prepare_electrode_contrasts results with matrices
    # (the pickle only stored summary numbers, not the matrices). Reload cohort.
    import pandas as pd
    hep_path = os.path.join(REPO, "6_hep_group_comparison.py")
    spec = importlib.util.spec_from_file_location("hep_module", hep_path)
    hep_mod = importlib.util.module_from_spec(spec)
    sys.modules["hep_module"] = hep_mod
    spec.loader.exec_module(hep_mod)

    raw_df = dash16.load_patient_data(
        hep_mod, ["Harvard_Electroencephalography"], STAGES,
        force_rebuild=False, apply_ica=False, min_eeg_channels=10,
    )
    selected_channels = R["selected_channels"]
    ehr_df = dash16.load_ehr_data(tuple(raw_df["patient_id"].unique()))
    healthy = dash16.select_non_diagnosis_cohort(raw_df, ehr_df)
    sick = raw_df[~raw_df["patient_id"].isin(healthy["patient_id"])].copy()
    sick["diagnosis_group"] = "Diagnosed (any)"
    combined = pd.concat([healthy, sick], ignore_index=True)

    clinical_df = dash16.load_cobrad_clinical()
    demo = dash16._patient_demographics(combined["patient_id"].unique(), clinical_df)
    valid_ages = demo.set_index("patient_id")["age"].dropna()
    age_median = float(valid_ages.median())
    if abs(age_median - R["age_median"]) >= 1e-6:
        print(f"[WARN] age median drifted between runs: {R['age_median']:.4f} -> {age_median:.4f} "
              f"(ongoing concurrent cache rebuilds on this machine; proceeding with the fresh value)")
    age_group_map = {pid: ("Older" if age >= age_median else "Younger") for pid, age in valid_ages.items()}
    grouped = combined.copy()
    grouped["diagnosis_group"] = grouped["patient_id"].astype(str).map(age_group_map)
    grouped = grouped[grouped["diagnosis_group"].isin(["Younger", "Older"])].copy()

    for stage in STAGES:
        result = dash16.prepare_waveform_contrast(
            grouped, "Younger", stage, "Older", stage, selected_channels, False, N_PERM, paired=False,
        )
        p_now = f"{result['contrast_p']:.4f}"
        p_before = next(r["p_formatted"] for r in R["age_split_results"] if r["stage"] == stage)
        if p_now != p_before:
            print(f"[WARN] {stage} age-split p-value drifted: {p_before} -> {p_now} "
                  f"(ongoing concurrent cache rebuilds; proceeding with the fresh value)")

        waveform_figure = plot_two_group_summary_and_tstat(
            dash16, result, "Younger", "Older", dash16.STAGE_LABELS[stage],
        )
        channel_results = dash16.prepare_electrode_contrasts(
            grouped, "Younger", stage, "Older", stage, selected_channels, False, N_PERM, paired=False,
        )
        waveform_figure = dash16.add_topomap_inset_to_waveform(
            waveform_figure, channel_results, "Younger", "Older", dash16.STAGE_LABELS[stage],
        )
        fig_path = os.path.join(FIG_DIR, f"agesplit_{stage}.png")
        waveform_figure.write_image(fig_path, scale=3)
        print(f"wrote {fig_path} ({os.path.getsize(fig_path)} bytes)")


if __name__ == "__main__":
    main()
