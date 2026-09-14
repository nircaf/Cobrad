"""
Cardiac field artifact (CFA) dashboard.

Shows the direct (model-free) CFA variance metric from
cfa_variance_explained.py: R^2 of each EEG channel's R-peak-locked HEP
evoked average regressed on the ECG's own evoked average (same epoch window
as this repo's HEP analysis, see HEP_TMIN/HEP_TMAX in
ica_ecg_component_variance.py) -- i.e. is the averaged heartbeat-evoked EEG
deflection just a copy of the ECG waveform, without depending on ICA being
correct. Reads straight from cfa_variance_explained_cache/{results,errors}/
*.parquet so it works while the batch run is still in progress.

Also fits ICA live for one chosen patient/recording (not from any cache --
just a few seconds of work) to plot a concrete before/after example: the
R-peak-locked HEP evoked average for one EEG channel, pre- and post-ICA
cleaning, next to the evoked ECG (QRS) waveform.

Run:
  streamlit run Paper1/ica_ecg_component_variance_dashboard.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyarrow.dataset as ds
import streamlit as st
from plotly.subplots import make_subplots

from ica_ecg_component_variance import HEP_TMIN, QRS_EXCLUDE_SEC, channel_names, epoch_around_r_peaks, quality

HERE = Path(__file__).resolve().parent
DEFAULT_CFA_CACHE_DIR = HERE / "cfa_variance_explained_cache"
DEFAULT_ICA_CACHE_DIR = HERE / "ica_ecg_component_variance_cache"

CFA_COLUMNS = [
    "patient_id", "recording_id", "edf_path", "eeg_channel", "cfa_r2_full_epoch", "cfa_r2_excl_qrs",
    "n_hep_epochs", "ecg_channel", "window_start_s", "window_duration_s",
]
# channel_hep_variance_pct_drop is per (recording, eeg_channel), duplicated
# across every component row in the raw ICA cache -- dedupe after loading.
ICA_DROP_COLUMNS = [
    "patient_id", "recording_id", "eeg_channel",
    "channel_hep_variance_pre_ica", "channel_hep_variance_post_ica", "channel_hep_variance_pct_drop",
]


@st.cache_data(ttl=120, show_spinner=False)
def load_cfa_results(cache_dir: str) -> pd.DataFrame:
    paths = sorted((Path(cache_dir) / "results").glob("*.parquet"))
    if not paths:
        return pd.DataFrame(columns=CFA_COLUMNS)
    progress = st.progress(0, text=f"Loading cached CFA regression results... 0% (0/{len(paths)})")
    frames = []
    for index, path in enumerate(paths, start=1):
        table = ds.dataset(str(path), format="parquet").to_table(columns=CFA_COLUMNS)
        frames.append(table.to_pandas())
        percent = round(100 * index / len(paths))
        progress.progress(
            index / len(paths),
            text=f"Loading cached CFA regression results... {percent}% ({index}/{len(paths)})",
        )
    progress.empty()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(ttl=120, show_spinner="Loading cached ICA cleaning-effect results...")
def load_ica_drop_results(cache_dir: str) -> pd.DataFrame:
    paths = sorted((Path(cache_dir) / "results").glob("*.parquet"))
    if not paths:
        return pd.DataFrame(columns=ICA_DROP_COLUMNS)
    table = ds.dataset([str(p) for p in paths], format="parquet").to_table(columns=ICA_DROP_COLUMNS)
    return table.to_pandas().drop_duplicates(["recording_id", "eeg_channel"]).reset_index(drop=True)


@st.cache_data(ttl=120)
def count_errors(cache_dir: str) -> int:
    return len(list((Path(cache_dir) / "errors").glob("*.parquet")))


@st.cache_data(ttl=600, show_spinner="Fitting ICA and epoching around R-peaks...")
def fit_and_clean(edf_path: str, window_start_s: float, window_duration_s: float):
    """Reproduce one recording's ICA cleaning live, then epoch pre/post EEG and ECG
    around R-peaks and average -- the HEP evoked waveform, not raw continuous traces."""
    import mne

    raw = mne.io.read_raw_edf(edf_path, preload=False, encoding="latin1", verbose="ERROR")
    eeg, ecg = channel_names(raw)
    raw.set_channel_types({ch: "eeg" for ch in eeg}, on_unit_change="ignore", verbose="ERROR")
    raw.set_channel_types({ecg: "ecg"}, on_unit_change="ignore", verbose="ERROR")
    segment = raw.copy().pick(eeg + [ecg])
    segment.crop(window_start_s, window_start_s + window_duration_s, include_tmax=False)
    segment.load_data(verbose="ERROR")
    _, qc = quality(segment, eeg, ecg)

    sfreq = float(segment.info["sfreq"])
    upper = min(100.0, sfreq / 2 - .5)
    segment.filter(1.0, upper, picks=eeg, verbose="ERROR")
    eeg_pre = segment.get_data(picks=eeg) * 1e6
    ecg_signal = segment.get_data(picks=[ecg])[0] * 1e6

    count = min(15, len(eeg) - 1)
    try:
        import picard  # noqa: F401
        method, params = "picard", {"ortho": False, "extended": True}
    except ImportError:
        method, params = "infomax", {"extended": True}
    ica = mne.preprocessing.ICA(n_components=count, method=method, fit_params=params, random_state=42, max_iter=500)
    ica.fit(segment, picks=eeg, verbose="ERROR")
    bad, scores = ica.find_bads_ecg(segment, ch_name=ecg, method="correlation", verbose="ERROR")
    scores = np.asarray(scores, float)
    if not bad and scores.size:
        bad = [int(np.nanargmax(np.abs(scores)))]

    cleaned = segment.copy()
    ica.apply(cleaned, exclude=bad, verbose="ERROR")
    eeg_post = cleaned.get_data(picks=eeg) * 1e6

    pre_epochs = epoch_around_r_peaks(eeg_pre, qc["r_peak_samples"], sfreq)
    post_epochs = epoch_around_r_peaks(eeg_post, qc["r_peak_samples"], sfreq)
    ecg_epochs = epoch_around_r_peaks(ecg_signal, qc["r_peak_samples"], sfreq)
    n_hep_epochs = pre_epochs.shape[0]

    # ECG -> EEG prediction with a recording-level, heartbeat-epoch holdout.
    # Fit one affine mapping per EEG channel on individual samples from 80% of
    # the heartbeats, then evaluate it on completely unseen heartbeats.  The
    # plotted traces below are held-out evoked averages, not training data.
    rng = np.random.default_rng(42)
    order = rng.permutation(n_hep_epochs)
    n_test = max(1, int(round(0.2 * n_hep_epochs)))
    test_idx, train_idx = order[:n_test], order[n_test:]
    if train_idx.size == 0:
        raise ValueError("at least two R-peak epochs are required for ECG-to-EEG prediction")
    x_train = ecg_epochs[train_idx].reshape(-1)
    x_mean = float(x_train.mean())
    x_centered = x_train - x_mean
    x_ss = float(x_centered @ x_centered)
    if x_ss <= np.finfo(float).eps:
        raise ValueError("ECG signal has no usable variance for EEG prediction")
    y_train = pre_epochs[train_idx].transpose(1, 0, 2).reshape(len(eeg), -1)
    y_mean = y_train.mean(axis=1)
    slopes = ((y_train - y_mean[:, None]) @ x_centered) / x_ss
    intercepts = y_mean - slopes * x_mean
    predicted_test = intercepts[None, :, None] + slopes[None, :, None] * ecg_epochs[test_idx, None, :]
    actual_test = pre_epochs[test_idx]
    residual_ss = np.sum((actual_test - predicted_test) ** 2, axis=(0, 2))
    baseline_ss = np.sum((actual_test - y_mean[None, :, None]) ** 2, axis=(0, 2))
    prediction_r2 = np.where(baseline_ss > 0, 1.0 - residual_ss / baseline_ss, np.nan)

    # Compact heartbeat-level inputs for the nonlinear feature models.  These
    # summarize ECG morphology without passing an entire resampled trace to a
    # model (which would be high-dimensional relative to the heartbeat count).
    qrs = np.abs(epoch_times := (np.arange(ecg_epochs.shape[1]) - int(round(-HEP_TMIN * sfreq))) / sfreq) <= QRS_EXCLUDE_SEC
    post_qrs = (epoch_times > QRS_EXCLUDE_SEC) & (epoch_times <= 0.30)
    ecg_diff = np.diff(ecg_epochs, axis=1)
    ecg_features = np.column_stack([
        ecg_epochs.mean(axis=1), ecg_epochs.std(axis=1),
        np.sqrt(np.mean(ecg_epochs ** 2, axis=1)), np.ptp(ecg_epochs, axis=1),
        np.max(np.abs(ecg_epochs), axis=1), np.mean(np.abs(ecg_epochs), axis=1),
        np.sqrt(np.mean(ecg_epochs[:, qrs] ** 2, axis=1)),
        np.sqrt(np.mean(ecg_epochs[:, post_qrs] ** 2, axis=1)),
        np.max(np.abs(ecg_diff), axis=1),
    ])
    eeg_feature_targets = {
        "Post-QRS mean amplitude": pre_epochs[:, :, post_qrs].mean(axis=2),
        "Post-QRS RMS": np.sqrt(np.mean(pre_epochs[:, :, post_qrs] ** 2, axis=2)),
        "Full-epoch peak-to-peak": np.ptp(pre_epochs, axis=2),
        "Full-epoch RMS": np.sqrt(np.mean(pre_epochs ** 2, axis=2)),
    }

    pre_samples = int(round(-HEP_TMIN * sfreq))
    epoch_times = (np.arange(ecg_epochs.shape[1]) - pre_samples) / sfreq
    sem_scale = max(n_hep_epochs, 1) ** 0.5

    return {
        "eeg_channels": eeg, "epoch_times": epoch_times, "n_hep_epochs": n_hep_epochs,
        "pre_mean": pre_epochs.mean(axis=0), "pre_sem": pre_epochs.std(axis=0) / sem_scale,
        "post_mean": post_epochs.mean(axis=0), "post_sem": post_epochs.std(axis=0) / sem_scale,
        "ecg_mean": ecg_epochs.mean(axis=0), "bad_components": bad,
        "prediction_actual_mean": actual_test.mean(axis=0),
        "prediction_predicted_mean": predicted_test.mean(axis=0),
        "prediction_r2": prediction_r2, "prediction_n_train": len(train_idx),
        "prediction_n_test": len(test_idx),
        "ecg_features": ecg_features, "eeg_feature_targets": eeg_feature_targets,
        "ecg_epochs": ecg_epochs,
        "prediction_train_idx": train_idx, "prediction_test_idx": test_idx,
    }


def _add_evoked_trace(fig, x, mean, sem, color_rgb, name, secondary_y=False):
    r, g, b = color_rgb
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]), y=np.concatenate([mean + sem, (mean - sem)[::-1]]),
        fill="toself", fillcolor=f"rgba({r},{g},{b},0.15)", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ), secondary_y=secondary_y)


def make_variance_drop_topomap(frame: pd.DataFrame):
    """Map channels present in >50% of patients, weighting each patient once."""
    import re
    import matplotlib.pyplot as plt
    import mne
    from matplotlib.ticker import FuncFormatter

    montage = mne.channels.make_standard_montage("standard_1020")
    canonical = {name.upper(): name for name in montage.ch_names}
    positions = montage.get_positions()["ch_pos"]
    n_patients = frame["patient_id"].nunique()
    patient_channel = (
        frame.groupby(["patient_id", "eeg_channel"], as_index=False)["channel_hep_variance_pct_drop"]
        .mean()
    )
    coverage = patient_channel.groupby("eeg_channel")["patient_id"].nunique() / max(n_patients, 1)
    eligible = set(coverage[coverage > 0.50].index)
    low_coverage = sorted(set(patient_channel["eeg_channel"]) - eligible)
    channel_summary = (
        patient_channel.groupby("eeg_channel")["channel_hep_variance_pct_drop"]
        .agg(patient_count="count", mean_drop="mean", std_drop="std")
        .reset_index()
        .rename(columns={"eeg_channel": "Electrode"})
    )
    channel_summary["patient_coverage"] = channel_summary["patient_count"] / max(n_patients, 1)
    channel_summary["shown_on_topomap"] = channel_summary["Electrode"].map(
        lambda channel: (
            channel in eligible
            and "-" not in re.sub(r"^EEG\s+", "", str(channel), flags=re.I).strip()
            and re.sub(r"^EEG\s+", "", str(channel), flags=re.I).strip().upper() in canonical
        )
    )
    channel_summary = channel_summary.sort_values(
        ["shown_on_topomap", "patient_coverage", "Electrode"], ascending=[False, False, True],
    ).reset_index(drop=True)

    mapped: dict[str, list[float]] = {}
    unmapped = set()
    for channel, value in patient_channel[["eeg_channel", "channel_hep_variance_pct_drop"]].itertuples(index=False):
        if channel not in eligible:
            continue
        label = re.sub(r"^EEG\s+", "", str(channel), flags=re.I).strip()
        # Bipolar derivations do not have a single electrode coordinate.
        name = canonical.get(label.upper()) if "-" not in label else None
        if name is None:
            unmapped.add(str(channel))
        elif np.isfinite(value):
            mapped.setdefault(name, []).append(float(value))
    if len(mapped) < 3:
        return None, sorted(unmapped), low_coverage, len(mapped), n_patients, channel_summary

    names = sorted(mapped)
    values = np.asarray([np.mean(mapped[name]) for name in names])
    pos = np.asarray([positions[name][:2] for name in names])
    limit = max(float(np.nanmax(np.abs(values))), 0.01)
    fig, ax = plt.subplots(figsize=(7, 5.8))
    image, _ = mne.viz.plot_topomap(
        values, pos, axes=ax, show=False, cmap="RdBu_r", vlim=(-limit, limit),
        contours=6, sensors=True, names=names, size=4,
    )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.78, pad=0.08)
    colorbar.set_label("Mean HEP variance drop after ICA")
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.set_title(f"Mean pre- vs post-ICA HEP variance drop ({len(names)} channels)")
    fig.tight_layout()
    return fig, sorted(unmapped), low_coverage, len(mapped), n_patients, channel_summary
    fig.add_trace(go.Scatter(
        x=x, y=mean, name=name, line=dict(color=f"rgb({r},{g},{b})"),
    ), secondary_y=secondary_y)


@st.cache_data(persist="disk", max_entries=128, show_spinner="Training ECG-to-EEG feature models...")
def compare_feature_models(ecg_features, target, train_idx, test_idx):
    """Fit diverse regressors and return honest held-out predictions/scores."""
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models = {
        "Ridge": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            RidgeCV(alphas=np.logspace(-3, 3, 13)),
        ),
        "Random forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=80, min_samples_leaf=4, max_features="sqrt", n_jobs=2, random_state=42),
        ),
        "Extra Trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(n_estimators=80, min_samples_leaf=4, max_features=1.0, n_jobs=2, random_state=42),
        ),
        "Gradient boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            GradientBoostingRegressor(n_estimators=80, learning_rate=0.05, max_depth=2, loss="huber", random_state=42),
        ),
    }
    x_train, x_test = ecg_features[train_idx], ecg_features[test_idx]
    y_train, y_test = target[train_idx], target[test_idx]
    def fit_one(name, model):
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        correlation = np.corrcoef(y_test, predicted)[0, 1] if np.std(predicted) > 0 and np.std(y_test) > 0 else np.nan
        row = {
            "Algorithm": name, "Held-out R²": r2_score(y_test, predicted),
            "MAE (µV)": mean_absolute_error(y_test, predicted),
            "Correlation": correlation,
        }
        return row, name, predicted

    # Model families are independent, so fit them concurrently. Threading is
    # preferable here because the input arrays stay shared rather than being
    # copied into four worker processes.
    from joblib import Parallel, delayed
    fitted = Parallel(n_jobs=min(4, len(models)), prefer="threads")(
        delayed(fit_one)(name, model) for name, model in models.items()
    )
    rows = [item[0] for item in fitted]
    predictions = {item[1]: item[2] for item in fitted}
    return pd.DataFrame(rows).sort_values("Held-out R²", ascending=False), predictions, y_test


@st.cache_data(persist="disk", max_entries=64, show_spinner="Training CNN and LSTM on raw ECG epochs...")
def compare_deep_models(ecg_epochs, target, train_idx, test_idx, epochs=12):
    """Train compact raw-waveform networks without touching held-out heartbeats."""
    import torch
    from torch import nn

    torch.manual_seed(42)
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
    train_idx = np.asarray(train_idx)
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(train_idx)
    n_val = max(2, int(round(0.15 * len(shuffled))))
    val_idx, fit_idx = shuffled[:n_val], shuffled[n_val:]

    # At most 96 time points retains the ECG morphology while making recurrent
    # training several times faster on CPU.
    sample_idx = np.linspace(0, ecg_epochs.shape[1] - 1, min(96, ecg_epochs.shape[1])).round().astype(int)
    ecg_epochs = ecg_epochs[:, sample_idx]
    x_mean, x_std = ecg_epochs[fit_idx].mean(), ecg_epochs[fit_idx].std() + 1e-8
    y_mean, y_std = target[fit_idx].mean(), target[fit_idx].std() + 1e-8
    x = torch.tensor((ecg_epochs - x_mean) / x_std, dtype=torch.float32)
    y = torch.tensor((target - y_mean) / y_std, dtype=torch.float32)

    class CNN1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 8, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(8, 16, 5, padding=2), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.15), nn.Linear(16, 1),
            )
        def forward(self, values):
            return self.net(values[:, None, :]).squeeze(1)

    class ECG_LSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(1, 12, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(0.15), nn.Linear(12, 1))
        def forward(self, values):
            sequence, _ = self.rnn(values[:, :, None])
            return self.head(sequence.mean(dim=1)).squeeze(1)

    def train(model, seed):
        model_rng = np.random.default_rng(seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        loss_fn = nn.HuberLoss()
        best_state, best_loss, patience = None, float("inf"), 0
        for _ in range(int(epochs)):
            model.train()
            for batch in np.array_split(model_rng.permutation(fit_idx), max(1, int(np.ceil(len(fit_idx) / 128)))):
                optimizer.zero_grad()
                loss = loss_fn(model(x[batch]), y[batch])
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(x[val_idx]), y[val_idx]))
            if val_loss < best_loss - 1e-5:
                best_loss, patience = val_loss, 0
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 3:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            return model(x[test_idx]).numpy() * y_std + y_mean

    actual = target[test_idx]
    def fit_one(name, model, seed):
        predicted = train(model, seed)
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        row = {
            "Algorithm": name, "Held-out R²": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            "MAE (µV)": np.mean(np.abs(actual - predicted)),
            "Correlation": np.corrcoef(actual, predicted)[0, 1] if np.std(predicted) > 0 else np.nan,
        }
        return row, name, predicted

    # CNN and LSTM are independent and train simultaneously. Limit each to two
    # Torch threads so the pair does not monopolize the dashboard host.
    from concurrent.futures import ThreadPoolExecutor
    model_specs = [("1D CNN", CNN1D(), 42), ("LSTM", ECG_LSTM(), 43)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        fitted = list(pool.map(lambda spec: fit_one(*spec), model_specs))
    rows = [item[0] for item in fitted]
    predictions = {item[1]: item[2] for item in fitted}
    return pd.DataFrame(rows).sort_values("Held-out R²", ascending=False), predictions, actual


def main() -> None:
    st.set_page_config(page_title="Cardiac Field Artifact", layout="wide")
    st.title("🔗 Cardiac field artifact — direct variance explained")
    st.caption(
        "R² of each EEG channel's R-peak-locked HEP evoked average regressed on the ECG's own "
        "evoked average -- the fraction of the averaged heartbeat-evoked EEG deflection that's "
        "just a copy of the ECG waveform, independent of whether ICA correctly isolated it."
    )

    cache_dir = st.sidebar.text_input("Cache dir", str(DEFAULT_CFA_CACHE_DIR))
    if st.sidebar.button("Refresh now"):
        st.cache_data.clear()

    cfa_frame = load_cfa_results(cache_dir)
    n_errors = count_errors(cache_dir)

    col1, col2, col3 = st.columns(3)
    col1.metric("Recordings processed", f"{cfa_frame['recording_id'].nunique() if not cfa_frame.empty else 0:,}")
    col2.metric("Recordings errored", f"{n_errors:,}")
    col3.metric("Channel rows", f"{len(cfa_frame):,}")

    if cfa_frame.empty:
        st.info("No cached CFA regression results yet -- run cfa_variance_explained.py first.")
        return

    metric_choice = st.radio(
        "CFA metric", ["Full epoch (-0.3s to 0.4s)", "Excluding QRS window (|t| > 50ms)"], horizontal=True,
    )
    metric_col = "cfa_r2_full_epoch" if metric_choice.startswith("Full") else "cfa_r2_excl_qrs"

    st.subheader("Explained variance per channel")
    fig_cfa = px.histogram(
        cfa_frame, x=metric_col, nbins=60,
        labels={metric_col: "% of evoked HEP variance explained by ECG (direct R²)"},
    )
    fig_cfa.update_xaxes(tickformat=".1%")
    st.plotly_chart(fig_cfa, use_container_width=True)

    per_recording = cfa_frame.groupby(["patient_id", "recording_id"], as_index=False)[metric_col].mean()
    per_recording = per_recording.rename(columns={metric_col: "mean_cfa_r2"})
    st.metric("Median mean-channel CFA R² per recording", f"{per_recording['mean_cfa_r2'].median():.2%}")
    fig_per_rec = px.histogram(per_recording, x="mean_cfa_r2", nbins=50)
    fig_per_rec.update_xaxes(tickformat=".1%", title="Mean per-channel CFA R², per recording")
    st.plotly_chart(fig_per_rec, use_container_width=True)

    st.subheader("Per-channel detail")
    st.dataframe(cfa_frame.sort_values(metric_col, ascending=False), use_container_width=True)

    st.divider()
    st.header("🧪 Example patient: HEP evoked average, pre- vs post-ICA cleaning")
    st.caption(
        "Fits ICA live for one chosen recording (not from any cache), epochs the EEG channel "
        "and ECG around every R-peak, and plots the mean ± SEM evoked waveform before and after "
        "removing the ICA component ICA flagged as ECG-related. Takes a few seconds."
    )

    recordings = cfa_frame[["patient_id", "recording_id", "edf_path", "window_start_s", "window_duration_s"]].drop_duplicates("recording_id")
    # Default to the recording with the strongest measured CFA, so the example is visibly informative.
    default_recording = per_recording.sort_values("mean_cfa_r2", ascending=False)["recording_id"].iloc[0]
    recording_choice = st.selectbox(
        "Recording", recordings["recording_id"].sort_values(),
        index=int(np.flatnonzero(recordings["recording_id"].sort_values().to_numpy() == default_recording)[0]),
    )
    meta = recordings[recordings["recording_id"] == recording_choice].iloc[0]
    channel_options = sorted(cfa_frame.loc[cfa_frame["recording_id"] == recording_choice, "eeg_channel"].unique())
    channel_choice = st.selectbox("EEG channel", channel_options)

    result = fit_and_clean(meta["edf_path"], float(meta["window_start_s"]), float(meta["window_duration_s"]))
    channel_index = result["eeg_channels"].index(channel_choice)
    times = result["epoch_times"]

    fig_clean = make_subplots(specs=[[{"secondary_y": True}]])
    fig_clean.add_vrect(
        x0=-QRS_EXCLUDE_SEC, x1=QRS_EXCLUDE_SEC, fillcolor="gray", opacity=0.15, line_width=0,
        annotation_text="QRS window", annotation_position="top left",
    )
    _add_evoked_trace(fig_clean, times, result["pre_mean"][channel_index], result["pre_sem"][channel_index], (214, 39, 40), f"{channel_choice} (pre-ICA)")
    _add_evoked_trace(fig_clean, times, result["post_mean"][channel_index], result["post_sem"][channel_index], (31, 119, 180), f"{channel_choice} (post-ICA)")
    fig_clean.add_trace(go.Scatter(
        x=times, y=result["ecg_mean"], name="ECG (evoked)", line=dict(color="#2ca02c", dash="dot"),
    ), secondary_y=True)
    fig_clean.update_xaxes(title="Time relative to R-peak (s)")
    fig_clean.update_yaxes(title="EEG (µV)", secondary_y=False)
    fig_clean.update_yaxes(title="ECG (µV)", secondary_y=True)
    fig_clean.update_layout(title=f"{meta['patient_id']} / {recording_choice}", height=500)
    st.plotly_chart(fig_clean, use_container_width=True)

    var_pre = float(np.var(result["pre_mean"][channel_index]))
    var_post = float(np.var(result["post_mean"][channel_index]))
    pct_drop = (var_pre - var_post) / var_pre if var_pre > 0 else float("nan")
    st.metric(f"HEP evoked variance drop on {channel_choice}", f"{pct_drop:.1%}")
    st.caption(f"ICA excluded component index(es): {result['bad_components']} -- averaged over {result['n_hep_epochs']:,} R-peak epochs")

    st.subheader("ECG → EEG prediction on held-out heartbeats")
    st.caption(
        "A linear model (EEG = intercept + slope × ECG) is fitted on 80% of this recording's "
        "R-peak epochs and predicts the selected EEG channel on the other 20%. The score is "
        "out-of-sample R² across all samples in the held-out epochs; values below zero mean that "
        "the ECG model predicts worse than the training-set mean EEG."
    )
    fig_prediction = go.Figure()
    fig_prediction.add_vrect(
        x0=-QRS_EXCLUDE_SEC, x1=QRS_EXCLUDE_SEC, fillcolor="gray", opacity=0.15,
        line_width=0, annotation_text="QRS window", annotation_position="top left",
    )
    fig_prediction.add_trace(go.Scatter(
        x=times, y=result["prediction_actual_mean"][channel_index],
        name=f"Measured {channel_choice} (held out)", line=dict(color="#1f77b4"),
    ))
    fig_prediction.add_trace(go.Scatter(
        x=times, y=result["prediction_predicted_mean"][channel_index],
        name="Predicted from ECG", line=dict(color="#ff7f0e", dash="dash"),
    ))
    fig_prediction.update_xaxes(title="Time relative to R-peak (s)")
    fig_prediction.update_yaxes(title="EEG (µV)")
    fig_prediction.update_layout(height=430, title=f"Held-out ECG prediction of {channel_choice}")
    st.plotly_chart(fig_prediction, use_container_width=True)
    pred_col1, pred_col2, pred_col3 = st.columns(3)
    pred_col1.metric("Held-out prediction R²", f"{result['prediction_r2'][channel_index]:.2%}")
    pred_col2.metric("Training heartbeats", f"{result['prediction_n_train']:,}")
    pred_col3.metric("Test heartbeats", f"{result['prediction_n_test']:,}")

    st.subheader("Advanced models: predict a specific EEG feature")
    st.caption(
        "Each row is one heartbeat. Nine ECG morphology features (including QRS energy, "
        "post-QRS energy, amplitude, variability, and maximum slope) predict one selected EEG "
        "feature. All algorithms use exactly the same held-out heartbeats."
    )
    feature_choice = st.selectbox(
        "EEG feature to predict", list(result["eeg_feature_targets"]),
        index=1, key="eeg_prediction_feature",
    )
    feature_scores, feature_predictions, feature_actual = compare_feature_models(
        result["ecg_features"],
        result["eeg_feature_targets"][feature_choice][:, channel_index],
        result["prediction_train_idx"], result["prediction_test_idx"],
    )
    display_scores = feature_scores.copy()
    display_scores["Held-out R²"] = display_scores["Held-out R²"].map(lambda value: f"{value:.2%}")
    display_scores["Correlation"] = display_scores["Correlation"].map(lambda value: f"{value:.3f}")
    display_scores["MAE (µV)"] = display_scores["MAE (µV)"].map(lambda value: f"{value:.3f}")
    st.dataframe(display_scores, use_container_width=True, hide_index=True)

    best_model = feature_scores.iloc[0]["Algorithm"]
    feature_plot = pd.DataFrame({
        "Held-out heartbeat": np.arange(1, len(feature_actual) + 1),
        "Measured": feature_actual,
        f"Predicted ({best_model})": feature_predictions[best_model],
    }).melt("Held-out heartbeat", var_name="Series", value_name="EEG feature (µV)")
    fig_feature = px.line(
        feature_plot, x="Held-out heartbeat", y="EEG feature (µV)", color="Series",
        markers=True, title=f"{feature_choice} on {channel_choice}: best held-out model is {best_model}",
    )
    st.plotly_chart(fig_feature, use_container_width=True)
    st.caption(
        "Negative held-out R² is a valid result: it means ECG alone does not generalize well "
        "enough to predict that EEG feature for this recording."
    )

    st.markdown("#### Raw-waveform neural networks")
    st.caption(
        "The compact 1D CNN learns local ECG morphology; the LSTM learns sequential context. "
        "ECG epochs are resampled to at most 96 points and training uses up to 12 epochs with early stopping. "
        "They use raw ECG epochs and the identical untouched test heartbeats used above. The two "
        "networks train in parallel, and completed results are cached on disk across restarts."
    )
    if st.button("Train fast CNN and LSTM", type="primary"):
        deep_scores, deep_predictions, deep_actual = compare_deep_models(
            result["ecg_epochs"],
            result["eeg_feature_targets"][feature_choice][:, channel_index],
            result["prediction_train_idx"], result["prediction_test_idx"],
        )
        combined_scores = pd.concat([feature_scores, deep_scores], ignore_index=True).sort_values(
            "Held-out R²", ascending=False,
        )
        formatted = combined_scores.copy()
        formatted["Held-out R²"] = formatted["Held-out R²"].map(lambda value: f"{value:.2%}")
        formatted["Correlation"] = formatted["Correlation"].map(lambda value: f"{value:.3f}")
        formatted["MAE (µV)"] = formatted["MAE (µV)"].map(lambda value: f"{value:.3f}")
        st.markdown("**Classical + neural leaderboard**")
        st.dataframe(formatted, use_container_width=True, hide_index=True)

    st.info(
        "Transfer learning is not estimated by this single-recording panel. A valid transfer result "
        "must pretrain on other patients, freeze or fine-tune the ECG encoder, and test on a patient "
        "excluded from all pretraining and model selection. Reusing this recording would inflate the score."
    )

    st.divider()
    st.header("📉 Fleet-wide: HEP variance drop, pre- vs post-ICA")
    st.caption(
        "From the ica_ecg_component_variance.py batch cache: for every processed recording/channel, "
        "the % drop in R-peak-locked HEP evoked variance after removing the ICA component(s) flagged "
        "as ECG-related, vs. before."
    )
    ica_cache_dir = st.sidebar.text_input("ICA cache dir", str(DEFAULT_ICA_CACHE_DIR))
    ica_drop_frame = load_ica_drop_results(ica_cache_dir)
    ica_errors = count_errors(ica_cache_dir)

    ica_col1, ica_col2, ica_col3 = st.columns(3)
    ica_col1.metric("Recordings processed", f"{ica_drop_frame['recording_id'].nunique() if not ica_drop_frame.empty else 0:,}")
    ica_col2.metric("Recordings errored", f"{ica_errors:,}")
    ica_col3.metric("Channel rows", f"{len(ica_drop_frame):,}")

    if ica_drop_frame.empty:
        st.info("No cached ICA results yet -- run ica_ecg_component_variance.py first.")
        return

    st.metric("Median HEP variance drop per channel", f"{ica_drop_frame['channel_hep_variance_pct_drop'].median():.1%}")
    fig_drop = px.histogram(ica_drop_frame, x="channel_hep_variance_pct_drop", nbins=60)
    fig_drop.update_xaxes(tickformat=".0%", title="% drop in HEP evoked variance after ICA cleaning")
    st.plotly_chart(fig_drop, use_container_width=True)

    st.subheader("Average HEP variance drop by scalp channel")
    st.caption(
        "Each electrode value is the patient-weighted mean pre- vs post-ICA HEP variance drop. "
        "Only electrodes available in more than half of patients are shown. Red indicates a variance "
        "reduction; blue indicates an increase after cleaning."
    )
    (
        topo_figure, unmapped_channels, low_coverage_channels, mapped_count,
        topo_patient_count, topo_channel_summary,
    ) = make_variance_drop_topomap(ica_drop_frame)
    if topo_figure is None:
        st.warning(f"Only {mapped_count} channels could be mapped to standard 10–20 coordinates; at least 3 are required.")
    else:
        st.pyplot(topo_figure, use_container_width=False)
    st.caption(
        f"Coverage filter: shown electrodes occur in >50% of {topo_patient_count:,} unique processed patients. "
        "Repeated recordings are averaged within patient before the across-patient mean."
    )
    if unmapped_channels:
        st.caption("Not shown (no single standard 10–20 position): " + ", ".join(unmapped_channels))
    if low_coverage_channels:
        st.caption("Not shown (≤50% patient coverage): " + ", ".join(low_coverage_channels))

    st.markdown("#### Electrode values used in the topomap")
    shown_summary = topo_channel_summary[topo_channel_summary["shown_on_topomap"]].copy()
    shown_summary = shown_summary.rename(columns={
        "patient_count": "Patients",
        "patient_coverage": "Patients (%)",
        "mean_drop": "Mean HEP variance drop (%)",
        "std_drop": "Patient STD (%)",
    })
    shown_summary["Patients (%)"] *= 100
    shown_summary["Mean HEP variance drop (%)"] *= 100
    shown_summary["Patient STD (%)"] *= 100
    st.dataframe(
        shown_summary[[
            "Electrode", "Patients", "Patients (%)",
            "Mean HEP variance drop (%)", "Patient STD (%)",
        ]],
        use_container_width=True, hide_index=True,
        column_config={
            "Patients (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Mean HEP variance drop (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Patient STD (%)": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )
    st.caption(
        "STD is calculated across patients after repeated recordings for the same patient/electrode "
        "are averaged. It measures between-patient variability, not uncertainty of the mean."
    )

    st.subheader("Per-channel detail")
    st.dataframe(
        ica_drop_frame.sort_values("channel_hep_variance_pct_drop", ascending=False),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
