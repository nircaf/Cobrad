"""Streamlit application for exploring single EEG/EDF recordings."""
from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional, Sequence, Tuple

import plotly.graph_objects as go
import mne
import numpy as np
import pandas as pd
import streamlit as st
from mne.time_frequency import psd_array_welch

try:  # Optional imports used in the cleaning pipeline
    from autoreject import AutoReject
except Exception:  # pragma: no cover - optional dependency
    AutoReject = None

try:  # Optional dependency for the PREP pipeline
    from pyprep.prep_pipeline import PrepPipeline
except Exception:  # pragma: no cover - optional dependency
    PrepPipeline = None

try:  # Optional dependency for spectrogram computation
    from scipy.signal import spectrogram
except Exception:  # pragma: no cover - optional dependency
    spectrogram = None

DEFAULT_FILE = Path("EDF_Format/Controls/Controls/61-70 data/FA0011KC_1-2.edf")
NUMERIC_KINDS = {"eeg", "seeg", "ecog", "dbs", "misc", "eog", "emg", "ecg", "meg"}
PSWE_EPS = 1e-12

mne.set_log_level("WARNING")
st.set_page_config(page_title="Single Scan EEG Overview", layout="wide")
# Custom CSS for better presentation
st.markdown("""
<style>
    .section-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stExpander > div > div {
        background-color: #f8f9fa;
    }
        .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-metric {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .warning-metric {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .error-metric {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

def read_recording(path: Path) -> mne.io.BaseRaw:
    """Read a recording from disk using MNE."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    reader = None
    if ext in {".edf", ".bdf", ".gdf", ".rec", ".eeg"}:
        reader = mne.io.read_raw_edf
    elif ext == ".fif":
        reader = mne.io.read_raw_fif
    elif ext == ".set":
        reader = mne.io.read_raw_eeglab

    if reader is None:
        raise ValueError(f"Unsupported file extension '{ext}'. Supported: EDF, BDF, GDF, REC, EEG, FIF, SET")

    raw = reader(path.as_posix(), preload=True, verbose="ERROR")
    return raw


def load_uploaded_file(upload) -> Tuple[mne.io.BaseRaw, str]:
    """Persist the uploaded file to a temporary location and load it with MNE."""
    suffix = Path(upload.name).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.getbuffer())
        tmp_path = Path(tmp.name)

    try:
        raw = read_recording(tmp_path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass

    identifier = f"upload:{upload.name}:{getattr(upload, 'size', tmp_path.stat().st_size)}"
    return raw, identifier


def load_default_file(path: Path) -> Tuple[mne.io.BaseRaw, str]:
    raw = read_recording(path)
    return raw, f"default:{path.as_posix()}"


def get_numeric_picks(raw: mne.io.BaseRaw) -> List[int]:
    """Return indices of numeric channels only."""
    picks: List[int] = []
    for idx, ch in enumerate(raw.info["chs"]):
        kind = ch.get("kind", None)
        try:
            ch_type = raw.get_channel_types(picks=[idx])[0]
        except Exception:
            ch_type = None
        if ch_type and ch_type.lower() in NUMERIC_KINDS:
            picks.append(idx)
            continue

        try:
            data = raw.get_data(picks=[idx], reject_by_annotation=False)
        except Exception:
            continue
        if data.size == 0:
            continue
        if np.issubdtype(data.dtype, np.floating):
            picks.append(idx)
    return picks


def count_spikes(signal: np.ndarray, multiplier: float = 5.0) -> int:
    """Simple spike detector using MAD-based thresholding."""
    if signal.size == 0:
        return 0
    baseline = np.nanmedian(signal)
    deviation = np.abs(signal - baseline)
    mad = np.nanmedian(deviation)
    if not np.isfinite(mad) or mad == 0:
        mad = np.nanstd(signal)
    if not np.isfinite(mad) or mad == 0:
        mad = np.nanmean(deviation) if np.nanmean(deviation) != 0 else 1.0
    threshold = multiplier * mad * 1.4826
    return int(np.sum(deviation > threshold))


def compute_channel_statistics(data: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    """Compute descriptive statistics for each channel."""
    rows = []
    for idx, name in enumerate(names):
        channel = data[idx]
        rows.append(
            {
                "Channel": name,
                "Minimum": float(np.nanmin(channel)),
                "Maximum": float(np.nanmax(channel)),
                "Mean": float(np.nanmean(channel)),
                "Std": float(np.nanstd(channel)),
                "Var": float(np.nanvar(channel)),
                "Spike Count": count_spikes(channel),
            }
        )
    df = pd.DataFrame(rows)
    return df.set_index("Channel")


def compute_pswe(psd_values: np.ndarray) -> float:
    """Compute the Power Spectral Wavelet Entropy (approximated via normalized PSD entropy)."""
    total_power = float(np.sum(psd_values))
    if total_power <= 0:
        return 0.0
    probabilities = psd_values / (total_power + PSWE_EPS)
    n_bins = len(probabilities)
    if n_bins <= 1:
        return 0.0
    entropy = -np.sum(probabilities * np.log(probabilities + PSWE_EPS))
    return float(entropy / np.log(n_bins))


def plot_raw_signals(times: np.ndarray, data: np.ndarray, names: Sequence[str]) -> go.Figure:
    fig = go.Figure()
    if data.size == 0:
        fig.update_layout(
            title="Raw channel preview",
            xaxis_title="Time (s)",
            yaxis=dict(showticklabels=False),
        )
        return fig
    scale = np.nanpercentile(np.abs(data), 95)
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    for idx, name in enumerate(names):
        offset = idx * scale * 2.0
        fig.add_trace(
            go.Scatter(
                x=times,
                y=(data[idx] + offset),
                mode="lines",
                name=name,
                line=dict(width=1),
                hovertemplate="t=%{x:.3f}s<br>value=%{y:.3f}<extra>" + name + "</extra>",
            )
        )
    fig.update_layout(
        title="Raw channel preview",
        xaxis_title="Time (s)",
        yaxis=dict(title="Amplitude + offset", showticklabels=False, zeroline=False, showgrid=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=40, b=40),
        template="plotly_white",
    )
    return fig


def plot_psd(freqs: np.ndarray, psds: np.ndarray, names: Sequence[str]) -> go.Figure:
    fig = go.Figure()
    if psds.size == 0:
        fig.update_layout(title="Power Spectral Density")
        return fig
    for idx, name in enumerate(names):
        fig.add_trace(
            go.Scatter(
                x=freqs,
                y=psds[idx],
                mode="lines",
                name=name,
                line=dict(width=1),
                opacity=0.3,
                hoverinfo="skip",
            )
        )
    mean_psd = np.nanmean(psds, axis=0)
    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=mean_psd,
            mode="lines",
            name="Average",
            line=dict(width=2, color="black"),
        )
    )
    fig.update_layout(
        title="Power Spectral Density",
        xaxis_title="Frequency (Hz)",
        yaxis=dict(title="PSD (V^2/Hz)", type="log", showgrid=True),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def plot_spectrogram(
    times: np.ndarray,
    signal: np.ndarray,
    sfreq: float,
    channel_name: str,
) -> Optional[go.Figure]:
    if spectrogram is None:
        return None
    if signal.size < 16 or not np.isfinite(signal).any():
        return None
    nperseg = int(min(len(signal), max(64, sfreq * 4)))
    if nperseg <= 0:
        return None
    noverlap = nperseg // 2
    freqs, time_points, spec = spectrogram(signal, fs=sfreq, nperseg=nperseg, noverlap=noverlap)
    if spec.size == 0:
        return None
    z = 10 * np.log10(spec + PSWE_EPS)
    x = times[0] + time_points
    y = freqs
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale="Viridis",
            colorbar=dict(title="Power (dB)"),
        )
    )
    fig.update_layout(
        title=f"Spectrogram – {channel_name}",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig






def apply_autoreject_pipeline(raw: mne.io.BaseRaw) -> Tuple[mne.io.BaseRaw, str]:
    if AutoReject is None:
        raise RuntimeError("autoreject is not installed")
    epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True, reject_by_annotation=False)
    if len(epochs) == 0:
        raise RuntimeError("not enough data for AutoReject epochs")
    ar = AutoReject(random_state=97, n_jobs=1, verbose=False)
    ar.fit(epochs)
    reject_log = ar.get_reject_log(epochs)

    cleaned = raw.copy().load_data()
    sfreq = raw.info["sfreq"]
    epoch_duration = epochs.tmax - epochs.tmin
    for idx, bad in enumerate(reject_log.bad_epochs):
        if not bad:
            continue
        onset = epochs.events[idx, 0] / sfreq + epochs.tmin
        cleaned.annotations.append(onset, epoch_duration, "bad_autoreject")

    bad_channels: List[str] = []
    for channel_list in reject_log.bad_channels:
        bad_channels.extend(channel_list)
    if bad_channels:
        cleaned.info["bads"] = sorted(set(bad_channels))
        cleaned.interpolate_bads(reset_bads=True)
    return cleaned, "AutoReject applied (bad epochs annotated and channels interpolated)."


def apply_prep_pipeline(raw: mne.io.BaseRaw) -> Tuple[mne.io.BaseRaw, str]:
    if PrepPipeline is None:
        raise RuntimeError("pyprep is not installed")
    prep_params = {
        "ref_chs": raw.ch_names,
        "reref_chs": raw.ch_names,
        "line_freqs": np.array([50.0]),
    }
    pipeline = PrepPipeline(raw.copy(), prep_params, montage=None)
    pipeline.fit()
    return pipeline.raw, "PREP pipeline applied."


def run_cleaning_pipeline(raw: mne.io.BaseRaw) -> Tuple[mne.io.BaseRaw, List[str]]:
    log: List[str] = []
    cleaned = raw.copy().load_data()
    try:
        cleaned.notch_filter(50.0)
        log.append("50 Hz notch filter applied.")
    except Exception as exc:
        log.append(f"Notch filter failed: {exc}")
    try:
        cleaned.filter(0.1, 100.0)
        log.append("0.1–100 Hz band-pass filter applied.")
    except Exception as exc:
        log.append(f"Band-pass filter failed: {exc}")

    try:
        cleaned, message = apply_autoreject_pipeline(cleaned)
        log.append(message)
    except Exception as exc:
        log.append(f"AutoReject skipped: {exc}")

    try:
        cleaned, message = apply_prep_pipeline(cleaned)
        log.append(message)
    except Exception as exc:
        log.append(f"PREP pipeline skipped: {exc}")

    return cleaned, log


def format_stats_table(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    for column in ["Minimum", "Maximum", "Mean", "Std", "Var"]:
        formatted[column] = formatted[column].map(lambda x: f"{x:.4f}")
    formatted["Spike Count"] = formatted["Spike Count"].astype(int)
    return formatted


def main() -> None:
    st.title("Single Scan EEG/EDF Overview")
    st.markdown(
        """
        Upload an EDF/EEG recording or explore the default control scan. The app highlights numeric channels,
        visualises the raw signal, summarises channel statistics, and offers optional cleaning tools (notch + band-pass
        filtering, AutoReject, and the PREP pipeline). Frequency-domain views, spectrograms, entropy metrics, and
        topographic summaries are included below. All content renders on a single scrollable page using containers
        and expanders.
        """
    )

    with st.container():
        st.subheader("1. Recording selection")
        uploaded = st.file_uploader("Upload EDF/EEG file", type=["edf", "EDF", "eeg", "EEG", "bdf", "BDF", "gdf", "GDF", "fif", "FIF", "set", "SET"])
        default_available = DEFAULT_FILE.exists()
        if default_available:
            st.caption(f"Default file: {DEFAULT_FILE}")
        else:
            st.warning("Default file not found in repository. Upload a file to continue.")
        use_default = st.checkbox("Use default recording when no upload is provided", value=default_available)

    raw: Optional[mne.io.BaseRaw] = None
    file_identifier: Optional[str] = None
    load_error: Optional[str] = None

    if uploaded is not None:
        try:
            raw, file_identifier = load_uploaded_file(uploaded)
        except Exception as exc:  # pragma: no cover - depends on external files
            load_error = f"Unable to load uploaded file: {exc}"
    elif use_default and default_available:
        try:
            raw, file_identifier = load_default_file(DEFAULT_FILE)
        except Exception as exc:
            load_error = f"Unable to load default file: {exc}"

    if load_error:
        st.error(load_error)
        return

    if raw is None:
        st.info("Upload a compatible file or enable the default recording to begin.")
        return

    raw.load_data()

    if st.session_state.get("current_file_id") != file_identifier:
        st.session_state["current_file_id"] = file_identifier
        st.session_state["cleaned_raw"] = None
        st.session_state["cleaning_log"] = []

    cleaned_raw = st.session_state.get("cleaned_raw")
    cleaning_log: List[str] = st.session_state.get("cleaning_log", [])
    active_raw = cleaned_raw if cleaned_raw is not None else raw

    numeric_picks = get_numeric_picks(active_raw)
    if not numeric_picks:
        st.error("No numeric channels were detected in the recording.")
        return

    channel_names = [active_raw.ch_names[idx] for idx in numeric_picks]
    data = active_raw.get_data(picks=numeric_picks)
    sfreq = float(active_raw.info["sfreq"])
    duration = active_raw.n_times / sfreq if sfreq > 0 else 0

    with st.container():
        st.subheader("2. Recording overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Channels", f"{len(channel_names)}")
        col2.metric("Sampling rate (Hz)", f"{sfreq:.2f}")
        col3.metric("Duration (s)", f"{duration:.2f}")
        col4.metric("Data source", "Cleaned" if cleaned_raw is not None else "Raw")

        with st.expander("Channel statistics", expanded=False):
            stats_df = compute_channel_statistics(data, channel_names)
            st.dataframe(format_stats_table(stats_df))

    with st.container():
        st.subheader("3. Cleaning pipeline")
        st.write(
            "Apply a cleaning pipeline (50 Hz notch, 0.1–100 Hz band-pass, AutoReject, PREP). Each step logs whether it succeeded or was skipped."
        )
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("Run cleaning pipeline", type="primary"):
                with st.spinner("Running cleaning pipeline..."):
                    cleaned, log = run_cleaning_pipeline(raw)
                    st.session_state["cleaned_raw"] = cleaned
                    st.session_state["cleaning_log"] = log
                    st.rerun()
        with col_b:
            if st.button("Reset to raw data"):
                st.session_state["cleaned_raw"] = None
                st.session_state["cleaning_log"] = []
                st.rerun()

        if cleaning_log:
            with st.expander("Cleaning log", expanded=True):
                for entry in cleaning_log:
                    st.write(f"• {entry}")

    active_raw = st.session_state.get("cleaned_raw") or raw
    if st.session_state["cleaned_raw"] is not None:
        st.success("Cleaning pipeline completed.")

    numeric_picks = get_numeric_picks(active_raw)
    channel_names = [active_raw.ch_names[idx] for idx in numeric_picks]
    data = active_raw.get_data(picks=numeric_picks)
    sfreq = float(active_raw.info["sfreq"])

    with st.container():
        st.subheader("4. Raw signal inspection")
        if duration <= 0:
            st.warning("Unable to determine recording duration.")
        else:
            max_time = max(duration, 1.0)
            default_end = min(30.0, max_time)
            if default_end <= 0:
                default_end = max_time
            time_window = st.slider(
                "Select time window (seconds)",
                min_value=0.0,
                max_value=float(max_time),
                value=(0.0, float(default_end)),
                step=0.5,
            )
            start_sample = int(max(0, time_window[0] * sfreq))
            stop_sample = int(min(data.shape[1], max(start_sample + 1, time_window[1] * sfreq)))
            window_times = np.arange(start_sample, stop_sample) / sfreq
            window_data = data[:, start_sample:stop_sample]

            default_channels = channel_names[: min(6, len(channel_names))]
            # add a button to select all
            if st.button("Select all channels"):
                default_channels = channel_names
            selected_channels = st.multiselect(
                "Channels to plot",
                channel_names,
                default=default_channels,
            )
            if not selected_channels:
                st.warning("Select at least one channel to visualise raw traces.")
            else:
                indices = [channel_names.index(name) for name in selected_channels]
                fig = plot_raw_signals(window_times, window_data[indices], selected_channels)
                st.plotly_chart(fig, use_container_width=True)

    psds, freqs = psd_array_welch(data, sfreq=sfreq, fmin=0.1, fmax=100.0, average="mean")
    pswe_values = np.array([compute_pswe(psd) for psd in psds])

    with st.container():
        st.subheader("5. Spectral analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig_psd = plot_psd(freqs, psds, channel_names)
            st.plotly_chart(fig_psd, use_container_width=True)
        with col2:
            summary_df = pd.DataFrame({"Channel": channel_names, "PSWE": pswe_values, "Spike Count": [count_spikes(ch) for ch in data]}).set_index("Channel")
            with st.expander("PSWE and spike summary", expanded=False):
                st.dataframe(summary_df)

        if spectrogram is None:
            st.info("Install SciPy to enable spectrogram visualisations.")
        else:
            first_idx = 0
            if len(channel_names) > 1:
                selection = st.selectbox("Channel for spectrogram", channel_names, index=0)
                first_idx = channel_names.index(selection)
            signal = data[first_idx]
            time_axis = active_raw.times
            fig_spec = plot_spectrogram(time_axis, signal, sfreq, channel_names[first_idx])
            if fig_spec is None:
                st.info("Unable to compute spectrogram for the selected channel.")
            else:
                st.plotly_chart(fig_spec, use_container_width=True)


if __name__ == "__main__":
    main()
