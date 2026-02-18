import streamlit as st
import os
st.set_page_config(layout="wide")
import pickle
import numpy as np
import mne
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
from scipy import signal

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

try:
    import pynapple as nap
    PYNAPPLE_AVAILABLE = True
except ImportError:
    PYNAPPLE_AVAILABLE = False
    st.error("Pynapple (nap) not installed. Please install it.")

try:
    import wfdb
    import wfdb.processing
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False

try:
    import heartpy as hp
    HEARTPY_AVAILABLE = True
except ImportError:
    HEARTPY_AVAILABLE = False
def bandpass_ecg(x, fs, lo=3.0, hi=40.0, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lo/nyq, hi/nyq], btype="bandpass")
    return signal.filtfilt(b, a, x)

def detect_rpeaks_abs(x_filt, fs):
    # Make it robust to inversion: detect on absolute signal
    x = np.abs(x_filt)

    # Smooth a bit to emphasize QRS energy
    win = int(0.08 * fs)  # ~80 ms
    win = max(win, 3) | 1  # odd >= 3
    x_smooth = signal.savgol_filter(x, win, polyorder=2)

    # Adaptive-ish threshold using MAD
    mad = np.median(np.abs(x_smooth - np.median(x_smooth))) + 1e-12
    thr = np.median(x_smooth) + 3.5 * mad

    # Refractory period to avoid double detections
    distance = int(0.25 * fs)  # 250 ms
    peaks, _ = signal.find_peaks(x_smooth, height=thr, distance=distance)
    return peaks

def decide_inversion_from_template(x_filt, rpeaks, fs, qrs_ms=120):
    # Build an average beat template centered at R
    half = int((qrs_ms / 1000.0) * fs)  # +- qrs_ms
    if len(rpeaks) < 5:
        return False, None  # not enough info

    # Keep only peaks that allow full window extraction
    rpeaks = rpeaks[(rpeaks > half) & (rpeaks < len(x_filt) - half)]
    if len(rpeaks) < 5:
        return False, None

    beats = np.stack([x_filt[p-half:p+half+1] for p in rpeaks], axis=0)

    # Robust average: median across beats resists outliers
    template = np.median(beats, axis=0)

    # Polarity decision: is the dominant QRS deflection negative?
    pos_peak = np.max(template)
    neg_peak = np.min(template)  # negative value
    
    # FLIP ONLY the scans were 90% of max is smaller than abs(min) of the r peaks.
    inverted = (abs(neg_peak) > 0.9 * abs(pos_peak))

    return inverted, template

def fix_inverted_ecg(ecg, fs, lo=3.0, hi=40.0):
    ecg = np.asarray(ecg).astype(float)

    ecg_f = bandpass_ecg(ecg, fs, lo=lo, hi=hi)
    rpeaks = detect_rpeaks_abs(ecg_f, fs)

    inverted, template = decide_inversion_from_template(ecg_f, rpeaks, fs)

    ecg_fixed = -ecg if inverted else ecg
    info = {
        "inverted_detected": bool(inverted),
        "n_rpeaks_used": int(len(rpeaks)),
        "rpeaks": rpeaks,
        "template_filtered": template,  # median beat on filtered signal
    }
    return ecg_fixed, info

def clean_ecg_high_fidelity(ecg_signal, sampling_rate, wavelet='db1', wavelet_levels=3, zscore_threshold=5.0):
    """
    Improved ECG cleaning to preserve R-peak amplitude and HEP components.
    """
    if len(ecg_signal) == 0:
        return ecg_signal, {'methods_applied': [], 'n_extreme_samples': 0}
    
    methods_applied = []
    info = {}
    cleaned_signal = ecg_signal.copy()
    
    # --- Step 1: Gentle Median Filter ---
    # REDUCED: 300ms was killing the R-peak. 
    # 10-20ms is enough to kill 'pops' without touching the R-wave.
    median_window_ms = 20 
    median_window_samples = int(np.round(median_window_ms * sampling_rate / 1000.0))
    if median_window_samples % 2 == 0: median_window_samples += 1
    
    median_filtered = median_filter(cleaned_signal, size=max(3, median_window_samples))
    cleaned_signal = median_filtered
    methods_applied.append('gentle_median_filter')
    
    # --- Step 2: Bandpass Filter (3-40 Hz) ---
    # Replaces Wavelet Denoising as per user request.
    try:
        lowcut = 3.0
        highcut = 40.0
        nyquist = 0.5 * sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(2, [low, high], btype='band') # Order 2 is usually sufficient for ECG
        
        cleaned_signal = filtfilt(b, a, cleaned_signal)
        methods_applied.append('bandpass_3_40Hz')
    except Exception as e:
        if 'st' in globals():
            st.warning(f"Bandpass filter failed: {e}")
        # Fallback to original signal if filter fails
        cleaned_signal = cleaned_signal

    # --- Step 3: Outlier Handling (Z-score) ---
    # INCREASED: 3.0 often catches the R-wave itself. 5.0-6.0 catches actual artifacts.
    mean_val = np.mean(cleaned_signal)
    std_val = np.std(cleaned_signal)
    if std_val > 0:
        z_scores = (cleaned_signal - mean_val) / std_val
        extreme_mask = np.abs(z_scores) > zscore_threshold
        # Instead of replacing with median, consider clipping or ignoring 
        # to avoid 'flat-top' peaks.
        cleaned_signal[extreme_mask] = np.sign(cleaned_signal[extreme_mask]) * (zscore_threshold * std_val)
        methods_applied.append('zscore_clipping')

    info['methods_applied'] = methods_applied
    return cleaned_signal, info

def detect_rpeaks_robust(ecg_signal_clean, sfreq):
    """
    Detects R-peaks using WFDB XQRS, refines them to local maxima within 0.05s window,
    and filters them to ensure a minimum distance of 501ms between consecutive peaks.
    """
    if not WFDB_AVAILABLE:
        return np.array([], dtype=int)
        
    try:
        # 1. WFDB Detection
        rpeaks = wfdb.processing.xqrs_detect(sig=ecg_signal_clean, fs=sfreq, verbose=False)
        
        # 2. Refine to local maxima (0.05s window)
        if len(rpeaks) > 0:
            window = int(0.1 * sfreq)
            refined_rpeaks = []
            for peak in rpeaks:
                start = max(0, peak - window)
                end = min(len(ecg_signal_clean), peak + window + 1)
                # Find index of max value in the window
                local_max = start + np.argmax(ecg_signal_clean[start:end])
                refined_rpeaks.append(local_max)
            rpeaks = np.array(refined_rpeaks)
            rpeaks = np.unique(rpeaks)
            rpeaks = np.sort(rpeaks)
    
            # 3. Filter peaks: if < 550ms between each other, keep first, remove second
            min_dist = int(0.550 * sfreq)
            if len(rpeaks) > 0:
                filtered_rpeaks = [rpeaks[0]]
                for i in range(1, len(rpeaks)):
                    if rpeaks[i] - filtered_rpeaks[-1] >= min_dist:
                        filtered_rpeaks.append(rpeaks[i])
                rpeaks = np.array(filtered_rpeaks)
                
        return rpeaks
    except Exception as e:
        if 'st' in globals():
            # Avoid spamming warnings if called in a loop, but useful for debug
            pass
        return np.array([], dtype=int)


def process_file_data(raw, patient_id):
    """
    Cleans ECG and extracts R-peaks using the provided logic.
    Returns the raw data, sfreq, and R-peak times if successful.
    """
    # Parameters
    minmax = (-0.3, .4)
    
    # Get sampling frequency and data
    sfreq = raw.info['sfreq']
    ch_names = raw.ch_names
    data = raw.get_data() * 1e6
    
    # Process ECG
    ch_lower = [ch.lower() for ch in ch_names]
    ecg_indices = [i for i, ch in enumerate(ch_lower) if 'ecg' in ch or 'ekg' in ch]
    
    if not ecg_indices:
        if 'st' in globals():
            st.warning(f"No ECG channel found in {patient_id}")
        return None
        
    ecg_ch_idx = ecg_indices[0]
    ecg_signal = data[ecg_ch_idx, :]
    
    # Fix inverted ECG
    ecg_signal, _ = fix_inverted_ecg(ecg_signal, sfreq)
    
    # Clean ECG
    ecg_signal_clean, _ = clean_ecg_high_fidelity(
        ecg_signal, 
        sampling_rate=sfreq
    )
    # Detect R-peaks using robust method
    rpeaks = detect_rpeaks_robust(ecg_signal_clean, sfreq)

    if len(rpeaks) < 2:
        if 'st' in globals():
            st.warning(f"Not enough R-peaks found. Patient ID: {patient_id}")
        return None

    rpeak_times = rpeaks / sfreq
    rpeak_ts = nap.Ts(t=rpeak_times, time_units="s")
    return raw, sfreq, rpeak_ts, rpeaks, minmax


def compute_hep_avg(raw, rpeaks, sfreq, minmax=(-0.5, 1.0), rpeak_ts=None):
    """
    Computes HEP (Heartbeat Evoked Potential) for all EEG channels.
    """
    tmin, tmax = minmax
    
    # Identify EEG channels
    ch_names = raw.ch_names
    eeg_indices = [i for i, ch in enumerate(ch_names) 
                  if 'eeg' in ch.lower() or any(elec in ch.upper() for elec in ['FP', 'F', 'C', 'P', 'O', 'T', 'A'])]
    
    if not eeg_indices:
        return None, None, None
        
    # Pick EEG channels
    eeg_ch_names = [ch_names[i] for i in eeg_indices]
    
    # Pynapple implementation
    data = raw.get_data(picks=eeg_indices).T # (n_times, n_channels)
    tsd_frame = nap.TsdFrame(t=raw.times, d=data, columns=eeg_ch_names)
    perievent = nap.compute_perievent_continuous(tsd_frame, rpeak_ts, minmax=minmax)
    # Average across trials (axis 1)
    mean_data = perievent.nanmean(axis=1).values.T # (n_channels, n_times)
    return mean_data, perievent.t, eeg_ch_names



def compute_ecg_hep_avg(raw, rpeaks, sfreq, minmax=(-0.5, 1.0), rpeak_ts=None):
    """
    Computes HEP (Heartbeat Evoked Potential) for the ECG channel.
    """
    tmin, tmax = minmax
    ch_names = raw.ch_names
    ch_lower = [ch.lower() for ch in ch_names]
    ecg_indices = [i for i, ch in enumerate(ch_lower) if 'ecg' in ch or 'ekg' in ch]
    
    if not ecg_indices:
        return None, None, None
        
    ecg_ch_name = ch_names[ecg_indices[0]]

    # Pynapple implementation
    data = raw.get_data(picks=[ecg_indices[0]]).T # (n_times, 1)
    tsd_frame = nap.TsdFrame(t=raw.times, d=data, columns=[ecg_ch_name])
    perievent = nap.compute_perievent_continuous(tsd_frame, rpeak_ts, minmax=minmax)
    # Average across trials (axis 1)
    mean_data = perievent.nanmean(axis=1).values.T # (1, n_times)
    return mean_data, perievent.t, [ecg_ch_name]

@st.cache_data
def get_group_data(group_name, sleep_stage, base_path):
    """
    Loads all files for a group/sleep_stage, processes them, and computes the Grand Average HEP.
    """
    group_dir = os.path.join(base_path, group_name, sleep_stage)
    if not os.path.exists(group_dir):
        return None, None, 0

    files = [f for f in os.listdir(group_dir) if f.endswith('.pkl')]
    if not files:
        return None, None, 0

    group_hep_sum = None
    count = 0
    common_times = None

    for f_name in files:
        f_path = os.path.join(group_dir, f_name)
        patient_id = f_name.replace('.pkl', '').replace('.edf', '')
        
        with open(f_path, 'rb') as f:
            raw = pickle.load(f)
        
        # Use user snippet logic
        result = process_file_data(raw, patient_id)
        if result is None:
            continue
        
        raw, sfreq, rpeak_ts, rpeaks, minmax = result
        
        # Compute HEP
        hep_data, times, _ = compute_hep_avg(raw, rpeaks, sfreq, minmax, rpeak_ts=rpeak_ts)
        
        if hep_data is None:
            continue
            
        # Average across channels to get "Global Field Power" or "Mean HEP" logic?
        # User asks for "Plot of amp vs time". Typically this is mean across channels OR per channel.
        # "for all the groups".
        # Usually we plot the Global Mean (mean across all EEG electrodes).
        
        mean_hep = np.mean(hep_data, axis=0) # Mean across channels -> (time,)
        
        if group_hep_sum is None:
            group_hep_sum = mean_hep
            common_times = times
            count = 1
        else:
            # Ensure dimensions match
            if len(mean_hep) == len(group_hep_sum):
                group_hep_sum += mean_hep
                count += 1
            elif len(mean_hep) > len(group_hep_sum):
                 group_hep_sum += mean_hep[:len(group_hep_sum)]
                 count += 1
            else: 
                 # Pad or skip?
                 # Simple: trim to min length
                 l = len(mean_hep)
                 group_hep_sum = group_hep_sum[:l] + mean_hep
                 common_times = common_times[:l]
                 count += 1

    if count > 0:
        return group_hep_sum / count, common_times, count
    else:
        return None, None, 0

@st.cache_data
def get_group_individuals(group_name, sleep_stage, base_path):
    """
    Loads all files for a group/sleep_stage and returns individual HEPs.
    Returns: list of (patient_id, hep_data, times, ch_names)
    """
    group_dir = os.path.join(base_path, group_name, sleep_stage)
    if not os.path.exists(group_dir):
        return []

    files = [f for f in os.listdir(group_dir) if f.endswith('.pkl')]
    if not files:
        return []

    individuals = []

    for f_name in files:
        f_path = os.path.join(group_dir, f_name)
        patient_id = f_name.replace('.pkl', '').replace('.edf', '')
        
        with open(f_path, 'rb') as f:
            raw = pickle.load(f)
        
        results = process_file_data(raw, patient_id)
        if results is None:
            continue
        raw, sfreq, rpeak_ts, rpeaks, minmax = results
                
        hep_data, times, ch_names = compute_hep_avg(raw, rpeaks, sfreq, minmax, rpeak_ts=rpeak_ts)
        ecg_hep_data, _, ecg_ch_names = compute_ecg_hep_avg(raw, rpeaks, sfreq, minmax, rpeak_ts=rpeak_ts)
                   
        # Keep full data: (patient_id, hep_data, times, ch_names, rpeaks, ecg_hep_data, ecg_ch_names)
        individuals.append((patient_id, hep_data, times, ch_names, rpeaks, ecg_hep_data, ecg_ch_names))

    return individuals

from mne.stats import permutation_cluster_1samp_test
import numpy as np
from scipy import stats
from scipy.ndimage import label

def permutation_cluster_jitter_test(avg_hep, times, n_permutations=100, p_threshold=0.05, jitter_sec=None):
    """
    Performs cluster-based permutation test with a controlled jitter range to identify
    statistically significant time-windows in HEP data.
    
    Parameters:
    -----------
    avg_hep : np.ndarray
        Data matrix of shape (n_subjects, n_times).
    times : np.ndarray
        Time points (in seconds) corresponding to the columns of avg_hep.
    n_permutations : int
        Number of random permutations to perform (default: 100).
    p_threshold : float
        Alpha level for cluster significance (default: 0.05).
    jitter_sec : float or None
        The max duration in seconds to randomly shift each subject's data. 
        If None, shifts can span the entire epoch.
    """
    n_subjects, n_times = avg_hep.shape
    
    # Calculate Sampling Rate (Hz) to convert time-jitter into sample-jitter
    sfreq = 1 / np.mean(np.diff(times))
    if jitter_sec is not None:
        max_shift = int(jitter_sec * sfreq)
    else:
        # If no jitter range specified, allow shifting across all available time points
        max_shift = n_times

    # Determine t-statistic threshold for a single time-point significance
    # based on the number of subjects (degrees of freedom = n-1)
    t_thresh = stats.t.ppf(1 - p_threshold / 2, df=n_subjects - 1)
    
    # 1. Observed Clusters: Calculate t-stats for the actual data
    t_obs, _ = stats.ttest_1samp(avg_hep, 0)
    
    # Identify contiguous time points where the t-statistic exceeds the threshold
    obs_mask = np.abs(t_obs) > t_thresh
    obs_labels, n_clusters = label(obs_mask) # Label contiguous clusters
    
    if n_clusters == 0:
        raise ValueError("No clusters found in observed data.")

    # Calculate "Cluster Mass" (sum of absolute t-values within each cluster)
    obs_cluster_masses = [np.sum(np.abs(t_obs[obs_labels == i+1])) for i in range(n_clusters)]

    # 2. Permutation Loop: Build a null distribution by shuffling the data
    null_dist = np.zeros(n_permutations)
    for p in range(n_permutations):
        hep_jittered = np.zeros_like(avg_hep)
        for s in range(n_subjects):
            # For each subject, apply a random circular shift (jitter)
            # This breaks the temporal alignment to the heartbeat while preserving the signal's structure
            shift = np.random.randint(-max_shift, max_shift + 1)
            hep_jittered[s, :] = np.roll(avg_hep[s, :], shift)
        
        # Calculate t-stats for the jittered data
        t_null, _ = stats.ttest_1samp(hep_jittered, 0)
        
        # Identify clusters in the null data
        null_mask = np.abs(t_null) > t_thresh
        null_labels, n_null_clusters = label(null_mask)
        
        # Store only the maximum cluster mass found in this iteration
        # This builds a distribution of the "best possible" cluster by chance
        if n_null_clusters > 0:
            null_dist[p] = max([np.sum(np.abs(t_null[null_labels == i+1])) for i in range(n_null_clusters)])

    # 3. Compile Results: Compare observed cluster masses against the null distribution
    significant_windows = []
    for i in range(n_clusters):
        # Calculate p-value: proportion of null clusters stronger than the observed one
        p_val = np.mean(null_dist >= obs_cluster_masses[i])
        
        # If the p-value is below our alpha threshold, the cluster is significant
        if p_val < p_threshold:
            indices = np.where(obs_labels == i + 1)[0]
            significant_windows.append({
                'start': times[indices[0]],
                'end': times[indices[-1]],
                'p_value': p_val
            })

    return significant_windows, t_obs

def finalize_plot(fig, ax, title, avg_hep=None, times=None, n_subjects=None, significant_windows=None):
    """
    Applies common styling to the plot, optionally plots the average and significance, and displays it.
    """
    if avg_hep is not None and times is not None:
        label = f"Group Average (n={n_subjects})" if n_subjects is not None else "Group Average"
        ax.plot(times, avg_hep * 1e6, color='blue', linewidth=2, label=label)

    if significant_windows:
        ylim = ax.get_ylim()
        for window in significant_windows:
            start, end = window['start'], window['end']
            p_val = window['p_value']
            
            # Plot rectangle
            ax.axvspan(start, end, color='orange', alpha=0.2)
            
            # Plot asterisk
            mid_point = (start + end) / 2
            
            # Determine asterisks
            if p_val < 0.001:
                asterisks = "***"
            elif p_val < 0.01:
                asterisks = "**"
            else:
                asterisks = "*"
            current_ymax = ax.get_ylim()[1]
            # p_val_str = f"{p_val:.1e}"
            ax.text(mid_point, current_ymax-.4, asterisks, ha='center', va='bottom', fontsize=16, color='orange', fontweight='bold')
            # ax.text(mid_point, current_ymax-.6, p_val_str, ha='center', va='bottom', fontsize=16, color='orange', fontweight='bold')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (μV)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.axvline(0, color='r', linestyle='--', alpha=0.5)
    # set ax.set_xlim to be the same as times
    ax.set_xlim(times[0], times[-1])
    # tight_layout
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

def run_compare_groups_analysis(base_path, selected_stage):
    """
    Logic for Comparing Groups mode.
    """
    # Verify groups exist
    available_groups = [g for g in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, g))]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    has_data = False
    
    for group in available_groups:
        st.write(f"Processing Group: {group}...")
        avg_hep, times, n_subjects = get_group_data(group, selected_stage, base_path)
        
        if avg_hep is not None:
                ax.plot(times, avg_hep * 1e6, label=f"{group} (n={n_subjects})") # Convert to uV
                has_data = True
        else:
                st.warning(f"No data found for group {group} in stage {selected_stage}")

    if has_data:
        finalize_plot(fig, ax, f"HEP Comparison - Sleep Stage {selected_stage}")
    else:
        st.error("No valid data found for any group.")

import re

def plot_ecg_cleaning_comparison(raw, patient_id):
    """
    Plots a 10 second segment of raw vs cleaned ECG and compares 5 R-peak detection methods.
    """
    sfreq = raw.info['sfreq']
    ch_names = raw.ch_names
    ch_lower = [ch.lower() for ch in ch_names]
    ecg_indices = [i for i, ch in enumerate(ch_lower) if 'ecg' in ch or 'ekg' in ch]
    
    if not ecg_indices:
        st.warning(f"No ECG channel found for {patient_id}")
        return

    ecg_ch_idx = ecg_indices[0]
    ecg_ch_name = ch_names[ecg_ch_idx]
    
    total_len = raw.n_times
    segment_duration = 10 
    n_samples = int(segment_duration * sfreq)
    
    start_sample = max(0, (total_len // 2) - (n_samples // 2))
    end_sample = min(total_len, start_sample + n_samples)
    
    raw_segment = raw.get_data(picks=[ecg_ch_idx], start=start_sample, stop=end_sample)[0] * 1e6
    clean_segment, _ = clean_ecg_high_fidelity(raw_segment, sampling_rate=sfreq)
    times = np.arange(len(raw_segment)) / sfreq

    # --- Compute 5 Methods ---
    methods_peaks = {}

    # 1. NeuroKit2
    _, rpk_nk = nk.ecg_peaks(clean_segment, sampling_rate=sfreq)
    methods_peaks['NeuroKit2'] = rpk_nk['ECG_R_Peaks']

    # 2. WFDB XQRS
    # 2. WFDB XQRS
    methods_peaks['WFDB (XQRS)'] = detect_rpeaks_robust(clean_segment, sfreq)

    # 3. HeartPy
    if HEARTPY_AVAILABLE:
        try:
            working_data, _ = hp.process(clean_segment, sample_rate=sfreq)
            methods_peaks['HeartPy'] = working_data['peaklist']
        except:
            methods_peaks['HeartPy (Failed)'] = []
    else:
        methods_peaks['HeartPy (Missing)'] = []

    # 4. SciPy (No Threshold)
    distance = int(0.6 * sfreq) # Assume max HR of ~100 bpm
    peaks_scipy, _ = find_peaks(clean_segment, distance=distance) 
    methods_peaks['SciPy'] = peaks_scipy

    # 5. MNE (Peak Finder)
    # Using MNE's peak_finder function
    import mne
    rpeaks_mne, _ = mne.preprocessing.peak_finder(clean_segment, thresh=None, extrema=1, verbose=False)
    methods_peaks['MNE (Peak Finder)'] = rpeaks_mne

    # --- Plotting ---
    method_names = list(methods_peaks.keys())
    fig, axes = plt.subplots(len(method_names), 1, figsize=(16, 3 * len(method_names)), sharex=True)
    
    if len(method_names) == 1:
        axes = [axes]

    for idx, (method_name, rpeaks) in enumerate(methods_peaks.items()):
        ax = axes[idx]
        ax.plot(times, raw_segment, color='gray', alpha=0.3, label='Raw' if idx == 0 else None)
        ax.plot(times, clean_segment, color='red', alpha=0.7, label='Cleaned' if idx == 0 else None)
        
        for i, peak in enumerate(rpeaks):
            peak_time = peak / sfreq
            ax.axvline(peak_time, color='green', linestyle='--', alpha=0.6, 
                       label='R-peak' if (idx == 0 and i == 0) else None)
        
        ax.set_ylabel("Amp (μV)")
        ax.set_title(f"Method: {method_name}")
        if idx == 0:
            ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

def plot_ecg_hep_individuals(individuals, selected_group, selected_stage):
    """
    Plots ECG HEP for all individuals and their average.
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    all_ecg_heps = []
    # allow user to select how many individuals to plot
    n_individuals = st.slider("Number of individuals to plot", min_value=1, max_value=len(individuals), value=len(individuals))
    for ind in individuals[:n_individuals]:
        # ind structure: (patient_id, hep_data, times, ch_names, rpeaks, ecg_hep_data, ecg_ch_names)
        ecg_hep = ind[5]
        if ecg_hep is not None:
            hep = ecg_hep[0]
            ax.plot(ind[2], hep * 1e6, color='gray', alpha=0.3, linewidth=1)
            all_ecg_heps.append(hep)
    
    if all_ecg_heps:
        avg_ecg = np.nanmean(all_ecg_heps, axis=0)
        finalize_plot(
            fig, ax, 
            f"ECG - Group: {selected_group} - Stage: {selected_stage}",
            avg_hep=avg_ecg, 
            times=individuals[0][2], 
            n_subjects=len(all_ecg_heps)
        )
        
        # Calculate epochs stats
        epochs_data = [{"Patient ID": ind[0], "Epochs": len(ind[4])} for ind in individuals]
        if epochs_data:
            df_epochs = pd.DataFrame(epochs_data)
            avg_epochs = df_epochs["Epochs"].mean()
            st.write(f"**Average Epochs per Patient:** {avg_epochs:.2f}")
            
            with st.expander("See Epochs per Patient"):
                st.dataframe(df_epochs)
                
    else:
        st.warning("No ECG data found for this group.")

def analyze_ecg_repetitive_pattern(ecg_signal, sfreq, patient_id=""):
    """
    Detects repetitive waveform templates/motifs in ECG signal.
    This identifies identical wave patterns that repeat throughout the recording.
    
    Returns:
    --------
    pattern_metrics : dict
        Dictionary containing:
        - 'patient_id': Patient identifier
        - 'template': Extracted repetitive template waveform
        - 'template_times': Time axis for template
        - 'match_locations': Sample indices where template repeats
        - 'match_correlations': Correlation strength at each match location
        - 'avg_correlation': Average correlation strength
        - 'repetition_rate': Matches per second
        - 'has_repetitive_pattern': Boolean indicating significant repetition
    """
    from scipy import signal as sp_signal
    from scipy.signal import correlate, find_peaks
    
    # Use a manageable segment for analysis (first 60 seconds)
    max_samples = min(len(ecg_signal), int(60 * sfreq))
    signal_segment = ecg_signal[:max_samples]
    
    # Template extraction parameters
    # Try different template durations to find repeating patterns
    template_durations = [0.05, 0.1, 0.2, 0.3, 0.5]  # seconds
    
    best_template = None
    best_matches = []
    best_correlations = []
    best_avg_corr = 0
    best_duration = 0
    
    for template_dur in template_durations:
        template_len = int(template_dur * sfreq)
        
        if template_len >= len(signal_segment) // 4:
            continue
        
        # Strategy: Find the most repetitive segment by testing multiple candidates
        # Sample multiple starting points
        n_candidates = min(20, len(signal_segment) // template_len)
        candidate_starts = np.linspace(0, len(signal_segment) - template_len, n_candidates, dtype=int)
        
        for start_idx in candidate_starts:
            # Extract candidate template
            template = signal_segment[start_idx:start_idx + template_len]
            
            # Normalize template
            template_norm = (template - np.mean(template)) / (np.std(template) + 1e-10)
            
            # Normalize signal segment
            signal_norm = (signal_segment - np.mean(signal_segment)) / (np.std(signal_segment) + 1e-10)
            
            # Cross-correlate to find matches
            correlation = correlate(signal_norm, template_norm, mode='valid')
            
            # Find peaks in correlation (high correlation = good match)
            # Require minimum separation between matches
            min_separation = template_len // 2
            threshold = 0.7  # Correlation threshold for a "good match"
            
            peak_indices, properties = find_peaks(correlation, 
                                                   height=threshold, 
                                                   distance=min_separation)
            
            if len(peak_indices) > 0:
                avg_corr = np.mean(correlation[peak_indices])
                
                # Keep the template with the most repetitions and highest avg correlation
                if avg_corr > best_avg_corr and len(peak_indices) >= 3:
                    best_avg_corr = avg_corr
                    best_template = template
                    best_matches = peak_indices
                    best_correlations = correlation[peak_indices]
                    best_duration = template_dur
    
    # Prepare return values
    if best_template is not None:
        template_times = np.arange(len(best_template)) / sfreq
        repetition_rate = len(best_matches) / (len(signal_segment) / sfreq)
        has_repetitive_pattern = len(best_matches) >= 5 and best_avg_corr >= 0.75
    else:
        # No strong repetitive pattern found
        template_times = np.array([])
        repetition_rate = 0
        has_repetitive_pattern = False
        best_template = np.array([])
        best_matches = []
        best_correlations = []
    
    return {
        'patient_id': patient_id,
        'template': best_template,
        'template_times': template_times,
        'template_duration': best_duration,
        'match_locations': best_matches,
        'match_correlations': best_correlations,
        'avg_correlation': best_avg_corr,
        'repetition_rate': repetition_rate,
        'n_matches': len(best_matches),
        'has_repetitive_pattern': has_repetitive_pattern,
        'signal_segment': signal_segment,  # For visualization
        'sfreq': sfreq
    }

def handle_ecg_noise_detection(base_path, selected_group, selected_stage):
    """
    Analyzes ECG for repetitive waveform patterns in all patients and displays results.
    """
    st.subheader(f"ECG Repetitive Pattern Analysis - {selected_group} {selected_stage}")
    
    st.info("This analysis detects identical waveform patterns that repeat throughout each patient's ECG recording.")
    
    group_dir = os.path.join(base_path, selected_group, selected_stage)
    files = [f for f in os.listdir(group_dir) if f.endswith('.pkl')]
    
    if not files:
        st.warning("No files found for pattern analysis.")
        return
    
    all_metrics = []
    
    progress_bar = st.progress(0)
    
    for idx, f_name in enumerate(files):
        progress_bar.progress((idx + 1) / len(files))
        
        f_path = os.path.join(group_dir, f_name)
        patient_id = f_name.replace('.pkl', '').replace('.edf', '')
        
        try:
            with open(f_path, 'rb') as f:
                raw = pickle.load(f)
            
            # Extract ECG channel
            sfreq = raw.info['sfreq']
            ch_names = raw.ch_names
            ch_lower = [ch.lower() for ch in ch_names]
            ecg_indices = [i for i, ch in enumerate(ch_lower) if 'ecg' in ch or 'ekg' in ch]
            
            if not ecg_indices:
                continue
            
            ecg_idx = ecg_indices[0]
            ecg_signal = raw.get_data(picks=[ecg_idx])[0] * 1e6  # Convert to uV
            
            # Analyze for repetitive patterns
            metrics = analyze_ecg_repetitive_pattern(ecg_signal, sfreq, patient_id)
            all_metrics.append(metrics)
            
        except Exception as e:
            st.warning(f"Error processing {patient_id}: {e}")
            continue
    
    progress_bar.empty()
    
    if not all_metrics:
        st.error("No valid ECG data found for pattern analysis.")
        return
    
    # --- Summary Statistics ---
    st.markdown("### Group Summary")
    
    n_patients = len(all_metrics)
    n_with_pattern = sum(1 for m in all_metrics if m['has_repetitive_pattern'])
    
    # Calculate averages only for patients with detected patterns
    valid_metrics = [m for m in all_metrics if len(m['template']) > 0]
    if valid_metrics:
        avg_repetition_rate = np.mean([m['repetition_rate'] for m in valid_metrics])
        avg_correlation = np.mean([m['avg_correlation'] for m in valid_metrics])
        avg_n_matches = np.mean([m['n_matches'] for m in valid_metrics])
    else:
        avg_repetition_rate = 0
        avg_correlation = 0
        avg_n_matches = 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", n_patients)
    with col2:
        st.metric("With Repetitive Pattern", f"{n_with_pattern} ({100*n_with_pattern/n_patients:.1f}%)")
    with col3:
        st.metric("Avg Repetition Rate", f"{avg_repetition_rate:.2f} /sec")
    with col4:
        st.metric("Avg Correlation", f"{avg_correlation:.3f}")
    
    # --- Per-Patient Table ---
    st.markdown("### Per-Patient Pattern Metrics")
    
    df_data = []
    for m in all_metrics:
        if len(m['template']) > 0:
            df_data.append({
                'Patient ID': m['patient_id'],
                'Pattern Duration (ms)': f"{m['template_duration']*1000:.1f}",
                'Num Repetitions': m['n_matches'],
                'Repetition Rate (/sec)': f"{m['repetition_rate']:.2f}",
                'Avg Correlation': f"{m['avg_correlation']:.3f}",
                'Has Strong Pattern': '✓' if m['has_repetitive_pattern'] else '✗'
            })
        else:
            df_data.append({
                'Patient ID': m['patient_id'],
                'Pattern Duration (ms)': 'N/A',
                'Num Repetitions': 0,
                'Repetition Rate (/sec)': '0.00',
                'Avg Correlation': 'N/A',
                'Has Strong Pattern': '✗'
            })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)
    
    # --- Visualization: Individual Patient Templates ---
    st.markdown("### Individual Patient Templates")
    st.markdown("Shows the extracted repetitive waveform pattern for each patient")
    
    # Filter patients with valid templates
    patients_with_templates = [m for m in all_metrics if len(m['template']) > 0]
    
    if not patients_with_templates:
        st.warning("No repetitive patterns detected in any patient.")
        return
    
    n_show = st.slider("Number of patients to show", min_value=1, 
                       max_value=min(12, len(patients_with_templates)), 
                       value=min(6, len(patients_with_templates)), 
                       key="pattern_patient_slider")
    
    # Create grid of template plots
    n_cols = 3
    n_rows = int(np.ceil(n_show / n_cols))
    
    fig_templates, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(n_show):
        ax = axes[i]
        m = patients_with_templates[i]
        
        # Plot the template
        ax.plot(m['template_times'] * 1000, m['template'], color='red', linewidth=2)
        
        pattern_status = "STRONG" if m['has_repetitive_pattern'] else "WEAK"
        ax.set_title(f"{m['patient_id']} - {pattern_status}\n"
                     f"Repeats: {m['n_matches']}, Corr: {m['avg_correlation']:.2f}", 
                     fontsize=9)
        ax.set_xlabel('Time (ms)', fontsize=8)
        ax.set_ylabel('Amplitude (μV)', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    
    # Hide unused subplots
    for j in range(n_show, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig_templates, use_container_width=True)
    
    # --- Detailed View: Template in Context ---
    st.markdown("### Detailed Pattern View")
    st.markdown("View the repetitive pattern in context of the full ECG signal")
    
    selected_patient_idx = st.selectbox(
        "Select patient for detailed view",
        range(len(patients_with_templates)),
        format_func=lambda i: patients_with_templates[i]['patient_id']
    )
    
    m = patients_with_templates[selected_patient_idx]
    
    # Show template
    st.markdown(f"**Patient: {m['patient_id']}**")
    st.markdown(f"- Pattern duration: {m['template_duration']*1000:.1f} ms")
    st.markdown(f"- Found {m['n_matches']} repetitions (rate: {m['repetition_rate']:.2f} /sec)")
    st.markdown(f"- Average correlation: {m['avg_correlation']:.3f}")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Plot template
        fig_temp, ax_temp = plt.subplots(figsize=(7, 4))
        ax_temp.plot(m['template_times'] * 1000, m['template'], color='red', linewidth=2)
        ax_temp.set_title(f"Extracted Repetitive Template", fontsize=12, fontweight='bold')
        ax_temp.set_xlabel('Time (ms)')
        ax_temp.set_ylabel('Amplitude (μV)')
        ax_temp.grid(True, alpha=0.3)
        st.pyplot(fig_temp, use_container_width=True)
    
    with col_b:
        # Plot correlation values at match locations
        fig_corr, ax_corr = plt.subplots(figsize=(7, 4))
        ax_corr.bar(range(len(m['match_correlations'])), m['match_correlations'], 
                    color='steelblue', alpha=0.7)
        ax_corr.axhline(0.75, color='red', linestyle='--', label='Strong Match Threshold')
        ax_corr.set_title(f"Correlation Strength per Repetition", fontsize=12, fontweight='bold')
        ax_corr.set_xlabel('Repetition Number')
        ax_corr.set_ylabel('Correlation Coefficient')
        ax_corr.set_ylim(0, 1)
        ax_corr.legend()
        ax_corr.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig_corr, use_container_width=True)
    
    # Show signal with marked repetitions
    st.markdown("**Signal with Pattern Locations Marked**")
    
    # User can select time window
    signal_duration = len(m['signal_segment']) / m['sfreq']
    view_window = st.slider("View window (seconds)", 
                            min_value=2, 
                            max_value=min(30, int(signal_duration)), 
                            value=10,
                            key="pattern_window_slider")
    
    start_time = st.slider("Start time (seconds)", 
                          min_value=0.0, 
                          max_value=max(0.0, signal_duration - view_window),
                          value=0.0,
                          step=0.5,
                          key="pattern_start_slider")
    
    start_sample = int(start_time * m['sfreq'])
    end_sample = min(len(m['signal_segment']), int((start_time + view_window) * m['sfreq']))
    
    signal_view = m['signal_segment'][start_sample:end_sample]
    times_view = np.arange(len(signal_view)) / m['sfreq'] + start_time
    
    fig_signal, ax_signal = plt.subplots(figsize=(15, 5))
    ax_signal.plot(times_view, signal_view, color='black', linewidth=1, alpha=0.7, label='ECG Signal')
    
    # Mark pattern locations in the view window
    for match_loc in m['match_locations']:
        match_time = match_loc / m['sfreq']
        if start_time <= match_time <= start_time + view_window:
            ax_signal.axvline(match_time, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    
    ax_signal.set_title(f"ECG Signal with Repetitive Pattern Locations (Red lines)", fontsize=12)
    ax_signal.set_xlabel('Time (s)')
    ax_signal.set_ylabel('Amplitude (μV)')
    ax_signal.grid(True, alpha=0.3)
    ax_signal.legend()
    st.pyplot(fig_signal, use_container_width=True)
    
    # --- Average Template Across All Patients ---
    st.markdown("### Average Template Across All Patients")
    
    # Find common duration (use median)
    template_lengths = [len(m['template']) for m in patients_with_templates]
    median_length = int(np.median(template_lengths))
    
    # Interpolate all templates to median length and average
    from scipy.interpolate import interp1d
    
    aligned_templates = []
    for m in patients_with_templates:
        if len(m['template']) > 0:
            # Interpolate to median length
            x_old = np.linspace(0, 1, len(m['template']))
            x_new = np.linspace(0, 1, median_length)
            f = interp1d(x_old, m['template'], kind='linear')
            template_aligned = f(x_new)
            aligned_templates.append(template_aligned)
    
    if aligned_templates:
        avg_template = np.mean(aligned_templates, axis=0)
        std_template = np.std(aligned_templates, axis=0)
        
        # Use median duration for time axis
        median_duration = np.median([m['template_duration'] for m in patients_with_templates])
        avg_times = np.linspace(0, median_duration * 1000, median_length)
        
        fig_avg, ax_avg = plt.subplots(figsize=(10, 5))
        ax_avg.plot(avg_times, avg_template, color='blue', linewidth=3, label=f'Average (n={len(aligned_templates)})')
        ax_avg.fill_between(avg_times, avg_template - std_template, avg_template + std_template,
                           color='blue', alpha=0.2, label='± 1 SD')
        
        ax_avg.set_title(f"Average Repetitive Pattern - {selected_group} {selected_stage}", 
                        fontsize=14, fontweight='bold')
        ax_avg.set_xlabel('Time (ms)', fontsize=11)
        ax_avg.set_ylabel('Amplitude (μV)', fontsize=11)
        ax_avg.grid(True, alpha=0.3)
    ax_avg.legend()
    st.pyplot(fig_avg, use_container_width=True)

    # --- Cleaning Methods Comparison ---
    st.markdown("### Cleaning Methods Comparison")
    st.markdown("Visualizing the effect of different artifact removal methods on the selected patient's signal.")
    
    if 'm' in locals() and len(m['template']) > 0:
        # Use the same patient selected in "Detailed Pattern View"
        
        # 1. Define Cleaning Functions locally or call them if defined outside
        # Defining them here for clarity in context, or better move to global scope. 
        # I'll define them here to access 'm' easily if needed, but standard practice is outside.
        # Let's call the helper functions defined above (we will add them to the file scope in a moment).
        # Since I can't add them to file scope in this single Replace block easily without replacing the whole file,
        # I will define them as nested functions or just implement the logic inline for this block.
        # Actually, I can add the helper functions *before* this function in a previous edit, or just include them here.
        # Given the constraint of one Replace block, I will include the logic here.
        
        # ... Wait, I should add the helper functions properly at module level. 
        # But this tool call is for `replace_file_content`. I can only do one contiguous block.
        # I will implement the logic inside `handle_ecg_noise_detection` or helper functions *inside* this block 
        # if I want to do it in one go. However, to keep code clean, I should probably use `multi_replace` 
        # if I want to insert functions elsewhere. 
        # BUT, for now, let's implement the cleaning logic directly here for the specific signal view.
        
        signal_seg = m['signal_segment']
        template = m['template']
        match_locs = m['match_locations']
        template_len = len(template)
        
        # Method 1: Global Template Subtraction
        clean_global = signal_seg.copy()
        for start_idx in match_locs:
            if start_idx + template_len <= len(clean_global):
                clean_global[start_idx:start_idx+template_len] -= template
                
        # Method 2: Scaled Template Subtraction
        clean_scaled = signal_seg.copy()
        template_norm_sq = np.sum(template**2)
        if template_norm_sq > 0:
            for start_idx in match_locs:
                if start_idx + template_len <= len(clean_scaled):
                    segment = clean_scaled[start_idx:start_idx+template_len]
                    scale = np.dot(segment, template) / template_norm_sq
                    # Limit scale to reasonable bounds to avoid subtracting noise that looks like template
                    # scale = np.clip(scale, 0.5, 2.0) # Optional
                    clean_scaled[start_idx:start_idx+template_len] -= scale * template
        
        # Method 3: Local Moving Average
        clean_local = signal_seg.copy()
        window_size = 10
        n_matches = len(match_locs)
        
        for i, start_idx in enumerate(match_locs):
            # Find local window indices
            start_w = max(0, i - window_size // 2)
            end_w = min(n_matches, start_w + window_size)
            if end_w - start_w < window_size and start_w > 0:
                start_w = max(0, end_w - window_size)
            
            local_indices = match_locs[start_w:end_w]
            
            # Compute local template
            local_segments = []
            for loc in local_indices:
                if loc + template_len <= len(signal_seg):
                    local_segments.append(signal_seg[loc:loc+template_len])
            
            if local_segments:
                local_temp = np.mean(local_segments, axis=0)
                if start_idx + template_len <= len(clean_local):
                    clean_local[start_idx:start_idx+template_len] -= local_temp

        # --- Plotting ---
        # Reuse the zoomed window from earlier
        # start_sample, end_sample, times_view defined above in "Signal with Pattern Locations Marked"
        
        # We need to slice the cleaned signals to match the view window
        # Ensure we have the view variables
        if 'start_sample' in locals() and 'end_sample' in locals() and 'times_view' in locals():
            sig_view_raw = signal_seg[start_sample:end_sample]
            sig_view_global = clean_global[start_sample:end_sample]
            sig_view_scaled = clean_scaled[start_sample:end_sample]
            sig_view_local = clean_local[start_sample:end_sample]
            
            fig_clean, axes_clean = plt.subplots(4, 1, figsize=(15, 10), sharex=True, sharey=True)
            
            # 1. Original
            axes_clean[0].plot(times_view, sig_view_raw, color='black', alpha=0.8, linewidth=1)
            axes_clean[0].set_title("Original Signal (Red lines = detected patterns)", fontsize=10, fontweight='bold')
            # Add markers
            for match_loc in match_locs:
                match_time = match_loc / m['sfreq']
                if times_view[0] <= match_time <= times_view[-1]:
                    axes_clean[0].axvline(match_time, color='red', linestyle='--', alpha=0.5)
            
            # 2. Global
            axes_clean[1].plot(times_view, sig_view_global, color='#1f77b4', linewidth=1)
            axes_clean[1].set_title("Method 1: Global Template Subtraction", fontsize=10, fontweight='bold')
            
            # 3. Scaled
            axes_clean[2].plot(times_view, sig_view_scaled, color='#ff7f0e', linewidth=1)
            axes_clean[2].set_title("Method 2: Scaled Template Subtraction (Best Fit)", fontsize=10, fontweight='bold')
            
            # 4. Local
            axes_clean[3].plot(times_view, sig_view_local, color='#2ca02c', linewidth=1)
            axes_clean[3].set_title(f"Method 3: Local Moving Average (Window={window_size})", fontsize=10, fontweight='bold')
            
            for ax in axes_clean:
                ax.grid(True, alpha=0.3)
                ax.set_ylabel("Amplitude (μV)")
            
            axes_clean[-1].set_xlabel("Time (s)")
            
            plt.tight_layout()
            st.pyplot(fig_clean, use_container_width=True)
        else:
            st.warning("Please interact with the Signal View slider above to initialize the plot.")

    st.markdown("""
    **Interpretation Guide:**
    - **Template**: The extracted waveform pattern that repeats throughout the recording
    - **Repetition Rate**: How many times per second the pattern appears
    - **Correlation**: How similar each repetition is to the template (1.0 = perfect match)
    - **Strong Pattern**: Pattern repeats ≥5 times with correlation ≥0.75
    - **Red vertical lines**: Locations where the repetitive pattern was detected
    """)

def handle_ecg_cleaning_comparison(base_path, selected_group, selected_stage):
    """
    Reloads the first patient in a group and plots the raw vs cleaned ECG comparison.
    """
    group_dir = os.path.join(base_path, selected_group, selected_stage)
    files = [f for f in os.listdir(group_dir) if f.endswith('.pkl')]
    if files:
        f_path = os.path.join(group_dir, files[0])
        patient_id = files[0].replace('.pkl', '').replace('.edf', '')
        with open(f_path, 'rb') as f:
            raw_first = pickle.load(f)
        plot_ecg_cleaning_comparison(raw_first, patient_id)

def handle_single_patient_view(individuals, selected_group, selected_stage, base_path):
    """
    Handles the selection and plotting of all channels for a single patient,
    plus a raw ECG viewer with time controls.
    """
    patient_ids = [ind[0] for ind in individuals]
    selected_pid = st.selectbox("Select Patient for All Channels", patient_ids)
    
    # --- Raw ECG Viewer ---
    st.markdown("### Raw ECG Viewer")
    
    # Locate and load the file for the selected patient
    # We need to find the file that starts with selected_pid in the group folder
    group_dir = os.path.join(base_path, selected_group, selected_stage)
    ecg_raw_data = None
    ecg_sfreq = None
    
    if os.path.exists(group_dir):
        # Find file starting with patient_id
        # Note: ind[0] might be just the base ID, but filenames might be longer.
        # Let's assume filenames start with the ID.
        potential_files = [f for f in os.listdir(group_dir) if f.startswith(selected_pid) and f.endswith('.pkl')]
        if potential_files:
            file_path = os.path.join(group_dir, potential_files[0])
            try:
                with open(file_path, 'rb') as f:
                    raw_obj = pickle.load(f)
                    
                ecg_sfreq = raw_obj.info['sfreq']
                ch_names = raw_obj.ch_names
                ch_lower = [ch.lower() for ch in ch_names]
                ecg_indices = [i for i, ch in enumerate(ch_lower) if 'ecg' in ch or 'ekg' in ch]
                
                if ecg_indices:
                    ecg_idx = ecg_indices[0]
                    # Get all data for this channel
                    ecg_raw_data = raw_obj.get_data(picks=[ecg_idx])[0] * 1e6 # Convert to uV
                else:
                    st.warning("No ECG channel found in the raw file.")
                    
            except Exception as e:
                st.error(f"Error loading raw file for ECG viewer: {e}")
        else:
             st.warning(f"Could not find source file for {selected_pid} to display raw ECG.")
             
    if ecg_raw_data is not None and ecg_sfreq is not None:
        duration_sec = len(ecg_raw_data) / ecg_sfreq
        
        col1, col2 = st.columns(2)
        with col1:
            start_time = st.number_input("Start Time (s)", min_value=0.0, max_value=duration_sec, value=0.0, step=1.0)
        with col2:
            view_duration = st.slider("View Duration (s)", min_value=1, max_value=60, value=10)
            
        # Plot
        start_sample = int(start_time * ecg_sfreq)
        end_sample = min(len(ecg_raw_data), start_sample + int(view_duration * ecg_sfreq))
        
        # Add plot mode selector
        plot_mode = st.radio("ECG Plot Mode", ["Overlay", "Separate Subplots"], horizontal=True)

        segment = ecg_raw_data[start_sample:end_sample]
        segment_times = np.linspace(start_time, start_time + len(segment)/ecg_sfreq, len(segment))
        
        # Buffer and Clean logic
        clean_segment = None
        # Initialize fixed_segment to match raw segment length initially (or None)
        fixed_segment = None 
        
        try:
             # Add buffer
            buffer_samples = int(1.0 * ecg_sfreq) # 1 sec buffer
            start_buf = max(0, start_sample - buffer_samples)
            end_buf = min(len(ecg_raw_data), end_sample + buffer_samples)
            
            segment_buf = ecg_raw_data[start_buf:end_buf]
            
            # 1. Fix Inverted
            fixed_segment_buf, fit_info = fix_inverted_ecg(segment_buf, fs=ecg_sfreq)

            # 2. Clean based on fixed signal
            clean_segment_buf, _ = clean_ecg_high_fidelity(fixed_segment_buf, sampling_rate=ecg_sfreq)
            
            # Crop back to original window
            crop_start = start_sample - start_buf
            crop_end = crop_start + (end_sample - start_sample)
            
            clean_segment = clean_segment_buf[crop_start:crop_end]
            fixed_segment = fixed_segment_buf[crop_start:crop_end]
            
        except Exception as e:
            st.warning(f"Could not clean ECG segment: {e}")

        # Plotting based on mode
        if plot_mode == "Overlay":
            fig_ecg_raw, ax_ecg = plt.subplots(figsize=(12, 4))
            ax_ecg.plot(segment_times, segment, color='black', linewidth=1, label='Raw ECG', alpha=0.5)
            
            if fixed_segment is not None and len(fixed_segment) == len(segment):
                 # Plot fixed signal if valid
                 ax_ecg.plot(segment_times, fixed_segment, color='blue', linestyle='--', linewidth=1, alpha=0.7, label='Inverted (Fixed)')

            if clean_segment is not None and len(clean_segment) == len(segment):
                 ax_ecg.plot(segment_times, clean_segment, color='red', linewidth=1, alpha=0.8, label='Cleaned ECG')
            
            ax_ecg.set_title(f"Raw vs Fixed vs Cleaned ECG - {selected_pid} (Start: {start_time}s)")
            ax_ecg.set_xlabel("Time (s)")
            ax_ecg.set_ylabel("Amplitude (μV)")
            ax_ecg.grid(True, alpha=0.3)
            ax_ecg.legend(loc='upper right')
            st.pyplot(fig_ecg_raw, use_container_width=True)

            # --- Cleaned ECG with WFDB R-peaks ---
            if clean_segment is not None and WFDB_AVAILABLE:
                st.markdown("#### Cleaned ECG with WFDB R-peaks")
                try:
                    # Detect R-peaks on the cleaned segment
                    # Detect R-peaks on the cleaned segment
                    rpeaks_loc = detect_rpeaks_robust(clean_segment, ecg_sfreq)
                    
                    fig_rpeaks, ax_rpeaks = plt.subplots(figsize=(12, 4))
                    ax_rpeaks.plot(segment_times, clean_segment, color='red', linewidth=1, label='Cleaned ECG')
                    
                    if len(rpeaks_loc) > 0:
                        # rpeaks_loc are indices relative to start of clean_segment
                        rpeak_times = segment_times[rpeaks_loc]
                        rpeak_amps = clean_segment[rpeaks_loc]
                        
                        ax_rpeaks.plot(rpeak_times, rpeak_amps, 'bo', label='WFDB R-peaks')
                        
                        for rt in rpeak_times:
                             ax_rpeaks.axvline(rt, color='blue', linestyle='--', alpha=0.3)
                    else:
                        st.info("No R-peaks detected by WFDB in this segment.")
                        
                    ax_rpeaks.set_title(f"Cleaned ECG with WFDB R-peaks - {selected_pid}")
                    ax_rpeaks.set_xlabel("Time (s)")
                    ax_rpeaks.set_ylabel("Amplitude (μV)")
                    ax_rpeaks.grid(True, alpha=0.3)
                    ax_rpeaks.legend(loc='upper right')
                    st.pyplot(fig_rpeaks, use_container_width=True)
                except Exception as e:
                    st.warning(f"Error detecting R-peaks with WFDB: {e}")

        else: # Separate Subplots
            fig_ecg_raw, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
            
            # Raw
            axes[0].plot(segment_times, segment, color='black', linewidth=1)
            axes[0].set_title("Raw ECG")
            axes[0].set_ylabel("Amp (μV)")
            axes[0].grid(True, alpha=0.3)
            
            # Fixed
            if fixed_segment is not None and len(fixed_segment) == len(segment):
                axes[1].plot(segment_times, fixed_segment, color='blue', linewidth=1)
            else:
                axes[1].text(0.5, 0.5, "Processing Failed", ha='center', va='center')
            axes[1].set_title("Inverted (Fixed) ECG")
            axes[1].set_ylabel("Amp (μV)")
            axes[1].grid(True, alpha=0.3)

            # Cleaned
            if clean_segment is not None and len(clean_segment) == len(segment):
                axes[2].plot(segment_times, clean_segment, color='red', linewidth=1)
            else:
                axes[2].text(0.5, 0.5, "Cleaning Failed", ha='center', va='center')
                
            axes[2].set_title("Cleaned ECG")
            axes[2].set_ylabel("Amp (μV)")
            axes[2].set_xlabel("Time (s)")
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig_ecg_raw, use_container_width=True)

        # --- Raw EEG Viewer ---
        st.markdown(f"**Raw EEG Viewer ({selected_pid})**")
        # Filter for EEG channels only (exclude ECG, EOG, etc.)
        eeg_channels = [ch for ch in ch_names if not any(x in ch.lower() for x in ['ecg', 'ekg', 'eog', 'emg', 'resp'])]
        selected_eeg_ch = st.selectbox("Select EEG Electrode", eeg_channels)
        
        if selected_eeg_ch:
             try:
                 eeg_ch_idx = ch_names.index(selected_eeg_ch)
                 eeg_raw_data = raw_obj.get_data(picks=[eeg_ch_idx])[0] * 1e6
                 
                 eeg_segment = eeg_raw_data[start_sample:end_sample]
                 
                 fig_eeg_raw, ax_eeg_raw = plt.subplots(figsize=(12, 4))
                 ax_eeg_raw.plot(segment_times, eeg_segment, color='blue', linewidth=1)
                 ax_eeg_raw.set_title(f"Raw EEG - {selected_eeg_ch} (Time Aligned)")
                 ax_eeg_raw.set_xlabel("Time (s)")
                 ax_eeg_raw.set_ylabel("Amplitude (μV)")
                 ax_eeg_raw.grid(True, alpha=0.3)
                 st.pyplot(fig_eeg_raw, use_container_width=True)
                 
             except Exception as e:
                 st.error(f"Error extracting EEG channel {selected_eeg_ch}: {e}")



        # --- HRV Analysis (RR Intervals) ---
        st.markdown(f"**HRV Analysis (RR Intervals) - {selected_pid}**")
        
        # We need to get the rpeaks for this patient.
        # Check individuals list for the selected patient
        rpeaks_current = None
        for ind_chk in individuals:
            if ind_chk[0] == selected_pid:
                # ind structure: (patient_id, hep_data, times, ch_names, rpeaks, ecg_hep_data, ecg_ch_names)
                rpeaks_current = ind_chk[4]
                break
        
        if rpeaks_current is not None and len(rpeaks_current) > 1:
            # Calculate RR intervals in ms
            # fs is needed. Use ecg_sfreq from raw file above, or assume it's same for group.
            if ecg_sfreq is None:
                 pass

            if ecg_sfreq:
                rr_intervals_samples = np.diff(rpeaks_current)
                rr_intervals_ms = (rr_intervals_samples / ecg_sfreq) * 1000
                
                # Filter outliers (550ms to 1300ms)
                valid_rr = rr_intervals_ms[(rr_intervals_ms >= 550) & (rr_intervals_ms <= 1300)]
                n_excluded = len(rr_intervals_ms) - len(valid_rr)
                
                if len(valid_rr) > 0:
                    rr_mean = np.mean(valid_rr)
                    rr_std = np.std(valid_rr)
                    rr_median = np.median(valid_rr)
                    hr_mean = 60000 / rr_mean
                    
                    st.write(f"**Mean RR:** {rr_mean:.1f} ms | **Median RR:** {rr_median:.1f} ms | **SDNN:** {rr_std:.1f} ms | **Mean HR:** {hr_mean:.1f} bpm")
                    if n_excluded > 0:
                        st.caption(f"(Excluded {n_excluded} intervals outside 550-1300ms range)")
                    
                    fig_hrv, ax_hrv = plt.subplots(figsize=(10, 4))
                    ax_hrv.hist(valid_rr, bins=30, color='purple', alpha=0.7, edgecolor='black')
                    ax_hrv.set_title(f"RR Interval Distribution - {selected_pid} (550-1300ms) - {hr_mean:.1f} BPM")
                    ax_hrv.set_xlabel("RR Interval (ms)")
                    ax_hrv.set_ylabel("Count")
                    ax_hrv.set_xlim(550, 1300)
                    ax_hrv.grid(True, alpha=0.3)
                    
                    # Add vertical line for mean
                    ax_hrv.axvline(rr_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {rr_mean:.1f}ms')
                    ax_hrv.legend()
                    
                    st.pyplot(fig_hrv, use_container_width=True)
                else:
                    st.warning("No valid RR intervals found (all outside 550-1300ms).")
            else:
                 st.warning("Sampling rate unknown (file load failed), cannot calculate RR intervals in ms.")
        else:
            st.warning("Not enough R-peaks to calculate HRV.")

    for ind in individuals:
        if ind[0] == selected_pid:
            # Determine grid size
            ch_names = ind[3]
            n_channels = len(ch_names)
            n_cols = 5
            n_rows = int(np.ceil(n_channels / n_cols))
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows), sharex=True, sharey=True)
            axes = axes.flatten()
            
            times = ind[2]
            full_hep = ind[1] # (n_ch, n_times)
            
            # Get ECG HEP data if available
            ecg_hep = ind[5] # (1, n_times)
            
            # remove LOC ROC channels
            full_hep = full_hep[:-2]
            ch_names = ch_names[:-2]
            # Determine symmetric limits for alignment
            # EEG Limits
            max_eeg = np.nanmax(np.abs(full_hep * 1e6)) * 1.1 # 10% margin
            ylim_eeg = (-max_eeg, max_eeg)

            # ECG Limits
            if ecg_hep is not None:
                max_ecg = np.nanmax(np.abs(ecg_hep[0] * 1e6)) * 1.1 # 10% margin
                ylim_ecg = (-max_ecg, max_ecg)
            
            for i, (ch_name, ch_data) in enumerate(zip(ch_names, full_hep)):
                ax = axes[i]
                ax.plot(times, ch_data * 1e6)
                ax.set_title(ch_name)
                ax.grid(True)
                ax.axvline(0, color='r', linestyle='--', alpha=0.5)
                ax.set_ylim(ylim_eeg)
                
                # Add ECG HEP on secondary axis
                if ecg_hep is not None:
                    ax2 = ax.twinx()
                    # ecg_hep is (1, n_times), so take [0]
                    ax2.plot(times, ecg_hep[0] * 1e6, color='gray', linestyle='--', alpha=0.7,linewidth=1)
                    ax2.set_ylim(ylim_ecg)
                    
                    # Only show right y-axis labels for the rightmost column to reduce clutter
                    if (i + 1) % n_cols == 0:
                        ax2.set_ylabel("ECG (μV)", color='gray', fontsize=8)
                    else:
                        ax2.set_yticklabels([])
                    
                    ax2.tick_params(axis='y', labelcolor='gray', labelsize=8)

                if i % n_cols == 0:
                    ax.set_ylabel("Amp (μV)")
                if i >= n_channels - n_cols:
                    ax.set_xlabel("Time (s)")
                    
            # Hide unused subplots
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')
                
            fig.suptitle(f"All Channels - {selected_pid} - {selected_group} {selected_stage}", fontsize=16)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)



            # --- Butterfly Plot (Z-Scored) ---
            st.subheader(f"All Channels Z-Scored ({selected_pid})")
            fig_z, ax_z = plt.subplots(figsize=(14, 6)) # Increased width for legend

            # Compute Common Average Reference (CAR)
            # Use only EEG channels for CAR (exclude potential artifact channels)
            # Filter: strictly exclude ECG/EOG/EMG and require digit or 'z' suffix
            eeg_indices_car = [
                idx for idx, name in enumerate(ch_names)
                if not any(excl in name.upper() for excl in ['ECG', 'EKG', 'EOG', 'EMG', 'RESP', 'PULSE'])
                and (name[-1].isdigit() or name.lower().endswith('z'))
            ]
            
            if eeg_indices_car:
                car_signal = np.mean(full_hep[eeg_indices_car], axis=0)
            else:
                # Fallback if no specific EEG channels found
                car_signal = np.mean(full_hep, axis=0)

            for i, (ch_name, ch_data) in enumerate(zip(ch_names, full_hep)):
                # Filter: last char must be digit
                if not ch_name[-1].isdigit():
                    continue
                    
                last_digit = int(ch_name[-1])
                
                # Color logic: Even=Blue, Odd=Red
                if last_digit % 2 == 0:
                    color = 'blue'
                else:
                    color = 'red'

                # Subtract CAR
                ch_data_car = ch_data - car_signal

                # Calculate Z-score per channel on CAR-subtracted data
                if np.std(ch_data_car) > 0:
                    z_data = (ch_data_car - np.mean(ch_data_car)) / np.std(ch_data_car)
                else:
                    z_data = np.zeros_like(ch_data_car)
                    
                ax_z.plot(times, z_data, color=color, alpha=0.3, linewidth=1, label=ch_name)
            
            # Plot CAR signal (Z-scored)
            if np.std(car_signal) > 0:
                z_car = (car_signal - np.mean(car_signal)) / np.std(car_signal)
                ax_z.plot(times, z_car, color='black', alpha=0.8, linewidth=2, linestyle='--', label='CAR (EEG only)')

            ax_z.set_title(f"Butterfly Plot - Z-Scored (CAR Subtracted, Even=Blue, Odd=Red) - {selected_pid}")
            ax_z.set_xlabel("Time (s)")
            ax_z.set_ylabel("Z-Score")
            ax_z.axvline(0, color='r', linestyle='--', alpha=0.8)
            ax_z.grid(True, alpha=0.3)
            
            # Add legend outside
            ax_z.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0., fontsize='small', ncol=1)
            
            st.pyplot(fig_z, use_container_width=True)

            # --- Butterfly Plot (Raw uV) with ECG ---
            st.subheader(f"Butterfly Plot - Raw uV ({selected_pid})")
            fig_raw, ax_raw = plt.subplots(figsize=(14, 6))

            # Plot EEG channels
            for i, (ch_name, ch_data) in enumerate(zip(ch_names, full_hep)):
                 # Filter: last char must be digit
                if not ch_name[-1].isdigit():
                    continue
                last_digit = int(ch_name[-1])
                color = 'blue' if last_digit % 2 == 0 else 'red'
                
                ax_raw.plot(times, ch_data * 1e6, color=color, alpha=0.3, linewidth=1)

            ax_raw.set_xlabel("Time (s)")
            ax_raw.set_ylabel("EEG Amplitude (μV)")
            ax_raw.axvline(0, color='r', linestyle='--', alpha=0.8)
            ax_raw.grid(True, alpha=0.3)

            # Secondary Axis for ECG
            if ecg_hep is not None:
                ax_ecg = ax_raw.twinx()
                ax_ecg.plot(times, ecg_hep[0] * 1e6, color='green', alpha=0.6, linewidth=2, linestyle='--', label='ECG')
                ax_ecg.set_ylabel("ECG Amplitude (μV)", color='green')
                ax_ecg.tick_params(axis='y', labelcolor='green')
                # Add legend for ECG
                ax_ecg.legend(loc='upper right')

            ax_raw.set_title(f"Butterfly Plot - Raw uV (Even=Blue, Odd=Red) - {selected_pid}")
            
            # Add legend for Even/Odd EEG (optional, purely visual)
            from matplotlib.lines import Line2D
            custom_lines = [Line2D([0], [0], color='blue', lw=2),
                            Line2D([0], [0], color='red', lw=2)]
            ax_raw.legend(custom_lines, ['Even Channels', 'Odd Channels'], loc='upper left')

            st.pyplot(fig_raw, use_container_width=True)

            # --- Butterfly Plot - All Channels Colored by Region ---
            st.subheader(f"Butterfly Plot - Region Colored ({selected_pid})")
            fig_reg_all, ax_reg_all = plt.subplots(figsize=(14, 6))

            region_color_map = {
                'F': 'blue',
                'C': 'green',
                'T': 'red',
                'P': 'purple',
                'O': 'orange'
            }
            
            # Helper to determine region
            def get_region_color(ch_name):
                name_upper = ch_name.upper()
                # Check regions in specific order to handle overlaps if any (e.g. FC -> Frontal-Central)
                # Typically F first checks F, FC, Fp. C checks C, CP.
                # Simplest heuristic: check letter presence.
                for reg, col in region_color_map.items():
                   if reg in name_upper:
                       return col, reg
                return 'gray', 'Other'

            # Plot EEG channels
            used_regions = set()
            for i, (ch_name, ch_data) in enumerate(zip(ch_names, full_hep)):
                color, region = get_region_color(ch_name)
                ax_reg_all.plot(times, ch_data * 1e6, color=color, alpha=0.3, linewidth=1)
                used_regions.add(region)

            ax_reg_all.set_xlabel("Time (s)")
            ax_reg_all.set_ylabel("EEG Amplitude (μV)")
            ax_reg_all.axvline(0, color='r', linestyle='--', alpha=0.8)
            ax_reg_all.grid(True, alpha=0.3)

            # Secondary Axis for ECG
            if ecg_hep is not None:
                ax_ecg_reg = ax_reg_all.twinx()
                ax_ecg_reg.plot(times, ecg_hep[0] * 1e6, color='black', alpha=0.6, linewidth=2, linestyle='--', label='ECG')
                ax_ecg_reg.set_ylabel("ECG Amplitude (μV)", color='black')
                ax_ecg_reg.tick_params(axis='y', labelcolor='black')
                ax_ecg_reg.legend(loc='upper right')

            ax_reg_all.set_title(f"Butterfly Plot - Region Colored - {selected_pid}")
            
            # Custom legend for regions
            custom_lines_reg = [Line2D([0], [0], color=region_color_map[reg], lw=2) for reg in region_color_map if reg in used_regions]
            custom_labels_reg = [f"Region {reg}" for reg in region_color_map if reg in used_regions]
            if 'Other' in used_regions:
                custom_lines_reg.append(Line2D([0], [0], color='gray', lw=2))
                custom_labels_reg.append("Region Other")
            
            ax_reg_all.legend(custom_lines_reg, custom_labels_reg, loc='upper left')

            st.pyplot(fig_reg_all, use_container_width=True)

            # --- Butterfly Plot - Odd Channels (Region Colored) ---
            st.subheader(f"Butterfly Plot - Odd Channels ({selected_pid})")
            fig_odd, ax_odd = plt.subplots(figsize=(14, 6))
            
            used_regions_odd = set()
            has_odd = False
            for i, (ch_name, ch_data) in enumerate(zip(ch_names, full_hep)):
                if not ch_name[-1].isdigit(): continue
                if int(ch_name[-1]) % 2 == 0: continue # Skip even

                color, region = get_region_color(ch_name)
                ax_odd.plot(times, ch_data * 1e6, color=color, alpha=0.3, linewidth=1)
                used_regions_odd.add(region)
                has_odd = True

            if has_odd:
                ax_odd.set_xlabel("Time (s)")
                ax_odd.set_ylabel("EEG Amplitude (μV)")
                ax_odd.axvline(0, color='r', linestyle='--', alpha=0.8)
                ax_odd.grid(True, alpha=0.3)

                if ecg_hep is not None:
                    ax_ecg_odd = ax_odd.twinx()
                    ax_ecg_odd.plot(times, ecg_hep[0] * 1e6, color='black', alpha=0.6, linewidth=2, linestyle='--', label='ECG')
                    ax_ecg_odd.set_ylabel("ECG Amplitude (μV)", color='black')
                    ax_ecg_odd.tick_params(axis='y', labelcolor='black')
                    ax_ecg_odd.legend(loc='upper right')

                ax_odd.set_title(f"Butterfly Plot - Odd Channels - {selected_pid}")
                
                 # Legend
                custom_lines_odd = [Line2D([0], [0], color=region_color_map[reg], lw=2) for reg in region_color_map if reg in used_regions_odd]
                custom_labels_odd = [f"Region {reg}" for reg in region_color_map if reg in used_regions_odd]
                if 'Other' in used_regions_odd:
                    custom_lines_odd.append(Line2D([0], [0], color='gray', lw=2))
                    custom_labels_odd.append("Region Other")
                
                ax_odd.legend(custom_lines_odd, custom_labels_odd, loc='upper left')
                
                st.pyplot(fig_odd, use_container_width=True)
            else:
                plt.close(fig_odd)

            # --- Butterfly Plot - Even Channels (Region Colored) ---
            st.subheader(f"Butterfly Plot - Even Channels ({selected_pid})")
            fig_even, ax_even = plt.subplots(figsize=(14, 6))
            
            used_regions_even = set()
            has_even = False
            for i, (ch_name, ch_data) in enumerate(zip(ch_names, full_hep)):
                if not ch_name[-1].isdigit(): continue
                if int(ch_name[-1]) % 2 != 0: continue # Skip odd

                color, region = get_region_color(ch_name)
                ax_even.plot(times, ch_data * 1e6, color=color, alpha=0.3, linewidth=1)
                used_regions_even.add(region)
                has_even = True

            if has_even:
                ax_even.set_xlabel("Time (s)")
                ax_even.set_ylabel("EEG Amplitude (μV)")
                ax_even.axvline(0, color='r', linestyle='--', alpha=0.8)
                ax_even.grid(True, alpha=0.3)

                if ecg_hep is not None:
                    ax_ecg_even = ax_even.twinx()
                    ax_ecg_even.plot(times, ecg_hep[0] * 1e6, color='black', alpha=0.6, linewidth=2, linestyle='--', label='ECG')
                    ax_ecg_even.set_ylabel("ECG Amplitude (μV)", color='black')
                    ax_ecg_even.tick_params(axis='y', labelcolor='black')
                    ax_ecg_even.legend(loc='upper right')

                ax_even.set_title(f"Butterfly Plot - Even Channels - {selected_pid}")

                # Legend
                custom_lines_even = [Line2D([0], [0], color=region_color_map[reg], lw=2) for reg in region_color_map if reg in used_regions_even]
                custom_labels_even = [f"Region {reg}" for reg in region_color_map if reg in used_regions_even]
                if 'Other' in used_regions_even:
                    custom_lines_even.append(Line2D([0], [0], color='gray', lw=2))
                    custom_labels_even.append("Region Other")
                
                ax_even.legend(custom_lines_even, custom_labels_even, loc='upper left')

                st.pyplot(fig_even, use_container_width=True)
            else:
                plt.close(fig_even)

            # --- Regional Butterfly Plots ---
            regions = ['F', 'T', 'P', 'O']
            for region in regions:
                # Find channels containing the region letter (e.g., 'F' in 'Fp1', 'F3', 'Fz')
                region_indices = [idx for idx, name in enumerate(ch_names) if region in name.upper()]
                
                if not region_indices:
                    continue

                st.subheader(f"Butterfly Plot - Region {region} ({selected_pid})")
                fig_reg, ax_reg = plt.subplots(figsize=(14, 6))
                
                # Setup colormap
                cmap = plt.get_cmap('tab10')
                has_channels = False
                
                # Collect handles for legend
                region_handles = []
                region_labels = []

                for k, idx in enumerate(region_indices):
                    ch_name = ch_names[idx]
                    ch_data = full_hep[idx]
                    
                    # Use a unique color for each channel
                    color = cmap(k % 10)
                    
                    line, = ax_reg.plot(times, ch_data * 1e6, color=color, alpha=0.6, linewidth=1.5, label=ch_name)
                    has_channels = True
                    region_handles.append(line)
                    region_labels.append(ch_name)

                if not has_channels:
                    plt.close(fig_reg)
                    continue

                ax_reg.set_xlabel("Time (s)")
                ax_reg.set_ylabel("EEG Amplitude (μV)")
                ax_reg.axvline(0, color='r', linestyle='--', alpha=0.8)
                ax_reg.grid(True, alpha=0.3)
                
                # Secondary Axis for ECG
                if ecg_hep is not None:
                    ax_ecg_reg = ax_reg.twinx()
                    ecg_line, = ax_ecg_reg.plot(times, ecg_hep[0] * 1e6, color='black', alpha=0.4, linewidth=2, linestyle='--', label='ECG')
                    ax_ecg_reg.set_ylabel("ECG Amplitude (μV)", color='black')
                    ax_ecg_reg.tick_params(axis='y', labelcolor='black')
                    
                    # Add ECG to legend
                    region_handles.append(ecg_line)
                    region_labels.append('ECG')

                ax_reg.set_title(f"Butterfly Plot - Region {region} - {selected_pid}")
                
                # Add legend
                ax_reg.legend(region_handles, region_labels, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
                
                st.pyplot(fig_reg, use_container_width=True)

            break

def plot_patients_butterfly_comparison(individuals, selected_group, selected_stage):
    """
    Plots a grid of butterfly plots, one method per patient.
    """
    n_pats = len(individuals)
    n_cols = 2
    n_rows = int(np.ceil(n_pats / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows), sharex=True, sharey=True)
    if n_pats == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    for idx, ind in enumerate(individuals):
        ax = axes[idx]
        patient_id = ind[0]
        full_hep = ind[1]
        times = ind[2]
        ch_names = ind[3]
        ecg_hep = ind[5]
        
        # Plot EEG channels
        for i, (ch_name, ch_data) in enumerate(zip(ch_names, full_hep)):
            # Filter: last char must be digit
            if not ch_name[-1].isdigit():
                continue
            last_digit = int(ch_name[-1])
            color = 'blue' if last_digit % 2 == 0 else 'red'
            
            ax.plot(times, ch_data * 1e6, color=color, alpha=0.3, linewidth=1)

        ax.grid(True, alpha=0.3)
        ax.set_ylim(-30, 30)
        ax.axvline(0, color='r', linestyle='--', alpha=0.8)
        
        # Secondary Axis for ECG
        if ecg_hep is not None:
            ax_ecg = ax.twinx()
            ax_ecg.plot(times, ecg_hep[0] * 1e6, color='green', alpha=0.6, linewidth=1.5, linestyle='--')
            
            ax_ecg.set_ylim(-150, 150)

            # Only label rightmost plots
            if (idx + 1) % n_cols == 0:
                 ax_ecg.set_ylabel("ECG (μV)", color='green')
            else:
                 ax_ecg.set_yticklabels([])
                 
            ax_ecg.tick_params(axis='y', labelcolor='green')

        ax.set_title(f"{patient_id}")
        
        if idx % n_cols == 0:
            ax.set_ylabel("EEG (μV)")
        if idx >= n_pats - n_cols:
            ax.set_xlabel("Time (s)")

    # Hide unused subplots
    for j in range(len(individuals), len(axes)):
        axes[j].axis('off')
        
    fig.suptitle(f"Patients Comparison (Butterfly EEG + ECG) - {selected_group} {selected_stage}", fontsize=16)
    st.pyplot(fig, use_container_width=True)

def plot_group_ecg_analysis(individuals, selected_group, selected_stage):
    """
    Plots the average ECG of the whole group and individual patient ECGs.
    """
    st.subheader(f"Group ECG Analysis - {selected_group} {selected_stage}")

    # 1. Group Average ECG
    all_ecg_heps = []
    for ind in individuals:
        ecg_hep = ind[5]
        if ecg_hep is not None:
            all_ecg_heps.append(ecg_hep[0])
    
    if all_ecg_heps:
        avg_ecg = np.nanmean(all_ecg_heps, axis=0)
        times = individuals[0][2]
        
        fig_avg, ax_avg = plt.subplots(figsize=(10, 5))
        ax_avg.plot(times, avg_ecg * 1e6, color='black', linewidth=2, label='Group Average ECG')
        
        # Optional: Add standard error shading
        std_ecg = np.nanstd(all_ecg_heps, axis=0)
        sem_ecg = std_ecg / np.sqrt(len(all_ecg_heps))
        ax_avg.fill_between(times, (avg_ecg - sem_ecg) * 1e6, (avg_ecg + sem_ecg) * 1e6, color='gray', alpha=0.3, label='SEM')

        ax_avg.set_title(f"Group Average ECG (n={len(all_ecg_heps)})")
        ax_avg.set_xlabel("Time (s)")
        ax_avg.set_ylabel("Amplitude (μV)")
        ax_avg.axvline(0, color='r', linestyle='--', alpha=0.5)
        ax_avg.legend()
        ax_avg.grid(True, alpha=0.3)
        st.pyplot(fig_avg, use_container_width=True)
    else:
        st.warning("No ECG data found for this group to calculate average.")

    # 2. Individual Patients
    st.markdown("#### Individual Patient ECGs")
    n_individuals = st.slider("Number of individual patients to show", min_value=1, max_value=len(individuals), value=min(4, len(individuals)), key="group_ecg_slider")
    
    # Grid layout for individuals
    n_cols = 2
    n_rows = int(np.ceil(n_individuals / n_cols))
    
    fig_ind, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True, sharey=True)
    if n_individuals == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    for i in range(n_individuals):
        ind = individuals[i]
        ax = axes[i]
        patient_id = ind[0]
        ecg_hep = ind[5]
        times = ind[2]
        
        if ecg_hep is not None:
             ax.plot(times, ecg_hep[0] * 1e6, color='green', linewidth=1.5)
        else:
             ax.text(0.5, 0.5, "No ECG Data", ha='center', va='center')
             
        ax.set_title(f"{patient_id}")
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='r', linestyle='--', alpha=0.5)
        
        if i % n_cols == 0:
            ax.set_ylabel("ECG (μV)")
        if i >= n_individuals - n_cols:
            ax.set_xlabel("Time (s)")
            
    # Hide unused
    if hasattr(axes, '__len__'):    
        for j in range(n_individuals, len(axes)):
            axes[j].axis('off')
            
    st.pyplot(fig_ind, use_container_width=True)

def run_single_group_analysis(base_path, selected_stage):
    """
    Logic for Single Group Analysis mode.
    """
    # Get available groups
    available_groups = [g for g in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, g))]
    selected_group = st.selectbox("Select Group", available_groups,index=1)
    
    st.write(f"Processing individuals for {selected_group}...")
    individuals = get_group_individuals(selected_group, selected_stage, base_path)
    # make st number_input to ger from user jitter_sec defult .1
    jitter_sec = st.number_input("Jitter (seconds)", min_value=0.01, max_value=.5, value=0.1, step=0.05)
    # checkbox 'show ecg only plots'
    show_ecg_only = st.checkbox("Show ECG Only Plots", value=False)
    show_cleaning_comparison = st.checkbox("Show ECG Cleaning Comparison", value=False)
    show_noise_analysis = st.checkbox("Show ECG Repetitive Noise Analysis", value=False)
    show_single_patient_all = st.checkbox("Show Single Patient All Channels", value=True)
    show_patients_comparison = st.checkbox("Show EEG-ECG Patients Comparison", value=False)
    show_group_ecg = st.checkbox("Show Group ECG Analysis", value=False)
    
    if individuals:
        if show_cleaning_comparison:
            handle_ecg_cleaning_comparison(base_path, selected_group, selected_stage)
        
        if show_noise_analysis:
            handle_ecg_noise_detection(base_path, selected_group, selected_stage)

        if show_single_patient_all:
            handle_single_patient_view(individuals, selected_group, selected_stage, base_path)

        if show_patients_comparison:
            n_compare = st.slider("Number of patients to compare", min_value=1, max_value=len(individuals), value=min(4, len(individuals)))
            plot_patients_butterfly_comparison(individuals[:n_compare], selected_group, selected_stage)

        if show_ecg_only:
            plot_ecg_hep_individuals(individuals, selected_group, selected_stage)

        if show_group_ecg:
            plot_group_ecg_analysis(individuals, selected_group, selected_stage)


        # Identify common channels across all individuals
        all_channel_sets = [set(ind[3]) for ind in individuals]
        common_channels = sorted(list(set.intersection(*all_channel_sets)))
        # Channel must be letter and number or letter and 'z'
        common_channels = [ch for ch in common_channels if re.match(r'^[a-zA-Z][0-9]*$', ch) or re.match(r'^[a-zA-Z]z$', ch)]

        if not common_channels:
            st.error("No common EEG channels found across all individuals in this group.")
            return

        # run per channel
        for ch_name in common_channels:
            fig, ax = plt.subplots(figsize=(16, 9))
            
            # Plot individuals
            all_full_heps = []
            
            for pid, hep_full, times, ch_names, rpeaks, ecg_hep, ecg_ch in individuals:
                ch_idx = ch_names.index(ch_name)
                hep = hep_full[ch_idx]
                ax.plot(times, hep * 1e6, color='gray', alpha=0.3, linewidth=1)
                all_full_heps.append(hep)
            
            avg_hep = None
            sig_windows = None
            if all_full_heps:
                avg_hep = np.nanmean(all_full_heps, axis=0)
                sig_windows, _ = permutation_cluster_jitter_test(np.array(all_full_heps), times,jitter_sec=jitter_sec)
            # Finalize with Average
            finalize_plot(
                fig, ax, 
                f"Channel: {ch_name} - Group: {selected_group} - Stage: {selected_stage}",
                avg_hep=avg_hep, 
                times=times, 
                n_subjects=len(individuals),
                significant_windows=sig_windows
            )
    else:
        st.error(f"No data found for group {selected_group} in stage {selected_stage}")

def run_compare_sleep_stages_analysis(base_path):
    """
    Logic for Compare Sleep Stages mode.
    Plots ECG HEP for a single patient across different sleep stages.
    """
    # 1. Select Group
    available_groups = [g for g in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, g))]
    if not available_groups:
        st.error("No groups found.")
        return
    selected_group = st.selectbox("Select Group", available_groups, index=1)

    # 2. Select Patient
    # Scan through sleep stages to find a list of all patients
    sleep_stages = ['W', 'N1', 'N2', 'N3', 'R']
    all_patients = set()
    
    for stage in sleep_stages:
        stage_dir = os.path.join(base_path, selected_group, stage)
        if os.path.exists(stage_dir):
            files = [f for f in os.listdir(stage_dir) if f.endswith('.pkl')]
            for f in files:
                pid_full = f.replace('.pkl', '').replace('.edf', '')
                # Split by '_' and take the first part to get the base patient ID
                pid_base = pid_full.split('_')[0]
                all_patients.add(pid_base)
    
    if not all_patients:
        st.warning(f"No patients found in group {selected_group}")
        return

    selected_pid = st.selectbox("Select Patient", sorted(list(all_patients)))

    # 3. Process Stages - Collect Data First
    st.subheader(f"Comparisons across Sleep Stages - {selected_pid} - {selected_group}")
    
    stage_data = {}
    
    # Define colors for stages
    stage_colors = {
        'W': 'orange',
        'N1': 'yellow',
        'N2': 'lightblue',
        'N3': 'blue',
        'R': 'red'
    }

    progress_bar = st.progress(0)
    
    for idx, stage in enumerate(sleep_stages):
        progress_bar.progress((idx + 1) / len(sleep_stages))
        
        # Find the specific file for this patient in this stage
        # Since filename might have extra info, search for startswith selected_pid
        stage_dir = os.path.join(base_path, selected_group, stage)
        if not os.path.exists(stage_dir):
            continue
            
        stage_files = [f for f in os.listdir(stage_dir) if f.startswith(selected_pid) and f.endswith('.pkl')]
        
        if not stage_files:
            continue
            
        # Assuming only one file matches the base ID per stage
        file_path = os.path.join(stage_dir, stage_files[0])
        
        with open(file_path, 'rb') as f:
            try:
                raw = pickle.load(f)
            except Exception as e:
                st.warning(f"Error loading {file_path}: {e}")
                continue

        # Process Data
        # Pass the full found ID just in case, or base ID? The function uses it for logging/title mostly
        results = process_file_data(raw, selected_pid)
        if results is None:
            continue
            
        raw, sfreq, rpeak_ts, rpeaks, minmax = results
        
        # Compute ECG HEP
        ecg_hep_data, times, _ = compute_ecg_hep_avg(raw, rpeaks, sfreq, minmax, rpeak_ts=rpeak_ts)
        # Compute EEG HEP
        eeg_hep_data, _, ch_names = compute_hep_avg(raw, rpeaks, sfreq, minmax, rpeak_ts=rpeak_ts)
        
        stage_data[stage] = {
            'ecg_hep': ecg_hep_data,
            'eeg_hep': eeg_hep_data,
            'times': times,
            'ch_names': ch_names,
            'n_epochs': len(rpeaks)
        }

    progress_bar.empty()
    
    if not stage_data:
        st.warning("No data found for any stage.")
        return

    # --- Plot ECG ---
    fig_ecg, ax_ecg = plt.subplots(figsize=(10, 6))
    has_ecg_data = False
    
    for stage, data in stage_data.items():
        ecg_hep = data['ecg_hep']
        times = data['times']
        
        if ecg_hep is not None:
             # ecg_hep is (1, n_times)
             n_epochs = data['n_epochs']
             ax_ecg.plot(times, ecg_hep[0] * 1e6, label=f"{stage} (n={n_epochs})", color=stage_colors.get(stage, 'gray'), linewidth=2, alpha=0.8)
             has_ecg_data = True
             
    if has_ecg_data:
        ax_ecg.set_title(f"ECG HEP across Sleep Stages - {selected_pid}")
        ax_ecg.set_xlabel("Time (s)")
        ax_ecg.set_ylabel("Amplitude (μV)")
        ax_ecg.grid(True)
        ax_ecg.legend()
        ax_ecg.axvline(0, color='black', linestyle='--', alpha=0.5)
        st.pyplot(fig_ecg, use_container_width=True)
    else:
        st.write("No ECG data available.")

    # --- Plot EEG by Regions ---
    # Determine all unique channels available (use first stage as reference, or intersection?)
    # Usually montage is constant. Let's use the first available stage.
    first_stage = next(iter(stage_data))
    all_channels = stage_data[first_stage]['ch_names']
    
    if not all_channels:
        st.warning("No EEG channels found.")
        return

    regions = ['F', 'C', 'T', 'P']
    region_map = {r: [] for r in regions}

    for ch in all_channels:
        name = ch.upper()
        # Simple categorization heuristic
        if 'F' in name:
            region_map['F'].append(ch)
        elif 'C' in name:
            region_map['C'].append(ch)
        elif 'T' in name:
            region_map['T'].append(ch)
        elif 'P' in name:
            region_map['P'].append(ch)
            
    for region in regions:
        channels = region_map[region]
        if not channels:
            continue
            
        st.subheader(f"Region: {region} ({len(channels)} channels)")
        
        # Create subplots
        n_channels = len(channels)
        n_cols = 4
        n_rows = int(np.ceil(n_channels / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows), sharex=True, sharey=True)
        if n_channels == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        # Determine symmetric limits for this region across all stages
        max_abs_val = 0
        for stage, data in stage_data.items():
            if data['eeg_hep'] is None: continue
            
            # Get indices for these channels
            # Note: ch_names in stage_data might differ if montage changed? Assuming constant for now.
            try:
                indices = [data['ch_names'].index(ch) for ch in channels if ch in data['ch_names']]
                if indices:
                    region_data = data['eeg_hep'][indices]
                    curr_max = np.nanmax(np.abs(region_data * 1e6))
                    if curr_max > max_abs_val:
                        max_abs_val = curr_max
            except:
                pass
                
        ylim = (-max_abs_val * 1.1, max_abs_val * 1.1) if max_abs_val > 0 else None

        for i, ch_name in enumerate(channels):
            ax = axes[i]
            
            for stage, data in stage_data.items():
                if data['eeg_hep'] is None: continue
                
                if ch_name in data['ch_names']:
                    ch_idx = data['ch_names'].index(ch_name)
                    hep = data['eeg_hep'][ch_idx]
                    times = data['times']
                    
                    ax.plot(times, hep * 1e6, label=stage, color=stage_colors.get(stage, 'gray'), linewidth=1.5, alpha=0.8)
            
            ax.set_title(ch_name)
            ax.grid(True, alpha=0.3)
            ax.axvline(0, color='r', linestyle='--', alpha=0.5)
            if ylim:
                ax.set_ylim(ylim)
            
            # Only legend on first plot to avoid clutter? Or external?
            if i == 0:
                ax.legend(fontsize='small')
                
        # Hide unused
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        fig.tight_layout()

def main():
    st.title("HEP Group Comparison Dashboard")
    st.write("Comparing Amplitude vs Time (Heartbeat Evoked Potential).")

    base_path = "/storage/pblab_shared_data/Nir/Cobrad/pickles_sleep_stage"

    # Select Sleep Stage
    sleep_stages = ['N1', 'N2', 'N3', 'R', 'W']
    selected_stage = st.selectbox("Select Sleep Stage", sleep_stages)
    
    # Analysis Mode Selection
    # Analysis Mode Selection
    mode = st.radio("Analysis Mode", ["Single Group Analysis", "Compare Groups", "Compare Sleep Stages"], index=0)

    if mode == "Compare Groups":
        run_compare_groups_analysis(base_path, selected_stage)
    elif mode == "Compare Sleep Stages":
        run_compare_sleep_stages_analysis(base_path)
    else: # Single Group Analysis
        run_single_group_analysis(base_path, selected_stage)

if __name__ == "__main__":
    main()