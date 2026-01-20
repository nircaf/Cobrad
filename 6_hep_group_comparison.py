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
from scipy.signal import find_peaks

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
    
    # --- Step 2: Wavelet Denoising (Conservative) ---
    if PYWT_AVAILABLE:
        # Use 'db1' (Haar) for better preservation of sharp spikes (R-waves)
        # Reduce levels so we don't smooth out the HEP components
        coeffs = pywt.wavedec(cleaned_signal, wavelet, level=wavelet_levels)
        
        # Universal threshold can be too aggressive. Use a multiplier (0.5) to keep more detail.
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(cleaned_signal))) * 0.5 
        
        # Apply threshold ONLY to high-frequency detail coefficients, not all levels
        coeffs_thresh = [coeffs[0]] # Keep Approximation
        for i in range(1, len(coeffs)):
            # Soft thresholding only on the highest frequency level
            if i == len(coeffs) - 1:
                coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
            else:
                coeffs_thresh.append(coeffs[i]) # Keep intermediate details raw
        
        cleaned_signal = pywt.waverec(coeffs_thresh, wavelet)[:len(ecg_signal)]
        methods_applied.append('targeted_wavelet_denoising')

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

def process_file_data(raw, patient_id):
    """
    Cleans ECG and extracts R-peaks using the provided logic.
    Returns the raw data, sfreq, and R-peak times if successful.
    """
    # Parameters
    minmax = (-0.5, 1.0)
    
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
    
    # Clean ECG
    ecg_signal_clean, _ = clean_ecg_high_fidelity(
        ecg_signal, 
        sampling_rate=sfreq
    )
    rpeaks = wfdb.processing.xqrs_detect(sig=ecg_signal_clean, fs=sfreq, verbose=False)

    # Refine R-peaks to be local maxima within 0.05s
    if len(rpeaks) > 0:
        window = int(0.05 * sfreq)
        refined_rpeaks = []
        for peak in rpeaks:
            start = max(0, peak - window)
            end = min(len(ecg_signal_clean), peak + window + 1)
            # Find index of max value in the window
            local_max = start + np.argmax(ecg_signal_clean[start:end])
            refined_rpeaks.append(local_max)
        rpeaks = np.array(refined_rpeaks)

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
    if WFDB_AVAILABLE:
        rpeaks_wfdb = wfdb.processing.xqrs_detect(sig=clean_segment, fs=sfreq, verbose=False)
        methods_peaks['WFDB (XQRS)'] = rpeaks_wfdb
    else:
        methods_peaks['WFDB (Missing)'] = []

    # 3. HeartPy
    if HEARTPY_AVAILABLE:
        try:
            working_data, _ = hp.process(clean_segment, sample_rate=sfreq)
            methods_peaks['HeartPy'] = working_data['peaklist']
        except:
            methods_peaks['HeartPy (Failed)'] = []
    else:
        methods_peaks['HeartPy (Missing)'] = []

    # 4. SciPy (Simple Peak Finding)
    # Use a simple height threshold based on signal stats
    height = np.percentile(clean_segment, 95)
    distance = int(0.6 * sfreq) # Assume max HR of ~100 bpm
    peaks_scipy, _ = find_peaks(clean_segment, height=height, distance=distance)
    methods_peaks['SciPy (Threshold)'] = peaks_scipy

    # 5. MNE (find_ecg_events logic)
    # MNE usually works on Raw, but we can mock it or use another NK variant
    # Let's use NK's Pan-Tompkins for variety
    _, rpk_pt = nk.ecg_peaks(clean_segment, sampling_rate=sfreq, method="pantompkins")
    methods_peaks['NeuroKit (Pan-Tompkins)'] = rpk_pt['ECG_R_Peaks']

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
    show_single_patient_all = st.checkbox("Show Single Patient All Channels", value=False)
    
    if individuals:
        if show_cleaning_comparison:
            handle_ecg_cleaning_comparison(base_path, selected_group, selected_stage)

        if show_single_patient_all:
            patient_ids = [ind[0] for ind in individuals]
            selected_pid = st.selectbox("Select Patient for All Channels", patient_ids)
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
                    
                    for i, (ch_name, ch_data) in enumerate(zip(ch_names, full_hep)):
                        ax = axes[i]
                        ax.plot(times, ch_data * 1e6)
                        ax.set_title(ch_name)
                        ax.grid(True)
                        ax.axvline(0, color='r', linestyle='--', alpha=0.5)
                        
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
                    break

        if show_ecg_only:
            plot_ecg_hep_individuals(individuals, selected_group, selected_stage)

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

def main():
    st.title("HEP Group Comparison Dashboard")
    st.write("Comparing Amplitude vs Time (Heartbeat Evoked Potential).")

    base_path = "/storage/pblab_shared_data/Nir/Cobrad/pickles_sleep_stage"

    # Select Sleep Stage
    sleep_stages = ['N1', 'N2', 'N3', 'R', 'W']
    selected_stage = st.selectbox("Select Sleep Stage", sleep_stages)
    
    # Analysis Mode Selection
    mode = st.radio("Analysis Mode", ["Single Group Analysis","Compare Groups"])

    if mode == "Compare Groups":
        run_compare_groups_analysis(base_path, selected_stage)
    else: # Single Group Analysis
        run_single_group_analysis(base_path, selected_stage)

if __name__ == "__main__":
    main()