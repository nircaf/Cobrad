import os
import re
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mne
import neurokit2 as nk
from scipy.signal import butter, filtfilt
from scipy.stats import wilcoxon, zscore, pearsonr, entropy, ranksums, linregress
from statsmodels.stats.multitest import fdrcorrection

from sklearn.feature_selection import mutual_info_regression
from mne_connectivity import SpectralConnectivity as spectral_connectivity

# Import for EEG cleaning pipeline
from autoreject import AutoReject
from pyprep.prep_pipeline import PrepPipeline
from contextlib import contextmanager
import sys
import os
AUTOREJECT_AVAILABLE = True

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

try:
    import streamlit as st
    is_streamlit = True
except ImportError:
    is_streamlit = False

# Project utils: expects power_bands and compute_network_features at least.
# sys append mother folder
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.eeg_utils import *

# ------------------------------------------------------------------------------
# Global settings
# ------------------------------------------------------------------------------
SAVE_DIR = "figures_HEP/compute_brain_heart_coupling"
TEMPS_DIR = "parquets_HEP"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(TEMPS_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def is_patient_processed(patient_id, temps_dir=TEMPS_DIR):
    """
    Check if a patient was already processed by looking for their parquet files.
    
    Args:
        patient_id (str): Patient ID to check
        temps_dir (str): Directory containing the parquet files
        
    Returns:
        bool: True if patient was already processed, False otherwise
    """
    # Expected bands based on the code
    expected_bands = ['alpha', 'beta', 'delta', 'gamma', 'theta']
    
    for band in expected_bands:
        parquet_file = os.path.join(temps_dir, f"{patient_id}_results_{band}.parquet")
        if not os.path.exists(parquet_file):
            return False
    
    return True

def get_processed_patients(temps_dir=TEMPS_DIR):
    """
    Get a list of all patients that have been processed (have all parquet files).
    
    Args:
        temps_dir (str): Directory containing the parquet files
        
    Returns:
        list: List of patient IDs that have been processed
    """
    if not os.path.exists(temps_dir):
        return []
    
    # Get all parquet files in the directory
    parquet_files = glob.glob(os.path.join(temps_dir, "*_results_*.parquet"))
    
    # Extract patient IDs from filenames
    patient_ids = set()
    for file_path in parquet_files:
        filename = os.path.basename(file_path)
        # Extract patient ID from filename like "0345-010_results_alpha.parquet"
        match = re.match(r'^(.+)_results_\w+\.parquet$', filename)
        if match:
            patient_ids.add(match.group(1))
    
    # Filter to only include patients with all expected bands
    processed_patients = []
    for patient_id in patient_ids:
        if is_patient_processed(patient_id, temps_dir):
            processed_patients.append(patient_id)
    
    return sorted(processed_patients)
def clean_ecg_signal(ecg_signal, sfreq, lowcut=0.5, highcut=40, order=4):
    nyq = 0.5 * sfreq
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, ecg_signal)

def _interpolate_nans_1d(arr):
    arr = np.asarray(arr, dtype=float)
    if not np.isnan(arr).any():
        return arr
    x = np.arange(len(arr))
    m = ~np.isnan(arr)
    if m.sum() == 0:
        return np.zeros_like(arr)
    arr_interp = np.copy(arr)
    arr_interp[~m] = np.interp(x[~m], x[m], arr[m])
    return arr_interp

def _ensure_finite_1d(arr):
    arr = np.asarray(arr, dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    arr = _interpolate_nans_1d(arr)
    if np.isnan(arr).any() or (~np.isfinite(arr)).any():
        mean_val = np.nanmean(arr)
        if not np.isfinite(mean_val):
            mean_val = 0.0
        arr = np.nan_to_num(arr, nan=mean_val, posinf=mean_val, neginf=mean_val)
    return arr

def plot_ecg_signal(ecg_signal, sfreq, name, save_plot, save_dir, label, bool_plots=True):
    if not bool_plots:
        return
    os.makedirs(save_dir, exist_ok=True)
    # Full signal
    plt.figure(figsize=(10, 4))
    plt.plot(ecg_signal)
    plt.title(f'ECG Signal {label}')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    if save_plot:
        fname = f"{save_dir}/{name}_ecg_signal_{label}.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

    # 60-120s segment
    seg_start_sec = 60
    seg_end_sec = 120
    seg_start = int(seg_start_sec * sfreq)
    seg_end = int(seg_end_sec * sfreq)
    if len(ecg_signal) > seg_end:
        plt.figure(figsize=(10, 4))
        t = np.arange(seg_start, seg_end) / sfreq
        plt.plot(t, ecg_signal[seg_start:seg_end])
        plt.title(f'ECG Segment {seg_start_sec}-{seg_end_sec}s {label}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        if save_plot:
            fname = f"{save_dir}/{name}_ecg_signal_{seg_start_sec}to{seg_end_sec}s_{label}.png"
            plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

def filter_by_ecg_quality(ecg_clean, data_all, ecg_quality, threshold=0.5):
    mask = ecg_quality > threshold
    ecg_clean_interp = np.copy(ecg_clean)
    if not np.all(mask):
        x = np.arange(len(ecg_quality))
        good = mask
        bad = ~mask
        ecg_clean_interp[bad] = np.interp(x[bad], x[good], ecg_clean[good])
    return ecg_clean_interp, data_all

def joint_entropy(x, y, bins=50):
    c_xy = np.histogram2d(x, y, bins)[0]
    c_xy = c_xy / np.sum(c_xy)
    c_xy = c_xy[c_xy > 0]
    return entropy(c_xy, base=2)



# ------------------------------------------------------------------------------
# Core computation with data_all list of Raw objects
# ------------------------------------------------------------------------------
def compute_brain_heart_coupling(data_all, patient_id, bool_plots=False, save_plot=False, step_sec=5):
    """
    Compute time-varying EEG network metrics and HRV indices, then their coupling,
    across a list of MNE Raw objects (data_all). Results are averaged across the list.

    Parameters
    ----------
    data_all : list[mne.io.Raw]
        List of MNE Raw objects to aggregate (e.g., 6 EDFs for a patient).
    patient_id : str
        Patient identifier for naming/saving.
    bool_plots : bool
        Whether to show plots.
    save_plot : bool
        Whether to save plots.
    step_sec : int
        Step size (in seconds) for sliding window.

    Returns
    -------
    dict[str, pd.DataFrame]
        Averaged DataFrame per band across all raws.
    """
    band_to_dfs = {band: [] for band in power_bands}

    for i, raw in enumerate(data_all):
        print(f"[{patient_id}] Processing EDF {i+1}/{len(data_all)}")
        
        # Clean the EEG data using the comprehensive pipeline
        print(f"[{patient_id}] Cleaning EEG data for EDF {i+1}")
        # raw_clean = clean_eeg_data(raw)
        
        window_data = raw.get_data()
        ch_names = raw.ch_names
        sfreq = int(raw.info['sfreq'])
        print(f"Sampling frequency after cleaning: {sfreq}")
        name_prefix = f"{patient_id}_edf{i+1}"

        # Extract ECG and detect R-peaks
        ch_lower = [ch.lower() for ch in ch_names]
        ecg_indices = [i for i, ch in enumerate(ch_lower) if 'ecg' in ch or 'ekg' in ch]
        if not ecg_indices:
            print(f"[{patient_id}] EDF {i+1} has no ECG channel; skipping.")
            continue

        ecg_idx = ecg_indices[0]  # Use the first ECG channel found
        ecg_signal = np.asarray(window_data[ecg_idx], dtype=float)
        ecg_signal = _ensure_finite_1d(_interpolate_nans_1d(ecg_signal))
        plot_ecg_signal(ecg_signal, sfreq, name_prefix, save_plot, SAVE_DIR, label="raw", bool_plots=bool_plots)

        ecg_signal_filt = clean_ecg_signal(ecg_signal, sfreq, lowcut=0.5, highcut=40, order=4)
        ecg_signal_filt = _ensure_finite_1d(_interpolate_nans_1d(ecg_signal_filt))
        plot_ecg_signal(ecg_signal_filt, sfreq, name_prefix, save_plot, SAVE_DIR, label="bandpass_filtered", bool_plots=bool_plots)

        try:
            signals, info = nk.ecg_process(ecg_signal_filt, sampling_rate=sfreq)
        except Exception as e:
            print(f"Warning: nk.ecg_process failed ({e}). Falling back to nk.ecg_clean.")
            ecg_clean_fallback = nk.ecg_clean(ecg_signal_filt, sampling_rate=sfreq, method='neurokit')
            signals = pd.DataFrame({
                'ECG_Clean': ecg_clean_fallback,
                'ECG_Quality': np.ones_like(ecg_clean_fallback, dtype=float)
            })
            info = {'method': 'fallback'}

        ecg_clean = signals['ECG_Clean'].values
        plot_ecg_signal(ecg_clean, sfreq, name_prefix, save_plot, SAVE_DIR, label="cleaned", bool_plots=bool_plots)
        ecg_clean, window_data = filter_by_ecg_quality(ecg_clean, window_data, signals['ECG_Quality'].values, threshold=0.5)
        plot_ecg_signal(ecg_clean, sfreq, name_prefix, save_plot, SAVE_DIR, label="quality_cleaned_filtered", bool_plots=bool_plots)

        _, rpk = nk.ecg_peaks(ecg_clean, sampling_rate=sfreq)
        rpeaks = rpk['ECG_R_Peaks']
        r_times = rpeaks / sfreq

        # EEG channels subset
        eeg_indices = [ch_names.index(ch) for ch in eeg_channels if ch in ch_names]
        if len(eeg_indices) == 0:
            print(f"[{patient_id}] EDF {i+1} has no expected EEG channels; skipping.")
            continue
        data = window_data[eeg_indices]
        n_nodes, n_samples = data.shape
        print(f"EEG data shape: {data.shape}")

        # Sliding window parameters
        w_eeg_sec = 15
        w_eeg = int(w_eeg_sec * sfreq)
        step = int(step_sec * sfreq)
        n_windows = int((n_samples - w_eeg) / step) + 1
        n_windows = max(n_windows, 0)
        print(f"Number of windows: {n_windows}, Window size: {w_eeg}, Step: {step}. Sampled duration: {n_samples/sfreq/60:.1f}m")

        eff_ts = {band: [] for band in power_bands}
        clu_ts = {band: [] for band in power_bands}
        mod_ts = {band: [] for band in power_bands}
        ass_ts = {band: [] for band in power_bands}
        cvi_ts, csi_ts = [], []

        for w in range(n_windows):
            start = w * step
            end = start + w_eeg
            segment = data[:, start:end]
            # Compute features per band
            band_features = {}
            for band_name, (fmin, fmax) in power_bands.items():
                eff, clu, mod, ass = compute_network_features(segment, sfreq, [fmin, fmax])
                band_features[band_name] = (eff, clu, mod, ass)
            # Store features
            for band_name, (eff, clu, mod, ass) in band_features.items():
                eff_ts[band_name].append(eff)
                clu_ts[band_name].append(clu)
                mod_ts[band_name].append(mod)
                ass_ts[band_name].append(ass)

            # HRV within same window (SD1/SD2 proxies)
            t0, t1 = start / sfreq, end / sfreq
            idx = np.where((r_times >= t0) & (r_times <= t1))[0]
            if len(idx) > 2:
                ibi = np.diff(rpeaks[idx]) / sfreq
                dibi = np.diff(ibi)
                sd1 = np.std(dibi) / np.sqrt(2)
                sd2 = np.sqrt(max(0, 2 * np.std(ibi)**2 - 0.5 * np.std(dibi)**2))
            else:
                sd1 = np.nan; sd2 = np.nan
            cvi_ts.append(sd1)
            csi_ts.append(sd2)

        # Convert to arrays and build per-band DataFrames for this EDF
        cvi_arr = np.array(cvi_ts)
        csi_arr = np.array(csi_ts)

        for band in power_bands:
            eff_arr = np.array(eff_ts[band])
            clu_arr = np.array(clu_ts[band])
            mod_arr = np.array(mod_ts[band])
            ass_arr = np.array(ass_ts[band])

            valid = ~np.isnan(cvi_arr) & ~np.isnan(csi_arr)
            if valid.sum() == 0:
                print(f"[{patient_id}] EDF {i+1} band {band}: no valid samples after NaN filtering.")
                continue
            eff_arr, clu_arr = eff_arr[valid], clu_arr[valid]
            mod_arr, ass_arr = mod_arr[valid], ass_arr[valid]
            cvi_arr_band, csi_arr_band = cvi_arr[valid], csi_arr[valid]

            # Mutual information placeholders (set to NaN to avoid heavy compute; uncomment if needed)
            mi_results = {}
            metrics = {'Efficiency': eff_arr, 'Clustering': clu_arr, 'Modularity': mod_arr, 'Assortativity': ass_arr}
            for name, arr in metrics.items():
                X = arr.reshape(-1, 1)
                mic_sym = np.nan
                mic_vag = np.nan
                # mic_sym = mutual_info_regression(X, csi_arr_band, random_state=0)[0]
                # mic_vag = mutual_info_regression(X, cvi_arr_band, random_state=0)[0]
                mi_results[name] = {'Sympathetic MI': mic_sym, 'Vagal MI': mic_vag}

            results_df = pd.DataFrame({
                'Efficiency': eff_arr,
                'Clustering': clu_arr,
                'Modularity': mod_arr,
                'Assortativity': ass_arr,
                'Vagal_SD1': cvi_arr_band,
                'Sympathetic_SD2': csi_arr_band
            })
            results_df.attrs['mutual_info'] = mi_results

            band_to_dfs[band].append(results_df)


    # Average across all EDFs in data_all per band
    results_df_dict_avg = {}
    for band, dfs in band_to_dfs.items():
        if len(dfs) == 0:
            results_df_dict_avg[band] = pd.DataFrame()
            continue

        all_columns = set()
        for df in dfs:
            all_columns.update(df.columns)
        all_columns = list(all_columns)

        max_len = max(len(df) for df in dfs)
        dfs_aligned = []
        for df in dfs:
            df_aligned = df.reindex(columns=all_columns)
            df_aligned = df_aligned.reindex(range(max_len)).reset_index(drop=True)
            dfs_aligned.append(df_aligned)

        arrs = np.stack([d.values for d in dfs_aligned], axis=0)
        n_files = len(dfs)
        n_rows, n_cols = arrs.shape[1], arrs.shape[2]
        mean_arr = np.full((n_rows, n_cols), np.nan)
        for i in range(n_rows):
            for j in range(n_cols):
                vals = arrs[:, i, j]
                n_not_nan = np.sum(~np.isnan(vals))
                if n_not_nan >= n_files / 2:
                    mean_arr[i, j] = np.nanmean(vals)
                else:
                    mean_arr[i, j] = np.nan
        mean_df = pd.DataFrame(mean_arr, columns=all_columns)
        mean_df = mean_df.dropna(how="all")
        results_df_dict_avg[band] = mean_df

    return results_df_dict_avg

# ------------------------------------------------------------------------------
# Aggregation and plotting helpers
# ------------------------------------------------------------------------------
def plot_patient_band_means(patient_id, bands=None, step_sec=5, temps_dir=TEMPS_DIR, save_dir=SAVE_DIR):
    if bands is None:
        bands = list(power_bands.keys())
    for band in bands:
        pattern = os.path.join(temps_dir, f"*{patient_id}*_results_{band}.csv")
        file_list = glob.glob(pattern)
        if not file_list:
            print(f"No files found for patient {patient_id}, band {band}")
            continue
        dfs = []
        for f in file_list:
            try:
                df = pd.read_csv(f)
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {f}: {e}")
        if not dfs:
            print(f"No valid DataFrames for patient {patient_id}, band {band}")
            continue

        all_columns = set()
        for df in dfs:
            all_columns.update(df.columns)
        all_columns = list(all_columns)
        max_len = max(len(df) for df in dfs)
        dfs_aligned = []
        for df in dfs:
            df_aligned = df.reindex(columns=all_columns)
            df_aligned = df_aligned.reindex(range(max_len)).reset_index(drop=True)
            dfs_aligned.append(df_aligned)

        arrs = np.stack([d.values for d in dfs_aligned], axis=0)
        n_files = len(dfs)
        n_rows, n_cols = arrs.shape[1], arrs.shape[2]
        mean_arr = np.full((n_rows, n_cols), np.nan)
        for i in range(n_rows):
            for j in range(n_cols):
                vals = arrs[:, i, j]
                n_not_nan = np.sum(~np.isnan(vals))
                if n_not_nan >= n_files / 2:
                    mean_arr[i, j] = np.nanmean(vals)
                else:
                    mean_arr[i, j] = np.nan
        mean_df = pd.DataFrame(mean_arr, columns=all_columns)
        mean_df = mean_df.dropna(how="all")
        only_plots(mean_df, save_plot=True, save_dir=save_dir, edf_pickle_name=patient_id, band=band, step_sec=step_sec)
        print(f"Plotted mean for patient {patient_id}, band {band}")

def plot_all_patients_band_means(bands=None, step_sec=5, temps_dir=TEMPS_DIR, save_dir=SAVE_DIR):
    if bands is None:
        bands = list(power_bands.keys())
    for band in bands:
        pattern = os.path.join(temps_dir, f"*results_{band}.csv")
        file_list = glob.glob(pattern)
        if not file_list:
            print(f"No files found for band {band} (all patients)")
            continue
        dfs = []
        for f in file_list:
            try:
                df = pd.read_csv(f)
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {f}: {e}")
        if not dfs:
            print(f"No valid DataFrames for band {band} (all patients)")
            continue

        all_columns = set()
        for df in dfs:
            all_columns.update(df.columns)
        all_columns = list(all_columns)
        max_len = max(len(df) for df in dfs)
        dfs_aligned = []
        for df in dfs:
            df_aligned = df.reindex(columns=all_columns)
            df_aligned = df_aligned.reindex(range(max_len)).reset_index(drop=True)
            dfs_aligned.append(df_aligned)

        arrs = np.stack([d.values for d in dfs_aligned], axis=0)
        n_files = len(dfs)
        n_rows, n_cols = arrs.shape[1], arrs.shape[2]
        mean_arr = np.full((n_rows, n_cols), np.nan)
        for i in range(n_rows):
            for j in range(n_cols):
                vals = arrs[:, i, j]
                n_not_nan = np.sum(~np.isnan(vals))
                if n_not_nan >= n_files / 2:
                    mean_arr[i, j] = np.nanmean(vals)
                else:
                    mean_arr[i, j] = np.nan
        mean_df = pd.DataFrame(mean_arr, columns=all_columns)
        mean_df = mean_df.dropna(how="all")
        only_plots(mean_df, save_plot=True, save_dir=save_dir, edf_pickle_name="ALL_PATIENTS", band=band, step_sec=step_sec)
        print(f"Plotted mean for ALL_PATIENTS, band {band}")

# ------------------------------------------------------------------------------
# EDF discovery and RAM-conscious processing
# ------------------------------------------------------------------------------
def group_edf_files_by_patient(edf_root="pickles/EDF"):
    """
    Walk pickle directory and group .pkl file paths by patient_id extracted via regex (\d{4}-\d{3}).
    """
    patient_to_files = {}
    for root, dirs, files in os.walk(edf_root):
        for file in files:
            if not file.lower().endswith(".pkl"):
                continue
            fpath = os.path.join(root, file)
            m = re.search(r'(\d{4}-\d{3})', fpath)
            if m:
                pid = m.group(1)
            else:
                # try parent directory name as fallback
                pid = file.split('.')[0]
            patient_to_files.setdefault(pid, []).append(fpath)

    return patient_to_files

def group_edf_files_by_patient_edf(edf_root="EDF_Format/EDF"):
    """
    Walk EDF directory and group .edf file paths by patient_id extracted via regex (\d{4}-\d{3}).
    """
    patient_to_files = {}
    for root, dirs, files in os.walk(edf_root):
        for file in files:
            if not file.lower().endswith(".edf"):
                continue
            fpath = os.path.join(root, file)
            m = re.search(r'(\d{4}-\d{3})', fpath)
            if m:
                pid = m.group(1)
            else:
                # try parent directory name as fallback
                pid = file.split('.')[0]
            patient_to_files.setdefault(pid, []).append(fpath)

    return patient_to_files

def edf_has_ecg(edf_path):
    """
    Check whether a pickled MNE Raw object (.pkl) contains an ECG channel.
    """
    import sys, mne.io.array
    sys.modules['mne.io.array.array'] = mne.io.array
    try:
        with open(edf_path, 'rb') as f:
            raw = pickle.load(f)
        if hasattr(raw, 'ch_names'):
            ch_names = raw.ch_names
        elif hasattr(raw, 'info') and 'ch_names' in raw.info:
            ch_names = raw.info['ch_names']
        else:
            ch_names = []
        ch_lower = [ch.lower() for ch in ch_names]
        if any('ecg' in ch or 'ekg' in ch for ch in ch_lower):
            return True
        else:
            ecg_channels, channel_metrics = detect_ecg_channels_from_data(raw)
            if len(ecg_channels) > 0:
                return True
            else:
                return False
    except Exception as e:
        print(f"Failed to read pickle {edf_path}: {e}")
        return False

def detect_ecg_channels_from_data(raw, sfreq=None):
    """
    Analyze MNE Raw EEG data to detect ECG channels based on signal characteristics.
    
    Parameters:
    -----------
    raw : mne.io.Raw
        MNE Raw object containing EEG data
    sfreq : float, optional
        Sampling frequency. If None, will be extracted from raw.info['sfreq']
    
    Returns:
    --------
    list : List of channel names that are detected as ECG channels
    dict : Dictionary with detailed metrics for each channel
    
    Criteria for ECG detection:
    - Dominant frequency: ~1 Hz (ECG) vs 4-12 Hz (EEG)
    - Kurtosis: High (ECG) vs Low-moderate (EEG)  
    - Amplitude: 0.2-2 mV (ECG) vs 10-100 µV (EEG)
    - Channel must meet ALL criteria to be classified as ECG
    """
    import sys, mne.io.array
    sys.modules['mne.io.array.array'] = mne.io.array
    # clean data; resample 256, notch 50, bandpass 0.5-40
    sf = 256
    raw.resample(sf)
    # notch every 10hz
    raw.notch_filter(np.arange(10, sf/2, 10))
    raw.filter(0.5, 40)
    # Get channel names and data
    ch_names = raw.ch_names
    data = raw.get_data()  # Shape: (n_channels, n_times)
    
    ecg_channels = []
    channel_metrics = {}
    
    for i, ch_name in enumerate(ch_names):
        signal = data[i, :]
        # Calculate metrics
        metrics = {}
        
        # 1. Dominant frequency analysis
        from scipy.fft import fft, fftfreq
        freqs = fftfreq(len(signal), 1/sf)
        fft_vals = np.abs(fft(signal))
        
        # Find dominant frequency (excluding DC component)
        freqs_positive = freqs[1:len(freqs)//2]
        fft_positive = fft_vals[1:len(fft_vals)//2]
        dominant_freq_idx = np.argmax(fft_positive)
        dominant_freq = freqs_positive[dominant_freq_idx]
        metrics['dominant_frequency'] = dominant_freq
        
        # 2. Kurtosis (measure of tail heaviness)
        from scipy.stats import kurtosis
        kurt = kurtosis(signal)
        metrics['kurtosis'] = kurt
        
        # 3. Amplitude (peak-to-peak amplitude)
        amplitude_pp = np.max(signal) - np.min(signal)
        # Convert to mV if needed (assuming data is in V)
        amplitude_mv = amplitude_pp * 1000  # Convert V to mV
        metrics['amplitude_mv'] = amplitude_mv
        
        # 4. Additional ECG-specific metrics
        # Heart rate variability (if we can detect peaks)
        try:
            # Simple peak detection for heart rate estimation
            from scipy.signal import find_peaks
            # Filter for typical ECG frequency range
            from scipy.signal import butter, filtfilt
            nyquist = sf / 2
            low = 0.5 / nyquist
            high = 5.0 / nyquist
            b, a = butter(4, [low, high], btype='band')
            filtered_signal = filtfilt(b, a, signal)
            
            # Find peaks
            peaks, _ = find_peaks(filtered_signal, distance=int(sf * 0.3))  # Min 0.3s between peaks
            
            if len(peaks) > 1:
                # Calculate heart rate
                rr_intervals = np.diff(peaks) / sf  # in seconds
                heart_rate = 60 / np.mean(rr_intervals)  # BPM
                metrics['heart_rate_bpm'] = heart_rate
                metrics['peak_count'] = len(peaks)
            else:
                metrics['heart_rate_bpm'] = None
                metrics['peak_count'] = 0
                
        except Exception as e:
            metrics['heart_rate_bpm'] = None
            metrics['peak_count'] = 0
        
        channel_metrics[ch_name] = metrics
        
        # Apply ECG detection criteria
        is_ecg = True
        
        # Criterion 1: Dominant frequency around 1 Hz (ECG) vs 4-12 Hz (EEG)
        # ECG should have dominant frequency close to heart rate (~1 Hz)
        if not (0.8 <= dominant_freq <= 3.0):  # Allow some tolerance around 1 Hz
            is_ecg = False
        
        # Criterion 2: High kurtosis (ECG has sharp peaks)
        # ECG typically has higher kurtosis due to sharp QRS complexes
        if kurt < 4:  # Threshold for high kurtosis
            is_ecg = False
        
        # Criterion 3: Amplitude in ECG range (0.2-2 mV)
        # Convert criteria to same units (mV)
        if not (0.2 <= amplitude_mv <= 2.0):
            is_ecg = False
        
        # Additional criterion: Should have detectable heart rate
        if metrics['heart_rate_bpm'] is not None:
            if not (40 <= metrics['heart_rate_bpm'] <= 200):  # Reasonable heart rate range
                is_ecg = False
        else:
            is_ecg = False
        
        if is_ecg:
            ecg_channels.append(ch_name)
    
    return ecg_channels, channel_metrics
    # plot raw 


def select_random_edfs_with_ecg(file_list, k=6, seed=42):
    rng = np.random.default_rng(seed)
    eligible = []
    # Check ECG presence without preloading data
    for fp in file_list:
        if edf_has_ecg(fp):
            eligible.append(fp)
    if len(eligible) == 0:
        return []
    if len(eligible) <= k:
        return eligible
    idx = rng.choice(len(eligible), size=k, replace=False)
    return [eligible[i] for i in idx]

def load_raws(file_list):
    """
    Load pickled MNE Raw objects from .pkl files.
    """
    raws = []
    import sys, mne.io.array
    sys.modules['mne.io.array.array'] = mne.io.array
    for fp in file_list:
        try:
            print(f"Loading PKL: {fp}")
            with open(fp, 'rb') as f:
                raw = pickle.load(f)
            raws.append(raw)
        except Exception as e:
            print(f"Error loading {fp}: {e}")
    return raws

def process_patients_random6(edf_root="EDF", k=6, step_sec=5, seed=42):
    """
    For each patient_id:
      - find all their EDF files,
      - randomly select up to k EDFs that contain an ECG channel,
      - load those k files (only), and
      - run compute_brain_heart_coupling with data_all as that list of Raw objects.
    """
    # get form edf_root the project name the is after the last /
    project_name = edf_root.split('/')[-1]
    # get TEMPS_DIR from globals
    TEMPS_DIR = globals()['TEMPS_DIR']
    TEMPS_DIR = os.path.join(TEMPS_DIR, project_name)
    patient_to_files = group_edf_files_by_patient(edf_root=edf_root)
    if not patient_to_files:
        print(f"No EDF files found under {edf_root}.")
        return
    
    # Check which patients are already processed
    processed_patients = get_processed_patients(temps_dir=TEMPS_DIR)
    if processed_patients:
        print(f"Already processed patients: {', '.join(processed_patients)}")
    else:
        print("No patients have been processed yet.")

    for patient_id, files in sorted(patient_to_files.items()):
        print(f"Patient {patient_id}: {len(files)} PKL files found.")
        
        # Check if patient was already processed
        if is_patient_processed(patient_id, temps_dir=TEMPS_DIR):
            print(f"Patient {patient_id}: already processed (parquet files exist). Skipping.")
            continue
        
        selected = select_random_edfs_with_ecg(files, k=k, seed=seed)
        if len(selected) == 0:
            print(f"Patient {patient_id}: no PKLs with ECG channel. Skipping.")
            continue
        print(f"Patient {patient_id}: selected {len(selected)} PKLs for processing.")
        raws = load_raws(selected)
        if len(raws) == 0:
            print(f"Patient {patient_id}: failed to load selected PKLs. Skipping.")
            continue

        results_df_dict = compute_brain_heart_coupling(
            data_all=raws,
            patient_id=patient_id,
            bool_plots=False,
            save_plot=True,
            step_sec=step_sec
        )

        # Save averaged results per band for this patient
        for band, results_df in results_df_dict.items():
            results_path = os.path.join(TEMPS_DIR, f"{patient_id}_results_{band}.parquet")
            print(f"Saving results for patient {patient_id}, band {band} to {results_path}")
            # create the directory if it doesn't exist
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            results_df.to_parquet(results_path, index=False)

    # Aggregate across all patients
    plot_all_patients_band_means(bands=power_bands.keys(), step_sec=step_sec, temps_dir=TEMPS_DIR, save_dir=SAVE_DIR)
    print("All processing and plotting complete.")

def process_patients_n1_stage_hep(edf_root="EDF_Format/EDF", step_sec=5, n1_duration_min=1, stage='N1'):
    """
    For each patient_id:
      - find all their EDF files from EDF_Format/EDF directory,
      - load each EDF file using mne.io.read_raw_edf(),
      - apply clean_mne_raw() to clean the data,
      - use YASA to detect sleep stages,
      - find specified sleep stage segments (N1, N2, N3, R, W),
      - crop the MNE raw data to the selected stage segments,
      - run compute_brain_heart_coupling on stage segments,
      - save results to parquets_HEP with stage-specific naming.
    
    Parameters
    ----------
    edf_root : str
        Root directory containing EDF files (default: EDF_Format/EDF)
    step_sec : int
        Step size in seconds for sliding window analysis
    n1_duration_min : int
        Duration of stage segments to extract in minutes (default: 1)
    stage : str
        Sleep stage to extract (N1, N2, N3, R, or W) (default: N1)
    """
    import yasa
    import mne
    import importlib.util
    spec = importlib.util.spec_from_file_location("edf_cleaning", "1_edf_cleaning.py")
    edf_cleaning = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(edf_cleaning)
    clean_mne_raw = edf_cleaning.clean_mne_raw
    
    # Get project name from edf_root
    project_name = edf_root.split('/')[-1]
    # Get TEMPS_DIR from globals
    TEMPS_DIR = globals()['TEMPS_DIR']
    TEMPS_DIR = os.path.join(TEMPS_DIR, f"{project_name}_{stage}")
    
    # Find EDF files instead of pickle files
    patient_to_files = group_edf_files_by_patient_edf(edf_root=edf_root)
    if not patient_to_files:
        print(f"No EDF files found under {edf_root}.")
        return
    
    # Check which patients are already processed
    processed_patients = get_processed_patients(temps_dir=TEMPS_DIR)
    if processed_patients:
        print(f"Already processed patients: {', '.join(processed_patients)}")
    else:
        print("No patients have been processed yet.")

    # patient_to_files flip order
    patient_to_files = dict(sorted(patient_to_files.items(), key=lambda x: x[0], reverse=True))
    for patient_id, files in patient_to_files.items():
        print(f"Patient {patient_id}: {len(files)} EDF files found.")
        
        # Check if patient was already processed
        if is_patient_processed(patient_id, temps_dir=TEMPS_DIR):
            print(f"Patient {patient_id}: already processed (parquet files exist). Skipping.")
            continue
        
        # Process each file to find stage segments
        # flip order of files
        stage_segments = []
        for file_path in files:
            print(f"Processing {file_path} for {stage} sleep stages...")
            
            try:
                # Load the EDF file using mne
                raw = mne.io.read_raw_edf(file_path, preload=True)
                # do notch 50 and low pass high pass
                raw.notch_filter(np.arange(50, raw.info['sfreq']/2, 50))
                raw.filter(l_freq=0.5, h_freq=raw.info['sfreq']/2 - 0.1)
                # raw is in V. change to microV
                # raw._data *= 1e6  # Convert from V to µV in-place at the numpy array level
                # resample to 256 Hz
                raw.resample(256)
                raw = rename_channels(raw)
                # Check if file has ECG channel
                ch_lower = [ch.lower() for ch in raw.ch_names]
                ecg_indices = [i for i, ch in enumerate(ch_lower) if 'ecg' in ch or 'ekg' in ch]
                if not ecg_indices:
                    print(f"No ECG channel found in {file_path}. Skipping.")
                    continue
                
                # Use YASA for sleep staging
                # Find the best channels for sleep staging
                available_eeg_channels = [ch for ch in raw.ch_names if ch in eeg_channels]
                if not available_eeg_channels:
                    print(f"No suitable EEG channels found in {file_path}. Skipping.")
                    continue
                
                # Select optimal EEG channel (prefer central electrodes like C3, C4, Cz)
                central_eeg_channels = ['Cz','C3', 'C4']
                eeg_name = None
                for ch in central_eeg_channels:
                    if ch in available_eeg_channels:
                        eeg_name = ch
                        break
                
                # If no central electrode, use the first available EEG channel
                if eeg_name is None:
                    eeg_name = available_eeg_channels[0]
                
                print(f"Using EEG channel: {eeg_name}")
                
                # Find EOG channel
                eog_name = None
                if 'EOG+' in raw.ch_names:
                    eog_name = 'EOG+'
                    print(f"Using EOG channel: {eog_name}")
                else:
                    print("No EOG channel found")
                
                # Find EMG channel (prefer EMG1+ over EMG2+)
                emg_name = None
                if 'EMG1+' in raw.ch_names:
                    emg_name = 'EMG1+'
                    print(f"Using EMG channel: {emg_name}")
                elif 'EMG2+' in raw.ch_names:
                    emg_name = 'EMG2+'
                    print(f"Using EMG channel: {emg_name}")
                else:
                    print("No EMG channel found")
                
                # Run YASA sleep staging on the raw (uncleaned) data with all available channels
                print(f"Running YASA sleep staging on {file_path}...")
                ss = yasa.SleepStaging(raw, eeg_name=eeg_name, eog_name=eog_name, emg_name=emg_name)
                predicted_stages = ss.predict()
                
                # Find specified sleep stage epochs (N1, N2, N3, R, or W)
                stage_epochs = np.where(predicted_stages == stage)[0]

                if len(stage_epochs) == 0:
                    print(f"No {stage} sleep stages found in {file_path}. Skipping.")
                    continue
                
                # Find the longest continuous streak of stage epochs (allowing gaps of up to 3 epochs)
                def find_longest_stage_streak(stage_epochs, max_gap=3):
                    """Find the longest continuous streak of stage epochs, allowing gaps of up to max_gap epochs."""
                    if len(stage_epochs) == 0:
                        return None, 0
                    
                    longest_start = None
                    longest_length = 0
                    current_start = stage_epochs[0]
                    current_length = 1
                    current_end = stage_epochs[0]
                    
                    for i in range(1, len(stage_epochs)):
                        gap = stage_epochs[i] - current_end
                        
                        if gap <= max_gap + 1:  # Allow gap of up to max_gap epochs
                            # Still part of the same streak
                            current_length += 1
                            current_end = stage_epochs[i]
                        else:
                            # Gap too large, check if this is the longest streak
                            if current_length > longest_length:
                                longest_start = current_start
                                longest_length = current_length
                            
                            # Start new streak
                            current_start = stage_epochs[i]
                            current_length = 1
                            current_end = stage_epochs[i]
                    
                    # Check the last streak
                    if current_length > longest_length:
                        longest_start = current_start
                        longest_length = current_length
                    
                    return longest_start, longest_length
                
                # Find the longest stage streak
                longest_start, longest_length = find_longest_stage_streak(stage_epochs)
                
                if longest_start is None or longest_length == 0:
                    print(f"No continuous {stage} epochs found in {file_path}. Skipping.")
                    continue
                
                # YASA uses 30-second epochs by default
                epoch_duration_sec = 30
                min_stage_epochs = int(n1_duration_min * 60 / epoch_duration_sec)
                
                # Check if the longest streak meets the minimum duration requirement
                if longest_length >= min_stage_epochs:
                    segment_start_sec = longest_start * epoch_duration_sec
                    segment_end_sec = (longest_start + longest_length) * epoch_duration_sec
                    stage_segments_in_file = [(segment_start_sec, segment_end_sec)]
                    print(f"Found longest {stage} streak: {longest_length} epochs ({longest_length * epoch_duration_sec / 60:.1f} minutes) in {file_path}")
                else:
                    print(f"Longest {stage} streak ({longest_length} epochs, {longest_length * epoch_duration_sec / 60:.1f} minutes) is shorter than required {n1_duration_min} minutes in {file_path}. Skipping.")
                    continue
                
                # Crop the raw data to stage segments, clean them, and add to list
                for start_sec, end_sec in stage_segments_in_file:
                    try:
                        # Crop the raw data to the stage segment
                        raw_stage = raw.copy().crop(tmin=start_sec, tmax=end_sec)
                        
                        # Apply cleaning to the stage segment
                        print(f"Cleaning {stage} segment: {start_sec:.1f}s - {end_sec:.1f}s ({end_sec-start_sec:.1f}s duration)")
                        raw_stage_cleaned = clean_mne_raw(raw_stage,file_path)
                        
                        stage_segments.append(raw_stage_cleaned)
                        print(f"Added cleaned {stage} segment: {start_sec:.1f}s - {end_sec:.1f}s ({end_sec-start_sec:.1f}s duration)")
                    except Exception as e:
                        print(f"Error processing {stage} segment {start_sec}-{end_sec}s: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        if len(stage_segments) == 0:
            print(f"Patient {patient_id}: no {stage} segments found. Skipping.")
            continue
        
        print(f"Patient {patient_id}: found {len(stage_segments)} {stage} segments for processing.")
        
        # Run brain-heart coupling analysis on stage segments
        results_df_dict = compute_brain_heart_coupling(
            data_all=stage_segments,
            patient_id=f"{patient_id}_{stage}",
            bool_plots=False,
            save_plot=True,
            step_sec=step_sec
        )

        # Save averaged results per band for this patient
        for band, results_df in results_df_dict.items():
            results_path = os.path.join(TEMPS_DIR, f"{patient_id}_{stage}_results_{band}.parquet")
            print(f"Saving {stage} results for patient {patient_id}, band {band} to {results_path}")
            # create the directory if it doesn't exist
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            results_df.to_parquet(results_path, index=False)

    # Aggregate across all patients
    plot_all_patients_band_means(bands=power_bands.keys(), step_sec=step_sec, temps_dir=TEMPS_DIR, save_dir=SAVE_DIR)
    print(f"All {stage} stage processing and plotting complete.")

# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process EDF files for brain-heart coupling analysis')
    parser.add_argument('-c', '--edf_root', type=str, default="EDF_Format/EDF",
                        help='Root directory containing EDF files (default: EDF_Format/EDF)')
    parser.add_argument('--mode', type=str, choices=['random', 'n1'], default='n1',
                        help='Processing mode: random (select k random files) or n1 (extract N1 sleep stages)')
    # Controls for random mode
    parser.add_argument('-k', '--k_files', type=int, default=6,
                        help='Number of random EDF files to select per patient (default: 6)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible file selection (default: 42)')
    # Controls for N1 mode
    parser.add_argument('--n1_duration_min', type=int, default=1,
                        help='Duration of stage segments to extract in minutes (default: 1)')
    parser.add_argument('--stage', type=str, choices=['N1', 'N2', 'N3', 'R', 'W'], default='N2',
                        help='Sleep stage to extract (N1, N2, N3, R, or W) (default: N2)')
    # Common controls
    parser.add_argument('-s', '--step_sec', type=int, default=5,
                        help='Step size in seconds for sliding window (default: 5)')
    
    args = parser.parse_args()
    
    if args.mode == 'random':
        # RAM-conscious processing: k random PKLs per patient from specified edf_root
        process_patients_random6(edf_root=args.edf_root, k=args.k_files, step_sec=args.step_sec, seed=args.seed)
    elif args.mode == 'n1':
        # Stage-specific sleep stage processing
        process_patients_n1_stage_hep(edf_root=args.edf_root, step_sec=args.step_sec, n1_duration_min=args.n1_duration_min, stage=args.stage)
