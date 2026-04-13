import os
import re
import glob
import pickle
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne
import neurokit2 as nk
from scipy.signal import butter, filtfilt
from scipy.stats import entropy


# Import for EEG cleaning pipeline
from contextlib import contextmanager
import sys
import socket
import logging

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/hep_parquet_generation.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('HEP_parquet_generation')
logger.info(f"Script started on host: {socket.gethostname()}, PID: {os.getpid()}")

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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.eeg_utils import power_bands, compute_network_features, only_plots, rename_channels, eeg_channels, eeg_dict_convertion
import edf_cleaning
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
def get_project_name(path):
    """
    Extract the first folder after 'EDF_Format' in the given path.
    If 'EDF_Format' is not in the path, returns the basename of the path.
    """
    norm_path = os.path.normpath(path)
    parts = norm_path.split(os.sep)
    if 'EDF_Format' in parts:
        idx = parts.index('EDF_Format')
        if len(parts) > idx + 1:
            return parts[idx + 1]
    return os.path.basename(norm_path)

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

def get_processed_patients(temps_dir=TEMPS_DIR, pickle_base_dir="pickles_sleep_stage"):
    """
    Get a list of all patients that have been fully processed.

    A patient is considered processed when they have at least one pickle file in
    ``pickles_sleep_stage/<project>/<stage>/``.

    The project name and stage are inferred from the last component of ``temps_dir``
    (e.g. ``parquets_HEP/EDF_N1`` → project=``EDF``, stage=``N1``).
    When ``temps_dir`` has no recognisable ``<project>_<stage>`` suffix the function
    returns an empty list (cannot determine pickle dir).

    Args:
        temps_dir (str): Used only to derive the project/stage suffix.
        pickle_base_dir (str): Root directory for sleep-stage pickles
                               (default: ``"pickles_sleep_stage"``).

    Returns:
        list: Sorted list of patient IDs that have been fully processed.
    """
    # --- Derive pickle directory from temps_dir suffix ---
    folder_name = os.path.basename(os.path.normpath(temps_dir))
    stage_match = re.match(r'^(.+)_(N1|N2|N3|R|W)$', folder_name)
    if not stage_match:
        return []  # Cannot determine project/stage

    project_name = stage_match.group(1)
    stage = stage_match.group(2)
    pickle_dir = os.path.join(pickle_base_dir, project_name, stage)

    if not os.path.isdir(pickle_dir):
        return []

    # --- Collect patient IDs that have at least one pickle ---
    processed_patients = set()
    for pkl_path in glob.glob(os.path.join(pickle_dir, f"*_{stage}_*.pkl")):
        filename = os.path.basename(pkl_path)
        # filename pattern: <bare_patient_id>_<stage>_*.pkl
        m = re.match(rf'^(.+)_{re.escape(stage)}_', filename)
        if m:
            bare_patient_id = m.group(1)
            processed_patients.add(f"{bare_patient_id}_{stage}")

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
                sd1 = np.nan
                sd2 = np.nan
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
                # X was used for mutual info regression, safe to skip since we nan it out
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
        n_not_nan = np.sum(~np.isnan(arrs), axis=0)
        # Suppress warnings for all-NaN slices
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_arr_raw = np.nanmean(arrs, axis=0)
        mean_arr = np.where(n_not_nan >= n_files / 2, mean_arr_raw, np.nan)
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
        pattern = os.path.join(temps_dir, f"*{patient_id}*_results_{band}.parquet")
        file_list = glob.glob(pattern)
        if not file_list:
            print(f"No files found for patient {patient_id}, band {band}")
            continue
        dfs = []
        for f in file_list:
            try:
                df = pd.read_parquet(f)
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
        n_not_nan = np.sum(~np.isnan(arrs), axis=0)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_arr_raw = np.nanmean(arrs, axis=0)
        mean_arr = np.where(n_not_nan >= n_files / 2, mean_arr_raw, np.nan)
        mean_df = pd.DataFrame(mean_arr, columns=all_columns)
        mean_df = mean_df.dropna(how="all")
        only_plots(mean_df, save_plot=True, save_dir=save_dir, edf_pickle_name=patient_id, band=band, step_sec=step_sec)
        print(f"Plotted mean for patient {patient_id}, band {band}")

def plot_all_patients_band_means(bands=None, step_sec=5, temps_dir=TEMPS_DIR, save_dir=SAVE_DIR):
    if bands is None:
        bands = list(power_bands.keys())
    for band in bands:
        pattern = os.path.join(temps_dir, f"*results_{band}.parquet")
        file_list = glob.glob(pattern)
        if not file_list:
            print(f"No files found for band {band} (all patients)")
            continue
        dfs = []
        for f in file_list:
            try:
                df = pd.read_parquet(f)
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
        n_not_nan = np.sum(~np.isnan(arrs), axis=0)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_arr_raw = np.nanmean(arrs, axis=0)
        mean_arr = np.where(n_not_nan >= n_files / 2, mean_arr_raw, np.nan)
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
    
    if os.path.isfile(edf_root):
        if edf_root.lower().endswith(".pkl"):
            m = re.search(r'(\d{4}-\d{3})', edf_root)
            if m:
                pid = m.group(1)
            else:
                pid = os.path.basename(edf_root).split('.')[0]
            patient_to_files.setdefault(pid, []).append(edf_root)
        return patient_to_files

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
    
    if os.path.isfile(edf_root):
        if edf_root.lower().endswith(".edf"):
            m = re.search(r'(\d{4}-\d{3})', edf_root)
            if m:
                pid = m.group(1)
            else:
                pid = os.path.basename(edf_root).split('.')[0]
            patient_to_files.setdefault(pid, []).append(edf_root)
        return patient_to_files

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
    import sys
    import mne.io.array
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
    import sys
    import mne.io.array
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
                
        except Exception:
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
    import sys
    import mne.io.array
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

def smooth_sleep_stages(predicted_stages):
    """
    Smoothing: Merge short runs (< 3 epochs) into the prior run.
    Operates on a copy of the array.
    """
    smooth_stages = predicted_stages.copy()
    n_epochs = len(smooth_stages)
    
    if n_epochs > 0:
        current_label = smooth_stages[0]
        run_start_idx = 0
        
        for i in range(1, n_epochs):
            if smooth_stages[i] != current_label:
                # Run ended at i-1
                run_length = i - run_start_idx
                
                # Check if this finished run was too short
                if run_length < 3 and run_start_idx > 0:
                    # Merge into prior
                    prev_label = smooth_stages[run_start_idx - 1]
                    smooth_stages[run_start_idx:i] = prev_label
                
                run_start_idx = i
                current_label = smooth_stages[i]
        
        # Check last run
        run_length = n_epochs - run_start_idx
        if run_length < 3 and run_start_idx > 0:
             prev_label = smooth_stages[run_start_idx - 1]
             smooth_stages[run_start_idx:] = prev_label
             
    return smooth_stages

def process_patients_n1_stage_hep(edf_root="EDF_Format/EDF", step_sec=5, n1_duration_min=10, stage='N1'):
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
        Duration of stage segments to extract in minutes (default: 10)
    stage : str
        Sleep stage to extract (N1, N2, N3, R, or W) (default: N1)
    """
    import yasa
    clean_mne_raw = edf_cleaning.clean_mne_raw
    
    # Get project name from edf_root (the first folder from EDF_Format)
    project_name = get_project_name(edf_root)
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
    patient_to_files = dict(sorted(patient_to_files.items(), key=lambda x: x[0], reverse=False))
    
    # Diagnostic tracking
    patient_stats = {
        'total': len(patient_to_files),
        'processed': 0,
        'skipped_no_segments': 0,
        'skipped_already_processed': 0,
        'patient_filter_reasons': {
            'no_ecg': 0,  # All files failed: no ECG
            'no_eeg': 0,  # All files failed: no EEG
            'no_stage': 0,  # All files failed: no N3 stages detected
            'no_continuous': 0,  # All files failed: no continuous N3 epochs
            'too_short': 0,  # All files failed: N3 streak too short
            'processing_error': 0,  # All files failed: processing errors
            'mixed': 0  # Multiple different reasons across files
        },
        'file_level_stats': {
            'no_ecg': 0,
            'no_eeg': 0,
            'no_stage': 0,
            'no_continuous': 0,
            'too_short': 0,
            'processing_error': 0
        }
    }
    
    for patient_id, files in patient_to_files.items():
        logger.info(f"[{stage}] Patient {patient_id}: {len(files)} EDF files found.")
        logger.info(f"[{stage}] Patient {patient_id}: started processing")
        
        # Check if patient was already processed
        if is_patient_processed(f"{patient_id}_{stage}", temps_dir=TEMPS_DIR):
            logger.warning(f"[{stage}] Patient {patient_id}: already processed (parquet files exist). Skipping.")
            logger.info(f"[{stage}] Patient {patient_id}: already processed. Skipping.")
            patient_stats['skipped_already_processed'] += 1
            continue
        
        # Process each file to find stage segments
        stage_segments = []
        patient_file_stats = {
            'no_ecg': 0,
            'no_eeg': 0,
            'no_stage': 0,
            'no_continuous': 0,
            'too_short': 0,
            'processing_error': 0,
            'successful': 0
        }
        
        # Check if patient already has pickles in pickles_sleep_stage
        pickle_dir = os.path.join("pickles_sleep_stage", project_name, stage)
        existing_pickles = []
        if os.path.exists(pickle_dir):
            existing_pickles = glob.glob(os.path.join(pickle_dir, f"{patient_id}_{stage}_*.pkl"))
            
        if existing_pickles:
            print(f"Patient {patient_id}: Found {len(existing_pickles)} existing cleaned pickle files in {pickle_dir}. Loading pickles directly.")
            logger.info(f"[{stage}] Patient {patient_id}: Found {len(existing_pickles)} existing pickles. Skipping EDF extraction.")
            for pkl_file in existing_pickles:
                try:
                    with open(pkl_file, 'rb') as f:
                        raw_stage_cleaned = pickle.load(f)
                        stage_segments.append(raw_stage_cleaned)
                    patient_file_stats['successful'] += 1
                except Exception as e:
                    print(f"Error loading pickle {pkl_file}: {e}")
                    logger.error(f"[{stage}] Patient {patient_id}: Error loading pickle {pkl_file}: {e}")
                    patient_file_stats['processing_error'] += 1
                    patient_stats['file_level_stats']['processing_error'] += 1
            # Empty the files list so the EDF scanning loop is bypassed
            files = []

        for file_path in files:
            logger.info(f"[{stage}] Processing {file_path} for {stage} sleep stages...")
            
            try:
                # Load the EDF file using mne
                raw = mne.io.read_raw_edf(file_path, preload=True, encoding='latin1')
                # do notch 50 and low pass high pass
                raw.notch_filter(np.arange(50, raw.info['sfreq']/2, 50))
                raw.filter(l_freq=0.5, h_freq=raw.info['sfreq']/2 - 0.1)
                # raw is in V. change to microV
                # raw._data *= 1e6  # Convert from V to µV in-place at the numpy array level
                # resample to 256 Hz
                raw.resample(256)
                raw = rename_channels(raw)
                ch_lower = [ch.lower() for ch in raw.ch_names]                
                ecg_indices = [i for i, ch in enumerate(ch_lower) if 'ecg' in ch or 'ekg' in ch]
                if not ecg_indices:
                    logger.warning(f"[{stage}] No ECG channel found in {file_path}. Skipping.")
                    patient_file_stats['no_ecg'] += 1
                    patient_stats['file_level_stats']['no_ecg'] += 1
                    continue
                
                # Use YASA for sleep staging
                # Find the best channels for sleep staging
                available_eeg_channels = [ch for ch in raw.ch_names if ch in eeg_channels]
                if not available_eeg_channels:
                    logger.warning(f"[{stage}] No suitable EEG channels found in {file_path}. Skipping.")
                    patient_file_stats['no_eeg'] += 1
                    patient_stats['file_level_stats']['no_eeg'] += 1
                    continue
                
                # Resolve EOG channel
                eog_name = None
                if 'EOG+' in raw.ch_names:
                    eog_name = 'EOG+'
                else:
                    loc = 'LOC' if 'LOC' in raw.ch_names else None
                    roc = 'ROC' if 'ROC' in raw.ch_names else None
                    if loc and roc:
                        raw = mne.set_bipolar_reference(raw, anode=[loc], cathode=[roc], ch_name=['LOC-ROC'], drop_refs=False)
                        eog_name = 'LOC-ROC'
                    elif loc:
                        eog_name = loc
                    elif roc:
                        eog_name = roc
                logger.debug(f"[{stage}] EOG channel: {eog_name}")

                # Resolve EMG channel
                emg_name = None
                if 'EMG1+' in raw.ch_names:
                    emg_name = 'EMG1+'
                elif 'EMG2+' in raw.ch_names:
                    emg_name = 'EMG2+'
                logger.debug(f"[{stage}] EMG channel: {emg_name}")

                # Run YASA sleep staging — retry with next electrode if streak too short
                logger.info(f"[{stage}] Running YASA sleep staging on {file_path}...")

                def find_longest_stage_streak(stage_epochs, max_gap=3):
                    """Longest continuous streak of stage epochs (gaps up to max_gap allowed)."""
                    if len(stage_epochs) == 0:
                        return None, 0
                    longest_start = None
                    longest_length = 0
                    current_start = stage_epochs[0]
                    current_length = 1
                    current_end = stage_epochs[0]
                    for i in range(1, len(stage_epochs)):
                        gap = stage_epochs[i] - current_end
                        if gap <= max_gap + 1:
                            current_length += 1
                            current_end = stage_epochs[i]
                        else:
                            if current_length > longest_length:
                                longest_start = current_start
                                longest_length = current_length
                            current_start = stage_epochs[i]
                            current_length = 1
                            current_end = stage_epochs[i]
                    if current_length > longest_length:
                        longest_start = current_start
                        longest_length = current_length
                    return longest_start, longest_length

                # Build ordered candidate list (same logic as run_yasa_staging)
                ordered = [ch for ch in _YASA_EEG_PRIORITY if ch in available_eeg_channels]
                remainder = [ch for ch in available_eeg_channels if ch not in ordered]
                eeg_candidates = ordered + remainder

                tried_electrodes = []
                raw_staged = None
                eeg_name = None
                predicted_stages = None
                stage_segments_in_file = None  # will be set once a valid electrode is found

                for candidate in eeg_candidates:
                    try:
                        logger.debug(f"[{stage}] Trying EEG electrode: {candidate}")
                        raw_try = prep_for_yasa(raw=raw.copy(), eeg_name=candidate)
                        ss = yasa.SleepStaging(raw_try, eeg_name=candidate,
                                               eog_name=eog_name, emg_name=emg_name)
                        preds = ss.predict()
                    except Exception as exc:
                        logger.warning(f"[{stage}] YASA failed with '{candidate}': {exc}")
                        tried_electrodes.append(candidate)
                        continue

                    smooth = smooth_sleep_stages(preds)

                    if stage == 'W':
                        # W handling — accept any result (W segments found later)
                        raw_staged = raw_try
                        eeg_name = candidate
                        predicted_stages = preds
                        logger.info(f"[{stage}] Accepted electrode '{candidate}' for W staging.")
                        break

                    # Non-W: check streak length
                    s_epochs = np.where(smooth == stage)[0]
                    if len(s_epochs) == 0:
                        logger.warning(f"[{stage}] No {stage} epochs with electrode '{candidate}'. Trying next.")
                        tried_electrodes.append(candidate)
                        continue

                    lon_start, lon_len = find_longest_stage_streak(s_epochs)
                    epoch_duration_sec = 30
                    min_stage_epochs = int(n1_duration_min * 60 / epoch_duration_sec)

                    if lon_len >= min_stage_epochs:
                        raw_staged = raw_try
                        eeg_name = candidate
                        predicted_stages = preds
                        seg_start = lon_start * epoch_duration_sec
                        seg_end   = (lon_start + lon_len) * epoch_duration_sec
                        stage_segments_in_file = [(seg_start, seg_end)]
                        logger.info(
                            f"[{stage}] Electrode '{candidate}': longest streak "
                            f"{lon_len} epochs ({lon_len * epoch_duration_sec / 60:.1f} min). Accepted."
                        )
                        break
                    else:
                        logger.warning(
                            f"[{stage}] Electrode '{candidate}': longest streak "
                            f"{lon_len} epochs ({lon_len * epoch_duration_sec / 60:.1f} min) "
                            f"< required {n1_duration_min} min. Trying next electrode."
                        )
                        tried_electrodes.append(candidate)

                if raw_staged is None or predicted_stages is None:
                    logger.warning(
                        f"[{stage}] All electrodes exhausted for {file_path}. "
                        f"Tried: {tried_electrodes}. Skipping."
                    )
                    patient_file_stats['too_short'] += 1
                    patient_stats['file_level_stats']['too_short'] += 1
                    continue

                raw = raw_staged
                smooth_stages = smooth_sleep_stages(predicted_stages)
                logger.info(f"[{stage}] Final electrode: '{eeg_name}'. "
                            f"Stage counts: {pd.Series(predicted_stages).value_counts().to_dict()}")


                # Special handling for W: select 10 min of Wake right after the last adequate non-W run
                
                # --- Smoothing: Merge short runs (< 3 epochs) into the prior run ---
                # Apply smoothing for ALL cases as requested
                # smooth_stages = smooth_sleep_stages(predicted_stages) # Already done above

                if stage == 'W':
                    # YASA uses 30-second epochs by default
                    epoch_duration_sec = 30
                    # Require at least 5 minutes of any non-W stage before the Wake segment
                    min_non_w_epochs = int(np.ceil(5 * 60 / epoch_duration_sec))
                    # min_w_epochs is not currently used

                    # Build runs of identical stages: (label, start_idx, length)
                    runs = []
                    if len(smooth_stages) > 0:
                        cur_label = smooth_stages[0]
                        cur_start = 0
                        cur_len = 1
                        for i in range(1, len(smooth_stages)):
                            if smooth_stages[i] == cur_label:
                                cur_len += 1
                            else:
                                runs.append((cur_label, cur_start, cur_len))
                                cur_label = smooth_stages[i]
                                cur_start = i
                                cur_len = 1
                        runs.append((cur_label, cur_start, cur_len))

                    # leave only runs > min_non_w_epochs
                    runs = [run for run in runs if run[2] > min_non_w_epochs]

                    # save in stage_segments_in_file the start sec and end sec of the W segments
                    stage_segments_in_file = [(run[1] * epoch_duration_sec, (run[1] + run[2]) * epoch_duration_sec) for run in runs if run[0] == 'W']
                else:
                    # For non-W stages, stage_segments_in_file is already set by the retry loop
                    # If it's still None, it means no suitable segment was found even after retries
                    if stage_segments_in_file is None or len(stage_segments_in_file) == 0:
                        print(f"No continuous {stage} epochs found in {file_path} after electrode retry. Skipping.")
                        patient_file_stats['no_continuous'] += 1
                        patient_stats['file_level_stats']['no_continuous'] += 1
                        continue
                
                # Crop the raw data to stage segments, clean them, and add to list
                for start_sec, end_sec in stage_segments_in_file:
                    try:
                        # Crop the raw data to the stage segment
                        raw_stage = raw.copy().crop(tmin=start_sec, tmax=end_sec)
                        
                        # Apply cleaning to the stage segment
                        original_duration = raw_stage.times[-1] if len(raw_stage.times) > 0 else 0
                        logger.info(f"[{stage}] Cleaning {stage} segment: {start_sec:.1f}s - {end_sec:.1f}s (Duration: {original_duration/60:.1f} min)")
                        
                        raw_stage_cleaned = clean_mne_raw(raw_stage, file_path)
                        
                        if raw_stage_cleaned is not None:
                            cleaned_duration = raw_stage_cleaned.times[-1] if len(raw_stage_cleaned.times) > 0 else 0
                            snipped_sec = original_duration - cleaned_duration
                            snipped_pct = (snipped_sec / original_duration * 100) if original_duration > 0 else 0
                            logger.info(f"[{stage}] Cleaned segment. Snipped {snipped_sec:.1f}s ({snipped_pct:.1f}%). Remaining: {cleaned_duration/60:.1f} min")
                        else:
                            logger.error(f"[{stage}] clean_mne_raw returned None for {file_path}")
                        
                        # Save the cleaned MNE raw object to pickle
                        duration = int(end_sec - start_sec)
                        pickle_dir = os.path.join("pickles_sleep_stage", project_name,stage)
                        os.makedirs(pickle_dir, exist_ok=True)
                        pickle_filename = f"{patient_id}_{stage}_{duration}_{step_sec}.pkl"
                        pickle_path = os.path.join(pickle_dir, pickle_filename)
                        with open(pickle_path, 'wb') as f:
                            pickle.dump(raw_stage_cleaned, f)
                        logger.info(f"[{stage}] Saved cleaned MNE raw to {pickle_path}")
                        
                        stage_segments.append(raw_stage_cleaned)
                        patient_file_stats['successful'] += 1
                        print(f"Added cleaned {stage} segment: {start_sec:.1f}s - {end_sec:.1f}s ({end_sec-start_sec:.1f}s duration)")
                    except Exception as e:
                        print(f"Error processing {stage} segment {start_sec}-{end_sec}s: {e}")
                        patient_file_stats['processing_error'] += 1
                        patient_stats['file_level_stats']['processing_error'] += 1
                        continue
                        
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                logger.error(f"[{stage}] Patient {patient_id}: CRASHED during processing {file_path}. Error: {e}", exc_info=True)
                patient_file_stats['processing_error'] += 1
                patient_stats['file_level_stats']['processing_error'] += 1
                continue
        
        if len(stage_segments) == 0:
            print(f"Patient {patient_id}: no {stage} segments found. Skipping.")
            logger.info(f"[{stage}] Patient {patient_id}: no {stage} segments found. Skipping.")
            print(f"  Patient {patient_id} file statistics: {patient_file_stats}")
            patient_stats['skipped_no_segments'] += 1
            
            # Determine the primary reason why this patient was filtered
            total_failed_files = sum([
                patient_file_stats['no_ecg'],
                patient_file_stats['no_eeg'],
                patient_file_stats['no_stage'],
                patient_file_stats['no_continuous'],
                patient_file_stats['too_short'],
                patient_file_stats['processing_error']
            ])
            
            if total_failed_files == 0:
                # All files failed at exception level (before reaching specific filters)
                patient_stats['patient_filter_reasons']['processing_error'] += 1
            else:
                # Find the primary reason (the one that affected all or most files)
                # Check if one reason accounts for all failures
                if patient_file_stats['no_ecg'] == total_failed_files and total_failed_files > 0:
                    patient_stats['patient_filter_reasons']['no_ecg'] += 1
                elif patient_file_stats['no_eeg'] == total_failed_files and total_failed_files > 0:
                    patient_stats['patient_filter_reasons']['no_eeg'] += 1
                elif patient_file_stats['no_stage'] == total_failed_files and total_failed_files > 0:
                    patient_stats['patient_filter_reasons']['no_stage'] += 1
                elif patient_file_stats['no_continuous'] == total_failed_files and total_failed_files > 0:
                    patient_stats['patient_filter_reasons']['no_continuous'] += 1
                elif patient_file_stats['too_short'] == total_failed_files and total_failed_files > 0:
                    patient_stats['patient_filter_reasons']['too_short'] += 1
                elif patient_file_stats['processing_error'] == total_failed_files and total_failed_files > 0:
                    patient_stats['patient_filter_reasons']['processing_error'] += 1
                else:
                    # Multiple different reasons - count as mixed if no single reason accounts for all failures
                    # Count how many different reasons have non-zero counts
                    non_zero_reasons = sum([
                        1 if patient_file_stats['no_ecg'] > 0 else 0,
                        1 if patient_file_stats['no_eeg'] > 0 else 0,
                        1 if patient_file_stats['no_stage'] > 0 else 0,
                        1 if patient_file_stats['no_continuous'] > 0 else 0,
                        1 if patient_file_stats['too_short'] > 0 else 0,
                        1 if patient_file_stats['processing_error'] > 0 else 0
                    ])
                    if non_zero_reasons > 1:
                        # Multiple different reasons
                        patient_stats['patient_filter_reasons']['mixed'] += 1
                    else:
                        # Shouldn't happen, but assign to the single reason
                        reasons = {
                            'no_ecg': patient_file_stats['no_ecg'],
                            'no_eeg': patient_file_stats['no_eeg'],
                            'no_stage': patient_file_stats['no_stage'],
                            'no_continuous': patient_file_stats['no_continuous'],
                            'too_short': patient_file_stats['too_short'],
                            'processing_error': patient_file_stats['processing_error']
                        }
                        primary_reason = max(reasons, key=reasons.get)
                        if reasons[primary_reason] > 0:
                            patient_stats['patient_filter_reasons'][primary_reason] += 1
                        else:
                            patient_stats['patient_filter_reasons']['mixed'] += 1
            
            continue
        
        print(f"Patient {patient_id}: found {len(stage_segments)} {stage} segments for processing.")
        if len(stage_segments) > 3:
            stage_segments = sorted(stage_segments, key=lambda r: r.n_times, reverse=True)[:3]
            print(f"Patient {patient_id}: keeping longest 3 segments out of original set.")
        patient_stats['processed'] += 1
        
        try:
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
                
            logger.info(f"[{stage}] Patient {patient_id}: successfully finished processing.")
            
        except Exception as e:
            logger.error(f"[{stage}] Patient {patient_id}: CRASHED during processing. Error: {e}", exc_info=True)

    # Aggregate across all patients
    plot_all_patients_band_means(bands=power_bands.keys(), step_sec=step_sec, temps_dir=TEMPS_DIR, save_dir=SAVE_DIR)
    
    # Print diagnostic summary
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC SUMMARY for {stage} stage processing:")
    print(f"{'='*80}")
    print(f"Total patients found: {patient_stats['total']}")
    print(f"Patients successfully processed: {patient_stats['processed']}")
    print(f"Patients skipped (already processed): {patient_stats['skipped_already_processed']}")
    print(f"Patients skipped (no {stage} segments found): {patient_stats['skipped_no_segments']}")
    print(f"\n{'='*80}")
    print("PATIENT-LEVEL FILTER BREAKDOWN (why patients were filtered):")
    print(f"{'='*80}")
    print(f"  Patients filtered - No ECG channel (all files): {patient_stats['patient_filter_reasons']['no_ecg']}")
    print(f"  Patients filtered - No suitable EEG channels (all files): {patient_stats['patient_filter_reasons']['no_eeg']}")
    print(f"  Patients filtered - No {stage} stages detected (all files): {patient_stats['patient_filter_reasons']['no_stage']}")
    print(f"  Patients filtered - No continuous {stage} epochs (all files): {patient_stats['patient_filter_reasons']['no_continuous']}")
    print(f"  Patients filtered - {stage} streak too short (all files, < {n1_duration_min} min): {patient_stats['patient_filter_reasons']['too_short']}")
    print(f"  Patients filtered - Processing errors (all files): {patient_stats['patient_filter_reasons']['processing_error']}")
    print(f"  Patients filtered - Mixed reasons (multiple different failures): {patient_stats['patient_filter_reasons']['mixed']}")
    print(f"\n{'='*80}")
    print("FILE-LEVEL FILTERING STATISTICS (across all patients):")
    print(f"{'='*80}")
    print(f"  Files skipped - No ECG channel: {patient_stats['file_level_stats']['no_ecg']}")
    print(f"  Files skipped - No suitable EEG channels: {patient_stats['file_level_stats']['no_eeg']}")
    print(f"  Files skipped - No {stage} stages detected: {patient_stats['file_level_stats']['no_stage']}")
    print(f"  Files skipped - No continuous {stage} epochs: {patient_stats['file_level_stats']['no_continuous']}")
    print(f"  Files skipped - {stage} streak too short (< {n1_duration_min} min): {patient_stats['file_level_stats']['too_short']}")
    print(f"  Files skipped - Processing errors: {patient_stats['file_level_stats']['processing_error']}")
    print(f"{'='*80}\n")
    
    print(f"All {stage} stage processing and plotting complete.")

def rename_edfs_to_upper(edf_root):
    """
    Renames all .edf files under edf_root to upper case (both filename and extension).
    """
    print(f"Checking for lowercase filenames (renaming to uppercase) in {edf_root}...")
    count = 0
    
    if os.path.isfile(edf_root):
        file = os.path.basename(edf_root)
        if file.lower().endswith(".edf") and file != file.upper():
            root = os.path.dirname(edf_root)
            new_name = file.upper()
            new_path = os.path.join(root, new_name)
            if not os.path.exists(new_path):
                try:
                    os.rename(edf_root, new_path)
                    print(f"Renamed: {file} -> {new_name}")
                    return new_path
                except Exception as e:
                    print(f"Failed to rename {edf_root} -> {new_path}: {e}")
        return edf_root

    # Walk top-down so we can rename files inside directories
    for root, dirs, files in os.walk(edf_root):
        for file in files:
            # Check if it is an .edf file (case-insensitive)
            if file.lower().endswith(".edf"):
                # If name is not fully uppercase
                if file != file.upper():
                    old_path = os.path.join(root, file)
                    new_name = file.upper()
                    new_path = os.path.join(root, new_name)

                    # Only rename if destination doesn't exist (unless it's the same file on case-insensitive FS)
                    # On Linux (ext4), case sensitive, so foo.edf and FOO.EDF are distinct.
                    if os.path.exists(new_path):
                        print(f"Skipping rename for {file} -> {new_name}: Target exists.")
                        continue
                    
                    try:
                        os.rename(old_path, new_path)
                        print(f"Renamed: {file} -> {new_name}")
                        count += 1
                    except Exception as e:
                        print(f"Failed to rename {old_path} -> {new_path}: {e}")

    print(f"Renamed {count} files to uppercase.")
    return edf_root

# YASA-recommended EEG channel priority (central > frontal > parietal > occipital)
_YASA_EEG_PRIORITY = [
    'Cz', 'C3', 'C4',           # central (best for YASA)
    'Fz', 'F3', 'F4',           # frontal
    'Pz', 'P3', 'P4',           # parietal
    'Oz', 'O1', 'O2',           # occipital
    'F7', 'F8', 'T3', 'T4',     # temporal / lateral
    'T5', 'T6', 'P7', 'P8',
]


def run_yasa_staging(raw, available_eeg_channels, eog_name, emg_name, stage):
    """
    Try YASA sleep staging with EEG channels in priority order.

    For each candidate channel (ordered by _YASA_EEG_PRIORITY, then whatever
    is left in available_eeg_channels):
      1. Apply prep_for_yasa() on a *copy* of raw so other channels are untouched.
      2. Run yasa.SleepStaging and ss.predict().
      3. If it succeeds, return (raw_prepped, eeg_name_used, predicted_stages).
      4. If it fails, log the error and try the next channel.

    Returns None if every channel fails.
    """
    # Build an ordered candidate list: priority channels first, remainder after
    ordered = [ch for ch in _YASA_EEG_PRIORITY if ch in available_eeg_channels]
    remainder = [ch for ch in available_eeg_channels if ch not in ordered]
    candidates = ordered + remainder

    for eeg_name in candidates:
        try:
            logger.debug(f"[{stage}] Trying EEG channel for YASA: {eeg_name}")
            raw_try = prep_for_yasa(raw=raw.copy(), eeg_name=eeg_name)
            ss = yasa.SleepStaging(raw_try, eeg_name=eeg_name,
                                    eog_name=eog_name, emg_name=emg_name)
            predicted_stages = ss.predict()
            logger.info(f"[{stage}] YASA succeeded with EEG channel: {eeg_name}")
            return raw_try, eeg_name, predicted_stages
        except Exception as exc:
            logger.warning(
                f"[{stage}] YASA failed with EEG channel '{eeg_name}': {exc}. "
                "Trying next channel..."
            )

    logger.error(f"[{stage}] All EEG channels exhausted — YASA staging failed.")
    return None


def prep_for_yasa(raw, eeg_name):
    """
    Checks and fixes amplitude scaling and filtering on the EEG channel
    in-place on `raw` to optimise YASA sleep staging.
    Returns the modified `raw` object (same instance, channel replaced).
    """
    # 1. Ensure data is in memory – required for in-place resampling / filtering.
    if not raw.preload:
        logger.debug("[prep_for_yasa] Data not preloaded – loading now.")
        raw.load_data()
    # 1. Work on a single-channel copy for scaling / filtering
    raw_prep = raw.copy().pick_channels([eeg_name])
    data = raw.get_data(picks=[eeg_name]).squeeze()  # (n_times,)

    # Robust amplitude estimate: 99th–1st percentile avoids rail/clip artefacts
    amplitude = float(np.percentile(data, 99) - np.percentile(data, 1))

    if amplitude <= 0:
        logger.warning(f"[_infer_and_fix_eeg_units] '{eeg_name}' appears flat, skipping.")
        return

    log_amp = np.log10(amplitude)

    # Log-space centroids for each unit assumption
    centroids = {
        "V":  np.log10(1e-4),   # -4.0
        "mV": np.log10(1e-1),   # -1.0
        "µV": np.log10(1e+2),   # +2.0
    }
    scales = {"V": 1, "mV": 1e-3, "µV": 1e-6}

    detected = min(centroids, key=lambda u: abs(log_amp - centroids[u]))
    distance = abs(log_amp - centroids[detected])

    logger.debug(
        f"[_infer_and_fix_eeg_units] '{eeg_name}' robust p-p={amplitude:.2e}, "
        f"log10={log_amp:.2f} → nearest centroid: {detected} (dist={distance:.2f} decades)"
    )

    if distance > 1.4:
        # More than halfway to the next centroid — something is genuinely odd
        logger.warning(
            f"[_infer_and_fix_eeg_units] '{eeg_name}' amplitude={amplitude:.2e} is "
            f"{distance:.1f} log-decades from '{detected}' centroid. "
            "Data may be heavily artefacted or the unit is unexpected. Proceeding anyway."
        )

    if scales[detected] is None:
        logger.debug(f"[_infer_and_fix_eeg_units] Already in V – no rescaling needed.")
        

    logger.warning(
        f"[_infer_and_fix_eeg_units] Detected {detected} "
        f"(amplitude={amplitude:.2e}), rescaling by {scales[detected]:.0e} → V."
    )
    raw.apply_function(lambda x: x * scales[detected], picks="all", channel_wise=False)
    # 3. FIX FILTERING: apply standard sleep bandpass (0.3 – 35 Hz)
    print("[*] Applying bandpass filter (0.1 - 40 Hz)")
    raw_prep.filter(l_freq=0.1, h_freq=40, fir_design='firwin', verbose=False)

    # 4. Write the prepared channel back into the original Raw object
    ch_idx = raw.ch_names.index(eeg_name)
    raw._data[ch_idx, :] = raw_prep.get_data()[0, :]
    # resample raw to 100hz
    raw.resample(100)
    return raw
    raw.info['sfreq']    

def clean_duplicate_pickles(base_dir="pickles_sleep_stage"):
    """
    Remove duplicate patient pickles from the same sleep stage,
    keeping only the one with the longest duration.
    A patient is considered the same if their ID after an optional hyphen matches.
    """
    files = glob.glob(os.path.join(base_dir, "**", "*.pkl"), recursive=True)
    groups = {}
    for f in files:
        basename = os.path.basename(f)
        parts = basename.replace('.pkl', '').split('_')
        if len(parts) < 4:
            continue
        
        stage = parts[-3]
        try:
            duration = int(parts[-2])
        except ValueError:
            continue
            
        bare_patient_id = "_".join(parts[:-3])
        
        # Canonical patient ID: if there's a hyphen, take the part after it.
        if '-' in bare_patient_id:
            canon_id = bare_patient_id.split('-')[-1]
        else:
            canon_id = bare_patient_id
            
        # Group by canonical ID, stage, and the parent directory to avoid cross-project merging
        key = (canon_id, stage, os.path.dirname(f))
        groups.setdefault(key, []).append((duration, f, bare_patient_id))
        
    for key, items in groups.items():
        if len(items) > 1:
            # Sort by duration descending
            items.sort(key=lambda x: x[0], reverse=True)
            logger.info(f"Cleaning duplicates for patient {key[0]} in stage {key[1]}:")
            print(f"Cleaning duplicates for patient {key[0]} in stage {key[1]}:")
            logger.info(f"  Keeping: {items[0][1]} (duration={items[0][0]})")
            print(f"  Keeping: {items[0][1]} (duration={items[0][0]})")
            for item in items[1:]:
                logger.info(f"  Removing: {item[1]} (duration={item[0]})")
                print(f"  Removing: {item[1]} (duration={item[0]})")
                try:
                    os.remove(item[1])
                except Exception as e:
                    logger.error(f"  Failed to remove {item[1]}: {e}")
                    print(f"  Failed to remove {item[1]}: {e}")

# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process EDF files for brain-heart coupling analysis')
    parser.add_argument('-c', '--edf_root', type=str, default="EDF_Format/Seeg",
                        help='Root directory containing EDF files (default: EDF_Format/EDF)')
    parser.add_argument('--mode', type=str, choices=['random', 'stage'], default='stage',
                        help='Processing mode: random (select k random files) or stage (extract selected sleep stages)')
    # Controls for random mode
    parser.add_argument('-k', '--k_files', type=int, default=6,
                        help='Number of random EDF files to select per patient (default: 6)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible file selection (default: 42)')
    # Controls for N1 mode
    parser.add_argument('--n1_duration_min', type=int, default=10,
                        help='Duration of stage segments to extract in minutes (default: 10)')
    parser.add_argument('--stage', type=str, choices=['N1', 'N2', 'N3', 'R', 'W'], default='N3',
                        help='Sleep stage to extract (N1, N2, N3, R, or W) (default: N2)')
    # Common controls
    parser.add_argument('-s', '--step_sec', type=int, default=5,
                        help='Step size in seconds for sliding window (default: 5)')
    
    args = parser.parse_args()

    # Setup run-specific logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = get_project_name(args.edf_root)
    run_log_filename = f"logs/run_{project_name}_{args.mode}_{args.stage}_{timestamp}.log"
    
    run_handler = logging.FileHandler(run_log_filename)
    run_handler.setLevel(logging.DEBUG)
    run_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(run_handler)
    
    logger.info(f"--- STARTED NEW RUN ---")
    logger.info(f"Arguments: {vars(args)}")

    # Clean duplicates in pickles_sleep_stage before processing
    clean_duplicate_pickles("pickles_sleep_stage")
    
    # Rename .edf files to uppercase
    args.edf_root = rename_edfs_to_upper(args.edf_root)

    process_patients_n1_stage_hep(edf_root=args.edf_root, step_sec=args.step_sec, n1_duration_min=args.n1_duration_min, stage=args.stage)
