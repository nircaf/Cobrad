import os
import re
import glob
import pickle
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
from bct import efficiency_bin, transitivity_bu, modularity_und, assortativity_bin

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
TEMPS_DIR = "temps_EDF_HEP"
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
        window_data = raw.get_data()
        ch_names = raw.ch_names
        sfreq = int(raw.info['sfreq'])
        print(f"Sampling frequency: {sfreq}")
        name_prefix = f"{patient_id}_edf{i+1}"

        # Extract ECG and detect R-peaks
        ch_lower = [ch.lower() for ch in ch_names]
        if 'ecg' not in ch_lower:
            print(f"[{patient_id}] EDF {i+1} has no ECG channel; skipping.")
            continue

        ecg_idx = ch_lower.index('ecg')
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
        eeg_channels = ['Fpz', 'F7', 'T3', 'T5', 'Fp1', 'F3', 'C3', 'P3', 'Oz', 'F8', 'T4', 'T6', 'Fp2', 'F4', 'C4', 'P4', 'Fz', 'Cz']
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
        return 'ecg' in ch_lower
    except Exception as e:
        print(f"Failed to read pickle {edf_path}: {e}")
        return False

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
    patient_to_files = group_edf_files_by_patient(edf_root=edf_root)
    if not patient_to_files:
        print(f"No EDF files found under {edf_root}.")
        return
    
    # Check which patients are already processed
    processed_patients = get_processed_patients()
    if processed_patients:
        print(f"Already processed patients: {', '.join(processed_patients)}")
    else:
        print("No patients have been processed yet.")

    for patient_id, files in sorted(patient_to_files.items()):
        print(f"Patient {patient_id}: {len(files)} PKL files found.")
        
        # Check if patient was already processed
        if is_patient_processed(patient_id):
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
            results_df.to_parquet(results_path, index=False)

    # Aggregate across all patients
    plot_all_patients_band_means(bands=power_bands.keys(), step_sec=step_sec, temps_dir=TEMPS_DIR, save_dir=SAVE_DIR)
    print("All processing and plotting complete.")

# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # RAM-conscious processing: 6 random PKLs per patient from pickles/EDF
    process_patients_random6(edf_root="pickles/EDF", k=6, step_sec=5, seed=42)
