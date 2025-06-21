import mne 
import os 
import pandas as pd
import pickle

import numpy as np
import pandas as pd
import neurokit2 as nk
from sklearn.feature_selection import mutual_info_regression
from mne_connectivity import SpectralConnectivity as spectral_connectivity
from bct import efficiency_bin, transitivity_bu, modularity_und, assortativity_bin
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from scipy.signal import coherence
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities, modularity

try:
    import streamlit as st
    is_streamlit = True
except ImportError:
    is_streamlit = False

def compute_brain_heart_coupling(edf_results, key, motor_symptoms=None, bool_plots=False):
    """
    Compute time-varying EEG network metrics and HRV indices, then their mutual-information coupling.

    Parameters
    ----------
    edf_results : dict
        Mapping keys to MNE Raw objects.
    key : str
        Key of the recording to analyze.
    motor_symptoms : array-like, optional
        Δ motor symptom values for correlation plots.
    bool_plots : bool
        If True, display plots (supports Streamlit).

    Returns
    -------
    results_df : pd.DataFrame
        Mutual information between network metrics and HRV indices.
    """
    raw = edf_results[key]
    data_all = raw.get_data()
    ch_names = raw.ch_names
    eeg_channels = ['Fpz', 'F7', 'T3', 'T5', 'Fp1', 'F3', 'C3', 'P3', 'Oz', 'F8', 'T4', 'T6', 'Fp2', 'F4', 'C4', 'P4', 'Fz', 'Cz']
    sfreq = int(raw.info['sfreq'])

    # Extract ECG and detect R-peaks
    if 'ecg' not in ch_names:
        raise ValueError("No 'ecg' channel found")
    ecg_signal = data_all[ch_names.index('ecg')]
    _, rpk = nk.ecg_peaks(ecg_signal, sampling_rate=sfreq)
    rpeaks = rpk['ECG_R_Peaks']
    # Times in seconds
    r_times = rpeaks / sfreq

    # Extract EEG data (excluding ECG)
    data = data_all[[ch_names.index(ch) for ch in eeg_channels if ch in ch_names]]
    n_nodes, n_samples = data.shape
    # Sliding window parameters
    w_eeg_sec = 15
    step_sec = 5
    w_eeg = int(w_eeg_sec * sfreq)
    step = int(step_sec * sfreq)
    n_windows = int((n_samples - w_eeg) / step) + 1

    # Initialize time series lists
    eff_ts, clu_ts, mod_ts, ass_ts = [], [], [], []
    cvi_ts, csi_ts = [], []

    for w in range(n_windows):
        start = w * step
        end = start + w_eeg
        segment = data[:, start:end]

        # Connectivity via coherence (8-45 Hz)
        con = np.zeros((n_nodes, n_nodes))
        fmin, fmax = 8, 45
        for i in range(n_nodes):
            for j in range(i, n_nodes):
                f, Cxy = coherence(segment[i], segment[j], fs=sfreq, nperseg=w_eeg//10)
                mask = (f >= fmin) & (f <= fmax)
                val = np.nanmean(Cxy[mask]) if mask.any() else 0
                if np.isnan(val):
                    val = 0
                con[i, j] = con[j, i] = val

        # Binarize at 90th percentile
        thr = np.percentile(con, 90)
        con_bin = (con >= thr).astype(int)

        # Compute network metrics
        eff_ts.append(efficiency_bin(con_bin))
        clu_ts.append(transitivity_bu(con_bin))
        G = nx.from_numpy_array(con_bin)
        comm = list(greedy_modularity_communities(G))
        if G.number_of_edges() > 0 and len(comm) > 0:
            mod_ts.append(modularity(G, comm))
        else:
            mod_ts.append(np.nan)  # or 0, or skip, depending on your needs
        ass_ts.append(assortativity_bin(con_bin))

        # HRV within same window
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

    # Convert to arrays, mask out NaNs
    eff_arr = np.array(eff_ts)
    clu_arr = np.array(clu_ts)
    mod_arr = np.array(mod_ts)
    ass_arr = np.array(ass_ts)
    cvi_arr = np.array(cvi_ts)
    csi_arr = np.array(csi_ts)
    valid = ~np.isnan(cvi_arr) & ~np.isnan(csi_arr)

    # Truncate arrays
    eff_arr, clu_arr = eff_arr[valid], clu_arr[valid]
    mod_arr, ass_arr = mod_arr[valid], ass_arr[valid]
    cvi_arr, csi_arr = cvi_arr[valid], csi_arr[valid]

    # Mutual information coupling
    results = {}
    metrics = {'Efficiency': eff_arr, 'Clustering': clu_arr,
               'Modularity': mod_arr, 'Assortativity': ass_arr}
    for name, arr in metrics.items():
        X = arr.reshape(-1, 1)
        mic_sym = mutual_info_regression(X, csi_arr, random_state=0)[0]
        mic_vag = mutual_info_regression(X, cvi_arr, random_state=0)[0]
        results[name] = {'Sympathetic MI': mic_sym, 'Vagal MI': mic_vag}

    results_df = pd.DataFrame(results).T

    # Optional plots
    if bool_plots:
        # Scatter MI vs motor symptoms
        if motor_symptoms is not None:
            fig, ax = plt.subplots(figsize=(5, 4))
            xs = results_df['Vagal MI'].values
            ys = motor_symptoms
            ax.scatter(xs, ys)
            ax.set_xlabel('Vagal MI')
            ax.set_ylabel('Δ Motor Symptoms')
            stat, p = wilcoxon(ys)
            n = len(ys)
            z = (stat - n*(n+1)/4) / np.sqrt(n*(n+1)*(2*n+1)/24)
            ax.set_title(f'p={p:.3f}, Z={z:.2f}')
            plt.tight_layout()
            if is_streamlit: st.pyplot(fig)
            else: plt.show()

        # Example time series HRV vs network metric
        fig, ax = plt.subplots(figsize=(6, 4))
        t = np.arange(len(eff_arr)) * step_sec
        ax.plot(t, eff_arr, label='Efficiency')
        ax.plot(t, csi_arr, label='Sympathetic (SD2)')
        ax.legend()
        ax.set_xlabel('Time (s)')
        ax.set_title('Dynamic Efficiency vs Sympathetic')
        plt.tight_layout()
        if is_streamlit: st.pyplot(fig)
        else: plt.show()

    return results_df

def load_edf_pickles_with_ecg(pickle_dir='pickles/EDF'):
    """
    Iterate over all pickle files in the given directory, load each,
    and return None for files where 'ecg' is not in the channel names.
    Returns a dict: {filename: raw or None}
    """
    results = {}
    for fname in os.listdir(pickle_dir):
        if not fname.endswith('.pkl'):
            continue
        fpath = os.path.join(pickle_dir, fname)
        with open(fpath, 'rb') as f:
            try:
                raw = pickle.load(f)
                if hasattr(raw, 'ch_names'):
                    ch_names = raw.ch_names
                elif hasattr(raw, 'info') and 'ch_names' in raw.info:
                    ch_names = raw.info['ch_names']
                else:
                    ch_names = []
                if 'ecg' not in [ch.lower() for ch in ch_names]:
                    results[fname] = None
                else:
                    results[fname] = raw
                    break
            except Exception as e:
                print(f"Error loading {fname}: {e}")
                results[fname] = None
    return results

# Example usage:
edf_results = load_edf_pickles_with_ecg()
# remove keys with None values
edf_results = {k: v for k, v in edf_results.items() if v is not None}
edf_results['0345-010.edf_600_1.pkl'].ch_names
compute_brain_heart_coupling(edf_results, '0345-010.edf_600_1.pkl', bool_plots=True)
pass
# for fname, raw in edf_results.items():
#     print(fname, 'Has ECG' if raw is not None else 'No ECG')

