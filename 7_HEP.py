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
from scipy.stats import wilcoxon, zscore, pearsonr
from scipy.signal import coherence
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities, modularity
from scipy.stats import pearsonr, entropy, ranksums
try:
    import streamlit as st
    is_streamlit = True
except ImportError:
    is_streamlit = False

def joint_entropy(x, y, bins=50):
    """Compute joint entropy of two arrays."""
    c_xy = np.histogram2d(x, y, bins)[0]
    c_xy = c_xy / np.sum(c_xy)
    c_xy = c_xy[c_xy > 0]
    return entropy(c_xy, base=2)


def generate_control_table(group1_data, group2_data,
                           group1_label='Group1', group2_label='Group2'):
    """
    Perform Wilcoxon signed‐rank tests on each network metric and cardiac feature
    between two paired groups, and return a multi‐indexed DataFrame of p‐ and Z‐values.

    Parameters
    ----------
    group1_data : dict
        Nested dict of the form
          {
            'Alpha network': {'Clustering': array, 'Efficiency': array, ...},
            'Beta network':  {...},
            'Gamma network': {...},
            'Cardiac features': {'Sympathetic': array, 'Vagal': array}
          }
    group2_data : dict
        Same structure as group1_data, for the second condition.
    group1_label, group2_label : str
        Names to use in the single output column header.

    Returns
    -------
    pd.DataFrame
        MultiIndex rows (network category, metric) × 1 column named
        "{group1_label} vs. {group2_label}", with entries "p = …, Z = …".
    """
    # Define which metrics to test in each category
    metrics_map = {
        'Alpha network':   ['Clustering', 'Efficiency', 'Assortativity', 'Modularity'],
        'Beta network':    ['Clustering', 'Efficiency', 'Assortativity', 'Modularity'],
        'Gamma network':   ['Clustering', 'Efficiency', 'Assortativity', 'Modularity'],
        'Cardiac features': ['Sympathetic', 'Vagal']
    }

    # Build the MultiIndex
    tuples = []
    for cat, mets in metrics_map.items():
        for m in mets:
            tuples.append((cat, m))
    idx = pd.MultiIndex.from_tuples(tuples,
                                    names=['Feature category', 'Metric'])

    col_name = f"{group1_label} vs. {group2_label}"
    table = pd.DataFrame(index=idx, columns=[col_name])

    # Fill in the Wilcoxon p‐ and Z‐values
    for cat, mets in metrics_map.items():
        for m in mets:
            x = np.asarray(group1_data[cat][m])
            y = np.asarray(group2_data[cat][m])
            # drop any nan pairs
            mask = ~np.isnan(x) & ~np.isnan(y)
            x, y = x[mask], y[mask]
            if len(x) < 1:
                table.loc[(cat, m), col_name] = "n/a"
                continue
            W, p = wilcoxon(x, y)
            n = len(x)
            # Z‐score approximation for Wilcoxon signed‐rank
            mu = n * (n + 1) / 4
            sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
            Z = (W - mu) / sigma if sigma > 0 else np.nan
            table.loc[(cat, m), col_name] = f"p = {p:.4f}, Z = {Z:.2f}"

    return table

def _plot_metric_vs_hrv(t, metric_arr, hrv_arr, metric_name, hrv_label,
                        save_plot, edf_name, save_dir):
    """
    Plot a network metric against an HRV index over time, with optional saving.
    Also saves normalized (min joint entropy) plot and Wilcoxon signed-rank test values.
    """
    from scipy.stats import entropy
    import os
    import numpy as np

    # Raw plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, metric_arr, label=metric_name)
    ax.plot(t, hrv_arr, label=hrv_label)
    # Wilcoxon signed-rank test (paired)
    try:
        stat_raw, p_raw = wilcoxon(metric_arr, hrv_arr)
    except ValueError:
        stat_raw, p_raw = np.nan, np.nan
    ax.set_title(f'Dynamic {metric_name} vs {hrv_label}\nWilcoxon signed-rank stat={stat_raw:.2f}, p={p_raw:.2g}')
    ax.set_xlabel('Time (s)')
    ax.legend()
    plt.tight_layout()
    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"{save_dir}/{edf_name}_{metric_name.lower()}_{hrv_label.lower()}.png"
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        if 'is_streamlit' in globals() and is_streamlit:
            st.pyplot(fig)
        else:
            plt.show()

    # Normalization using minimum joint entropy (log2 min)
    je = joint_entropy(metric_arr, hrv_arr)
    min_je = np.log2(je) if je > 0 else 1  # avoid log2(0)
    metric_norm = metric_arr / min_je
    hrv_norm = hrv_arr / min_je
    fig_norm, ax_norm = plt.subplots(figsize=(6, 4))
    ax_norm.plot(t, metric_norm, label=f'{metric_name} (norm)')
    ax_norm.plot(t, hrv_norm, label=f'{hrv_label} (norm)')
    try:
        stat_norm, p_norm = wilcoxon(metric_norm, hrv_norm)
    except ValueError:
        stat_norm, p_norm = np.nan, np.nan
    ax_norm.set_title(f'Normalized {metric_name} vs {hrv_label}\nWilcoxon signed-rank stat={stat_norm:.2f}, p={p_norm:.2g}')
    ax_norm.set_xlabel('Time (s)')
    ax_norm.legend()
    plt.tight_layout()
    if save_plot:
        fname_norm = f"{save_dir}/{edf_name}_{metric_name.lower()}_{hrv_label.lower()}_norm.png"
        fig_norm.savefig(fname_norm, dpi=300, bbox_inches='tight')
        plt.close(fig_norm)
    else:
        if 'is_streamlit' in globals() and is_streamlit:
            st.pyplot(fig_norm)
        else:
            plt.show()

    # Save Wilcoxon signed-rank results to a text file
    if save_plot:
        fname_txt = f"{save_dir}/{edf_name}_{metric_name.lower()}_{hrv_label.lower()}_wilcoxon_signedrank.txt"
        with open(fname_txt, 'w') as f:
            f.write(f"Wilcoxon signed-rank stat (raw): {stat_raw:.4f}, p-value: {p_raw:.4g}\n")
            f.write(f"Wilcoxon signed-rank stat (norm): {stat_norm:.4f}, p-value: {p_norm:.4g}\n")

def compute_brain_heart_coupling(edf_results, key, motor_symptoms=None, bool_plots=False, save_plot=False, edf_pickle_name="",step_sec=5):
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
    save_plot : bool
        If True, save the plot to a file.
    edf_pickle_name : str
        Base name for the EDF pickle file, used for saving plot filenames.

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
    step_sec = step_sec
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
    mi_results = {}
    metrics = {'Efficiency': eff_arr, 'Clustering': clu_arr,
               'Modularity': mod_arr, 'Assortativity': ass_arr}
    for name, arr in metrics.items():
        X = arr.reshape(-1, 1)
        mic_sym = mutual_info_regression(X, csi_arr, random_state=0)[0]
        mic_vag = mutual_info_regression(X, cvi_arr, random_state=0)[0]
        mi_results[name] = {'Sympathetic MI': mic_sym, 'Vagal MI': mic_vag}

    # Build results_df with all arrays as columns
    results_df = pd.DataFrame({
        'Efficiency': eff_arr,
        'Clustering': clu_arr,
        'Modularity': mod_arr,
        'Assortativity': ass_arr,
        'Vagal_SD1': cvi_arr,
        'Sympathetic_SD2': csi_arr
    })
    # Optionally, add MI results as a separate DataFrame or as metadata
    results_df.attrs['mutual_info'] = mi_results

    # Now loop over the DataFrame to plot
    # for idx, row in results_df.iterrows():
    #     t = np.arange(len(row['Efficiency'])) * step_sec
    #     save_dir = "figures_HEP/compute_brain_heart_coupling"
    #     edf_pickle_name = row['edf_pickle_name']  # or the relevant identifier
    
    #     # Plot Vagal vs Sympathetic
    #     _plot_metric_vs_hrv(t, row['Vagal_SD1'], row['Sympathetic_SD2'],
    #                         'Vagal_SD1', 'Sympathetic_SD2', save_plot, edf_pickle_name, save_dir)
    
    #     # Plot each metric vs Sympathetic and Vagal
    #     for m_name in ['Efficiency', 'Clustering', 'Modularity', 'Assortativity']:
    #         _plot_metric_vs_hrv(t, row[m_name], row['Sympathetic_SD2'],
    #                             m_name, 'Sympathetic_SD2', save_plot, edf_pickle_name, save_dir)
    #         _plot_metric_vs_hrv(t, row[m_name], row['Vagal_SD1'],
    #                             m_name, 'Vagal_SD1', save_plot, edf_pickle_name, save_dir)


    return results_df

def only_plots(results_df,save_plot,save_dir, step_sec=5):
    
    # Now loop over the DataFrame to plot
    for idx, row in results_df.iterrows():
        t = np.arange(len(row['Efficiency'])) * step_sec
        edf_pickle_name = row['edf_pickle_name']  # or the relevant identifier
    
        # Plot Vagal vs Sympathetic
        _plot_metric_vs_hrv(t, row['Vagal_SD1'], row['Sympathetic_SD2'],
                            'Vagal_SD1', 'Sympathetic_SD2', save_plot, edf_pickle_name, save_dir)
    
        # Plot each metric vs Sympathetic and Vagal
        for m_name in ['Efficiency', 'Clustering', 'Modularity', 'Assortativity']:
            _plot_metric_vs_hrv(t, row[m_name], row['Sympathetic_SD2'],
                                m_name, 'Sympathetic_SD2', save_plot, edf_pickle_name, save_dir)
            _plot_metric_vs_hrv(t, row[m_name], row['Vagal_SD1'],
                                m_name, 'Vagal_SD1', save_plot, edf_pickle_name, save_dir)
            
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
                    ### DEV RUN
                    # break
            except Exception as e:
                print(f"Error loading {fname}: {e}")
                results[fname] = None
    return results

def process_all_patients(edf_results, step_sec=5):
    os.makedirs('temps_EDF_HEP', exist_ok=True)
    patient_results = {}
    # 1. Compute and save all results_df per scan
    for edf_key, raw in edf_results.items():
        # Extract patient ID (e.g., '010' from '0345-010.edf_600_1.pkl')
        patient_id = edf_key.split('-')[1].split('.')[0]
        edf_pickle_name = patient_id
        results_df = compute_brain_heart_coupling(
            edf_results, edf_key, bool_plots=False, save_plot=False,
            edf_pickle_name=edf_pickle_name, step_sec=step_sec
        )
        # Save per scan
        results_path = f"temps_EDF_HEP/{edf_key}_results.csv"
        results_df.to_csv(results_path, index=False)
        # Collect for patient
        if patient_id not in patient_results:
            patient_results[patient_id] = []
        patient_results[patient_id].append(results_df)
    # 2. Average all results_df for each patient, then plot
    for patient_id, dfs in patient_results.items():
        # Align columns and average
        avg_df = pd.concat(dfs).groupby(level=0).mean(numeric_only=True)
        # Use the first edf_pickle_name for naming
        edf_pickle_name = patient_id
        only_plots(avg_df, edf_pickle_name=edf_pickle_name, save_plot=True)
    # 3. Average across all patients and plot
    if patient_results:
        # Concatenate all patient average DataFrames
        all_avg_dfs = []
        for dfs in patient_results.values():
            # Each dfs is a list of per-scan DataFrames for a patient
            avg_df = pd.concat(dfs).groupby(level=0).mean(numeric_only=True)
            all_avg_dfs.append(avg_df)
        # Now average across all patients
        grand_avg_df = pd.concat(all_avg_dfs).groupby(level=0).mean(numeric_only=True)
        # Use a generic name for the group plot
        only_plots(grand_avg_df, edf_pickle_name='ALL_PATIENTS', save_plot=True)

# Example usage:
edf_results = load_edf_pickles_with_ecg()
edf_results = {k: v for k, v in edf_results.items() if v is not None}
process_all_patients(edf_results, step_sec=5)

print("Processing and plotting complete for all patients.")