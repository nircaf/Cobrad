import sys
import os

def check_tmux():
    if 'TMUX' in os.environ:
        return True
    return False

is_tmux = check_tmux()

import mne 
import os 
import pandas as pd
import pickle
import numpy as np
import neurokit2 as nk
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import dabest
from sklearn.feature_selection import mutual_info_regression
from mne_connectivity import SpectralConnectivity as spectral_connectivity
from bct import efficiency_bin, transitivity_bu, modularity_und, assortativity_bin
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, zscore, pearsonr
from scipy.signal import coherence
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities, modularity
from scipy.stats import pearsonr, entropy, ranksums
from scipy.signal import coherence, windows
from itertools import combinations
from scipy.stats import linregress
import seaborn as sns

try:
    import streamlit as st
    is_streamlit = True
except ImportError:
    is_streamlit = False
from scipy.stats import zscore

def clean_ecg_signal(ecg_signal, sfreq, lowcut=0.5, highcut=40, order=4):
    """
    Bandpass filter the ECG signal to remove noise and baseline wander.
    """
    nyq = 0.5 * sfreq
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, ecg_signal)

def plot_ecg_signal(ecg_signal, sfreq, edf_pickle_name, save_plot, save_dir, label, bool_plots=True):
    if not bool_plots:
        return
    os.makedirs(save_dir, exist_ok=True)
    # Plot full signal
    plt.figure(figsize=(10, 4))
    plt.plot(ecg_signal)
    plt.title(f'ECG Signal {label}')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    if save_plot:
        fname = f"{save_dir}/{edf_pickle_name}_ecg_signal_{label}.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')
    # Plot a 60-120 second segment (if available)
    seg_start_sec = 60
    seg_end_sec = 120
    seg_start = int(seg_start_sec * sfreq)
    seg_end = int(seg_end_sec * sfreq)
    if len(ecg_signal) > seg_end:
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(seg_start, seg_end) / sfreq, ecg_signal[seg_start:seg_end])
        plt.title(f'ECG Segment {seg_start_sec}-{seg_end_sec}s {label}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        if save_plot:
            fname = f"{save_dir}/{edf_pickle_name}_ecg_signal_{seg_start_sec}to{seg_end_sec}s_{label}.png"
            plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()


def filter_by_ecg_quality(ecg_clean, data_all, ecg_quality, threshold=0.5):
    """
    Remove timepoints from ecg_clean and data_all where ecg_quality <= threshold.
    Returns filtered ecg_clean and filtered data_all.
    """
    mask = ecg_quality > threshold
    ecg_clean_filt = ecg_clean[mask]
    if data_all.ndim == 2:
        data_all_filt = data_all[:, mask]
    else:
        data_all_filt = data_all[mask]
    return ecg_clean_filt, data_all_filt

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

from functools import partial

def _plot_metric_vs_hrv(t, arrs, labels, save_plot, edf_name, save_dir):
    """
    Plot multiple arrays (metrics/HRV indices) over time, with optional saving.
    Also saves normalized (min joint entropy) plot and Wilcoxon signed-rank test values for the first two arrays.
    Parameters
    ----------
    t : array-like
        Time axis.
    arrs : list of np.ndarray
        List of arrays to plot.
    labels : list of str
        List of labels for each array.
    save_plot : bool
        Whether to save the plot.
    edf_name : str
        Name for saving.
    save_dir : str
        Directory to save plots.
    """


    # If exactly two arrays, use dabest for paired estimation
    if len(arrs) == 2:
        # Prepare data for dabest: long-form DataFrame
        df = pd.DataFrame({
            'value': np.concatenate([arrs[0], arrs[1]]),
            'group': [labels[0]] * len(arrs[0]) + [labels[1]] * len(arrs[1]),
            'pair_id': list(range(len(arrs[0]))) + list(range(len(arrs[1])))
        })
        # Paired estimation plot
        dabest_obj = dabest.load(
            data=df,
            x='group',
            y='value',
            id_col='pair_id',
            paired=True
        )
        fig = dabest_obj.plot(
            swarm_label="Value",
            contrast_label="Mean difference",
            custom_palette=None,
            show_pairs=True,
            raw_marker_size=4,
            halfviolin_width=0.6,
            fig_size=(6, 4)
        )
        if save_plot:
            os.makedirs(save_dir, exist_ok=True)
            fname = f"{save_dir}/dabest_{labels[0]}_vs_{labels[1]}_{edf_name}.png"
            fig.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            if 'is_streamlit' in globals() and is_streamlit:
                st.pyplot(fig)
            else:
                plt.show()
    elif len(arrs) > 2:
        # For more than two arrays, plot all as swarm plots (no dabest)
        n = len(arrs[0])
        df = pd.DataFrame({
            'value': np.concatenate(arrs),
            'group': np.concatenate([[label]*n for label in labels]),
            'pair_id': np.tile(np.arange(n), len(arrs))
        })
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.swarmplot(data=df, x='group', y='value', ax=ax)
        ax.set_title("Swarm plot of all metrics and HRV indices")
        ax.set_xlabel('Metric/Index')
        ax.set_ylabel('Value')
        plt.tight_layout()
        if save_plot:
            os.makedirs(save_dir, exist_ok=True)
            fname = f"{save_dir}/swarmplot_all_{edf_name}.png"
            fig.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            if 'is_streamlit' in globals() and is_streamlit:
                st.pyplot(fig)
            else:
                plt.show()
        # Also plot all arrays as time series (x-y plot)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for arr, label in zip(arrs, labels):
            ax2.plot(t, arr, label=label)
        ax2.set_title("Time series of all metrics and HRV indices")
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Value')
        ax2.legend()
        plt.tight_layout()
        if save_plot:
            fname2 = f"{save_dir}/xyplot_all_{edf_name}.png"
            fig2.savefig(fname2, dpi=300, bbox_inches='tight')
            plt.close(fig2)
        else:
            if 'is_streamlit' in globals() and is_streamlit:
                st.pyplot(fig2)
            else:
                plt.show()
        # Also plot all arrays as pairwise scatterplots with regression (PairGrid)
        df_pair = pd.DataFrame({label: arr for label, arr in zip(labels, arrs)})
        g = sns.PairGrid(df_pair, diag_sharey=False)
        def scatter_with_stats(x, y, **kwargs):
            ax = plt.gca()
            ax.scatter(x, y, alpha=0.7)
            # Regression line
            if len(x) > 1 and len(y) > 1:
                sns.regplot(x=x, y=y, scatter=False, ax=ax, color='red', truncate=False)
                r, p = pearsonr(x, y)
                ax.annotate(f"$R^2$={r**2:.2f}\np={p:.3g}",
                            xy=(0.05, 0.85), xycoords='axes fraction', fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))
        g.map_upper(scatter_with_stats)
        # Lower: plot x and y as functions of t
        def lineplot_with_time(x, y, t, **kwargs):
            ax = plt.gca()
            ax.plot(t, x, label='x', alpha=0.7)
            ax.plot(t, y, label='y', alpha=0.7)
            # Compute and annotate R^2 and p-value
            from scipy.stats import pearsonr
            if len(x) > 1 and len(y) > 1:
                r, p = pearsonr(x, y)
                ax.annotate(f"$R^2$={r**2:.2f}\np={p:.3g}",
                            xy=(0.05, 0.85), xycoords='axes fraction', fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))
            ax.legend(fontsize=8)
            ax.set_xlabel('t')
            ax.set_ylabel('Value')
        g.map_lower(partial(lineplot_with_time, t=t))
        g.map_diag(sns.histplot, kde=True)
        plt.suptitle("Pairwise relationships of all metrics and HRV indices", y=1.02)
        plt.tight_layout()
        if save_plot:
            fname2 = f"{save_dir}/pairgrid_all_{edf_name}.png"
            plt.savefig(fname2, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            if 'is_streamlit' in globals() and is_streamlit:
                st.pyplot(plt.gcf())
            else:
                plt.show()
    else:
        # Fallback: plot single array as line plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t, arrs[0], label=labels[0])
        ax.set_title(labels[0])
        ax.set_xlabel('Time (s)')
        ax.legend()
        plt.tight_layout()
        if save_plot:
            os.makedirs(save_dir, exist_ok=True)
            fname = f"{save_dir}/{labels[0]}_{edf_name}.png"
            fig.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            if 'is_streamlit' in globals() and is_streamlit:
                st.pyplot(fig)
            else:
                plt.show()

def compute_network_features(eeg_data, sfreq, freq_band, threshold_density=0.2):
    """
    Compute network features from EEG data using coherence in a given frequency band.

    Parameters:
        eeg_data (ndarray): EEG data (channels x samples)
        sfreq (float): Sampling frequency
        freq_band (tuple): Frequency band (low, high) in Hz
        threshold_density (float): Proportion of top connections to retain in binarized graph

    Returns:
        dict: Network metrics (clustering, efficiency, assortativity, modularity)
    """
    n_channels = eeg_data.shape[0]
    coh_matrix = np.zeros((n_channels, n_channels))

    # Compute coherence between all pairs
    for i, j in combinations(range(n_channels), 2):
        f, Cxy = coherence(eeg_data[i], eeg_data[j], fs=sfreq, window=windows.hann(sfreq),
                           nperseg=sfreq, noverlap=sfreq//2)
        freq_mask = (f >= freq_band[0]) & (f <= freq_band[1])
        mean_coh = np.mean(Cxy[freq_mask])
        coh_matrix[i, j] = coh_matrix[j, i] = mean_coh

    # Binarize: keep top X% of connections (thresholding by density)
    n_possible = n_channels * (n_channels - 1) / 2
    n_edges = int(threshold_density * n_possible)
    triu_vals = coh_matrix[np.triu_indices(n_channels, k=1)]
    thresh_val = np.sort(triu_vals)[-n_edges] if n_edges > 0 else 0
    binarized = (coh_matrix >= thresh_val).astype(int)
    np.fill_diagonal(binarized, 0)

    # Build graph and compute features
    G = nx.from_numpy_array(binarized)

    clustering = nx.transitivity(G)
    efficiency = nx.global_efficiency(G)
    assortativity = nx.degree_pearson_correlation_coefficient(G)
    modularity = nx.algorithms.community.modularity(G, nx.community.label_propagation_communities(G))

    return efficiency, clustering, assortativity, modularity
    {
        'efficiency': efficiency,
        'clustering': clustering,
        'assortativity': assortativity,
        'modularity': modularity
    }

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
    print(f"Starting compute_brain_heart_coupling for key: {key}")
    raw = edf_results[key]
    data_all = raw.get_data()
    data_len = data_all.shape[1]
    ch_names = raw.ch_names
    print(f"Channels found: {ch_names}")
    eeg_channels = ['Fpz', 'F7', 'T3', 'T5', 'Fp1', 'F3', 'C3', 'P3', 'Oz', 'F8', 'T4', 'T6', 'Fp2', 'F4', 'C4', 'P4', 'Fz', 'Cz']
    sfreq = int(raw.info['sfreq'])
    print(f"Sampling frequency: {sfreq}")

    # Extract ECG and detect R-peaks
    if 'ecg' not in ch_names:
        raise ValueError("No 'ecg' channel found")
    print("Extracting ECG signal and detecting R-peaks...")
    ecg_signal = data_all[ch_names.index('ecg')]
    save_dir = "figures_HEP/compute_brain_heart_coupling"
    # Plot raw ECG
    plot_ecg_signal(ecg_signal, sfreq, edf_pickle_name, save_plot, save_dir, label="raw", bool_plots=bool_plots)
    # Process ECG signal
    signals, info = nk.ecg_process(ecg_signal, sampling_rate=sfreq)
    ecg_clean = signals['ECG_Clean'].values
    # Plot cleaned ECG
    plot_ecg_signal(ecg_clean, sfreq, edf_pickle_name, save_plot, save_dir, label="cleaned", bool_plots=bool_plots)
    # Remove from ecg_clean and from data_all where signals['ECG_Quality'] <= 0.5
    ecg_clean, data_all = filter_by_ecg_quality(ecg_clean, data_all, signals['ECG_Quality'].values, threshold=0.5)
    plot_ecg_signal(ecg_clean, sfreq, edf_pickle_name, save_plot, save_dir, label="quality_cleaned_filtered", bool_plots=bool_plots)
    # Use cleaned ECG for peak detection
    _, rpk = nk.ecg_peaks(ecg_clean, sampling_rate=sfreq)
    rpeaks = rpk['ECG_R_Peaks']
    # Times in seconds
    r_times = rpeaks / sfreq
    rpeaks = rpk['ECG_R_Peaks']
    # Times in seconds
    r_times = rpeaks / sfreq
    # --- Print average HR per minute ---
    if len(r_times) > 1:
        total_duration = r_times[-1]
        num_minutes = int(np.ceil(total_duration / 60))
        print("Average HR per minute:")
        for minute in range(num_minutes):
            start_t = minute * 60
            end_t = (minute + 1) * 60
            # R-peaks in this minute
            idx = np.where((r_times >= start_t) & (r_times < end_t))[0]
            if len(idx) > 1:
                rr_intervals = np.diff(r_times[idx])
                avg_hr = 60.0 / np.mean(rr_intervals)
                print(f"  Minute {minute+1}: {avg_hr:.2f} bpm")
            else:
                print(f"  Minute {minute+1}: insufficient data")
    else:
        print("Not enough R-peaks to compute HR per minute.")

    # Extract EEG data (excluding ECG)
    data = data_all[[ch_names.index(ch) for ch in eeg_channels if ch in ch_names]]
    n_nodes, n_samples = data.shape
    print(f"EEG data shape: {data.shape}")
    # Sliding window parameters
    w_eeg_sec = 15
    step_sec = step_sec
    w_eeg = int(w_eeg_sec * sfreq)
    step = int(step_sec * sfreq)
    n_windows = int((n_samples - w_eeg) / step) + 1
    print(f"Number of windows: {n_windows}, Window size: {w_eeg}, Step: {step}")

    # Initialize time series dicts for each band
    bands = {
        "delta": (1, 4),
        "alpha": (8, 12),
        "beta": (12, 30),
        "gamma": (30, 45)
    }
    eff_ts = {band: [] for band in bands}
    clu_ts = {band: [] for band in bands}
    mod_ts = {band: [] for band in bands}
    ass_ts = {band: [] for band in bands}
    cvi_ts, csi_ts = [], []

    for w in range(n_windows):
        start = w * step
        end = start + w_eeg
        segment = data[:, start:end]
        print(f"Processing window {w+1}/{n_windows} (samples {start}:{end})")
        band_features = {}
        for band_name, (fmin, fmax) in bands.items():
            eff, clu, mod, ass = compute_network_features(segment,sfreq,[fmin,fmax])
            band_features[band_name] = (eff, clu, mod, ass)
        # Store features for each band
        for band_name, (eff, clu, mod, ass) in band_features.items():
            eff_ts[band_name].append(eff)
            clu_ts[band_name].append(clu)
            mod_ts[band_name].append(mod)
            ass_ts[band_name].append(ass)

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

    # Convert to arrays, mask out NaNs, and plot for each band
    cvi_arr = np.array(cvi_ts)
    csi_arr = np.array(csi_ts)
    save_dir = "figures_HEP/compute_brain_heart_coupling"
    edf_pickle_name = edf_pickle_name # or the relevant identifier

    results_df_dict = {}
    for band in bands:
        eff_arr = np.array(eff_ts[band])
        clu_arr = np.array(clu_ts[band])
        mod_arr = np.array(mod_ts[band])
        ass_arr = np.array(ass_ts[band])
        # Mask out NaNs in HRV
        valid = ~np.isnan(cvi_arr) & ~np.isnan(csi_arr)
        eff_arr, clu_arr = eff_arr[valid], clu_arr[valid]
        mod_arr, ass_arr = mod_arr[valid], ass_arr[valid]
        cvi_arr_band, csi_arr_band = cvi_arr[valid], csi_arr[valid]
        print(f"Valid samples after NaN filtering for {band}: {len(eff_arr)}")

        # Mutual information coupling
        mi_results = {}
        metrics = {'Efficiency': eff_arr, 'Clustering': clu_arr,
                   'Modularity': mod_arr, 'Assortativity': ass_arr}
        for name, arr in metrics.items():
            X = arr.reshape(-1, 1)
            mic_sym = np.nan
            mic_vag = np.nan
            # mic_sym = mutual_info_regression(X, csi_arr_band, random_state=0)[0]
            # mic_vag = mutual_info_regression(X, cvi_arr_band, random_state=0)[0]
            # print(f"MI for {name} ({band}): Sympathetic={mic_sym:.4f}, Vagal={mic_vag:.4f}")
            mi_results[name] = {'Sympathetic MI': mic_sym, 'Vagal MI': mic_vag}

        # Build results_df with all arrays as columns
        results_df = pd.DataFrame({
            'Efficiency': eff_arr,
            'Clustering': clu_arr,
            'Modularity': mod_arr,
            'Assortativity': ass_arr,
            'Vagal_SD1': cvi_arr_band,
            'Sympathetic_SD2': csi_arr_band
        })
        results_df.attrs['mutual_info'] = mi_results
        results_df_dict[band] = results_df

        # Plot all metrics and HRV indices together for this band
        t = np.arange(len(eff_arr)) * step_sec
        arrs = [cvi_arr_band, csi_arr_band, eff_arr, clu_arr, mod_arr, ass_arr]
        labels = ['Vagal_SD1', 'Sympathetic_SD2', 'Efficiency', 'Clustering', 'Modularity', 'Assortativity']
        _plot_metric_vs_hrv(
            t, arrs, labels, save_plot, f"{edf_pickle_name}_{band}", save_dir
        )
        # Also plot z-scored arrays
        arrs_zscore = [zscore(arr) if np.std(arr) > 0 else arr for arr in arrs]
        _plot_metric_vs_hrv(
            t, arrs_zscore, labels, save_plot, f"{edf_pickle_name}_{band}_zscore", save_dir
        )

    # Return the results for the alpha band for compatibility
    return results_df_dict

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
    print(f"Loading EDF pickles from directory: {pickle_dir}")
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
                    print(f"File {fname} does not contain ECG channel. Skipping.")
                    results[fname] = None
                else:
                    results[fname] = raw
                    if not is_tmux:
                        ### DEV RUN
                        break
            except Exception as e:
                print(f"Error loading {fname}: {e}")
                results[fname] = None
    print(f"Finished loading EDF pickles. Total valid: {sum(1 for v in results.values() if v is not None)} / {len(results)}")
    return results

def process_all_patients(edf_results, step_sec=5):
    print(f"Processing all patients, step_sec={step_sec}")
    os.makedirs('temps_EDF_HEP', exist_ok=True)
    patient_results = {}
    # 1. Compute and save all results_df per scan
    for edf_key, raw in edf_results.items():
        print(f"Processing scan: {edf_key}")
        # Extract patient ID (e.g., '010' from '0345-010.edf_600_1.pkl')
        patient_id = edf_key.split('-')[1].split('.')[0]
        edf_pickle_name = patient_id
        results_df_dict = compute_brain_heart_coupling(
            edf_results, edf_key, bool_plots=True, save_plot=True,
            edf_pickle_name=edf_pickle_name, step_sec=step_sec
        )
        # Save per scan, per band
        for band, results_df in results_df_dict.items():
            results_path = f"temps_EDF_HEP/{edf_key}_results_{band}.csv"
            print(f"Saving results for scan {edf_key}, band {band} to {results_path}")
            results_df.to_csv(results_path, index=False)
        # Collect for patient (use alpha band for compatibility)
        if patient_id not in patient_results:
            patient_results[patient_id] = []
        patient_results[patient_id].append(results_df_dict.get("alpha", None))
    # 2. Average all results_df for each patient, then plot
    for patient_id, dfs in patient_results.items():
        print(f"Averaging and plotting for patient: {patient_id}")
        # Align columns and average
        avg_df = pd.concat(dfs).groupby(level=0).mean(numeric_only=True)
        # Use the first edf_pickle_name for naming
        edf_pickle_name = patient_id
        only_plots(avg_df, edf_pickle_name=edf_pickle_name, save_plot=True)
    # 3. Average across all patients and plot
    if patient_results:
        print("Averaging and plotting across all patients")
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
    print("All processing and plotting complete.")

# Example usage:
edf_results = load_edf_pickles_with_ecg()
edf_results = {k: v for k, v in edf_results.items() if v is not None}
process_all_patients(edf_results, step_sec=5)

print("Processing and plotting complete for all patients.")
