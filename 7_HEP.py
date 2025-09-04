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
from statsmodels.stats.multitest import fdrcorrection
import glob
try:
    import streamlit as st
    is_streamlit = True
except ImportError:
    is_streamlit = False
from scipy.stats import zscore
save_dir = "figures_HEP/compute_brain_heart_coupling"
# Initialize time series dicts for each band
bands = {
    "delta": (1, 4),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 45)
}
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
    Interpolate (not remove) timepoints in ecg_clean and data_all where ecg_quality <= threshold.
    Returns interpolated ecg_clean and data_all.
    """
    mask = ecg_quality > threshold
    # Interpolate ecg_clean
    ecg_clean_interp = np.copy(ecg_clean)
    if not np.all(mask):
        x = np.arange(len(ecg_quality))
        good = mask
        bad = ~mask
        ecg_clean_interp[bad] = np.interp(x[bad], x[good], ecg_clean[good])
    # Interpolate data_all
    # if data_all.ndim == 2:
    #     data_all_interp = np.copy(data_all)
    #     for i in range(data_all.shape[0]):
    #         if not np.all(mask):
    #             data_all_interp[i, bad] = np.interp(x[bad], x[good], data_all[i, good])
    # else:
    #     data_all_interp = np.copy(data_all)
    #     if not np.all(mask):
    #         data_all_interp[bad] = np.interp(x[bad], x[good], data_all[good])
    return ecg_clean_interp, data_all

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
from scipy.stats import linregress
def _plot_metric_vs_hrv(t, arrs, labels, save_plot, edf_name, save_dir,is_streamlit=False):
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
            if is_streamlit:
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
            if is_streamlit:
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
            if is_streamlit:
                st.pyplot(fig2)
            else:
                plt.show()

        # ------------------ PairGrid with scatter and line plots ------------------
        df_pair = pd.DataFrame({label: arr for label, arr in zip(labels, arrs)})
        g = sns.PairGrid(df_pair, diag_sharey=False)
        # Compute all unique pairwise p-values for FDR correction
        n_vars = len(df_pair.columns)
        var_names = list(df_pair.columns)
        pair_indices = []
        r_vals = []
        p_vals = []
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                x = df_pair.iloc[:, i]
                y = df_pair.iloc[:, j]
                mask = ~np.isnan(x) & ~np.isnan(y)
                x_clean = np.array(x)[mask]
                y_clean = np.array(y)[mask]
                if len(x_clean) > 1 and len(y_clean) > 1:
                    r, p = pearsonr(x_clean, y_clean)
                else:
                    r, p = np.nan, np.nan
                pair_indices.append((i, j))
                r_vals.append(r)
                p_vals.append(p)
        # FDR correction
        p_vals_array = np.array([p if not np.isnan(p) else 1.0 for p in p_vals])
        reject, pvals_fdr = fdrcorrection(p_vals_array, alpha=0.05, method='indep')
        # Map (i, j) -> (r, p_fdr)
        pair_to_stats = {}
        for idx, (i, j) in enumerate(pair_indices):
            pair_to_stats[(i, j)] = (r_vals[idx], pvals_fdr[idx])

        def scatter_with_stats(x, y, **kwargs):
            ax = kwargs.get('ax', plt.gca())
            # Find which pair this is
            i, j = None, None
            for idx, col in enumerate(var_names):
                if np.all(x == df_pair[col]):
                    i = idx
                if np.all(y == df_pair[col]):
                    j = idx
            if i is not None and j is not None and i != j:
                key = (min(i, j), max(i, j))
                r, p_fdr = pair_to_stats.get(key, (np.nan, np.nan))
            else:
                r, p_fdr = np.nan, np.nan
            # Scatter plot for the joint distribution
            mask = ~np.isnan(x) & ~np.isnan(y)
            x_clean = np.array(x)[mask]
            y_clean = np.array(y)[mask]
            sns.scatterplot(x=x_clean, y=y_clean, ax=ax, alpha=0.7, s=20)
            # Optionally add regression line
            if len(x_clean) > 1 and len(y_clean) > 1:
                sns.regplot(x=x_clean, y=y_clean, ax=ax, scatter=False, color='gray', line_kws={'alpha':0.5})
            # Annotate with r^2 and FDR-corrected p-value only if significant
            if not np.isnan(r) and not np.isnan(p_fdr) and p_fdr < 0.05:
                ax.annotate(f"$R^2$={r**2:.2f}\np={p_fdr:.3g}",
                            xy=(0.05, 0.85), xycoords='axes fraction', fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))

        def lineplot_with_time(x, y, **kwargs):
            ax = kwargs.get('ax', plt.gca())
            # t_temp = linspace between min and max of x and y
            t_temp = np.linspace(min(min(x), min(y)), max(max(x), max(y)), num=len(x))
            # For time plots, use the same FDR-corrected p-value as for the pair
            i, j = None, None
            for idx, col in enumerate(var_names):
                if np.all(x == df_pair[col]):
                    i = idx
                if np.all(y == df_pair[col]):
                    j = idx
            if i is not None and j is not None and i != j:
                key = (min(i, j), max(i, j))
                r, p_fdr = pair_to_stats.get(key, (np.nan, np.nan))
            else:
                r, p_fdr = np.nan, np.nan
            ax.plot(t_temp, x, label='x', alpha=0.7)
            ax.plot(t_temp, y, label='y', alpha=0.7)
            if not np.isnan(r) and not np.isnan(p_fdr) and p_fdr < 0.05:
                ax.annotate(f"$R^2$={r**2:.2f}\np={p_fdr:.3g}",
                            xy=(0.05, 0.85), xycoords='axes fraction', fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))
            ax.legend(fontsize=8)
            ax.set_xlabel('t')
            ax.set_ylabel('Value')

        g.map_diag(sns.histplot, kde=True, bins=10)
        g.map_upper(scatter_with_stats)
        g.map_lower(lineplot_with_time)
        g.fig.suptitle("Pairwise relationships", y=1.02)
        plt.tight_layout()
        if save_plot:
            os.makedirs(save_dir, exist_ok=True)
            fname2 = f"{save_dir}/pairgrid_all_{edf_name}.png"
            g.savefig(fname2, dpi=300, bbox_inches='tight')
            plt.close(g.fig)
        else:
            if is_streamlit:
                st.pyplot(g.fig)
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
            if is_streamlit:
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
    save_dir = "figures_HEP/compute_brain_heart_coupling"

    # Extract ECG and detect R-peaks
    if 'ecg' not in ch_names:
        raise ValueError("No 'ecg' channel found")
    print("Extracting ECG signal and detecting R-peaks...")
    ecg_signal = data_all[ch_names.index('ecg')]
    # Plot raw ECG
    plot_ecg_signal(ecg_signal, sfreq, edf_pickle_name, save_plot, save_dir, label="raw", bool_plots=bool_plots)
    # Bandpass filter ECG before processing
    ecg_signal_filt = clean_ecg_signal(ecg_signal, sfreq, lowcut=0.5, highcut=40, order=4)
    plot_ecg_signal(ecg_signal_filt, sfreq, edf_pickle_name, save_plot, save_dir, label="bandpass_filtered", bool_plots=bool_plots)
    # Process ECG signal
    signals, info = nk.ecg_process(ecg_signal_filt, sampling_rate=sfreq)
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
        from scipy.signal import medfilt
        total_duration = r_times[-1]
        num_minutes = int(np.ceil(total_duration / 60))
        hr_per_minute = []
        for minute in range(num_minutes):
            start_t = minute * 60
            end_t = (minute + 1) * 60
            idx = np.where((r_times >= start_t) & (r_times < end_t))[0]
            if len(idx) > 1:
                rr_intervals = np.diff(r_times[idx])
                avg_hr = 60.0 / np.mean(rr_intervals)
                hr_per_minute.append(avg_hr)
            else:
                hr_per_minute.append(np.nan)
        # Remove implausible HR values
        hr_per_minute = np.array(hr_per_minute)
        hr_per_minute[(hr_per_minute < 40) | (hr_per_minute > 180)] = np.nan
        # Median filter (window size 3)
        print("Average HR per minute (raw):")
        for i, hr in enumerate(hr_per_minute):
            print(f"  Minute {i+1}: {hr:.2f} bpm" if not np.isnan(hr) else f"  Minute {i+1}: insufficient/invalid data")
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
        # Add z-scored columns
        results_df['Vagal_SD1_zscore'] = zscore(cvi_arr_band) if np.std(cvi_arr_band) > 0 else cvi_arr_band
        results_df['Sympathetic_SD2_zscore'] = zscore(csi_arr_band) if np.std(csi_arr_band) > 0 else csi_arr_band
        results_df['Efficiency_zscore'] = zscore(eff_arr) if np.std(eff_arr) > 0 else eff_arr
        results_df['Clustering_zscore'] = zscore(clu_arr) if np.std(clu_arr) > 0 else clu_arr
        results_df['Modularity_zscore'] = zscore(mod_arr) if np.std(mod_arr) > 0 else mod_arr
        results_df['Assortativity_zscore'] = zscore(ass_arr) if np.std(ass_arr) > 0 else ass_arr

        results_df.attrs['mutual_info'] = mi_results
        results_df_dict[band] = results_df

        # Plot all metrics and HRV indices together for this band
        t = np.arange(len(eff_arr)) * step_sec
        arrs = [cvi_arr_band, csi_arr_band, eff_arr, clu_arr, mod_arr, ass_arr]
        labels = ['Vagal_SD1', 'Sympathetic_SD2', 'Efficiency', 'Clustering', 'Modularity', 'Assortativity']
        # Also plot z-scored arrays
        arrs_zscore = [zscore(arr) if np.std(arr) > 0 else arr for arr in arrs]
        if bool_plots:
            _plot_metric_vs_hrv(
                t, arrs_zscore, labels, save_plot, f"{edf_pickle_name}_{band}_zscore", save_dir
            )

    # Return the results for the alpha band for compatibility
    return results_df_dict

def only_plots(results_df, save_plot, save_dir, edf_pickle_name="plot", band="band", step_sec=5):
    """
    Utility to plot all z-scored metrics and HRV indices from a results_df.
    """
    # Define the order and labels
    zscore_cols = [
        'Vagal_SD1_zscore',
        'Sympathetic_SD2_zscore',
        'Efficiency_zscore',
        'Clustering_zscore',
        'Modularity_zscore',
        'Assortativity_zscore'
    ]
    labels = [
        'Vagal_SD1',
        'Sympathetic_SD2',
        'Efficiency',
        'Clustering',
        'Modularity',
        'Assortativity'
    ]
    arrs_zscore = [results_df[col].values for col in zscore_cols if col in results_df]
    t = np.arange(len(results_df)) * step_sec
    _plot_metric_vs_hrv(
        t, arrs_zscore, labels, save_plot, f"{edf_pickle_name}_{band}_zscore", save_dir
    )

def plot_patient_band_means(patient_id, bands=None, step_sec=5, temps_dir="temps_EDF_HEP", save_dir="figures_HEP/compute_brain_heart_coupling"):
    """
    For a given patient_id, for each band, read all the patient's .csv files, average them, then run only_plots with the mean DataFrame.
    """
    for band in bands:
        # Find all files for this patient and band
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
        # Align columns and average row-wise using nanmean logic
        # 1. Get union of all columns
        all_columns = set()
        for df in dfs:
            all_columns.update(df.columns)
        all_columns = list(all_columns)
        # 2. Reindex columns and rows for all dfs
        max_len = max(len(df) for df in dfs)
        dfs_aligned = []
        for df in dfs:
            df_aligned = df.reindex(columns=all_columns)
            df_aligned = df_aligned.reindex(range(max_len)).reset_index(drop=True)
            dfs_aligned.append(df_aligned)
        # 3. Stack into 3D array (n_files, n_rows, n_cols)
        arrs = np.stack([d.values for d in dfs_aligned], axis=0)
        # 4. For each cell, count non-NaN, if >= half, nanmean, else NaN
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
        # Drop rows where all columns are NaN (or, optionally, where most columns are NaN)
        mean_df = mean_df.dropna(how="all")
        edf_pickle_name = patient_id
        only_plots(mean_df, save_plot=True, save_dir=save_dir, edf_pickle_name=edf_pickle_name, band=band, step_sec=step_sec)
        print(f"Plotted mean for patient {patient_id}, band {band}")

            
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
                    print(f"Loaded {fname} with ECG channel.")
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
            edf_results, edf_key, bool_plots=False, save_plot=True,
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
        plot_patient_band_means(            patient_id, bands=bands.keys(),
            step_sec=step_sec, temps_dir="temps_EDF_HEP", save_dir="figures_HEP/compute_brain_heart_coupling"
        )
        
    # 3. Average across all patients and plot
    def plot_all_patients_band_means(bands=None, step_sec=5, temps_dir="temps_EDF_HEP", save_dir="figures_HEP/compute_brain_heart_coupling"):
        """
        For each band, average all *_results_{band}.csv files across all patients, then plot.
        """
        if bands is None:
            bands = ["delta", "alpha", "beta", "gamma"]
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
            # Align columns and average row-wise using nanmean logic
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

    plot_all_patients_band_means(bands=bands.keys(), step_sec=step_sec, temps_dir="temps_EDF_HEP", save_dir="figures_HEP/compute_brain_heart_coupling")

    print("All processing and plotting complete.")

# Example usage:
edf_results = load_edf_pickles_with_ecg()
edf_results = {k: v for k, v in edf_results.items() if v is not None}
process_all_patients(edf_results, step_sec=5)

print("Processing and plotting complete for all patients.")

# Example: plot mean band results for a specific patient (replace "010" with desired patient_id)
# plot_patient_band_means("010")
