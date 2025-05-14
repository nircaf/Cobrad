import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import iqr
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.stats.multitest as smm
from scipy.signal import spectrogram
import statsmodels.api as sm
import streamlit as st
import json
import pickle, uuid
import re, io
import h5py
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from scipy.signal import welch

eeg_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7',
       'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'A1','A2', 'Fpz', 'Oz']

eeg_dict_convertion = {
       'Fp2-F4': 'Fp2',
       'F4-C4': 'F4',
       'C4-P4': 'C4',
       'P4-O2': 'P4',
       'Fp2-F8': 'Oz',
       'F8-T4': 'F8',
       'T4-T6': 'T4',
       'T6-O2': 'T6',
       'Fz-Cz': 'Fz',
       'Cz-Pz': 'Cz',
       'Fp1-F3': 'Fp1',
       'F3-C3': 'F3',
       'C3-P3': 'C3',
       'P3-O1': 'P3',
       'Fp1-F7': 'Fpz',
       'F7-T3': 'F7',
       'T3-T5': 'T3',
       'T5-O1': 'T5',
       'EOG1+': 'eog',
       'EOG2+': 'eog',
       'ECG1+': 'ecg',
        'ECG2+': 'ecg',
}

# Define power bands as a dictionary
power_bands = {
    "delta": [.5, 4],
    "theta": [4, 8],
    "alpha": [8, 12],
    "beta": [12, 30],
    "gamma": [30, 100]
}

def mean_of_resized_arrays(arrays):
    # Get the shapes of all arrays
    shapes = np.array([arr.shape for arr in arrays])
    
    # Compute median dimensions
    median_shape = tuple(np.median(shapes, axis=0).astype(int))
    
    # Resize all arrays to the median shape
    resized_arrays = np.array([np.resize(arr, median_shape) for arr in arrays])
    
    # Compute the mean
    return np.mean(resized_arrays, axis=0)

def stat_text_get(group_data, col=None):
    lower_bound = -1e-30
    upper_bound = 1e30
    if col is None:
        stats_dict = {
            "N": len(group_data),
            "Mean": np.clip(group_data.mean(), lower_bound, upper_bound),
            "Median": np.clip(group_data.median(), lower_bound, upper_bound),
            "Max": np.clip(group_data.max(), lower_bound, upper_bound),
            "Min": np.clip(group_data.min(), lower_bound, upper_bound),
            "Std": np.clip(group_data.std(), lower_bound, upper_bound),
        }
    else:
        stats_dict = {
            "N": len(group_data),
            "Mean": np.clip(group_data[col].mean(), lower_bound, upper_bound),
            "Median": np.clip(group_data[col].median(), lower_bound, upper_bound),
            "Max": np.clip(group_data[col].max(), lower_bound, upper_bound),
            "Min": np.clip(group_data[col].min(), lower_bound, upper_bound),
            "Std": np.clip(group_data[col].std(), lower_bound, upper_bound),
        }

    stats_text = (
        f"N = {stats_dict['N']}, "
        f"Mean = {stats_dict['Mean']:.2e}, "
        f"Median = {stats_dict['Median']:.2e}, "
        f"Max = {stats_dict['Max']:.2e}, "
        f"Min = {stats_dict['Min']:.2e}, "
        f"Std = {stats_dict['Std']:.2e}"
    )
    return stats_text, stats_dict

def boxplot_plot(results_df, combined_df, col, output_dir,figures_dir=None,is_streamlit=False,analysis_type=None, show_histograms=False):
    # Function to remove outliers based on 5 standard deviations
    def remove_outliers(df, col, group_col, threshold=5):
        def filter_group(group):
            mean = group[col].mean()
            std = group[col].std()
            return group[np.abs(group[col] - mean) <= threshold * std]
        
        return df.groupby(group_col).apply(filter_group).reset_index(drop=True)

    # Remove outliers from each group
    cleaned_df = remove_outliers(combined_df, col, 'Group')

    # Plot the cleaned data
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Group', y=col, data=cleaned_df, showfliers=False)
    # Add stripplot
    sns.stripplot(x='Group', y=col, data=cleaned_df, alpha=0.5, jitter=True, color='black')
    # Add significance markers
    filtered_df = results_df[results_df['Variable'] == col]
    if filtered_df.empty:
        return
    row = results_df[results_df['Variable'] == col].iloc[0]
    if row['adj_p_value'] < 0.001:
        sig_symbol = '***'
    elif row['adj_p_value'] < 0.01:
        sig_symbol = '**'
    elif row['adj_p_value'] < 0.05:
        sig_symbol = '*'
    else:
        sig_symbol = 'ns'
    # if sig_symbol != 'ns':
    title_text = (
        f"{row['Test']}\n"
        f"p = {row['adj_p_value']:.3e} ({sig_symbol})\n"
        f"Cohen's d = {row['Cohen_d']:.2f}\n"
        f"{col} Comparison"
    )
    if sig_symbol != 'ns' or analysis_type == 'Full':
        plt.title(title_text, ha='center')
        # Add sample size to x-axis labels
        group_counts = combined_df['Group'].value_counts()
        ax = plt.gca()
        ax.set_xticklabels([f"{label.get_text()}\nn={group_counts[label.get_text()]}" for label in ax.get_xticklabels()])
        plt.tight_layout()
        if is_streamlit:
            st.divider()
            st.subheader(f"Boxplot of {col} by Group")
            for group_val in cleaned_df['Group'].unique():
                group_data = cleaned_df[cleaned_df['Group'] == group_val][col]
                stats_text, stat_dict = stat_text_get(group_data)
                st.write(f"{group_val} mean {stat_dict['Mean']:.2e} ± {stat_dict['Std']:.2e}")
            st.write(f"P-value: {row['adj_p_value']:.3e}, Effect size: d {row['Cohen_d']:.2f}")
            st_pyplot_func(plt,filename=f"{col}_comparison")
        else:
            os.makedirs(f'{figures_dir}/boxplots/{output_dir}', exist_ok=True)
            plt.savefig(f"{figures_dir}/boxplots/{output_dir}/{col}_comparison.png")
            plt.close()
        if show_histograms:
            # Plot histograms for each group and both groups together
            plt.figure(figsize=(10, 6))
            sns.histplot(data=cleaned_df, x=col, hue='Group', element='step', stat='density', common_norm=False)
            for i, group in enumerate(cleaned_df['Group'].unique()):
                group_data = cleaned_df[cleaned_df['Group'] == group][col]
                stats_text, _ = stat_text_get(group_data)
                plt.annotate(stats_text, xy=(0.25, 0.95 - i * 0.1), xycoords='axes fraction', fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
            plt.title(f"{col} Histogram by Group")
            if is_streamlit:
                st.write(f"Histogram of {col} by Group")
                st_pyplot_func(plt)
            else:
                os.makedirs(f'{figures_dir}/hist/{output_dir}', exist_ok=True)
                plt.savefig(f"{figures_dir}/hist/{output_dir}/{col}_hist_by_group.png")
                plt.close()
            plt.figure(figsize=(10, 6))
            sns.histplot(data=cleaned_df, x=col, element='step', stat='density')
            combined_data = cleaned_df[col]
            stats_text, _ = stat_text_get(combined_data)
            plt.annotate(stats_text, xy=(0.25, 0.95), xycoords='axes fraction', fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
            plt.title(f"{col} Histogram Combined")
            if is_streamlit:
                st.write(f"Histogram of {col}")
                st_pyplot_func(plt)
            else:
                plt.savefig(f"{figures_dir}/hist/{output_dir}/{col}_hist_combined.png")
                plt.close()
    return results_df

def st_pyplot_func(plt,filename='plot'):
    """Function to display matplotlib figures in Streamlit."""
    # bbox_inches="tight":
    plt.tight_layout()
    st.pyplot(plt)
    # Save the plot as an SVG file in memory
    svg_buffer = io.BytesIO()
    plt.savefig(svg_buffer, format="svg")
    svg_buffer.seek(0)
    uuid4 = str(uuid.uuid4())
    # Add a download button for the SVG file
    st.download_button(
        label="Download plot as SVG",
        data=svg_buffer,
        file_name=f"{filename}.svg",
        mime="image/svg+xml",
        key=uuid4,
    )
    # plt.close()
    
def scatter_plot_with_regression(results_df, combined_df, x_col, y_col, output_dir,figures_dir= None,is_streamlit=False,analysis_type=None, show_histograms=False):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_col, y=y_col, data=combined_df, alpha=0.5, color='black')
    sns.regplot(x=x_col, y=y_col, data=combined_df, scatter=False, color='blue')
    # Perform linear regression
    X = sm.add_constant(combined_df[x_col])
    y = combined_df[y_col]
    model = sm.OLS(y, X).fit()
    p_value = model.pvalues[1]  # p-value for the slope
    r_squared = model.rsquared  # R-squared value
    # Determine significance symbol
    if (p_value < 0.001):
        sig_symbol = '***'
    elif (p_value < 0.01):
        sig_symbol = '**'
    elif (p_value < 0.05):
        sig_symbol = '*'
    else:
        sig_symbol = 'ns'
    if sig_symbol != 'ns' or analysis_type == 'Full':
        # Add stats results to the title
        plt.title(
            f"{x_col} vs {y_col} Regression\n"
            f"Slope p = {p_value:.3e} ({sig_symbol}), R^2 = {r_squared:.2f}, n = {len(combined_df)}",
            ha='center'
        )
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.tight_layout()
        # Make folder {figures_dir}
        if is_streamlit:
            st.divider()
            st.subheader(f"Scatterplot of {x_col} vs {y_col}")
            # write {x_col} mean ± std
            st.write(f"Mean: {x_col} = {combined_df[x_col].mean():.2e} ± {combined_df[x_col].std():.2e}")
            st.write(f"Mean: {y_col} = {combined_df[y_col].mean():.2e} ± {combined_df[y_col].std():.2e}")
            st.write(f"P-value {p_value:.3e}, R^2 {r_squared:.2f}")
            st_pyplot_func(plt,filename=f"{x_col}_vs_{y_col}_regression")
        else:
            os.makedirs(f'{figures_dir}/scatterplots/{output_dir}', exist_ok=True)
            plt.savefig(f"{figures_dir}/scatterplots/{output_dir}/{y_col}_regression.png")
            plt.close()
    
def analyze_and_correct(combined_df, columns_to_analyze, groups=['Control', 'WNV']):
    def analyze_groups(combined_df, col, groups):
        results = []
        for i, group1 in enumerate(groups):
            for group2 in groups[i+1:]:
                control_data = combined_df[combined_df['Group'] == group1][col].dropna()
                case_data = combined_df[combined_df['Group'] == group2][col].dropna()
                if len(control_data) < 2 or len(case_data) < 2:
                    continue
                # Normality test
                _, normal_p = stats.normaltest(combined_df[col].dropna())
                # Choose appropriate test
                if normal_p < 0.05:  # Non-parametric
                    stat, p = stats.mannwhitneyu(control_data, case_data)
                    test_used = "Mann-Whitney U"
                else:  # Parametric
                    stat, p = stats.ttest_ind(control_data, case_data)
                    test_used = "T-test"
                # Effect size
                cohen_d = (case_data.mean() - control_data.mean()) / np.sqrt((
                    (len(case_data)-1)*case_data.std()**2 + 
                    (len(control_data)-1)*control_data.std()**2) / 
                    (len(case_data) + len(control_data) - 2))
                results.append({
                    'Variable': col,
                    'Group1': group1,
                    'Group2': group2,
                    'Test': test_used,
                    'Statistic': stat,
                    'p_value': p,
                    'Cohen_d': cohen_d,
                    'Mean_Group1': control_data.mean(),
                    'Mean_Group2': case_data.mean(),
                    'Std_Group1': control_data.std(),
                    'Std_Group2': case_data.std(),
                    'median_Group1': control_data.median(),
                    'median_Group2': case_data.median(),
                    'MAD_Group1': stats.median_abs_deviation(control_data),
                    'MAD_Group2': stats.median_abs_deviation(case_data),
                })
        return results

    all_results = []    
    # Statistical analysis
    for col in columns_to_analyze:
        results = analyze_groups(combined_df, col, groups)
        if results:
            all_results.extend(results)

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Multiple testing correction
    if not results_df.empty:
        rej, adj_p, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
        results_df['adj_p_value'] = adj_p
        results_df['Significant'] = adj_p < 0.05

    return results_df

def topomap_group_data( band, montage,control_data,wnv_data,output_dir,figures_dir,is_streamlit=False):
    # Calculate p-values for each channel
    common_channels = control_data.columns.intersection(wnv_data.columns)
    dict_p_values = {}
    for common_channel in common_channels:
        control_channel = control_data[common_channel].dropna()
        wnv_channel = wnv_data[common_channel].dropna()
        if len(control_channel) < 2 or len(wnv_channel) < 2:
            continue
        _, p = stats.mannwhitneyu(control_channel, wnv_channel)
        dict_p_values[common_channel] = p
    df_p_values = pd.DataFrame(dict_p_values, index=[0]).T
    reject, pvals_corrected, _, _ = smm.multipletests(df_p_values[0].values, alpha=0.05, method='fdr_bh')
    df_p_values['pvals_corrected'] = pvals_corrected
    if any(df_p_values['pvals_corrected'] < 0.05):
        ch_names = df_p_values.index.tolist()
        # Create an info object
        info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types='eeg')
        info.set_montage(montage)
        # Create an EvokedArray object for p-values
        p_evoked = mne.EvokedArray(df_p_values['pvals_corrected'].values.reshape(-1, 1), info)
        # Plot the topomap of p-values
        fig, ax = plt.subplots()
        vlim_max = min(0.05, df_p_values['pvals_corrected'].max())
        im, cm = mne.viz.plot_topomap(p_evoked.data[:, 0], p_evoked.info, axes=ax, show=False, cmap='jet_r', vlim=[0, vlim_max])
        fig.colorbar(im, ax=ax)
        plt.title(f"{band} {output_dir} P-Value Topomap")
        # if any value in pvals_corrected is less than 0.05
        # make folder {figures_dir}/boxplots/{output_dir}
        # Save the figure
        if is_streamlit:
            st.write(f"Topomap for {band} {output_dir} P-Value")
            st_pyplot_func(plt)
        else:
            os.makedirs(f'{figures_dir}/topomaps_p_values/{output_dir}', exist_ok=True)
            plt.savefig(f"{figures_dir}/topomaps_p_values/{output_dir}/p_values_{band}_topomap.png")
            plt.close()

def process_group_data(group, run_df, frequency_bands, eeg_dict_convertion, eeg_channels, montage,group_data):
    # get only columns that say EEG
    eeg_df = run_df.filter(like='EEG')
    group_data[group] = {}
    for band in frequency_bands:
        # get the columns which have the band name
        power_df = eeg_df.filter(like=band)
        # column name split ' '[-1]
        power_df.columns = power_df.columns.str.split(' ').str[-1]
        power_df.columns = [eeg_dict_convertion[col] if col in eeg_dict_convertion else col for col in power_df.columns]
        # leave only the eeg channels
        # Get the channels that exist in the DataFrame
        existing_channels = [ch for ch in eeg_channels if ch in power_df.columns]
        # Filter the DataFrame to include only the existing channels
        power_df = power_df[existing_channels]
        # drop columns more than half of the values are NaN
        power_df = power_df.dropna(axis=1, thresh=power_df.shape[0]//2)
        # drop duplicates columns
        power_df.columns
        # change column names to only be the channel. if not
        # Extract the power values for the current band
        # power_values = power_df[band].values
        ch_names = power_df.columns.tolist()
        # Create an info object
        info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types='eeg')
        info.set_montage(montage)
        power_values = power_df.T.values
        group_data[group][band] = power_df

        # Create an EvokedArray object
        evoked = mne.EvokedArray(power_values, info)
        # Plot the topomap
        fig, ax = plt.subplots()
        im, cm = mne.viz.plot_topomap(evoked.data[:, 0], evoked.info, axes=ax, show=False)
        fig.colorbar(im, ax=ax)
        # plt.title(f"{band} Topomap")
        # # Save the figure
        # os.makedirs(f'{figures_dir}/topomaps', exist_ok=True)
        # plt.savefig(f"{figures_dir}/topomaps/{group}_{band}_topomap.png")
        # plt.close()
    return group_data 

def weighted_avg(df, weight_col, numeric_cols):
    # Calculate weighted average for numeric columns, ignoring NaNs
    weighted_df = df[numeric_cols].multiply(df[weight_col], axis=0).sum(skipna=True) / df[weight_col].sum(skipna=True)
    # Preserve non-numeric columns by taking the first non-null value
    non_numeric_cols = df.columns.difference(numeric_cols)
    preserved_df = df[non_numeric_cols].iloc[0]
    # Combine the results
    return pd.concat([preserved_df, weighted_df])
#%% WNV
def wnv_get_files():
    # load clinical data from WNV_merged_291224_KP.xlsx
    df_wnv = pd.read_excel('WNV_merged_291224_KP.xlsx')
    # df_wnv replace '.' in column names with '_'
    df_wnv.columns = df_wnv.columns.str.replace('.', '_')
    # df_wnv replace 'NA' with np.nan
    df_wnv = df_wnv.replace('NA', np.nan)
    # Configuration
    patients_folder = "west_nile_virus"
    case_file = f"{patients_folder}.csv"
    # Read and prepare data
    controls = pd.read_csv(f'{patients_folder}_controls.csv')
    cases = pd.read_csv(case_file)
    wnv_ids = [file.split('/')[-2] for file in cases['file_path']]
    # to int
    wnv_ids = [int(id) for id in wnv_ids]
    cases['ID'] = wnv_ids
    # merge the dataframes
    df_merged_outer = pd.merge(df_wnv, cases, on='ID', how='outer',indicator=True)
    df_merged = df_merged_outer[df_merged_outer['_merge'] == 'both']
    wnv_files = os.listdir(f'west_nile_virus')
    # remove .DS_Store
    wnv_files = [file for file in wnv_files if 'DS_Store' not in file]
    wnv_files = [file.split('.edf')[0] for file in wnv_files]
    # also split '-'
    wnv_files = [file.split('-')[0] for file in wnv_files]
    # remove duplicates
    wnv_files = list(set(wnv_files))
    # to int
    wnv_files = [int(file) for file in wnv_files]
    # get df in column ID matches with wnv_files
    # print what wnv_files are not in df
    print([file for file in wnv_files if file not in df_wnv['ID'].values])
    df_wnv2 = df_wnv[df_wnv['ID'].isin(wnv_files)]
    # avg lines with same ID
    numeric_cols = df_merged.select_dtypes(include=[np.number]).columns
    # Group by ID and calculate the mean of each numeric column
    df_wnv2 = df_merged.groupby('ID').apply(weighted_avg, weight_col='duration_min', numeric_cols=numeric_cols).reset_index(drop=True)
    # delta = MRS_FOLLOW_UP - MRS_prior
    df_wnv2['MRS_delta_follow_up'] = df_wnv2['MRS_FOLLOW_UP'] - df_wnv2['MRS_prior']
    df_wnv2['MRS_delta_at_peak_illness'] = df_wnv2['MRS_at_peak_illness'] - df_wnv2['MRS_prior']
    df_wnv2['MRS_delta_peak_minus_follow_up'] = df_wnv2['MRS_at_peak_illness'] - df_wnv2['MRS_FOLLOW_UP']
    cases_group_name = 'WNV'
    return df_wnv,patients_folder,controls,df_wnv2,cases_group_name 
    # df_wnv2 to csv 'wnv_grouped.csv'
    df_wnv2.to_csv('wnv_grouped.csv', index=False)
    # sum ENCEPHALITIS
    df_wnv2['ENCEPHALITIS'].sum()
    # sum MENINGITIS
    df_wnv2['MENINGITIS'].sum()
    # df_wnv2.columns where MRS
    [col for col in df_wnv2.columns if 'MRS' in col]
    df_wnv2.columns.tolist()
    df_merged['ID'].to_list()

def find_consecutive_sequences(events, min_length=5):
    sequences = []
    temp_seq = [events[0]]

    for i in range(1, len(events)):
        if events[i] == events[i - 1] + 1:
            temp_seq.append(events[i])
        else:
            if len(temp_seq) >= min_length:
                sequences.append(temp_seq)
            temp_seq = [events[i]]

    if len(temp_seq) >= min_length:  # Check the last sequence
        sequences.append(temp_seq)

    return sequences

def read_edf_mne(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True, encoding='latin1')
    metadata = {
        'file_name': os.path.basename(file_path),
        'start_date': raw.info['meas_date'],
        'duration_sec': raw.times[-1],
        'duration_min': raw.times[-1] / 60,
        'number_of_signals': len(raw.ch_names),
        'signal_labels': raw.ch_names,
        'sampling_frequency': raw.info['sfreq'],
        'highpass': raw.info['highpass'],
        'lowpass': raw.info['lowpass'],
        'annotations': raw.annotations if raw.annotations else None,
        'n_samples': raw.n_times,
        'bad_channels': raw.info['bads']
    }
    return metadata, raw

def eeg_data_to_features(raw, window_size_sec=5, min_duration_sec=5):
    raw_eeg = raw.copy().pick_types(eeg=True)
    eeg_data = raw_eeg.get_data()
    sf = raw_eeg.info['sfreq']
    window_size = int(window_size_sec * sf)

    # Initialize storage
    psd_list = []
    fft_list = []
    mpf_list = []
    df_list = []
    dfv_list = []
    pswe_events_per_channel = []

    # Loop channels
    for channel_data in eeg_data:
        ch_psd, ch_fft, ch_mpf, ch_df, ch_pswe = [], [], [], [], []

        # Slide windows
        for start in range(0, len(channel_data) - window_size + 1, window_size):
            win = channel_data[start:start + window_size]

            # PSD
            psd_vals, freqs = mne.time_frequency.psd_array_welch(
                win, sf, fmin=1, fmax=40, n_fft=int(sf)
            )
            psd_vals = psd_vals.squeeze()

            # FFT
            fft_vals = np.fft.fft(win)

            # MPF and dominant freq
            mpf = np.sum(psd_vals * freqs) / np.sum(psd_vals)
            dfreq = freqs[np.argmax(psd_vals)]

            ch_psd.append(psd_vals)
            ch_fft.append(fft_vals)
            ch_mpf.append(mpf)
            ch_df.append(dfreq)

            # PSWE detection
            if mpf < 6.0:
                ch_pswe.append(start / sf)

        # Store per-channel
        psd_list.append(ch_psd)
        fft_list.append(ch_fft)
        mpf_list.append(ch_mpf)
        df_list.append(ch_df)
        dfv_list.append(np.std(ch_df))
        pswe_events_per_channel.append(ch_pswe)

    # Convert to arrays
    psd_arr = np.array(psd_list)
    fft_arr = np.array(fft_list)
    mpf_arr = np.array(mpf_list)
    df_arr = np.array(df_list)
    dfv_arr = np.array(dfv_list)

    # PSWE stats
    total_duration = eeg_data.shape[1] / sf
    overall_pswe_durations = []
    pswe_stats = []

    for events in pswe_events_per_channel:
        if events:
            # find_consecutive_sequences should return lists of indices or times
            sequences = find_consecutive_sequences(np.array(events), min_duration_sec)
            durations = [len(seq) for seq in sequences]
        else:
            durations = []

        pswe_total = sum(durations)
        pct = (pswe_total / total_duration) * 100
        per_min = len(durations) / (total_duration / 60)
        avg_len = np.mean(durations) if durations else 0

        pswe_stats.append({
            'pswe_percentage': pct,
            'pswe_events_per_minute': per_min,
            'pswe_avg_length': avg_len
        })
        overall_pswe_durations.extend(durations)

    # Overall PSWE
    overall_pswe_median = np.median(overall_pswe_durations) if overall_pswe_durations else 0
    overall_pct = (overall_pswe_median / total_duration) * 100
    overall_evpm = len(overall_pswe_durations) / ((total_duration * len(pswe_events_per_channel)) / 60)
    overall_avg = np.mean(overall_pswe_durations) if overall_pswe_durations else 0

    # Bandpower
    def bandpower(data, sf, band, window_sec=None, relative=False):
        low, high = band
        nperseg = int((window_sec or (2/low)) * sf)
        freqs, psd = welch(data, sf, nperseg=nperseg)
        idx = np.logical_and(freqs >= low, freqs <= high)
        bp = np.trapz(psd[:, idx], freqs[idx], axis=1)
        if relative:
            bp /= np.trapz(psd, freqs, axis=1)
        return bp

    band_powers = {name: bandpower(eeg_data, sf, rng,window_sec=window_size_sec) for name, rng in power_bands.items()}

    # Time-domain stats
    skew = np.apply_along_axis(lambda x: pd.Series(x).skew(), 1, eeg_data)
    kurt = np.apply_along_axis(lambda x: pd.Series(x).kurt(), 1, eeg_data)

    # Compile metadata
    meta = {
        'overall_pswe_median_percentage': overall_pct,
        'overall_pswe_events_per_minute': overall_evpm,
        'overall_pswe_avg_length': overall_avg,
        'overall_min': eeg_data.min(),
        'overall_max': eeg_data.max(),
        'overall_mean': eeg_data.mean(),
        'overall_median': np.median(eeg_data),
        'overall_std': eeg_data.std(),
        'overall_psd_mean': psd_arr.mean(),
        'overall_psd_std': psd_arr.std(),
        'overall_fft_mean': np.abs(fft_arr).mean(),
        'overall_fft_std': np.abs(fft_arr).std(),
        'overall_delta_power': band_powers['delta'].mean(),
        'overall_theta_power': band_powers['theta'].mean(),
        'overall_alpha_power': band_powers['alpha'].mean(),
        'overall_beta_power': band_powers['beta'].mean(),
        'overall_gamma_power': band_powers['gamma'].mean(),
        'overall_skewness': skew.mean(),
        'overall_kurtosis': kurt.mean(),
        'overall_mpf_mean': mpf_arr.mean(),
        'overall_mpf_median': np.median(mpf_arr),
        'overall_df_mean': df_arr.mean(),
        'overall_dfv_std': dfv_arr.mean()
    }

    # Channel-specific metadata
    for i, ch in enumerate(raw_eeg.ch_names):
        meta.update({
            f'min_EEG {ch}': eeg_data[i].min(),
            f'max_EEG {ch}': eeg_data[i].max(),
            f'mean_EEG {ch}': eeg_data[i].mean(),
            f'median_EEG {ch}': np.median(eeg_data[i]),
            f'std_EEG {ch}': eeg_data[i].std(),
            f'psd_mean_EEG {ch}': psd_arr[i].mean(),
            f'psd_std_EEG {ch}': psd_arr[i].std(),
            f'fft_mean_EEG {ch}': np.abs(fft_arr[i]).mean(),
            f'fft_std_EEG {ch}': np.abs(fft_arr[i]).std(),
            f'delta_power_EEG {ch}': band_powers['delta'][i],
            f'theta_power_EEG {ch}': band_powers['theta'][i],
            f'alpha_power_EEG {ch}': band_powers['alpha'][i],
            f'beta_power_EEG {ch}': band_powers['beta'][i],
            f'gamma_power_EEG {ch}': band_powers['gamma'][i],
            f'skewness_EEG {ch}': skew[i],
            f'kurtosis_EEG {ch}': kurt[i],
            f'mean_mpf_EEG {ch}': mpf_arr[i].mean(),
            f'median_mpf_EEG {ch}': np.median(mpf_arr[i]),
            f'pswe_percentage_EEG {ch}': pswe_stats[i]['pswe_percentage'],
            f'pswe_events_per_minute_EEG {ch}': pswe_stats[i]['pswe_events_per_minute'],
            f'pswe_avg_length_EEG {ch}': pswe_stats[i]['pswe_avg_length'],
            f'dfv_mean_EEG {ch}': df_arr[i].mean(),
            f'dfv_std_EEG {ch}': dfv_arr[i]
        })
    return meta

def process_file(file, pickles_location, sample_window_size):
    """
    Process a single file to extract metadata.
    """
    print(f"Processing file: {file}")
    # Extract patient ID from the file name
    patient_id = re.search(r'(\d{4})-\d{3}', file).group(1)
    print(f"Patient ID: {patient_id}")
    # Load the raw EEG data from the pickle file
    raw = pickle.load(open(os.path.join(pickles_location, file), 'rb'))
    metadata_window = eeg_data_to_features(raw, window_size_sec=sample_window_size)
    return patient_id, metadata_window
#%% COBRAD
def cobrad_get_files(sample_window_size=0,only_awake=False,sleep_only=False):
    patients_folder = "EDF"
    sheets_to_read = ['clinical', 'medications', 'npi-q', 'epworth', 'isi', 'ecog_12','Sheet4','seizures']
    dfs = pd.read_excel('COBRAD_clinical_24022025.xlsx', sheet_name=sheets_to_read)
    def get_df_wnv():
        # read sheets clinical, medications, npi-q, epworth,isi, ecpg_12 from COBRAD_clinical_24022025.xlsx
        sheets_to_sum_vals = ['epworth', 'isi', 'ecog_12','Sheet4','npi-q','seizures', 'medications']
        # Rename 'record_id' to 'ID' in each DataFrame and convert to string
        for sheet in sheets_to_read:
            dfs[sheet] = dfs[sheet].rename(columns={'record_id': 'ID'}).astype(str)
            # drop col contain has_eeg or has eeg . ignore case
            dfs[sheet] = dfs[sheet].drop(columns=[col for col in dfs[sheet].columns if 'has eeg' in col.lower() or 'has_eeg' in col.lower()])
            # One-hot encode drugs in the medications sheet
            if sheet == 'medications':
                # Extract the drug names no 'nan'
                drug_names = dfs[sheet]['name_drug_1'].dropna().unique()
                # remove 'nan'
                drug_names = [drug for drug in drug_names if 'nan' not in drug]
                # Create one-hot encoded columns for each drug
                for drug in drug_names:
                    dfs[sheet][f'{drug}'] = dfs[sheet]['name_drug_1'].apply(lambda x: 1 if x == drug else 0)
                # Keep only the ID and one-hot encoded drug columns
                drug_columns = [f'{drug}' for drug in drug_names]
                dfs[sheet] = dfs[sheet][['ID'] + drug_columns]  
                # merge columns per ID
                dfs[sheet] = dfs[sheet].groupby('ID').sum().reset_index()   
                # dict how many ID take a drug
                drug_counts = dfs[sheet].drop(columns='ID').sum().to_dict() 
                # only above 3
                drug_counts = {k: v for k, v in drug_counts.items() if v > 3}  
                # read utils/drug_groups.json
                with open('utils/drug_groups.json', 'r') as f:
                    drug_groups = json.load(f)
                for key_number_groups in drug_groups:
                    for key_group, value_group in drug_groups[key_number_groups].items():
                        # value_group remove /n and stip
                        value_group = [group.strip() for group in value_group]
                        # get the columns that contain the key_group
                        columns = [col for col in dfs[sheet].columns if any(group in col for group in value_group)]
                        # sum the columns and create a new column with the name of the group
                        dfs[sheet][f'{key_number_groups}_groups_{key_group}'] = dfs[sheet][columns].sum(axis=1)
            if sheet in sheets_to_sum_vals:
                # to numeric all columns but ID
                dfs[sheet] = pd.concat([dfs[sheet]['ID'], dfs[sheet].drop(columns='ID').apply(pd.to_numeric, errors='coerce')], axis=1)
                dfs[sheet][f'{sheet}_sum'] = dfs[sheet].drop(columns='ID').sum(axis=1)
                if sheet =='seizures':
                    # replace -9 with np.nan
                    dfs[sheet] = dfs[sheet].replace(-9, np.nan)
            # add the name of sheet at beggining of column all cols but ID
            dfs[sheet].columns = [f'{sheet}_{col}' if col != 'ID' else col for col in dfs[sheet].columns]

        # Merge all DataFrames on 'ID'
        df_wnv = dfs[sheets_to_read[0]]
        for sheet in sheets_to_read[1:]:
            df_wnv = pd.merge(df_wnv, dfs[sheet], on='ID', how='outer')
        return df_wnv
    def get_cobrad_controls():
        controls = pd.read_csv(f'{patients_folder}_controls.csv')
        controls['ID'] = controls['file_name'].apply(lambda x: x.split('_')[0]).astype(str)
        numeric_cols = controls.select_dtypes(include=[np.number]).columns
        controls = controls.groupby('ID').apply(
            lambda x: (x[numeric_cols].multiply(x['duration_min'], axis=0)).sum(skipna=False) / x['duration_min'].sum(skipna=False)
        ).reset_index()
        return controls
    controls = get_cobrad_controls()

    def get_cases_cobrad():
        if sample_window_size == 0:
            if only_awake:
                case_file = f"{patients_folder}_awake.csv"
            else:
                case_file = f"{patients_folder}.csv"
            cases = pd.read_csv(case_file)
            cases['ID'] = cases['csv_file_name'].apply(id_from_csv_filename).astype(str)
            # Sort by ID
            cases = cases.sort_values(by='ID')
        else:
            # Process pickles to calculate average values for each patient
            case_folder = 'wake_EDF' if only_awake else 'EDF'
            pickles_location = f'pickles/{case_folder}'
            files = [file for file in os.listdir(pickles_location) if file.endswith('.pkl')]

            # Dictionary to store metadata for each patient
            patient_metadata = {}
            # Use ProcessPoolExecutor for parallel processing
            with ProcessPoolExecutor() as executor:
                results = list(tqdm(executor.map(process_file, files, [pickles_location] * len(files), [sample_window_size] * len(files)), 
                                    total=len(files), desc="Processing files"))

            # Aggregate metadata for each patient
            for patient_id, metadata_window in results:
                if patient_id not in patient_metadata:
                    patient_metadata[patient_id] = []
                patient_metadata[patient_id].append(metadata_window)

            # Calculate average values for each patient
            cases = []
            for patient_id, metadata_list in patient_metadata.items():
                # Convert list of metadata dictionaries to a DataFrame
                patient_df = pd.DataFrame(metadata_list)
                # Calculate mean for numeric columns
                patient_avg = patient_df.mean(numeric_only=True).to_dict()
                patient_avg['ID'] = patient_id
                cases.append(patient_avg)

            # Convert the list of dictionaries to a DataFrame
            cases = pd.DataFrame(cases)

        return cases
    cases = get_cases_cobrad()
    df_wnv = get_df_wnv()
    df_wnv.columns = df_wnv.columns.str.replace('.', '_').str.replace('<', '').str.replace('>', '')

    df_merged = pd.merge(df_wnv, cases, on='ID', how='outer',indicator=True)
    # Get all files that end with .edf from EDF folder and subfolders
    eeg_files = []
    for root, dirs, files in os.walk('EDF'):
        for file in files:
            if file.endswith('.edf'):
                eeg_files.append(os.path.join(root, file))
    # print outer
    failed_ids = df_merged[df_merged['_merge'] == 'left_only']['ID'].unique()
    # check if they exist in eeg_files
    for id_to_check in failed_ids:
        # check if id exists contains in eeg_files
        if any(id_to_check in file for file in eeg_files):
            # print(f'{id_to_check}')
            pass
    df_merged = df_merged[df_merged['_merge'] == 'both'].drop(columns='_merge')
    numeric_cols = df_merged.select_dtypes(include=[np.number]).columns
    # Group by ID and apply the weighted average function
    df_wnv2 = df_merged.groupby('ID').apply(weighted_avg, weight_col='duration_min', numeric_cols=numeric_cols).reset_index(drop=True)
    # remove highpass, lowpass, n_samples, size, patient_number
    cols_to_drop = ['highpass', 'lowpass', 'n_samples', 'size', 'patient_number','duration_sec']
    # drop if conatins any of the cols_to_drop
    df_wnv2 = df_wnv2.drop(columns=[col for col in df_wnv2.columns if any([drop in col for drop in cols_to_drop])])
    # replace all nan with np.nan ignore case
    df_wnv2 = df_wnv2.applymap(lambda x: np.nan if isinstance(x, str) and 'nan' in x.lower() else x)
    # numeric strings to float or int
    for col in df_wnv2.columns:
        try:
            df_wnv2[col] = pd.to_numeric(df_wnv2[col])
        except:
            # print(f'Could not convert {col} to numeric')
            pass
    cases_group_name = 'COBRAD'
    # if 'COBRAD_descriptive.xlsx' doesnt exist
    if not os.path.exists('COBRAD_descriptive.xlsx'):
        # Create a dictionary to store descriptive statistics for each sheet
        desc_stats = {}
        for sheet in sheets_to_read:
            # get only the ids that are in df_wnv2
            dfs[sheet] = dfs[sheet][dfs[sheet]['ID'].isin(df_wnv2['ID'])]
            df_desc = custom_describe(dfs[sheet])
            desc_stats[sheet] = df_desc
        # Save all descriptive statistics to one Excel file
        with pd.ExcelWriter('COBRAD_descriptive.xlsx') as writer:
            for sheet_name, df_desc in desc_stats.items():
                df_desc.to_excel(writer, sheet_name=sheet_name)
    # df_wnv save to csv
    return df_wnv,patients_folder,controls,df_wnv2,cases_group_name
    df_merged['ID'].unique()
    df_merged['_merge'].unique()
    df_merged[df_merged['_merge'] == 'right_only']['ID'].unique()
    df_wnv2['ID'].unique()

def get_clinical_and_boxplot_cols(df_wnv2):
       boxplot_columns = [col for col in df_wnv2.columns if 'overall' in col.lower()]
       # split file name .[0] and then '-'[0] to get the ID
       clinical_columns_all = df_wnv2.columns[3:].tolist()
       # remove boxplot_columns from clinical_columns
       clinical_columns = [col for col in clinical_columns_all if col not in boxplot_columns]
       # Remove columns that contain 'EEG'
       clinical_columns = [col for col in clinical_columns if 'EEG' not in col]
       return clinical_columns,boxplot_columns

def save_raw_data(df, figures_dir):
    # Ensure the directory exists
    os.makedirs(f'{figures_dir}/raw_data', exist_ok=True)
    # Save the DataFrame to a CSV file
    df.to_csv(f'{figures_dir}/raw_data/raw_data.csv', index=False)
   
# Custom descriptive statistics function
def custom_describe(df):
    # Convert columns with object dtype to more relevant types
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except (ValueError, TypeError):
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                pass  # If conversion fails, keep the column as object
    stats = {}
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in df.columns:
        # if col values are numeric cases.select_dtypes(include=[np.number]).columns
        if col in numeric_columns:
            stats[col] = {
                'count': df[col].count(),
                'mean': df[col].mean(),
                'std': df[col].std(),
                'median': df[col].median(),
                'unique': df[col].nunique(),
                'min': df[col].min(),
                'max': df[col].max(),
                'iqr': iqr(df[col].dropna())  # IQR requires non-null values
            }
        else:  # Non-numeric columns (e.g., record_id)
            stats[col] = {
                'count': df[col].count(),
                'unique': df[col].nunique()
            }
    df_ret = pd.DataFrame(stats)
    # round 2
    return df_ret.round(2)

import mne
import yasa
import matplotlib.pyplot as plt
import numpy as np

def detect_sleep_stages(raw, plot=True):
    """
    Detect sleep stages using YASA on a selected EEG channel.

    Parameters:
    -----------
    raw : mne.io.Raw
        MNE Raw object containing the EEG data.
    channel : str, optional
        The channel name to use for sleep staging (default is 'Fpz').
    epoch_sec : int, optional
        The length (in seconds) of each epoch (default is 30).
    plot : bool, optional
        If True, plots the hypnogram of the sleep stages.

    Returns:
    --------
    predicted_stages : numpy.ndarray
        Array of predicted sleep stage labels for each epoch.
    """
    # Check if the chosen channel is available in the raw data
    channels = raw.info['ch_names']
    # Fz if exists if not C4 if not C3
    eeg_name = 'Fz' if 'Fz' in channels else 'C4' if 'C4' in channels else 'C3'
    # Initialize and run YASA's sleep staging
    ss = yasa.SleepStaging(raw,eeg_name=eeg_name)
    predicted_stages = ss.predict()

    # Plot hypnogram if requested
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(predicted_stages, drawstyle='steps-post')
        plt.xlabel('Epochs (sec segments)')
        plt.ylabel('Sleep Stage')
        plt.title(f'Sleep Staging Hypnogram from {eeg_name}')
        # Map stages to labels (0 = Wake, 1 = N1, 2 = N2, 3 = N3, 5 = REM)
        plt.yticks([0, 1, 2, 3, 5], ['W', 'N1', 'N2', 'N3', 'REM'])
        plt.grid(True)
        plt.show(block=False)
    return predicted_stages

def fix_predicted_stages(predicted_stages, min_non_w=10):
    """
    Fix the predicted_stages array by turning sequences of fewer than `min_non_w` 'W' values
    into the nearest neighboring stage.

    Parameters:
    -----------
    predicted_stages : np.ndarray
        Array of predicted sleep stage labels.
    min_non_w : int
        Minimum number of consecutive 'W' values to keep as 'W'.

    Returns:
    --------
    np.ndarray
        Fixed predicted_stages array.
    """
    fixed_stages = predicted_stages.copy()
    w_indices = np.where(predicted_stages == 'W')[0]  # Indices of 'W' values

    # Group consecutive indices of 'W' values
    groups = np.split(w_indices, np.where(np.diff(w_indices) != 1)[0] + 1)

    for group in groups:
        if len(group) < min_non_w and len(group) > 0:
            # Replace with the nearest neighboring stage
            start_idx = group[0]
            end_idx = group[-1]

            # Get the value before the group (if it exists)
            before_value = fixed_stages[start_idx - 1] if start_idx > 0 else None
            # Get the value after the group (if it exists)
            after_value = fixed_stages[end_idx + 1] if end_idx + 1 < len(fixed_stages) else None

            # Decide the replacement value
            if before_value == after_value:
                replacement_value = before_value  # Use the same value if both neighbors are the same
            elif before_value is not None:
                replacement_value = before_value  # Prefer the value before the group
            elif after_value is not None:
                replacement_value = after_value  # Otherwise, use the value after the group
            else:
                replacement_value = 'W'  # Default to 'W' if no neighbors exist

            # Replace the group with the determined value
            fixed_stages[group] = replacement_value

    return fixed_stages

def detect_sleep(cases_group_name, save_sleep_only=False):
    # Get pickle files from pickles/{cases_group_name}/*.pkl
    case_files = []
    for root, dirs, files in os.walk(f'pickles/{cases_group_name}'):
        for file in files:
            if file.endswith('.pkl'):
                case_files.append(os.path.join(root, file))

    # Create output directory for wake data
    wake_dir = f'pickles/wake_{cases_group_name}'
    os.makedirs(wake_dir, exist_ok=True)
    if save_sleep_only:
        sleep_dir = f'pickles/sleep_{cases_group_name}'
        os.makedirs(sleep_dir, exist_ok=True)

    for case_file in case_files:
        # Load the file
        try:
            with open(case_file, 'rb') as f:
                raw = pickle.load(f)
        except Exception as e:
            print(f"Failed to load {case_file}: {e}")
            continue

        # Get the ID from the file name
        ID = os.path.basename(case_file).split('.')[0].split(' ')[0]

        # Detect sleep stages
        predicted_stages = detect_sleep_stages(raw, plot=False)
        duration_sec = raw.times[-1] - raw.times[0]
        epoch_length_sec = int(duration_sec / len(predicted_stages))
        # min_non_w to be 10 minutes. based on epoch_length_sec
        min_non_w = int(10 * 60 / epoch_length_sec)
        # Fix the predicted_stages array
        predicted_stages = fix_predicted_stages(predicted_stages, min_non_w=min_non_w)

        # If there are no awake ('W') segments, skip this file
        if not any(predicted_stages == 'W'):
            print(f"No awake segments found for ID {ID}. Skipping...")
            continue

        # Get the indices of the awake ('W') segments
        awake_indices = np.where(predicted_stages == 'W')[0]
        # Group consecutive awake indices
        awake_groups = np.split(awake_indices, np.where(np.diff(awake_indices) != 1)[0] + 1)
        # Initialize an empty list to store the awake segments
        awake_segments = []
        # Convert awake groups to time in seconds and crop the raw data
        for group in awake_groups:
            start_time = group[0] * epoch_length_sec
            end_time = (group[-1] + 1) * epoch_length_sec
            awake_segments.append(raw.copy().crop(tmin=start_time, tmax=end_time, include_tmax=False))

        # Concatenate all awake segments into a single raw object
        awake_raw = mne.concatenate_raws(awake_segments)

        # Save the clipped raw data to a new pickle file
        wake_file_path = os.path.join(f'{case_file}')
        # remove .edf and .csv
        wake_file_path = wake_file_path.replace('.edf', '').replace('.csv', '')
        # replace 'EDF' with 'wake_EDF'
        wake_file_path = wake_file_path.replace('EDF', f'wake_{cases_group_name}')
        with open(wake_file_path, 'wb') as f:
            pickle.dump(awake_raw, f)

        print(f"Saved awake data for ID {ID} to {wake_file_path}")

        # Save sleep segments if requested
        if save_sleep_only:
            # Find indices where predicted_stages != 'W'
            sleep_indices = np.where(predicted_stages != 'W')[0]
            if len(sleep_indices) == 0:
                print(f"No sleep segments found for ID {ID}.")
                continue
            sleep_groups = np.split(sleep_indices, np.where(np.diff(sleep_indices) != 1)[0] + 1)
            sleep_segments = []
            for group in sleep_groups:
                start_time = group[0] * epoch_length_sec
                end_time = (group[-1] + 1) * epoch_length_sec
                sleep_segments.append(raw.copy().crop(tmin=start_time, tmax=end_time, include_tmax=False))
            if sleep_segments:
                sleep_raw = mne.concatenate_raws(sleep_segments)
                sleep_file_path = case_file.replace('.edf', '').replace('.csv', '')
                sleep_file_path = sleep_file_path.replace('EDF', f'sleep_{cases_group_name}')
                with open(sleep_file_path, 'wb') as f:
                    pickle.dump(sleep_raw, f)
                print(f"Saved sleep data for ID {ID} to {sleep_file_path}")
    return

def read_pkl_files(path):
    """
    Read all pickle files from a given path, process them per ID, and save individual metadata as CSV files.
    Merge all individual CSV files into one consolidated CSV file.

    Parameters:
    -----------
    path : str
        Path to the directory containing the pickle files.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing merged metadata for all file IDs.
    """
    pkl_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pkl')]
    metadata_list = []
    temp_dir = 'temps_awake_EDF'
    os.makedirs(temp_dir, exist_ok=True)

    for file in pkl_files:
        with open(file, 'rb') as f:
            raw = pickle.load(f)
            # Extract ID from the filename (assuming ID is part of the filename)
            file_id = os.path.basename(file).split(' ')[0]
            metadata = eeg_data_to_features(raw)
            metadata['ID'] = file_id  # Add ID to metadata
            metadata['duration_sec'] = raw.times[-1] - raw.times[0]
            metadata['duration_min'] = metadata['duration_sec'] / 60
            metadata_list.append(metadata)
            # replace .pkl with .csv
            file_name = os.path.basename(file).replace('.pkl', '')
            # Save individual metadata to a CSV file
            individual_csv_path = os.path.join(temp_dir, f'{file_name}.csv')
            pd.DataFrame([metadata]).to_csv(individual_csv_path, index=False)
    # Merge all individual CSV files into one consolidated CSV file
    merge_csv_files(temp_dir)

    
def id_from_csv_filename(filename):
    # re get '0345-{3%d}' from filename
    match = re.search(r'0345-\d{3}', filename)
    if match:
        return match.group(0)[1:]
    else:
        print(f"ID not found in filename: {filename}")
        return None
    
def merge_csv_files(temp_dir):
    # Merge all individual CSV files into one consolidated CSV file
    all_metadata_df = pd.DataFrame()
    for f in os.listdir(temp_dir):
        if f.endswith('.csv'):
            temp_df = pd.read_csv(os.path.join(temp_dir, f))

            temp_df['file_name'] = f.split(' ')[0]  # Extract file name from filename
            temp_df['csv_file_name'] = f  # Extract file name from filename
            # with regex find 0345-%d
            temp_df['ID'] = temp_df['csv_file_name'].apply(id_from_csv_filename)
            # remove ID from file name
            all_metadata_df = pd.concat([all_metadata_df, temp_df], ignore_index=True)
    consolidated_csv_path = os.path.join('EDF_awake.csv')
    # move ID to the first column
    cols = all_metadata_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('ID')))
    all_metadata_df = all_metadata_df[cols]
    # Save the consolidated DataFrame to a CSV file
    all_metadata_df.to_csv(consolidated_csv_path, index=False)
    return
    # print how many unique ID
    f"Total unique IDs in the consolidated CSV: {all_metadata_df['ID'].nunique()}"
    all_metadata_df['ID'].unique()

def raw_run(cases_group_name='EDF'):
    group = 'west_nile_virus' if cases_group_name == 'WNV' else 'EDF'
    pickle_files = [f for f in os.listdir(f'pickles/{group}') if f.endswith('.pkl')]
    pickle_file = st.selectbox('Select a file', pickle_files)
    
    with open(f'pickles/{group}/{pickle_file}', 'rb') as f:
        raw = pickle.load(f)
    
    total_duration = raw.times[-1]
    st.write(f"Total duration: {total_duration:.2f} seconds ({total_duration / 60:.2f} minutes)")

    start_time = st.number_input("Select start time (seconds)", min_value=0.0, max_value=max(0.0, total_duration), value=0.0, step=10.0, format="%.1f")
    end_time = min(start_time + 10, total_duration)
    cropped_raw = raw.copy().crop(tmin=start_time, tmax=end_time, include_tmax=False)

    st.write(f"Displaying data from {start_time:.2f} to {end_time:.2f} seconds")
    fig = cropped_raw.plot(show=False, block=False)
    st_pyplot_func(fig)
    
    # Compute power spectral density (PSD) for delta band (0.5-4 Hz)
    psd, freqs = raw.compute_psd(fmin=0.5, fmax=4.0).get_data(return_freqs=True)
    delta_power = psd.mean(axis=1)  # Average across frequencies for each channel

    # Find the channel with the highest delta power
    max_channel_idx = delta_power.argmax()
    max_channel_name = raw.info['ch_names'][max_channel_idx]
    st.write(f"Channel with most delta power: {max_channel_name}")

    window_size = st.sidebar.slider("Select window size in seconds", 10, int(total_duration//2), 5)
    # Plot the channel with the highest delta power
    fig_max_channel = raw.plot(start=start_time, duration=window_size, picks=[max_channel_idx], show=False, block=False)
    st.write(f"Plot of {max_channel_name} with most delta power")
    st_pyplot_func(fig_max_channel)

    # Find the channel with the lowest delta power
    min_channel_idx = delta_power.argmin()
    min_channel_name = raw.info['ch_names'][min_channel_idx]
    st.write(f"Channel with least delta power: {min_channel_name}")

    # Plot the channel with the lowest delta power
    fig_min_channel = raw.plot(start=start_time, duration=window_size, picks=[min_channel_idx], show=False, block=False)
    st.write(f"Plot of {min_channel_name} with least delta power")
    st_pyplot_func(fig_min_channel)

    # Find region with the highest delta power across all channels
    max_power_idx = delta_power.argmax()
    start_slow = max_power_idx * window_size
    end_slow = min(start_slow + window_size, total_duration)
    slowing_raw = raw.copy().crop(tmin=start_slow, tmax=end_slow, include_tmax=False)
    st.write(f"Most delta power: {start_slow:.2f} to {end_slow:.2f} seconds")
    fig_slow = slowing_raw.plot(show=False, block=False)
    st_pyplot_func(fig_slow)

    # Find region with the lowest delta power across all channels
    min_power_idx = delta_power.argmin()
    start_fast = min_power_idx * window_size
    end_fast = min(start_fast + window_size, total_duration)
    speeding_raw = raw.copy().crop(tmin=start_fast, tmax=end_fast, include_tmax=False)
    st.write(f"Least delta power: {start_fast:.2f} to {end_fast:.2f} seconds")
    fig_fast = speeding_raw.plot(show=False, block=False)
    st_pyplot_func(fig_fast)

def spectogram_run(group,figures_dir=None,win_sec=5):
    if figures_dir is None:
        # read f'pickles/group_mean/{group}_mean.pkl'
        with open(f'pickles/group_mean/{group}_mean.pkl', 'rb') as f:
            di = pickle.load(f)
        raw = di['raw']
        arr_mean = di['arr_mean']
    else:
        # Ensure the directory exists
        os.makedirs(f'{figures_dir}/spectograms', exist_ok=True)
        # Read all pickle files from pickles/{group}
        pickle_files = [f for f in os.listdir(f'pickles/{group}') if f.endswith('.pkl')] 
        arr = []
        for i, pickle_file in enumerate(pickle_files):
            # Load the data
            raw = pd.read_pickle(f'pickles/{group}/{pickle_file}')
            data = raw.get_data()
            arr.append(data)
        arr_mean = mean_of_resized_arrays(arr)
        # save arr_mean to pkl
        os.makedirs(f'pickles/group_mean', exist_ok=True)
        with open(f"pickles/group_mean/{figures_dir.split('_')[0]}_mean.pkl", 'wb') as f:
            pickle.dump({'raw': raw, 'arr_mean': arr_mean}, f)
    #%%  spectrogram
    # get eeg channels that are in raw.info['ch_names'] and in eeg_channels
    channels = [ch for ch in eeg_channels if ch in raw.info['ch_names']]
    sf = raw.info['sfreq']
    show_per_channel = True
    if figures_dir is None:
        show_per_channel = st.button("Show spectrogram per channel")
    if show_per_channel:
        for i, ch in enumerate(channels):
            # plot spectrogram
            fig = yasa.plot_spectrogram(arr_mean[i, :], sf, win_sec=win_sec, ch_names=[ch], cmap='jet')
            if figures_dir is None:
                    st.subheader(f'{ch}')
                    st_pyplot_func(fig)
            else:
                fig.savefig(f'{figures_dir}/spectograms/{group}_{ch}_spectrogram.png')
            plt.close(fig)
    # mean over all channels
    fig = yasa.plot_spectrogram(arr_mean.mean(axis=0), sf,win_sec=win_sec, ch_names=['mean'],cmap='jet')
    if figures_dir is None:
        st.title(f'Spectrogram mean')
        st_pyplot_func(fig)
    else:
        fig.savefig(f'{figures_dir}/spectograms/{group}_mean_spectrogram.png')
        
# if name == main
if __name__ == '__main__':
    # Example usage
    detect_sleep('EDF', save_sleep_only=True)
    # read_pkl_files('pickles/wake_EDF')

    # merge_csv_files('temps_awake_EDF')

    # df_wnv,patients_folder,controls,df_wnv2,cases_group_name = cobrad_get_files(sample_window_size=600,only_awake=True)
    
    # df_wnv,patients_folder,controls,df_wnv2,cases_group_name = wnv_get_files()
