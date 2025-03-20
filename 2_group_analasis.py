import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import numpy as np
import os
import mne
from scipy.stats import ttest_ind
import statsmodels.stats.multitest as smm
from scipy.signal import spectrogram
import yasa
from sklearn.decomposition import PCA
import statsmodels.api as sm
from collections import Counter
import scienceplots
from eeg_utils import *

# plt.style.use('science')
# make {figures_dir} prettier
sns.set_context('talk')
sns.set_style('white')
# put grid in all {figures_dir}
plt.rcParams['axes.grid'] = True
# add ticks to both sides 
plt.rc('xtick', bottom   = True)
plt.rc('ytick', left = True)
plt.rc('font',  family='serif',)
plt.rc('text',  usetex=False)
# make labels slightly smaller 
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
plt.rc('axes',  labelsize=11)
plt.rc('legend',  handlelength=4.0)
plt.rc('axes',  titlesize=12)  # Set title size to be the same as x and y labels

def boxplot_plot(results_df,combined_df, col, output_dir):
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
    if sig_symbol != 'ns':
        title_text = (
            f"{row['Test']}\n"
            f"p = {row['adj_p_value']:.3e} ({sig_symbol})\n"
            f"Cohen's d = {row['Cohen_d']:.2f}\n"
            f"{col} Comparison"
        )
        plt.title(title_text, ha='center')
        # Add sample size to x-axis labels
        group_counts = combined_df['Group'].value_counts()
        ax = plt.gca()
        ax.set_xticklabels([f"{label.get_text()}\nn={group_counts[label.get_text()]}" for label in ax.get_xticklabels()])
        plt.tight_layout()
        os.makedirs(f'{figures_dir}/boxplots/{output_dir}', exist_ok=True)
        plt.savefig(f"{figures_dir}/boxplots/{output_dir}/{col}_comparison.png")
    plt.close()

def scatter_plot_with_regression(results_df, combined_df, x_col, y_col, output_dir):
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
    # Add significance markers
    max_y = combined_df[y_col].max()
    if sig_symbol != 'ns':
        # Add stats results to the title
        plt.title(
            f"{x_col} vs {y_col} Regression\n"
            f"Slope p = {p_value:.3e} ({sig_symbol}), R^2 = {r_squared:.2f}, n = {len(combined_df)}",
            ha='center'
        )
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    if sig_symbol != 'ns':
        # Make folder {figures_dir}
        os.makedirs(f'{figures_dir}/scatterplots/{output_dir}', exist_ok=True)
        plt.savefig(f"{figures_dir}/scatterplots/{output_dir}/{x_col}_vs_{y_col}_regression.png")
    plt.close()
    
def analyze_and_correct(combined_df, columns_to_analyze, groups=['Control', 'WNV']):
    def analyze_groups(combined_df, col, groups):
        # Split data
        control_data = combined_df[combined_df['Group'] == groups[0]][col].dropna()
        case_data = combined_df[combined_df['Group'] == groups[1]][col].dropna()
        if len(control_data) < 2 or len(case_data) < 2:
            return None
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
        return {
            'Variable': col,
            'Test': test_used,
            'Statistic': stat,
            'p_value': p,
            'Cohen_d': cohen_d
        }

    results = []    
    # Statistical analysis
    for col in columns_to_analyze:
        result = analyze_groups(combined_df, col, groups)
        if result is not None:
            results.append(result)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Multiple testing correction
    if not results_df.empty:
        rej, adj_p, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
        results_df['adj_p_value'] = adj_p
        results_df['Significant'] = adj_p < 0.05

    return results_df

def topomap_group_data( band, montage,control_data,wnv_data,output_dir):
    # Calculate p-values for each channel
    common_channels = control_data.columns.intersection(wnv_data.columns)
    dict_p_values = {}
    for common_channel in common_channels:
        control_channel = control_data[common_channel].dropna()
        wnv_channel = wnv_data[common_channel].dropna()
        if len(control_channel) < 2 or len(wnv_channel) < 2:
            continue
        _, p = ttest_ind(control_channel, wnv_channel)
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
        os.makedirs(f'{figures_dir}/topomaps_p_values_vs_controls/{output_dir}', exist_ok=True)
        # Save the figure
        plt.savefig(f"{figures_dir}/topomaps_p_values_vs_controls/{output_dir}/p_values_{band}_topomap.png")
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
        plt.title(f"{band} Topomap")
        # Save the figure
        os.makedirs(f'{figures_dir}/topomaps', exist_ok=True)
        plt.savefig(f"{figures_dir}/topomaps/{group}_{band}_topomap.png")
        plt.close()
    return group_data   

#%% Choose project
# df_wnv,patients_folder,control_folder,controls,cases,df_wnv2,cases_group_name = wnv_get_files()
df_wnv,patients_folder,control_folder,controls,df_wnv2,cases_group_name = cobrad_get_files()
#%% Initialize variables
figures_dir = f'{cases_group_name}_figures'
# Add group labels
controls['Group'] = 'Control'
df_wnv2['Group'] = cases_group_name
# all columns that have EEG, their split(' ')[-1] needs to be uppercase first letter, all rest lower case
df_wnv2.columns = [' '.join([part.capitalize() if i == len(col.split(' ')) - 1 else part for i, part in enumerate(col.split(' '))]) if 'EEG' in col else col for col in df_wnv2.columns]
controls.columns = [' '.join([part.capitalize() if i == len(col.split(' ')) - 1 else part for i, part in enumerate(col.split(' '))]) if 'EEG' in col else col for col in controls.columns]
# Combine datasets
combined_df = pd.concat([controls, df_wnv2], ignore_index=True)
combined_df['sampling_frequency']
combined_df.columns.tolist()
# Initialize results storage
results = []
cols_to_skip = ['Group', 'patient_number']
columns_to_analyze = [col for col in combined_df.columns if col not in cols_to_skip
                     and pd.api.types.is_numeric_dtype(combined_df[col])]
# make folder {figures_dir}/topomaps
os.makedirs(f'{figures_dir}/topomaps', exist_ok=True)
# Define the channel locations (you need to have this information)
montage = mne.channels.make_standard_montage('standard_1020')
eeg_channels = eeg_channels
eeg_dict_convertion = eeg_dict_convertion
# Iterate over each frequency band and plot the topomap
frequency_bands = ['delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power','pswe_events_per_minute_EEG','pswe_avg_length_EEG','mean_mpf','dfv_std','dfv_mean']

#%% clinical data analysis
clinical_columns, boxplot_columns = get_clinical_and_boxplot_cols(df_wnv2=df_wnv2)
# Visualization
numeric_cols = df_wnv2.select_dtypes(include=[np.number]).columns
all_group_data = []
# Iterate over clinical columns
for col in clinical_columns:
    df_wnv3 = df_wnv2[df_wnv2[col].notna()].copy()
    unique_values = df_wnv3[col].unique()
    print(f'Analyzing {col} with {len(unique_values)} unique values')
    if df_wnv3.shape[0] < 3 or unique_values.shape[0] < 2:
        continue
    if len(unique_values) == 2:  # Check if binary
        # check that there are at least 3 in each group (0,1)
        if len(df_wnv3[df_wnv3[col] == 1]) < 3 or len(df_wnv3[df_wnv3[col] == 0]) < 3:
            continue
        for band in boxplot_columns:
            if col == 'sex':
                # if 1 'f' else 'm'
                df_wnv3['Group'] = df_wnv3[col].apply(lambda x: 'f' if x == 1 else 'm')
            elif col == 'sex, 1=male':
                df_wnv3['Group'] = df_wnv3[col].apply(lambda x: 'm' if x == 1 else 'f')
            else:
                # group values based on band if =1, else f'not {band}'
                df_wnv3['Group'] = df_wnv3[col].apply(lambda x: col if x == 1 else f'not {col}')
            results_df = analyze_and_correct(df_wnv3, [band], groups=df_wnv3['Group'].unique())
            boxplot_plot(results_df, df_wnv3, band, f'{col}')
        # if frequency band is contained in the column name
        group_data = {}
        for value in unique_values:
            group = col if value == 1 else f'not {col}'
            run_df = df_wnv3[df_wnv3[col] == value]
            group_data = process_group_data(group, run_df, frequency_bands, eeg_dict_convertion, eeg_channels, montage,group_data)
        all_group_data.append(group_data)
    # If numeric non-binary
    elif col in numeric_cols:
        for band in boxplot_columns:
            scatter_plot_with_regression({}, df_wnv3, col, band, f'{col}_scatterplots')
#%% Topomap per clinical column
# Calculate p-values for each band and channel
for group_data2 in all_group_data:
    keys = list(group_data2.keys())
    # output_dir  is the key which doesnt say not. if key is m, output_dir is sex
    output_dir = [key for key in keys if 'not' not in key][0]
    if 'm' in keys:
        output_dir = 'sex'
    for band in frequency_bands:
        control_data = group_data2[keys[0]][band]
        wnv_data = group_data2[keys[1]][band]
        topomap_group_data(band, montage,control_data,wnv_data,output_dir)
#%% Topomap per group CONTROLS
# run over controls and cases
group_data = {}
for group in ['Control', cases_group_name]:
    run_df = combined_df[combined_df['Group'] == group]
    group_data = process_group_data(group, run_df, frequency_bands, eeg_dict_convertion, eeg_channels, montage,group_data)

### Topomap P-Value
# Calculate p-values for each band and channel
for band in frequency_bands:
    control_data = group_data['Control'][band]
    wnv_data = group_data[cases_group_name][band]
    topomap_group_data(band, montage,control_data,wnv_data,'topomaps_p_values_vs_controls')

results_df = analyze_and_correct(combined_df, columns_to_analyze,groups=['Control', cases_group_name])
# Save statistical results
results_df.to_csv(f"{patients_folder}_analysis_results.csv", index=False)

### Boxplot Group Comparison
# columns to analyze which contains overall
# Visualization
for col in boxplot_columns:
    curr_data = combined_df[[col, 'Group']].dropna()
    num_groups = curr_data['Group'].nunique()
    if num_groups < 2:
        continue
    boxplot_plot(results_df,curr_data, col, 'vs_controls')

def mean_of_resized_arrays(arrays):
    # Get the shapes of all arrays
    shapes = np.array([arr.shape for arr in arrays])
    
    # Compute median dimensions
    median_shape = tuple(np.median(shapes, axis=0).astype(int))
    
    # Resize all arrays to the median shape
    resized_arrays = np.array([np.resize(arr, median_shape) for arr in arrays])
    
    # Compute the mean
    return np.mean(resized_arrays, axis=0)
#%% Spectrogram
def spectogram_run():
    # Ensure the directory exists
    os.makedirs(f'{figures_dir}/spectograms', exist_ok=True)
    for group in [patients_folder, control_folder]:
        # Read all pickle files from pickles/{group}
        pickle_files = [f for f in os.listdir(f'pickles/{group}') if f.endswith('.pkl')] 
        arr = []
        for i, pickle_file in enumerate(pickle_files):
            # Load the data
            raw = pd.read_pickle(f'pickles/{group}/{pickle_file}')
            data = raw.get_data()
            arr.append(data)
            if i ==0:
                sf = raw.info['sfreq']
        arr_mean = mean_of_resized_arrays(arr)
        #%%  spectrogram
        # pca of eeg_data
        pca = PCA(n_components=0.95)
        pca.fit(arr_mean.T)
        eeg_data_pca = pca.transform(arr_mean.T)
        print(pca.explained_variance_ratio_)
        # plot spectrogram
        fig = yasa.plot_spectrogram(eeg_data_pca.T[0,:], sf,win_sec=1)
        fig.savefig(f'{figures_dir}/spectograms/{group}_spectrogram.png')
        plt.close(fig)
    
# spectogram_run()