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
from utils.eeg_utils import *


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

  

#%% Choose project
# df_wnv,patients_folder,control_folder,controls,df_wnv2,cases_group_name = wnv_get_files()
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
# Define the channel locations (you need to have this information)
montage = mne.channels.make_standard_montage('standard_1020')
eeg_channels = eeg_channels
eeg_dict_convertion = eeg_dict_convertion
# Iterate over each frequency band and plot the topomap
frequency_bands = ['delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power','pswe_events_per_minute','pswe_avg_length','mean_mpf','dfv_std','dfv_mean']

#%% clinical data analysis
clinical_columns, boxplot_columns = get_clinical_and_boxplot_cols(df_wnv2=df_wnv2)
# map all 'nan' to np.nan
df_wnv2 = df_wnv2.replace('nan', np.nan)
# all numeric string to float
for col in df_wnv2.columns:
    try:
        df_wnv2[col] = pd.to_numeric(df_wnv2[col])
    except:
        # print(f'Could not convert {col} to numeric')
        pass
numeric_cols = df_wnv2.select_dtypes(include=[np.number]).columns
all_group_data = []
save_raw_data(df_wnv2, figures_dir)
# Iterate over clinical columns
for col in clinical_columns:
    # drop Nan, None, nan values
    df_wnv3 = df_wnv2.dropna(subset=[col]).copy()
    unique_values = df_wnv3[col].unique()
    # Save the raw data
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
            boxplot_plot(results_df, df_wnv3, band, f'{col}', figures_dir)
        # if frequency band is contained in the column name
        group_data = {}
        for value in unique_values:
            group = col if value == 1 else f'not {col}'
            run_df = df_wnv3[df_wnv3[col] == value]
            group_data = process_group_data(group, run_df, frequency_bands, eeg_dict_convertion, eeg_channels, montage, group_data)
        all_group_data.append(group_data)
    # if col name has ( and )
    elif '(' in col and ')' in col:
        for band in boxplot_columns:
            df_wnv3['Group'] = df_wnv3[col]
            # do boxplot for each band
            results_df = analyze_and_correct(df_wnv3, [band], groups=df_wnv3['Group'].unique())
            boxplot_plot(results_df, df_wnv3, band, f'{col}', figures_dir)
    # If numeric non-binary
    elif col in numeric_cols:
        for band in boxplot_columns:
            scatter_plot_with_regression({}, df_wnv3, col, band, f'{col}', figures_dir)
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
        topomap_group_data(band, montage,control_data,wnv_data,output_dir,figures_dir=figures_dir)
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
    topomap_group_data(band, montage,control_data,wnv_data,'vs_controls',figures_dir=figures_dir)

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
    boxplot_plot(results_df,curr_data, col, 'vs_controls', figures_dir)

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