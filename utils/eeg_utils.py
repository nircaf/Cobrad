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
       'T5-O1': 'T5'
}

def stat_text_get(group_data, col=None):
    lower_bound = -1e-20
    upper_bound = 1e20
    if col is None:
        stats_text = (
            f"N = {len(group_data)}, "
            f"Mean = {np.clip(group_data.mean(), lower_bound, upper_bound):.2e}, "
            f"Median = {np.clip(group_data.median(), lower_bound, upper_bound):.2e}, "
            f"Max = {np.clip(group_data.max(), lower_bound, upper_bound):.2e}, "
            f"Min = {np.clip(group_data.min(), lower_bound, upper_bound):.2e}, "
            f"Std = {np.clip(group_data.std(), lower_bound, upper_bound):.2e}"
        )
    else:
        stats_text = (
            f"N = {len(group_data)}, "
            f"Mean = {np.clip(group_data[col].mean(), lower_bound, upper_bound):.2e}, "
            f"Median = {np.clip(group_data[col].median(), lower_bound, upper_bound):.2e}, "
            f"Max = {np.clip(group_data[col].max(), lower_bound, upper_bound):.2e}, "
            f"Min = {np.clip(group_data[col].min(), lower_bound, upper_bound):.2e}, "
            f"Std = {np.clip(group_data[col].std(), lower_bound, upper_bound):.2e}"
        )
    return stats_text

def boxplot_plot(results_df, combined_df, col, output_dir,figures_dir):
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
    if sig_symbol != 'ns':
        plt.title(title_text, ha='center')
        # Add sample size to x-axis labels
        group_counts = combined_df['Group'].value_counts()
        ax = plt.gca()
        ax.set_xticklabels([f"{label.get_text()}\nn={group_counts[label.get_text()]}" for label in ax.get_xticklabels()])
        plt.tight_layout()
        os.makedirs(f'{figures_dir}/boxplots/{output_dir}', exist_ok=True)
        plt.savefig(f"{figures_dir}/boxplots/{output_dir}/{col}_comparison.png")
        plt.close()
        # Plot histograms for each group and both groups together
        plt.figure(figsize=(10, 6))
        sns.histplot(data=cleaned_df, x=col, hue='Group', element='step', stat='density', common_norm=False)
        for i, group in enumerate(cleaned_df['Group'].unique()):
            group_data = cleaned_df[cleaned_df['Group'] == group][col]
            stats_text = stat_text_get(group_data)
            plt.annotate(stats_text, xy=(0.25, 0.95 - i * 0.1), xycoords='axes fraction', fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
        plt.title(f"{col} Histogram by Group")
        os.makedirs(f'{figures_dir}/hist/{output_dir}', exist_ok=True)
        plt.savefig(f"{figures_dir}/hist/{output_dir}/{col}_hist_by_group.png")
        plt.close()
        plt.figure(figsize=(10, 6))
        sns.histplot(data=cleaned_df, x=col, element='step', stat='density')
        combined_data = cleaned_df[col]
        stats_text = stat_text_get(combined_data)
        plt.annotate(stats_text, xy=(0.25, 0.95), xycoords='axes fraction', fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
        plt.title(f"{col} Histogram Combined")
        plt.savefig(f"{figures_dir}/hist/{output_dir}/{col}_hist_combined.png")
        plt.close()
    
def scatter_plot_with_regression(results_df, combined_df, x_col, y_col, output_dir,figures_dir):
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
        # Make folder {figures_dir}
        os.makedirs(f'{figures_dir}/scatterplots/{output_dir}', exist_ok=True)
        plt.savefig(f"{figures_dir}/scatterplots/{output_dir}/{y_col}_regression.png")
        plt.close()

        # Plot histogram of X
        plt.figure(figsize=(10, 6))
        sns.histplot(combined_df[x_col], color='blue', kde=True, stat='density', element='step')
        x_stats = stat_text_get(combined_df, x_col)
        plt.annotate(x_stats, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
        plt.title(f"Histogram of {x_col}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/scatterplots/{output_dir}/{x_col}_histogram.png")
        plt.close()

        # Plot histogram of Y
        plt.figure(figsize=(10, 6))
        sns.histplot(combined_df[y_col], color='red', kde=True, stat='density', element='step')
        y_stats = stat_text_get(combined_df, y_col)
        plt.annotate(y_stats, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
        plt.title(f"Histogram of {y_col}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/scatterplots/{output_dir}/{y_col}_histogram.png")
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
                    'Cohen_d': cohen_d
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

def topomap_group_data( band, montage,control_data,wnv_data,output_dir,figures_dir):
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
        os.makedirs(f'{figures_dir}/topomaps_p_values/{output_dir}', exist_ok=True)
        # Save the figure
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
    # Configuration
    patients_folder = "west_nile_virus"
    control_folder = f"{patients_folder}_controls"
    case_file = f"{patients_folder}.csv"
    # Read and prepare data
    controls = pd.read_csv(f'{control_folder}.csv')
    cases = pd.read_csv(case_file)
    wnv_ids = [file.split('/')[-2] for file in cases['file_path']]
    # to int
    wnv_ids = [int(id) for id in wnv_ids]
    cases['ID'] = wnv_ids
    # merge the dataframes
    df_merged = pd.merge(df_wnv, cases, on='ID', how='inner')
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
    cases_group_name = 'WNV'
    return df_wnv,patients_folder,control_folder,controls,df_wnv2,cases_group_name

#%% COBRAD
def cobrad_get_files():
    # read sheets clinical, medications, npi-q, epworth,isi, ecpg_12 from COBRAD_clinical_24022025.xlsx
    sheets_to_read = ['clinical', 'medications', 'npi-q', 'epworth', 'isi', 'ecog_12','Sheet4','seizures']
    sheets_to_sum_vals = ['epworth', 'isi', 'ecog_12','Sheet4','npi-q']
    dfs = pd.read_excel('COBRAD_clinical_24022025.xlsx', sheet_name=sheets_to_read)
    # Rename 'record_id' to 'ID' in each DataFrame and convert to string
    for sheet in sheets_to_read:
        dfs[sheet] = dfs[sheet].rename(columns={'record_id': 'ID'}).astype(str)
        # drop col contain has_eeg or has eeg . ignore case
        dfs[sheet] = dfs[sheet].drop(columns=[col for col in dfs[sheet].columns if 'has eeg' in col.lower() or 'has_eeg' in col.lower()])
        if sheet in sheets_to_sum_vals:
            # to numeric all columns but ID
            dfs[sheet] = pd.concat([dfs[sheet]['ID'], dfs[sheet].drop(columns='ID').apply(pd.to_numeric, errors='coerce')], axis=1)
            dfs[sheet][f'{sheet}_sum'] = dfs[sheet].drop(columns='ID').sum(axis=1)

    # Merge all DataFrames on 'ID'
    df_wnv = dfs[sheets_to_read[0]]
    for sheet in sheets_to_read[1:]:
        df_wnv = pd.merge(df_wnv, dfs[sheet], on='ID', how='outer')
    patients_folder = "EDF"
    control_folder = f"{patients_folder}_controls"
    case_file = f"{patients_folder}.csv"
    controls = pd.read_csv(f'{control_folder}.csv')
    controls['ID'] = controls['file_name'].apply(lambda x: x.split('_')[0]).astype(str)
    numeric_cols = controls.select_dtypes(include=[np.number]).columns
    controls = controls.groupby('ID').apply(
        lambda x: (x[numeric_cols].multiply(x['duration_min'], axis=0)).sum(skipna=False) / x['duration_min'].sum(skipna=False)
    ).reset_index()
    cases = pd.read_csv(case_file)
    cases['ID'] = cases['csv_file_name'].apply(lambda x: x.split('.')[0]).astype(str)
    # remove first letter of ID
    cases['ID'] = cases['ID'].apply(lambda x: x[1:])
    # split ' ' and get first element
    cases['ID'] = cases['ID'].apply(lambda x: x.split(' ')[0])
    # sort id ID
    cases = cases.sort_values(by='ID')
    df_merged = pd.merge(df_wnv, cases, on='ID', how='outer',indicator=True)
    
    # Get all files that end with .edf from EDF folder and subfolders
    eeg_files = []
    for root, dirs, files in os.walk('EDF'):
        for file in files:
            if file.endswith('.edf'):
                eeg_files.append(os.path.join(root, file))
    # print outer
    print('Only clinical data - eeg data currupted')
    failed_ids = df_merged[df_merged['_merge'] == 'left_only']['ID'].unique()
    # check if they exist in eeg_files
    for id_to_check in failed_ids:
        # check if id exists contains in eeg_files
        if any(id_to_check in file for file in eeg_files):
            print(f'{id_to_check}')
    df_merged = df_merged[df_merged['_merge'] == 'both'].drop(columns='_merge')
    numeric_cols = df_merged.select_dtypes(include=[np.number]).columns
    # Group by ID and apply the weighted average function
    df_wnv2 = df_merged.groupby('ID').apply(weighted_avg, weight_col='duration_min', numeric_cols=numeric_cols).reset_index(drop=True)
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
    return df_wnv,patients_folder,control_folder,controls,df_wnv2,cases_group_name
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