import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import iqr

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
    sheets_to_read = ['clinical', 'medications', 'npi-q', 'epworth', 'isi', 'ecog_12','Sheet4']
    sheets_to_sum_vals = ['epworth', 'isi', 'ecog_12','Sheet4']
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
        df_wnv = pd.merge(df_wnv, dfs[sheet], on='ID', how='inner')
    # rename record_id to ID
    df_wnv = df_wnv.rename(columns={'record_id':'ID'}).astype(str)
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
    cases['ID'] = cases['file_name'].apply(lambda x: x.split('.')[0]).astype(str)
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
    return df_wnv,patients_folder,control_folder,controls,df_wnv2,cases_group_name
    df_merged['ID'].unique()

def get_clinical_and_boxplot_cols(df_wnv2):
       boxplot_columns = [col for col in df_wnv2.columns if 'overall' in col.lower()]
       # split file name .[0] and then '-'[0] to get the ID
       clinical_columns_all = df_wnv2.columns[3:].tolist()
       # remove boxplot_columns from clinical_columns
       clinical_columns = [col for col in clinical_columns_all if col not in boxplot_columns]
       # Remove columns that contain 'EEG'
       clinical_columns = [col for col in clinical_columns if 'EEG' not in col]
       return clinical_columns,boxplot_columns
   
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