import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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
    sheets_to_read = ['clinical', 'medications', 'npi-q', 'epworth', 'isi', 'ecog_12']
    dfs = pd.read_excel('COBRAD_clinical_24022025.xlsx', sheet_name=sheets_to_read)
    # Rename 'record_id' to 'ID' in each DataFrame and convert to string
    for sheet in sheets_to_read:
        dfs[sheet] = dfs[sheet].rename(columns={'record_id': 'ID'}).astype(str)
        # drop col contain has_eeg
        dfs[sheet] = dfs[sheet].drop(columns=[col for col in dfs[sheet].columns if 'has_eeg' in col])
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
    df_merged = pd.merge(df_wnv, cases, on='ID', how='inner')
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
    return df_wnv,patients_folder,control_folder,controls,df_wnv2,cases_group_name