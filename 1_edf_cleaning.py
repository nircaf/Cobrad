import os
from collections import defaultdict
import pandas as pd
import sys
import pyedflib
import numpy as np
import warnings
import mne
import re
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import concurrent.futures
import utils.eeg_utils as eeg_utils
from scipy.signal import spectrogram
import pickle
from autoreject import AutoReject
from pyprep.prep_pipeline import PrepPipeline
import matplotlib.pyplot as plt
import yasa
# sklearn.preprocessing.MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from contextlib import contextmanager
from utils.eeg_utils import *

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
# sys arrent cur dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())

is_prod = not any('vscode' in arg.lower() for arg in sys.argv)
use_multiprocessing = False
# Suppress specific RuntimeWarnings
warnings.filterwarnings("ignore", message="Channels contain different highpass filters. Highest filter setting will be stored.")
warnings.filterwarnings("ignore", message="Channels contain different lowpass filters. Lowest filter setting will be stored.")
warnings.filterwarnings("ignore", message="Effective window size : 1.000 (s)")
getcwd = os.getcwd()

#%% INITIALIZATION
cases_project_name = 'west_nile_virus'
# cases_project_name = 'EDF'
directory = os.path.join(getcwd, cases_project_name)
# directory = os.path.join(getcwd, 'Controls')
os_splittor = '\\' if 'nt' in os.name else '/'


#%% Load the data
# df_wnv = pd.read_excel(f'WNV_merged_291224_KP.xlsx')
project_name = directory.split(os_splittor)[-1]
temp_dir = f'temps_{project_name}' 
os.makedirs(temp_dir, exist_ok=True)
# make folder
os.makedirs(f'pickles/{project_name}', exist_ok=True)

def clean_df_demographics(df_demographics,patient_names):
    # strip all values
    df_demographics = df_demographics.map(lambda x: x.strip() if isinstance(x, str) else x)
    # find what column has values 'נקבה'
    sex_col_num = df_demographics.columns[df_demographics.isin(['נקבה']).any()][0]
    # move to be first col
    df_demographics = pd.concat([df_demographics.loc[:, [sex_col_num]], df_demographics.drop(columns=[sex_col_num])], axis=1)
    # rename to gender
    df_demographics.rename(columns={sex_col_num: 'Gender'}, inplace=True)
    # what column contains patient_names at least 1
    id_col_num = df_demographics.columns[df_demographics.isin(patient_names).any()][0]
    # move to be first col
    df_demographics = pd.concat([df_demographics.loc[:, [id_col_num]], df_demographics.drop(columns=[id_col_num])], axis=1)
    # rename to ID
    df_demographics.rename(columns={id_col_num: 'ID'}, inplace=True)
    return df_demographics
    df_demographics.iloc[:,0]


def controls_match(sexes,ages,controls_ratio=4):
    # Define the age groups
    age_groups = [
        '18-30 data', '31-40 data', '41-50 data', '51-60 data',
        '61-70 data', '71-80 data', '81-90 data', '90+ data'
    ]
    age_groups_int = [
        (18, 30), (31, 40), (41, 50), (51, 60),
        (61, 70), (71, 80), (81, 90), (90, 120)
    ]
    good_files_info = []
    # Count the number of ages and sexes in each age group
    for age, sex in tqdm(zip(ages, sexes), total=len(ages)):
        for group, (start, end) in zip(age_groups, age_groups_int):
            if start <= age <= end:
                break
        group_name = group.split(' ')[0]
        #list files in directory/{group} 
        group_files = os.listdir(f'{directory}/{group}')
        # get file that end with .edf
        group_files = [file for file in group_files if file.endswith('.edf')]
        patient_names = [file.split('_')[0] for file in group_files]
        # read the directory/{group}.xlsx 
        df_demographics = pd.read_excel(f'{directory}/{group}/{group_name}.xlsx', header=None)
        df_demographics = clean_df_demographics(df_demographics,patient_names)
        if start >= 61 and end <= 90:
            df_demographics2 = pd.read_excel(f'{directory}/{group}/{group_name} final data.xlsx', header=None)
            df_demographics2 = clean_df_demographics(df_demographics2,patient_names)
            df_demographics = pd.concat([df_demographics, df_demographics2],ignore_index=True)
        # remove duplicates column ID and nan
        df_demographics = df_demographics.loc[:,~df_demographics.columns.duplicated()].dropna(subset=['ID'])
        sex_col_num = 1
        df_demographics.iloc[:, sex_col_num] = df_demographics.iloc[:, sex_col_num].apply(lambda x: 'f' if x == 'נקבה' else 'm')
        # get df_demographics that match with sex
        df_demographics = df_demographics[df_demographics.iloc[:,sex_col_num] == sex]
        # get the group files in the indices that match the df_demographics.iloc[:,11] with patient_names
        filtered_group_files = []
        for i in range(len(group_files)):
            # check if exists in all values in df_demographics
            if patient_names[i] in df_demographics.iloc[:,0].values:
                filtered_group_files.append(group_files[i])
            else:
                print(f'{patient_names[i]} not in df_demographics')
        group_files = filtered_group_files
        
        good_files = 0
        for edf_files in group_files:
            if good_files >= controls_ratio:
                break
            file_path= f'{directory}/{group}/{edf_files}'
            try:
                metadata, raw = read_edf_mne(file_path)
            except:
                continue
            # check that duration min 
            if metadata['duration_min'] > 14:
                file_size = os.path.getsize(file_path)
                file_name = os.path.basename(file_path)
                # if file_name.split('_')[0] exists in good_files_info
                if file_name.split('_')[0] in [file['file_name'].split('_')[0] for file in good_files_info]:
                    print(f'{file_name} already exists. duration: {metadata["duration_min"]}')
                    continue
                good_files_info.append({
                    'file_path': file_path,
                    'size': file_size,
                    'file_name': file_name,
                })
                good_files += 1
    # Convert the list to a DataFrame
    good_files_df = pd.DataFrame(good_files_info)
    # remove duplicates
    good_files_df.drop_duplicates(subset='file_name', inplace=True)
    return good_files_df

def choose_controls_WNV(directory,controls_ratio=4):
    # read the WNV_merged_291224_KP.xlsx
    # list files in temps_west_nile_virus folder
    wnv_files = os.listdir(f'temps_west_nile_virus')
    # remove .DS_Store
    wnv_files = [file for file in wnv_files if file != '.DS_Store']
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
    ages = df_wnv2['age']
    sexes = df_wnv2['sex']
    # Convert sexes to 'f' for female and 'm' for male
    sexes = sexes.apply(lambda x: 'f' if x == 1 or x == 'נקבה' else 'm')
    print(f'Male: {sexes.value_counts(normalize=True).get("m", 0) * 100:.2f}% age: {ages.mean():.2f} ± {ages.std():.2f}')
    good_files_df = controls_match(sexes=sexes,ages=ages,controls_ratio=controls_ratio)
    return good_files_df       

def choose_controls_EDF(directory,controls_ratio=4):
    edf_files = os.listdir(f'temps_EDF')
    # remove .DS_Store
    edf_files = [file for file in edf_files if file != '.DS_Store']
    edf_files = [file.split('.edf')[0] for file in edf_files]
    # re get from edf_files 4 digits-3 digits like 0345-042
    edf_files = [re.search(r'\d{4}-\d{3}', file).group() if re.search(r'\d{4}-\d{3}', file) else None for file in edf_files]
    # remove duplicates
    edf_files = list(set(edf_files))
    # remove the first letter
    edf_files = [file[1:] for file in edf_files]
    # read excel COBRAD_clinical_24022025.xlsx
    clinical_files = pd.read_excel('COBRAD_clinical_24022025.xlsx')
    # record_id column
    clinical_files2 = clinical_files[clinical_files['record_id'].isin(edf_files)]
    # from column sex, 1=male 2=f
    sexes = clinical_files2['sex, 1=male'].apply(lambda x: 'f' if x == 2 else 'm')
    ages = clinical_files2['age_at_visit'].astype(int)
    good_files_df = controls_match(sexes=sexes,ages=ages,controls_ratio=controls_ratio)
    return good_files_df

def list_files_and_find_duplicates(directory):
    file_size_map = defaultdict(list)

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.edf'):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                file_size_map[file_size].append(file_path)

    data = []
    for size, files in file_size_map.items():
        for file in files:
            data.append({"file_path": file, "size": size})

    df = pd.DataFrame(data)
    # files sort size asc
    df.sort_values(by='size', inplace=True)
    # remove files less than 100000 bytes
    df = df[df['size'] > 100000]
    df['file_name'] = df.file_path.apply(lambda x: x.split(os_splittor)[-1])
    # file name: 0345-042 (1).edf  -> patient_number: 042
    df['patient_number'] = df.file_name.apply(lambda x: re.search(r'\d{3}-(\d{3})', x).group(1) if re.search(r'\d{3}-(\d{3})', x) else None)
    # sort by patient number
    df.sort_values(by='patient_number', inplace=True)
    return df

def get_edf_files(directory):
    edf_folder = os.path.join(directory, 'EDF')
    file_list = []

    for root, dirs, files in os.walk(edf_folder):
        for file in files:
            if file.endswith('.edf'):
                file_path = os.path.join(root, file)
                file_list.append({'file_name': file, 'file_path': file_path})

    df = pd.DataFrame(file_list)
    return df


def plot_not_prod(raw,is_prod,filename):
    if not is_prod:
        fig = raw.plot(show=False)
        # make folder if not exist
        os.makedirs(f'figures/data_cleaning', exist_ok=True)
        fig.savefig(f'figures/data_cleaning/{filename}.png')
        plt.close(fig)

def analyze_eeg_data(raw,is_prod,filename):
    raw_copy = raw.copy()
    # raw = raw_copy.copy()
    # Step 1: Preprocessing with PyPrep
    channels = raw.ch_names
    try:
        # remove channels that don't have EEG, ECG, or EOG in their name from raw
        # raw.drop_channels([channel for channel in raw.ch_names if 'EEG' not in channel])
        # remove EEG from channel name
        raw.rename_channels({channel: channel.replace('EEG', '').strip() for channel in raw.ch_names})
        # all channels that have EEG, their split(' ')[-1] needs to be uppercase first letter, all rest lower case
        raw.rename_channels({channel: ' '.join([part.capitalize() if i == len(channel.split(' ')) - 1 else part for i, part in enumerate(channel.split(' '))]) if 'EEG' in channel else channel for channel in raw.ch_names})
        # rename channels based on eeg_utils.eeg_dict_convertion
        valid_channels = set(raw.info['ch_names'])
        valid_rename_dict = {k: v for k, v in eeg_utils.eeg_dict_convertion.items() if k in valid_channels}
        raw.rename_channels(valid_rename_dict)
    except:
        pass
    for channel in raw.ch_names:
        if channel in eeg_utils.eeg_channels:
            raw.set_channel_types({channel: 'eeg'})
        elif channel == 'eog':
            raw.set_channel_types({channel: 'eog'})
        elif channel == 'ecg':
            raw.set_channel_types({channel: 'ecg'})
        else:
            raw.set_channel_types({channel: 'misc'})
    plot_not_prod(raw,is_prod,'pre_clean1')
    # Clean data
    # Filter the data
    nyquist_freq = raw.info['sfreq'] / 2
    raw.filter(l_freq=1.0, h_freq=nyquist_freq - 0.1)
    raw.notch_filter(np.arange(50, nyquist_freq, 50), filter_length='auto', phase='zero')
    plot_not_prod(raw,is_prod,'filter2')
    # picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)
    # raw.pick(picks)
    raw.resample(256.)
    # Initialize the PrepPipeline
    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(50, nyquist_freq, 50),
    }
    prep = PrepPipeline(raw, prep_params, montage="standard_1020",ransac=False)
    try:
        with suppress_stdout():
            prep.fit()  # Run the pipeline without writing to console
    except Exception as e:
        print(f'Error in PrepPipeline: {e}')
        return
    plot_not_prod(prep.raw,is_prod,'PrepPipeline4')
    raw = prep.raw  # Get cleaned data
    raw.interpolate_bads()
    # raw.preload
    # Remove bad windows using autoreject
    # raw.load_data()
    ar = AutoReject()
    epochs = mne.make_fixed_length_epochs(raw, duration=2, overlap=0.5,preload=True)
    try:
        epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
    except Exception as e:
        print(f'Error in AutoReject: {e}')
        return
    # Assuming epochs_clean is an instance of mne.Epochs
    epochs_data = epochs_clean.get_data()  # Shape: (n_epochs, n_channels, n_times)
    
    # Reshape the data to (n_channels, n_times_total)
    n_epochs, n_channels, n_times = epochs_data.shape
    reshaped_data = epochs_data.transpose(1, 0, 2).reshape(n_channels, -1)
    # Create a new RawArray object
    info = epochs_clean.info  # Use the info from the epochs
    raw = mne.io.RawArray(reshaped_data, info)
    plot_not_prod(raw,is_prod,'AutoReject5')
    # save spec_data to pickle in pickles/project_name/filename
    with open(f'pickles/{project_name}/{filename}.pkl', 'wb') as f:
        pickle.dump(raw, f)
    metadata = eeg_data_to_features(raw)
    return metadata



def process_file(row,filename,is_prod):
    metadata, raw = read_edf_mne(row['file_path'])
    metadata.update(row)
    # make folder if not exist
    os.makedirs(temp_dir, exist_ok=True)
    # if file temps/{row['file_name']}.csv exists
    if f'{row["file_name"]}.csv' in os.listdir(temp_dir):
        return 
    if metadata:
        channels = raw.ch_names
        # Check the duration of the recording
        duration_s = raw.times[-1]  # Convert duration to milliseconds
        duration_skip = 10 * 60  # Skip recordings less than 10 minutes
        if duration_s < duration_skip:
            return
        # Split the data into segments
        max_duration_s = 60 * 60  # 60 minutes in seconds
        
        if duration_s > max_duration_s:
            eeg_metadata = None
            start_i = 0
            while eeg_metadata is None:
                if max_duration_s == 0:
                    pd.DataFrame().to_csv(f'{temp_dir}/{row["file_name"]}_{max_duration_s}_{start_i}.csv', index=False)
                    start_i += 1
                    max_duration_s = 60 * 60
                    print(f'Error processing {row["file_name"]}_{max_duration_s}_{start_i}, skipping...')
                # Split the data into 60-minute segments
                n_segments = int(np.ceil(duration_s / max_duration_s))
                for i in range(start_i, n_segments):
                    segment_filename = f'{row["file_name"]}_{max_duration_s}_{i + 1}.csv'
                    if os.path.exists(f'{temp_dir}/{segment_filename}'):
                        continue
                    start = i * max_duration_s  # Start time in seconds
                    stop = min((i + 1) * max_duration_s, raw.times[-1])  # Stop time in seconds
                    raw_segment = raw.copy().crop(tmin=start, tmax=stop)
                    eeg_metadata = analyze_eeg_data(raw_segment, is_prod, segment_filename)
                    if eeg_metadata is None:
                        # Reduce max_duration_s by 10 minutes but not below 0
                        print(f'Error processing {row["file_name"]}_{max_duration_s}_{start_i}, retrying with max_duration_s={max_duration_s-600}...')
                        max_duration_s = max(max_duration_s - 5 * 60, 0)
                        break  # Exit the for loop to recalculate segments with new max_duration_s
                    # Update and write segment metadata
                    metadata.update(eeg_metadata)
                    segment_metadata = metadata.copy()
                    segment_metadata['segment'] = i + 1
                    segment_metadata['start_time'] = start
                    segment_metadata['end_time'] = stop
                    segment_metadata['duration_sec'] = stop - start
                    segment_metadata['duration_min'] = (stop - start) / 60
        
                    # Write segment metadata to CSV
                    df_segment = pd.DataFrame([segment_metadata])
                    df_segment.to_csv(f'{temp_dir}/{segment_filename}', index=False)
        else:
            # load file name csv
            try:
                df_csv = pd.read_csv(filename)
                # if file name in csv
                if row['file_name'] in df_csv['file_name'].values:
                    return 
            except:
                pass
            eeg_metadata = analyze_eeg_data(raw,is_prod,row["file_name"])
            if eeg_metadata is None:
                # save empty csv file
                # pd.DataFrame().to_csv(f'{temp_dir}/{row["file_name"]}.csv', index=False)
                print(f'Error processing {row["file_name"]}')
            else:
                metadata.update(eeg_metadata)
                # Write metadata to CSV
                df = pd.DataFrame([metadata])
                df.to_csv(f'{temp_dir}/{row["file_name"]}.csv', index=False)
        # return metadata
    # return metadata

if __name__ == "__main__":
    if project_name == 'Controls':
        temp_dir += f'_{cases_project_name}'
        if cases_project_name == 'west_nile_virus':
            df = choose_controls_WNV(directory)    
        elif cases_project_name == 'EDF':
            df = choose_controls_EDF(directory)
    else:
        df = list_files_and_find_duplicates(directory)
    # remove duplicates subset file_name
    df.drop_duplicates(subset='file_name', inplace=True)
    # leave only the files that contains 
    # df = df[df['file_name'].str.contains('010')]
    # Set multiprocessing flag
    filename = f'{project_name}.csv'
    if use_multiprocessing:
        print(f'Processing {len(df)} files in parallel...')
        # Process files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor: # max_workers=10
            futures = [executor.submit(process_file, row, filename, is_prod) for _, row in df.iterrows()]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                future.result()
    else:
        print(f'Processing {len(df)} files sequentially...')
        # Process files sequentially
        metadata_list = [process_file(row, filename,is_prod) for _, row in tqdm(df.iterrows(), total=len(df))]
    if project_name == 'Controls':
        # Combine all the temporary CSV files into a single CSV file
        filename = f'{cases_project_name}_controls.csv'
    # Process temporary files in batches
    batch_size = 100  # Adjust the batch size as needed
    # make folder if not exist
    all_files = os.listdir(temp_dir)
    # remove .DS_Store
    all_files = [file for file in all_files if file != '.DS_Store']
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i + batch_size]
        df_list = []
        for file in batch_files:
            try:
                df = pd.read_csv(os.path.join(temp_dir, file))
                df['csv_file_name'] = file
                df_list.append(df)
            except pd.errors.EmptyDataError:
                print(f"Skipping empty file: {file}")
        if df_list:
            df_batch = pd.concat(df_list)
            # move col csv_file_name to first
            df_batch = pd.concat([df_batch['csv_file_name'], df_batch.drop(columns=['csv_file_name'])], axis=1)
            # Determine if the header should be written
            write_header = not os.path.exists(filename)
            # Append the DataFrame to the CSV file
            df_batch.to_csv(filename, mode='a', header=write_header, index=False)