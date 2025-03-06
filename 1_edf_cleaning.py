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
import eeg_utils
from scipy.signal import spectrogram
import pickle
from autoreject import AutoReject
from pyprep.prep_pipeline import PrepPipeline
import matplotlib.pyplot as plt
import yasa
# sklearn.preprocessing.MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
# sys arrent cur dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())

is_prod = False
use_multiprocessing = False
# Suppress specific RuntimeWarnings
warnings.filterwarnings("ignore", message="Channels contain different highpass filters. Highest filter setting will be stored.")
warnings.filterwarnings("ignore", message="Channels contain different lowpass filters. Lowest filter setting will be stored.")
warnings.filterwarnings("ignore", message="Effective window size : 1.000 (s)")

directory = '/Users/nircafri/Desktop/Scripts/Nir/cobrad/EDF'
# directory = '/Users/nircafri/Desktop/Scripts/Nir/cobrad/west_nile_virus'
# directory = '/Users/nircafri/Desktop/Scripts/Nir/cobrad/Controls'
cases_project_name = 'west_nile_virus'
# cases_project_name = 'EDF'
df_wnv = pd.read_excel(f'WNV_merged_291224_KP.xlsx')
project_name = directory.split('/')[-1]
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
    df['file_name'] = df.file_path.apply(lambda x: x.split('/')[-1])
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
        # remove channels that don't have EEG in their name from raw
        raw.drop_channels([channel for channel in raw.ch_names if 'EEG' not in channel])
        # remove EEG from channel name
        raw.rename_channels({channel: channel.replace('EEG', '').strip() for channel in raw.ch_names})
        # rename channels based on eeg_utils.eeg_dict_convertion
        raw.rename_channels(eeg_utils.eeg_dict_convertion)
    except:
        pass
    # all channels that have EEG, their split(' ')[-1] needs to be uppercase first letter, all rest lower case
    raw.rename_channels({channel: ' '.join([part.capitalize() if i == len(channel.split(' ')) - 1 else part for i, part in enumerate(channel.split(' '))]) if 'EEG' in channel else channel for channel in raw.ch_names})
    # drop channels that are not in eeg_utils.eeg_channels
    raw.drop_channels([channel for channel in raw.ch_names if channel not in eeg_utils.eeg_channels])
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
        prep.fit()  # Run the pipeline
    except:
        return
    plot_not_prod(prep.raw,is_prod,'PrepPipeline4')
    raw = prep.raw  # Get cleaned data
    raw.interpolate_bads()
    # raw.preload
    # Remove bad windows using autoreject
    # raw.load_data()
    ar = AutoReject()
    epochs = mne.make_fixed_length_epochs(raw, duration=2, overlap=0.5,preload=True)
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
    # Assuming epochs_clean is an instance of mne.Epochs
    epochs_data = epochs_clean.get_data()  # Shape: (n_epochs, n_channels, n_times)
    
    # Reshape the data to (n_channels, n_times_total)
    n_epochs, n_channels, n_times = epochs_data.shape
    reshaped_data = epochs_data.transpose(1, 0, 2).reshape(n_channels, -1)
    # Create a new RawArray object
    info = epochs_clean.info  # Use the info from the epochs
    raw = mne.io.RawArray(reshaped_data, info)
    plot_not_prod(raw,is_prod,'AutoReject5')
    eeg_data = raw.get_data()
    # eeg_data sklearn.preprocessing.MinMaxScaler minmax norm
    # eeg_data = MinMaxScaler().fit_transform(eeg_data)
    # Define window parameters
    window_size_sec = 1  # 1-second windows for sliding
    sf = raw.info['sfreq']
    window_size = int(window_size_sec * sf)
    min_duration_sec = 5  # Minimum duration for PSWE
    
    # Initialize lists to store results
    psd_list = []
    mpf_list = []
    pswe_events_per_channel = []
    df_list = []
    dfv_list = []
    
    # Compute power spectral density (PSD) and MPF for each window and each channel
    for channel_data in eeg_data:
        channel_psd_list = []
        channel_mpf_list = []
        channel_pswe_events = []
        channel_df_list = []
        
        for start in range(0, channel_data.shape[0] - window_size + 1, window_size):
            window_data = channel_data[start:start + window_size]
            window_psd, window_freqs = mne.time_frequency.psd_array_welch(window_data, sf, fmin=1, fmax=40, n_fft=int(sf))
            window_psd = window_psd.squeeze()
            window_mpf = np.sum(window_psd * window_freqs) / np.sum(window_psd)
            dominant_freq = window_freqs[np.argmax(window_psd)]
    
            channel_psd_list.append(window_psd)
            channel_mpf_list.append(window_mpf)
            channel_df_list.append(dominant_freq)
            
            # Detect Paroxysmal Slow Wave Events (PSWE)
            if window_mpf < 6.0:
                channel_pswe_events.append(start / sf)
        
        psd_list.append(channel_psd_list)
        mpf_list.append(channel_mpf_list)
        pswe_events_per_channel.append(channel_pswe_events)
        df_list.append(channel_df_list)
        dfv_list.append(np.std(channel_df_list))
    
    # Convert lists to arrays
    psd_array = np.array(psd_list)
    mpf_array = np.array(mpf_list)
    df_array = np.array(df_list)
    dfv_array = np.array(dfv_list)
    
    # Aggregate PSWE events per channel
    pswe_stats = []
    overall_pswe_durations = []

    for channel_pswe_events in pswe_events_per_channel:
        channel_pswe_events = np.array(channel_pswe_events)
        if len(channel_pswe_events) > 0:
            pswe_sequences = find_consecutive_sequences(channel_pswe_events, min_duration_sec)
            pswe_durations = [len(seq) for seq in pswe_sequences]
        else:
            pswe_durations = np.array([])

        # Calculate PSWE statistics per channel
        total_duration = eeg_data.shape[1] / sf
        pswe_total_duration = np.sum(pswe_durations)
        pswe_percentage = (pswe_total_duration / total_duration) * 100
        pswe_events_per_minute = len(pswe_durations) / (total_duration / 60)
        pswe_avg_length = np.mean(pswe_durations) if len(pswe_durations) > 0 else 0

        pswe_stats.append({
            'pswe_percentage': pswe_percentage,
            'pswe_events_per_minute': pswe_events_per_minute,
            'pswe_avg_length': pswe_avg_length
        })

        # Collect overall PSWE durations
        overall_pswe_durations.extend(pswe_durations)

    # Calculate overall PSWE statistics
    overall_pswe_total_duration = np.median(overall_pswe_durations)
    overall_pswe_median_percentage = (overall_pswe_total_duration / total_duration) * 100
    overall_pswe_events_per_minute = len(overall_pswe_durations) / ((total_duration*len(pswe_events_per_channel)) / 60)
    overall_pswe_avg_length = np.mean(overall_pswe_durations) if len(overall_pswe_durations) > 0 else 0
    
    # Compute band power
    def bandpower(data, sf, band, window_sec=None, relative=False):
        from scipy.signal import welch
        band = np.asarray(band)
        low, high = band

        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2 / low) * sf

        freqs, psd = welch(data, sf, nperseg=nperseg)
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        bp = np.trapz(psd[:, idx_band], freqs[idx_band], axis=1)

        if relative:
            bp /= np.trapz(psd, freqs, axis=1)

        return bp
    sf = raw.info['sfreq']
    delta_power = bandpower(eeg_data, sf, [1, 4])
    theta_power = bandpower(eeg_data, sf, [4, 8])
    alpha_power = bandpower(eeg_data, sf, [8, 12])
    beta_power = bandpower(eeg_data, sf, [12, 30])
    gamma_power = bandpower(eeg_data, sf, [30, 100])
    
    # Compute additional statistics
    skewness = np.apply_along_axis(lambda x: pd.Series(x).skew(), 1, eeg_data)
    kurtosis = np.apply_along_axis(lambda x: pd.Series(x).kurt(), 1, eeg_data)
    
    # Compute FFT
    fft_data = np.fft.fft(eeg_data, axis=1)
    fft_freqs = np.fft.fftfreq(eeg_data.shape[1], 1/sf)
    # save spec_data to pickle in pickles/project_name/filename
    with open(f'pickles/{project_name}/{filename}.pkl', 'wb') as f:
        pickle.dump(raw, f)
    metadata = {}
    metadata['overall_pswe_median_percentage'] = overall_pswe_median_percentage
    metadata['overall_pswe_events_per_minute'] = overall_pswe_events_per_minute
    metadata['overall_pswe_avg_length'] = overall_pswe_avg_length
    metadata['overall_min'] = eeg_data.min()
    metadata['overall_max'] = eeg_data.max()
    metadata['overall_mean'] = eeg_data.mean()
    metadata['overall_median'] = np.median(eeg_data)
    metadata['overall_std'] = eeg_data.std()
    metadata['overall_psd_mean'] = psd_array.mean()
    metadata['overall_psd_std'] = psd_array.std()
    metadata['overall_fft_mean'] = np.abs(fft_data).mean()
    metadata['overall_fft_std'] = np.abs(fft_data).std()
    metadata['overall_delta_power'] = delta_power.mean()
    metadata['overall_theta_power'] = theta_power.mean()
    metadata['overall_alpha_power'] = alpha_power.mean()
    metadata['overall_beta_power'] = beta_power.mean()
    metadata['overall_gamma_power'] = gamma_power.mean()
    metadata['overall_skewness'] = skewness.mean()
    metadata['overall_kurtosis'] = kurtosis.mean()
    metadata['overall_mpf_mean'] = mpf_array.mean()
    metadata['overall_mpf_median'] = np.median(mpf_array)  
    metadata['overall_df_mean'] = df_array.mean()
    metadata['overall_dfv_std'] = dfv_array.mean()  
    for i, channel in enumerate(raw.ch_names):
        metadata[f'min_EEG {channel}'] = eeg_data.min(axis=1)[i]
        metadata[f'max_EEG {channel}'] = eeg_data.max(axis=1)[i]
        metadata[f'mean_EEG {channel}'] = eeg_data.mean(axis=1)[i]
        metadata[f'median_EEG {channel}'] = np.median(eeg_data, axis=1)[i]
        metadata[f'std_EEG {channel}'] = eeg_data.std(axis=1)[i]
        metadata[f'psd_mean_EEG {channel}'] = psd_array.mean(axis=2).mean(axis=1)[i]
        metadata[f'psd_std_EEG {channel}'] = psd_array.std(axis=2).mean(axis=1)[i]
        metadata[f'fft_mean_EEG {channel}'] = np.abs(fft_data).mean(axis=1)[i]
        metadata[f'fft_std_EEG {channel}'] = np.abs(fft_data).std(axis=1)[i]
        metadata[f'delta_power_EEG {channel}'] = delta_power[i]
        metadata[f'theta_power_EEG {channel}'] = theta_power[i]
        metadata[f'alpha_power_EEG {channel}'] = alpha_power[i]
        metadata[f'beta_power_EEG {channel}'] = beta_power[i]
        metadata[f'gamma_power_EEG {channel}'] = gamma_power[i]
        metadata[f'skewness_EEG {channel}'] = skewness[i]
        metadata[f'kurtosis_EEG {channel}'] = kurtosis[i]
        metadata[f'mean_mpf_EEG {channel}'] = mpf_array.mean(axis=1)[i]
        metadata[f'median_mpf_EEG {channel}'] = np.median(mpf_array, axis=1)[i]
        metadata[f'pswe_percentage_EEG {channel}'] = pswe_stats[i]['pswe_percentage']
        metadata[f'pswe_events_per_minute_EEG {channel}'] = pswe_stats[i]['pswe_events_per_minute']
        metadata[f'pswe_avg_length_EEG {channel}'] = pswe_stats[i]['pswe_avg_length']
        metadata[f'dfv_mean_EEG {channel}'] = df_array.mean(axis=1)[i]
        metadata[f'dfv_std_EEG {channel}'] = dfv_array[i]
        
    return metadata

def process_file(row,filename,is_prod):
    metadata, raw = read_edf_mne(row['file_path'])
    metadata.update(row)
    # if file temps/{row['file_name']}.csv exists
    if f'{row["file_name"]}.csv' in os.listdir(temp_dir):
        return 
    # load file name csv
    try:
        df_csv = pd.read_csv(filename)
        # if file name in csv
        if row['file_name'] in df_csv['file_name'].values:
            return 
    except:
        pass
    if metadata:
        channels = raw.ch_names
        # Check the duration of the recording
        duration_s = raw.times[-1]  # Convert duration to milliseconds
        duration_h = duration_s / 3600
        duration_skip = 10 * 60  # Skip recordings less than 10 minutes
        if duration_s < duration_skip:
            return
        # Split the data into 30-minute segments
        max_duration_s = 60 * 60   # 60 minutes in seconds
        if duration_s > max_duration_s:
            # Split the data into 60-minute segments
            n_segments = int(np.ceil(duration_s / max_duration_s))
            for i in range(n_segments):
                segment_filename = f'{row["file_name"]}_segment_{i + 1}.csv'
                if os.path.exists(f'{temp_dir}/{segment_filename}'):
                    continue
                start = i * max_duration_s   # Convert to seconds
                stop = min((i + 1) * max_duration_s , raw.times[-1])  # Convert to seconds
                raw_segment = raw.copy().crop(tmin=start, tmax=stop)
                eeg_metadata = analyze_eeg_data(raw_segment,is_prod,segment_filename)
                if eeg_metadata is None:
                    print(f'Error processing segment {i + 1} of {row["file_name"]}')
                    continue
                metadata.update(eeg_metadata)
                # add row to metadata
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
            eeg_metadata = analyze_eeg_data(raw,is_prod,row["file_name"])
            if eeg_metadata is None:
                return
            metadata.update(eeg_metadata)
            # Write metadata to CSV
            df = pd.DataFrame([metadata])
            df.to_csv(f'{temp_dir}/{row["file_name"]}.csv', index=False)
        # return metadata
    # return metadata

if __name__ == "__main__":
    # read COBRAD concent sheets.xlsx
    df_concents = pd.read_excel("COBRAD concent sheets.xlsx")
    df_concents = df_concents[['ID','Signed future research']]
    df_concents.loc[:, 'patient_number'] = df_concents.loc[:, 'ID'].apply(lambda x: f'{int(x):03d}')
    if project_name == 'Controls':
        temp_dir += f'_{cases_project_name}'
        if cases_project_name == 'west_nile_virus':
            df = choose_controls_WNV(directory)    
        elif cases_project_name == 'EDF':
            df = choose_controls_EDF(directory)
    else:
        df = list_files_and_find_duplicates(directory)
    # Set multiprocessing flag
    filename = f'{project_name}.csv'
    if use_multiprocessing:
        with concurrent.futures.ThreadPoolExecutor() as executor: # max_workers=10
            futures = [executor.submit(process_file, row, filename, is_prod) for _, row in df.iterrows()]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                future.result()
    else:
        # Process files sequentially
        metadata_list = [process_file(row, filename,is_prod) for _, row in tqdm(df.iterrows(), total=len(df))]
    if project_name == 'Controls':
        # Combine all the temporary CSV files into a single CSV file
        filename = f'{cases_project_name}_controls.csv'
    # Process temporary files in batches
    batch_size = 100  # Adjust the batch size as needed
    # make folder if not exist
    all_files = os.listdir(temp_dir)
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i + batch_size]
        df_batch = pd.concat([pd.read_csv(os.path.join(temp_dir, file)) for file in batch_files])
        # Determine if the header should be written
        write_header = not os.path.exists(filename)
        # Append the DataFrame to the CSV file
        df_batch.to_csv(filename, mode='a', header=write_header, index=False)