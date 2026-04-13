import os
from collections import defaultdict
import pandas as pd
import sys
try:
    import pyedflib
except ImportError:
    pyedflib = None  # Optional dependency
import numpy as np
import warnings
import mne
import re
from tqdm import tqdm
import pickle
from autoreject import AutoReject
from pyprep.prep_pipeline import PrepPipeline
import matplotlib.pyplot as plt
# sklearn.preprocessing.MinMaxScaler
from contextlib import contextmanager
from utils.eeg_utils import clean_df_demographics, read_edf_mne, rename_channels, raise_error, eeg_data_to_features
import ray
# Import and run the HEP processing
import glob

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
# Detect if running inside tmux
def is_tmux():
    return 'TMUX' in os.environ

use_multiprocessing = is_tmux()
use_multiprocessing = False
# Suppress specific RuntimeWarnings
warnings.filterwarnings("ignore", message="Channels contain different highpass filters. Highest filter setting will be stored.")
warnings.filterwarnings("ignore", message="Channels contain different lowpass filters. Lowest filter setting will be stored.")
warnings.filterwarnings("ignore", message="Effective window size : 1.000 (s)")
getcwd = os.getcwd()

#%% INITIALIZATION
# Parse command line arguments
cases_project_name = 'ASSY' # 'Controls' #'Seeg' #'CAP_Sleep_Database/CAP_Sleep_Database'

edf_dir = 'EDF_Format'
# Where to load the data from 
# If cases_project_name already contains 'EDF_Format', ignore it from edf_dir
if 'EDF_Format' in cases_project_name:
    directory = os.path.join(getcwd, cases_project_name)
else:
    directory = os.path.join(getcwd, edf_dir, cases_project_name)
# directory = os.path.join(getcwd, 'Controls')
os_splittor = '\\' if 'nt' in os.name else '/'

#%% Load the data
# df_wnv = pd.read_excel(f'WNV_merged_291224_KP.xlsx')
project_name = directory.split(os_splittor)[-1]
# Where to save the data to
temp_dir = f'parquet_results/{project_name}' 
os.makedirs(temp_dir, exist_ok=True)
# make folder
os.makedirs(f'pickles/{project_name}', exist_ok=True)

def get_seeg_clinical_data():
    """
    Reads SEEG clinical data, merges with folder names, and combines with parquet data for each patient.
    Returns a DataFrame with clinical and EEG features for each patient.
    """
    import pandas as pd
    import os
    # 1. Read clinical Excel
    excel_path = 'EDF_Format/Seeg/SEEG_26AUG25-ANONYMIZED.xlsx'
    sheet_name = 'SEEG_PATIENTS'
    clinical_df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # 2. Get folder names from Seeg folder (sub_code_PHASE2)
    seeg_folder = 'EDF_Format/Seeg'
    subfolders = [f for f in os.listdir(seeg_folder) if os.path.isdir(os.path.join(seeg_folder, f))]
    # Assume sub_code_PHASE2 is a column in clinical_df, and matches folder names
    clinical_df['folder_exists'] = clinical_df['sub_code_PHASE2'].apply(lambda x: x in subfolders)

    # 3. For each patient, get their parquet data
    parquet_dir = 'parquet_results/Seeg'
    parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
    # Map: patient_code -> parquet file(s)
    # Assume patient_code is in clinical_df['sub_code_PHASE2'] and in parquet file name
    patient_tables = {}
    for _, row in clinical_df.iterrows():
        patient_code = str(row['sub_code_PHASE2'])
        # Find parquet files for this patient
        matching_files = [f for f in parquet_files if patient_code in f]
        patient_data = []
        for pf in matching_files:
            try:
                df_parquet = pd.read_parquet(os.path.join(parquet_dir, pf))
                # Add clinical columns to each row
                for col in clinical_df.columns:
                    df_parquet[col] = row[col]
                patient_data.append(df_parquet)
            except Exception as e:
                print(f"Error reading {pf}: {e}")
        if patient_data:
            patient_tables[patient_code] = pd.concat(patient_data, ignore_index=True)
        else:
            patient_tables[patient_code] = pd.DataFrame([row])
    return patient_tables



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
            except Exception:
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
    wnv_files = os.listdir('temps_west_nile_virus')
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
    if 'df_wnv' not in globals():
        global df_wnv
        try:
            df_wnv = pd.read_excel('WNV_merged_291224_KP.xlsx')
        except Exception:
            df_wnv = pd.DataFrame(columns=['ID', 'age', 'sex'])

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
    edf_files = os.listdir('temps_EDF')
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

    # raise if file_size_map is empty
    if not file_size_map:
        raise ValueError('.edf files not found in directory: ' + directory)
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
    df['patient_number'] = df.file_name.apply(
        lambda x: re.search(r'\d{3}-(\d{3})', x).group(1) 
                if re.search(r'\d{3}-(\d{3})', x) 
                else x.split('.')[0]
    )
    # sort by patient number desc
    df.sort_values(by='patient_number', inplace=True, ascending=False)
    return df


def plot_not_prod(raw,is_prod,filename):
    if not is_prod:
        fig = raw.plot(show=False)
        # make folder if not exist
        os.makedirs('figures/data_cleaning', exist_ok=True)
        fig.savefig(f'figures/data_cleaning/{filename}.png')
        plt.close(fig)



def manual_epoch_rejection(epochs, filename):
    """
    Manual epoch rejection when AutoReject fails.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to clean
    filename : str
        Filename for debugging
        
    Returns
    -------
    mne.Epochs
        Cleaned epochs
    """
    print(f'[{filename}] Applying manual epoch rejection...')
    
    # Get epoch data
    epochs_data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = epochs_data.shape
    
    # Calculate rejection criteria using standard deviations
    # 1. Amplitude-based rejection (channels with extreme values)
    max_amplitudes = np.max(np.abs(epochs_data), axis=2)  # Max amplitude per epoch per channel
    amp_mean = np.mean(max_amplitudes)
    amp_std = np.std(max_amplitudes)
    amplitude_threshold = amp_mean + 3 * amp_std  # 3 SD above mean
    
    # 2. Variance-based rejection (channels with very low or very high variance)
    epoch_variances = np.var(epochs_data, axis=2)  # Variance per epoch per channel
    var_mean = np.mean(epoch_variances)
    var_std = np.std(epoch_variances)
    var_low_threshold = var_mean - 3 * var_std  # 3 SD below mean
    var_high_threshold = var_mean + 3 * var_std  # 3 SD above mean
    
    # 3. Gradient-based rejection (channels with sudden jumps)
    gradients = np.diff(epochs_data, axis=2)  # Time derivative
    max_gradients = np.max(np.abs(gradients), axis=2)  # Max gradient per epoch per channel
    grad_mean = np.mean(max_gradients)
    grad_std = np.std(max_gradients)
    gradient_threshold = grad_mean + 3 * grad_std  # 3 SD above mean
    
    # Find bad epochs
    bad_epochs = []
    for epoch_idx in range(n_epochs):
        epoch_bad = False
        
        # Check amplitude
        if np.any(max_amplitudes[epoch_idx] > amplitude_threshold):
            epoch_bad = True
            
        # Check variance
        if np.any(epoch_variances[epoch_idx] < var_low_threshold) or \
           np.any(epoch_variances[epoch_idx] > var_high_threshold):
            epoch_bad = True
            
        # Check gradient
        if np.any(max_gradients[epoch_idx] > gradient_threshold):
            epoch_bad = True
            
        if epoch_bad:
            bad_epochs.append(epoch_idx)
    
    print(f'[{filename}] Rejecting {len(bad_epochs)}/{n_epochs} epochs ({len(bad_epochs)/n_epochs*100:.1f}%)')
    
    # Create good epochs mask
    good_epochs_mask = np.ones(n_epochs, dtype=bool)
    good_epochs_mask[bad_epochs] = False
    
    # Return cleaned epochs
    if np.sum(good_epochs_mask) == 0:
        print(f'[{filename}] Warning: All epochs rejected! Using original epochs.')
        return epochs
    else:
        return epochs[good_epochs_mask]



def clean_mne_raw(raw,filename):
    raw = rename_channels(raw)
    plot_not_prod(raw,is_prod,'pre_clean1')
    # Clean data
    # Filter the data
    raw.resample(256.)
    nyquist_freq = raw.info['sfreq'] / 2
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    if len(picks) == 0:
        print('No EEG channels found after picking. Skipping this file.')
        return None
    raw.filter(l_freq=.5, h_freq=nyquist_freq - 0.1, picks=picks)
    raw.notch_filter(np.arange(50, nyquist_freq, 50), filter_length='auto', phase='zero', picks=picks)
    plot_not_prod(raw,is_prod,'filter2')
    # picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)
    # raw.pick(picks)
    # Initialize the PrepPipeline with fallback
    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(50, nyquist_freq, 50),
    }
    
    try:
        prep = PrepPipeline(raw, prep_params, montage="standard_1020", ransac=False)
        prep.fit()  # Run the pipeline without writing to console
        plot_not_prod(prep.raw, is_prod, 'PrepPipeline4')
        raw = prep.raw  # Get cleaned data
        
        # Interpolate bad channels (only if digitization info is available)
        try:
            raw.interpolate_bads()
        except RuntimeError as e:
            if "digitization" in str(e).lower() or "headshape" in str(e).lower():
                print(f'Cannot interpolate bad channels (no digitization info): {e}')
                print('Dropping bad channels instead...')
                # Drop bad channels instead of interpolating
                if len(raw.info['bads']) > 0:
                    raw.drop_channels(raw.info['bads'])
            else:
                raise_error(e, f"Unexpected interpolation error for {filename}")
    except Exception as e:
        print('--------------------------------------------------')
        print(f'Error in PrepPipeline: {e} for file: {filename}')
        print('Falling back to manual preprocessing...')
        print('--------------------------------------------------')
        
        
        # Fallback: Manual bad channel detection and standard referencing
        data = raw.get_data()
        ch_names = raw.ch_names
        
        # Define threshold factors
        var_thresh_factor = 0.01  # Channels with variance < 10% of median are bad
        corr_thresh_factor = 0.3  # Channels with correlation < 30% of median are bad
        
        # --- 1. Variance-based detection ---
        variances = np.var(data, axis=1)
        median_var = np.median(variances)
        variance_threshold = var_thresh_factor * median_var
        bad_var = [ch_names[i] for i, v in enumerate(variances) if v < variance_threshold]

        # --- 2. Correlation-based detection ---
        corr_matrix = np.corrcoef(data)
        mean_corr = np.mean(np.abs(corr_matrix), axis=1)  # abs avoids sign issues
        median_corr = np.median(mean_corr)
        corr_threshold = corr_thresh_factor * median_corr
        bad_corr = [ch_names[i] for i, c in enumerate(mean_corr) if c < corr_threshold]
        
        # Combine both methods
        bad_channels = list(set(bad_var + bad_corr))
        
        # Mark bad channels
        raw.info['bads'] = bad_channels
        print(f'Detected {len(bad_channels)} bad channels: {bad_channels}')
        
        # Standard average reference (instead of robust reference)
        raw.set_eeg_reference('average', projection=True)
        raw.apply_proj()
        
        # Interpolate bad channels (only if digitization info is available)
        if len(bad_channels) > 0:
            try:
                raw.interpolate_bads()
            except RuntimeError as e:
                if "digitization" in str(e).lower() or "headshape" in str(e).lower():
                    print(f'Cannot interpolate bad channels (no digitization info): {e}')
                    print('Dropping bad channels instead...')
                    # Drop bad channels instead of interpolating
                    raw.drop_channels(bad_channels)
                else:
                    raise_error(e, f"Unexpected interpolation error in fallback for {filename}")
            # raw.preload
    # Remove bad windows using autoreject
    # raw.load_data()
    epochs = mne.make_fixed_length_epochs(raw, duration=2, overlap=0.5, preload=True)
    
    # Try AutoReject first
    try:
        ar = AutoReject()
        epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
        print(f'[{filename}] AutoReject successfully applied')
    except Exception as e:
        print('--------------------------------------------------')
        print(f'Error in AutoReject: {e} for file: {filename}')
        print('Falling back to manual epoch rejection...')
        print('--------------------------------------------------')
        
        # Fallback: Manual epoch rejection using variance and amplitude criteria
        epochs_clean = manual_epoch_rejection(epochs, filename)
    # Assuming epochs_clean is an instance of mne.Epochs
    epochs_data = epochs_clean.get_data()  # Shape: (n_epochs, n_channels, n_times)
    
    # Reshape the data to (n_channels, n_times_total)
    n_epochs, n_channels, n_times = epochs_data.shape
    reshaped_data = epochs_data.transpose(1, 0, 2).reshape(n_channels, -1)
    # Create a new RawArray object
    info = epochs_clean.info  # Use the info from the epochs
    raw = mne.io.RawArray(reshaped_data, info)
    return raw

def analyze_eeg_data(raw,is_prod,filename):
    raw = clean_mne_raw(raw,filename)
    if raw is None:
        return None, None
    plot_not_prod(raw,is_prod,'AutoReject5')
    # save spec_data to pickle in pickles/project_name/filename
    with open(f'pickles/{project_name}/{filename}.pkl', 'wb') as f:
        pickle.dump(raw, f)
    metadata, conn_df = eeg_data_to_features(raw)
    return metadata, conn_df



def process_file(row,filename,is_prod):
    print(f'Processing {row["file_name"]}')
    file_name = row['file_name']
    # read file
    metadata, raw = read_edf_mne(row['file_path'])
    if raw is None or metadata is None:
        return None
    metadata.update(row)
    # make folder if not exist
    os.makedirs(temp_dir, exist_ok=True)
    # if file temps/{row['file_name']}.parquet exists
    if f'{file_name}.parquet' in os.listdir(temp_dir):
        print(f'{file_name}.parquet already exists')
        return 
    if metadata:
        # Check the duration of the recording
        duration_s = raw.times[-1]  # Convert duration to milliseconds
        # duration_skip = 3 * 60  # Skip recordings less than 10 minutes
        # if duration_s < duration_skip:
        #     return
        # Split the data into segments
        max_duration_s = 60 * 60  # 60 minutes in seconds
        
        if duration_s > max_duration_s:
            eeg_metadata = None
            start_i = 0
            # while eeg_metadata is None:
                # if max_duration_s == 0:
                #     pd.DataFrame().to_parquet(f'{temp_dir}/{file_name}_{max_duration_s}_{start_i}.parquet', index=False)
                #     start_i += 1
                #     max_duration_s = 60 * 60
                #     print(f'Error processing {file_name}_{max_duration_s}_{start_i}, skipping...')
                # Split the data into 60-minute segments
            n_segments = int(np.ceil(duration_s / max_duration_s))
            for i in range(start_i, n_segments):
                segment_filename = f'{file_name}_{max_duration_s}_{i + 1}.parquet'
                if os.path.exists(f'{temp_dir}/{segment_filename}'):
                    continue
                start = i * max_duration_s  # Start time in seconds
                stop = min((i + 1) * max_duration_s, raw.times[-1])  # Stop time in seconds
                raw_segment = raw.copy().crop(tmin=start, tmax=stop)
                eeg_metadata, conn_df = analyze_eeg_data(raw_segment, is_prod, segment_filename.replace(".parquet", ""))
                if eeg_metadata is None:
                    # Reduce max_duration_s by 10 minutes but not below 0
                    continue
                    # print(f'Error processing {file_name}_{max_duration_s}_{start_i}, retrying with max_duration_s={max_duration_s-600}...')
                    # max_duration_s = max(max_duration_s - 5 * 60, 0)
                    # break  # Exit the for loop to recalculate segments with new max_duration_s
                # Save connectivity DataFrame (conn_df) to its own Parquet folder
                try:

                    if conn_df is not None:
                        # If conn_df is a DataFrame, save directly; if it's a dict, convert to DataFrame first
                        if isinstance(conn_df, pd.DataFrame):
                            pass
                        else:
                            try:
                                pd.DataFrame(conn_df)
                            except Exception:
                                pass
                        # if out_df is not None and not out_df.empty:
                        #     temp_conn_dir = f"{temp_dir}_conn_df"
                        #     os.makedirs(temp_conn_dir, exist_ok=True)
                        #     out_path = os.path.join(temp_conn_dir, f"{segment_filename.replace('.parquet','')}_conn.parquet")
                        #     out_df.to_parquet(out_path, index=True)
                except Exception as e:
                    print(f"Failed to save conn_df for {segment_filename}: {e}")
                # Update and write segment metadata
                metadata.update(eeg_metadata)
                # metadata annotation to str
                metadata = {k: str(v) if not isinstance(v, (int, float, bool)) else v for k, v in metadata.items()}
                segment_metadata = metadata.copy()
                segment_metadata['segment'] = i + 1
                segment_metadata['start_time'] = start
                segment_metadata['end_time'] = stop
                segment_metadata['duration_sec'] = stop - start
                segment_metadata['duration_min'] = (stop - start) / 60
    
                # Write segment metadata to Parquet
                df_segment = pd.DataFrame([segment_metadata])
                df_segment.to_parquet(f'{temp_dir}/{segment_filename}', index=False)
        else:
            # load file name parquet
            try:
                df_parquet = pd.read_parquet(filename)
                # if file name in parquet
                if row['file_name'] in df_parquet['file_name'].values:
                    return 
            except Exception:
                pass
            eeg_metadata, conn_df = analyze_eeg_data(raw,is_prod,row["file_name"])
            if eeg_metadata is None:
                # save empty parquet file
                # pd.DataFrame().to_parquet(f'{temp_dir}/{row["file_name"]}.parquet', index=False)
                print(f'Error processing {row["file_name"]}')
                return None
            else:
                metadata.update(eeg_metadata)
                # metadata annotation to str
                metadata['annotations'] = str(metadata['annotations'])
                # Write metadata to Parquet
                df = pd.DataFrame([metadata])
                df.to_parquet(f'{temp_dir}/{row["file_name"]}.parquet', index=False)
                # Save connectivity DataFrame (conn_df) to its own Parquet folder
                # try:
                #     temp_conn_dir = f"{temp_dir}_conn_df"
                #     os.makedirs(temp_conn_dir, exist_ok=True)
                #     if conn_df is not None:
                #         if isinstance(conn_df, pd.DataFrame):
                #             out_df = conn_df
                #         else:
                #             try:
                #                 out_df = pd.DataFrame(conn_df)
                #             except Exception:
                #                 out_df = None
                #         if out_df is not None and not out_df.empty:
                #             out_path = os.path.join(temp_conn_dir, f"{row['file_name']}_conn.parquet")
                #             out_df.to_parquet(out_path, index=True)
                # except Exception as e:
                #     print(f"Failed to save conn_df for {row['file_name']}: {e}")
        # return metadata
    # return metadata

def find_eeg_files(directory):
        eeg_files = glob.glob(os.path.join(directory, '**', '*.EEG'), recursive=True)
        return eeg_files

def convert_and_remove_eeg(eeg_files):
    min_size_mb = 5  # Minimum file size in MB
    min_size_bytes = min_size_mb * 1024 * 1024  # Convert to bytes
    
    for eeg_file_path in eeg_files:
        # Check file size before processing
        try:
            file_size = os.path.getsize(eeg_file_path)
            file_size_mb = file_size / (1024 * 1024)
            
            if file_size < min_size_bytes:
                print(f"⏭️  Skipping {eeg_file_path} - Size: {file_size_mb:.2f}MB (less than {min_size_mb}MB)")
                continue
            else:
                print(f"📁 Processing {eeg_file_path} - Size: {file_size_mb:.2f}MB")
        except Exception as e:
            print(f"⚠️  Could not get file size for {eeg_file_path}: {e}")
            continue
        
        base, _ = os.path.splitext(eeg_file_path)
        edf_file_path = base + '.edf'
        try:
            raw = mne.io.read_raw_nihon(eeg_file_path, preload=True)
            raw.export(edf_file_path, fmt='edf', overwrite=True)
            print(f"Converted {eeg_file_path} to {edf_file_path}")
            os.remove(eeg_file_path)
            print(f"Deleted original EEG file: {eeg_file_path}")
        except Exception as e:
            print(f"Failed to convert {eeg_file_path}: {e}")

def eeg_edf_cleaning_pipeline():
    # Find and convert .EEG files to .edf, then remove .EEG files
    eeg_files = find_eeg_files(directory)
    convert_and_remove_eeg(eeg_files)
    df = list_files_and_find_duplicates(directory)
    # remove duplicates subset file_name
    df.drop_duplicates(subset='file_name', inplace=True)
    # Set multiprocessing flag
    filename = f'{project_name}.parquet'
    if use_multiprocessing:
        print(f'Processing {len(df)} files in parallel with Ray...')
        ray.init(ignore_reinit_error=True, num_cpus=15)
        # Process files in parallel using Ray
        @ray.remote
        def process_file_ray(row):
            return process_file(row, filename, is_prod)
        
        futures = [process_file_ray.remote(row) for _, row in df.iterrows()]
        for _ in tqdm(ray.get(futures), total=len(futures)):
            pass
        ray.shutdown()
    else:
        print(f'Processing {len(df)} files sequentially...')
        # Process files sequentially
        metadata_list = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            file_name = row['file_name']
            metadata_list.append(process_file(row, filename, is_prod))
    if project_name == 'Controls':
        # Combine all the temporary Parquet files into a single Parquet file
        filename = f'{cases_project_name}_controls.parquet'
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
                df = pd.read_parquet(os.path.join(temp_dir, file))
                df['parquet_file_name'] = file
                df_list.append(df)
            except Exception as e:
                print(f"Skipping file {file}: {e}")
        if df_list:
            df_batch = pd.concat(df_list)
            # move col parquet_file_name to first
            df_batch = pd.concat([df_batch['parquet_file_name'], df_batch.drop(columns=['parquet_file_name'])], axis=1)
            # For Parquet, we need to collect all data and write at once
            # Append the DataFrame to the list for final write
            if i == 0:
                final_df = df_batch.copy()
            else:
                final_df = pd.concat([final_df, df_batch], ignore_index=True)
    
    # Write the final combined DataFrame to Parquet
    if 'final_df' in locals():
        final_df.to_parquet(filename, index=False)
    

    # process_patients_random6(edf_root=f"parquet_HEP/{cases_project_name}_HEP",)

if __name__ == "__main__":
    eeg_edf_cleaning_pipeline()