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
from collections import defaultdict

directory = '/Users/nircafri/Desktop/Scripts/Nir/cobrad/west_nile_virus'
prject_name = directory.split('/')[-1]
file_size_map = defaultdict(list)

for root, _, files in os.walk(directory):
    for file in files:
        file_path = os.path.join(root, file)
        file_size = os.path.getsize(file_path)
        file_size_map[file_size].append(file_path)
data = []
for size, files in file_size_map.items():
    for file in files:
        data.append({"file_path": file, "size": size})
df = pd.DataFrame(data)

#%% 
# file name to path split [-1]
df['file_name'] = df['file_path'].str.split('/').str[-1]
# .EEG files
df_eegs_format = df[df['file_path'].str.contains('.EEG')]
df_eegs_format['patient_id'] = df_eegs_format['file_path'].str.split('/').str[-2].astype(int)
df_eegs_format['patient_id'].tolist()

def convert_eeg_to_edf(df_eegs_format):
    for index, row in df_eegs_format.iterrows():
        eeg_file_path = row['file_path']
        edf_file_path = eeg_file_path.replace('.EEG', '.edf')
        # os system run EEG2EDF_Matlab/EEG_to_edf('/EEG_data/', 'edf_data/')  
        os.system(f'EEG2EDF_Matlab/EEG_to_edf("{eeg_file_path}", "{edf_file_path}")')
        # Load the .eeg file using mne
        raw = mne.io.read_raw_eeglab(edf_file_path, preload=True)
        
        # Save the data as .edf
        raw.export(edf_file_path, fmt='edf')
        print(f"Converted {eeg_file_path} to {edf_file_path}")

# Call the function
convert_eeg_to_edf(df_eegs_format)