import mne 
import os 
import pandas as pd
import pickle


# read pickles/EDF/0345-002.edf_600_2.pkl
with open('pickles/EDF/0345-010.edf_600_1.pkl', 'rb') as f:
    data = pickle.load(f)

pass
# get raw data from mne raw array
raw_data = data
data_raw = data.get_data()
channels = data.ch_names