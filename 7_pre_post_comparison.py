import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import os
import re
from utils.eeg_utils import eeg_channels, rename_channels, read_edf_mne, load_pre_post_data
import mne
from scipy import stats, signal
import dabest
import warnings
import io
from datetime import datetime
warnings.filterwarnings('ignore')

# Set plotting styles
sns.set_context('talk')
sns.set_style('white')
plt.rcParams['axes.grid'] = True
plt.rc('xtick', bottom=True)
plt.rc('ytick', left=True)
plt.rc('font', family='serif')
plt.rc('text', usetex=False)
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
plt.rc('axes', labelsize=11)
plt.rc('legend', handlelength=4.0)
plt.rc('axes', titlesize=12)

# Define montage
montage = mne.channels.make_standard_montage('standard_1020')


def get_eeg_features(df):
    """Extract EEG features from the dataframe."""
    # Get columns that contain EEG data (typically start with 'overall_' or contain frequency band names)
    eeg_columns = []
    for col in df.columns:
        if any(band in col.lower() for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']) or 'overall_' in col:
            eeg_columns.append(col)
    return eeg_columns

def create_dabest_plot(pre_data, post_data, feature, title_suffix=""):
    """Create dabest estimation plot for paired pre vs post comparison."""
    
    # Ensure we have the same subjects in both datasets
    pre_subjects = set(pre_data['Subject_ID'].unique()) if len(pre_data) > 0 else set()
    post_subjects = set(post_data['Subject_ID'].unique()) if len(post_data) > 0 else set()
    common_subjects = pre_subjects.intersection(post_subjects)
    
    if len(common_subjects) < 2:
        return None, None, None
    
    # Prepare paired data for dabest
    paired_data = []
    for subject in common_subjects:
        pre_val = pre_data[pre_data['Subject_ID'] == subject][feature].iloc[0] if len(pre_data[pre_data['Subject_ID'] == subject]) > 0 else np.nan
        post_val = post_data[post_data['Subject_ID'] == subject][feature].iloc[0] if len(post_data[post_data['Subject_ID'] == subject]) > 0 else np.nan
        
        if not np.isnan(pre_val) and not np.isnan(post_val):
            paired_data.append({
                'Subject_ID': subject,
                'PRE': pre_val,
                'POST': post_val
            })
    
    if len(paired_data) < 2:
        return None, None, None
    
    df_paired = pd.DataFrame(paired_data)

    # Convert to long format for dabest
    df_long = df_paired.melt(id_vars='Subject_ID', value_vars=['PRE', 'POST'],
                             var_name='Timepoint', value_name=feature)

    # Create dabest object for paired analysis
    dabest_obj = dabest.load(df_long, x='Timepoint', y=feature, id_col='Subject_ID',
                             paired='sequential', idx=['PRE', 'POST'])
    # Get statistical results
    results = dabest_obj.mean_diff.results
    p_value = results['pvalue_wilcoxon'].values[0] if 'pvalue_wilcoxon' in results else np.nan
    
    # Get Cohen's d for paired data
    try:
        cohen_d = dabest_obj.cohens_d.results['difference'].values[0]
    except Exception:
        cohen_d = np.nan
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    dabest_obj.mean_diff.plot(ax=ax, raw_marker_size=6, show_pairs=True)
    
    # Add title with statistics
    plt.title(f"{feature}{title_suffix}\n"
              f"Paired Analysis (N={len(paired_data)})\n"
              f"p = {p_value:.3e}, Cohen's d = {cohen_d:.3f}")
    
    return fig, p_value, cohen_d

def extract_channel_values_for_feature(df, feature_name):
    """
    Extract channel-specific values for a given feature name.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing EEG features
    feature_name : str
        Feature name to extract (e.g., 'mean', 'median', 'std', 'pswe_events_per_minute', 
        'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power')
    
    Returns
    -------
    dict : Dictionary mapping channel names to values
    """
    channel_values = {}
    
    # Define feature patterns to search for
    # Check if feature is a frequency band name
    band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    is_band = any(band in feature_name.lower() for band in band_names)
    
    # Normalize feature name for matching
    feature_lower = feature_name.lower().strip()
    
    # Pattern matching for different feature types
    if is_band:
        # For frequency bands: e.g., 'delta_power_EEG Fp1' or 'delta_power_EEG Fp1'
        # Try both with and without '_power'
        if '_power' in feature_lower:
            pattern = f"{feature_lower}_EEG"
        else:
            pattern = f"{feature_lower}_power_EEG"
    elif 'pswe_events_per_minute' in feature_lower or 'pswe events per min' in feature_lower or 'pswe_events' in feature_lower:
        # For PSWE: e.g., 'pswe_events_per_minute_EEG Fp1'
        pattern = "pswe_events_per_minute_EEG"
    elif 'mean' in feature_lower and 'median' not in feature_lower:
        # For mean: e.g., 'mean_EEG Fp1'
        pattern = "mean_EEG"
    elif 'median' in feature_lower:
        # For median: e.g., 'median_EEG Fp1'
        pattern = "median_EEG"
    elif 'std' in feature_lower:
        # For std: e.g., 'std_EEG Fp1'
        pattern = "std_EEG"
    else:
        # Try generic pattern: e.g., '{feature}_EEG {ch}'
        pattern = f"{feature_lower}_EEG"
    
    # Search for columns matching the pattern
    for col in df.columns:
        col_lower = col.lower()
        if pattern.lower() in col_lower:
            # Extract channel name (should be after 'EEG')
            # Handle both 'EEG ' and 'EEG' formats
            if 'EEG ' in col:
                parts = col.split('EEG ')
            elif 'EEG' in col:
                parts = col.split('EEG')
            else:
                continue
            
            if len(parts) > 1:
                # Extract potential channel name (everything after 'EEG' or 'EEG ')
                potential_ch = parts[-1].strip()
                
                # Check if extracted name is exactly a valid channel
                if potential_ch in eeg_channels:
                    values = df[col].dropna()
                    if len(values) > 0:
                        channel_values[potential_ch] = values.mean()
                else:
                    # Try to find channel name as a word boundary match
                    # Check each channel to see if it appears as a complete word in the column
                    for ch in eeg_channels:
                        # Match channel name with word boundaries (not substring)
                        # e.g., "Fp1" should match "mean_EEG Fp1" but not "mean_EEG Fp10"
                        pattern = r'\b' + re.escape(ch) + r'\b'
                        if re.search(pattern, col):
                            values = df[col].dropna()
                            if len(values) > 0:
                                channel_values[ch] = values.mean()
                            break
    
    return channel_values

def create_topomap_comparison(pre_data, post_data, feature, montage, p_value=None):
    """Create topomap comparison for pre vs post groups."""
    
    # For overall features, try to extract channel-specific data
    if 'overall_' in feature:
        # Try to extract the base feature name (e.g., 'mean' from 'overall_mean')
        base_feature = feature.replace('overall_', '').strip()
        pre_channel_vals = extract_channel_values_for_feature(pre_data, base_feature)
        post_channel_vals = extract_channel_values_for_feature(post_data, base_feature)
        
        if len(pre_channel_vals) < 3 and len(post_channel_vals) < 3:
            return None, None, None
    else:
        # Extract channel data for the feature
        pre_channel_vals = extract_channel_values_for_feature(pre_data, feature)
        post_channel_vals = extract_channel_values_for_feature(post_data, feature)
        
        if len(pre_channel_vals) < 3 and len(post_channel_vals) < 3:
            return None, None, None
    
    # Get all unique channels from both datasets
    all_channels = set(pre_channel_vals.keys()) | set(post_channel_vals.keys())
    
    if len(all_channels) < 3:
        return None, None, None
    
    # Create ordered lists of values for all channels
    ch_names = sorted(list(all_channels))
    pre_vals = [pre_channel_vals.get(ch, np.nan) for ch in ch_names]
    post_vals = [post_channel_vals.get(ch, np.nan) for ch in ch_names]
    
    # Handle missing channels (interpolate or use 0)
    pre_vals = [v if not np.isnan(v) else 0 for v in pre_vals]
    post_vals = [v if not np.isnan(v) else 0 for v in post_vals]
    
    # Create info object
    info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types='eeg')
    info.set_montage(montage)
    
    # Create evoked objects
    pre_evoked = mne.EvokedArray(np.array(pre_vals).reshape(-1, 1), info)
    post_evoked = mne.EvokedArray(np.array(post_vals).reshape(-1, 1), info)
    
    # Calculate difference
    diff_vals = np.array(post_vals) - np.array(pre_vals)
    diff_evoked = mne.EvokedArray(diff_vals.reshape(-1, 1), info)
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Determine common scale
    all_vals = pre_vals + post_vals
    vmin, vmax = np.min(all_vals), np.max(all_vals)
    
    # Create title with p-value if available
    if p_value is not None:
        title_suffix = f'\np = {p_value:.3e}'
        if p_value < 0.05:
            title_suffix += ' *'
    else:
        title_suffix = ''
    
    # per-group unique-subject counts (Subject_ID identifies patients here)
    n_pre = pre_data['Subject_ID'].nunique() if 'Subject_ID' in pre_data.columns else len(pre_data)
    n_post = post_data['Subject_ID'].nunique() if 'Subject_ID' in post_data.columns else len(post_data)

    # PRE topomap
    im1, _ = mne.viz.plot_topomap(pre_evoked.data[:, 0], pre_evoked.info,
                                  axes=axes[0], show=False, vlim=(vmin, vmax))
    axes[0].set_title(f'PRE (n={n_pre})')

    # POST topomap
    im2, _ = mne.viz.plot_topomap(post_evoked.data[:, 0], post_evoked.info,
                                  axes=axes[1], show=False, vlim=(vmin, vmax))
    axes[1].set_title(f'POST (n={n_post})')
    
    # Difference topomap
    im3, _ = mne.viz.plot_topomap(diff_evoked.data[:, 0], diff_evoked.info, 
                                  axes=axes[2], show=False)
    axes[2].set_title('POST - PRE')
    
    # Add overall title with p-value
    fig.suptitle(f'{feature}{title_suffix}', fontsize=14, fontweight='bold', y=1.02)
    
    # Add colorbars
    plt.colorbar(im1, ax=axes[0])
    plt.colorbar(im2, ax=axes[1])
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    return fig, pre_evoked, post_evoked

def create_topomaps_for_all_features(pre_df, post_df, eeg_features, montage, pre_label, post_label, 
                                     pairing_info=None, is_mixed=False, significant_only=True):
    """
    Create topomaps for all EEG features that have channel-specific data.
    Optionally filter to only significant features.
    
    Parameters
    ----------
    pre_df : pd.DataFrame
        PRE group data
    post_df : pd.DataFrame
        POST group data
    eeg_features : list
        List of EEG feature names
    montage : mne.channels.DigMontage
        MNE montage object
    pre_label : str
        Label for PRE group
    post_label : str
        Label for POST group
    pairing_info : dict or set, optional
        Pairing information for statistical tests
    is_mixed : bool
        Whether analysis is mixed paired/unpaired
    significant_only : bool
        If True, only show significant features (p < 0.05)
    
    Returns
    -------
    dict : Dictionary mapping feature names to (figure, p_value) tuples
    """
    topomap_figs = {}
    
    # Define feature types to create topomaps for
    feature_types = ['mean', 'median', 'std', 'pswe_events_per_minute', 
                     'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power']
    
    # Process each feature type
    for feature_type in feature_types:
        # Check if this feature type exists in the data
        pre_vals = extract_channel_values_for_feature(pre_df, feature_type)
        post_vals = extract_channel_values_for_feature(post_df, feature_type)
        
        if len(pre_vals) >= 3 or len(post_vals) >= 3:
            # Calculate p-value for this feature
            # First, try to find a matching column that contains this feature type
            p_value = None
            if pairing_info is not None:
                # Look for a column that matches this feature type
                matching_col = None
                for col in pre_df.columns:
                    if feature_type.lower() in col.lower() and col in post_df.columns:
                        matching_col = col
                        break
                
                if matching_col:
                    try:
                        result = calculate_effect_size_and_power(pre_df, post_df, matching_col, pairing_info)
                        if result[0] is not None:
                            p_value = result[0]
                    except (KeyError, Exception):
                        # If calculation fails, continue without p-value
                        pass
            
            # Filter for significant features if requested
            if significant_only and (p_value is None or p_value >= 0.05):
                continue
            
            # Create topomap for this feature
            fig, _, _ = create_topomap_comparison(pre_df, post_df, feature_type, montage, p_value=p_value)
            if fig is not None:
                topomap_figs[feature_type] = (fig, p_value)
    
    # Also process individual features from eeg_features list
    for feature in eeg_features:
        # Skip if we already processed it as a feature type
        if feature in topomap_figs:
            continue
        
        # Check if feature has channel-specific data
        pre_vals = extract_channel_values_for_feature(pre_df, feature)
        post_vals = extract_channel_values_for_feature(post_df, feature)
        
        if len(pre_vals) >= 3 or len(post_vals) >= 3:
            # Calculate p-value for this feature
            p_value = None
            if pairing_info is not None:
                result = calculate_effect_size_and_power(pre_df, post_df, feature, pairing_info)
                if result[0] is not None:
                    p_value = result[0]
            
            # Filter for significant features if requested
            if significant_only and (p_value is None or p_value >= 0.05):
                continue
            
            fig, _, _ = create_topomap_comparison(pre_df, post_df, feature, montage, p_value=p_value)
            if fig is not None:
                topomap_figs[feature] = (fig, p_value)
    
    return topomap_figs

def create_spectrograms(pre_df, post_df, dataset_name, pre_label, post_label):
    """
    Create spectrograms (Hz vs time) for PRE and POST groups.
    For Pre-Post_XCOPRI dataset, averages spectrograms across all patients with both PRE and POST files,
    grouped by brain regions (Frontal, Temporal, Central, Occipital).
    For other datasets, loads representative EDF files and creates spectrograms.
    
    Parameters
    ----------
    pre_df : pd.DataFrame
        PRE group data
    post_df : pd.DataFrame
        POST group data
    dataset_name : str
        Name of the dataset
    pre_label : str
        Label for PRE group
    post_label : str
        Label for POST group
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure with averaged spectrograms by region (4 rows × 2 columns: regions × PRE/POST) 
        for Pre-Post_XCOPRI, or two spectrograms side by side for other datasets
    """
    try:
        # Find EDF files for PRE and POST
        edf_dir = f"EDF_Format/{dataset_name}"
        if not os.path.exists(edf_dir):
            # Try alternative path
            edf_dir = dataset_name
            if not os.path.exists(edf_dir):
                return None
        
        # Special handling for Pre-Post_XCOPRI: find all patients with both PRE and POST pickle files
        if dataset_name == "Pre-Post_XCOPRI":
            # Find all pickle files
            pickle_dir = "pickles/Pre-Post_XCOPRI"
            if not os.path.exists(pickle_dir):
                return None
            
            # Parse pickle files to find patients with both PRE and POST
            pickle_files = [f for f in os.listdir(pickle_dir) if f.endswith('.pkl')]
            
            # Group by patient ID and PRE/POST status
            patient_files = {}  # {patient_id: {'PRE': file, 'POST': file}}
            
            for pickle_file in pickle_files:
                # Parse filename: e.g., "01_PRE_FA0014CD.edf.pkl" or "01_POST_FA0014Q2.edf.pkl"
                parts = pickle_file.replace('.edf.pkl', '').split('_')
                if len(parts) >= 2:
                    patient_id = parts[0]
                    status = parts[1].upper()  # PRE or POST
                    
                    if patient_id not in patient_files:
                        patient_files[patient_id] = {}
                    
                    if status in ['PRE', 'POST']:
                        patient_files[patient_id][status] = pickle_file
            
            # Only keep patients with both PRE and POST
            patient_dirs = []
            for patient_id, files in patient_files.items():
                if 'PRE' in files and 'POST' in files:
                    patient_dirs.append({
                        'patient_id': patient_id,
                        'pre_file': os.path.join(pickle_dir, files['PRE']),
                        'post_file': os.path.join(pickle_dir, files['POST'])
                    })
            
            if len(patient_dirs) == 0:
                return None
            
            # Define brain regions
            brain_regions = {
                'F': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz'],  # Frontal
                'T': ['T3', 'T4', 'T5', 'T6', 'T1', 'T2'],  # Temporal
                'C': ['C3', 'C4', 'Cz'],  # Central
                'O': ['O1', 'O2', 'P3', 'P4', 'Pz']  # Occipital
            }
            
            # Initialize progress bar
            total_patients = len(patient_dirs)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # FIRST PASS: Load all raw files and collect data for global z-score calculation
            status_text.text('Loading all files to calculate global statistics...')
            pre_data_by_region = {region: [] for region in brain_regions.keys()}
            post_data_by_region = {region: [] for region in brain_regions.keys()}
            
            # Fix for MNE pickle loading
            import sys
            import pickle
            sys.modules['mne.io.array.array'] = mne.io.array
            
            for patient_idx, patient_info in enumerate(patient_dirs):
                patient_id = patient_info['patient_id']
                pre_file = patient_info['pre_file']
                post_file = patient_info['post_file']
                
                # Update progress bar
                progress = (patient_idx + 1) / (total_patients * 2)  # Two passes
                progress_bar.progress(progress)
                status_text.text(f'Loading patient {patient_id} ({patient_idx + 1}/{total_patients})...')
                
                # Load PRE pickle file
                try:
                    with open(pre_file, 'rb') as f:
                        pre_raw = pickle.load(f)
                    
                    pre_raw = rename_channels(pre_raw)
                    ch_names = pre_raw.ch_names
                    
                    # Process each region
                    for region, region_chs in brain_regions.items():
                        available_chs = [ch for ch in ch_names if ch in region_chs]
                        if len(available_chs) == 0:
                            continue
                        
                        try:
                            pre_raw_region = pre_raw.copy().pick_channels(available_chs)
                            total_duration = pre_raw_region.times[-1]
                            segment_duration = min(300, total_duration)
                            if total_duration > segment_duration:
                                start_time = (total_duration - segment_duration) / 2
                                end_time = start_time + segment_duration
                                pre_raw_crop = pre_raw_region.copy().crop(tmin=start_time, tmax=end_time)
                            else:
                                pre_raw_crop = pre_raw_region.copy()
                            
                            data = pre_raw_crop.get_data()
                            data_avg = np.mean(data, axis=0)  # Average across channels
                            pre_data_by_region[region].append(data_avg)
                        except Exception:
                            continue
                except Exception:
                    continue
                
                # Load POST pickle file
                try:
                    with open(post_file, 'rb') as f:
                        post_raw = pickle.load(f)
                    
                    post_raw = rename_channels(post_raw)
                    ch_names = post_raw.ch_names
                    
                    # Process each region
                    for region, region_chs in brain_regions.items():
                        available_chs = [ch for ch in ch_names if ch in region_chs]
                        if len(available_chs) == 0:
                            continue
                        
                        try:
                            post_raw_region = post_raw.copy().pick_channels(available_chs)
                            total_duration = post_raw_region.times[-1]
                            segment_duration = min(300, total_duration)
                            if total_duration > segment_duration:
                                start_time = (total_duration - segment_duration) / 2
                                end_time = start_time + segment_duration
                                post_raw_crop = post_raw_region.copy().crop(tmin=start_time, tmax=end_time)
                            else:
                                post_raw_crop = post_raw_region.copy()
                            
                            data = post_raw_crop.get_data()
                            data_avg = np.mean(data, axis=0)  # Average across channels
                            post_data_by_region[region].append(data_avg)
                        except Exception:
                            continue
                except Exception:
                    continue
            
            # Calculate global mean and std for each region (combining PRE and POST)
            global_stats_by_region = {}
            for region in brain_regions.keys():
                all_data = pre_data_by_region[region] + post_data_by_region[region]
                if len(all_data) > 0:
                    # Concatenate all data for this region
                    all_data_concat = np.concatenate(all_data)
                    global_mean = np.nanmean(all_data_concat)
                    global_std = np.nanstd(all_data_concat)
                    global_stats_by_region[region] = {'mean': global_mean, 'std': global_std}
            
            # SECOND PASS: Process each patient with z-scoring
            status_text.text('Computing spectrograms with z-scoring...')
            pre_spectrograms_by_region = {region: [] for region in brain_regions.keys()}
            post_spectrograms_by_region = {region: [] for region in brain_regions.keys()}
            common_freqs = None
            common_times = None
            
            for patient_idx, patient_info in enumerate(patient_dirs):
                patient_id = patient_info['patient_id']
                pre_file = patient_info['pre_file']
                post_file = patient_info['post_file']
                
                # Update progress bar
                progress = (total_patients + patient_idx + 1) / (total_patients * 2)
                progress_bar.progress(progress)
                status_text.text(f'Processing patient {patient_id} ({patient_idx + 1}/{total_patients})...')
                
                # Load and process PRE pickle file with z-scoring
                try:
                    with open(pre_file, 'rb') as f:
                        pre_raw = pickle.load(f)
                    
                    pre_raw = rename_channels(pre_raw)
                    ch_names = pre_raw.ch_names
                    
                    # Process each region
                    for region, region_chs in brain_regions.items():
                        if region not in global_stats_by_region:
                            continue
                        
                        available_chs = [ch for ch in ch_names if ch in region_chs]
                        if len(available_chs) == 0:
                            continue
                        
                        try:
                            pre_raw_region = pre_raw.copy().pick_channels(available_chs)
                            total_duration = pre_raw_region.times[-1]
                            segment_duration = min(300, total_duration)
                            if total_duration > segment_duration:
                                start_time = (total_duration - segment_duration) / 2
                                end_time = start_time + segment_duration
                                pre_raw_crop = pre_raw_region.copy().crop(tmin=start_time, tmax=end_time)
                            else:
                                pre_raw_crop = pre_raw_region.copy()
                            
                            # Get data and z-score using global statistics
                            data = pre_raw_crop.get_data()
                            data_avg = np.mean(data, axis=0)  # Average across channels
                            # Z-score using global statistics
                            global_mean = global_stats_by_region[region]['mean']
                            global_std = global_stats_by_region[region]['std']
                            if global_std > 0:
                                data_avg_z = (data_avg - global_mean) / global_std
                            else:
                                data_avg_z = data_avg
                            
                            sfreq = pre_raw_crop.info['sfreq']
                            
                            nperseg = int(2 * sfreq)
                            noverlap = int(nperseg * 0.5)
                            freqs, times, Sxx = signal.spectrogram(
                                data_avg_z, sfreq, nperseg=nperseg, noverlap=noverlap, window='hann'
                            )
                            
                            freq_mask = (freqs >= 1) & (freqs <= 50)
                            freqs_filtered = freqs[freq_mask]
                            Sxx_filtered = Sxx[freq_mask, :]
                            
                            if common_freqs is None:
                                common_freqs = freqs_filtered
                                common_times = times
                                pre_spectrograms_by_region[region].append(Sxx_filtered)
                            else:
                                if len(times) != len(common_times) or not np.allclose(freqs_filtered, common_freqs):
                                    from scipy.interpolate import RectBivariateSpline
                                    f_interp = RectBivariateSpline(freqs_filtered, times, Sxx_filtered, kx=1, ky=1)
                                    Sxx_interp = f_interp(common_freqs, common_times)
                                    pre_spectrograms_by_region[region].append(Sxx_interp)
                                else:
                                    pre_spectrograms_by_region[region].append(Sxx_filtered)
                        except Exception:
                            continue
                except Exception:
                    continue
                
                # Load and process POST pickle file with z-scoring
                try:
                    with open(post_file, 'rb') as f:
                        post_raw = pickle.load(f)
                    
                    post_raw = rename_channels(post_raw)
                    ch_names = post_raw.ch_names
                    
                    # Process each region
                    for region, region_chs in brain_regions.items():
                        if region not in global_stats_by_region:
                            continue
                        
                        available_chs = [ch for ch in ch_names if ch in region_chs]
                        if len(available_chs) == 0:
                            continue
                        
                        try:
                            post_raw_region = post_raw.copy().pick_channels(available_chs)
                            total_duration = post_raw_region.times[-1]
                            segment_duration = min(300, total_duration)
                            if total_duration > segment_duration:
                                start_time = (total_duration - segment_duration) / 2
                                end_time = start_time + segment_duration
                                post_raw_crop = post_raw_region.copy().crop(tmin=start_time, tmax=end_time)
                            else:
                                post_raw_crop = post_raw_region.copy()
                            
                            # Get data and z-score using global statistics
                            data = post_raw_crop.get_data()
                            data_avg = np.mean(data, axis=0)  # Average across channels
                            # Z-score using global statistics
                            global_mean = global_stats_by_region[region]['mean']
                            global_std = global_stats_by_region[region]['std']
                            if global_std > 0:
                                data_avg_z = (data_avg - global_mean) / global_std
                            else:
                                data_avg_z = data_avg
                            
                            sfreq = post_raw_crop.info['sfreq']
                            
                            nperseg = int(2 * sfreq)
                            noverlap = int(nperseg * 0.5)
                            freqs, times, Sxx = signal.spectrogram(
                                data_avg_z, sfreq, nperseg=nperseg, noverlap=noverlap, window='hann'
                            )
                            
                            freq_mask = (freqs >= 1) & (freqs <= 50)
                            freqs_filtered = freqs[freq_mask]
                            Sxx_filtered = Sxx[freq_mask, :]
                            
                            if len(times) != len(common_times) or not np.allclose(freqs_filtered, common_freqs):
                                from scipy.interpolate import RectBivariateSpline
                                f_interp = RectBivariateSpline(freqs_filtered, times, Sxx_filtered, kx=1, ky=1)
                                Sxx_interp = f_interp(common_freqs, common_times)
                                post_spectrograms_by_region[region].append(Sxx_interp)
                            else:
                                post_spectrograms_by_region[region].append(Sxx_filtered)
                        except Exception:
                            continue
                except Exception:
                    continue
            
            # Clear progress bar
            progress_bar.empty()
            status_text.empty()
            
            # Average spectrograms across all patients for each region
            pre_avg_by_region = {}
            post_avg_by_region = {}
            region_labels = {'F': 'Frontal', 'T': 'Temporal', 'C': 'Central', 'O': 'Occipital'}
            
            for region in brain_regions.keys():
                if len(pre_spectrograms_by_region[region]) > 0:
                    pre_avg_by_region[region] = np.nanmean(np.stack(pre_spectrograms_by_region[region], axis=0), axis=0)
                if len(post_spectrograms_by_region[region]) > 0:
                    post_avg_by_region[region] = np.nanmean(np.stack(post_spectrograms_by_region[region], axis=0), axis=0)
            
            if len(pre_avg_by_region) == 0 or len(post_avg_by_region) == 0:
                return None
            
            # Create figure with grid: 4 rows (regions) × 2 columns (PRE, POST)
            fig, axes = plt.subplots(4, 2, figsize=(16, 12))
            
            # Plot spectrograms for each region
            for idx, region in enumerate(['F', 'T', 'C', 'O']):
                if region in pre_avg_by_region and region in post_avg_by_region:
                    # PRE spectrogram
                    im1 = axes[idx, 0].pcolormesh(common_times, common_freqs, 
                                                  10 * np.log10(pre_avg_by_region[region] + 1e-10), 
                                                  cmap='jet', shading='gouraud')
                    axes[idx, 0].set_xlabel('Time (s)', fontsize=10)
                    axes[idx, 0].set_ylabel('Frequency (Hz)', fontsize=10)
                    axes[idx, 0].set_title(f'{pre_label} - {region_labels[region]} (N={len(pre_spectrograms_by_region[region])})', 
                                          fontsize=12, fontweight='bold')
                    axes[idx, 0].set_ylim([1, 50])
                    plt.colorbar(im1, ax=axes[idx, 0], label='Power (dB)')
                    
                    # POST spectrogram
                    im2 = axes[idx, 1].pcolormesh(common_times, common_freqs, 
                                                  10 * np.log10(post_avg_by_region[region] + 1e-10), 
                                                  cmap='jet', shading='gouraud')
                    axes[idx, 1].set_xlabel('Time (s)', fontsize=10)
                    axes[idx, 1].set_ylabel('Frequency (Hz)', fontsize=10)
                    axes[idx, 1].set_title(f'{post_label} - {region_labels[region]} (N={len(post_spectrograms_by_region[region])})', 
                                          fontsize=12, fontweight='bold')
                    axes[idx, 1].set_ylim([1, 50])
                    plt.colorbar(im2, ax=axes[idx, 1], label='Power (dB)')
                else:
                    # If region data not available, show message
                    axes[idx, 0].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[idx, 0].transAxes)
                    axes[idx, 0].set_title(f'{pre_label} - {region_labels[region]} - No data', fontsize=12)
                    axes[idx, 1].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[idx, 1].transAxes)
                    axes[idx, 1].set_title(f'{post_label} - {region_labels[region]} - No data', fontsize=12)
            
            plt.tight_layout()
            return fig
        
        # Original logic for other datasets: use representative files
        # Get a representative subject from each group
        pre_subjects = pre_df['Subject_ID'].unique() if len(pre_df) > 0 else []
        post_subjects = post_df['Subject_ID'].unique() if len(post_df) > 0 else []
        
        if len(pre_subjects) == 0 or len(post_subjects) == 0:
            return None
        
        # Find EDF files
        pre_edf_file = None
        post_edf_file = None
        
        # Search for EDF files matching subject IDs
        for root, dirs, files in os.walk(edf_dir):
            for file in files:
                if file.lower().endswith('.edf'):
                    file_lower = file.lower()
                    # Check if file matches a PRE subject
                    if pre_edf_file is None:
                        for subject in pre_subjects[:5]:  # Check first 5 subjects
                            if subject.lower().replace('_', '') in file_lower.replace('_', '').replace('-', ''):
                                pre_edf_file = os.path.join(root, file)
                                break
                    # Check if file matches a POST subject
                    if post_edf_file is None:
                        for subject in post_subjects[:5]:  # Check first 5 subjects
                            if subject.lower().replace('_', '') in file_lower.replace('_', '').replace('-', ''):
                                post_edf_file = os.path.join(root, file)
                                break
                    if pre_edf_file and post_edf_file:
                        break
            if pre_edf_file and post_edf_file:
                break
        
        if pre_edf_file is None or post_edf_file is None:
            return None
        
        # Load EDF files using MNE
        try:
            pre_raw = mne.io.read_raw_edf(pre_edf_file, preload=True, verbose=False)
            post_raw = mne.io.read_raw_edf(post_edf_file, preload=True, verbose=False)
        except Exception:
            # Try using the utility function
            try:
                pre_raw = read_edf_mne(pre_edf_file)
                post_raw = read_edf_mne(post_edf_file)
                if pre_raw is None or post_raw is None:
                    return None
            except Exception:
                return None
        
        # Pick a representative EEG channel (prefer central channels)
        eeg_chs = [ch for ch in pre_raw.ch_names if 'EEG' in ch.upper() or 'eeg' in ch.lower()]
        if len(eeg_chs) == 0:
            return None
        
        # Prefer central channels (Cz, C3, C4, Fz, Pz)
        preferred_chs = ['Cz', 'C3', 'C4', 'Fz', 'Pz']
        selected_ch = None
        for pref in preferred_chs:
            for ch in eeg_chs:
                if pref.upper() in ch.upper():
                    selected_ch = ch
                    break
            if selected_ch:
                break
        
        if selected_ch is None:
            selected_ch = eeg_chs[0]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Compute and plot spectrogram for PRE
        try:
            pre_raw.pick_channels([selected_ch])
            # Use shorter duration for faster computation (first 60 seconds or full duration)
            duration = min(60, pre_raw.times[-1])
            pre_raw_crop = pre_raw.copy().crop(tmax=duration)
            
            # Get data
            data = pre_raw_crop.get_data()[0]
            sfreq = pre_raw_crop.info['sfreq']
            
            # Compute spectrogram using scipy
            nperseg = int(2 * sfreq)  # 2 second windows
            noverlap = int(nperseg * 0.5)  # 50% overlap
            freqs, times, Sxx = signal.spectrogram(
                data, 
                sfreq, 
                nperseg=nperseg, 
                noverlap=noverlap,
                window='hann'
            )
            
            # Filter frequencies (1-50 Hz)
            freq_mask = (freqs >= 1) & (freqs <= 50)
            freqs_filtered = freqs[freq_mask]
            Sxx_filtered = Sxx[freq_mask, :]
            
            # Plot spectrogram
            im1 = ax1.pcolormesh(times, freqs_filtered, 10 * np.log10(Sxx_filtered + 1e-10), 
                                 cmap='jet', shading='gouraud')
            ax1.set_xlabel('Time (s)', fontsize=12)
            ax1.set_ylabel('Frequency (Hz)', fontsize=12)
            ax1.set_title(f'{pre_label} - {selected_ch}', fontsize=14, fontweight='bold')
            ax1.set_ylim([1, 50])
            plt.colorbar(im1, ax=ax1, label='Power (dB)')
        except Exception as e:
            ax1.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title(f'{pre_label} - Error', fontsize=14)
        
        # Compute and plot spectrogram for POST
        try:
            post_raw.pick_channels([selected_ch])
            # Use shorter duration for faster computation
            duration = min(60, post_raw.times[-1])
            post_raw_crop = post_raw.copy().crop(tmax=duration)
            
            # Get data
            data = post_raw_crop.get_data()[0]
            sfreq = post_raw_crop.info['sfreq']
            
            # Compute spectrogram using scipy
            nperseg = int(2 * sfreq)  # 2 second windows
            noverlap = int(nperseg * 0.5)  # 50% overlap
            freqs, times, Sxx = signal.spectrogram(
                data, 
                sfreq, 
                nperseg=nperseg, 
                noverlap=noverlap,
                window='hann'
            )
            
            # Filter frequencies (1-50 Hz)
            freq_mask = (freqs >= 1) & (freqs <= 50)
            freqs_filtered = freqs[freq_mask]
            Sxx_filtered = Sxx[freq_mask, :]
            
            # Plot spectrogram
            im2 = ax2.pcolormesh(times, freqs_filtered, 10 * np.log10(Sxx_filtered + 1e-10), 
                                cmap='jet', shading='gouraud')
            ax2.set_xlabel('Time (s)', fontsize=12)
            ax2.set_ylabel('Frequency (Hz)', fontsize=12)
            ax2.set_title(f'{post_label} - {selected_ch}', fontsize=14, fontweight='bold')
            ax2.set_ylim([1, 50])
            plt.colorbar(im2, ax=ax2, label='Power (dB)')
        except Exception as e:
            ax2.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title(f'{post_label} - Error', fontsize=14)
        
        plt.tight_layout()
        return fig
        
    except Exception:
        # Return None if anything fails
        return None

def create_simple_comparison_plot(pre_data, post_data, feature):
    """Create a simple comparison plot for overall features."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Box plot comparison
    data_for_box = []
    labels = []
    
    if len(pre_data) > 0:
        pre_vals = pre_data[feature].dropna()
        if len(pre_vals) > 0:
            data_for_box.append(pre_vals)
            labels.append('PRE')
    
    if len(post_data) > 0:
        post_vals = post_data[feature].dropna()
        if len(post_vals) > 0:
            data_for_box.append(post_vals)
            labels.append('POST')
    
    if data_for_box:
        ax1.boxplot(data_for_box, labels=labels)
        ax1.set_ylabel(feature)
        ax1.set_title(f'Box Plot: {feature}')
        ax1.grid(True, alpha=0.3)
    
    # Violin plot comparison
    if data_for_box:
        ax2.violinplot(data_for_box, positions=range(1, len(data_for_box) + 1))
        ax2.set_xticks(range(1, len(data_for_box) + 1))
        ax2.set_xticklabels(labels)
        ax2.set_ylabel(feature)
        ax2.set_title(f'Violin Plot: {feature}')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_individual_change_plot(pre_df, post_df, feature, pre_label, post_label, pairing_info=None):
    """
    Create a plot showing individual PRE/POST changes with connecting lines for each subject.
    
    Parameters
    ----------
    pre_df : pd.DataFrame
        PRE group data
    post_df : pd.DataFrame
        POST group data
    feature : str
        Feature name
    pre_label : str
        Label for PRE group
    post_label : str
        Label for POST group
    pairing_info : dict or set, optional
        Pairing information
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure object if successful, None otherwise
    """
    # Get paired data
    pre_subjects = set(pre_df['Subject_ID'].unique()) if len(pre_df) > 0 else set()
    post_subjects = set(post_df['Subject_ID'].unique()) if len(post_df) > 0 else set()
    common_subjects = pre_subjects.intersection(post_subjects)
    
    if len(common_subjects) < 1:
        return None
    
    # Prepare paired data
    paired_data = []
    for subject in common_subjects:
        pre_val = pre_df[pre_df['Subject_ID'] == subject][feature].iloc[0] if len(pre_df[pre_df['Subject_ID'] == subject]) > 0 else np.nan
        post_val = post_df[post_df['Subject_ID'] == subject][feature].iloc[0] if len(post_df[post_df['Subject_ID'] == subject]) > 0 else np.nan
        
        if not np.isnan(pre_val) and not np.isnan(post_val):
            paired_data.append({
                'subject': subject,
                'pre': pre_val,
                'post': post_val,
                'change': post_val - pre_val
            })
    
    if len(paired_data) == 0:
        return None
    
    # Sort by change magnitude for better visualization
    paired_data.sort(key=lambda x: x['change'])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # X positions for PRE and POST
    x_pre = 0
    x_post = 1
    
    # Determine alpha and line width based on number of subjects
    n_subjects = len(paired_data)
    if n_subjects > 50:
        line_alpha = 0.3
        point_alpha = 0.5
        linewidth = 1.0
        point_size = 40
    elif n_subjects > 20:
        line_alpha = 0.4
        point_alpha = 0.6
        linewidth = 1.2
        point_size = 60
    else:
        line_alpha = 0.6
        point_alpha = 0.8
        linewidth = 1.5
        point_size = 80
    
    # Plot individual lines and points
    for i, data in enumerate(paired_data):
        # Color based on direction of change
        if data['change'] > 0:
            color = 'red'
        else:
            color = 'blue'
        
        # Draw connecting line
        ax.plot([x_pre, x_post], [data['pre'], data['post']], 
               color=color, linewidth=linewidth, alpha=line_alpha, zorder=1)
        
        # Plot PRE point
        ax.scatter(x_pre, data['pre'], color='blue', s=point_size, 
                  alpha=point_alpha, zorder=3, edgecolors='darkblue', linewidths=1)
        
        # Plot POST point
        ax.scatter(x_post, data['post'], color='red', s=point_size, 
                  alpha=point_alpha, zorder=3, edgecolors='darkred', linewidths=1)
    
    # Add mean lines
    pre_mean = np.mean([d['pre'] for d in paired_data])
    post_mean = np.mean([d['post'] for d in paired_data])
    
    ax.axhline(y=pre_mean, xmin=0, xmax=0.5, color='blue', linestyle='--', 
              linewidth=2, alpha=0.7, label=f'{pre_label} Mean')
    ax.axhline(y=post_mean, xmin=0.5, xmax=1, color='red', linestyle='--', 
              linewidth=2, alpha=0.7, label=f'{post_label} Mean')
    
    # Customize plot
    ax.set_xticks([x_pre, x_post])
    ax.set_xticklabels([pre_label, post_label], fontsize=12, fontweight='bold')
    ax.set_ylabel(feature, fontsize=12, fontweight='bold')
    ax.set_title(f'Individual Changes: {feature}\n(N={len(paired_data)} paired subjects)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Add summary statistics text
    mean_change = np.mean([d['change'] for d in paired_data])
    std_change = np.std([d['change'] for d in paired_data], ddof=1)
    ax.text(0.02, 0.98, f'Mean Change: {mean_change:.3f} ± {std_change:.3f}', 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    return fig

def create_single_feature_forest_plot(pre_vals, post_vals, feature, pre_label, post_label, p_value=None, is_mixed=False):
    """
    Create a forest plot for a single feature showing pre vs post with significance line.
    Values are z-scored before plotting.
    
    Parameters
    ----------
    pre_vals : list
        PRE group values
    post_vals : list
        POST group values
    feature : str
        Feature name
    pre_label : str
        Label for PRE group
    post_label : str
        Label for POST group
    p_value : float, optional
        P-value for significance indication
    is_mixed : bool
        Whether analysis is mixed paired/unpaired
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure object if successful, None otherwise
    """
    if len(pre_vals) == 0 or len(post_vals) == 0:
        return None
    
    # Z-score the values using combined distribution
    all_vals = np.array(pre_vals + post_vals)
    combined_mean = np.mean(all_vals)
    combined_std = np.std(all_vals, ddof=1)
    
    if combined_std == 0:
        # If std is 0, all values are the same, skip z-scoring
        pre_vals_z = np.array(pre_vals)
        post_vals_z = np.array(post_vals)
    else:
        # Z-score both groups using combined mean and std
        pre_vals_z = (np.array(pre_vals) - combined_mean) / combined_std
        post_vals_z = (np.array(post_vals) - combined_mean) / combined_std
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Calculate statistics on z-scored values
    pre_mean = np.mean(pre_vals_z)
    pre_std = np.std(pre_vals_z, ddof=1)
    pre_se = pre_std / np.sqrt(len(pre_vals_z))
    pre_ci_lower = pre_mean - 1.96 * pre_se
    pre_ci_upper = pre_mean + 1.96 * pre_se
    
    post_mean = np.mean(post_vals_z)
    post_std = np.std(post_vals_z, ddof=1)
    post_se = post_std / np.sqrt(len(post_vals_z))
    post_ci_lower = post_mean - 1.96 * post_se
    post_ci_upper = post_mean + 1.96 * post_se
    
    # Calculate difference on z-scored values
    mean_diff = post_mean - pre_mean
    if is_mixed:
        # Unpaired: use pooled standard error
        pooled_se = np.sqrt((pre_std**2 / len(pre_vals_z)) + (post_std**2 / len(post_vals_z)))
    else:
        # Paired: use difference scores on z-scored values
        differences = post_vals_z - pre_vals_z
        diff_std = np.std(differences, ddof=1)
        pooled_se = diff_std / np.sqrt(len(differences))
    
    diff_ci_lower = mean_diff - 1.96 * pooled_se
    diff_ci_upper = mean_diff + 1.96 * pooled_se
    
    # Determine significance color
    is_significant = p_value is not None and p_value < 0.05
    color = 'red' if is_significant else 'gray'
    marker_style = '*' if is_significant else 'o'
    
    # Y positions for PRE, POST, and Difference
    y_positions = [0, 1, 2]
    labels = [pre_label, post_label, f'{post_label} - {pre_label}']
    
    # Plot PRE
    ax.plot([pre_ci_lower, pre_ci_upper], [y_positions[0], y_positions[0]], 
           color='blue', linewidth=2.5, alpha=0.7)
    ax.scatter(pre_mean, y_positions[0], color='blue', s=100, 
              marker='o', alpha=0.8, zorder=5, edgecolors='darkblue', linewidths=1.5)
    
    # Plot POST
    ax.plot([post_ci_lower, post_ci_upper], [y_positions[1], y_positions[1]], 
           color='red', linewidth=2.5, alpha=0.7)
    ax.scatter(post_mean, y_positions[1], color='red', s=100, 
              marker='s', alpha=0.8, zorder=5, edgecolors='darkred', linewidths=1.5)
    
    # Plot Difference with significance indication
    ax.plot([diff_ci_lower, diff_ci_upper], [y_positions[2], y_positions[2]], 
           color=color, linewidth=2.5, alpha=0.7)
    ax.scatter(mean_diff, y_positions[2], color=color, s=120, 
              marker=marker_style, alpha=0.8, zorder=5, edgecolors='black', linewidths=1.5)
    
    # Add vertical line at 0 (no difference)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1.5, label='No difference')
    
    # Add significance line if significant
    if is_significant:
        # Add a vertical line at the mean difference to highlight significance
        ax.axvline(x=mean_diff, color=color, linestyle=':', alpha=0.3, linewidth=1)
    
    # Customize plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Z-scored Value (Mean ± 95% CI)', fontsize=10, fontweight='bold')
    ax.set_title(f'{feature}\n{"* p < 0.05" if is_significant else "ns"}', 
                fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Add text annotation for p-value if available
    if p_value is not None:
        ax.text(0.98, 0.02, f'p = {p_value:.3e}', transform=ax.transAxes,
               ha='right', va='bottom', fontsize=9, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig

def create_paired_scatterplot(pre_df, post_df, feature, pre_label, post_label, pairing_info, p_value=None):
    """
    Create a paired line plot with boxplots showing PRE vs POST values.
    X-axis has Pre and Post only, with lines connecting each patient and boxplots with STD.
    
    Parameters
    ----------
    pre_df : pd.DataFrame
        PRE group data
    post_df : pd.DataFrame
        POST group data
    feature : str
        Feature name
    pre_label : str
        Label for PRE group
    post_label : str
        Label for POST group
    pairing_info : dict or set
        Pairing information
    p_value : float, optional
        P-value for title
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure object if successful, None otherwise
    """
    # Get paired subjects
    pre_subjects = set(pre_df['Subject_ID'].unique()) if len(pre_df) > 0 else set()
    post_subjects = set(post_df['Subject_ID'].unique()) if len(post_df) > 0 else set()
    common_subjects = pre_subjects.intersection(post_subjects)
    
    if len(common_subjects) < 2:
        return None
    
    # Prepare paired data
    pre_vals = []
    post_vals = []
    subject_ids = []
    
    for subject in common_subjects:
        pre_val = pre_df[pre_df['Subject_ID'] == subject][feature].iloc[0] if len(pre_df[pre_df['Subject_ID'] == subject]) > 0 else np.nan
        post_val = post_df[post_df['Subject_ID'] == subject][feature].iloc[0] if len(post_df[post_df['Subject_ID'] == subject]) > 0 else np.nan
        
        if not np.isnan(pre_val) and not np.isnan(post_val):
            pre_vals.append(pre_val)
            post_vals.append(post_val)
            subject_ids.append(subject)
    
    if len(pre_vals) < 2:
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # X positions for Pre and Post
    x_pre = 0
    x_post = 1
    x_positions = [x_pre, x_post]
    x_labels = [pre_label, post_label]
    
    # Prepare data for boxplots
    boxplot_data = [pre_vals, post_vals]
    
    # Create boxplots with STD
    ax.boxplot(boxplot_data, positions=x_positions, widths=0.3, 
                    patch_artist=True, showmeans=True, meanline=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.5),
                    medianprops=dict(color='black', linewidth=2),
                    meanprops=dict(color='red', linewidth=1.5, linestyle='--'),
                    whiskerprops=dict(color='black', linewidth=1),
                    capprops=dict(color='black', linewidth=1),
                    zorder=1)
    
    # Add STD information to boxplots
    pre_mean = np.mean(pre_vals)
    pre_std = np.std(pre_vals)
    post_mean = np.mean(post_vals)
    post_std = np.std(post_vals)
    
    # Draw STD error bars on boxplots
    ax.errorbar(x_pre, pre_mean, yerr=pre_std, fmt='o', color='blue', 
                capsize=5, capthick=2, markersize=8, label=f'{pre_label} Mean ± STD', zorder=3)
    ax.errorbar(x_post, post_mean, yerr=post_std, fmt='s', color='red', 
                capsize=5, capthick=2, markersize=8, label=f'{post_label} Mean ± STD', zorder=3)
    
    # Plot connecting lines for each patient
    for i, (pre_val, post_val) in enumerate(zip(pre_vals, post_vals)):
        ax.plot([x_pre, x_post], [pre_val, post_val], 'gray', alpha=0.4, linewidth=1, zorder=2)
    
    # Plot individual points
    ax.scatter([x_pre] * len(pre_vals), pre_vals, color='blue', s=60, alpha=0.6, 
              marker='o', zorder=4, edgecolors='darkblue', linewidths=1)
    ax.scatter([x_post] * len(post_vals), post_vals, color='red', s=60, alpha=0.6, 
              marker='s', zorder=4, edgecolors='darkred', linewidths=1)
    
    # Customize plot
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{feature} Value', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.3, 1.3)
    
    # Add title with p-value
    if p_value is not None:
        sig_marker = ' *' if p_value < 0.05 else ''
        title = f'{feature}\np = {p_value:.3e}{sig_marker} (Paired, N={len(pre_vals)})'
    else:
        title = f'{feature}\n(Paired, N={len(pre_vals)})'
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    return fig

def create_simple_paired_scatterplot(pre_df, post_df, feature, pre_label, post_label, pairing_info, p_value=None):
    """
    Create a simple scatterplot with PRE and POST on x-axis, connecting lines for each patient.
    
    Parameters
    ----------
    pre_df : pd.DataFrame
        PRE group data
    post_df : pd.DataFrame
        POST group data
    feature : str
        Feature name
    pre_label : str
        Label for PRE group
    post_label : str
        Label for POST group
    pairing_info : dict or set
        Pairing information
    p_value : float, optional
        P-value for title
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure object if successful, None otherwise
    """
    # Get paired subjects
    pre_subjects = set(pre_df['Subject_ID'].unique()) if len(pre_df) > 0 else set()
    post_subjects = set(post_df['Subject_ID'].unique()) if len(post_df) > 0 else set()
    common_subjects = pre_subjects.intersection(post_subjects)
    
    if len(common_subjects) < 2:
        return None
    
    # Prepare paired data
    pre_vals = []
    post_vals = []
    subject_ids = []
    
    for subject in common_subjects:
        pre_val = pre_df[pre_df['Subject_ID'] == subject][feature].iloc[0] if len(pre_df[pre_df['Subject_ID'] == subject]) > 0 else np.nan
        post_val = post_df[post_df['Subject_ID'] == subject][feature].iloc[0] if len(post_df[post_df['Subject_ID'] == subject]) > 0 else np.nan
        
        if not np.isnan(pre_val) and not np.isnan(post_val):
            pre_vals.append(pre_val)
            post_vals.append(post_val)
            subject_ids.append(subject)
    
    if len(pre_vals) < 2:
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # X positions for PRE and POST
    x_pre = 0
    x_post = 1
    x_positions = [x_pre, x_post]
    x_labels = [pre_label, post_label]
    
    # Plot connecting lines for each patient
    for i, (pre_val, post_val) in enumerate(zip(pre_vals, post_vals)):
        ax.plot([x_pre, x_post], [pre_val, post_val], 'gray', alpha=0.3, linewidth=1, zorder=1)
    
    # Plot PRE points
    ax.scatter([x_pre] * len(pre_vals), pre_vals, color='blue', s=100, alpha=0.6, 
              marker='o', label=pre_label, zorder=2, edgecolors='darkblue', linewidths=1.5)
    
    # Plot POST points
    ax.scatter([x_post] * len(post_vals), post_vals, color='red', s=100, alpha=0.6, 
              marker='s', label=post_label, zorder=3, edgecolors='darkred', linewidths=1.5)
    
    # Add mean lines
    pre_mean = np.mean(pre_vals)
    post_mean = np.mean(post_vals)
    ax.axhline(y=pre_mean, color='blue', linestyle='--', alpha=0.5, linewidth=1.5, label=f'{pre_label} Mean')
    ax.axhline(y=post_mean, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label=f'{post_label} Mean')
    
    # Customize plot
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{feature} Value', fontsize=12, fontweight='bold')
    
    # Add title with p-value
    if p_value is not None:
        sig_marker = ' *' if p_value < 0.05 else ''
        title = f'{feature}\np = {p_value:.3e}{sig_marker} (Paired, N={len(pre_vals)})'
    else:
        title = f'{feature}\n(Paired, N={len(pre_vals)})'
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    return fig

def create_simple_paired_scatterplots_for_significant_features(pre_df, post_df, eeg_features, pre_label, post_label, 
                                                                 pairing_info, is_mixed, significant_only=True):
    """
    Create simple scatterplots for all significant features showing PRE and POST on x-axis with connecting lines.
    
    Parameters
    ----------
    pre_df : pd.DataFrame
        PRE group data
    post_df : pd.DataFrame
        POST group data
    eeg_features : list
        List of EEG feature names
    pre_label : str
        Label for PRE group
    post_label : str
        Label for POST group
    pairing_info : dict or set
        Pairing information
    is_mixed : bool
        Whether analysis is mixed paired/unpaired
    significant_only : bool
        If True, only show significant features (p < 0.05)
    
    Returns
    -------
    dict : Dictionary mapping feature names to figure objects
    """
    scatterplot_figs = {}
    
    for feature in eeg_features:
        # Calculate p-value for this feature
        p_value = None
        if pairing_info is not None:
            try:
                result = calculate_effect_size_and_power(pre_df, post_df, feature, pairing_info)
                if result[0] is not None:
                    p_value = result[0]
            except (KeyError, Exception):
                pass
        
        # Filter for significant features if requested
        if significant_only and (p_value is None or p_value >= 0.05):
            continue
        
        # Create scatterplot for this feature
        fig = create_simple_paired_scatterplot(pre_df, post_df, feature, pre_label, post_label, pairing_info, p_value=p_value)
        if fig is not None:
            scatterplot_figs[feature] = fig
    
    return scatterplot_figs

def create_paired_scatterplots_for_significant_features(pre_df, post_df, eeg_features, pre_label, post_label, 
                                                         pairing_info, is_mixed, significant_only=True):
    """
    Create scatterplots for all significant features showing paired PRE vs POST comparisons.
    
    Parameters
    ----------
    pre_df : pd.DataFrame
        PRE group data
    post_df : pd.DataFrame
        POST group data
    eeg_features : list
        List of EEG feature names
    pre_label : str
        Label for PRE group
    post_label : str
        Label for POST group
    pairing_info : dict or set
        Pairing information
    is_mixed : bool
        Whether analysis is mixed paired/unpaired
    significant_only : bool
        If True, only show significant features (p < 0.05)
    
    Returns
    -------
    dict : Dictionary mapping feature names to figure objects
    """
    scatterplot_figs = {}
    
    for feature in eeg_features:
        # Calculate p-value for this feature
        p_value = None
        if pairing_info is not None:
            try:
                result = calculate_effect_size_and_power(pre_df, post_df, feature, pairing_info)
                if result[0] is not None:
                    p_value = result[0]
            except (KeyError, Exception):
                pass
        
        # Filter for significant features if requested
        if significant_only and (p_value is None or p_value >= 0.05):
            continue
        
        # Create scatterplot for this feature
        fig = create_paired_scatterplot(pre_df, post_df, feature, pre_label, post_label, pairing_info, p_value=p_value)
        if fig is not None:
            scatterplot_figs[feature] = fig
    
    return scatterplot_figs

def create_median_std_forest_plot(pre_df, post_df, eeg_features, pre_label, post_label, is_mixed, pairing_info, significant_only=True):
    """
    Create a forest plot showing median ± STD for EEG features.
    Optionally filter to only significant features.
    
    Parameters
    ----------
    pre_df : pd.DataFrame
        PRE group data
    post_df : pd.DataFrame
        POST group data
    eeg_features : list
        List of EEG feature names
    pre_label : str
        Label for PRE group
    post_label : str
        Label for POST group
    is_mixed : bool
        Whether analysis is mixed paired/unpaired
    pairing_info : dict or set
        Pairing information
    significant_only : bool
        If True, only show significant features (p < 0.05)
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure object if successful, None otherwise
    """
    plot_data = []
    
    for feature in eeg_features:
        # Get data for this feature
        if is_mixed:
            # For mixed analysis, get all data
            pre_vals = pre_df[feature].dropna().tolist()
            post_vals = post_df[feature].dropna().tolist()
        else:
            # For paired analysis, get only paired data
            pre_subjects = set(pre_df['Subject_ID'].unique()) if len(pre_df) > 0 else set()
            post_subjects = set(post_df['Subject_ID'].unique()) if len(post_df) > 0 else set()
            common_subjects = pre_subjects.intersection(post_subjects)
            
            pre_vals = []
            post_vals = []
            for subject in common_subjects:
                pre_val = pre_df[pre_df['Subject_ID'] == subject][feature].iloc[0] if len(pre_df[pre_df['Subject_ID'] == subject]) > 0 else np.nan
                post_val = post_df[post_df['Subject_ID'] == subject][feature].iloc[0] if len(post_df[post_df['Subject_ID'] == subject]) > 0 else np.nan
                
                if not np.isnan(pre_val) and not np.isnan(post_val):
                    pre_vals.append(pre_val)
                    post_vals.append(post_val)
        
        # Calculate p-value if filtering for significant features
        if significant_only and pairing_info is not None:
            try:
                result = calculate_effect_size_and_power(pre_df, post_df, feature, pairing_info)
                if result[0] is not None:
                    p_value = result[0]
                    # Skip if not significant
                    if p_value >= 0.05:
                        continue
                else:
                    # Skip if p-value cannot be calculated
                    continue
            except (KeyError, Exception):
                # Skip if calculation fails
                continue
        
        # Calculate median and STD for each group
        if len(pre_vals) > 0:
            pre_median = np.median(pre_vals)
            pre_std = np.std(pre_vals, ddof=1)
            pre_lower = pre_median - pre_std
            pre_upper = pre_median + pre_std
        else:
            pre_median = np.nan
            pre_std = np.nan
            pre_lower = np.nan
            pre_upper = np.nan
        
        if len(post_vals) > 0:
            post_median = np.median(post_vals)
            post_std = np.std(post_vals, ddof=1)
            post_lower = post_median - post_std
            post_upper = post_median + post_std
        else:
            post_median = np.nan
            post_std = np.nan
            post_lower = np.nan
            post_upper = np.nan
        
        if not (np.isnan(pre_median) and np.isnan(post_median)):
            plot_data.append({
                'feature': feature,
                'pre_median': pre_median,
                'pre_lower': pre_lower,
                'pre_upper': pre_upper,
                'pre_std': pre_std,
                'pre_n': len(pre_vals),
                'post_median': post_median,
                'post_lower': post_lower,
                'post_upper': post_upper,
                'post_std': post_std,
                'post_n': len(post_vals)
            })
    
    if not plot_data:
        return None
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, max(8, len(plot_data) * 0.6)))
    
    y_positions = np.arange(len(plot_data))
    
    # Plot PRE and POST separately
    for i, data in enumerate(plot_data):
        y_pos = y_positions[i]
        
        # Plot PRE group
        if not np.isnan(data['pre_median']):
            # Error bar for PRE (median ± STD)
            ax.plot([data['pre_lower'], data['pre_upper']], [y_pos - 0.15, y_pos - 0.15], 
                   color='blue', linewidth=2.5, alpha=0.7, label='PRE' if i == 0 else '')
            ax.scatter(data['pre_median'], y_pos - 0.15, color='blue', s=120, 
                      marker='o', alpha=0.8, zorder=5, edgecolors='darkblue', linewidths=1.5)
        
        # Plot POST group
        if not np.isnan(data['post_median']):
            # Error bar for POST (median ± STD)
            ax.plot([data['post_lower'], data['post_upper']], [y_pos + 0.15, y_pos + 0.15], 
                   color='red', linewidth=2.5, alpha=0.7, label='POST' if i == 0 else '')
            ax.scatter(data['post_median'], y_pos + 0.15, color='red', s=120, 
                      marker='s', alpha=0.8, zorder=5, edgecolors='darkred', linewidths=1.5)
    
    # Customize plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels([data['feature'] for data in plot_data])
    ax.set_xlabel('Value (Median ± STD)', fontsize=12, fontweight='bold')
    if significant_only:
        ax.set_title('Forest Plot: Significant EEG Features (Median ± STD, p < 0.05)', fontsize=14, fontweight='bold', pad=20)
    else:
        ax.set_title('Forest Plot: All EEG Features (Median ± STD)', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Add vertical line at median of all medians for reference
    all_medians = [d['pre_median'] for d in plot_data if not np.isnan(d['pre_median'])] + \
                  [d['post_median'] for d in plot_data if not np.isnan(d['post_median'])]
    if all_medians:
        overall_median = np.median(all_medians)
        ax.axvline(x=overall_median, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.text(overall_median, len(plot_data) - 0.5, f'Overall Median: {overall_median:.3f}', 
               rotation=90, va='bottom', ha='right', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    return fig

def calculate_effect_size_and_power(pre_data, post_data, feature, pairing_info=None):
    """
    Calculate effect size and statistical power for paired or mixed paired/unpaired data.
    
    Parameters
    ----------
    pre_data : pd.DataFrame
        PRE group data
    post_data : pd.DataFrame
        POST group data
    feature : str
        Feature name to analyze
    pairing_info : dict or set, optional
        If dict, contains pairing information for mixed analysis.
        If set, contains paired subject IDs (legacy format).
        If None, will compute pairing from data.
    
    Returns
    -------
    p_value : float
        P-value from statistical test
    cohen_d : float
        Cohen's d effect size
    effect_size_desc : str
        Effect size description
    power : float
        Statistical power
    test_type : str
        Type of test used ('paired' or 'unpaired')
    """
    
    # Determine if we have mixed paired/unpaired data
    is_mixed = isinstance(pairing_info, dict)
    
    # Ensure we have the same subjects in both datasets
    pre_subjects = set(pre_data['Subject_ID'].unique()) if len(pre_data) > 0 else set()
    post_subjects = set(post_data['Subject_ID'].unique()) if len(post_data) > 0 else set()
    common_subjects = pre_subjects.intersection(post_subjects)
    
    if is_mixed:
        # Mixed analysis: use paired test for paired subjects, unpaired for all data
        paired_subjects = pairing_info.get('paired', set())
        
        # Prepare paired data
        pre_vals_paired = []
        post_vals_paired = []
        for subject in paired_subjects:
            if subject in pre_subjects and subject in post_subjects:
                pre_val = pre_data[pre_data['Subject_ID'] == subject][feature].iloc[0] if len(pre_data[pre_data['Subject_ID'] == subject]) > 0 else np.nan
                post_val = post_data[post_data['Subject_ID'] == subject][feature].iloc[0] if len(post_data[post_data['Subject_ID'] == subject]) > 0 else np.nan
                
                if not np.isnan(pre_val) and not np.isnan(post_val):
                    pre_vals_paired.append(pre_val)
                    post_vals_paired.append(post_val)
        
        # Prepare unpaired data
        unpaired_pre_subjects = pairing_info.get('unpaired_pre', set())
        unpaired_post_subjects = pairing_info.get('unpaired_post', set())
        
        pre_vals_unpaired = []
        for subject in unpaired_pre_subjects:
            if subject in pre_subjects:
                pre_val = pre_data[pre_data['Subject_ID'] == subject][feature].iloc[0] if len(pre_data[pre_data['Subject_ID'] == subject]) > 0 else np.nan
                if not np.isnan(pre_val):
                    pre_vals_unpaired.append(pre_val)
        
        post_vals_unpaired = []
        for subject in unpaired_post_subjects:
            if subject in post_subjects:
                post_val = post_data[post_data['Subject_ID'] == subject][feature].iloc[0] if len(post_data[post_data['Subject_ID'] == subject]) > 0 else np.nan
                if not np.isnan(post_val):
                    post_vals_unpaired.append(post_val)
        
        # Combine all data for unpaired test
        all_pre_vals = pre_vals_paired + pre_vals_unpaired
        all_post_vals = post_vals_paired + post_vals_unpaired
        
        if len(all_pre_vals) < 2 or len(all_post_vals) < 2:
            return None, None, None, None, None
        
        # Use Mann-Whitney U test for unpaired comparison of all data
        try:
            stat, p_value = stats.mannwhitneyu(all_post_vals, all_pre_vals, alternative='two-sided')
            test_type = 'unpaired'
        except Exception:
            p_value = np.nan
            test_type = 'unpaired'
        
        # Cohen's d for unpaired data (pooled standard deviation)
        try:
            pooled_std = np.sqrt((((len(all_pre_vals) - 1) * np.std(all_pre_vals, ddof=1)**2 + 
                                  (len(all_post_vals) - 1) * np.std(all_post_vals, ddof=1)**2) / 
                                 (len(all_pre_vals) + len(all_post_vals) - 2)))
            if pooled_std > 0:
                cohen_d = (np.mean(all_post_vals) - np.mean(all_pre_vals)) / pooled_std
            else:
                cohen_d = np.nan
        except Exception:
            cohen_d = np.nan
        
        # Statistical power for unpaired t-test
        try:
            from statsmodels.stats.power import TTestIndPower
            analysis = TTestIndPower()
            power = analysis.power(effect_size=abs(cohen_d) if not np.isnan(cohen_d) else 0, 
                                  nobs1=len(all_pre_vals), 
                                  ratio=len(all_post_vals)/len(all_pre_vals) if len(all_pre_vals) > 0 else 1,
                                  alpha=0.05, 
                                  alternative='two-sided')
        except Exception:
            power = np.nan
        
    else:
        # Standard paired analysis
        if len(common_subjects) < 2:
            return None, None, None, None, None
        
        # Prepare paired data
        pre_vals = []
        post_vals = []
        for subject in common_subjects:
            pre_val = pre_data[pre_data['Subject_ID'] == subject][feature].iloc[0] if len(pre_data[pre_data['Subject_ID'] == subject]) > 0 else np.nan
            post_val = post_data[post_data['Subject_ID'] == subject][feature].iloc[0] if len(post_data[post_data['Subject_ID'] == subject]) > 0 else np.nan
            
            if not np.isnan(pre_val) and not np.isnan(post_val):
                pre_vals.append(pre_val)
                post_vals.append(post_val)
        
        if len(pre_vals) < 2:
            return None, None, None, None, None
        
        # Wilcoxon signed-rank test for paired data
        try:
            stat, p_value = stats.wilcoxon(post_vals, pre_vals, alternative='two-sided')
            test_type = 'paired'
        except Exception:
            p_value = np.nan
            test_type = 'paired'
        
        # Cohen's d for paired data (using difference scores)
        try:
            differences = np.array(post_vals) - np.array(pre_vals)
            cohen_d = np.mean(differences) / np.std(differences, ddof=1)
        except Exception:
            cohen_d = np.nan
        
        # Statistical power for paired t-test
        try:
            from statsmodels.stats.power import TTestPower
            analysis = TTestPower()
            power = analysis.power(effect_size=abs(cohen_d) if not np.isnan(cohen_d) else 0, 
                                  nobs=len(pre_vals), 
                                  alpha=0.05, 
                                  alternative='two-sided')
        except Exception:
            power = np.nan
    
    # Effect size interpretation
    if cohen_d is None or np.isnan(cohen_d):
        effect_size_desc = "unknown"
    elif abs(cohen_d) < 0.2:
        effect_size_desc = "negligible"
    elif abs(cohen_d) < 0.5:
        effect_size_desc = "small"
    elif abs(cohen_d) < 0.8:
        effect_size_desc = "medium"
    else:
        effect_size_desc = "large"
    
    return p_value, cohen_d, effect_size_desc, power, test_type

def generate_excel_data(pre_df, post_df, eeg_features, pairing_info, is_mixed, pre_label, post_label):
    """
    Generate Excel data with Subject_ID, PRE and POST values for each feature.
    
    Parameters
    ----------
    pre_df : pd.DataFrame
        PRE group data
    post_df : pd.DataFrame
        POST group data
    eeg_features : list
        List of EEG feature names
    pairing_info : dict or set
        Pairing information
    is_mixed : bool
        Whether analysis is mixed paired/unpaired
    pre_label : str
        Label for PRE group
    post_label : str
        Label for POST group
    
    Returns
    -------
    bytes : Excel file as bytes
    """
    # Get all unique subjects from both datasets
    pre_subjects = set(pre_df['Subject_ID'].unique()) if len(pre_df) > 0 else set()
    post_subjects = set(post_df['Subject_ID'].unique()) if len(post_df) > 0 else set()
    all_subjects = sorted(list(pre_subjects | post_subjects))
    
    if len(all_subjects) == 0:
        return None
    
    # Create a dictionary to store the data
    excel_data = {'Subject_ID': all_subjects}
    
    # For each feature, add PRE and POST columns
    for feature in eeg_features:
        pre_col_name = f"{feature}_{pre_label}"
        post_col_name = f"{feature}_{post_label}"
        
        pre_values = []
        post_values = []
        
        for subject in all_subjects:
            # Get PRE value
            pre_subject_data = pre_df[pre_df['Subject_ID'] == subject]
            if len(pre_subject_data) > 0 and feature in pre_subject_data.columns:
                pre_val = pre_subject_data[feature].iloc[0] if not pre_subject_data[feature].isna().all() else np.nan
            else:
                pre_val = np.nan
            pre_values.append(pre_val)
            
            # Get POST value
            post_subject_data = post_df[post_df['Subject_ID'] == subject]
            if len(post_subject_data) > 0 and feature in post_subject_data.columns:
                post_val = post_subject_data[feature].iloc[0] if not post_subject_data[feature].isna().all() else np.nan
            else:
                post_val = np.nan
            post_values.append(post_val)
        
        excel_data[pre_col_name] = pre_values
        excel_data[post_col_name] = post_values
    
    # Create DataFrame
    excel_df = pd.DataFrame(excel_data)
    
    # Create Excel file in memory
    excel_buffer = io.BytesIO()
    try:
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            excel_df.to_excel(writer, index=False, sheet_name='Analysis Data')
        excel_buffer.seek(0)
        return excel_buffer.getvalue()
    except ImportError:
        raise ImportError("openpyxl is required for Excel export. Please install it with: pip install openpyxl")
    except Exception as e:
        raise Exception(f"Error creating Excel file: {str(e)}")

def generate_pdf_report(pre_df, post_df, eeg_features, pairing_info, is_mixed, pre_label, post_label, 
                        montage, selected_feature, show_significant_only):
    """
    Generate a PDF report with all plots and results.
    
    Returns
    -------
    bytes : PDF file as bytes
    """
    pdf_buffer = io.BytesIO()
    
    with PdfPages(pdf_buffer) as pdf:
        # Title page
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.9, 'VNS Pre/Post EEG Comparison Report', 
                ha='center', va='top', fontsize=20, fontweight='bold')
        fig.text(0.5, 0.8, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                ha='center', va='top', fontsize=12)
        fig.text(0.5, 0.7, f'Analysis: {selected_feature}', 
                ha='center', va='top', fontsize=14)
        fig.text(0.5, 0.6, f'{pre_label} vs {post_label}', 
                ha='center', va='top', fontsize=14)
        if is_mixed:
            fig.text(0.5, 0.5, 'Mixed paired/unpaired analysis', 
                    ha='center', va='top', fontsize=12)
        else:
            fig.text(0.5, 0.5, 'Paired analysis', 
                    ha='center', va='top', fontsize=12)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Collect all forest plots for significant features
        if selected_feature == "All Features":
            for feature in eeg_features:
                # Get data for this feature
                if is_mixed:
                    pre_vals = pre_df[feature].dropna().tolist()
                    post_vals = post_df[feature].dropna().tolist()
                else:
                    pre_subjects = set(pre_df['Subject_ID'].unique()) if len(pre_df) > 0 else set()
                    post_subjects = set(post_df['Subject_ID'].unique()) if len(post_df) > 0 else set()
                    common_subjects = pre_subjects.intersection(post_subjects)
                    
                    pre_vals = []
                    post_vals = []
                    for subject in common_subjects:
                        pre_val = pre_df[pre_df['Subject_ID'] == subject][feature].iloc[0] if len(pre_df[pre_df['Subject_ID'] == subject]) > 0 else np.nan
                        post_val = post_df[post_df['Subject_ID'] == subject][feature].iloc[0] if len(post_df[post_df['Subject_ID'] == subject]) > 0 else np.nan
                        
                        if not np.isnan(pre_val) and not np.isnan(post_val):
                            pre_vals.append(pre_val)
                            post_vals.append(post_val)
                
                if len(pre_vals) > 0 and len(post_vals) > 0:
                    result = calculate_effect_size_and_power(pre_df, post_df, feature, pairing_info)
                    if result[0] is not None:
                        p_value = result[0]
                    else:
                        p_value = None
                    
                    # Only include significant features if filtering is enabled
                    if not show_significant_only or (p_value is not None and p_value < 0.05):
                        fig = create_single_feature_forest_plot(
                            pre_vals, post_vals, feature, pre_label, post_label, 
                            p_value=p_value, is_mixed=is_mixed
                        )
                        if fig is not None:
                            pdf.savefig(fig, bbox_inches='tight')
                            plt.close(fig)
            
            # Add topomaps for significant features
            topomap_figs = create_topomaps_for_all_features(
                pre_df, post_df, eeg_features, montage, pre_label, post_label,
                pairing_info=pairing_info, is_mixed=is_mixed, significant_only=True
            )
            
            for feature_name, (fig, p_val) in topomap_figs.items():
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
    
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

def main():
    st.set_page_config(page_title="VNS Pre/Post EEG Comparison", layout="wide")
    
    # Dataset selection
    dataset_options = {
        "reading_epilepsy_cut": "Reading Epilepsy Cut Analysis",
        "VNS_PRE_POST_25": "VNS Pre/Post Analysis",
        "Pre-Post_XCOPRI": "Pre-Post_XCOPRI Analysis",
    }
    
    dataset_keys = list(dataset_options.keys())
    default_index = dataset_keys.index("Pre-Post_XCOPRI") if "Pre-Post_XCOPRI" in dataset_keys else 0
    selected_dataset = st.selectbox(
        "Select Dataset:",
        options=dataset_keys,
        format_func=lambda x: dataset_options[x],
        index=default_index
    )
    
    st.markdown(f"**Selected Dataset:** {dataset_options[selected_dataset]}")
    
    # Label selection
    st.markdown("### Group Labels")
    col1, col2 = st.columns(2)
    with col1:
        pre_label = st.text_input("PRE Group Label:", value="PRE", help="Label for the first group (e.g., 'Before', 'Baseline', 'Non-Reading')")
    with col2:
        post_label = st.text_input("POST Group Label:", value="POST", help="Label for the second group (e.g., 'After', 'Treatment', 'Reading')")
    
    # Analysis level selection
    st.markdown("### Analysis Level")
    analysis_level = st.radio(
        "Choose analysis level:",
        ["Per Patient (averaged)", "Per Scan (individual sessions)", "All Samples (mixed paired/non-paired)"],
        help="Per Patient: Average values across all sessions for each patient (paired only). Per Scan: Analyze each individual session separately (paired only). All Samples: Show all samples, handling both paired and non-paired data with multiple scans per patient."
    )
    
    # Results filtering
    st.markdown("### Results Display")
    show_significant_only = st.checkbox(
        "Show only significant results (p < 0.05)",
        value=True,
        help="When checked, only displays plots and statistics for measurements with p < 0.05. When unchecked, shows all results."
    )
    
    st.markdown("---")
    
    # Load data
    with st.spinner(f"Loading {dataset_options[selected_dataset]} data..."):
        pre_df, post_df, pairing_info = load_pre_post_data(selected_dataset, analysis_level)
    
    if pre_df is None and post_df is None:
        st.error("Failed to load data. Please check the data directories.")
        return
    
    # Process data based on analysis level
    if analysis_level == "Per Patient (averaged)":
        # Data is already averaged per patient from load_pre_post_data
        st.info("📊 Analysis: Per Patient (averaged across sessions, paired only)")
    elif analysis_level == "All Samples (mixed paired/non-paired)":
        st.info("📊 Analysis: All Samples (mixed paired/non-paired, multiple scans per patient averaged)")
    else:
        # For per-scan analysis, we need to load individual sessions
        st.info("📊 Analysis: Per Scan (individual sessions, paired only)")
        # Note: The current data structure already contains individual sessions
        # The averaging happens in load_pre_post_data, so we need to modify that function
        # For now, we'll work with the current data structure
    
    # Determine if pairing_info is dict (mixed) or set (paired only)
    is_mixed = isinstance(pairing_info, dict)
    
    # Display data summary
    if is_mixed:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric(f"{pre_label} Sessions", pairing_info.get('total_pre', len(pre_df) if len(pre_df) > 0 else 0))
        with col2:
            st.metric(f"{post_label} Sessions", pairing_info.get('total_post', len(post_df) if len(post_df) > 0 else 0))
        with col3:
            st.metric("Paired Subjects", pairing_info.get('paired_count', 0))
        with col4:
            st.metric(f"Unpaired {pre_label}", pairing_info.get('unpaired_pre_count', 0))
        with col5:
            st.metric(f"Unpaired {post_label}", pairing_info.get('unpaired_post_count', 0))
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{pre_label} Sessions", len(pre_df) if len(pre_df) > 0 else 0)
        with col2:
            st.metric(f"{post_label} Sessions", len(post_df) if len(post_df) > 0 else 0)
        with col3:
            st.metric("Paired Subjects", len(pairing_info) if pairing_info else 0)
    
    if len(pre_df) == 0 or len(post_df) == 0:
        st.warning("Insufficient data for comparison. Need both PRE and POST data.")
        return
    
    # Show paired subjects info
    if is_mixed:
        pairing_info.get('paired', set())
        pairing_info.get('unpaired_pre', set())
        pairing_info.get('unpaired_post', set())
        info_text = "Analysis includes:\n"
        info_text += f"- {pairing_info.get('paired_count', 0)} paired subjects (both PRE and POST)\n"
        if pairing_info.get('unpaired_pre_count', 0) > 0:
            info_text += f"- {pairing_info.get('unpaired_pre_count', 0)} unpaired PRE subjects\n"
        if pairing_info.get('unpaired_post_count', 0) > 0:
            info_text += f"- {pairing_info.get('unpaired_post_count', 0)} unpaired POST subjects\n"
        info_text += "\nUsing unpaired statistical tests (Mann-Whitney U) for all data."
        st.info(info_text)
    elif pairing_info:
        st.info(f"Analysis includes {len(pairing_info)} subjects with both PRE and POST data: {', '.join(sorted(list(pairing_info)[:10]))}{'...' if len(pairing_info) > 10 else ''}")
    else:
        st.error("No subjects found with both PRE and POST data!")
        return
    
    # Get EEG features
    eeg_features = get_eeg_features(pre_df)
    eeg_features_forest = [feature for feature in eeg_features if 'overall_' in feature]
    if not eeg_features:
        st.error("No EEG features found in the data.")
        return
    
    # Feature selection
    st.sidebar.header("Analysis Settings")
    feature_options = ["All Features"] + eeg_features
    selected_feature = st.sidebar.selectbox("Select EEG Feature", feature_options)
    
    # Analysis options
    show_dabest = st.sidebar.checkbox("Show Dabest Plots", value=True)
    show_topomaps = st.sidebar.checkbox("Show Topomaps", value=True)
    show_statistics = st.sidebar.checkbox("Show Detailed Statistics", value=True)
    
    # PDF Export - Generate automatically
    st.sidebar.markdown("---")
    st.sidebar.header("Export")
    
    # Create a unique key for this session based on data and settings
    session_key = f"pdf_{selected_feature}_{pre_label}_{post_label}_{is_mixed}_{show_significant_only}"
    
    # Check if PDF is already generated in session state
    if session_key not in st.session_state:
        st.sidebar.info("Generating PDF report...")
        try:
            pdf_bytes = generate_pdf_report(
                pre_df, post_df, eeg_features, pairing_info, is_mixed, 
                pre_label, post_label, montage, selected_feature, show_significant_only
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eeg_comparison_report_{timestamp}.pdf"
            
            # Store in session state
            st.session_state[session_key] = pdf_bytes
            st.session_state[f"{session_key}_filename"] = filename
        except Exception as e:
            st.sidebar.error(f"Error generating PDF: {str(e)}")
            st.session_state[session_key] = None
    
    # Show download button if PDF is available
    if session_key in st.session_state and st.session_state[session_key] is not None:
        st.sidebar.success("PDF ready!")
        st.sidebar.download_button(
            label="📥 Download PDF",
            data=st.session_state[session_key],
            file_name=st.session_state.get(f"{session_key}_filename", "eeg_comparison_report.pdf"),
            mime="application/pdf",
            help="Download the complete analysis report as PDF"
        )
    
    # Excel Export
    excel_session_key = f"excel_{pre_label}_{post_label}_{is_mixed}"
    
    # Check if Excel is already generated in session state
    if excel_session_key not in st.session_state:
        st.sidebar.info("Generating Excel data...")
        try:
            excel_bytes = generate_excel_data(
                pre_df, post_df, eeg_features, pairing_info, is_mixed, 
                pre_label, post_label
            )
            if excel_bytes is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"eeg_analysis_data_{timestamp}.xlsx"
                
                # Store in session state
                st.session_state[excel_session_key] = excel_bytes
                st.session_state[f"{excel_session_key}_filename"] = filename
            else:
                st.session_state[excel_session_key] = None
        except Exception as e:
            st.sidebar.error(f"Error generating Excel: {str(e)}")
            st.session_state[excel_session_key] = None
    
    # Show download button if Excel is available
    if excel_session_key in st.session_state and st.session_state[excel_session_key] is not None:
        st.sidebar.success("Excel ready!")
        st.sidebar.download_button(
            label="📊 Download Excel",
            data=st.session_state[excel_session_key],
            file_name=st.session_state.get(f"{excel_session_key}_filename", "eeg_analysis_data.xlsx"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download analysis data as Excel file (Subject_ID, PRE and POST values for each feature)"
        )
    
    # Check if "All Features" is selected
    if selected_feature == "All Features":
        st.markdown("## Analysis of: All EEG Features")
        
        # Create a grid layout for all features
        num_features = len(eeg_features)
        cols_per_row = 3  # Number of columns per row
        (num_features + cols_per_row - 1) // cols_per_row
        

        
        # Forest plot for measurements
        if show_significant_only:
            st.markdown("## 🌲 Forest Plot - Significant Overall Measurements (p < 0.05)")
        else:
            st.markdown("## 🌲 Forest Plot - All Overall Measurements")
        
        # Collect results based on user selection
        plot_results = []
        for feature in eeg_features_forest:
            # get all values of pre and post in single list
            all_values = pre_df[feature].dropna().tolist() + post_df[feature].dropna().tolist()
            all_values_zscore = (np.array(all_values) - np.mean(all_values)) / np.std(all_values)
            pre_df_non_na_indices = pre_df[feature].dropna().index
            post_df_non_na_indices = post_df[feature].dropna().index
            pre_df.loc[pre_df_non_na_indices, feature] = all_values_zscore[:len(pre_df_non_na_indices)]
            post_df.loc[post_df_non_na_indices, feature] = all_values_zscore[len(pre_df_non_na_indices):]
            pre_vals = pre_df[feature].dropna().tolist()
            post_vals = post_df[feature].dropna().tolist()
            if len(pre_vals) > 0 and len(post_vals) > 0:
                result = calculate_effect_size_and_power(pre_df, post_df, feature, pairing_info)
                if result[0] is not None:
                    p_value, cohen_d, effect_size_desc, power, test_type = result
                else:
                    p_value, cohen_d, effect_size_desc, power, test_type = None, None, None, None, None
                
                # Include results based on user selection
                include_result = True
                if show_significant_only and (p_value is None or p_value >= 0.05):
                    include_result = False
                
                if include_result:
                    mean_diff = np.mean(post_vals) - np.mean(pre_vals)
                    
                    # Calculate CI based on test type
                    if is_mixed or test_type == 'unpaired':
                        # Unpaired: use pooled standard error
                        n_pre, n_post = len(pre_vals), len(post_vals)
                        std_pre = np.std(pre_vals, ddof=1)
                        std_post = np.std(post_vals, ddof=1)
                        pooled_se = np.sqrt((std_pre**2 / n_pre) + (std_post**2 / n_post))
                        ci_lower = mean_diff - 1.96 * pooled_se
                        ci_upper = mean_diff + 1.96 * pooled_se
                        n_total = n_pre + n_post
                    else:
                        # Paired: use difference scores
                        differences = np.array(post_vals) - np.array(pre_vals)
                        std_diff = np.std(differences, ddof=1)
                        se_diff = std_diff / np.sqrt(len(differences))
                        ci_lower = mean_diff - 1.96 * se_diff
                        ci_upper = mean_diff + 1.96 * se_diff
                        n_total = len(pre_vals)
                    
                    plot_results.append({
                        'feature': feature,
                        'mean_diff': mean_diff,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'p_value': p_value,
                        'cohen_d': cohen_d,
                        'n': n_total if is_mixed else len(pre_vals)
                    })
        
        if plot_results:
            # Create forest plot
            fig, ax = plt.subplots(figsize=(12, max(6, len(plot_results) * 0.5)))
            
            y_positions = range(len(plot_results))
            feature_names = [result['feature'] for result in plot_results]
            mean_diffs = [result['mean_diff'] for result in plot_results]
            ci_lowers = [result['ci_lower'] for result in plot_results]
            ci_uppers = [result['ci_upper'] for result in plot_results]
            p_values = [result['p_value'] for result in plot_results]
            cohen_ds = [result['cohen_d'] for result in plot_results]
            
            # Plot confidence intervals
            for i, (mean_diff, ci_lower, ci_upper, p_val, cohen_d) in enumerate(zip(mean_diffs, ci_lowers, ci_uppers, p_values, cohen_ds)):
                # Color based on effect size (handle None values)
                if cohen_d is not None:
                    if abs(cohen_d) >= 0.8:
                        color = 'red'  # Large effect
                    elif abs(cohen_d) >= 0.5:
                        color = 'orange'  # Medium effect
                    else:
                        color = 'blue'  # Small effect
                else:
                    color = 'gray'  # Unknown effect size
                
                # Plot confidence interval
                ax.plot([ci_lower, ci_upper], [i, i], color=color, linewidth=2, alpha=0.7)
                
                # Plot mean difference
                ax.scatter(mean_diff, i, color=color, s=100, alpha=0.8, zorder=5)
                
                # Add p-value and effect size as text
                if p_val is not None and cohen_d is not None:
                    ax.text(ci_upper + 0.1, i, f'p={p_val:.3f}, d={cohen_d:.2f}', 
                           va='center', fontsize=8, alpha=0.8)
                elif p_val is not None:
                    ax.text(ci_upper + 0.1, i, f'p={p_val:.3f}', 
                           va='center', fontsize=8, alpha=0.8)
                else:
                    ax.text(ci_upper + 0.1, i, 'No data', 
                           va='center', fontsize=8, alpha=0.8)
            
            # Add vertical line at 0
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
            # Customize plot
            ax.set_yticks(y_positions)
            ax.set_yticklabels(feature_names)
            ax.set_xlabel('Mean Difference (95% CI)')
            if show_significant_only:
                ax.set_title('Forest Plot: Significant Overall EEG Measurements (p < 0.05)')
            else:
                ax.set_title('Forest Plot: All Overall EEG Measurements')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', alpha=0.7, label='Large Effect (|d| ≥ 0.8)'),
                Patch(facecolor='orange', alpha=0.7, label='Medium Effect (|d| ≥ 0.5)'),
                Patch(facecolor='blue', alpha=0.7, label='Small Effect (|d| < 0.5)')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
            
            st.pyplot(fig)
            plt.close(fig)
            
            # Summary table
            if show_significant_only:
                st.markdown("### Summary of Significant Results")
            else:
                st.markdown("### Summary of All Results")
            
            summary_data = []
            for result in plot_results:
                # Handle None values for cohen_d and p_value
                cohen_d = result['cohen_d']
                p_value = result['p_value']
                
                # Determine effect size
                if cohen_d is not None:
                    if abs(cohen_d) >= 0.8:
                        effect_size = 'Large'
                    elif abs(cohen_d) >= 0.5:
                        effect_size = 'Medium'
                    else:
                        effect_size = 'Small'
                else:
                    effect_size = 'Unknown'
                
                # Format values
                p_value_str = f"{p_value:.3e}" if p_value is not None else "N/A"
                cohen_d_str = f"{cohen_d:.3f}" if cohen_d is not None else "N/A"
                significant = 'Yes' if p_value is not None and p_value < 0.05 else 'No'
                
                summary_data.append({
                    'Feature': result['feature'],
                    'Mean Difference': f"{result['mean_diff']:.3f}",
                    '95% CI': f"[{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]",
                    'P-value': p_value_str,
                    'Cohen\'s d': cohen_d_str,
                    'Effect Size': effect_size,
                    'Significant': significant,
                    'N': result['n']
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
        else:
            if show_significant_only:
                st.info("No significant measurements found (p < 0.05)")
            else:
                st.info("No measurements found")
        
        # Forest plot with median ± STD for significant features only
        st.markdown("## 🌳 Forest Plot - Significant EEG Features (Median ± STD, p < 0.05)")
        median_std_fig = create_median_std_forest_plot(pre_df, post_df, eeg_features_forest, pre_label, post_label, is_mixed, pairing_info, significant_only=True)
        if median_std_fig is not None:
            st.pyplot(median_std_fig)
            plt.close(median_std_fig)
            
            # Create summary table for median ± STD (significant features only)
            st.markdown("### Summary Table: Median ± STD (Significant Features, p < 0.05)")
            summary_median_data = []
            
            # Loop through all features to find significant ones
            for feature in eeg_features_forest:
                # Check if feature is significant
                is_significant_feature = True
                if pairing_info is not None:
                    try:
                        result = calculate_effect_size_and_power(pre_df, post_df, feature, pairing_info)
                        if result[0] is not None:
                            p_value = result[0]
                            # Mark as not significant if p >= 0.05
                            if p_value >= 0.05:
                                is_significant_feature = False
                        else:
                            # Mark as not significant if p-value cannot be calculated
                            is_significant_feature = False
                    except (KeyError, Exception):
                        # Mark as not significant if calculation fails
                        is_significant_feature = False
                else:
                    is_significant_feature = False

                if not is_significant_feature:
                    continue  # Skip non-significant features
                
                # Get paired data for this feature
                pre_subjects = set(pre_df['Subject_ID'].unique()) if len(pre_df) > 0 else set()
                post_subjects = set(post_df['Subject_ID'].unique()) if len(post_df) > 0 else set()
                common_subjects = pre_subjects.intersection(post_subjects)
                
                pre_vals = []
                post_vals = []
                for subject in common_subjects:
                    pre_val = pre_df[pre_df['Subject_ID'] == subject][feature].iloc[0] if len(pre_df[pre_df['Subject_ID'] == subject]) > 0 else np.nan
                    post_val = post_df[post_df['Subject_ID'] == subject][feature].iloc[0] if len(post_df[post_df['Subject_ID'] == subject]) > 0 else np.nan
                    
                    if not np.isnan(pre_val) and not np.isnan(post_val):
                        pre_vals.append(pre_val)
                        post_vals.append(post_val)
                
                if len(pre_vals) > 0 or len(post_vals) > 0:
                    pre_median = np.median(pre_vals) if len(pre_vals) > 0 else np.nan
                    pre_std = np.std(pre_vals, ddof=1) if len(pre_vals) > 0 else np.nan
                    post_median = np.median(post_vals) if len(post_vals) > 0 else np.nan
                    post_std = np.std(post_vals, ddof=1) if len(post_vals) > 0 else np.nan
                    
                    summary_median_data.append({
                        'Feature': feature,
                        f'{pre_label} Median': f"{pre_median:.3f}" if not np.isnan(pre_median) else "N/A",
                        f'{pre_label} STD': f"{pre_std:.3f}" if not np.isnan(pre_std) else "N/A",
                        f'{pre_label} Median ± STD': f"{pre_median:.3f} ± {pre_std:.3f}" if not (np.isnan(pre_median) or np.isnan(pre_std)) else "N/A",
                        f'{pre_label} N': len(pre_vals),
                        f'{post_label} Median': f"{post_median:.3f}" if not np.isnan(post_median) else "N/A",
                        f'{post_label} STD': f"{post_std:.3f}" if not np.isnan(post_std) else "N/A",
                        f'{post_label} Median ± STD': f"{post_median:.3f} ± {post_std:.3f}" if not (np.isnan(post_median) or np.isnan(post_std)) else "N/A",
                        f'{post_label} N': len(post_vals)
                    })
            
            if summary_median_data:
                summary_median_df = pd.DataFrame(summary_median_data)
                st.dataframe(summary_median_df, use_container_width=True)
        else:
            st.warning("Insufficient data to create median ± STD forest plot")
        
        # Paired scatterplots for significant features
        if show_significant_only:
            st.markdown("## 📊 Paired Scatterplots - Significant Features (p < 0.05)")
        else:
            st.markdown("## 📊 Paired Scatterplots - All Features")
        st.markdown("Scatterplots showing paired PRE vs POST comparisons with connecting lines. Points above the diagonal line indicate increases from PRE to POST.")
        
        with st.spinner("Generating paired scatterplots..."):
            scatterplot_figs = create_paired_scatterplots_for_significant_features(
                pre_df, post_df, eeg_features_forest, pre_label, post_label,
                pairing_info, is_mixed, significant_only=show_significant_only
            )
        
        if scatterplot_figs:
            # Display scatterplots in a grid (2 columns)
            len(scatterplot_figs)
            cols_per_row = 2
            
            for idx, (feature_name, fig) in enumerate(scatterplot_figs.items()):
                if idx % cols_per_row == 0:
                    # Create new row
                    cols = st.columns(cols_per_row)
                
                with cols[idx % cols_per_row]:
                    st.markdown(f"### {feature_name}")
                    st.pyplot(fig)
                    plt.close(fig)
        else:
            if show_significant_only:
                st.info("No significant features found for paired scatterplots (p < 0.05)")
            else:
                st.info("No features found for paired scatterplots")
        
        # Simple paired scatterplots (PRE and POST on x-axis)
        if show_significant_only:
            st.markdown("## 📈 Simple Paired Scatterplots - Significant Features (p < 0.05)")
        else:
            st.markdown("## 📈 Simple Paired Scatterplots - All Features")
        st.markdown("Simple scatterplots with PRE and POST on x-axis, showing individual patient trajectories with connecting lines.")
        
        with st.spinner("Generating simple paired scatterplots..."):
            simple_scatterplot_figs = create_simple_paired_scatterplots_for_significant_features(
                pre_df, post_df, eeg_features_forest, pre_label, post_label,
                pairing_info, is_mixed, significant_only=show_significant_only
            )
        
        if simple_scatterplot_figs:
            # Display scatterplots in a grid (2 columns)
            len(simple_scatterplot_figs)
            cols_per_row = 2
            
            for idx, (feature_name, fig) in enumerate(simple_scatterplot_figs.items()):
                if idx % cols_per_row == 0:
                    # Create new row
                    cols = st.columns(cols_per_row)
                
                with cols[idx % cols_per_row]:
                    st.markdown(f"### {feature_name}")
                    st.pyplot(fig)
                    plt.close(fig)
        else:
            if show_significant_only:
                st.info("No significant features found for simple paired scatterplots (p < 0.05)")
            else:
                st.info("No features found for simple paired scatterplots")
        
        # Topomaps for all EEG features (significant only)
        st.markdown("## 🧠 Topographic Maps - Significant EEG Features (p < 0.05)")
        st.markdown("Topomaps showing spatial distribution of significant EEG features across channels for PRE and POST groups.")
        
        with st.spinner("Generating topomaps for significant EEG features..."):
            topomap_figs = create_topomaps_for_all_features(
                pre_df, post_df, eeg_features, montage, pre_label, post_label,
                pairing_info=pairing_info, is_mixed=is_mixed, significant_only=True
            )
        
        if topomap_figs:
            # Group features by type for better organization
            feature_type_order = ['mean', 'median', 'std', 'pswe_events_per_minute', 
                                  'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power']
            
            # Display standard feature types first
            displayed_features = set()
            for feature_type in feature_type_order:
                if feature_type in topomap_figs:
                    displayed_features.add(feature_type)
                    fig, p_val = topomap_figs[feature_type]
                    st.markdown(f"### {feature_type.replace('_', ' ').title()}")
                    st.pyplot(fig)
                    plt.close(fig)
            
            # Display remaining features
            remaining_features = [f for f in topomap_figs.keys() if f not in displayed_features]
            if remaining_features:
                st.markdown("### Other Features")
                for feature in remaining_features:
                    fig, p_val = topomap_figs[feature]
                    st.markdown(f"#### {feature}")
                    st.pyplot(fig)
                    plt.close(fig)
        else:
            st.info("No significant channel-specific topomaps available (p < 0.05).")
        
        # Spectrograms
        st.markdown("## 📊 Spectrograms - Frequency vs Time")
        st.markdown("Spectrograms showing frequency content over time for representative PRE and POST recordings.")
        
        with st.spinner("Generating spectrograms..."):
            spectrogram_fig = create_spectrograms(pre_df, post_df, selected_dataset, pre_label, post_label)
        
        if spectrogram_fig is not None:
            st.pyplot(spectrogram_fig)
            plt.close(spectrogram_fig)
        else:
            st.info("Spectrograms could not be generated. EDF files may not be available or accessible.")
    else:
        st.markdown(f"## Analysis of: {selected_feature}")
        
        # Create expandable containers for different analyses
        with st.expander("📊 Descriptive Statistics", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            # Get data for this feature
            if is_mixed:
                # For mixed analysis, get all data
                pre_vals = pre_df[selected_feature].dropna().tolist()
                post_vals = post_df[selected_feature].dropna().tolist()
                group_label = ""
                change_label = "Mean Difference (POST - PRE)"
            else:
                # For paired analysis, get only paired data
                pre_subjects = set(pre_df['Subject_ID'].unique()) if len(pre_df) > 0 else set()
                post_subjects = set(post_df['Subject_ID'].unique()) if len(post_df) > 0 else set()
                common_subjects = pre_subjects.intersection(post_subjects)
                
                pre_vals = []
                post_vals = []
                for subject in common_subjects:
                    pre_val = pre_df[pre_df['Subject_ID'] == subject][selected_feature].iloc[0] if len(pre_df[pre_df['Subject_ID'] == subject]) > 0 else np.nan
                    post_val = post_df[post_df['Subject_ID'] == subject][selected_feature].iloc[0] if len(post_df[post_df['Subject_ID'] == subject]) > 0 else np.nan
                    
                    if not np.isnan(pre_val) and not np.isnan(post_val):
                        pre_vals.append(pre_val)
                        post_vals.append(post_val)
                group_label = " (Paired)"
                change_label = "Change (POST - PRE)"
            
            with col1:
                st.subheader(f"{pre_label} Group{group_label}")
                if len(pre_vals) > 0:
                    st.write(f"**N:** {len(pre_vals)}")
                    st.write(f"**Mean ± SD:** {np.mean(pre_vals):.3f} ± {np.std(pre_vals):.3f}")
                    st.write(f"**Median (IQR):** {np.median(pre_vals):.3f} ({np.percentile(pre_vals, 25):.3f} - {np.percentile(pre_vals, 75):.3f})")
                    st.write(f"**Range:** {np.min(pre_vals):.3f} - {np.max(pre_vals):.3f}")
                else:
                    st.write(f"No {pre_label} data available")
            
            with col2:
                st.subheader(f"{post_label} Group{group_label}")
                if len(post_vals) > 0:
                    st.write(f"**N:** {len(post_vals)}")
                    st.write(f"**Mean ± SD:** {np.mean(post_vals):.3f} ± {np.std(post_vals):.3f}")
                    st.write(f"**Median (IQR):** {np.median(post_vals):.3f} ({np.percentile(post_vals, 25):.3f} - {np.percentile(post_vals, 75):.3f})")
                    st.write(f"**Range:** {np.min(post_vals):.3f} - {np.max(post_vals):.3f}")
                else:
                    st.write(f"No {post_label} data available")
            
            with col3:
                st.subheader(change_label)
                if is_mixed:
                    # For mixed analysis, show mean difference
                    if len(pre_vals) > 0 and len(post_vals) > 0:
                        mean_diff = np.mean(post_vals) - np.mean(pre_vals)
                        st.write(f"**N (PRE):** {len(pre_vals)}")
                        st.write(f"**N (POST):** {len(post_vals)}")
                        st.write(f"**Mean Difference:** {mean_diff:.3f}")
                        st.write(f"**PRE Mean ± SD:** {np.mean(pre_vals):.3f} ± {np.std(pre_vals):.3f}")
                        st.write(f"**POST Mean ± SD:** {np.mean(post_vals):.3f} ± {np.std(post_vals):.3f}")
                    else:
                        st.write("Insufficient data")
                else:
                    # For paired analysis, show paired differences
                    if len(pre_vals) > 0 and len(post_vals) > 0:
                        differences = np.array(post_vals) - np.array(pre_vals)
                        st.write(f"**N:** {len(differences)}")
                        st.write(f"**Mean ± SD:** {np.mean(differences):.3f} ± {np.std(differences):.3f}")
                        st.write(f"**Median (IQR):** {np.median(differences):.3f} ({np.percentile(differences, 25):.3f} - {np.percentile(differences, 75):.3f})")
                        st.write(f"**Range:** {np.min(differences):.3f} - {np.max(differences):.3f}")
                    else:
                        st.write("No paired data available")
    
        # Statistical analysis
        if show_statistics:
            analysis_title = "📈 Statistical Analysis (Mixed)" if is_mixed else "📈 Statistical Analysis (Paired)"
            with st.expander(analysis_title, expanded=True):
                result = calculate_effect_size_and_power(pre_df, post_df, selected_feature, pairing_info)
                if result[0] is not None:
                    p_value, cohen_d, effect_size_desc, power, test_type = result
                else:
                    p_value, cohen_d, effect_size_desc, power, test_type = None, None, None, None, None
            
                if p_value is not None:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    test_name = "Mann-Whitney U" if test_type == 'unpaired' else "Wilcoxon"
                    cohen_label = "Cohen's d" if test_type == 'unpaired' else "Cohen's d (Paired)"
                    
                    with col1:
                        st.metric(f"P-value ({test_name})", f"{p_value:.3e}")
                    with col2:
                        st.metric(cohen_label, f"{cohen_d:.3f}")
                    with col3:
                        st.metric("Effect Size", effect_size_desc.title())
                    with col4:
                        st.metric("Statistical Power", f"{power:.1%}" if not np.isnan(power) else "N/A")
                    
                    # Significance interpretation
                    if p_value < 0.001:
                        sig_text = "*** (p < 0.001)"
                    elif p_value < 0.01:
                        sig_text = "** (p < 0.01)"
                    elif p_value < 0.05:
                        sig_text = "* (p < 0.05)"
                    else:
                        sig_text = "ns (p ≥ 0.05)"
                    
                    st.write(f"**Significance ({test_name} test):** {sig_text}")
                    
                    # Effect size interpretation
                    st.write(f"**Effect Size Interpretation:** {effect_size_desc.title()} effect size (Cohen's d = {cohen_d:.3f})")
                    
                    # Additional info about analysis type
                    if is_mixed:
                        st.info("This analysis uses unpaired comparisons (Mann-Whitney U test) for all data, including both paired and unpaired subjects.")
                    else:
                        st.info("This analysis uses paired comparisons (Wilcoxon signed-rank test) for subjects with both PRE and POST data.")
                else:
                    st.warning("Insufficient data for statistical analysis")
    
        # Dabest plots
        if show_dabest:
            with st.expander("🎯 Dabest Estimation Plots", expanded=True):
                fig, p_val, cohen_d = create_dabest_plot(pre_df, post_df, selected_feature)
                if fig is not None:
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning("Insufficient data for Dabest plot")
        
        # Topomaps or Simple Comparison Plots
        if show_topomaps:
            with st.expander("🧠 Topographic Maps", expanded=True):
                # Calculate p-value for topomap title
                result = calculate_effect_size_and_power(pre_df, post_df, selected_feature, pairing_info)
                topomap_p_value = result[0] if result[0] is not None else None
                
                if 'overall_' in selected_feature:
                    # For overall features, show simple comparison plots
                    fig = create_simple_comparison_plot(pre_df, post_df, selected_feature)
                    if fig is not None:
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.warning("Insufficient data for comparison plot")
                else:
                    # For channel-specific features, show topomaps
                    fig, pre_evoked, post_evoked = create_topomap_comparison(
                        pre_df, post_df, selected_feature, montage, p_value=topomap_p_value
                    )
                    if fig is not None:
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.warning("Insufficient channel data for topomap generation")
    
        # Distribution plots
        with st.expander("📈 Distribution Comparison (Paired)", expanded=True):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get paired data for this feature
            pre_subjects = set(pre_df['Subject_ID'].unique()) if len(pre_df) > 0 else set()
            post_subjects = set(post_df['Subject_ID'].unique()) if len(post_df) > 0 else set()
            common_subjects = pre_subjects.intersection(post_subjects)
            
            pre_vals = []
            post_vals = []
            for subject in common_subjects:
                pre_val = pre_df[pre_df['Subject_ID'] == subject][selected_feature].iloc[0] if len(pre_df[pre_df['Subject_ID'] == subject]) > 0 else np.nan
                post_val = post_df[post_df['Subject_ID'] == subject][selected_feature].iloc[0] if len(post_df[post_df['Subject_ID'] == subject]) > 0 else np.nan
                
                if not np.isnan(pre_val) and not np.isnan(post_val):
                    pre_vals.append(pre_val)
                    post_vals.append(post_val)
            
            if len(pre_vals) > 0:
                ax.hist(pre_vals, alpha=0.7, label='PRE (Paired)', bins=20, density=True)
            
            if len(post_vals) > 0:
                ax.hist(post_vals, alpha=0.7, label='POST (Paired)', bins=20, density=True)
            
            ax.set_xlabel(selected_feature)
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution Comparison: {selected_feature} (Paired Data)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close(fig)
        
        # Individual change plot with connecting lines
        with st.expander("📊 Individual Changes (PRE → POST)", expanded=True):
            individual_fig = create_individual_change_plot(
                pre_df, post_df, selected_feature, pre_label, post_label, pairing_info
            )
            if individual_fig is not None:
                st.pyplot(individual_fig)
                plt.close(individual_fig)
            else:
                st.warning("Insufficient paired data for individual change plot")
        
        # Box plots
        with st.expander("📦 Box Plot Comparison (Paired)", expanded=True):
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Get paired data for this feature
            pre_subjects = set(pre_df['Subject_ID'].unique()) if len(pre_df) > 0 else set()
            post_subjects = set(post_df['Subject_ID'].unique()) if len(post_df) > 0 else set()
            common_subjects = pre_subjects.intersection(post_subjects)
            
            pre_vals = []
            post_vals = []
            for subject in common_subjects:
                pre_val = pre_df[pre_df['Subject_ID'] == subject][selected_feature].iloc[0] if len(pre_df[pre_df['Subject_ID'] == subject]) > 0 else np.nan
                post_val = post_df[post_df['Subject_ID'] == subject][selected_feature].iloc[0] if len(post_df[post_df['Subject_ID'] == subject]) > 0 else np.nan
                
                if not np.isnan(pre_val) and not np.isnan(post_val):
                    pre_vals.append(pre_val)
                    post_vals.append(post_val)
            
            data_for_box = []
            labels = []
            
            if len(pre_vals) > 0:
                data_for_box.append(pre_vals)
                labels.append('PRE (Paired)')
            
            if len(post_vals) > 0:
                data_for_box.append(post_vals)
                labels.append('POST (Paired)')
            
            if data_for_box:
                # Create box plot
                bp = ax.boxplot(data_for_box, labels=labels, patch_artist=True)
                
                # Color the boxes
                colors = ['lightblue', 'lightcoral']
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Add individual data points with connecting lines
                if len(pre_vals) == len(post_vals) and len(pre_vals) > 0:
                    # Boxplot positions are 1-indexed
                    x_pre = 1
                    x_post = 2
                    for i in range(len(pre_vals)):
                        # Draw connecting line
                        ax.plot([x_pre, x_post], [pre_vals[i], post_vals[i]], 
                               color='gray', linewidth=0.8, alpha=0.3, zorder=1)
                        # Add individual points
                        ax.scatter(x_pre, pre_vals[i], color='blue', s=25, 
                                  alpha=0.6, zorder=2, edgecolors='darkblue', linewidths=0.5)
                        ax.scatter(x_post, post_vals[i], color='red', s=25, 
                                  alpha=0.6, zorder=2, edgecolors='darkred', linewidths=0.5)
                
                ax.set_ylabel(selected_feature)
                ax.set_title(f'Box Plot Comparison: {selected_feature} (Paired Data)')
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                plt.close(fig)

if __name__ == "__main__":
    main()
