import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
from utils.eeg_utils import *
import mne
from scipy import stats
from statsmodels.stats.multitest import multipletests
import dabest
from statsmodels.stats.power import TTestIndPower
import warnings
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
    except:
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

def create_topomap_comparison(pre_data, post_data, feature, montage):
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
    
    # PRE topomap
    im1, _ = mne.viz.plot_topomap(pre_evoked.data[:, 0], pre_evoked.info, 
                                  axes=axes[0], show=False, vlim=(vmin, vmax))
    axes[0].set_title('PRE')
    
    # POST topomap
    im2, _ = mne.viz.plot_topomap(post_evoked.data[:, 0], post_evoked.info, 
                                  axes=axes[1], show=False, vlim=(vmin, vmax))
    axes[1].set_title('POST')
    
    # Difference topomap
    im3, _ = mne.viz.plot_topomap(diff_evoked.data[:, 0], diff_evoked.info, 
                                  axes=axes[2], show=False)
    axes[2].set_title('POST - PRE')
    
    # Add colorbars
    plt.colorbar(im1, ax=axes[0])
    plt.colorbar(im2, ax=axes[1])
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    return fig, pre_evoked, post_evoked

def create_topomaps_for_all_features(pre_df, post_df, eeg_features, montage, pre_label, post_label):
    """
    Create topomaps for all EEG features that have channel-specific data.
    
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
    
    Returns
    -------
    dict : Dictionary mapping feature names to figure objects
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
            # Create topomap for this feature
            fig, _, _ = create_topomap_comparison(pre_df, post_df, feature_type, montage)
            if fig is not None:
                topomap_figs[feature_type] = fig
    
    # Also process individual features from eeg_features list
    for feature in eeg_features:
        # Skip if we already processed it as a feature type
        if feature in topomap_figs:
            continue
        
        # Check if feature has channel-specific data
        pre_vals = extract_channel_values_for_feature(pre_df, feature)
        post_vals = extract_channel_values_for_feature(post_df, feature)
        
        if len(pre_vals) >= 3 or len(post_vals) >= 3:
            fig, _, _ = create_topomap_comparison(pre_df, post_df, feature, montage)
            if fig is not None:
                topomap_figs[feature] = fig
    
    return topomap_figs

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

def create_median_std_forest_plot(pre_df, post_df, eeg_features, pre_label, post_label, is_mixed, pairing_info):
    """
    Create a forest plot showing median ± STD for all EEG features.
    
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
        except:
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
        except:
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
        except:
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
        except:
            p_value = np.nan
            test_type = 'paired'
        
        # Cohen's d for paired data (using difference scores)
        try:
            differences = np.array(post_vals) - np.array(pre_vals)
            cohen_d = np.mean(differences) / np.std(differences, ddof=1)
        except:
            cohen_d = np.nan
        
        # Statistical power for paired t-test
        try:
            from statsmodels.stats.power import TTestPower
            analysis = TTestPower()
            power = analysis.power(effect_size=abs(cohen_d) if not np.isnan(cohen_d) else 0, 
                                  nobs=len(pre_vals), 
                                  alpha=0.05, 
                                  alternative='two-sided')
        except:
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
        value=False,
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
        paired_subjects = pairing_info.get('paired', set())
        unpaired_pre = pairing_info.get('unpaired_pre', set())
        unpaired_post = pairing_info.get('unpaired_post', set())
        info_text = f"Analysis includes:\n"
        info_text += f"- {pairing_info.get('paired_count', 0)} paired subjects (both PRE and POST)\n"
        if pairing_info.get('unpaired_pre_count', 0) > 0:
            info_text += f"- {pairing_info.get('unpaired_pre_count', 0)} unpaired PRE subjects\n"
        if pairing_info.get('unpaired_post_count', 0) > 0:
            info_text += f"- {pairing_info.get('unpaired_post_count', 0)} unpaired POST subjects\n"
        info_text += f"\nUsing unpaired statistical tests (Mann-Whitney U) for all data."
        st.info(info_text)
    elif pairing_info:
        st.info(f"Analysis includes {len(pairing_info)} subjects with both PRE and POST data: {', '.join(sorted(list(pairing_info)[:10]))}{'...' if len(pairing_info) > 10 else ''}")
    else:
        st.error("No subjects found with both PRE and POST data!")
        return
    
    # Get EEG features
    eeg_features = get_eeg_features(pre_df)
    
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
    
    # Check if "All Features" is selected
    if selected_feature == "All Features":
        st.markdown("## Analysis of: All EEG Features")
        
        # Create a grid layout for all features
        num_features = len(eeg_features)
        cols_per_row = 3  # Number of columns per row
        num_rows = (num_features + cols_per_row - 1) // cols_per_row
        
        for row in range(num_rows):
            start_idx = row * cols_per_row
            end_idx = min(start_idx + cols_per_row, num_features)
            current_features = eeg_features[start_idx:end_idx]
            
            # Create columns for this row
            cols = st.columns(len(current_features))
            
            for i, feature in enumerate(current_features):
                with cols[i]:
                    st.markdown(f"### {feature}")
                    
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
                    
                    # Show basic statistics
                    if len(pre_vals) > 0 and len(post_vals) > 0:
                        # Calculate difference/change based on analysis type
                        if is_mixed:
                            mean_diff = np.mean(post_vals) - np.mean(pre_vals)
                            differences_for_display = [mean_diff]  # Just for consistency
                        else:
                            differences = np.array(post_vals) - np.array(pre_vals)
                            differences_for_display = differences
                        
                        # Statistical analysis
                        result = calculate_effect_size_and_power(pre_df, post_df, feature, pairing_info)
                        if result[0] is not None:
                            p_value, cohen_d, effect_size_desc, power, test_type = result
                        else:
                            p_value, cohen_d, effect_size_desc, power, test_type = None, None, None, None, None
                        
                        # Show results based on user selection
                        if show_significant_only:
                            # Only show significant results
                            if p_value is not None and p_value < 0.05:
                                st.write(f"**N ({pre_label}):** {len(pre_vals)}")
                                st.write(f"**N ({post_label}):** {len(post_vals)}")
                                st.write(f"**{pre_label}:** {np.mean(pre_vals):.3f} ± {np.std(pre_vals):.3f}")
                                st.write(f"**{post_label}:** {np.mean(post_vals):.3f} ± {np.std(post_vals):.3f}")
                                if is_mixed:
                                    st.write(f"**Mean Diff:** {np.mean(post_vals) - np.mean(pre_vals):.3f}")
                                else:
                                    st.write(f"**Change:** {np.mean(differences_for_display):.3f} ± {np.std(differences_for_display):.3f}")
                                st.write(f"**p:** {p_value:.3e} *")
                                st.write(f"**d:** {cohen_d:.3f}")
                                
                                # Create a small comparison plot
                                fig, ax = plt.subplots(figsize=(4, 3))
                                
                                data_for_box = [pre_vals, post_vals]
                                labels = [pre_label, post_label]
                                
                                ax.boxplot(data_for_box, labels=labels)
                                ax.set_ylabel(feature)
                                ax.set_title(f'{feature}')
                                ax.grid(True, alpha=0.3)
                                
                                st.pyplot(fig)
                                plt.close(fig)
                            else:
                                st.write("**p ≥ 0.05** - Not significant")
                        else:
                            # Show all results
                            st.write(f"**N ({pre_label}):** {len(pre_vals)}")
                            st.write(f"**N ({post_label}):** {len(post_vals)}")
                            st.write(f"**{pre_label}:** {np.mean(pre_vals):.3f} ± {np.std(pre_vals):.3f}")
                            st.write(f"**{post_label}:** {np.mean(post_vals):.3f} ± {np.std(post_vals):.3f}")
                            if is_mixed:
                                st.write(f"**Mean Diff:** {np.mean(post_vals) - np.mean(pre_vals):.3f}")
                            else:
                                st.write(f"**Change:** {np.mean(differences_for_display):.3f} ± {np.std(differences_for_display):.3f}")
                            
                            if p_value is not None:
                                if p_value < 0.05:
                                    st.write(f"**p:** {p_value:.3e} *")
                                else:
                                    st.write(f"**p:** {p_value:.3e}")
                                st.write(f"**d:** {cohen_d:.3f}")
                            else:
                                st.write("**Insufficient data**")
                            
                            # Create a small comparison plot
                            fig, ax = plt.subplots(figsize=(4, 3))
                            
                            data_for_box = [pre_vals, post_vals]
                            labels = [pre_label, post_label]
                            
                            ax.boxplot(data_for_box, labels=labels)
                            ax.set_ylabel(feature)
                            ax.set_title(f'{feature}')
                            ax.grid(True, alpha=0.3)
                            
                            st.pyplot(fig)
                            plt.close(fig)
                    else:
                        st.write("**No paired data**")
        
        # Forest plot for measurements
        if show_significant_only:
            st.markdown("## 🌲 Forest Plot - Significant Measurements (p < 0.05)")
        else:
            st.markdown("## 🌲 Forest Plot - All Measurements")
        
        # Collect results based on user selection
        plot_results = []
        for feature in eeg_features:
            # For mixed analysis, get all data; for paired-only, get only paired
            if is_mixed:
                # Get all PRE and POST values (paired + unpaired)
                pre_vals = pre_df[feature].dropna().tolist()
                post_vals = post_df[feature].dropna().tolist()
            else:
                # Get only paired data
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
            ax.set_title('Forest Plot: Significant EEG Measurements (p < 0.05)')
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
        
        # Forest plot with median ± STD for all features
        st.markdown("## 🌳 Forest Plot - All EEG Features (Median ± STD)")
        median_std_fig = create_median_std_forest_plot(pre_df, post_df, eeg_features, pre_label, post_label, is_mixed, pairing_info)
        if median_std_fig is not None:
            st.pyplot(median_std_fig)
            plt.close(median_std_fig)
            
            # Create summary table for median ± STD
            st.markdown("### Summary Table: Median ± STD")
            summary_median_data = []
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
        
        # Topomaps for all EEG features
        st.markdown("## 🧠 Topographic Maps - All EEG Features")
        st.markdown("Topomaps showing spatial distribution of EEG features across channels for PRE and POST groups.")
        
        with st.spinner("Generating topomaps for all EEG features..."):
            topomap_figs = create_topomaps_for_all_features(pre_df, post_df, eeg_features, montage, pre_label, post_label)
        
        if topomap_figs:
            # Group features by type for better organization
            feature_type_order = ['mean', 'median', 'std', 'pswe_events_per_minute', 
                                  'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power']
            
            # Display standard feature types first
            displayed_features = set()
            for feature_type in feature_type_order:
                if feature_type in topomap_figs:
                    displayed_features.add(feature_type)
                    st.markdown(f"### {feature_type.replace('_', ' ').title()}")
                    st.pyplot(topomap_figs[feature_type])
                    plt.close(topomap_figs[feature_type])
            
            # Display remaining features
            remaining_features = [f for f in topomap_figs.keys() if f not in displayed_features]
            if remaining_features:
                st.markdown("### Other Features")
                for feature in remaining_features:
                    st.markdown(f"#### {feature}")
                    st.pyplot(topomap_figs[feature])
                    plt.close(topomap_figs[feature])
        else:
            st.info("No channel-specific topomaps available. Feature data may not contain channel-specific measurements.")
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
                    fig, pre_evoked, post_evoked = create_topomap_comparison(pre_df, post_df, selected_feature, montage)
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
                ax.boxplot(data_for_box, labels=labels)
                ax.set_ylabel(selected_feature)
                ax.set_title(f'Box Plot Comparison: {selected_feature} (Paired Data)')
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                plt.close(fig)

if __name__ == "__main__":
    main()
