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

def load_pre_post_data():
    """Load pre and post data from parquet files and determine group membership from EDF directory structure."""
    
    # Load all parquet files
    parquet_dir = "parquet_results/VNS_PRE_POST_25"
    parquet_files = glob.glob(os.path.join(parquet_dir, "*.parquet"))
    
    pre_data = []
    post_data = []
    file_map = {}
    # EDF directory structure for determining pre/post
    edf_base_dir = "EDF Format/VNS_PRE_POST_25"
    
    for parquet_file in parquet_files:
        try:
            # Extract filename without extension
            filename = os.path.basename(parquet_file).replace('.edf.parquet', '')
            
            # Load parquet data
            df = pd.read_parquet(parquet_file)
            
            # Determine if this is pre or post based on EDF directory structure
            # Build a mapping from filename to (group, Subject_ID) for all files in edf_base_dir
                
            for root, dirs, files in os.walk(edf_base_dir):
                for file in files:
                    if filename in file:
                        key = file.split('.')[0]  # Remove extension for matching
                        group = root.split(os.sep)[-1].upper()
                        subject_id = root.split(os.sep)[-2]
                        file_map[key] = (group, subject_id)
                        print(f"Mapping file {key} to group {group}, subject {subject_id}. Filename: {filename}, file: {file}")
                        break
                # if key in file_map:
                if filename in file_map:
                    break
            load_pre_post_data._file_map = file_map

            # Try to match filename in the file_map
            group_subject = file_map.get(filename, (False, None))
            is_post, Subject_ID = group_subject
            
            # Add group label
            df['Group'] = is_post
            df['Subject_ID'] = Subject_ID
            
            if is_post == "POST":
                post_data.append(df)
            else:
                pre_data.append(df)
            # print how many in pre and post
        except Exception as e:
            st.warning(f"Could not load {parquet_file}: {e}")
            continue
    print(f"Loaded {len(pre_data)} PRE and {len(post_data)} POST files so far.")
    if not pre_data and not post_data:
        st.error("No data found!")
        return None, None, None
    
    pre_df = pd.concat(pre_data, ignore_index=True) if pre_data else pd.DataFrame()
    post_df = pd.concat(post_data, ignore_index=True) if post_data else pd.DataFrame()
    # mean over Subject_ID for numeric columns, first for non-numeric columns
    if len(pre_df) > 0:
        numeric_cols = pre_df.select_dtypes(include=np.number).columns.tolist()
        non_numeric_cols = [col for col in pre_df.columns if col not in numeric_cols and col != 'Subject_ID']
        pre_df = pre_df.groupby('Subject_ID').agg({**{col: 'mean' for col in numeric_cols}, **{col: 'first' for col in non_numeric_cols}}).reset_index()
    if len(post_df) > 0:
        numeric_cols = post_df.select_dtypes(include=np.number).columns.tolist()
        non_numeric_cols = [col for col in post_df.columns if col not in numeric_cols and col != 'Subject_ID']
        post_df = post_df.groupby('Subject_ID').agg({**{col: 'mean' for col in numeric_cols}, **{col: 'first' for col in non_numeric_cols}}).reset_index()
    # Find patients with both PRE and POST data
    pre_subjects = set(pre_df['Subject_ID'].unique()) if len(pre_df) > 0 else set()
    post_subjects = set(post_df['Subject_ID'].unique()) if len(post_df) > 0 else set()
    paired_subjects = pre_subjects.intersection(post_subjects)
    
    # Filter to only include paired subjects
    if len(paired_subjects) > 0:
        pre_paired = pre_df[pre_df['Subject_ID'].isin(paired_subjects)].copy()
        post_paired = post_df[post_df['Subject_ID'].isin(paired_subjects)].copy()
        
        # Sort by Subject_ID to ensure proper pairing
        pre_paired = pre_paired.sort_values('Subject_ID').reset_index(drop=True)
        post_paired = post_paired.sort_values('Subject_ID').reset_index(drop=True)
        print(f'len pre_paired: {len(pre_paired)}, len post_paired: {len(post_paired)}')
        return pre_paired, post_paired, paired_subjects
    else:
        st.warning("No patients found with both PRE and POST data!")
        return None, None, None

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

def create_topomap_comparison(pre_data, post_data, feature, montage):
    """Create topomap comparison for pre vs post groups."""
    
    # For overall features, we can't create channel-specific topomaps
    # Instead, we'll create a simple comparison plot
    if 'overall_' in feature:
        return None, None, None
    
    # Extract channel data for the feature
    # Look for columns that contain the feature name and channel information
    feature_cols = [col for col in pre_data.columns if feature in col and any(ch in col for ch in eeg_channels)]
    
    if not feature_cols:
        return None, None, None
    
    # Create topomaps for each group
    pre_vals = []
    post_vals = []
    ch_names = []
    
    for col in feature_cols:
        # Extract channel name from column name
        # This is a simplified approach - you may need to adjust based on your column naming convention
        ch_name = col.split('_')[-1] if '_' in col else col
        if ch_name in eeg_channels:
            ch_names.append(ch_name)
            pre_vals.append(pre_data[col].mean() if len(pre_data) > 0 else 0)
            post_vals.append(post_data[col].mean() if len(post_data) > 0 else 0)
    
    if len(ch_names) < 3:  # Need at least 3 channels for topomap
        return None, None, None
    
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

def calculate_effect_size_and_power(pre_data, post_data, feature):
    """Calculate effect size and statistical power for paired data."""
    
    # Ensure we have the same subjects in both datasets
    pre_subjects = set(pre_data['Subject_ID'].unique()) if len(pre_data) > 0 else set()
    post_subjects = set(post_data['Subject_ID'].unique()) if len(post_data) > 0 else set()
    common_subjects = pre_subjects.intersection(post_subjects)
    
    if len(common_subjects) < 2:
        return None, None, None, None
    
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
        return None, None, None, None
    
    # Wilcoxon signed-rank test for paired data
    try:
        stat, p_value = stats.wilcoxon(post_vals, pre_vals, alternative='two-sided')
    except:
        p_value = np.nan
    
    # Cohen's d for paired data (using difference scores)
    try:
        differences = np.array(post_vals) - np.array(pre_vals)
        cohen_d = np.mean(differences) / np.std(differences, ddof=1)
    except:
        cohen_d = np.nan
    
    # Effect size interpretation
    if abs(cohen_d) < 0.2:
        effect_size_desc = "negligible"
    elif abs(cohen_d) < 0.5:
        effect_size_desc = "small"
    elif abs(cohen_d) < 0.8:
        effect_size_desc = "medium"
    else:
        effect_size_desc = "large"
    
    # Statistical power for paired t-test
    try:
        from statsmodels.stats.power import TTestPower
        analysis = TTestPower()
        power = analysis.power(effect_size=abs(cohen_d), 
                              nobs=len(pre_vals), 
                              alpha=0.05, 
                              alternative='two-sided')
    except:
        power = np.nan
    
    return p_value, cohen_d, effect_size_desc, power

def main():
    st.set_page_config(page_title="VNS Pre/Post EEG Comparison", layout="wide")
    
    st.title("VNS Pre/Post EEG Comparison Analysis")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading pre/post data..."):
        pre_df, post_df, paired_subjects = load_pre_post_data()
    
    if pre_df is None and post_df is None:
        st.error("Failed to load data. Please check the data directories.")
        return
    
    # Display data summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("PRE Sessions", len(pre_df) if len(pre_df) > 0 else 0)
    with col2:
        st.metric("POST Sessions", len(post_df) if len(post_df) > 0 else 0)
    with col3:
        st.metric("Paired Subjects", len(paired_subjects) if paired_subjects else 0)
    
    if len(pre_df) == 0 or len(post_df) == 0:
        st.warning("Insufficient data for comparison. Need both PRE and POST data.")
        return
    
    # Show paired subjects info
    if paired_subjects:
        st.info(f"Analysis includes {len(paired_subjects)} subjects with both PRE and POST data: {', '.join(sorted(paired_subjects))}")
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
    selected_feature = st.sidebar.selectbox("Select EEG Feature", eeg_features)
    
    # Analysis options
    show_dabest = st.sidebar.checkbox("Show Dabest Plots", value=True)
    show_topomaps = st.sidebar.checkbox("Show Topomaps", value=True)
    show_statistics = st.sidebar.checkbox("Show Detailed Statistics", value=True)
    
    st.markdown(f"## Analysis of: {selected_feature}")
    
    # Create expandable containers for different analyses
    with st.expander("📊 Descriptive Statistics", expanded=True):
        col1, col2, col3 = st.columns(3)
        
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
        
        with col1:
            st.subheader("PRE Group (Paired)")
            if len(pre_vals) > 0:
                st.write(f"**N:** {len(pre_vals)}")
                st.write(f"**Mean ± SD:** {np.mean(pre_vals):.3f} ± {np.std(pre_vals):.3f}")
                st.write(f"**Median (IQR):** {np.median(pre_vals):.3f} ({np.percentile(pre_vals, 25):.3f} - {np.percentile(pre_vals, 75):.3f})")
                st.write(f"**Range:** {np.min(pre_vals):.3f} - {np.max(pre_vals):.3f}")
            else:
                st.write("No paired PRE data available")
        
        with col2:
            st.subheader("POST Group (Paired)")
            if len(post_vals) > 0:
                st.write(f"**N:** {len(post_vals)}")
                st.write(f"**Mean ± SD:** {np.mean(post_vals):.3f} ± {np.std(post_vals):.3f}")
                st.write(f"**Median (IQR):** {np.median(post_vals):.3f} ({np.percentile(post_vals, 25):.3f} - {np.percentile(post_vals, 75):.3f})")
                st.write(f"**Range:** {np.min(post_vals):.3f} - {np.max(post_vals):.3f}")
            else:
                st.write("No paired POST data available")
        
        with col3:
            st.subheader("Change (POST - PRE)")
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
        with st.expander("📈 Statistical Analysis (Paired)", expanded=True):
            p_value, cohen_d, effect_size_desc, power = calculate_effect_size_and_power(pre_df, post_df, selected_feature)
            
            if p_value is not None:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("P-value (Wilcoxon)", f"{p_value:.3e}")
                with col2:
                    st.metric("Cohen's d (Paired)", f"{cohen_d:.3f}")
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
                
                st.write(f"**Significance (Wilcoxon signed-rank test):** {sig_text}")
                
                # Effect size interpretation
                st.write(f"**Effect Size Interpretation:** {effect_size_desc.title()} effect size (Cohen's d = {cohen_d:.3f})")
                
                # Additional info about paired analysis
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
