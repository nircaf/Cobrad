import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.eeg_utils import *

# Suppress PyDev debugger warnings for large data structures
import os
os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '10.0'
import mne
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.stats.multitest as smm
from scipy.signal import spectrogram
import statsmodels.api as sm
from collections import Counter
import time

# Set plotting styles as specified
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


# plt.style.use('science')
# make {figures_dir} prettier
sns.set_context('talk')
sns.set_style('white')
# put grid in all {figures_dir}
plt.rcParams['axes.grid'] = True
# add ticks to both sides 
plt.rc('xtick', bottom   = True)
plt.rc('ytick', left = True)
plt.rc('font',  family='serif',)
plt.rc('text',  usetex=False)
# make labels slightly smaller 
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
plt.rc('axes',  labelsize=11)
plt.rc('legend',  handlelength=4.0)
plt.rc('axes',  titlesize=12)  # Set title size to be the same as x and y labels
montage = mne.channels.make_standard_montage('standard_1020')


def plot_electrode_importance_topomap(electrode_importance_dict, title="Electrode Importance Topomap"):
    """
    Plot electrode importance values as an MNE topomap.
    
    Parameters:
    -----------
    electrode_importance_dict : dict
        Dictionary mapping electrode names to importance values
    title : str
        Title for the plot
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object, or None if insufficient channels
    """
    # Filter to only electrodes with non-zero importance and that exist in eeg_channels
    available_channels = []
    values = []
    
    for electrode, importance in electrode_importance_dict.items():
        if electrode in eeg_channels and importance > 0:
            available_channels.append(electrode)
            values.append(importance)
    
    # Need at least 3 channels for a meaningful topomap
    if len(available_channels) < 3:
        return None
    
    try:
        # Create MNE info object
        info = mne.create_info(ch_names=available_channels, sfreq=256, ch_types='eeg')
        info.set_montage(montage)
        
        # Create EvokedArray with importance values
        values_array = np.array(values).reshape(-1, 1)
        evoked = mne.EvokedArray(values_array, info)
        
        # Plot topomap
        fig, ax = plt.subplots(figsize=(8, 6))
        im, _ = mne.viz.plot_topomap(
            evoked.data[:, 0], 
            evoked.info, 
            axes=ax, 
            show=False,
            cmap='Reds',
            vlim=(0, max(values)) if values else None
        )
        fig.colorbar(im, ax=ax, label='Importance')
        ax.set_title(title)
        plt.tight_layout()
        
        return fig
    except Exception as e:
        # If there's an error (e.g., channel positions not found), return None
        return None


def plot_electrode_pvalue_topomap(electrode_importance_dict, title="Electrode P-Value Topomap"):
    """
    Plot p-values comparing each electrode's importance to all others as an MNE topomap.
    
    For each electrode, compares its importance value to all other electrodes' values
    using a statistical test (Mann-Whitney U or one-sample test).
    
    Parameters:
    -----------
    electrode_importance_dict : dict
        Dictionary mapping electrode names to importance values
    title : str
        Title for the plot
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object, or None if insufficient channels
    """
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
    
    # Filter to only electrodes with non-zero importance and that exist in eeg_channels
    available_channels = []
    values = []
    
    for electrode, importance in electrode_importance_dict.items():
        if electrode in eeg_channels and importance >= 0:  # Include zero for comparison
            available_channels.append(electrode)
            values.append(importance)
    
    # Need at least 3 channels for meaningful comparison
    if len(available_channels) < 3:
        return None
    
    # Calculate p-values: for each electrode, compare to all others
    p_values = []
    all_values = np.array(values)
    
    for i, electrode_val in enumerate(values):
        # Get all other values
        other_values = np.delete(all_values, i)
        
        if len(other_values) < 2:
            p_values.append(1.0)
            continue
        
        # Compare this electrode's value to all others
        # Use Mann-Whitney U test (non-parametric)
        try:
            # Create two groups: [this electrode] vs [all others]
            # For single value comparison, use a one-sample approach
            # Compare this value to the distribution of others
            if electrode_val == 0 and np.all(other_values == 0):
                p_values.append(1.0)
            else:
                # Use a permutation-like approach: how many others are >= this value
                # Or use a one-sample test comparing to the mean of others
                from scipy.stats import mannwhitneyu
                # Create a small sample for comparison (repeat the value a few times for valid test)
                this_electrode_array = np.array([electrode_val] * min(3, len(other_values)))
                stat, p = mannwhitneyu(this_electrode_array, other_values, alternative='two-sided')
                p_values.append(p)
        except Exception:
            # Fallback: simple comparison based on rank
            # How extreme is this value compared to others?
            combined = np.concatenate([[electrode_val], other_values])
            rank = np.sum(combined <= electrode_val)
            p = 2 * min(rank / len(combined), 1 - rank / len(combined))
            p_values.append(p)
    
    # Apply multiple comparison correction (FDR)
    try:
        _, p_corrected, _, _ = multipletests(p_values, method='fdr_bh', alpha=0.05)
    except:
        p_corrected = np.array(p_values)
    
    try:
        # Create MNE info object
        info = mne.create_info(ch_names=available_channels, sfreq=256, ch_types='eeg')
        info.set_montage(montage)
        
        # Create EvokedArray with p-values
        p_array = np.array(p_corrected).reshape(-1, 1)
        evoked = mne.EvokedArray(p_array, info)
        
        # Plot topomap
        fig, ax = plt.subplots(figsize=(8, 6))
        # Use reversed colormap: lower p-values (more significant) shown in brighter colors
        vlim_max = min(0.05, max(p_corrected)) if len(p_corrected) > 0 else 0.05
        im, _ = mne.viz.plot_topomap(
            evoked.data[:, 0], 
            evoked.info, 
            axes=ax, 
            show=False,
            cmap='Reds_r',  # Reversed: red = significant (low p-value)
            vlim=(0, vlim_max)
        )
        fig.colorbar(im, ax=ax, label='P-value (FDR corrected)')
        ax.set_title(title)
        plt.tight_layout()
        
        return fig
    except Exception as e:
        # If there's an error, return None
        return None


def multiselect_pairplot(all_features):
    # groups are based on common split('_')[0] of all features
    feature_groups = [col.split('_')[0] for col in all_features]
    # Count occurrences of each group
    group_counts = Counter(feature_groups)
    # Separate groups with at least 8 occurrences
    valid_groups = {group for group, count in group_counts.items() if count >= 8}
    columns = []
    for group in valid_groups:
        # let sidebar multiselect for group
        st.sidebar.write(f"Group: {group}")
        group_columns = [col for col in all_features if col.startswith(group)]
        columns.extend(st.sidebar.multiselect("Select features for pairplot:", group_columns))
    return columns

def pairplot_columns(df, clinical_features, eeg_features, hue=None, output_dir=None):
    """
    Creates a pairplot and scatterplots for the specified columns in the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        columns (list): List of column names to include in the pairplot.
        hue (str, optional): Column name to use for color encoding (e.g., 'Group').
        output_dir (str, optional): Directory to save the pairplot image. If None, the plot is displayed in Streamlit.

    Returns:
        None
    """
    st.sidebar.subheader("Select Clinical Features")
    columns = multiselect_pairplot(clinical_features)
    st.sidebar.subheader("Select EEG Features")
    columns.extend(multiselect_pairplot(eeg_features))
    if columns:
        # Ensure the columns exist in the DataFrame
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The following columns are missing in the DataFrame: {missing_columns}")

        # Drop rows with NaN values in the specified columns
        df_cleaned = df[columns + ([hue] if hue else [])].dropna()

        # Create the pairplot
        pairplot = sns.pairplot(df_cleaned, hue=hue, diag_kind='kde')
        
        # Lower pairplot add scatterplot
        pairplot.map_lower(sns.scatterplot, alpha=0.5)
        
        # Add regression line to the lower triangle
        for i in range(len(columns)):
            for j in range(i):
                # Extract the correct data for x and y
                x = df_cleaned[columns[j]]
                y = df_cleaned[columns[i]]
                
                # Add regression line to the lower triangle
                sns.regplot(x=x, y=y, ax=pairplot.axes[i, j], scatter=False, color='red', line_kws={'alpha': 0.5})
                
                # Calculate R^2 and p-value
                X = sm.add_constant(x)
                model = sm.OLS(y, X).fit()
                r_squared = model.rsquared
                p_value = model.pvalues[1]  # p-value for the slope
                
                # Annotate the plot with R^2 and p-value
                pairplot.axes[i, j].annotate(
                    f"$R^2$={r_squared:.2f}\n$p$={p_value:.2e}",
                    xy=(0.05, 0.95),
                    xycoords='axes fraction',
                    ha='left',
                    va='top',
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white')
                )
        
        # Add a title and adjust layout to prevent text cutoff
        pairplot.fig.suptitle("Pairplot of Selected Columns", y=1.02)
        pairplot.fig.tight_layout()  # Automatically adjust subplots to fit within the figure area
        pairplot.fig.subplots_adjust(top=0.95)  # Add extra space at the top for the title
        
        # Display the pairplot in Streamlit
        st_pyplot_func(pairplot.fig,filename=f'pairplot_{"_".join(columns)}.png')

        # Display scatterplots for each x column with all y columns on one plot
        st.header("Scatterplots with R^2 and p-value (All Columns on One Plot)")
        for j in range(len(columns)):
            x = df_cleaned[columns[j]]
            plt.figure(figsize=(10, 8))
            
            for i in range(len(columns)):
                if i == j:
                    continue
                y = df_cleaned[columns[i]]
                
                # Apply Z-score normalization to y
                y_zscore = (y - y.mean()) / y.std()
                
                sns.scatterplot(x=x, y=y_zscore, alpha=0.5, label=columns[i]
                )
                sns.regplot(x=x, y=y_zscore, scatter=False, line_kws={'alpha': 0.5}, label=None)
                
                # Calculate R^2 and p-value
                X = sm.add_constant(x)
                model = sm.OLS(y_zscore, X).fit()
                r_squared = model.rsquared
                p_value = model.pvalues[1]
                
                # Annotate the plot with R^2 and p-value for each y column
                plt.annotate(
                    f"{columns[i]}: $R^2$={r_squared:.2f}, $p$={p_value:.2e}",
                    xy=(0.05, 0.95 - i * 0.05),
                    xycoords='axes fraction',
                    fontsize=10,
                    ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white')
                )
            
            # Add title and labels
            plt.title(f"{columns[j]} vs MRS")
            plt.xlabel(columns[j])
            plt.ylabel("Z-Score")
            plt.legend(title="", loc="upper right")
            st_pyplot_func(plt.gcf(), filename=f'scatterplot_{columns[j]}_vs_all_y_zscore.png')
            plt.close()

from scipy.stats import mannwhitneyu

def vs_controls_general_groups(project_name, df_wnv2, controls, boxplot_columns, analysis_type):
    """
    Run general groups vs controls analysis: demographics and boxplot columns.
    """
    df_wnv2['Group'] = project_name
    controls['Group'] = 'Controls'
    combined_df = pd.concat([df_wnv2, controls], ignore_index=True, axis=0)
    controls_dir = f'temps_Controls_EDF'
    try:
        st.header('Controls Demographics')
        # get controls demographic from get_controls_ages_genders(controls_dir)
        controls_demographics_df = get_controls_ages_genders(controls_dir)
        # Use renamed columns if available, otherwise use original names
        age_col = 'age' if 'age' in df_wnv2.columns else 'clinical_age_at_visit'
        gender_col = 'gender' if 'gender' in df_wnv2.columns else 'clinical_sex, 1=male'
        cobrad_ages = df_wnv2[age_col]
        cobrad_sexes = df_wnv2[gender_col]
        cobrad_sexes -= 1
        st.write(f"Controls Demographics: N= {controls_demographics_df.shape[0]}, mean age {controls_demographics_df['Age'].mean():.2f} ± {controls_demographics_df['Age'].std():.2f}")
        # Display mean gender (assuming 1=male, 0=female or similar coding)
        st.write(f"Mean gender (1=female): {controls_demographics_df['Gender'].mean():.2f} ± {controls_demographics_df['Gender'].std():.2f}")
        # Mann-Whitney U test for age
        age_stat, age_p = mannwhitneyu(cobrad_ages.dropna(), controls_demographics_df['Age'].dropna(), alternative='two-sided')
        st.write(f"Mann-Whitney U test for age: p={age_p:.3g}")
        # Mann-Whitney U test for gender
        if 'Gender' in controls_demographics_df.columns:
            gender_stat, gender_p = mannwhitneyu(cobrad_sexes.dropna(), controls_demographics_df['Gender'].dropna(), alternative='two-sided')
            st.write(f"Mann-Whitney U test for gender: p={gender_p:.3g}")
    except Exception as e:
        st.warning(f"Error occurred while processing controls demographics: {e}")
    for col in boxplot_columns:
        curr_data = combined_df[[col, 'Group']].dropna()
        num_groups = curr_data['Group'].nunique()
        if num_groups < 2:
            continue
        results_df = analyze_and_correct(curr_data, [col], groups=curr_data['Group'].unique())
        boxplot_plot_dabest(results_df, curr_data, col, 'vs_controls', is_streamlit=True, analysis_type=analysis_type)

def vs_controls_all_features(df_wnv2, controls, boxplot_columns, analysis_type):
    """
    Run analysis over all features vs controls: iterate through all clinical columns and create plots.
    """
    # Add section to go over columns of df_wnv2 and show features vs controls
    st.divider()
    st.header('Features vs Controls Analysis')

    # Get clinical features
    clinical_features, _ = get_clinical_and_boxplot_cols(df_wnv2=df_wnv2)
    
    # Get columns to iterate over (exclude metadata columns)
    cols_to_skip = ['ID', 'annotations', 'bad_channels', 'Group', 'patient_number', 'size', 'n_samples', 
                    'lowpass', 'highpass', 'duration_sec', 'parquet_file_name']
    # Add age and gender columns to skip (check both old and new names)
    age_cols_to_skip = [col for col in df_wnv2.columns if 'age' in col.lower()]
    gender_cols_to_skip = [col for col in df_wnv2.columns if 'gender' in col.lower() or ('sex' in col.lower() and 'age' not in col.lower())]
    cols_to_skip.extend(age_cols_to_skip)
    cols_to_skip.extend(gender_cols_to_skip)
    
    # Filter to only clinical features that are numeric and not in cols_to_skip
    feature_columns = [col for col in clinical_features if col not in cols_to_skip]
    numeric_columns = df_wnv2.select_dtypes(include=[np.number]).columns
    feature_columns = [col for col in feature_columns if col in numeric_columns]
    controls['Group'] = 'Controls'

    for idx, selected_feature in enumerate(feature_columns):
            # keep only rows that can be safely converted to float
            feature_data = (
                df_wnv2[selected_feature]
                .apply(pd.to_numeric, errors='coerce')  # turn invalids into NaN
                .dropna()
                .astype(float)
            )
            if feature_data.empty:
                reason = f"Skipping {selected_feature}: No valid numeric data after conversion"
                print(reason)
                st.write(f"⏭️ {reason}")
                continue  # Skip to next feature
            
            # Display selected feature header
            st.subheader(f"Feature: {selected_feature}")
            
            # Create df_wnv3 with the selected feature
            df_wnv3 = df_wnv2[df_wnv2[selected_feature].notna()].copy()
            
            # Clean ID column: remove .edf, .eeg, and other file extensions
            if 'ID' in df_wnv3.columns:
                df_wnv3['ID'] = df_wnv3['ID'].astype(str).str.replace('.edf', '', regex=False, case=False)
                df_wnv3['ID'] = df_wnv3['ID'].str.replace('.eeg', '', regex=False, case=False)
                df_wnv3['ID'] = df_wnv3['ID'].str.replace('.EDF', '', regex=False, case=False)
                df_wnv3['ID'] = df_wnv3['ID'].str.replace('.EEG', '', regex=False, case=False)
                # Remove any trailing dots
                df_wnv3['ID'] = df_wnv3['ID'].str.rstrip('.')
            
            unique_values = df_wnv3[selected_feature].unique()
            
            # Only process binary features (exactly 2 unique values)
            if df_wnv3.shape[0] < 3 :
                if df_wnv3.shape[0] < 3:
                    reason = f"Skipping {selected_feature}: Not enough rows ({df_wnv3.shape[0]} < 3)"
                else:
                    reason = f"Skipping {selected_feature}: Not binary feature (has {len(unique_values)} unique values, need exactly 2)"
                print(reason)
                st.write(f"⏭️ {reason}")
                continue
            
            # Convert to numeric and ensure binary (0 and 1)
            df_wnv3[selected_feature] = pd.to_numeric(df_wnv3[selected_feature], errors='coerce')
            df_wnv3 = df_wnv3[df_wnv3[selected_feature].notna()]
            unique_values = df_wnv3[selected_feature].unique()
            
            
            # Normalize to 0 and 1 if needed
            min_val = df_wnv3[selected_feature].min()
            max_val = df_wnv3[selected_feature].max()
            if min_val != 0 or max_val != 1:
                # Map to 0 and 1
                df_wnv3[selected_feature] = df_wnv3[selected_feature].apply(lambda x: 1 if x == max_val else 0)
            
            # Check that there are at least 3 in each group (0,1)
            group_0_count = len(df_wnv3[df_wnv3[selected_feature] == 0])
            group_1_count = len(df_wnv3[df_wnv3[selected_feature] == 1])
            if group_0_count < 3 or group_1_count < 3:
                reason = f"Skipping {selected_feature}: Not enough samples in each group (Group 0: {group_0_count}, Group 1: {group_1_count}, need at least 3 each)"
                print(reason)
                st.write(f"⏭️ {reason}")
                continue
            
            # Get group 1's age and gender for matching controls
            group_1_data = df_wnv3[df_wnv3[selected_feature] == 1].copy()
            # Use renamed columns if available, otherwise use original names
            age_col = 'age' if 'age' in group_1_data.columns else 'clinical_age_at_visit'
            gender_col = 'gender' if 'gender' in group_1_data.columns else 'clinical_sex, 1=male'
            group_1_ages = group_1_data[age_col].dropna()
            group_1_sexes = group_1_data[gender_col].dropna()
            
            if group_1_ages.empty or group_1_sexes.empty:
                if group_1_ages.empty:
                    reason = f"Skipping {selected_feature}: Missing age data for group 1"
                else:
                    reason = f"Skipping {selected_feature}: Missing gender data for group 1"
                print(reason)
                st.write(f"⏭️ {reason}")
                continue
            
            # Match controls by age and gender from group 1
            # Convert group 1 gender format to match controls (if needed)
            if gender_col == 'clinical_sex, 1=male':
                # Convert: 1=male, 2=female -> 0=female, 1=male (to match controls format)
                group_1_sexes_normalized = group_1_sexes - 1
            else:
                # Assume already in correct format (0=female, 1=male)
                group_1_sexes_normalized = group_1_sexes.copy()
            
            # Check if controls have age and gender columns
            if 'age' not in controls.columns or 'gender' not in controls.columns:
                reason = f"Skipping {selected_feature}: Controls dataframe missing 'age' or 'gender' column"
                print(reason)
                st.write(f"⏭️ {reason}")
                continue
            
            # Match controls: find controls with matching age range and gender
            matched_controls_indices = []
            for age, sex in zip(group_1_ages, group_1_sexes_normalized):
                # Find controls with similar age (within 2 years) and same gender
                age_mask = (controls['age'] >= age - 2) & (controls['age'] <= age + 2)
                gender_mask = (controls['gender'] == sex)
                matched = controls[age_mask & gender_mask]
                if not matched.empty:
                    # Get matched indices that don't already exist
                    matched_indices = [idx for idx in matched.index.tolist() if idx not in matched_controls_indices]
                    # Take first 3 that don't already exist
                    matched_controls_indices.extend(matched_indices[:3])
            
            if not matched_controls_indices:
                reason = f"Skipping {selected_feature}: No matched controls found for group 1's age and gender distribution"
                print(reason)
                st.write(f"⏭️ {reason}")
                continue
            
            # Remove duplicates and get matched controls
            matched_controls_indices = list(set(matched_controls_indices))
            controls_matched = controls.loc[matched_controls_indices].copy()
            
            if controls_matched.empty:
                reason = f"Skipping {selected_feature}: No matched controls data found after filtering"
                print(reason)
                st.write(f"⏭️ {reason}")
                continue
            
            # Create groups: group 0, group 1, and controls
            org_selected_feature_for_plot = org_selected_feature(selected_feature) if selected_feature != 'sex' and 'sex' not in selected_feature.lower() else ''
            
            # Assign group labels
            df_wnv3.loc[df_wnv3[selected_feature] == 0, 'Group'] = f'{org_selected_feature_for_plot} 0' if org_selected_feature_for_plot else 'Group 0'
            df_wnv3.loc[df_wnv3[selected_feature] == 1, 'Group'] = f'{org_selected_feature_for_plot} 1' if org_selected_feature_for_plot else 'Group 1'
            
            # Concatenate only matched controls (they already have 'Group' = 'Controls' from earlier)
            df_wnv3 = pd.concat([df_wnv3, controls_matched], ignore_index=True)
            # Verify we have 3 unique groups
            unique_groups = df_wnv3['Group'].unique()
            if len(unique_groups) != 3:
                reason = f"Skipping {selected_feature}: Expected 3 unique groups but found {len(unique_groups)} groups: {unique_groups}"
                print(reason)
                st.write(f"⏭️ {reason}")
                continue
            
            # Create statistics table for age, gender, and selected_feature
            # Get age and gender columns
            age_col = 'age' if 'age' in df_wnv3.columns else 'clinical_age_at_visit'
            gender_col = 'gender' if 'gender' in df_wnv3.columns else 'clinical_sex, 1=male'
            
            # Prepare data for table
            table_data = []
            
            # Get group 0 and group 1 labels
            group_0_label = df_wnv3.loc[df_wnv3[selected_feature] == 0, 'Group'].iloc[0] if len(df_wnv3[df_wnv3[selected_feature] == 0]) > 0 else None
            group_1_label = df_wnv3.loc[df_wnv3[selected_feature] == 1, 'Group'].iloc[0] if len(df_wnv3[df_wnv3[selected_feature] == 1]) > 0 else None
            group_0_data = df_wnv3[df_wnv3['Group'] == group_0_label].copy() if group_0_label is not None else pd.DataFrame()
            group_1_data = df_wnv3[df_wnv3['Group'] == group_1_label].copy() if group_1_label is not None else pd.DataFrame()
            
            # Function to calculate p-value between two groups
            def calc_pvalue(group1_data, group2_data, col_name):
                try:
                    g1 = group1_data[col_name].dropna()
                    g2 = group2_data[col_name].dropna()
                    if len(g1) < 2 or len(g2) < 2:
                        return None
                    stat, pval = mannwhitneyu(g1, g2, alternative='two-sided')
                    return pval
                except:
                    return None
            
            # Age statistics
            if age_col in df_wnv3.columns:
                controls_age = controls_matched[age_col].dropna()
                group_0_age = group_0_data[age_col].dropna() if not group_0_data.empty else pd.Series()
                group_1_age = group_1_data[age_col].dropna() if not group_1_data.empty else pd.Series()
                
                table_data.append({
                    'Variable': 'Age',
                    'Controls': f"{controls_age.mean():.2f} ± {controls_age.std():.2f} (n={len(controls_age)})",
                    'Group 0': f"{group_0_age.mean():.2f} ± {group_0_age.std():.2f} (n={len(group_0_age)})" if len(group_0_age) > 0 else "N/A",
                    'Group 1': f"{group_1_age.mean():.2f} ± {group_1_age.std():.2f} (n={len(group_1_age)})" if len(group_1_age) > 0 else "N/A",
                    'p-value (Controls vs Group 0)': f"{calc_pvalue(controls_matched, group_0_data, age_col):.3e}" if calc_pvalue(controls_matched, group_0_data, age_col) is not None else "N/A",
                    'p-value (Controls vs Group 1)': f"{calc_pvalue(controls_matched, group_1_data, age_col):.3e}" if calc_pvalue(controls_matched, group_1_data, age_col) is not None else "N/A",
                    'p-value (Group 0 vs Group 1)': f"{calc_pvalue(group_0_data, group_1_data, age_col):.3e}" if calc_pvalue(group_0_data, group_1_data, age_col) is not None else "N/A"
                })
            
            # Gender statistics
            if gender_col in df_wnv3.columns:
                controls_gender = controls_matched[gender_col].dropna()
                group_0_gender = group_0_data[gender_col].dropna() if not group_0_data.empty else pd.Series()
                group_1_gender = group_1_data[gender_col].dropna() if not group_1_data.empty else pd.Series()
                
                table_data.append({
                    'Variable': 'Gender',
                    'Controls': f"{controls_gender.mean():.2f} ± {controls_gender.std():.2f} (n={len(controls_gender)})",
                    'Group 0': f"{group_0_gender.mean():.2f} ± {group_0_gender.std():.2f} (n={len(group_0_gender)})" if len(group_0_gender) > 0 else "N/A",
                    'Group 1': f"{group_1_gender.mean():.2f} ± {group_1_gender.std():.2f} (n={len(group_1_gender)})" if len(group_1_gender) > 0 else "N/A",
                    'p-value (Controls vs Group 0)': f"{calc_pvalue(controls_matched, group_0_data, gender_col):.3e}" if calc_pvalue(controls_matched, group_0_data, gender_col) is not None else "N/A",
                    'p-value (Controls vs Group 1)': f"{calc_pvalue(controls_matched, group_1_data, gender_col):.3e}" if calc_pvalue(controls_matched, group_1_data, gender_col) is not None else "N/A",
                    'p-value (Group 0 vs Group 1)': f"{calc_pvalue(group_0_data, group_1_data, gender_col):.3e}" if calc_pvalue(group_0_data, group_1_data, gender_col) is not None else "N/A"
                })
            
            # Selected feature statistics
            if selected_feature in df_wnv3.columns and selected_feature in controls_matched.columns:
                controls_feat = controls_matched[selected_feature].dropna()
                group_0_feat = group_0_data[selected_feature].dropna() if not group_0_data.empty else pd.Series()
                group_1_feat = group_1_data[selected_feature].dropna() if not group_1_data.empty else pd.Series()
                
                table_data.append({
                    'Variable': selected_feature,
                    'Controls': f"{controls_feat.mean():.2f} ± {controls_feat.std():.2f} (n={len(controls_feat)})",
                    'Group 0': f"{group_0_feat.mean():.2f} ± {group_0_feat.std():.2f} (n={len(group_0_feat)})" if len(group_0_feat) > 0 else "N/A",
                    'Group 1': f"{group_1_feat.mean():.2f} ± {group_1_feat.std():.2f} (n={len(group_1_feat)})" if len(group_1_feat) > 0 else "N/A",
                    'p-value (Controls vs Group 0)': f"{calc_pvalue(controls_matched, group_0_data, selected_feature):.3e}" if calc_pvalue(controls_matched, group_0_data, selected_feature) is not None else "N/A",
                    'p-value (Controls vs Group 1)': f"{calc_pvalue(controls_matched, group_1_data, selected_feature):.3e}" if calc_pvalue(controls_matched, group_1_data, selected_feature) is not None else "N/A",
                    'p-value (Group 0 vs Group 1)': f"{calc_pvalue(group_0_data, group_1_data, selected_feature):.3e}" if calc_pvalue(group_0_data, group_1_data, selected_feature) is not None else "N/A"
                })
            
            # Display table
            if table_data :
                stats_df = pd.DataFrame(table_data)
                st.write("**Group Demographics**")
                st.dataframe(stats_df, use_container_width=True)
                # Add table to PPTX
                add_table_to_pptx(stats_df, title="Group Demographics")
            
            # Create filtered dataframe with only Group 1 and Controls for plotting
            df_wnv3_plot = df_wnv3[df_wnv3['Group'].isin(['Controls', group_1_label])].copy() if group_1_label is not None else df_wnv3[df_wnv3['Group'] == 'Controls'].copy()
            
            # run over boxplot_columns
            for band in boxplot_columns:
                results_df = analyze_and_correct(df_wnv3_plot, [band], groups=df_wnv3_plot['Group'].unique())
                time.sleep(0.5)  # Wait 0.5 seconds for things to load
                boxplot_plot_dabest(results_df, df_wnv3_plot, band, f'{selected_feature}', is_streamlit=True, selected_feature=selected_feature)
                results_df = analyze_and_correct(df_wnv3, [band], groups=df_wnv3['Group'].unique())
                time.sleep(0.5)  # Wait 0.5 seconds for things to load
                boxplot_plot_dabest(results_df, df_wnv3, band, f'{selected_feature}', is_streamlit=True, selected_feature=selected_feature)
            
            st.divider()

def vs_controls_run(project_name, df_wnv2, controls, boxplot_columns, analysis_type, feature_types_to_run=None):
    """
    Main function to run vs controls analysis. Calls both general groups and all features analysis.
    """
    # Run all features vs controls first (only if "vs_Controls" is in feature_types_to_run)
    if feature_types_to_run is None or "vs_Controls" in feature_types_to_run:
        vs_controls_all_features(df_wnv2, controls, boxplot_columns, analysis_type)
    
    # Run general groups vs controls (demographics and boxplot columns)
    vs_controls_general_groups(project_name, df_wnv2, controls, boxplot_columns, analysis_type)

def ml_plots_get_images(project_name, selected_feature):
    ml_plots_dir = f"{project_name}_figures/ml_plots"
    if os.path.exists(ml_plots_dir):
        # get all files in f"{project_name}_figures/ml_plots/{selected_feature}"
        ml_plot_files = [f for f in os.listdir(os.path.join(ml_plots_dir, selected_feature)) if f.endswith('.png')]
        for file in ml_plot_files:
            st.image(os.path.join(ml_plots_dir, selected_feature, file), caption=file)
    else:
        st.write(f"No ML plots found in {ml_plots_dir}")

def find_and_sort_ml_plots(ml_plots_dir):
    """
    Find all files in subfolders of ml_plots that match the pattern
    COBRAD_XGB_10_feat_imp_%d and sort them by %d.

    Parameters:
        ml_plots_dir (str): Path to the ml_plots directory.

    Returns:
        list: Sorted list of file paths.
    """
    pattern = r"COBRAD_XGB_10_feat_imp_(\d+)"  # Regex to extract the number %d
    matched_files = []

    # Walk through all subfolders and files in ml_plots_dir
    for root, _, files in os.walk(ml_plots_dir):
        for file in files:
            match = re.search(pattern, file)
            if match:
                # Extract the number %d and store it with the file path
                matched_files.append((int(match.group(1)), os.path.join(root, file)))

    # Sort the files by the extracted number %d
    matched_files.sort(key=lambda x: x[0], reverse=True)

    # Return only the sorted file paths
    return [file_path for _, file_path in matched_files]

def org_selected_feature(selected_feature):
    # if clinical_LBD_Cognitive_fluctuation change to LBD
    if selected_feature == 'clinical_LBD_Cognitive_fluctuation':
        return 'CF'
    return selected_feature

def get_cap_sleep_group(file_prefix):
    """
    Determine group (Dementia/Control) based on file prefix from CAP_Sleep_Database.
    Based on naming patterns observed in the database.
    """
    # Control group prefixes (typically start with 'n')
    control_prefixes = ['n3', 'n5', 'n8', 'n10', 'n11']
    
    # Check if any control prefix matches
    for prefix in control_prefixes:
        if file_prefix.startswith(prefix):
            return 'Control'
    
    # All other prefixes are considered Dementia cases
    return 'Dementia'

def HEP_plots2(project_name, df_wnv3, controls, boxplot_columns, analysis_type, selected_feature=None, size_feature=None):
    """
    New HEP analysis function that reads EDF files directly from pickles/EDF directory,
    calculates HEP measurements for each patient, and shows a pairgrid of results.
    """
    if project_name != 'COBRAD':
        return
    
    # Define the pickles directory
    pickles_dir = 'pickles/EDF'
    if not os.path.exists(pickles_dir):
        st.error(f"Directory not found: {pickles_dir}")
        return
    
    # Get all pickle files
    pickle_files = [f for f in os.listdir(pickles_dir) if f.endswith('.pkl')]
    
    if not pickle_files:
        st.warning(f"No pickle files found in {pickles_dir}")
        return
    
    # Group files by patient ID and select only the first file per patient
    patient_files = {}
    
    for pickle_file in pickle_files:
        # Extract patient ID from filename using regex pattern (\d{4}-\d{3})
        import re
        m = re.search(r'(\d{4}-\d{3})', pickle_file)
        if m:
            patient_id = m.group(1)
        else:
            # Fallback: use filename without extension
            patient_id = pickle_file.replace('.pkl', '')
        
        # If this is the first file for this patient, add it
        if patient_id not in patient_files:
            patient_files[patient_id] = pickle_file
    
    # Get the list of files to process (one per patient)
    files_to_process = list(patient_files.values())
    
    st.write(f"Found {len(pickle_files)} total files, processing {len(files_to_process)} files (1 per patient)")
    st.info("⚠️ Processing limited to first 5 minutes of data per patient for faster analysis")
    st.info("🚀 Processing patients in parallel for faster execution")
    st.info("📊 Converting raw data from volts to microvolts for standard EEG analysis")
    st.info("⚡ Calculating power bands for each time window")
    st.info("📈 Results will show mean values across patients for each time window")
    
    # Add checkbox for processing all patients
    process_all_patients = st.checkbox("Process all patients (unchecked = first 5 patients only)", value=False)
    
    # Check for existing cache and determine which patients need processing
    cache_dir = 'Cache'
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check which patients are already cached
    cached_patients = set()
    if os.path.exists(cache_dir):
        # Look for individual patient cache files
        patient_cache_files = [f for f in os.listdir(cache_dir) if f.startswith('HEP_patient_') and f.endswith('.parquet')]
        for cache_file in patient_cache_files:
            # Extract patient ID from filename (format: HEP_patient_XXXX-XXX_band.parquet)
            import re
            match = re.search(r'HEP_patient_(\d{4}-\d{3})_', cache_file)
            if match:
                patient_id = match.group(1)
                cached_patients.add(patient_id)
    
    st.info(f"📁 Found {len(cached_patients)} patients already cached: {sorted(cached_patients)}")
    
    # If we have cached data and not processing all patients, show cached results
    if cached_patients and not process_all_patients:
        try:
            # Load all cached patient data efficiently
            cached_dataframes = []
            for patient_id in cached_patients:
                for band_name in power_bands.keys():
                    cache_file = f"HEP_patient_{patient_id}_{band_name}.parquet"
                    cache_path = os.path.join(cache_dir, cache_file)
                    if os.path.exists(cache_path):
                        patient_df = pd.read_parquet(cache_path)
                        cached_dataframes.append(patient_df)
            
            if cached_dataframes:
                # Combine all cached data
                cached_df = pd.concat(cached_dataframes, ignore_index=True)
                # Clear the list to free memory immediately
                cached_dataframes.clear()
                
                # Apply z-score normalization to metric columns
                numeric_cols = cached_df.select_dtypes(include=[np.number]).columns.tolist()
                exclude_cols = ['patient_id', 'window_id', 'time_start', 'time_end']
                metric_cols = [col for col in numeric_cols if col not in exclude_cols]
                
                # Z-score normalize the metric columns
                from scipy import stats
                for col in metric_cols:
                    # Check for NaN values and replace them
                    if cached_df[col].isna().any():
                        cached_df[col] = cached_df[col].fillna(cached_df[col].mean())
                    
                    # Calculate standard deviation
                    std_val = cached_df[col].std()
                    
                    if std_val > 1e-10:  # Use a very small threshold instead of 0
                        z_scores = stats.zscore(cached_df[col], nan_policy='omit')
                        # Handle any remaining NaN values in z_scores
                        if np.isnan(z_scores).any():
                            cached_df[col] = cached_df[col]  # Keep original values
                        else:
                            cached_df[col] = z_scores
                    else:
                        # Keep original values instead of setting to 0
                        pass
                
                # Group by window_id and calculate mean
                mean_results = cached_df.groupby('window_id')[metric_cols].mean().reset_index()
                
                # Add time information
                time_info = cached_df.groupby('window_id')[['time_start', 'time_end']].mean().reset_index()
                mean_results = mean_results.merge(time_info, on='window_id')
                
                # Add patient count for each window
                patient_counts = cached_df.groupby('window_id')['patient_id'].nunique().reset_index()
                patient_counts.columns = ['window_id', 'patient_count']
                mean_results = mean_results.merge(patient_counts, on='window_id')
                
                # get where patient count is higher than 50
                mean_results = mean_results[mean_results['patient_count'] > 50]
                mean_results['Group'] = 'Dementia'
                
                st.info(f"📁 Loading cached results from {len(cached_patients)} patients")
                st.success(f"✅ Loaded {len(mean_results)} time windows from cache")
                
                # Display the cached data
                st.write("**Cached Results Preview:**")
                st.dataframe(mean_results.head())
                
                # Show plots using cached data
                for band_name in power_bands.keys():
                    st.write(f"Analyzing power band: {band_name}")
                    only_plots(
                        results_df=mean_results,
                        save_plot=False,
                        save_dir='',
                        edf_pickle_name=f"hep_analysis_{band_name}",
                        band=band_name,
                        step_sec=5,
                        is_streamlit=True
                    )
                    st.success(f"Completed analysis for {band_name} band with {len(mean_results)} time windows (from cache)")
                    st.divider()
                
                return  # Exit function after using cached data
                
        except Exception as e:
            st.warning(f"Could not load cached data: {e}")
            st.info("Proceeding with new data processing...")
    
    # Filter out already cached patients
    files_to_process_filtered = []
    for pickle_file in files_to_process:
        # Extract patient ID from filename
        import re
        m = re.search(r'(\d{4}-\d{3})', pickle_file)
        if m:
            patient_id = m.group(1)
            if patient_id not in cached_patients:
                files_to_process_filtered.append(pickle_file)
        else:
            # Fallback: use filename without extension
            patient_id = pickle_file.replace('.pkl', '')
            if patient_id not in cached_patients:
                files_to_process_filtered.append(pickle_file)
    
    # Limit to 5 patients unless checkbox is checked
    if not process_all_patients:
        # from the end
        files_to_process_filtered = files_to_process_filtered[-5:]
        st.info(f"🔢 Processing only {len(files_to_process_filtered)} new patients. Check the box above to process all {len(files_to_process_filtered)} patients.")
    else:
        st.info(f"🔢 Processing all {len(files_to_process_filtered)} new patients.")
    
    # Update files_to_process to the filtered list
    files_to_process = files_to_process_filtered
    
    if not files_to_process:
        st.info("✅ All patients are already cached! Loading from cache...")
        # Load and display cached results
        try:
            cached_dataframes = []
            for patient_id in cached_patients:
                for band_name in power_bands.keys():
                    cache_file = f"HEP_patient_{patient_id}_{band_name}.parquet"
                    cache_path = os.path.join(cache_dir, cache_file)
                    if os.path.exists(cache_path):
                        patient_df = pd.read_parquet(cache_path)
                        cached_dataframes.append(patient_df)
            
            if cached_dataframes:
                # Combine all cached data and process as before
                cached_df = pd.concat(cached_dataframes, ignore_index=True)
                # Clear the list to free memory immediately
                cached_dataframes.clear()
                
                # Apply z-score normalization to metric columns
                numeric_cols = cached_df.select_dtypes(include=[np.number]).columns.tolist()
                exclude_cols = ['patient_id', 'window_id', 'time_start', 'time_end']
                metric_cols = [col for col in numeric_cols if col not in exclude_cols]
                
                # Z-score normalize the metric columns
                from scipy import stats
                for col in metric_cols:
                    # Check for NaN values and replace them
                    if cached_df[col].isna().any():
                        cached_df[col] = cached_df[col].fillna(cached_df[col].mean())
                    
                    # Calculate standard deviation
                    std_val = cached_df[col].std()
                    
                    if std_val > 1e-10:  # Use a very small threshold instead of 0
                        z_scores = stats.zscore(cached_df[col], nan_policy='omit')
                        # Handle any remaining NaN values in z_scores
                        if np.isnan(z_scores).any():
                            cached_df[col] = cached_df[col]  # Keep original values
                        else:
                            cached_df[col] = z_scores
                    else:
                        # Keep original values instead of setting to 0
                        pass
                
                # Group by window_id and calculate mean
                mean_results = cached_df.groupby('window_id')[metric_cols].mean().reset_index()
                
                # Add time information
                time_info = cached_df.groupby('window_id')[['time_start', 'time_end']].mean().reset_index()
                mean_results = mean_results.merge(time_info, on='window_id')
                
                # Add patient count for each window
                patient_counts = cached_df.groupby('window_id')['patient_id'].nunique().reset_index()
                patient_counts.columns = ['window_id', 'patient_count']
                mean_results = mean_results.merge(patient_counts, on='window_id')
                
                # Filter to max patient count
                mean_results = mean_results[mean_results['patient_count'] == mean_results['patient_count'].max()]
                mean_results['Group'] = 'Dementia'
                
                # Show plots using cached data
                for band_name in power_bands.keys():
                    st.write(f"Analyzing power band: {band_name}")
                    only_plots(
                        results_df=mean_results,
                        save_plot=False,
                        save_dir='',
                        edf_pickle_name=f"hep_analysis_{band_name}",
                        band=band_name,
                        step_sec=5,
                        is_streamlit=True
                    )
                    st.success(f"Completed analysis for {band_name} band with {len(mean_results)} time windows (from cache)")
                    st.divider()
                
                return  # Exit function after using cached data
        except Exception as e:
            st.warning(f"Could not load cached data: {e}")
            st.info("Proceeding with new data processing...")
    
    # Process each frequency band
    for band_name, band_range in power_bands.items():
        st.write(f"Analyzing power band: {band_name}")
        
        all_patient_results = []
        
        # Create progress bar for file processing
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process files in parallel
        def process_single_file(pickle_file):
            """Process a single pickle file and return results"""
            try:
                # Extract patient ID from filename
                import re
                m = re.search(r'(\d{4}-\d{3})', pickle_file)
                if m:
                    patient_id = m.group(1)
                else:
                    patient_id = pickle_file.replace('.pkl', '')
                
                # Check if this patient is already cached for this band
                cache_file = f"HEP_patient_{patient_id}_{band_name}.parquet"
                cache_path = os.path.join(cache_dir, cache_file)
                
                if os.path.exists(cache_path):
                    try:
                        cached_df = pd.read_parquet(cache_path)
                        return cached_df, patient_id
                    except Exception as e:
                        # Continue with processing if cache load fails
                        pass
                
                # Load the raw EEG data
                with open(os.path.join(pickles_dir, pickle_file), 'rb') as f:
                    raw = pickle.load(f)
                
                # Calculate HEP measurements for this patient
                patient_results = calculate_hep_measurements_for_patient(raw, patient_id, band_name, band_range)
                
                if patient_results is not None:
                    # Save individual patient data to cache
                    try:
                        patient_results.to_parquet(cache_path, index=False)
                    except Exception as e:
                        pass
                    
                    return patient_results, patient_id
                else:
                    return None, patient_id
                    
            except Exception as e:
                return None, pickle_file
        
        # Use ThreadPoolExecutor for parallel processing
        max_workers = min(16, len(files_to_process))  # Limit to 4 workers to avoid overwhelming the system
        
        # Update status text to show current processing mode
        if len(files_to_process) <= 5:
            st.info(f"🔬 Processing {len(files_to_process)} patients for quick analysis")
        else:
            st.info(f"🔬 Processing {len(files_to_process)} patients for comprehensive analysis")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(process_single_file, pickle_file): pickle_file 
                             for pickle_file in files_to_process}
            
            # Process completed tasks
            completed_count = 0
            for future in as_completed(future_to_file):
                completed_count += 1
                pickle_file = future_to_file[future]
                
                # Update progress bar
                progress = completed_count / len(files_to_process)
                progress_bar.progress(progress)
                status_text.text(f"Processing file {completed_count}/{len(files_to_process)}: {pickle_file}")
                
                try:
                    patient_results, patient_id = future.result()
                    if patient_results is not None:
                        all_patient_results.append(patient_results)
                except Exception as e:
                    st.warning(f"Error processing file {pickle_file}: {e}")
                    continue
        
        if not all_patient_results:
            st.warning(f"No valid results for band {band_name}")
            continue
        
        # Combine all patient results
        results_df = pd.concat(all_patient_results, ignore_index=True)
        
        # Apply z-score normalization to metric columns
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['patient_id', 'window_id', 'time_start', 'time_end']
        metric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Z-score normalize the metric columns
        from scipy import stats
        for col in metric_cols:
            # Check for NaN values and replace them
            if results_df[col].isna().any():
                results_df[col] = results_df[col].fillna(results_df[col].mean())
            
            # Calculate standard deviation
            std_val = results_df[col].std()
            
            if std_val > 1e-10:  # Use a very small threshold instead of 0
                z_scores = stats.zscore(results_df[col], nan_policy='omit')
                # Handle any remaining NaN values in z_scores
                if np.isnan(z_scores).any():
                    results_df[col] = results_df[col]  # Keep original values
                else:
                    results_df[col] = z_scores
            else:
                # Keep original values instead of setting to 0
                pass
        
        # Calculate mean values across patients for each window_id
        
        # Group by window_id and calculate mean
        mean_results = results_df.groupby('window_id')[metric_cols].mean().reset_index()
        
        # Add time information (use mean time across patients for each window)
        time_info = results_df.groupby('window_id')[['time_start', 'time_end']].mean().reset_index()
        mean_results = mean_results.merge(time_info, on='window_id')
        
        # Add patient count for each window
        patient_counts = results_df.groupby('window_id')['patient_id'].nunique().reset_index()
        patient_counts.columns = ['window_id', 'patient_count']
        mean_results = mean_results.merge(patient_counts, on='window_id')
        # Save combined results to parquet file (for backward compatibility)
        patient_count_max = mean_results['patient_count'].max()
        combined_filename = f"HEP_time_avg_{int(patient_count_max)}.parquet"
        combined_path = os.path.join(cache_dir, combined_filename)
        
        try:
            mean_results.to_parquet(combined_path, index=False)
            st.success(f"💾 Combined results saved to {combined_filename}")
        except Exception as e:
            st.warning(f"Could not save combined results to parquet: {e}")
        # only the rows where mean_results patient_count is max
        mean_results = mean_results[mean_results['patient_count'] == mean_results['patient_count'].max()]
        

        
        # Use mean_results instead of results_df for plotting
        results_df = mean_results
        
        # Add group information
        results_df['Group'] = 'Dementia'  # All patients are from Dementia group
        
        # Add size feature if specified
        if size_feature and size_feature in df_wnv3.columns:
            for idx, row in results_df.iterrows():
                patient_id = row['patient_id']
                matching_rows = df_wnv3[df_wnv3['ID'] == patient_id]
                if not matching_rows.empty:
                    results_df.loc[idx, size_feature] = matching_rows[size_feature].iloc[0]
                else:
                    results_df.loc[idx, size_feature] = df_wnv3[size_feature].mean() if size_feature in df_wnv3.columns else 1.0
        
        st.write(f"Processed {len(results_df)} time windows for band {band_name}")
        
        # Use only_plots function instead of custom pairgrid
        only_plots(
            results_df=results_df,
            save_plot=False,
            save_dir='',
            edf_pickle_name=f"hep_analysis_{band_name}",
            band=band_name,
            step_sec=5,
            is_streamlit=True
        )
        
        # Clear progress bars
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"Completed analysis for {band_name} band with {len(results_df)} time windows (averaged across {len(all_patient_results)} patients)")
        st.info(f"📊 Results show mean values across patients for each time window")
        st.divider()


def calculate_hep_measurements_for_patient(raw, patient_id, band_name, band_range):
    """
    Calculate HEP measurements for a single patient's EEG data.
    Returns a DataFrame with measurements for each time window.
    """
    try:
        # Check for ECG channel first - skip patient if none found
        ecg_channel = None
        for i, ch_name in enumerate(raw.ch_names):
            if 'ecg' in ch_name.lower() or 'ekg' in ch_name.lower():
                ecg_channel = i
                break
        
        if ecg_channel is None:
            return None
        
        # Extract EEG data and convert from volts to microvolts
        eeg_data = raw.get_data() * 1e6  # Convert V to μV
        sfreq = raw.info['sfreq']
        
        # Define window parameters (same as ECG analysis)
        window_size_sec = 15  # 15-second windows
        step_size_sec = 5     # 5-second steps
        
        window_size = int(window_size_sec * sfreq)
        step_size = int(step_size_sec * sfreq)
        
        # Limit to first 5 minutes of data
        max_samples_5min = int(5 * 60 * sfreq)  # 5 minutes in samples
        if eeg_data.shape[1] > max_samples_5min:
            eeg_data = eeg_data[:, :max_samples_5min]
        
        # Calculate number of windows
        n_windows = max(0, (eeg_data.shape[1] - window_size) // step_size + 1)
        
        if n_windows == 0:
            return None
        
        # Initialize storage for this patient
        patient_results = []
        
        # Create progress bar for window processing
        if n_windows > 0:
            window_progress_bar = st.progress(0)
            window_status_text = st.empty()
        
        # Process each window
        for w in range(n_windows):
            # Update window progress bar
            if n_windows > 0:
                window_progress = (w + 1) / n_windows
                window_progress_bar.progress(window_progress)
                window_status_text.text(f"Processing window {w+1}/{n_windows} for patient {patient_id}")
            start = w * step_size
            end = start + window_size
            
            if end > eeg_data.shape[1]:
                break
                
            # Extract window data
            window_data = eeg_data[:, start:end]
            
            # Calculate network features for this frequency band
            try:
                efficiency, clustering, assortativity, modularity = compute_network_features(
                    window_data, sfreq, band_range
                )
            except Exception as e:
                continue
            
            # Calculate power bands for this window
            try:
                from scipy.signal import welch
                
                # Calculate power spectral density for each channel
                power_bands_window = {}
                for ch_idx in range(window_data.shape[0]):
                    # Calculate PSD for this channel
                    freqs, psd = welch(window_data[ch_idx], fs=sfreq, nperseg=min(256, len(window_data[ch_idx])//4))
                    
                    # Calculate power in each frequency band
                    for band_name, (fmin, fmax) in power_bands.items():
                        band_mask = (freqs >= fmin) & (freqs <= fmax)
                        band_power = np.trapezoid(psd[band_mask], freqs[band_mask])
                        
                        if band_name not in power_bands_window:
                            power_bands_window[band_name] = []
                        power_bands_window[band_name].append(band_power)
                
                # Calculate mean power across all channels for each band
                mean_power_bands = {}
                for band_name in power_bands_window:
                    mean_power_bands[f"{band_name}_power"] = np.mean(power_bands_window[band_name])
                
            except Exception as e:
                # Set default values for power bands
                mean_power_bands = {}
                for band_name in power_bands:
                    mean_power_bands[f"{band_name}_power"] = np.nan
            
            # Calculate HRV features (Vagal_SD1 and Sympathetic_SD2) using the same method as HEP_parquet_generation.py
            try:
                # ECG channel already verified at the beginning of the function
                # Get ECG data for this window and convert from volts to microvolts
                ecg_data = raw.get_data()[ecg_channel, start:end] * 1e6  # Convert V to μV
                
                # Use the same approach as in HEP_parquet_generation.py
                import neurokit2 as nk
                
                # Clean ECG signal
                ecg_clean = nk.ecg_clean(ecg_data, sampling_rate=sfreq)
                
                # Find R-peaks
                try:
                    signals, info = nk.ecg_process(ecg_clean, sampling_rate=sfreq)
                    rpeaks = signals['ECG_R_Peaks']
                    
                    # Calculate SD1 and SD2 using the same method as HEP_parquet_generation.py
                    if len(rpeaks) > 2:
                        # Convert R-peak indices to times
                        r_times = rpeaks / sfreq
                        
                        # Calculate inter-beat intervals (IBI)
                        ibi = np.diff(rpeaks) / sfreq
                        
                        # Calculate differences between consecutive IBIs
                        dibi = np.diff(ibi)
                        
                        # Calculate SD1 and SD2 using the same formulas
                        vagal_sd1 = np.std(dibi) / np.sqrt(2)
                        sympathetic_sd2 = np.sqrt(max(0, 2 * np.std(ibi)**2 - 0.5 * np.std(dibi)**2))
                        
                    else:
                        vagal_sd1 = np.nan
                        sympathetic_sd2 = np.nan
                    
                except Exception as e:
                    vagal_sd1 = np.nan
                    sympathetic_sd2 = np.nan
                    
            except Exception as e:
                vagal_sd1 = np.nan
                sympathetic_sd2 = np.nan
            
            # Store results for this window
            window_result = {
                'patient_id': patient_id,
                'window_id': w,
                'time_start': start / sfreq,
                'time_end': end / sfreq,
                'Efficiency': efficiency,
                'Clustering': clustering,
                'Assortativity': assortativity,
                'Modularity': modularity,
                'Vagal_SD1': vagal_sd1,
                'Sympathetic_SD2': sympathetic_sd2
            }
            
            # Add power bands to the result
            window_result.update(mean_power_bands)
            
            patient_results.append(window_result)
        
        # Clear window progress bar
        if n_windows > 0:
            window_progress_bar.empty()
            window_status_text.empty()
        
        return pd.DataFrame(patient_results)
        
    except Exception as e:
        st.warning(f"Error processing patient {patient_id}: {e}")
        return None


def HEP_plots(project_name, df_wnv3, controls, boxplot_columns, analysis_type, selected_feature=None, size_feature=None, context_key=None):
    # if size_feature is None:
    #     size_feature = 'clinical_moca'
    if project_name == 'COBRAD':
        # Ask user to select sleep stage
        sleep_stage = st.selectbox(
            "Select Sleep Stage",
            options=['N1','All',  'N2', 'N3', 'R', 'W'],
            index=1,  # Default to All
            key='sleep_stage_selection'
        )
        if sleep_stage == 'All':
            st.subheader("Analyzing all sleep stages: EDF_N1, EDF_N2, EDF_N3, EDF_R, EDF_W")
        else:
            st.subheader(f"Analyzing: EDF_{sleep_stage}")
        # Define directory if single stage
        edf_hep_dir = None if sleep_stage == 'All' else f'parquets_HEP/EDF_{sleep_stage}'
    else:
        return
    # If 'All' selected, run cross-stage comparison and return
    if project_name == 'COBRAD' and sleep_stage == 'All':
        compare_sleep_stages_default_pairs(df_wnv3, power_bands, size_feature)
        return

    # Run over power bands (single stage flow)
    for band_name, band_range in power_bands.items():
        st.write(f"Analyzing power band: {band_name}")
        dfs = []
        hue = None
        df_wnv3_clean = pd.DataFrame()
        # Process EDF_HEP directory (Dementia group)
        if os.path.exists(edf_hep_dir):
            edf_band_files = [f for f in os.listdir(edf_hep_dir) if f.endswith(f"_{band_name}.parquet")]
            for file in edf_band_files:
                file_path = os.path.join(edf_hep_dir, file)
                try:
                    df = pd.read_parquet(file_path)
                    if not df.empty:
                        # Extract patient ID from filename (assuming format like "patient_123_band.parquet")
                        patient_id = file.split('_')[0][1:]
                        matching_rows = df_wnv3[df_wnv3['ID'] == patient_id]
                        if not matching_rows.empty:
                            # df = pd.concat([matching_rows, df], ignore_index=True,axis=1)
                            # add to df_wnv3_clean matching_rows rows
                            df_wnv3_clean = pd.concat([df_wnv3_clean, matching_rows], ignore_index=True,axis=0)
                            df['Group'] = 'Dementia'
                            dfs.append(df)
                except Exception as e:
                    st.warning(f"Could not read {file} from EDF_HEP: {e}")
        
        if dfs:
            # Group by 'Group' and compute mean for each group
            group_dfs = []
            # Get unique groups from the loaded data (Dementia and Control)
            unique_groups = list(set([df['Group'].iloc[0] for df in dfs if 'Group' in df.columns and not df.empty]))
            
            for group in unique_groups:
                group_patients2 = []
                for df in dfs:
                    if 'Group' in df.columns and df['Group'].iloc[0] == group:
                        group_patients2.append(df)
                group_patients = [df for df in group_patients2]
                if group_patients:
                    # Get mean of each df (returns a Series) and concat to DataFrame
                    group_mean_df = pd.concat([df.drop(columns=['Group'], errors='ignore').mean() for df in group_patients], axis=1).T
                    group_mean_df['Group'] = group
                    group_dfs.append(group_mean_df)
            results_df = pd.concat(group_dfs, ignore_index=True)
            # # concat with df_wnv3_clean assume same index
            # results_df = pd.concat([pd.concat(group_dfs), df_wnv3_clean], axis=1)
            # remove cols more than 50% NaN
            results_df = results_df.loc[:, results_df.isnull().mean() < 0.5]
            # remove rows with any NaN
            results_df = results_df.dropna()
            hue = results_df['Group']
            st.write(f"Loaded {len(results_df)} group means for band {band_name}")
        else:
            st.write(f"No data found for band {band_name}")
        #  apply zscore each column (only numeric columns)
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns
        results_df[numeric_cols] = results_df[numeric_cols].apply(zscore)
        # results_df size_feature fillna max
        if size_feature:
            # remove rows where size_feature is NaN
            results_df = results_df[results_df[size_feature].notna()]
        st.dataframe(results_df)    
        if size_feature:
            # results_df leave columns 'Vagal_SD1','Sympathetic_SD2','Efficiency','Clustering','Modularity','Assortativity',size_feature
            results_df = results_df[['Vagal_SD1','Sympathetic_SD2','Efficiency','Clustering','Modularity','Assortativity',size_feature]]
        # Add cluster number selection
        n_clusters = 2
        
        # Use the original only_plots function
        only_plots(results_df, save_plot='', save_dir='', edf_pickle_name="plot", band=band_name, step_sec=5, is_streamlit=True, hue=hue, size_feature=size_feature)
        # Add pair-wise clustering analysis
        st.subheader("Pair-wise Clustering Analysis")
        pair_clustering_analysis(results_df, df_wnv3_clean, n_clusters=n_clusters, context_key=context_key)
        
        st.divider()


def compare_sleep_stages_default_pairs(df_wnv3, power_bands, size_feature=None):
    """Compare default feature pairs across all sleep stages.

    For each power band and for each of the default pairs, display per-stage:
    - Scatter plot (KMeans clusters) akin to plot_upper_cluster
    - Clinical feature-importance bar plot predicting clusters
    Also display averaged feature importance across stages and ranked importance.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import os
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier

    stages = ['N1', 'N2', 'N3', 'R', 'W']
    default_pairs = [
        ('Vagal_SD1', 'Efficiency'),
        ('Vagal_SD1', 'Clustering'),
        ('Vagal_SD1', 'Modularity'),
        ('Vagal_SD1', 'Assortativity')
    ]

    for band_name, _ in power_bands.items():
        st.header(f"Band: {band_name}")

        # Prepare per-stage datasets
        stage_results = {}
        stage_clinical = {}

        for stage in stages:
            edf_hep_dir = f'parquets_HEP/EDF_{stage}'
            dfs = []
            df_wnv3_clean = pd.DataFrame()
            if os.path.exists(edf_hep_dir):
                edf_band_files = [f for f in os.listdir(edf_hep_dir) if f.endswith(f"_{band_name}.parquet")]
                st.write(f"Loading files for stage {stage}, band {band_name} ({len(edf_band_files)} files)")
                progress_bar = st.progress(0)
                total_files = len(edf_band_files)
                processed_files = 0
                for file in edf_band_files:
                    file_path = os.path.join(edf_hep_dir, file)
                    try:
                        df = pd.read_parquet(file_path)
                        if not df.empty:
                            patient_id = file.split('_')[0][1:]
                            matching_rows = df_wnv3[df_wnv3['ID'] == patient_id]
                            if not matching_rows.empty:
                                df_wnv3_clean = pd.concat([df_wnv3_clean, matching_rows], ignore_index=True, axis=0)
                                df['Group'] = 'Dementia'
                                dfs.append(df)
                    except Exception:
                        continue
                    finally:
                        processed_files += 1
                        if total_files > 0:
                            progress_bar.progress(int(processed_files * 100 / total_files))
                # Clear the progress bar after stage load
                progress_bar.empty()

            if not dfs:
                continue

            # Build group-means results like in HEP_plots
            unique_groups = list(set([df['Group'].iloc[0] for df in dfs if 'Group' in df.columns and not df.empty]))
            group_dfs = []
            for group in unique_groups:
                group_patients = [df for df in dfs if 'Group' in df.columns and df['Group'].iloc[0] == group]
                if group_patients:
                    group_mean_df = pd.concat([df.drop(columns=['Group'], errors='ignore').mean() for df in group_patients], axis=1).T
                    group_mean_df['Group'] = group
                    group_dfs.append(group_mean_df)
            if not group_dfs:
                continue

            results_df = pd.concat(group_dfs, ignore_index=True)
            results_df = results_df.loc[:, results_df.isnull().mean() < 0.5]
            results_df = results_df.dropna()

            from scipy.stats import zscore
            numeric_cols = results_df.select_dtypes(include=[np.number]).columns
            results_df[numeric_cols] = results_df[numeric_cols].apply(zscore)

            if size_feature:
                if size_feature in results_df.columns:
                    results_df = results_df[['Vagal_SD1', 'Sympathetic_SD2', 'Efficiency', 'Clustering', 'Modularity', 'Assortativity', size_feature]]

            stage_results[stage] = results_df
            stage_clinical[stage] = df_wnv3_clean

        # Plot HEP plots for all stages side by side
        if any(stage in stage_results for stage in stages):
            st.subheader(f"HEP Plots: All Sleep Stages Comparison (Band: {band_name})")
            
            # Get available stages
            available_stages = [stage for stage in stages if stage in stage_results]
            
            if available_stages:
                # Combine all stages into one DataFrame with stage label
                labels = ['Vagal_SD1', 'Sympathetic_SD2', 'Efficiency', 'Clustering', 'Modularity', 'Assortativity']
                
                combined_dfs = []
                for stage in available_stages:
                    results_df_stage = stage_results[stage].copy()
                    # Keep only the HEP metrics
                    available_labels = [label for label in labels if label in results_df_stage.columns]
                    if available_labels:
                        stage_subset = results_df_stage[available_labels].copy()
                        stage_subset['Stage'] = stage
                        combined_dfs.append(stage_subset)
                
                if combined_dfs:
                    # Combine all stages
                    combined_df = pd.concat(combined_dfs, ignore_index=True)
                    
                    # Get common columns (HEP metrics)
                    hep_columns = [col for col in labels if col in combined_df.columns]
                    
                    if len(hep_columns) >= 2:
                        # Color palette for stages
                        stage_colors = {'N1': 'blue', 'N2': 'green', 'N3': 'orange', 'R': 'red', 'W': 'purple'}
                        
                        # Create palette list matching available_stages order
                        palette_list = [stage_colors.get(stage, 'gray') for stage in available_stages]
                        
                        # Create PairGrid with explicit palette
                        g = sns.PairGrid(combined_df, vars=hep_columns, hue='Stage', 
                                        diag_sharey=False, palette=palette_list, hue_order=available_stages)
                        
                        # LOWER: XY plot with 2 lines per pair (one for each variable) per stage
                        def plot_lower(x, y, **kwargs):
                            ax = kwargs.get('ax', plt.gca())
                            
                            # Get data for each stage
                            # x and y are already subsets from PairGrid, so use their indices
                            stage_data = {}
                            for stage in available_stages:
                                # Get stage mask for the current subset using indices
                                indices = x.index
                                stage_mask = combined_df.loc[indices, 'Stage'].values == stage
                                if stage_mask.any():
                                    # Get x and y values for this stage from the subset
                                    x_vals = x.values[stage_mask]
                                    y_vals = y.values[stage_mask]
                                    mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
                                    x_clean = x_vals[mask]
                                    y_clean = y_vals[mask]
                                    if len(x_clean) > 1:
                                        stage_data[stage] = {'x': x_clean, 'y': y_clean}
                            
                            # Create linspace for X axis
                            all_x_vals = []
                            all_y_vals = []
                            for stage_data_item in stage_data.values():
                                all_x_vals.extend(stage_data_item['x'])
                                all_y_vals.extend(stage_data_item['y'])
                            
                            if all_x_vals and all_y_vals:
                                # Plot scatter for each stage
                                for stage in available_stages:
                                    if stage in stage_data:
                                        data = stage_data[stage]
                                        x_clean = data['x']
                                        y_clean = data['y']
                                        
                                        # Plot scatter plot
                                        ax.scatter(x_clean, y_clean,
                                                  color=stage_colors.get(stage, 'gray'),
                                                  alpha=0.4,
                                                  s=30,
                                                  label=stage)
                                
                                ax.set_xlabel(x.name)
                                ax.set_ylabel(y.name)
                                ax.set_title(f'{x.name} vs {y.name}')
                                ax.legend(loc='best', fontsize=8)
                                ax.grid(True, alpha=0.3)
                            else:
                                ax.text(0.5, 0.5, 'No data available',
                                       transform=ax.transAxes,
                                       ha='center', va='center',
                                       fontsize=10)
                        
                        # UPPER: Regplot with asterisks based on p-value
                        def plot_upper(x, y, **kwargs):
                            ax = kwargs.get('ax', plt.gca())
                            
                            # Get data and calculate p-values for each stage
                            # x and y are already subsets from PairGrid, so use their indices
                            stage_data = {}
                            for stage in available_stages:
                                # Get stage mask for the current subset using indices
                                indices = x.index
                                stage_mask = combined_df.loc[indices, 'Stage'].values == stage
                                if stage_mask.any():
                                    # Get x and y values for this stage from the subset
                                    x_vals = x.values[stage_mask]
                                    y_vals = y.values[stage_mask]
                                    mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
                                    x_clean = x_vals[mask]
                                    y_clean = y_vals[mask]
                                    if len(x_clean) > 1:
                                        stage_data[stage] = {'x': x_clean, 'y': y_clean}
                                        
                                        # Regplot for this stage
                                        try:
                                            sns.regplot(x=x_clean, y=y_clean, 
                                                      ax=ax,
                                                      color=stage_colors.get(stage, 'gray'),
                                                      scatter=False,
                                                      line_kws={'alpha': 0.8, 'linewidth': 2.5, 'label': stage},
                                                      ci=95)
                                        except:
                                            # Fallback
                                            from scipy import stats
                                            slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
                                            x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
                                            y_line = slope * x_line + intercept
                                            ax.plot(x_line, y_line, 
                                                   color=stage_colors.get(stage, 'gray'),
                                                   linestyle='-', 
                                                   alpha=0.8, 
                                                   linewidth=2.5,
                                                   label=stage)
                            
                            # Calculate p-values and add asterisks
                            from scipy.stats import mannwhitneyu
                            from itertools import combinations
                            
                            p_value_text = []
                            for stage1, stage2 in combinations(available_stages, 2):
                                if stage1 in stage_data and stage2 in stage_data:
                                    data1 = stage_data[stage1]
                                    data2 = stage_data[stage2]
                                    try:
                                        _, p_val = mannwhitneyu(data1['y'], data2['y'], alternative='two-sided')
                                        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                                        p_value_text.append(f"{stage1} vs {stage2}: {p_val:.4f}{sig}")
                                    except:
                                        pass
                            
                            # Display p-values in upper corner
                            if p_value_text:
                                text = '\n'.join(p_value_text[:3])  # Show top 3
                                ax.text(0.98, 0.02, text,
                                       transform=ax.transAxes,
                                       fontsize=7,
                                       ha='right',
                                       va='bottom',
                                       family='monospace',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
                            
                            ax.legend(loc='best', fontsize=8)
                            ax.grid(True, alpha=0.3)
                        
                        # Map function for diagonal (histograms)
                        def plot_diag(x, **kwargs):
                            ax = kwargs.get('ax', plt.gca())
                            indices = x.index
                            
                            for stage in available_stages:
                                stage_mask = combined_df.loc[indices, 'Stage'] == stage
                                if stage_mask.any():
                                    x_vals = x[stage_mask].values
                                    x_clean = x_vals[~np.isnan(x_vals)]
                                    if len(x_clean) > 0:
                                        ax.hist(x_clean, alpha=0.5, 
                                               label=stage, 
                                               color=stage_colors.get(stage, 'gray'),
                                               bins=20)
                            ax.set_ylabel('Frequency')
                        
                        # Apply mapping functions
                        g.map_lower(plot_lower)
                        g.map_diag(plot_diag)
                        g.map_upper(plot_upper)
                        
                        # Add legend with proper color mapping
                        # Create custom legend handles with colors
                        import matplotlib.patches as mpatches
                        handles = [mpatches.Patch(color=stage_colors.get(stage, 'gray'), label=stage) 
                                  for stage in available_stages]
                        # Place legend on the figure, adjust layout to make room
                        g.fig.legend(handles=handles, title='Sleep Stage', 
                                   loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10, frameon=True)
                        
                        # Set title
                        g.fig.suptitle(f'HEP Metrics PairGrid: All Sleep Stages Comparison (Band: {band_name})', 
                                      fontsize=14, fontweight='bold', y=1.02)
                        
                        plt.tight_layout()
                        st.pyplot(g.fig)
                        plt.close(g.fig)
                        
                        # Create p-value table: rows are pairs, columns are stages
                        st.subheader(f"P-Values Table: Pair Comparisons Across Stages (Band: {band_name})")
                        
                        from scipy.stats import mannwhitneyu
                        from itertools import combinations
                        
                        # Get all pairs from hep_columns
                        pair_list = []
                        for i in range(len(hep_columns)):
                            for j in range(i+1, len(hep_columns)):
                                pair_list.append((hep_columns[i], hep_columns[j]))
                        
                        # Create DataFrame for p-values
                        # Rows: pairs, Columns: stage comparisons
                        pvalue_table_data = []
                        for pair_x, pair_y in pair_list:
                            row_data = {'Pair': f"{pair_x} vs {pair_y}"}
                            
                            # Get data for this pair across all stages
                            pair_data = {}
                            for stage in available_stages:
                                stage_mask = combined_df['Stage'] == stage
                                if stage_mask.any():
                                    x_vals = combined_df.loc[stage_mask, pair_x].values
                                    y_vals = combined_df.loc[stage_mask, pair_y].values
                                    # Use y-values for comparison between stages
                                    mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
                                    y_clean = y_vals[mask]
                                    if len(y_clean) > 1:
                                        pair_data[stage] = y_clean
                            
                            # Calculate p-values for each stage combination (columns)
                            stage_combinations = list(combinations(available_stages, 2))
                            for stage1, stage2 in stage_combinations:
                                combo_name = f"{stage1} vs {stage2}"
                                if stage1 in pair_data and stage2 in pair_data:
                                    try:
                                        _, p_val = mannwhitneyu(pair_data[stage1], pair_data[stage2], alternative='two-sided')
                                        # Format with asterisks
                                        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                                        row_data[combo_name] = f"{p_val:.4f}{sig}"
                                    except:
                                        row_data[combo_name] = "N/A"
                                else:
                                    row_data[combo_name] = "N/A"
                            
                            pvalue_table_data.append(row_data)
                        
                        # Create DataFrame and display
                        if pvalue_table_data:
                            pvalue_df = pd.DataFrame(pvalue_table_data)
                            # Set Pair as index for better display
                            pvalue_df = pvalue_df.set_index('Pair')
                            st.dataframe(pvalue_df, use_container_width=True)
                            
                            # Add legend
                            st.caption("*** p < 0.001, ** p < 0.01, * p < 0.05 | P-values compare y-variable distributions between stages for each pair")
                
                # Also call only_plots for each stage individually for detailed view
                st.subheader(f"Detailed HEP Plots per Stage (Band: {band_name})")
                for stage in available_stages:
                    results_df_stage = stage_results[stage]
                    hue_stage = results_df_stage.get('Group', None) if 'Group' in results_df_stage.columns else None
                    
                    st.write(f"**Stage: {stage}**")
                    only_plots(
                        results_df_stage,
                        save_plot='',
                        save_dir='',
                        edf_pickle_name=f"plot_{stage}",
                        band=band_name,
                        step_sec=5,
                        is_streamlit=True,
                        hue=hue_stage,
                        size_feature=size_feature
                    )
                    st.divider()

        # For each default pair, render per-stage comparisons
        for col1, col2 in default_pairs:
            st.subheader(f"Pair: {col1} vs {col2}")

            # Collect importances across stages
            per_stage_importances = {}

            tabs = st.tabs(stages)
            for idx, stage in enumerate(stages):
                with tabs[idx]:
                    if stage not in stage_results:
                        st.info(f"No data for stage {stage}")
                        continue

                    results_df = stage_results[stage]
                    df_wnv3_clean = stage_clinical.get(stage, pd.DataFrame())
                    if col1 not in results_df.columns or col2 not in results_df.columns:
                        st.warning(f"Missing features in {stage}")
                        continue

                    pair_df = results_df[[col1, col2]].dropna()
                    if pair_df.empty:
                        st.warning("No data for this pair")
                        continue

                    # Clean outliers: remove rows where either feature is beyond 3 SD
                    # Compute per-column mean and std, handle zero std safely
                    means = pair_df.mean()
                    stds = pair_df.std(ddof=0).replace(0, np.nan)
                    zscores = (pair_df - means) / stds
                    mask_inliers = (zscores[col1].abs() <= 3) & (zscores[col2].abs() <= 3)
                    pair_df_clean = pair_df[mask_inliers].dropna()
                    if pair_df_clean.shape[0] < 2:
                        st.warning("Not enough inlier data after 3-SD cleaning for this pair")
                        continue

                    # Cluster like plot_upper_cluster on cleaned data
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(pair_df_clean)
                    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(X_scaled)

                    # Layout: scatter (left) and feature-importance (right)
                    c1, c2 = st.columns(2)
                    with c1:
                        fig, ax = plt.subplots(figsize=(5, 4))
                        sns.scatterplot(x=pair_df_clean[col1], y=pair_df_clean[col2], hue=clusters, palette='viridis', ax=ax)
                        ax.set_title(f"{stage} clusters")
                        ax.set_xlabel(col1)
                        ax.set_ylabel(col2)
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close(fig)

                    # Clinical feature-importance predicting clusters
                    importance_df = None
                    if not df_wnv3_clean.empty:
                        # Align indices if possible; fallback to using all rows
                        clinical = df_wnv3_clean.select_dtypes(include=[np.number])
                        # Clean clinical
                        if not clinical.empty:
                            thresh = max(1, int(clinical.shape[0] / 2))
                            clinical = clinical.dropna(axis=1, thresh=thresh)
                            clinical = clinical.fillna(clinical.median())
                            # Truncate/align lengths to cleaned pair rows count
                            min_len = min(len(clinical), len(pair_df_clean))
                            X = clinical.iloc[:min_len, :]
                            y = pd.Series(clusters[:min_len])
                            try:
                                rf = RandomForestClassifier(n_estimators=200, random_state=42)
                                rf.fit(X, y)
                                importance_df = pd.DataFrame({
                                    'feature': X.columns,
                                    'importance': rf.feature_importances_
                                }).sort_values('importance', ascending=False)
                                per_stage_importances[stage] = importance_df.copy()
                            except Exception:
                                pass

                    with c2:
                        if importance_df is not None and not importance_df.empty:
                            top = importance_df.head(10)
                            fig, ax = plt.subplots(figsize=(5, 4))
                            sns.barplot(data=top, x='importance', y='feature', ax=ax)
                            ax.set_title(f"{stage} feature importance")
                            ax.set_xlabel('Importance')
                            ax.set_ylabel('Feature')
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                            
                            # Aggregate by electrode for this stage
                            electrode_importance = {}
                            for electrode in eeg_channels:
                                electrode_importance[electrode] = 0.0
                                for idx, feature in enumerate(importance_df['feature']):
                                    if isinstance(feature, str) and electrode in feature:
                                        electrode_importance[electrode] += importance_df.iloc[idx]['importance']
                            
                            electrode_df = pd.DataFrame({
                                'electrode': list(electrode_importance.keys()),
                                'total_importance': list(electrode_importance.values())
                            }).sort_values('total_importance', ascending=False)
                            
                        else:
                            st.info("Feature importance unavailable for this stage")

            # Group-based electrode importance analysis
            st.subheader("Group-Based Electrode Importance Analysis")
            
            # Get available clinical columns from the first stage that has data
            available_clinical_cols = []
            for stage in stages:
                if stage in stage_clinical and not stage_clinical[stage].empty:
                    numeric_cols = stage_clinical[stage].select_dtypes(include=[np.number]).columns.tolist()
                    # Filter out metadata columns
                    numeric_cols = [col for col in numeric_cols if col not in ['end_time', 'segment', 'start_time', 'duration_min', 'number_of_signals', 'ID']]
                    available_clinical_cols = numeric_cols
                    break
            
            if available_clinical_cols:
                # Let user select a column (default to clinical_moca if available)
                default_index = 0
                if 'clinical_moca' in available_clinical_cols:
                    default_index = available_clinical_cols.index('clinical_moca')
                
                selected_col = st.selectbox(
                    "Select clinical column for group comparison:",
                    options=available_clinical_cols,
                    index=default_index,
                    key=f'clinical_col_selection_{band_name}_{col1}_{col2}'
                )
                
                if selected_col:
                    # Determine threshold based on column type
                    threshold = None
                    group_name_high = "High"
                    group_name_low = "Low"
                    
                    if selected_col in ['clinical_moca', 'clinical_mmse']:
                        threshold = 18
                        group_name_high = f"{selected_col} > 18"
                        group_name_low = f"{selected_col} <= 18"
                    else:
                        # Check if binary (only 0 and 1)
                        all_values = []
                        for stage in stages:
                            if stage in stage_clinical and not stage_clinical[stage].empty:
                                col_values = stage_clinical[stage][selected_col].dropna().unique()
                                all_values.extend(col_values)
                        unique_vals = set([float(v) for v in all_values])
                        
                        if len(unique_vals) == 2 and unique_vals == {0.0, 1.0}:
                            threshold = 0.5  # Binary threshold
                            group_name_high = f"{selected_col} = 1"
                            group_name_low = f"{selected_col} = 0"
                        else:
                            # Ask user for threshold
                            threshold = st.number_input(
                                f"Enter threshold for {selected_col}:",
                                value=float(np.median([v for v in all_values if not pd.isna(v)])) if all_values else 0.0,
                                key=f'threshold_{band_name}_{selected_col}_{col1}_{col2}'
                            )
                            group_name_high = f"{selected_col} > {threshold}"
                            group_name_low = f"{selected_col} <= {threshold}"
                    
                    # Calculate electrode importance for each group across all stages
                    group_importances = {group_name_high: {}, group_name_low: {}}
                    group_counts = {group_name_high: 0, group_name_low: 0}  # Track sample sizes
                    
                    for stage in stages:
                        if stage not in stage_results or stage not in stage_clinical:
                            continue
                        
                        results_df = stage_results[stage]
                        df_wnv3_clean = stage_clinical[stage]
                        
                        if selected_col not in df_wnv3_clean.columns:
                            continue
                        
                        if col1 not in results_df.columns or col2 not in results_df.columns:
                            continue
                        
                        pair_df = results_df[[col1, col2]].dropna()
                        if pair_df.empty:
                            continue
                        
                        # Clean outliers
                        means = pair_df.mean()
                        stds = pair_df.std(ddof=0).replace(0, np.nan)
                        zscores = (pair_df - means) / stds
                        mask_inliers = (zscores[col1].abs() <= 3) & (zscores[col2].abs() <= 3)
                        pair_df_clean = pair_df[mask_inliers].dropna()
                        if pair_df_clean.shape[0] < 2:
                            continue
                        
                        # Get clinical features and align indices
                        clinical = df_wnv3_clean.select_dtypes(include=[np.number])
                        if clinical.empty:
                            continue
                        
                        # Clean clinical data
                        thresh = max(1, int(clinical.shape[0] / 2))
                        clinical = clinical.dropna(axis=1, thresh=thresh)
                        clinical = clinical.fillna(clinical.median())
                        
                        # Align clinical data with pair_df_clean by length (they may have different indices)
                        # Use positional indexing since indices may not match
                        min_len = min(len(pair_df_clean), len(clinical))
                        if min_len < 2:
                            continue
                        
                        # Get clinical column values aligned by position
                        clinical_col_data = df_wnv3_clean[selected_col].dropna()
                        min_len_col = min(min_len, len(clinical_col_data))
                        if min_len_col < 2:
                            continue
                        
                        # Get group masks based on clinical column values (using positional indexing)
                        clinical_col_values = clinical_col_data.iloc[:min_len_col].values
                        
                        if threshold == 0.5:  # Binary case
                            high_mask_bool = (clinical_col_values == 1) | (clinical_col_values == 1.0)
                            low_mask_bool = (clinical_col_values == 0) | (clinical_col_values == 0.0)
                        else:
                            high_mask_bool = clinical_col_values > threshold
                            low_mask_bool = clinical_col_values <= threshold
                        
                        # Process each group
                        for group_name, mask_bool in [(group_name_high, high_mask_bool), (group_name_low, low_mask_bool)]:
                            if mask_bool.sum() < 2:  # Need at least 2 samples
                                continue
                            
                            # Use boolean mask on aligned data (using positional indexing)
                            # Align all data to the same length
                            align_len = min(len(mask_bool), min_len)
                            mask_bool_aligned = mask_bool[:align_len]
                            
                            # Get aligned data slices
                            clinical_aligned = clinical.iloc[:align_len]
                            pair_df_aligned = pair_df_clean.iloc[:align_len]
                            
                            # Apply mask
                            X_group = clinical_aligned[mask_bool_aligned]
                            pair_df_group = pair_df_aligned[mask_bool_aligned]
                            
                            if len(pair_df_group) < 2 or len(X_group) < 2:
                                continue
                            
                            # Track sample size for this group
                            group_counts[group_name] += len(pair_df_group)
                            
                            # Cluster for this group
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(pair_df_group)
                            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                            clusters = kmeans.fit_predict(X_scaled)
                            
                            # Calculate feature importance
                            try:
                                rf = RandomForestClassifier(n_estimators=200, random_state=42)
                                rf.fit(X_group, clusters)
                                importance_df = pd.DataFrame({
                                    'feature': X_group.columns,
                                    'importance': rf.feature_importances_
                                })
                                
                                # Aggregate by electrode
                                for electrode in eeg_channels:
                                    if electrode not in group_importances[group_name]:
                                        group_importances[group_name][electrode] = 0.0
                                    for idx, feature in enumerate(importance_df['feature']):
                                        if isinstance(feature, str) and electrode in feature:
                                            group_importances[group_name][electrode] += importance_df.iloc[idx]['importance']
                            except Exception:
                                continue
                    
                    # Display group comparisons
                    if group_importances[group_name_high] or group_importances[group_name_low]:
                        st.write(f"**Electrode Importance Comparison: {group_name_high} vs {group_name_low}**")
                        
                        # Create DataFrames for both groups
                        all_electrodes = set()
                        all_electrodes.update(group_importances[group_name_high].keys())
                        all_electrodes.update(group_importances[group_name_low].keys())
                        
                        electrode_comparison_data = []
                        for electrode in all_electrodes:
                            high_imp = group_importances[group_name_high].get(electrode, 0.0)
                            low_imp = group_importances[group_name_low].get(electrode, 0.0)
                            electrode_comparison_data.append({
                                'electrode': electrode,
                                group_name_high: high_imp,
                                group_name_low: low_imp
                            })
                        
                        comparison_df = pd.DataFrame(electrode_comparison_data)
                        comparison_df = comparison_df[(comparison_df[group_name_high] > 0) | (comparison_df[group_name_low] > 0)]
                        comparison_df = comparison_df.sort_values(group_name_high, ascending=False)
                        
                        if not comparison_df.empty:
                            # Bar plot comparison
                            fig, ax = plt.subplots(figsize=(10, 6))
                            x = np.arange(len(comparison_df.head(15)))
                            width = 0.35
                            ax.bar(x - width/2, comparison_df[group_name_high].head(15), width, label=group_name_high)
                            ax.bar(x + width/2, comparison_df[group_name_low].head(15), width, label=group_name_low)
                            ax.set_xlabel('Electrode')
                            ax.set_ylabel('Total Importance')
                            n_high = group_counts.get(group_name_high, 0)
                            n_low = group_counts.get(group_name_low, 0)
                            ax.set_title(f'Top 15 Electrodes: {group_name_high} (N={n_high}) vs {group_name_low} (N={n_low})')
                            ax.set_xticks(x)
                            ax.set_xticklabels(comparison_df['electrode'].head(15), rotation=45, ha='right')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                            
                            # Topomaps for each group
                            col_top1, col_top2 = st.columns(2)
                            
                            with col_top1:
                                high_dict = dict(zip(comparison_df['electrode'], comparison_df[group_name_high]))
                                topomap_high = plot_electrode_importance_topomap(
                                    high_dict,
                                    title=f'{group_name_high} Electrode Importance'
                                )
                                if topomap_high:
                                    st.pyplot(topomap_high)
                                    plt.close(topomap_high)
                                else:
                                    st.info("Topomap unavailable")
                            
                            with col_top2:
                                low_dict = dict(zip(comparison_df['electrode'], comparison_df[group_name_low]))
                                topomap_low = plot_electrode_importance_topomap(
                                    low_dict,
                                    title=f'{group_name_low} Electrode Importance'
                                )
                                if topomap_low:
                                    st.pyplot(topomap_low)
                                    plt.close(topomap_low)
                                else:
                                    st.info("Topomap unavailable")
                            
                            # P-value topomaps
                            col_pval1, col_pval2 = st.columns(2)
                            
                            with col_pval1:
                                pval_topomap_high = plot_electrode_pvalue_topomap(
                                    high_dict,
                                    title=f'{group_name_high} P-Value Topomap'
                                )
                                if pval_topomap_high:
                                    st.pyplot(pval_topomap_high)
                                    plt.close(pval_topomap_high)
                                else:
                                    st.info("P-value topomap unavailable")
                            
                            with col_pval2:
                                pval_topomap_low = plot_electrode_pvalue_topomap(
                                    low_dict,
                                    title=f'{group_name_low} P-Value Topomap'
                                )
                                if pval_topomap_low:
                                    st.pyplot(pval_topomap_low)
                                    plt.close(pval_topomap_low)
                                else:
                                    st.info("P-value topomap unavailable")
            else:
                st.info("No clinical columns available for group comparison")

            # Averaged and ranked importance across stages
            if per_stage_importances:
                # Combine importances, fill missing with 0, average
                all_feats = sorted({f for df_imp in per_stage_importances.values() for f in df_imp['feature']})
                avg_series = pd.Series(0.0, index=all_feats)
                for imp_df in per_stage_importances.values():
                    s = imp_df.set_index('feature')['importance']
                    s = s.reindex(all_feats).fillna(0.0)
                    avg_series = avg_series.add(s, fill_value=0.0)
                avg_series = avg_series / len(per_stage_importances)
                avg_df = avg_series.sort_values(ascending=False).reset_index()
                avg_df.columns = ['feature', 'avg_importance']

                st.markdown("**Average feature importance across stages**")
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.barplot(data=avg_df.head(15), x='avg_importance', y='feature', ax=ax)
                ax.set_xlabel('Average Importance')
                ax.set_ylabel('Feature')
                st.pyplot(fig)
                plt.close(fig)
                
                # Aggregate averaged importance by electrode
                electrode_avg_importance = {}
                for electrode in eeg_channels:
                    electrode_avg_importance[electrode] = 0.0
                    for idx, feature in enumerate(avg_df['feature']):
                        if isinstance(feature, str) and electrode in feature:
                            electrode_avg_importance[electrode] += avg_df.iloc[idx]['avg_importance']
                
                electrode_avg_df = pd.DataFrame({
                    'electrode': list(electrode_avg_importance.keys()),
                    'avg_total_importance': list(electrode_avg_importance.values())
                }).sort_values('avg_total_importance', ascending=False)
                
                electrode_avg_df_nonzero = electrode_avg_df[electrode_avg_df['avg_total_importance'] > 0]
                top_electrodes_avg = electrode_avg_df_nonzero.head(10)
                
                if not top_electrodes_avg.empty:
                    st.markdown("**Top 10 EEG Electrodes by Average Importance Across All Stages**")
                    
                    # Create two columns: bar plot and topomap
                    col_avg_bar, col_avg_topomap = st.columns(2)
                    
                    with col_avg_bar:
                        fig, ax = plt.subplots(figsize=(7, 5))
                        sns.barplot(data=top_electrodes_avg, x='avg_total_importance', y='electrode', ax=ax)
                        ax.set_title('Top 10 EEG Electrodes (Averaged Across Stages)')
                        ax.set_xlabel('Average Total Importance')
                        ax.set_ylabel('EEG Electrode')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    with col_avg_topomap:
                        # Create topomap for averaged importance using all non-zero electrodes
                        electrode_avg_dict = dict(zip(electrode_avg_df_nonzero['electrode'], electrode_avg_df_nonzero['avg_total_importance']))
                        topomap_avg_fig = plot_electrode_importance_topomap(
                            electrode_avg_dict,
                            title='Average Electrode Importance Topomap\n(Across All Stages)'
                        )
                        if topomap_avg_fig:
                            st.pyplot(topomap_avg_fig)
                            plt.close(topomap_avg_fig)
                        else:
                            st.info("Topomap unavailable (insufficient channels or positioning data)")
                    
                    # P-value topomap for averaged importance: compare each electrode vs all others
                    electrode_avg_dict = dict(zip(electrode_avg_df_nonzero['electrode'], electrode_avg_df_nonzero['avg_total_importance']))
                    pvalue_topomap_avg_fig = plot_electrode_pvalue_topomap(
                        electrode_avg_dict,
                        title='Average Electrode P-Value Topomap (Each vs All Others)\n(Across All Stages)'
                    )
                    if pvalue_topomap_avg_fig:
                        st.pyplot(pvalue_topomap_avg_fig)
                        plt.close(pvalue_topomap_avg_fig)
                    else:
                        st.info("P-value topomap unavailable (insufficient channels or positioning data)")

def pair_clustering_analysis(results_df, df_wnv3_clean, n_clusters=2, context_key=None):
    """
    Perform clustering analysis on pairs of columns from results_df and classify clusters using group information.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with clustering features and Group column
    df_wnv3_clean : pd.DataFrame  
        DataFrame with clinical features for classification
    n_clusters : int
        Number of clusters for K-means (default: 2)
    """
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import seaborn as sns
    from itertools import combinations
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    

    
    # Classifier selection
    st.subheader("Classifier Configuration")
    classifier_options = {
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'LightGBM': LGBMClassifier(random_state=42),
        'MLP (Feedforward Neural Net)': MLPClassifier(random_state=42, max_iter=1000),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Stacking Classifier': StackingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingClassifier(random_state=42)),
                ('xgb', XGBClassifier(random_state=42))
            ],
            final_estimator=LogisticRegression(random_state=42)
        ),
    }
    # let user choose
    key_suffix = context_key or 'default'
    selected_classifier = st.selectbox('Select classifier', list(classifier_options.keys()), key=f'pair_clustering_classifier_{key_suffix}')
    
    classifier = classifier_options[selected_classifier]
    
    # Get numeric columns from results_df (excluding Group)
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Group' in numeric_cols:
        numeric_cols.remove('Group')
    
    if len(numeric_cols) < 2:
        st.error("Not enough numeric columns for pair analysis")
        return
    
    st.subheader("Pair-wise Clustering Analysis")
    st.write(f"Analyzing {len(numeric_cols)} features in pairs with {n_clusters} clusters each")
    st.write(f"Using classifier: **{selected_classifier}**")
    
    # Create all possible pairs
    all_pairs = list(combinations(numeric_cols, 2))
    total_pairs = len(all_pairs)
    st.write(f"Total pairs to analyze: {total_pairs}")
    
    # Add checkbox to run all pairs
    run_all_pairs = st.checkbox("Run all pairs", value=False)
    
    # Limit to 4 specific pairs unless checkbox is checked
    if not run_all_pairs:
        # Define the specific 4 pairs to run
        default_pairs = [
            ('Vagal_SD1', 'Efficiency'),
            ('Vagal_SD1', 'Clustering'),
            ('Vagal_SD1', 'Modularity'),
            ('Vagal_SD1', 'Assortativity')
        ]
        
        # Filter to only include pairs that exist in numeric_cols
        pairs = []
        missing_features = []
        for col1, col2 in default_pairs:
            if col1 in numeric_cols and col2 in numeric_cols:
                if (col1, col2) in all_pairs:
                    pairs.append((col1, col2))
                elif (col2, col1) in all_pairs:
                    pairs.append((col2, col1))
                else:
                    missing_features.append(f"{col1} vs {col2}")
            else:
                missing_features.append(f"{col1} vs {col2}")
        
        if missing_features:
            st.warning(f"Some requested pairs not available: {', '.join(missing_features)}")
        
        if pairs:
            st.info(f"Running {len(pairs)} specific pairs. Check 'Run all pairs' to analyze all {total_pairs} pairs.")
        else:
            st.warning("None of the requested pairs are available. Running first 2 pairs instead.")
            pairs = all_pairs[:2]
    else:
        pairs = all_pairs
    
    # Create columns for displaying results
    col1, col2 = st.columns(2)
    
    results_summary = []
    # Dictionary to store top 10 features per pair for overall aggregation
    all_pair_features = {}  # {pair_name: {feature: importance}}
    # cluster_labels is SD1 SD2 labels
    # Perform K-means clustering
    X_pair = results_df[['Sympathetic_SD2', 'Vagal_SD1']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pair)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    for i, (col1_name, col2_name) in enumerate(pairs):
        with st.expander(f"Pair {i+1}: {col1_name} vs {col2_name}", expanded=False):
            with st.spinner(f"Processing pair {i+1}: {col1_name} vs {col2_name}..."):
                # Prepare data for this pair
                pair_data = results_df[[col1_name, col2_name, 'Group']].dropna()
                
                if len(pair_data) < n_clusters:
                    st.warning(f"Not enough data points for pair {col1_name} vs {col2_name}")
                    continue
                # Add cluster labels to data
                pair_data_with_clusters = pair_data.copy()
                pair_data_with_clusters['cluster'] = cluster_labels
                # Find common indices between pair data and clinical data
                common_indices = pair_data_with_clusters.index.intersection(df_wnv3_clean.index)
                
                if len(common_indices) > 0:
                    # Get the pair features
                    pair_features = pair_data_with_clusters.loc[common_indices, [col1_name, col2_name, 'Group', 'cluster']]
                    
                    # Get clinical features (select a subset of important ones to avoid too wide table)
                    clinical_subset = df_wnv3_clean.loc[common_indices].select_dtypes(include=[np.number])
                    
                    # If too many clinical features, select top 10 most variable ones
                    if clinical_subset.shape[1] > 10:
                        clinical_variance = clinical_subset.var().sort_values(ascending=False)
                        clinical_subset = clinical_subset[clinical_variance.head(10).index]
                    
                    # Combine the data
                    combined_data = pd.concat([pair_features, clinical_subset], axis=1)
                    
                    # Add cluster column with proper naming
                    combined_data['Cluster_Assignment'] = combined_data['cluster'].map({i: f'Cluster_{i}' for i in range(n_clusters)})
                    
                    # Reorder columns to put cluster assignment near the beginning
                    cols = ['Cluster_Assignment', col1_name, col2_name, 'Group'] + [col for col in combined_data.columns if col not in ['Cluster_Assignment', col1_name, col2_name, 'Group', 'cluster']]
                    combined_data = combined_data[cols]
                    
                    st.write(f"Showing data for {len(combined_data)} samples with common indices")
                    st.dataframe(combined_data, use_container_width=True)
                    
                    # Show cluster distribution
                    st.subheader("Cluster Distribution")
                    cluster_dist = combined_data['Cluster_Assignment'].value_counts()
                    st.dataframe(cluster_dist.to_frame('Count'))
                    
                else:
                    st.warning("No common indices found between pair data and clinical data")
                
                # Create scatter plot with clusters - color each cluster differently
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                unique_clusters = pair_data_with_clusters['cluster'].unique()
                # Use a vibrant color palette for clusters
                color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
                cluster_colors = {cluster: color_palette[i % len(color_palette)] for i, cluster in enumerate(unique_clusters)}
                
                for cluster in unique_clusters:
                    mask = pair_data_with_clusters['cluster'] == cluster
                    ax.scatter(pair_data_with_clusters.loc[mask, col1_name], 
                               pair_data_with_clusters.loc[mask, col2_name],
                               c=[cluster_colors[cluster]], label=f'Cluster {cluster}', alpha=0.7, s=50)
                
                # Add cluster centers
                centers = scaler.inverse_transform(kmeans.cluster_centers_)
                ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
                
                ax.set_xlabel(col1_name)
                ax.set_ylabel(col2_name)
                ax.set_title(f'Clustering Results: {col1_name} vs {col2_name}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
                # Feature importance analysis - predict clusters using clinical features
                st.subheader("Feature Importance Analysis - Clinical Features Predicting Clusters")
                
                # Use clinical features from df_wnv3_clean to predict clusters
                try:
                    # Find common indices between results_df and df_wnv3_clean
                    common_indices = pair_data_with_clusters.index.intersection(df_wnv3_clean.index)
                    
                    if len(common_indices) < 10:  # Need sufficient data
                        # raise using func
                        raise_error(f"Not enough common data points ({len(common_indices)}) for clinical feature analysis")
                        return
                    # Get clinical features (numeric columns from df_wnv3_clean)
                    clinical_features = df_wnv3_clean.select_dtypes(include=[np.number]).columns.tolist()
                    if len(clinical_features) == 0:
                        st.warning("No numeric clinical features found in df_wnv3_clean")
                        continue
                    # remove clinical features end_time, segmnent, start_time, duration_min, number_of_signals
                    clinical_features = [feature for feature in clinical_features if feature not in ['end_time', 'segment', 'start_time', 'duration_min', 'number_of_signals']]
                    # Prepare data for classification
                    X_clinical = df_wnv3_clean[clinical_features]
                    # dropna columns where 50% + is Nan. otherwise, use median for the rest
                    X_clinical = X_clinical.dropna(axis=1, thresh=int(X_clinical.shape[0]/2))
                    X_clinical = X_clinical.fillna(X_clinical.median())
                    y_clusters = pair_data_with_clusters['cluster']
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_clinical, y_clusters, test_size=0.3, random_state=42, stratify=y_clusters
                    )
                    
                    # Train selected classifier
                    clf = classifier
                    clf.fit(X_train, y_train)
                    
                    # Get feature importance (if available)
                    if hasattr(clf, 'feature_importances_'):
                        # Align features with their importances
                        # Get the actual features used by the classifier (after dropping NaN columns)
                        actual_features = X_clinical.columns.tolist()
                        actual_importances = clf.feature_importances_
                        
                        st.write(f"Actual features used: {len(actual_features)}")
                        st.write(f"Feature importances: {len(actual_importances)}")
                        
                        # For tree-based models (Random Forest, Gradient Boosting, etc.)
                        importance_df = pd.DataFrame({
                            'feature': actual_features,
                            'importance': actual_importances
                        }).sort_values('importance', ascending=False)
                    elif hasattr(clf, 'coef_'):
                        # For linear models (Logistic Regression, SVM with linear kernel)
                        if len(clf.coef_.shape) > 1:
                            # Multi-class case - use mean of absolute coefficients
                            coef_values = np.mean(np.abs(clf.coef_), axis=0)
                        else:
                            # Binary case
                            coef_values = np.abs(clf.coef_[0])
                        importance_df = pd.DataFrame({
                            'feature': clinical_features,
                            'importance': coef_values
                        }).sort_values('importance', ascending=False)
                    else:
                        # For models without feature importance (KNN, Naive Bayes, etc.)
                        # Use permutation importance as fallback
                        try:
                            from sklearn.inspection import permutation_importance
                            perm_importance = permutation_importance(clf, X_test, y_test, random_state=42)
                            importance_df = pd.DataFrame({
                                'feature': clinical_features,
                                'importance': perm_importance.importances_mean
                            }).sort_values('importance', ascending=False)
                        except:
                            # Final fallback: equal importance
                            importance_df = pd.DataFrame({
                                'feature': clinical_features,
                                'importance': [1.0/len(clinical_features)] * len(clinical_features)
                            })
                    # Run classifier comparison on all features first
                    run_classifier_comparison_on_results(pair_data_with_clusters, X_clinical)
                    
                    # Plot feature importance - show top 10 features
                    top_features = importance_df.head(10)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=top_features, x='importance', y='feature', ax=ax)
                    ax.set_title(f'Top 10 Clinical Features Predicting Clusters ({selected_classifier})\nPair: {col1_name} vs {col2_name}')
                    ax.set_xlabel('Feature Importance')
                    ax.set_ylabel('Clinical Features')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Display importance values
                    st.write("Top Clinical Features Predicting Clusters:")
                    st.dataframe(top_features)
                    
                    # Aggregate feature importance by EEG electrode
                    electrode_importance = {}
                    
                    # For each electrode, find features that contain the electrode name and sum their importance
                    for electrode in eeg_channels:
                        electrode_importance[electrode] = 0.0
                        # Check each feature to see if it contains this electrode name
                        for idx, feature in enumerate(importance_df['feature']):
                            if isinstance(feature, str) and electrode in feature:
                                electrode_importance[electrode] += importance_df.iloc[idx]['importance']
                    
                    # Create DataFrame for electrode importance
                    electrode_df = pd.DataFrame({
                        'electrode': list(electrode_importance.keys()),
                        'total_importance': list(electrode_importance.values())
                    }).sort_values('total_importance', ascending=False)
                    
                    # Filter out electrodes with zero importance and show top 10
                    electrode_df_nonzero = electrode_df[electrode_df['total_importance'] > 0]
                    top_electrodes = electrode_df_nonzero.head(10)
                    
                    if not top_electrodes.empty:
                        st.subheader("Top 10 EEG Electrodes by Overall Feature Importance")
                        
                        # Create two columns: bar plot and topomap
                        col_bar, col_topomap = st.columns(2)
                        
                        with col_bar:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.barplot(data=top_electrodes, x='total_importance', y='electrode', ax=ax)
                            ax.set_title(f'Top 10 EEG Electrodes Contributing to Cluster Prediction\nPair: {col1_name} vs {col2_name}')
                            ax.set_xlabel('Total Importance (Sum across all features)')
                            ax.set_ylabel('EEG Electrode')
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                        
                        with col_topomap:
                            # Create topomap using all electrodes with non-zero importance (not just top 10)
                            electrode_dict = dict(zip(electrode_df_nonzero['electrode'], electrode_df_nonzero['total_importance']))
                            topomap_fig = plot_electrode_importance_topomap(
                                electrode_dict,
                                title=f'Electrode Importance Topomap\nPair: {col1_name} vs {col2_name}'
                            )
                            if topomap_fig:
                                st.pyplot(topomap_fig)
                                plt.close(topomap_fig)
                            else:
                                st.info("Topomap unavailable (insufficient channels or positioning data)")
                        
                        # P-value topomap: compare each electrode vs all others
                        electrode_dict = dict(zip(electrode_df_nonzero['electrode'], electrode_df_nonzero['total_importance']))
                        pvalue_topomap_fig = plot_electrode_pvalue_topomap(
                            electrode_dict,
                            title=f'Electrode P-Value Topomap (Each vs All Others)\nPair: {col1_name} vs {col2_name}'
                        )
                        if pvalue_topomap_fig:
                            st.pyplot(pvalue_topomap_fig)
                            plt.close(pvalue_topomap_fig)
                        else:
                            st.info("P-value topomap unavailable (insufficient channels or positioning data)")
                        
                        # Display electrode importance table
                        st.write("Electrode Importance (aggregated across all features):")
                        st.dataframe(top_electrodes)
                    else:
                        st.info("No EEG electrode features found in the clinical features.")
                    
                    # Classification report
                    y_pred = clf.predict(X_test)
                    st.write("Classification Report (Predicting Clusters from Clinical Features):")
                    st.text(classification_report(y_test, y_pred))
                    
                    # Confusion matrix with percentages
                    st.subheader("Confusion Matrix with Accuracy Metrics")
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(y_test, y_pred)
                    
                    # Calculate percentages
                    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
                    
                    # Create confusion matrix plot
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Raw counts
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                               xticklabels=[f'Pred Cluster {i}' for i in range(n_clusters)],
                               yticklabels=[f'True Cluster {i}' for i in range(n_clusters)])
                    ax1.set_title('Confusion Matrix (Counts)')
                    ax1.set_xlabel('Predicted Cluster')
                    ax1.set_ylabel('True Cluster')
                    
                    # Percentages
                    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='viridis', ax=ax2,
                               xticklabels=[f'Pred Cluster {i}' for i in range(n_clusters)],
                               yticklabels=[f'True Cluster {i}' for i in range(n_clusters)])
                    ax2.set_title('Confusion Matrix (Percentages)')
                    ax2.set_xlabel('Predicted Cluster')
                    ax2.set_ylabel('True Cluster')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Calculate and display accuracy metrics
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    metrics_df = pd.DataFrame({
                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                        'Value': [accuracy, precision, recall, f1],
                        'Percentage': [f'{accuracy*100:.1f}%', f'{precision*100:.1f}%', 
                                     f'{recall*100:.1f}%', f'{f1*100:.1f}%']
                    })
                    
                    st.write("Model Performance Metrics:")
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Store top 10 features for this pair
                    pair_name = f"{col1_name} vs {col2_name}"
                    top_10_features = importance_df.head(10)
                    all_pair_features[pair_name] = {
                        row['feature']: row['importance'] 
                        for _, row in top_10_features.iterrows()
                    }
                    
                    # Store results for summary
                    results_summary.append({
                        'pair': pair_name,
                        'n_samples': len(X_clinical),
                        'accuracy': clf.score(X_test, y_test),
                        'top_clinical_feature': importance_df.iloc[0]['feature'],
                        'top_importance': importance_df.iloc[0]['importance']
                    })
                    
                except Exception as e:
                    st.error(f"Error in feature importance analysis: {str(e)}")
                
    
    # Define the specific order based on labels array
    labels_order = [
        'Vagal_SD1',
        'Sympathetic_SD2', 
        'Efficiency',
        'Clustering',
        'Modularity',
        'Assortativity'
    ]
    
    # Filter to only include labels that exist in the data
    available_labels = [label for label in labels_order if label in results_df.columns]
    # Prepare data for pairgrid with specific order
    pairgrid_data = results_df[available_labels + ['Group']].dropna()
    # Create clustering results pairgrid with classifier performance groups
    st.subheader("Clustering Results Pairgrid - Classifier Performance Groups")
    
    if len(pairgrid_data) > 0:
        # Prepare X_clinical once for reuse if available
        X_clinical_global = pd.DataFrame()
        classifier_global = None
        if 'df_wnv3_clean' in locals() or 'df_wnv3_clean' in globals():
            try:
                # Find common indices between pairgrid_data and df_wnv3_clean
                common_indices = pairgrid_data.index.intersection(df_wnv3_clean.index)
                if len(common_indices) > 0:
                    clinical_features = df_wnv3_clean.select_dtypes(include=[np.number]).columns.tolist()
                    clinical_features = [f for f in clinical_features if f not in ['end_time', 'segment', 'start_time', 'duration_min', 'number_of_signals']]
                    if len(clinical_features) > 0:
                        X_clinical_temp = df_wnv3_clean.loc[common_indices, clinical_features]
                        X_clinical_temp = X_clinical_temp.dropna(axis=1, thresh=int(X_clinical_temp.shape[0]/2))
                        X_clinical_temp = X_clinical_temp.fillna(X_clinical_temp.median())
                        if len(X_clinical_temp) > 0 and len(common_indices) == len(pairgrid_data):
                            X_clinical_global = X_clinical_temp
                            classifier_global = classifier
            except:
                pass
        
        # Create pairgrid - we'll compute clustering and performance groups per pair inside the plotting functions
        g_cluster = sns.PairGrid(pairgrid_data, diag_sharey=False)
        
        def plot_upper_cluster(x, y, **kwargs):
            """Plot upper triangle with performance group-colored scatter plots."""
            ax = kwargs.get('ax', plt.gca())
            
            # Perform clustering on this specific pair (X_pair)
            X_pair = pd.DataFrame({x.name: x, y.name: y}).dropna()
            if len(X_pair) < n_clusters:
                # Not enough data, just plot without clustering
                ax.scatter(x, y, alpha=0.7, s=50)
                return
            
            scaler_pair = StandardScaler()
            X_scaled_pair = scaler_pair.fit_transform(X_pair)
            
            kmeans_pair = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels_pair = kmeans_pair.fit_predict(X_scaled_pair)
            
            # Get classifier predictions for this pair if available
            performance_groups = None
            if len(X_clinical_global) > 0 and classifier_global is not None:
                try:
                    # Align X_clinical_global with X_pair indices
                    X_clinical_aligned = X_clinical_global.loc[X_pair.index] if len(X_clinical_global) > 0 else pd.DataFrame()
                    if len(X_clinical_aligned) == len(X_pair):
                        classifier_predictions = classifier_global.predict(X_clinical_global)
                        
                        # Create performance groups based on classifier vs K-means clusters
                        performance_groups = []
                        for i, (true_cluster, pred_cluster) in enumerate(zip(cluster_labels_pair, classifier_predictions)):
                            if true_cluster == 1 and pred_cluster == 1:
                                performance_groups.append('TP')
                            elif true_cluster == 0 and pred_cluster == 0:
                                performance_groups.append('TN')
                            elif true_cluster == 0 and pred_cluster == 1:
                                performance_groups.append('FP')
                            elif true_cluster == 1 and pred_cluster == 0:
                                performance_groups.append('FN')
                except:
                    pass
            
            # Use specific colors for TP, TN, FP, FN
            performance_colors = {
                'TP': '#2E8B57',    # Green
                'TN': '#4169E1',    # Blue  
                'FP': '#FF6347',    # Red
                'FN': '#FFD700'     # Gold
            }
            
            # Plot with performance groups if available, otherwise use clusters
            if performance_groups is not None:
                unique_groups = set(performance_groups)
                for group in unique_groups:
                    mask = [g == group for g in performance_groups]
                    if sum(mask) > 0:
                        color = performance_colors.get(group, '#808080')  # Default gray
                        ax.scatter(X_pair.iloc[mask, 0], X_pair.iloc[mask, 1], 
                                  c=[color], 
                                  alpha=0.7, s=50)
            else:
                # Plot by cluster if no performance groups
                unique_clusters = np.unique(cluster_labels_pair)
                color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
                for cluster in unique_clusters:
                    mask = cluster_labels_pair == cluster
                    color = color_palette[cluster % len(color_palette)]
                    ax.scatter(X_pair.iloc[mask, 0], X_pair.iloc[mask, 1], 
                              c=[color], 
                              alpha=0.7, s=50)
            
            # Add cluster centers
            if len(np.unique(x)) > 1 and len(np.unique(y)) > 1:
                try:
                    centers = scaler_pair.inverse_transform(kmeans_pair.cluster_centers_)
                    ax.scatter(centers[:, 0], centers[:, 1], 
                             c='red', marker='x', s=200, linewidths=3, 
                             label='Centroids')
                except:
                    pass
        
        def plot_lower_cluster(x, y, **kwargs):
            """Plot lower triangle with performance group-colored scatter plots."""
            ax = kwargs.get('ax', plt.gca())
            
            # Perform clustering on this specific pair (X_pair)
            X_pair = pd.DataFrame({x.name: x, y.name: y}).dropna()
            if len(X_pair) < n_clusters:
                # Not enough data, just plot without clustering
                ax.scatter(x, y, alpha=0.6, s=30)
                return
            
            scaler_pair = StandardScaler()
            X_scaled_pair = scaler_pair.fit_transform(X_pair)
            
            kmeans_pair = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels_pair = kmeans_pair.fit_predict(X_scaled_pair)
            
            # Get classifier predictions for this pair if available
            performance_groups = None
            if len(X_clinical_global) > 0 and classifier_global is not None:
                try:
                    # Align X_clinical_global with X_pair indices
                    X_clinical_aligned = X_clinical_global.loc[X_pair.index] if len(X_clinical_global) > 0 else pd.DataFrame()
                    if len(X_clinical_aligned) == len(X_pair) and len(X_clinical_aligned) > 0:
                        classifier_predictions = classifier_global.predict(X_clinical_aligned)
                        
                        # Create performance groups
                        performance_groups = []
                        for i, (true_cluster, pred_cluster) in enumerate(zip(cluster_labels_pair, classifier_predictions)):
                            if true_cluster == 1 and pred_cluster == 1:
                                performance_groups.append('TP')
                            elif true_cluster == 0 and pred_cluster == 0:
                                performance_groups.append('TN')
                            elif true_cluster == 0 and pred_cluster == 1:
                                performance_groups.append('FP')
                            elif true_cluster == 1 and pred_cluster == 0:
                                performance_groups.append('FN')
                except:
                    pass
            
            performance_colors = {
                'TP': '#2E8B57',    # Green
                'TN': '#4169E1',    # Blue  
                'FP': '#FF6347',    # Red
                'FN': '#FFD700'     # Gold
            }
            
            # Plot with performance groups if available, otherwise use clusters
            if performance_groups is not None:
                unique_groups = set(performance_groups)
                for group in unique_groups:
                    mask = [g == group for g in performance_groups]
                    if sum(mask) > 0:
                        color = performance_colors.get(group, '#808080')  # Default gray
                        ax.scatter(X_pair.iloc[mask, 0], X_pair.iloc[mask, 1], 
                                  c=[color], 
                                  alpha=0.6, s=30)
            else:
                # Plot by cluster if no performance groups
                unique_clusters = np.unique(cluster_labels_pair)
                color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
                for cluster in unique_clusters:
                    mask = cluster_labels_pair == cluster
                    color = color_palette[cluster % len(color_palette)]
                    ax.scatter(X_pair.iloc[mask, 0], X_pair.iloc[mask, 1], 
                              c=[color], 
                              alpha=0.6, s=30)
        
        def plot_diag_cluster(x, **kwargs):
            """Plot diagonal with histograms for each performance group."""
            ax = kwargs.get('ax', plt.gca())
            
            # For diagonal, we need to get the pair context - use the x feature with itself
            # But since it's diagonal, we just show the distribution
            # Try to get performance groups if we can align with X_clinical_global
            performance_colors = {
                'TP': '#2E8B57',    # Green
                'TN': '#4169E1',    # Blue  
                'FP': '#FF6347',    # Red
                'FN': '#FFD700'     # Gold
            }
            
            # For diagonal plots, we'll just show a simple histogram
            # since we don't have a y feature for this pair
            try:
                # We can't compute clustering without a pair, so just show regular histogram
                ax.hist(x.dropna(), alpha=0.6, bins=15, density=True)
            except:
                pass
        
        # Apply the plotting functions
        g_cluster.map_upper(plot_upper_cluster)
        g_cluster.map_lower(plot_lower_cluster)
        g_cluster.map_diag(plot_diag_cluster)
        
        # Set title
        g_cluster.fig.suptitle(f'Classifier Performance Groups - Per Pair Analysis\nN={len(pairgrid_data)}, Performance Groups: TP, TN, FP, FN', 
                              fontsize=16, y=1.02)
        
        # Add single master legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2E8B57', label='TP (True Positive)'),
            Patch(facecolor='#4169E1', label='TN (True Negative)'),
            Patch(facecolor='#FF6347', label='FP (False Positive)'),
            Patch(facecolor='#FFD700', label='FN (False Negative)')
        ]
        g_cluster.fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=10)
        
        plt.tight_layout()
        st.pyplot(g_cluster.fig)
        plt.close(g_cluster.fig)
        
        # Note: Performance groups are computed per pair, so we can't aggregate them into a single summary
        st.info("Note: Clustering and performance groups are computed independently for each feature pair. Each plot shows TP, TN, FP, FN based on the specific pair's clustering results.")
    
    # Summary table
    if results_summary:
        st.subheader("Analysis Summary")
        summary_df = pd.DataFrame(results_summary)
        st.dataframe(summary_df)
        
        # Plot summary
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy by pair
        sns.barplot(data=summary_df, x='accuracy', y='pair', ax=ax1)
        ax1.set_title('Classification Accuracy by Feature Pair')
        ax1.set_xlabel('Accuracy')
        
        # Top clinical feature importance
        sns.barplot(data=summary_df, x='top_importance', y='pair', ax=ax2)
        ax2.set_title('Top Clinical Feature Importance by Pair')
        ax2.set_xlabel('Importance')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Aggregate top 10 features across all pairs
        if all_pair_features:
            st.subheader("Overall Best Clinical Features (Aggregated Across All Pairs)")
            
            # Aggregate feature scores across all pairs
            feature_total_scores = {}  # {feature: total_rank_score}
            feature_appearances = {}  # {feature: count_of_appearances}
            importance_sums = {}  # {feature: sum_of_importance_values}
            
            # Score each feature: 10 points for rank 1, 9 for rank 2, ..., 1 for rank 10
            # Also track total importance scores
            for pair_name, features_dict in all_pair_features.items():
                sorted_features = sorted(features_dict.items(), key=lambda x: x[1], reverse=True)
                for rank, (feature, importance) in enumerate(sorted_features, start=1):
                    score = 11 - rank  # 10 for rank 1, 9 for rank 2, etc.
                    
                    if feature not in feature_total_scores:
                        feature_total_scores[feature] = 0
                        feature_appearances[feature] = 0
                        importance_sums[feature] = 0.0
                    
                    feature_total_scores[feature] += score
                    feature_appearances[feature] += 1
                    importance_sums[feature] += importance
            
            # Create summary dataframe
            overall_features_df = pd.DataFrame({
                'feature': list(feature_total_scores.keys()),
                'total_rank_score': [feature_total_scores[f] for f in feature_total_scores.keys()],
                'appearances': [feature_appearances[f] for f in feature_total_scores.keys()],
                'avg_rank_score': [feature_total_scores[f] / feature_appearances[f] for f in feature_total_scores.keys()],
                'total_importance': [importance_sums[f] for f in feature_total_scores.keys()],
                'avg_importance': [importance_sums[f] / feature_appearances[f] for f in feature_total_scores.keys()]
            }).sort_values('total_rank_score', ascending=False)
            
            # Display table
            st.write("**Top Clinical Features Across All Pairs (sorted by total rank score)**")
            st.dataframe(overall_features_df, use_container_width=True)
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
            
            # Plot 1: Total rank score (top 20 features)
            top_20_by_score = overall_features_df.head(20)
            sns.barplot(data=top_20_by_score, y='feature', x='total_rank_score', ax=ax1, palette='viridis')
            ax1.set_title(f'Top 20 Clinical Features - Total Rank Score Across All {len(all_pair_features)} Pairs\n(Rank scoring: 10 points for #1, 9 for #2, ..., 1 for #10)', 
                         fontsize=12, fontweight='bold')
            ax1.set_xlabel('Total Rank Score', fontsize=11)
            ax1.set_ylabel('Clinical Feature', fontsize=11)
            ax1.grid(axis='x', alpha=0.3)
            
            # Plot 2: Average importance (top 20 features)
            top_20_by_importance = overall_features_df.sort_values('avg_importance', ascending=False).head(20)
            sns.barplot(data=top_20_by_importance, y='feature', x='avg_importance', ax=ax2, palette='plasma')
            ax2.set_title(f'Top 20 Clinical Features - Average Importance Across All {len(all_pair_features)} Pairs', 
                         fontsize=12, fontweight='bold')
            ax2.set_xlabel('Average Importance Score', fontsize=11)
            ax2.set_ylabel('Clinical Feature', fontsize=11)
            ax2.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # Additional summary statistics
            st.subheader("Summary Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Pairs Analyzed", len(all_pair_features))
            with col2:
                st.metric("Total Unique Features", len(feature_total_scores))
            with col3:
                st.metric("Most Frequent Feature", 
                         overall_features_df.sort_values('appearances', ascending=False).iloc[0]['feature'],
                         f"appears in {overall_features_df.sort_values('appearances', ascending=False).iloc[0]['appearances']} pairs")


def run_classifier_comparison_on_results(results_df, X_clinical):
    """
    Run all classifiers on scaled data from results_df and create comparison plots.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with numeric features and Group column
    X_clinical : pd.DataFrame
        DataFrame with clinical features
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    st.subheader("All Classifiers Performance Comparison")
    
    # Prepare data for classifier comparison
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'cluster' in numeric_cols:
        numeric_cols.remove('cluster')
    
    if len(numeric_cols) < 2:
        st.warning("Not enough numeric features for classifier comparison")
        return None
    
    # Use all numeric features for comparison
    y_groups = results_df['cluster']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clinical)
    
    # Run classifier comparison
    return compare_all_classifiers_on_scaled_data(X_scaled, y_groups)

def compare_all_classifiers_on_scaled_data(X_scaled, y_true, test_size=0.3):
    """
    Run all classifiers on X_scaled data and create a plot showing their accuracy.
    
    Parameters:
    -----------
    X_scaled : array-like
        Scaled feature matrix
    y_true : array-like
        True labels
    test_size : float
        Proportion of data to use for testing (default: 0.3)
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # Define all classifiers
    classifiers = {
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'LightGBM': LGBMClassifier(random_state=42),
        'MLP (Feedforward Neural Net)': MLPClassifier(random_state=42, max_iter=1000),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        # 'Stacking Classifier': StackingClassifier(
        #     estimators=[
        #         ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        #         ('gb', GradientBoostingClassifier(random_state=42)),
        #         ('xgb', XGBClassifier(random_state=42))
        #     ],
        #     final_estimator=LogisticRegression(random_state=42)
        # ),
    }
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_true, test_size=test_size, random_state=42, stratify=y_true
    )
    
    # Store results
    results = []
    
    st.subheader("Classifier Performance Comparison on Scaled Data")
    st.write(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
    
    # Run classifiers in parallel
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    
    def train_and_evaluate_classifier(name_classifier_tuple):
        """Train and evaluate a single classifier"""
        name, classifier = name_classifier_tuple
        try:
            # Train the classifier
            classifier.fit(X_train, y_train)
            
            # Make predictions
            y_pred = classifier.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            return {
                'Classifier': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'Classifier': name,
                'Accuracy': 0.0,
                'Precision': 0.0,
                'Recall': 0.0,
                'F1-Score': 0.0,
                'status': 'error',
                'error': str(e)
            }
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Run classifiers in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        future_to_name = {
            executor.submit(train_and_evaluate_classifier, (name, classifier)): name 
            for name, classifier in classifiers.items()
        }
        
        # Process completed tasks
        completed = 0
        total = len(classifiers)
        
        for future in as_completed(future_to_name):
            result = future.result()
            results.append(result)
            
            # Update progress
            completed += 1
            progress_bar.progress(completed / total)
            
            # Show status
            if result['status'] == 'success':
                status_text.write(f"✅ {result['Classifier']}: Accuracy = {result['Accuracy']:.3f}")
            else:
                status_text.write(f"❌ {result['Classifier']}: Error - {result['error']}")
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Create results DataFrame (remove status fields for display)
    display_results = []
    for result in results:
        display_results.append({
            'Classifier': result['Classifier'],
            'Accuracy': result['Accuracy'],
            'Precision': result['Precision'],
            'Recall': result['Recall'],
            'F1-Score': result['F1-Score']
        })
    
    results_df = pd.DataFrame(display_results)
    
    # Sort by accuracy
    results_df = results_df.sort_values('Accuracy', ascending=True)
    
    # Create the plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy plot
    bars1 = ax1.barh(results_df['Classifier'], results_df['Accuracy'], color='skyblue')
    ax1.set_xlabel('Accuracy')
    ax1.set_title('Classifier Accuracy Comparison')
    ax1.set_xlim(0, 1)
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    # Precision plot
    bars2 = ax2.barh(results_df['Classifier'], results_df['Precision'], color='lightgreen')
    ax2.set_xlabel('Precision')
    ax2.set_title('Classifier Precision Comparison')
    ax2.set_xlim(0, 1)
    
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    # Recall plot
    bars3 = ax3.barh(results_df['Classifier'], results_df['Recall'], color='lightcoral')
    ax3.set_xlabel('Recall')
    ax3.set_title('Classifier Recall Comparison')
    ax3.set_xlim(0, 1)
    
    for i, bar in enumerate(bars3):
        width = bar.get_width()
        ax3.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    # F1-Score plot
    bars4 = ax4.barh(results_df['Classifier'], results_df['F1-Score'], color='lightyellow')
    ax4.set_xlabel('F1-Score')
    ax4.set_title('Classifier F1-Score Comparison')
    ax4.set_xlim(0, 1)
    
    for i, bar in enumerate(bars4):
        width = bar.get_width()
        ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # Display results table
    st.subheader("Detailed Results Table")
    st.dataframe(results_df, use_container_width=True)
    
    # Find best classifier
    best_classifier = results_df.loc[results_df['Accuracy'].idxmax()]
    st.success(f"🏆 Best performing classifier: **{best_classifier['Classifier']}** with accuracy of {best_classifier['Accuracy']:.3f}")
    
    return results_df


def rf_classifier_analysis(df_wnv2, selected_feature, eeg_features, key_suffix=""):
    """
    Perform Gradient Boosting classifier analysis for binary features.
    If feature is not binary, ask user for threshold input.
    
    Parameters:
    - df_wnv2: DataFrame with the data
    - selected_feature: The feature to predict using EEG features
    - eeg_features: List of EEG feature columns to use as predictors
    - key_suffix: Unique suffix for streamlit keys
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
    
    # Checkbox to show/hide the entire classifier analysis section
    show_classifier = st.checkbox(
        f"Show Classifier Analysis for {selected_feature}",
        value=False,
        key=f"show_classifier_{selected_feature}_{key_suffix}"
    )
    
    if not show_classifier:
        return
    
    # Prepare data
    df_clean = df_wnv2[df_wnv2[selected_feature].notna()].copy()
    
    if df_clean.empty:
        return
    
    # Convert feature to numeric if possible
    feature_data = pd.to_numeric(df_clean[selected_feature], errors='coerce')
    df_clean[selected_feature] = feature_data
    
    # Check if binary
    unique_values = df_clean[selected_feature].dropna().unique()
    n_unique = len(unique_values)
    
    # Determine threshold and create binary target
    threshold = None
    if n_unique == 2:
        # Binary feature - use as is
        unique_vals = sorted(unique_values)
        y = df_clean[selected_feature].apply(lambda x: 1 if x == unique_vals[1] else 0)
        group_name_0 = f"{selected_feature} = {unique_vals[0]}"
        group_name_1 = f"{selected_feature} = {unique_vals[1]}"
    else:
        # Non-binary feature - ask for threshold
        threshold = st.number_input(
            f"Enter threshold for {selected_feature} to create binary classification:",
            value=18.0,
            key=f"threshold_{selected_feature}_rf_{key_suffix}",
            help="Values above threshold will be class 1, below or equal will be class 0"
        )
        y = df_clean[selected_feature].apply(lambda x: 1 if x > threshold else 0)
        group_name_0 = f"{selected_feature} <= {threshold}"
        group_name_1 = f"{selected_feature} > {threshold}"
    
    # Check if we have enough samples in each class
    if len(y[y == 0]) < 3 or len(y[y == 1]) < 3:
        st.warning(f"Insufficient samples for binary classification (need at least 3 in each class)")
        return
    
    # Get EEG features (only those that exist in df_clean)
    available_eeg_features = [f for f in eeg_features if f in df_clean.columns]
    
    if not available_eeg_features:
        st.warning(f"No EEG features available for {selected_feature}")
        return
    
    # Prepare X (EEG features)
    X = df_clean[available_eeg_features].copy()
    
    # Remove columns with too many NaN values (more than 50%)
    X = X.dropna(axis=1, thresh=int(X.shape[0]/2))
    
    # Remove columns with constant values
    X = X.loc[:, X.nunique() != 1]
    
    if X.empty:
        st.warning(f"No valid EEG features after cleaning for {selected_feature}")
        return
    
    # Align y with X (remove rows where X has NaN)
    valid_idx = X.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]
    
    # Fill remaining NaN with median
    X = X.fillna(X.median())
    
    # Split data
    if len(X) < 10:
        st.warning(f"Insufficient samples for train/test split (need at least 10, have {len(X)})")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Display sample counts for each group in training and testing
    train_counts = y_train.value_counts().sort_index()
    test_counts = y_test.value_counts().sort_index()
    
    st.write("**Sample Distribution:**")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Training Set:**")
        for class_val, count in train_counts.items():
            class_name = group_name_1 if class_val == 1 else group_name_0
            st.write(f"- {class_name}: N = {count}")
        st.write(f"**Total Training: N = {len(y_train)}**")
    with col2:
        st.write("**Testing Set:**")
        for class_val, count in test_counts.items():
            class_name = group_name_1 if class_val == 1 else group_name_0
            st.write(f"- {class_name}: N = {count}")
        st.write(f"**Total Testing: N = {len(y_test)}**")
    
    # Compare all classifiers first
    st.subheader("Classifier Performance Comparison")
    classifier_results = []
    
    # Define classifier factory functions to create fresh instances
    classifier_factories = {
        'Gradient Boosting': lambda: GradientBoostingClassifier(n_estimators=200, random_state=42, max_depth=10),
        'Random Forest': lambda: RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10),
        'XGBoost': lambda: XGBClassifier(random_state=42),
        'LightGBM': lambda: LGBMClassifier(random_state=42),
        'MLP (Feedforward Neural Net)': lambda: MLPClassifier(random_state=42, max_iter=1000),
        'Logistic Regression': lambda: LogisticRegression(random_state=42, max_iter=1000),
        'Stacking Classifier': lambda: StackingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingClassifier(random_state=42)),
                ('xgb', XGBClassifier(random_state=42))
            ],
            final_estimator=LogisticRegression(random_state=42)
        ),
    }
    
    with st.spinner("Training and comparing all classifiers..."):
        for name, factory in classifier_factories.items():
            try:
                clf = factory()  # Create fresh instance
                clf.fit(X_train, y_train)
                y_pred_temp = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred_temp)
                classifier_results.append({
                    'Classifier': name,
                    'Accuracy': acc
                })
            except Exception as e:
                classifier_results.append({
                    'Classifier': name,
                    'Accuracy': 0.0
                })
                st.warning(f"Error training {name}: {str(e)}")
    
    # Create comparison plot
    if classifier_results:
        results_df = pd.DataFrame(classifier_results).sort_values('Accuracy', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(results_df))]
        bars = ax.barh(results_df['Classifier'], results_df['Accuracy'], color=colors)
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Classifier')
        ax.set_title('Classifier Accuracy Comparison')
        ax.set_xlim(0, 1)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Display results table
        st.write("**Classifier Comparison Results:**")
        st.dataframe(results_df, use_container_width=True)
        
        # Highlight best classifier
        best_classifier = results_df.iloc[0]
        st.success(f"🏆 Best performing classifier: **{best_classifier['Classifier']}** with accuracy of {best_classifier['Accuracy']:.3f}")
    
    # Let user select classifier
    st.subheader("Select Classifier for Detailed Analysis")
    # Get best classifier index for default selection
    default_index = 0
    if classifier_results:
        results_df_temp = pd.DataFrame(classifier_results).sort_values('Accuracy', ascending=False)
        best_name = results_df_temp.iloc[0]['Classifier']
        classifier_names = list(classifier_factories.keys())
        if best_name in classifier_names:
            default_index = classifier_names.index(best_name)
    
    selected_classifier_name = st.selectbox(
        "Choose classifier for detailed analysis:",
        list(classifier_factories.keys()),
        index=default_index,
        key=f"classifier_select_{selected_feature}_{key_suffix}"
    )
    
    # Create fresh instance for selected classifier
    selected_classifier = classifier_factories[selected_classifier_name]()
    
    # Train selected classifier
    try:
        selected_classifier.fit(X_train, y_train)
        
        # Predictions
        y_pred = selected_classifier.predict(X_test)
        y_pred_proba = selected_classifier.predict_proba(X_test)[:, 1]
        
        # Display results
        st.subheader(f"{selected_classifier_name} Classifier Results: Predicting {selected_feature}")
        
        # Classification report
        st.write("**Classification Report:**")
        report = classification_report(y_test, y_pred, target_names=[group_name_0, group_name_1], output_dict=True)
        st.text(classification_report(y_test, y_pred, target_names=[group_name_0, group_name_1]))
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy:.3f}")
        col2.metric("Precision", f"{precision:.3f}")
        col3.metric("Recall", f"{recall:.3f}")
        col4.metric("F1-Score", f"{f1:.3f}")
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        # Show class descriptions as info
        st.info(f"Class 0: {group_name_0} | Class 1: {group_name_1}")
        cm = confusion_matrix(y_test, y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Raw counts - use simple 0/1 labels
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['0', '1'],
                   yticklabels=['0', '1'])
        ax1.set_title('Confusion Matrix (Counts)')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        
        # Percentages - use simple 0/1 labels
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='viridis', ax=ax2,
                   xticklabels=['0', '1'],
                   yticklabels=['0', '1'])
        ax2.set_title('Confusion Matrix (Percentages)')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Feature Importance
        st.subheader("Feature Importance")
        # Check if classifier has feature_importances_ attribute
        if hasattr(selected_classifier, 'feature_importances_'):
            importances = selected_classifier.feature_importances_
        elif hasattr(selected_classifier, 'coef_'):
            # For linear models like Logistic Regression, use coefficient absolute values
            importances = np.abs(selected_classifier.coef_[0])
        else:
            st.info("Feature importance not available for this classifier type.")
            importances = None
        
        if importances is not None:
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
        
            # Plot top 20 features
            top_n = min(20, len(importance_df))
            top_importance = importance_df.head(top_n)
            
            fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
            sns.barplot(data=top_importance, x='importance', y='feature', ax=ax, palette='viridis')
            ax.set_title(f'Top {top_n} Feature Importances ({selected_classifier_name})')
            ax.set_xlabel('Importance')
            ax.set_ylabel('Feature')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # Display feature importance table
            st.write(f"**Top {top_n} Most Important Features:**")
            st.dataframe(top_importance, use_container_width=True)
        
        # Aggregate feature importance by EEG electrode (only if feature importance is available)
        if importances is not None:
            st.subheader("Feature Importance by Electrode")
            electrode_importance = {}
            
            # For each electrode, find features that contain the electrode name and sum their importance
            for electrode in eeg_channels:
                electrode_importance[electrode] = 0.0
                # Check each feature to see if it contains this electrode name
                for idx, feature in enumerate(importance_df['feature']):
                    if isinstance(feature, str) and electrode in feature:
                        electrode_importance[electrode] += importance_df.iloc[idx]['importance']
            
            # Create DataFrame for electrode importance
            electrode_df = pd.DataFrame({
                'electrode': list(electrode_importance.keys()),
                'total_importance': list(electrode_importance.values())
            }).sort_values('total_importance', ascending=False)
            
            # Filter out electrodes with zero importance
            electrode_df_nonzero = electrode_df[electrode_df['total_importance'] > 0]
            
            if not electrode_df_nonzero.empty:
                # Plot electrode importance
                fig, ax = plt.subplots(figsize=(10, max(6, len(electrode_df_nonzero) * 0.3)))
                sns.barplot(data=electrode_df_nonzero, x='total_importance', y='electrode', ax=ax, palette='viridis')
                ax.set_title(f'Feature Importance by Electrode ({selected_classifier_name})')
                ax.set_xlabel('Total Importance (Sum across all features)')
                ax.set_ylabel('EEG Electrode')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
                # Display electrode importance table
                st.write(f"**Feature Importance by Electrode:**")
                st.dataframe(electrode_df_nonzero, use_container_width=True)
                
                # Try to create topomap if available
                try:
                    electrode_dict = dict(zip(electrode_df_nonzero['electrode'], electrode_df_nonzero['total_importance']))
                    topomap_fig = plot_electrode_importance_topomap(
                        electrode_dict,
                        title=f'Electrode Importance Topomap: Predicting {selected_feature}'
                    )
                    if topomap_fig:
                        st.subheader("Electrode Importance Topomap")
                        st.pyplot(topomap_fig)
                        plt.close(topomap_fig)
                except Exception as e:
                    st.info(f"Topomap unavailable: {str(e)}")
            else:
                st.info("No electrode importance found (no features matched electrode names)")
        
    except Exception as e:
        st.error(f"Error in RF classifier analysis: {str(e)}")


def rf_regressor_analysis(df_wnv2, selected_feature, eeg_features, key_suffix=""):
    """
    Perform regression analysis for numerical features using various regressors.
    
    Parameters:
    - df_wnv2: DataFrame with the data
    - selected_feature: The numerical feature to predict using EEG features
    - eeg_features: List of EEG feature columns to use as predictors
    - key_suffix: Unique suffix for streamlit keys
    """
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.neural_network import MLPRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler
    
    # Checkbox to show/hide the entire regressor analysis section
    show_regressor = st.checkbox(
        f"Show Regressor Analysis for {selected_feature}",
        value=False,
        key=f"show_regressor_{selected_feature}_{key_suffix}"
    )
    
    if not show_regressor:
        return
    
    # Prepare data
    df_clean = df_wnv2[df_wnv2[selected_feature].notna()].copy()
    
    if df_clean.empty:
        return
    
    # Convert feature to numeric if possible
    feature_data = pd.to_numeric(df_clean[selected_feature], errors='coerce')
    df_clean[selected_feature] = feature_data
    
    # Remove any remaining NaN values in target
    df_clean = df_clean[df_clean[selected_feature].notna()].copy()
    
    if df_clean.empty:
        st.warning(f"No valid numerical data for {selected_feature}")
        return
    
    # Get EEG features (only those that exist in df_clean)
    available_eeg_features = [f for f in eeg_features if f in df_clean.columns]
    
    if not available_eeg_features:
        st.warning(f"No EEG features available for {selected_feature}")
        return
    
    # Prepare X (EEG features)
    X = df_clean[available_eeg_features].copy()
    y = df_clean[selected_feature].copy()
    
    # Remove columns with too many NaN values (more than 50%)
    X = X.dropna(axis=1, thresh=int(X.shape[0]/2))
    
    # Remove columns with constant values
    X = X.loc[:, X.nunique() != 1]
    
    if X.empty:
        st.warning(f"No valid EEG features after cleaning for {selected_feature}")
        return
    
    # Align y with X (remove rows where X has NaN)
    valid_idx = X.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]
    
    # Fill remaining NaN with median
    X = X.fillna(X.median())
    
    # Split data
    if len(X) < 10:
        st.warning(f"Insufficient samples for train/test split (need at least 10, have {len(X)})")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Display sample counts
    st.write("**Sample Distribution:**")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Training Set: N = {len(y_train)}**")
        st.write(f"Target range: [{y_train.min():.2f}, {y_train.max():.2f}]")
        st.write(f"Target mean: {y_train.mean():.2f} ± {y_train.std():.2f}")
    with col2:
        st.write(f"**Testing Set: N = {len(y_test)}**")
        st.write(f"Target range: [{y_test.min():.2f}, {y_test.max():.2f}]")
        st.write(f"Target mean: {y_test.mean():.2f} ± {y_test.std():.2f}")
    
    # Compare all regressors first
    st.subheader("Regressor Performance Comparison")
    regressor_results = []
    
    # Define regressor factory functions to create fresh instances
    regressor_factories = {
        'Gradient Boosting': lambda: GradientBoostingRegressor(n_estimators=200, random_state=42, max_depth=10),
        'Random Forest': lambda: RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10),
        'XGBoost': lambda: XGBRegressor(random_state=42),
        'LightGBM': lambda: LGBMRegressor(random_state=42),
        'MLP (Feedforward Neural Net)': lambda: MLPRegressor(random_state=42, max_iter=1000),
        'Linear Regression': lambda: LinearRegression(),
        'Ridge Regression': lambda: Ridge(random_state=42),
        'Lasso Regression': lambda: Lasso(random_state=42, max_iter=1000),
        'Stacking Regressor': lambda: StackingRegressor(
            estimators=[
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingRegressor(random_state=42)),
                ('xgb', XGBRegressor(random_state=42))
            ],
            final_estimator=LinearRegression()
        ),
    }
    
    with st.spinner("Training and comparing all regressors..."):
        for name, factory in regressor_factories.items():
            try:
                reg = factory()  # Create fresh instance
                reg.fit(X_train, y_train)
                y_pred_temp = reg.predict(X_test)
                r2 = r2_score(y_test, y_pred_temp)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred_temp))
                mae = mean_absolute_error(y_test, y_pred_temp)
                regressor_results.append({
                    'Regressor': name,
                    'R² Score': r2,
                    'RMSE': rmse,
                    'MAE': mae
                })
            except Exception as e:
                regressor_results.append({
                    'Regressor': name,
                    'R² Score': -np.inf,
                    'RMSE': np.inf,
                    'MAE': np.inf
                })
                st.warning(f"Error training {name}: {str(e)}")
    
    # Create comparison plot
    if regressor_results:
        results_df = pd.DataFrame(regressor_results).sort_values('R² Score', ascending=False)
        
        # Plot R² scores
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(results_df))]
        bars = ax.barh(results_df['Regressor'], results_df['R² Score'], color=colors)
        ax.set_xlabel('R² Score')
        ax.set_ylabel('Regressor')
        ax.set_title('Regressor R² Score Comparison')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01 if width >= 0 else width - 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left' if width >= 0 else 'right', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Display results table
        st.write("**Regressor Comparison Results:**")
        st.dataframe(results_df, use_container_width=True)
        
        # Highlight best regressor
        best_regressor = results_df.iloc[0]
        st.success(f"🏆 Best performing regressor: **{best_regressor['Regressor']}** with R² score of {best_regressor['R² Score']:.3f}")
    
    # Let user select regressor
    st.subheader("Select Regressor for Detailed Analysis")
    # Get best regressor index for default selection
    default_index = 0
    if regressor_results:
        results_df_temp = pd.DataFrame(regressor_results).sort_values('R² Score', ascending=False)
        best_name = results_df_temp.iloc[0]['Regressor']
        regressor_names = list(regressor_factories.keys())
        if best_name in regressor_names:
            default_index = regressor_names.index(best_name)
    
    selected_regressor_name = st.selectbox(
        "Choose regressor for detailed analysis:",
        list(regressor_factories.keys()),
        index=default_index,
        key=f"regressor_select_{selected_feature}_{key_suffix}"
    )
    
    # Create fresh instance for selected regressor
    selected_regressor = regressor_factories[selected_regressor_name]()
    
    # Train selected regressor
    try:
        selected_regressor.fit(X_train, y_train)
        
        # Predictions
        y_pred = selected_regressor.predict(X_test)
        y_pred_train = selected_regressor.predict(X_train)
        
        # Display results
        st.subheader(f"{selected_regressor_name} Regressor Results: Predicting {selected_feature}")
        
        # Metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2_train = r2_score(y_train, y_pred_train)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R² Score (Test)", f"{r2:.3f}")
        col2.metric("RMSE (Test)", f"{rmse:.3f}")
        col3.metric("MAE (Test)", f"{mae:.3f}")
        col4.metric("R² Score (Train)", f"{r2_train:.3f}")
        
        # Prediction vs Actual plots
        st.subheader("Prediction vs Actual")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Test set
        axes[0].scatter(y_test, y_pred, alpha=0.6, s=50)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title(f'Test Set (R² = {r2:.3f})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Train set
        axes[1].scatter(y_train, y_pred_train, alpha=0.6, s=50, color='green')
        min_val = min(y_train.min(), y_pred_train.min())
        max_val = max(y_train.max(), y_pred_train.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        axes[1].set_xlabel('Actual Values')
        axes[1].set_ylabel('Predicted Values')
        axes[1].set_title(f'Train Set (R² = {r2_train:.3f})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Residuals plot
        st.subheader("Residuals Analysis")
        residuals = y_test - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6, s=50)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals distribution
        axes[1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residuals Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Feature Importance
        st.subheader("Feature Importance")
        # Check if regressor has feature_importances_ attribute
        if hasattr(selected_regressor, 'feature_importances_'):
            importances = selected_regressor.feature_importances_
        elif hasattr(selected_regressor, 'coef_'):
            # For linear models, use coefficient absolute values
            importances = np.abs(selected_regressor.coef_)
        else:
            st.info("Feature importance not available for this regressor type.")
            importances = None
        
        if importances is not None:
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
        
            # Plot top 20 features
            top_n = min(20, len(importance_df))
            top_importance = importance_df.head(top_n)
            
            fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
            sns.barplot(data=top_importance, x='importance', y='feature', ax=ax, palette='viridis')
            ax.set_title(f'Top {top_n} Feature Importances ({selected_regressor_name})')
            ax.set_xlabel('Importance')
            ax.set_ylabel('Feature')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # Display feature importance table
            st.write(f"**Top {top_n} Most Important Features:**")
            st.dataframe(top_importance, use_container_width=True)
        
            # Aggregate feature importance by EEG electrode (only if feature importance is available)
            st.subheader("Feature Importance by Electrode")
            electrode_importance = {}
            
            # Get eeg_channels if available (try to import or use a default list)
            try:
                from utils.eeg_utils import eeg_channels
            except:
                # If not available, try to extract from feature names
                eeg_channels = set()
                for feature in X.columns:
                    if isinstance(feature, str):
                        # Try to extract electrode names (this is a fallback)
                        for channel in ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
                                       'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']:
                            if channel in feature:
                                eeg_channels.add(channel)
                eeg_channels = list(eeg_channels) if eeg_channels else []
            
            # For each electrode, find features that contain the electrode name and sum their importance
            for electrode in eeg_channels:
                electrode_importance[electrode] = 0.0
                # Check each feature to see if it contains this electrode name
                for idx, feature in enumerate(importance_df['feature']):
                    if isinstance(feature, str) and electrode in feature:
                        electrode_importance[electrode] += importance_df.iloc[idx]['importance']
            
            # Create DataFrame for electrode importance
            electrode_df = pd.DataFrame({
                'electrode': list(electrode_importance.keys()),
                'total_importance': list(electrode_importance.values())
            }).sort_values('total_importance', ascending=False)
            
            # Filter out electrodes with zero importance
            electrode_df_nonzero = electrode_df[electrode_df['total_importance'] > 0]
            
            if not electrode_df_nonzero.empty:
                # Plot electrode importance
                fig, ax = plt.subplots(figsize=(10, max(6, len(electrode_df_nonzero) * 0.3)))
                sns.barplot(data=electrode_df_nonzero, x='total_importance', y='electrode', ax=ax, palette='viridis')
                ax.set_title(f'Feature Importance by Electrode ({selected_regressor_name})')
                ax.set_xlabel('Total Importance (Sum across all features)')
                ax.set_ylabel('EEG Electrode')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
                # Display electrode importance table
                st.write(f"**Feature Importance by Electrode:**")
                st.dataframe(electrode_df_nonzero, use_container_width=True)
                
                # Try to create topomap if available
                try:
                    electrode_dict = dict(zip(electrode_df_nonzero['electrode'], electrode_df_nonzero['total_importance']))
                    topomap_fig = plot_electrode_importance_topomap(
                        electrode_dict,
                        title=f'Electrode Importance Topomap: Predicting {selected_feature}'
                    )
                    if topomap_fig:
                        st.subheader("Electrode Importance Topomap")
                        st.pyplot(topomap_fig)
                        plt.close(topomap_fig)
                except Exception as e:
                    st.info(f"Topomap unavailable: {str(e)}")
            else:
                st.info("No electrode importance found (no features matched electrode names)")
        
    except Exception as e:
        st.error(f"Error in RF regressor analysis: {str(e)}")


# Streamlit App
def main():
    # options to choose from folders in pickles
    default_options = ["COBRAD", "WNV"] +  [f for f in os.listdir('parquet_results') if os.path.isdir(os.path.join('parquet_results', f))] 
    # allow multiple selection for project
    # default_options = ["COBRAD", "WNV"] reading_epilepsy_cut
    # Deduplicate while preserving order
    seen = set()
    opts = [x for x in default_options if not (x in seen or seen.add(x))]
    selected_projects = st.sidebar.pills("Select Project(s)", opts, default=[x for x in ["TGA"] if x in opts],selection_mode ="multi")
    if not selected_projects:
        st.error("Please select at least one project.")
        return
    df_wnv2 = None
    df_wnv2_others = []
    controls = None
    controls_others = []
    cases_group_name = None
    cases_group_name_others = []
    for idx, project_name in enumerate(selected_projects):
        if project_name == "COBRAD":
            awake_only = st.sidebar.checkbox("Awake Only", value=False)
            sample_window_size = st.sidebar.slider("Select the sample window size", 0, 12, 0)
            df_wnv, patients_folder, temp_controls, temp_df_wnv2, temp_cases_group_name = cobrad_get_files(sample_window_size, awake_only)
        elif project_name == "WNV":
            df_wnv, patients_folder, temp_controls, temp_df_wnv2, temp_cases_group_name = wnv_get_files()
        elif project_name == "TGA":
            df_wnv, patients_folder, temp_controls, temp_df_wnv2, temp_cases_group_name = tga_get_files()
        else:
            df_wnv, patients_folder, temp_controls, temp_df_wnv2, temp_cases_group_name = generic_get_files(project_name)
        
        if idx == 0:
            df_wnv2 = temp_df_wnv2
            controls = temp_controls
            controls['Group'] = 'Control'
            cases_group_name = temp_cases_group_name
        else:
            df_wnv2_others.append((project_name, temp_df_wnv2))
            controls_others.append((project_name, temp_controls))
            cases_group_name_others.append((project_name, temp_cases_group_name))
    project_name = selected_projects[0]  # Use the first selected project for downstream logic
    st.session_state['project_name'] = project_name  # Store project name for PPTX generation
    
    # Rename age column if there's exactly one column containing "age" as a whole word (case-insensitive)
    age_cols = [col for col in df_wnv2.columns if re.search(r'\bage\b', col, re.IGNORECASE)]
    if len(age_cols) == 1:
        df_wnv2 = df_wnv2.rename(columns={age_cols[0]: 'age'})
    
    # Rename gender/sex column if there's exactly one column containing "gender" or "sex" as whole words (case-insensitive)
    # Match gender/sex that is at word boundaries (including underscore as a boundary)
    gender_cols = [col for col in df_wnv2.columns if re.search(r'(^|_|[^a-zA-Z0-9_])gender(_|[^a-zA-Z0-9_]|$)', col, re.IGNORECASE) or (re.search(r'(^|_|[^a-zA-Z0-9_])sex(_|[^a-zA-Z0-9_]|$)', col, re.IGNORECASE) and not re.search(r'(^|_|[^a-zA-Z0-9_])age(_|[^a-zA-Z0-9_]|$)', col, re.IGNORECASE))]
    if len(gender_cols) == 1:
        df_wnv2 = df_wnv2.rename(columns={gender_cols[0]: 'gender'})
    # Store df_wnv2 in session state for Excel export
    st.session_state['df_wnv2'] = df_wnv2
    
    # Add Excel download button at the top of sidebar
    download_excel_button()

    st.title("EEG Analysis")
    # Iterate over each frequency band and plot the topomap
    cols_to_drop = ['annotations', 'bad_channels', 'patient_number', 'csv_file_name', 'file_name', 'file_path', 'signal_labels', 'number_of_signals', 'sampling_frequency', 'sampling_rate', 'duration_min']
    # Remove specified columns and those containing dates from df_wnv2
    df_wnv2 = df_wnv2.drop(columns=[col for col in df_wnv2.columns if col in cols_to_drop or 'date' in col.lower()])
    #%% clinical data analysis
    clinical_features, boxplot_columns = get_clinical_and_boxplot_cols(df_wnv2=df_wnv2)
    # Identify the separation point between clinical and EEG features
    separator_index = next((i for i, col in enumerate(df_wnv2.columns) if 'overall_' in col), None)
    if separator_index is None:
        st.error("No column with 'overall_' found to separate clinical and EEG features.")
        return
    
    # Split columns into clinical and EEG features
    eeg_features = [col for col in df_wnv2.columns[separator_index:] if col != 'Group']
    clinical_features_numeric = [col for col in clinical_features if pd.api.types.is_numeric_dtype(df_wnv2[col])]
    # Sidebar for feature selection
    st.sidebar.header("Feature Selection")
    # 'systole_diastole_comparison', 'ECG_Analysis', 'HEP',
    feature_types = ("vs_Controls",'All',  'Clinical Feature', "EEG Feature", "ml_plots",  "Pair Plot",'Raw','Spectrogram', ) # 'Longitudinal',
    feature_type = st.sidebar.selectbox("Select feature type to plot against the other type:", feature_types, index=0)
    
    # Sidebar for scatterplot size feature selection
    st.sidebar.header("Scatterplot Size Control")
    size_feature_options = ['None'] + clinical_features_numeric + eeg_features
    size_feature = st.sidebar.selectbox("Select feature for scatterplot circle size:", size_feature_options, index=0)
    if size_feature == 'None':
        size_feature = None
    if feature_type in ["Clinical Feature", "EEG Feature", "vs_Controls","All"]:
        # ask user if they want only significant, or full.
        st.sidebar.header("Select Analysis Type")
        analysis_type = st.sidebar.pills("Select Analysis Type", ["Full", "Significant"],default=["Significant"])
    else:
        analysis_type = "Full"
        
    dict_features = {}
    bool_all_features = False
    cols_to_skip = ['ID','annotations','bad_channels','Group','patient_number','size','n_samples']
    clinical_features = [feature for feature in clinical_features if feature not in cols_to_skip]
    if not clinical_features or not eeg_features:
        st.error("Could not identify clinical or EEG features based on the 'overall_' separator.")
        return
    # Determine which feature types to run
    if feature_type == "All":
        # remove all from feature_types
        feature_types_to_run = [feature for feature in feature_types if feature != 'All']
        st.title("All Feature Types Analysis")
    else:
        feature_types_to_run = [feature_type]
    
    # Run through each feature type
    for current_feature_type in feature_types_to_run:
        clinical_features, boxplot_columns = get_clinical_and_boxplot_cols(df_wnv2=df_wnv2)
        if feature_type == "All":
            st.header(f"{current_feature_type} Analysis")
            bool_all_features = True
        if current_feature_type == "vs_Controls":
            vs_controls_run(project_name, df_wnv2, controls, boxplot_columns, analysis_type)
        elif current_feature_type == "HEP":
            HEP_plots(project_name,df_wnv2,controls,boxplot_columns,analysis_type,size_feature=size_feature, context_key="hep_main")
        elif current_feature_type == "Pair Plot":
            pairplot_columns(df_wnv2, clinical_features, eeg_features)
        elif current_feature_type == "ml_plots":
            # get the names of folders that are in {figures_dir}/ml_plots
            sorted_files = find_and_sort_ml_plots(f"{project_name}_figures/ml_plots")
            ml_plots_features = [f.split('/')[2] for f in sorted_files]
            selected_feature = st.sidebar.radio("Select a feature for ML plots:", ml_plots_features)
            if selected_feature:
                ml_plots_get_images(project_name, selected_feature)
        elif current_feature_type == "Spectrogram":
            try:
                st.title("Spectrogram")
                # ask user for win_sec
                win_sec = st.sidebar.slider("Select window size in seconds", 1, 30, 5)
                st.subheader(f"{current_feature_type} {cases_group_name}")
                spectrogram_run(cases_group_name,win_sec=win_sec)
                st.divider()
                st.subheader("Spectrogram Controls")
                spectrogram_run(f'Controls',win_sec=win_sec)
            except Exception as e:
                st.error(f"Error running spectrogram: {e}")
        elif current_feature_type == "Raw":
            st.title("Raw Data")
            raw_run(cases_group_name)
        elif current_feature_type == "systole_diastole_comparison":
            # Sleep stage selection
            sleep_stage_options = ["N1","None (Full Recording)",  "N2", "N3", "R", "W"]
            selected_stage_display = st.sidebar.selectbox(
                "Select Sleep Stage to Extract:",
                sleep_stage_options,
                index=0,
                key="systole_diastole_stage"
            )
            sleep_stage = None if selected_stage_display == "None (Full Recording)" else selected_stage_display
            min_duration_min = st.sidebar.number_input(
                "Minimum Duration (minutes):",
                min_value=1,
                max_value=60,
                value=1,
                help="Minimum duration of the sleep stage segment. For W stage, 10 minutes is used.",
                key="systole_diastole_duration"
            )
            systole_diastole_comparison_run(cases_group_name, sleep_stage=sleep_stage, min_duration_min=min_duration_min)
        elif current_feature_type == "ECG_Analysis":
            # Sleep stage selection
            sleep_stage_options = ["N1","None (Full Recording)",  "N2", "N3", "R", "W"]
            selected_stage_display = st.sidebar.selectbox(
                "Select Sleep Stage to Extract:",
                sleep_stage_options,
                index=0
            )
            sleep_stage = None if selected_stage_display == "None (Full Recording)" else selected_stage_display
            min_duration_min = st.sidebar.number_input(
                "Minimum Duration (minutes):",
                min_value=1,
                max_value=60,
                value=1,
                help="Minimum duration of the sleep stage segment. For W stage, 10 minutes is used."
            )
            ecg_analysis_run(cases_group_name, sleep_stage=sleep_stage, min_duration_min=min_duration_min)
        elif current_feature_type == "Longitudinal":
            longitudinal_analysis(project_name)
        elif current_feature_type == "EEG Feature":
            eeg_feature_options =  eeg_features+ ["All Features"] 
            selected_feature = st.sidebar.selectbox("Select an EEG feature:", eeg_feature_options)
            if selected_feature == "All Features" or feature_type == "All":
                all_feat_list = eeg_features
                selected_feature = eeg_features[0]
            plot_title = f"Plots of {selected_feature} vs All Clinical Features"
            boxplot_columns = clinical_features_numeric
            
            # Checkbox for forest plot next to HEP checkbox position
            forest_plot_eeg = st.checkbox("Show Forest Plot vs All Clinical Features", value=False, key=f"forest_plot_eeg_{selected_feature}")
            
            # Add forest plot if requested
            if forest_plot_eeg:
                st.subheader("Forest Plot Analysis")
                forest_plot_all_features(df_wnv2, selected_feature, clinical_features_numeric, analysis_type)
        elif current_feature_type == "Clinical Feature":
            clinical_features_correlation = st.sidebar.checkbox("Show Clinical Features Correlation", value=False)
            forest_plot_clinical = st.sidebar.checkbox("Show Forest Plot vs All EEG Features", value=False)
            marked_clinical_features_w_all = clinical_features + ["All Features"]
            # Set default index to 'clinical_moca' if available, otherwise 0
            default_index = 0
            if 'clinical_moca' in marked_clinical_features_w_all:
                default_index = marked_clinical_features_w_all.index('clinical_moca')
            selected_feature = st.sidebar.selectbox("Select a Clinical feature:", marked_clinical_features_w_all, index=default_index)
            if selected_feature == "All Features" or feature_type == "All":
                all_feat_list = clinical_features
                selected_feature = clinical_features[0]
            plot_title = f"Plots of {selected_feature} vs All EEG Features"
            st.header(plot_title)
            def run_selected_feature(boxplot_columns, hep_checkbox=None):
                # keep only rows that can be safely converted to float
                feature_data = (
                    df_wnv2[selected_feature]
                    .apply(pd.to_numeric, errors='coerce')  # turn invalids into NaN
                    .dropna()
                    .astype(float)
                )
                if feature_data.empty:
                    st.warning(f"No valid numeric data available for the selected feature: {selected_feature}")
                    return  # Skip to next feature type
                col1, col2, col3 = st.columns(3)
                col1.metric("Mean", f"{feature_data.mean():.2f}")
                col2.metric("Median", f"{feature_data.median():.2f}")
                col3.metric("Std Dev", f"{feature_data.std():.2f}")
                col4, col5 ,col6 = st.columns(3)
                col4.metric("Minimum", f"{feature_data.min():.2f}")
                col5.metric("Maximum", f"{feature_data.max():.2f}")
                # col 6 is N with dropna
                col6.metric("N", f"{feature_data.dropna().count()}")
                # write text for {selected feature}, N= {}, mean ± std
                st.write(f'{selected_feature} N= {feature_data.dropna().count()}, mean {feature_data.mean():.2f} ± {feature_data.std():.2f}')
                numeric_colunms = df_wnv2.select_dtypes(include=[np.number]).columns
                # sidebar checkbox - Clinical Features Correlation
                if clinical_features_correlation:
                    # from clinical columns get
                    boxplot_columns = boxplot_columns + clinical_features - selected_feature
                # Display selected feature and plots
                df_wnv3 = df_wnv2[df_wnv2[selected_feature].notna()].copy()
                # Store df_wnv3 in session state for Excel export
                st.session_state['df_wnv3'] = df_wnv3
                unique_values = df_wnv3[selected_feature].unique()
                # Save the raw data
                print(f'Analyzing {selected_feature} with {len(unique_values)} unique values')
                if df_wnv3.shape[0] < 3 or unique_values.shape[0] < 2:
                    return
                if len(unique_values) == 2:  # Check if binary
                    # check that there are at least 3 in each group (0,1)
                    if len(df_wnv3[df_wnv3[selected_feature] == 1]) < 3 or len(df_wnv3[df_wnv3[selected_feature] == 0]) < 3:
                        return
                    if selected_feature == 'sex':
                        # if max is 2, decrease 1
                        if int(df_wnv3[selected_feature].max()) ==2:
                            df_wnv3[selected_feature] -= 1
                        # if 1 'f' else 'm'
                        df_wnv3['Group'] = df_wnv3[selected_feature].apply(lambda x: 'f' if x == 1 else 'm')
                        # Update session state
                        st.session_state['df_wnv3'] = df_wnv3
                    else:
                        # group values based on band if =1, else f'not {band}'
                        org_selected_feature_for_plot = org_selected_feature(selected_feature)
                        df_wnv3['Group'] = df_wnv3[selected_feature].apply(lambda x: f'{org_selected_feature_for_plot}+' if x == 1 else f'{org_selected_feature_for_plot}-')
                        # Update session state
                        st.session_state['df_wnv3'] = df_wnv3
                    # run over df_wnv2_others
                    for idx, (project_name_other, df_wnv2_other) in enumerate(df_wnv2_others):
                        if idx ==0:
                            # concat with controls, df_wnv3
                            df_wnv3 = pd.concat([controls, df_wnv3], ignore_index=True)
                            # Update session state
                            st.session_state['df_wnv3'] = df_wnv3
                        if selected_feature == 'sex':
                            # Build a safe string representation of sex (map numeric codes to 'f'/'m' if possible)
                            if 'sex' in df_wnv2_other.columns:
                                s = df_wnv2_other['sex']
                                if pd.api.types.is_numeric_dtype(s):
                                    s_num = s.fillna(-1).astype(float)
                                    # Some datasets use {1,2} where 2 represents female -> normalize to 0/1
                                    if np.nanmax(s_num) >= 2:
                                        s_num = s_num - 1
                                    s_str = s_num.fillna(-1).astype(int).map({1: 'f', 0: 'm'}).astype(str)
                                else:
                                    s_str = s.astype(str).str.strip()
                            else:
                                s_str = pd.Series([project_name_other] * len(df_wnv2_other), index=df_wnv2_other.index)
                            df_wnv2_other['Group'] = s_str + '_' + project_name_other
                            # df_wnv2_other['Group'] remove rows where df_wnv2_other['Group'] is NaN
                            df_wnv2_other = df_wnv2_other.dropna(subset=['sex'])
                        else:
                            df_wnv2_other['Group'] = project_name_other
                        # concat to df_wnv3
                        df_wnv3 = pd.concat([df_wnv3, df_wnv2_other], ignore_index=True)
                        # Update session state
                        st.session_state['df_wnv3'] = df_wnv3
                                    # st checkbox if HEP
                    if hep_checkbox is None:
                        hep_checkbox = st.checkbox("Show HEP & TSNE Plots", value=False, key=f"hep_checkbox_{selected_feature}")
                    if hep_checkbox:
                        st.subheader("HEP Plots")
                        HEP_plots(project_name, df_wnv3, controls, boxplot_columns, analysis_type, selected_feature, size_feature, context_key="hep_clinical")
                        plot_tsne_by_group(df_wnv3)
                    st.divider()
                    for band in boxplot_columns:
                        results_df = analyze_and_correct(df_wnv3, [band], groups=df_wnv3['Group'].unique())
                        boxplot_plot_dabest(results_df, df_wnv3, band, f'{selected_feature}',is_streamlit=True,analysis_type=analysis_type)
                elif '(' in selected_feature and ')' in selected_feature:
                    for band in boxplot_columns:
                        df_wnv3['Group'] = df_wnv3[selected_feature].astype(str)
                        df_wnv3[selected_feature] = df_wnv3[selected_feature].astype(float)
                        # do boxplot for each band
                        results_df = analyze_and_correct(df_wnv3, [band], groups=df_wnv3['Group'].unique())
                        boxplot_plot_dabest(results_df, df_wnv3, band, f'{selected_feature}',is_streamlit=True,analysis_type=analysis_type)
                elif selected_feature in numeric_colunms:
                    for band in boxplot_columns:
                        scatter_plot_with_regression({}, df_wnv3, selected_feature, band, f'{selected_feature}',is_streamlit=True,analysis_type=analysis_type)
            
            # if selected_feature_w_all is not None
            if bool_all_features:
                # Show HEP checkbox only once for all features
                hep_checkbox = st.checkbox("Show HEP & TSNE Plots", value=False, key="hep_checkbox_all")
                # remove size, n_samples, lowpass, highpass, duration_sec, parquet_file_name
                all_feat_list_fixed = [feature for feature in all_feat_list if feature not in ['size', 'n_samples', 'lowpass', 'highpass', 'duration_sec', 'parquet_file_name']]
                for feature in all_feat_list_fixed:
                    clinical_features, boxplot_columns = get_clinical_and_boxplot_cols(df_wnv2=df_wnv2)
                    selected_feature = feature
                    st.write(f"## Analyzing Feature: {selected_feature}")
                    # Add RF classifier analysis for binary features (or non-binary with threshold)
                    rf_classifier_analysis(df_wnv2, selected_feature, eeg_features, key_suffix=f"all_{feature}")
                    # Add RF regressor analysis for numerical features
                    rf_regressor_analysis(df_wnv2, selected_feature, eeg_features, key_suffix=f"all_{feature}")
                    run_selected_feature(boxplot_columns, hep_checkbox)
                    # Add forest plot if requested
                    if forest_plot_clinical:
                        st.subheader("Forest Plot Analysis")
                        forest_plot_all_features(df_wnv2, selected_feature, eeg_features, analysis_type)
                    st.divider()
            else:
                # Add RF classifier analysis for binary features (or non-binary with threshold)
                rf_classifier_analysis(df_wnv2, selected_feature, eeg_features, key_suffix="single")
                # Add RF regressor analysis for numerical features
                rf_regressor_analysis(df_wnv2, selected_feature, eeg_features, key_suffix="single")
                run_selected_feature(boxplot_columns)
                # Add forest plot if requested
                if forest_plot_clinical:
                    st.subheader("Forest Plot Analysis")
                    forest_plot_all_features(df_wnv2, selected_feature, eeg_features, analysis_type)
        
        # Add divider between feature types when running "All"
        if feature_type == "All" and current_feature_type != feature_types_to_run[-1]:
            st.divider()
    download_pptx_button()


def extract_sleep_stage_segment(raw, stage='N1', min_duration_min=1):
    """
    Extract a specific sleep stage segment from raw MNE data using YASA sleep staging.
    
    Parameters:
    -----------
    raw : mne.io.Raw
        MNE Raw object with EEG/ECG data
    stage : str
        Sleep stage to extract (N1, N2, N3, R, or W) (default: N1)
    min_duration_min : int
        Minimum duration of stage segment in minutes (default: 1)
        For W stage, this is ignored and 10 minutes is used
    
    Returns:
    --------
    mne.io.Raw or None
        Cropped raw object containing only the selected stage segment, or None if not found
    """
    import yasa
    import numpy as np
    
    try:
        # Check if file has ECG channel (optional but preferred)
        ch_lower = [ch.lower() for ch in raw.ch_names]
        ecg_indices = [i for i, ch in enumerate(ch_lower) if 'ecg' in ch or 'ekg' in ch]
        
        # Find the best channels for sleep staging
        available_eeg_channels = [ch for ch in raw.ch_names if ch in eeg_channels]
        if not available_eeg_channels:
            return None
        
        # Select optimal EEG channel (prefer central electrodes like C3, C4, Cz)
        central_eeg_channels = ['Cz', 'C3', 'C4']
        eeg_name = None
        for ch in central_eeg_channels:
            if ch in available_eeg_channels:
                eeg_name = ch
                break
        
        # If no central electrode, use the first available EEG channel
        if eeg_name is None:
            eeg_name = available_eeg_channels[0]
        
        # Find EOG channel
        eog_name = None
        if 'EOG+' in raw.ch_names:
            eog_name = 'EOG+'
        elif 'EOG' in raw.ch_names:
            eog_name = 'EOG'
        
        # Find EMG channel (prefer EMG1+ over EMG2+)
        emg_name = None
        if 'EMG1+' in raw.ch_names:
            emg_name = 'EMG1+'
        elif 'EMG2+' in raw.ch_names:
            emg_name = 'EMG2+'
        elif 'EMG' in raw.ch_names:
            emg_name = 'EMG'
        
        # Run YASA sleep staging
        ss = yasa.SleepStaging(raw, eeg_name=eeg_name, eog_name=eog_name, emg_name=emg_name)
        predicted_stages = ss.predict()
        
        # Special handling for W: select 10 min of Wake right after the last adequate non-W run
        if stage == 'W':
            # YASA uses 30-second epochs by default
            epoch_duration_sec = 30
            # Require at least 5 minutes of any non-W stage before the Wake segment
            min_non_w_epochs = int(np.ceil(5 * 60 / epoch_duration_sec))
            min_w_epochs = int(np.ceil(10 * 60 / epoch_duration_sec))  # 10 minutes of W

            # Build runs of identical stages: (label, start_idx, length)
            runs = []
            if len(predicted_stages) > 0:
                cur_label = predicted_stages[0]
                cur_start = 0
                cur_len = 1
                for i in range(1, len(predicted_stages)):
                    if predicted_stages[i] == cur_label:
                        cur_len += 1
                    else:
                        runs.append((cur_label, cur_start, cur_len))
                        cur_label = predicted_stages[i]
                        cur_start = i
                        cur_len = 1
                runs.append((cur_label, cur_start, cur_len))

            # Find the last non-W run (>= min_non_w_epochs) that is immediately followed by a W run (>= 10 min)
            selected_w_start_epoch = None
            for idx in range(len(runs) - 1, -1, -1):
                label, start_idx, length = runs[idx]
                if label == 'W' or length < min_non_w_epochs:
                    continue
                # Check next run exists and is Wake
                if idx + 1 < len(runs):
                    next_label, next_start, next_length = runs[idx + 1]
                    if next_label == 'W' and next_length >= min_w_epochs:
                        selected_w_start_epoch = next_start
                        break

            if selected_w_start_epoch is None:
                return None

            # Define the 10-minute Wake segment
            segment_start_sec = selected_w_start_epoch * epoch_duration_sec
            segment_end_sec = (selected_w_start_epoch + min_w_epochs) * epoch_duration_sec
            
            # Crop the raw data to the stage segment
            raw_stage = raw.copy().crop(tmin=segment_start_sec, tmax=segment_end_sec)
            return raw_stage

        else:
            # Find specified sleep stage epochs (N1, N2, N3, R)
            stage_epochs = np.where(predicted_stages == stage)[0]

            if len(stage_epochs) == 0:
                return None
            
            # Find the longest continuous streak of stage epochs (allowing gaps of up to 3 epochs)
            def find_longest_stage_streak(stage_epochs, max_gap=3):
                """Find the longest continuous streak of stage epochs, allowing gaps of up to max_gap epochs."""
                if len(stage_epochs) == 0:
                    return None, 0
                
                longest_start = None
                longest_length = 0
                current_start = stage_epochs[0]
                current_length = 1
                current_end = stage_epochs[0]
                
                for i in range(1, len(stage_epochs)):
                    gap = stage_epochs[i] - current_end
                    
                    if gap <= max_gap + 1:  # Allow gap of up to max_gap epochs
                        # Still part of the same streak
                        current_length += 1
                        current_end = stage_epochs[i]
                    else:
                        # Gap too large, check if this is the longest streak
                        if current_length > longest_length:
                            longest_start = current_start
                            longest_length = current_length
                        
                        # Start new streak
                        current_start = stage_epochs[i]
                        current_length = 1
                        current_end = stage_epochs[i]
                
                # Check the last streak
                if current_length > longest_length:
                    longest_start = current_start
                    longest_length = current_length
                
                return longest_start, longest_length
            
            # Find the longest stage streak
            longest_start, longest_length = find_longest_stage_streak(stage_epochs)
            
            if longest_start is None or longest_length == 0:
                return None
            
            # YASA uses 30-second epochs by default
            epoch_duration_sec = 30
            min_stage_epochs = int(min_duration_min * 60 / epoch_duration_sec)
            
            # Check if the longest streak meets the minimum duration requirement
            if longest_length >= min_stage_epochs:
                segment_start_sec = longest_start * epoch_duration_sec
                segment_end_sec = (longest_start + longest_length) * epoch_duration_sec
                
                # Crop the raw data to the stage segment
                raw_stage = raw.copy().crop(tmin=segment_start_sec, tmax=segment_end_sec)
                return raw_stage
            else:
                return None
                
    except Exception as e:
        return None


def compute_all_network_features(raw, patient_id):
    """
    Compute all EEG and HRV network features from raw MNE data.
    
    Parameters:
    -----------
    raw : mne.io.Raw
        MNE Raw object with EEG/ECG data
    patient_id : str
        Patient identifier
    
    Returns:
    --------
    dict : Dictionary with all computed features
    """
    features = {}
    
    try:
        # Get data and sampling frequency
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        ch_names = raw.ch_names
        
        if data.size == 0 or data.shape[0] < 2:
            return features
        
        # Separate EEG and ECG channels
        eeg_ch_idx = [i for i, ch in enumerate(ch_names) if 'eeg' in ch.lower() or any(elec in ch.upper() for elec in ['FP', 'F', 'C', 'P', 'O', 'T', 'A'])]
        ecg_ch_idx = [i for i, ch in enumerate(ch_names) if 'ecg' in ch.lower() or 'ekg' in ch.lower()]
        
        eeg_data = data[eeg_ch_idx, :] if eeg_ch_idx else data
        ecg_data = data[ecg_ch_idx, :] if ecg_ch_idx else None
        
        # Require EEG data, but ECG is optional
        if eeg_data is None or eeg_data.size == 0:
            return features

        n_eeg_channels = eeg_data.shape[0]
        
        if n_eeg_channels < 2:
            return features
        
        # Import required libraries
        from mne_connectivity import SpectralConnectivity as spectral_connectivity
        from scipy.signal import coherence, hilbert, welch
        from scipy.stats import pearsonr
        import networkx as nx
        from itertools import combinations
        import pandas as pd
        import mne.time_frequency
        
        # ========== EEG FEATURES (Overall and Per-Electrode) ==========
        
        # Define power bands
        power_bands = {
            "delta": [0.5, 4],
            "theta": [4, 8],
            "alpha": [8, 12],
            "beta": [12, 30],
            "gamma": [30, 100]
        }
        
        try:
            # Compute PSD and FFT for each channel
            window_size_sec = 5
            window_size = int(window_size_sec * sfreq)
            psd_list = []
            fft_list = []
            mpf_list = []
            df_list = []
            
            for i, channel_data in enumerate(eeg_data):
                ch_psd = []
                ch_fft = []
                ch_mpf = []
                ch_df = []
                
                # Slide windows
                for start in range(0, len(channel_data) - window_size + 1, window_size):
                    win = channel_data[start:start + window_size]
                    
                    # PSD
                    psd_vals, freqs = mne.time_frequency.psd_array_welch(
                        win, sfreq, fmin=1, fmax=40, n_fft=int(sfreq)
                    )
                    psd_vals = psd_vals.squeeze()
                    
                    # FFT
                    fft_vals = np.fft.fft(win)
                    
                    # MPF (Mean Power Frequency) and Dominant Frequency
                    mpf = np.sum(psd_vals * freqs) / np.sum(psd_vals) if np.sum(psd_vals) > 0 else 0.0
                    dfreq = freqs[np.argmax(psd_vals)] if len(psd_vals) > 0 else 0.0
                    
                    ch_psd.append(psd_vals)
                    ch_fft.append(fft_vals)
                    ch_mpf.append(mpf)
                    ch_df.append(dfreq)
                
                psd_list.append(ch_psd)
                fft_list.append(ch_fft)
                mpf_list.append(ch_mpf)
                df_list.append(ch_df)
            
            # Convert to arrays - handle lists of arrays
            # mpf_arr and df_arr are lists of lists, need to flatten for overall stats
            mpf_all = []
            df_all = []
            for ch_mpf in mpf_list:
                mpf_all.extend(ch_mpf)
            for ch_df in df_list:
                df_all.extend(ch_df)
            mpf_arr = np.array(mpf_all) if mpf_all else np.array([0.0])
            df_arr = np.array(df_all) if df_all else np.array([0.0])
            dfv_arr = np.array([np.std(ch_df) if len(ch_df) > 0 else 0.0 for ch_df in df_list])
            
            # Keep per-channel lists for per-channel stats
            mpf_list_per_ch = mpf_list
            df_list_per_ch = df_list
            
            # Compute mean PSD and FFT for each channel (from list of arrays)
            psd_means = []
            psd_stds = []
            fft_means = []
            fft_stds = []
            for ch_psd_list in psd_list:
                if len(ch_psd_list) > 0:
                    ch_psd_concat = np.concatenate(ch_psd_list)
                    psd_means.append(np.mean(ch_psd_concat))
                    psd_stds.append(np.std(ch_psd_concat))
                else:
                    psd_means.append(0.0)
                    psd_stds.append(0.0)
            
            for ch_fft_list in fft_list:
                if len(ch_fft_list) > 0:
                    ch_fft_concat = np.concatenate(ch_fft_list)
                    fft_means.append(np.mean(np.abs(ch_fft_concat)))
                    fft_stds.append(np.std(np.abs(ch_fft_concat)))
                else:
                    fft_means.append(0.0)
                    fft_stds.append(0.0)
            
            psd_means_arr = np.array(psd_means)
            psd_stds_arr = np.array(psd_stds)
            fft_means_arr = np.array(fft_means)
            fft_stds_arr = np.array(fft_stds)
            
            # Bandpower computation
            def bandpower(data, sf, band, window_sec=None, relative=False):
                low, high = band
                nperseg = int((window_sec or (2/low)) * sf)
                freqs, psd = welch(data, sf, nperseg=nperseg)
                idx = np.logical_and(freqs >= low, freqs <= high)
                bp = np.trapz(psd[:, idx], freqs[idx], axis=1)
                if relative:
                    bp /= np.trapz(psd, freqs, axis=1)
                return bp
            
            band_powers = {name: bandpower(eeg_data, sfreq, rng, window_sec=window_size_sec) 
                          for name, rng in power_bands.items()}
            
            # Time-domain stats (skewness and kurtosis)
            skew = np.apply_along_axis(lambda x: pd.Series(x).skew(), 1, eeg_data)
            kurt = np.apply_along_axis(lambda x: pd.Series(x).kurt(), 1, eeg_data)
            
            # Overall features
            features['overall_min'] = float(eeg_data.min())
            features['overall_max'] = float(eeg_data.max())
            features['overall_mean'] = float(eeg_data.mean())
            features['overall_median'] = float(np.median(eeg_data))
            features['overall_std'] = float(eeg_data.std())
            features['overall_psd_mean'] = float(psd_means_arr.mean())
            features['overall_psd_std'] = float(psd_stds_arr.mean())
            features['overall_fft_mean'] = float(fft_means_arr.mean())
            features['overall_fft_std'] = float(fft_stds_arr.mean())
            features['overall_delta_power'] = float(band_powers['delta'].mean())
            features['overall_theta_power'] = float(band_powers['theta'].mean())
            features['overall_alpha_power'] = float(band_powers['alpha'].mean())
            features['overall_beta_power'] = float(band_powers['beta'].mean())
            features['overall_gamma_power'] = float(band_powers['gamma'].mean())
            features['overall_skewness'] = float(skew.mean())
            features['overall_kurtosis'] = float(kurt.mean())
            features['overall_mpf_mean'] = float(mpf_arr.mean())
            features['overall_mpf_median'] = float(np.median(mpf_arr))
            features['overall_df_mean'] = float(df_arr.mean())
            features['overall_dfv_std'] = float(dfv_arr.mean())
            
            # Channel-specific features
            eeg_ch_names = [ch_names[i] for i in eeg_ch_idx] if eeg_ch_idx else ch_names[:n_eeg_channels]
            for i, ch in enumerate(eeg_ch_names):
                features[f'min_EEG {ch}'] = float(eeg_data[i].min())
                features[f'max_EEG {ch}'] = float(eeg_data[i].max())
                features[f'mean_EEG {ch}'] = float(eeg_data[i].mean())
                features[f'median_EEG {ch}'] = float(np.median(eeg_data[i]))
                features[f'std_EEG {ch}'] = float(eeg_data[i].std())
                features[f'psd_mean_EEG {ch}'] = float(psd_means_arr[i])
                features[f'psd_std_EEG {ch}'] = float(psd_stds_arr[i])
                features[f'fft_mean_EEG {ch}'] = float(fft_means_arr[i])
                features[f'fft_std_EEG {ch}'] = float(fft_stds_arr[i])
                features[f'delta_power_EEG {ch}'] = float(band_powers['delta'][i])
                features[f'theta_power_EEG {ch}'] = float(band_powers['theta'][i])
                features[f'alpha_power_EEG {ch}'] = float(band_powers['alpha'][i])
                features[f'beta_power_EEG {ch}'] = float(band_powers['beta'][i])
                features[f'gamma_power_EEG {ch}'] = float(band_powers['gamma'][i])
                features[f'skewness_EEG {ch}'] = float(skew[i])
                features[f'kurtosis_EEG {ch}'] = float(kurt[i])
                features[f'mean_mpf_EEG {ch}'] = float(np.mean(mpf_list_per_ch[i])) if len(mpf_list_per_ch[i]) > 0 else 0.0
                features[f'median_mpf_EEG {ch}'] = float(np.median(mpf_list_per_ch[i])) if len(mpf_list_per_ch[i]) > 0 else 0.0
                features[f'dfv_mean_EEG {ch}'] = float(np.mean(df_list_per_ch[i])) if len(df_list_per_ch[i]) > 0 else 0.0
                features[f'dfv_std_EEG {ch}'] = float(dfv_arr[i])
        except Exception as e:
            # If EEG feature computation fails, continue with network features
            pass
        
        # ========== EEG NETWORK FEATURES ==========
        
        # 1. Coherence Matrix
        try:
            data_for_conn = eeg_data[np.newaxis, :, :]
            conn_res = spectral_connectivity(
                data_for_conn,
                method='coh',
                sfreq=sfreq,
                fmin=1.0,
                fmax=40.0,
                faverage=True,
                verbose=False,
            )
            conn = conn_res[0]
            if conn.ndim == 3:
                coherence_matrix = np.mean(conn, axis=2)
            elif conn.ndim == 2:
                n_ch = eeg_data.shape[0]
                conn_mean = np.mean(conn, axis=1)
                coherence_matrix = np.zeros((n_ch, n_ch))
                triu_idx = np.triu_indices(n_ch, k=1)
                coherence_matrix[triu_idx] = conn_mean
                coherence_matrix[(triu_idx[1], triu_idx[0])] = conn_mean
            else:
                coherence_matrix = np.corrcoef(eeg_data)
                coherence_matrix = np.nan_to_num(coherence_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(coherence_matrix, 1.0)
            features['coherence_matrix'] = coherence_matrix
        except:
            coherence_matrix = np.corrcoef(eeg_data)
            coherence_matrix = np.nan_to_num(coherence_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(coherence_matrix, 1.0)
            features['coherence_matrix'] = coherence_matrix
        
        # 2. Imaginary Coherence
        try:
            conn_res = spectral_connectivity(
                data_for_conn,
                method='imcoh',
                sfreq=sfreq,
                fmin=1.0,
                fmax=40.0,
                faverage=True,
                verbose=False,
            )
            conn = conn_res[0]
            if conn.ndim == 3:
                imaginary_coherence = np.mean(np.abs(conn), axis=2)
            elif conn.ndim == 2:
                n_ch = eeg_data.shape[0]
                conn_mean = np.mean(np.abs(conn), axis=1)
                imaginary_coherence = np.zeros((n_ch, n_ch))
                triu_idx = np.triu_indices(n_ch, k=1)
                imaginary_coherence[triu_idx] = conn_mean
                imaginary_coherence[(triu_idx[1], triu_idx[0])] = conn_mean
            else:
                imaginary_coherence = np.abs(np.imag(np.fft.fft(eeg_data)))
            np.fill_diagonal(imaginary_coherence, 0.0)
            features['imaginary_coherence'] = imaginary_coherence
        except:
            features['imaginary_coherence'] = np.zeros((n_eeg_channels, n_eeg_channels))
        
        # 3. wPLI (Weighted Phase Lag Index)
        try:
            conn_res = spectral_connectivity(
                data_for_conn,
                method='wpli',
                sfreq=sfreq,
                fmin=1.0,
                fmax=40.0,
                faverage=True,
                verbose=False,
            )
            conn = conn_res[0]
            if conn.ndim == 3:
                wPLI_matrix = np.mean(conn, axis=2)
            elif conn.ndim == 2:
                n_ch = eeg_data.shape[0]
                conn_mean = np.mean(conn, axis=1)
                wPLI_matrix = np.zeros((n_ch, n_ch))
                triu_idx = np.triu_indices(n_ch, k=1)
                wPLI_matrix[triu_idx] = conn_mean
                wPLI_matrix[(triu_idx[1], triu_idx[0])] = conn_mean
            else:
                wPLI_matrix = np.zeros((n_eeg_channels, n_eeg_channels))
            np.fill_diagonal(wPLI_matrix, 0.0)
            features['wPLI_matrix'] = wPLI_matrix
        except:
            features['wPLI_matrix'] = np.zeros((n_eeg_channels, n_eeg_channels))
        
        # 4. PLI (Phase Lag Index)
        try:
            conn_res = spectral_connectivity(
                data_for_conn,
                method='pli',
                sfreq=sfreq,
                fmin=1.0,
                fmax=40.0,
                faverage=True,
                verbose=False,
            )
            conn = conn_res[0]
            if conn.ndim == 3:
                PLI_matrix = np.mean(conn, axis=2)
            elif conn.ndim == 2:
                n_ch = eeg_data.shape[0]
                conn_mean = np.mean(conn, axis=1)
                PLI_matrix = np.zeros((n_ch, n_ch))
                triu_idx = np.triu_indices(n_ch, k=1)
                PLI_matrix[triu_idx] = conn_mean
                PLI_matrix[(triu_idx[1], triu_idx[0])] = conn_mean
            else:
                PLI_matrix = np.zeros((n_eeg_channels, n_eeg_channels))
            np.fill_diagonal(PLI_matrix, 0.0)
            features['PLI_matrix'] = PLI_matrix
        except:
            features['PLI_matrix'] = np.zeros((n_eeg_channels, n_eeg_channels))
        
        # 5. Amplitude Envelope Correlation (AEC)
        try:
            # Compute Hilbert transform for amplitude envelope
            analytic_signal = hilbert(eeg_data, axis=1)
            amplitude_envelope = np.abs(analytic_signal)
            amplitude_envelope_corr = np.corrcoef(amplitude_envelope)
            amplitude_envelope_corr = np.nan_to_num(amplitude_envelope_corr, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(amplitude_envelope_corr, 1.0)
            features['amplitude_envelope_corr'] = amplitude_envelope_corr
        except:
            amplitude_envelope_corr = np.corrcoef(eeg_data)
            amplitude_envelope_corr = np.nan_to_num(amplitude_envelope_corr, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(amplitude_envelope_corr, 1.0)
            features['amplitude_envelope_corr'] = amplitude_envelope_corr
        
        # 6. Phase Locking Value (PLV)
        try:
            analytic_signal = hilbert(eeg_data, axis=1)
            phase = np.angle(analytic_signal)
            phase_locking_value = np.zeros((n_eeg_channels, n_eeg_channels))
            for i, j in combinations(range(n_eeg_channels), 2):
                phase_diff = phase[i] - phase[j]
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                phase_locking_value[i, j] = phase_locking_value[j, i] = plv
            np.fill_diagonal(phase_locking_value, 1.0)
            features['phase_locking_value'] = phase_locking_value
        except:
            features['phase_locking_value'] = np.zeros((n_eeg_channels, n_eeg_channels))
        
        # Use coherence_matrix as base for graph metrics
        if 'coherence_matrix' in features:
            conn_matrix = features['coherence_matrix']
        else:
            conn_matrix = np.corrcoef(eeg_data)
        
        # Threshold connectivity matrix (keep top 20% connections)
        threshold = np.percentile(conn_matrix[np.triu_indices_from(conn_matrix, k=1)], 80)
        binarized = (conn_matrix >= threshold).astype(float)
        np.fill_diagonal(binarized, 0)
        
        # Build network graph
        G = nx.from_numpy_array(binarized)
        
        # 7. Graph Degree
        try:
            degrees = dict(G.degree())
            graph_degree = np.array([degrees.get(i, 0) for i in range(n_eeg_channels)])
            features['graph_degree'] = graph_degree
        except:
            features['graph_degree'] = np.zeros(n_eeg_channels)
        
        # 8. Graph Strength (weighted degree)
        try:
            strengths = dict(G.degree(weight='weight'))
            graph_strength = np.array([strengths.get(i, 0) for i in range(n_eeg_channels)])
            features['graph_strength'] = graph_strength
        except:
            features['graph_strength'] = np.sum(conn_matrix, axis=1) - np.diag(conn_matrix)
        
        # 9. Clustering Coefficient
        try:
            clustering_coeff = nx.clustering(G)
            clustering_coefficient = np.array([clustering_coeff.get(i, 0) for i in range(n_eeg_channels)])
            features['clustering_coefficient'] = clustering_coefficient
        except:
            features['clustering_coefficient'] = np.zeros(n_eeg_channels)
        
        # 10. Global Efficiency
        try:
            global_efficiency = nx.global_efficiency(G)
            features['global_efficiency'] = global_efficiency if np.isfinite(global_efficiency) else 0.0
        except:
            features['global_efficiency'] = 0.0
        
        # 11. Characteristic Path Length
        try:
            if nx.is_connected(G):
                path_length = nx.average_shortest_path_length(G)
                features['characteristic_path_length'] = path_length if np.isfinite(path_length) else np.inf
            else:
                # For disconnected graphs, compute for each component
                components = list(nx.connected_components(G))
                path_lengths = []
                for comp in components:
                    subgraph = G.subgraph(comp)
                    if len(comp) > 1:
                        path_lengths.append(nx.average_shortest_path_length(subgraph))
                features['characteristic_path_length'] = np.mean(path_lengths) if path_lengths else np.inf
        except:
            features['characteristic_path_length'] = np.inf
        
        # 12. Modularity
        try:
            communities = nx.community.greedy_modularity_communities(G)
            modularity = nx.community.modularity(G, communities)
            features['modularity'] = modularity if np.isfinite(modularity) else 0.0
        except:
            features['modularity'] = 0.0
        
        # 13. Betweenness Centrality
        try:
            betweenness = nx.betweenness_centrality(G)
            betweenness_centrality = np.array([betweenness.get(i, 0) for i in range(n_eeg_channels)])
            features['betweenness_centrality'] = betweenness_centrality
        except:
            features['betweenness_centrality'] = np.zeros(n_eeg_channels)
        
        # 14. Eigenvector Centrality
        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=100)
            eigenvector_centrality = np.array([eigenvector.get(i, 0) for i in range(n_eeg_channels)])
            features['eigenvector_centrality'] = eigenvector_centrality
        except:
            features['eigenvector_centrality'] = np.zeros(n_eeg_channels)
        
        # 15. Participation Coefficient
        try:
            communities = nx.community.greedy_modularity_communities(G)
            community_dict = {}
            for idx, comm in enumerate(communities):
                for node in comm:
                    community_dict[node] = idx
            
            participation_coefficient = np.zeros(n_eeg_channels)
            for i in range(n_eeg_channels):
                ki = G.degree(i)
                if ki > 0:
                    kis = {}
                    for neighbor in G.neighbors(i):
                        comm = community_dict.get(neighbor, -1)
                        kis[comm] = kis.get(comm, 0) + 1
                    M = len(communities)
                    part_coef = 1 - sum((kis.get(m, 0) / ki) ** 2 for m in range(M))
                    participation_coefficient[i] = part_coef
            features['participation_coefficient'] = participation_coefficient
        except:
            features['participation_coefficient'] = np.zeros(n_eeg_channels)
        
        # 16. Small Worldness
        try:
            if nx.is_connected(G):
                C_rand = 2 * G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1))
                L_rand = np.log(G.number_of_nodes()) / np.log(np.mean([G.degree(n) for n in G.nodes()]))
                C = nx.transitivity(G)
                L = nx.average_shortest_path_length(G)
                small_worldness = (C / C_rand) / (L / L_rand) if L_rand > 0 and C_rand > 0 else 0.0
                features['small_worldness'] = small_worldness if np.isfinite(small_worldness) else 0.0
            else:
                features['small_worldness'] = 0.0
        except:
            features['small_worldness'] = 0.0
        
        # 17. Assortativity
        try:
            assortativity = nx.degree_pearson_correlation_coefficient(G)
            features['assortativity'] = assortativity if np.isfinite(assortativity) else 0.0
        except:
            features['assortativity'] = 0.0
        
        # 18. Network Entropy
        try:
            degrees = [G.degree(n) for n in G.nodes()]
            if sum(degrees) > 0:
                degree_dist = np.array(degrees) / sum(degrees)
                network_entropy = -np.sum(degree_dist * np.log2(degree_dist + 1e-10))
            else:
                network_entropy = 0.0
            features['network_entropy'] = network_entropy if np.isfinite(network_entropy) else 0.0
        except:
            features['network_entropy'] = 0.0
        
        # 19. Dynamic Connectivity Entropy (simplified - variance of connectivity over time)
        try:
            # Compute connectivity in sliding windows
            window_size = int(sfreq * 5)  # 5 second windows
            n_windows = max(1, eeg_data.shape[1] // window_size)
            conn_variance = []
            for w in range(n_windows):
                start = w * window_size
                end = min(start + window_size, eeg_data.shape[1])
                window_data = eeg_data[:, start:end]
                if window_data.shape[1] > 10:
                    try:
                        # Check for constant or near-constant channels before correlation
                        # Remove channels with insufficient variance
                        valid_channels = []
                        for ch_idx in range(window_data.shape[0]):
                            ch_data = window_data[ch_idx, :]
                            if np.std(ch_data) > 1e-10 and not np.allclose(ch_data, ch_data[0], atol=1e-10):
                                valid_channels.append(ch_idx)
                        
                        if len(valid_channels) >= 2:
                            window_data_clean = window_data[valid_channels, :]
                            window_conn = np.corrcoef(window_data_clean)
                            # Replace NaN and Inf values with 0
                            window_conn = np.nan_to_num(window_conn, nan=0.0, posinf=0.0, neginf=0.0)
                            var_val = np.var(window_conn)
                            if np.isfinite(var_val) and not np.isnan(var_val):
                                conn_variance.append(var_val)
                    except:
                        continue
            dynamic_connectivity_entropy = np.std(conn_variance) if conn_variance else 0.0
            if not np.isfinite(dynamic_connectivity_entropy):
                dynamic_connectivity_entropy = 0.0
            features['dynamic_connectivity_entropy'] = dynamic_connectivity_entropy
        except:
            features['dynamic_connectivity_entropy'] = 0.0
        
        # 20. Functional Connectivity Density
        try:
            functional_connectivity_density = np.mean(conn_matrix)
            features['functional_connectivity_density'] = functional_connectivity_density
        except:
            features['functional_connectivity_density'] = 0.0
        
        # 21. PCA Components (explaining 95% variance)
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Standardize the data
            scaler = StandardScaler()
            eeg_data_scaled = scaler.fit_transform(eeg_data.T).T  # Scale across time for each channel
            
            # Reshape for PCA: (n_samples, n_features) where samples are time points, features are channels
            eeg_data_for_pca = eeg_data_scaled.T  # Shape: (n_timepoints, n_channels)
            
            # Find number of components for 95% variance
            pca_full = PCA()
            pca_full.fit(eeg_data_for_pca)
            cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
            n_components_95 = np.argmax(cumsum_variance >= 0.95) + 1
            n_components_95 = min(n_components_95, eeg_data_for_pca.shape[1])
            
            # Apply PCA with 95% variance components
            pca = PCA(n_components=n_components_95)
            pca_components = pca.fit_transform(eeg_data_for_pca)
            
            # Store PCA components (transpose back to match original shape concept)
            features['pca_components_95'] = pca_components.T  # Shape: (n_components, n_timepoints)
            features['pca_explained_variance_ratio'] = pca.explained_variance_ratio_
            features['pca_n_components_95'] = n_components_95
            features['pca_total_variance_explained'] = np.sum(pca.explained_variance_ratio_)
        except Exception as e:
            features['pca_components_95'] = np.zeros((1, eeg_data.shape[1]))
            features['pca_explained_variance_ratio'] = np.array([0.0])
            features['pca_n_components_95'] = 0
            features['pca_total_variance_explained'] = 0.0
        
        # 22. Autoencoder Features (explaining 95% variance)
        try:
            # Use a simple autoencoder approach
            # For simplicity, we'll use PCA as a linear autoencoder baseline
            # A full autoencoder would require neural network libraries
            
            # Option 1: Use PCA as linear autoencoder (fast, always available)
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            scaler = StandardScaler()
            eeg_data_scaled = scaler.fit_transform(eeg_data.T).T
            eeg_data_for_ae = eeg_data_scaled.T
            
            # Find components for 95% variance
            pca_ae = PCA()
            pca_ae.fit(eeg_data_for_ae)
            cumsum_variance = np.cumsum(pca_ae.explained_variance_ratio_)
            n_components_95_ae = np.argmax(cumsum_variance >= 0.95) + 1
            n_components_95_ae = min(n_components_95_ae, eeg_data_for_ae.shape[1])
            
            # Encode and decode
            pca_encoder = PCA(n_components=n_components_95_ae)
            encoded = pca_encoder.fit_transform(eeg_data_for_ae)
            decoded = pca_encoder.inverse_transform(encoded)
            
            # Calculate reconstruction error
            reconstruction_error = np.mean((eeg_data_for_ae - decoded) ** 2)
            variance_explained = np.sum(pca_encoder.explained_variance_ratio_)
            
            # Store autoencoder features (encoded representation)
            features['autoencoder_features_95'] = encoded.T  # Shape: (n_components, n_timepoints)
            features['autoencoder_n_components_95'] = n_components_95_ae
            features['autoencoder_variance_explained'] = variance_explained
            features['autoencoder_reconstruction_error'] = reconstruction_error
            
            # Option 2: Try to use a neural network autoencoder if available
            try:
                import tensorflow as tf
                from tensorflow import keras
                from tensorflow.keras import layers
                
                # Build a simple autoencoder
                input_dim = eeg_data_for_ae.shape[1]
                encoding_dim = n_components_95_ae
                
                # Create encoder
                input_layer = keras.Input(shape=(input_dim,))
                encoded = layers.Dense(encoding_dim * 2, activation='relu')(input_layer)
                encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
                
                # Create decoder
                decoded = layers.Dense(encoding_dim * 2, activation='relu')(encoded)
                decoded = layers.Dense(input_dim, activation='linear')(decoded)
                
                # Create autoencoder
                autoencoder = keras.Model(input_layer, decoded)
                encoder = keras.Model(input_layer, encoded)
                
                # Compile and train (quick training for feature extraction)
                autoencoder.compile(optimizer='adam', loss='mse')
                
                # Train on a subset if data is too large
                max_samples = min(10000, eeg_data_for_ae.shape[0])
                train_data = eeg_data_for_ae[:max_samples]
                
                # Quick training (few epochs for speed)
                autoencoder.fit(train_data, train_data, epochs=5, batch_size=32, verbose=0)
                
                # Get encoded features for all data
                encoded_nn = encoder.predict(eeg_data_for_ae, verbose=0)
                
                # Calculate variance explained
                decoded_nn = autoencoder.predict(eeg_data_for_ae, verbose=0)
                reconstruction_error_nn = np.mean((eeg_data_for_ae - decoded_nn) ** 2)
                
                # Calculate variance explained by comparing to original variance
                original_var = np.var(eeg_data_for_ae, axis=0).sum()
                residual_var = np.var(eeg_data_for_ae - decoded_nn, axis=0).sum()
                variance_explained_nn = 1 - (residual_var / original_var) if original_var > 0 else 0.0
                
                # Store neural network autoencoder features
                features['autoencoder_features_95_nn'] = encoded_nn.T
                features['autoencoder_variance_explained_nn'] = variance_explained_nn
                features['autoencoder_reconstruction_error_nn'] = reconstruction_error_nn
                
            except ImportError:
                # TensorFlow not available, use PCA-based autoencoder only
                pass
            except Exception as e:
                # Neural network training failed, use PCA-based autoencoder only
                pass
                
        except Exception as e:
            features['autoencoder_features_95'] = np.zeros((1, eeg_data.shape[1]))
            features['autoencoder_n_components_95'] = 0
            features['autoencoder_variance_explained'] = 0.0
            features['autoencoder_reconstruction_error'] = np.inf
        
        # ========== HRV NETWORK / COUPLED FEATURES ==========
        
        if ecg_data is not None and ecg_data.shape[0] > 0:
            ecg_signal = ecg_data[0, :]  # Use first ECG channel
            
            # 21. Time Domain Matrix (HRV metrics covariance)
            try:
                # Compute HRV time-domain metrics in windows
                window_size = int(sfreq * 30)  # 30 second windows
                n_windows = max(1, len(ecg_signal) // window_size)
                hrv_metrics = []
                for w in range(n_windows):
                    start = w * window_size
                    end = min(start + window_size, len(ecg_signal))
                    window_ecg = ecg_signal[start:end]
                    if len(window_ecg) > 10:
                        # Simple HRV metrics
                        rr_intervals = np.diff(np.where(np.diff(window_ecg) > np.std(window_ecg))[0])
                        if len(rr_intervals) > 1:
                            hrv_metrics.append([
                                np.mean(rr_intervals),
                                np.std(rr_intervals),
                                np.var(rr_intervals)
                            ])
                if len(hrv_metrics) > 1:
                    time_domain_matrix = np.cov(np.array(hrv_metrics).T)
                else:
                    time_domain_matrix = np.zeros((3, 3))
                features['time_domain_matrix'] = time_domain_matrix
            except:
                features['time_domain_matrix'] = np.zeros((3, 3))
            
            # Heart-Brain Coherence
            try:
                eeg_power = np.mean(np.abs(np.fft.fft(eeg_data, axis=1)), axis=0)
                ecg_power = np.abs(np.fft.fft(ecg_signal))
                min_len = min(len(eeg_power), len(ecg_power))
                heart_brain_coherence = np.corrcoef(eeg_power[:min_len], ecg_power[:min_len])[0, 1]
                features['heart_brain_coherence'] = heart_brain_coherence if np.isfinite(heart_brain_coherence) else 0.0
            except:
                features['heart_brain_coherence'] = 0.0
            
            features['phase_amplitude_coupling'] = np.zeros(n_eeg_channels)
            features['mutual_information_matrix'] = np.zeros((n_eeg_channels, n_eeg_channels))
            features['graph_efficiency_heart_brain'] = 0.0
            features['network_synchrony_index'] = 0.0
        else:
            # No ECG data available
            for hrv_feat in ['time_domain_matrix', 'frequency_coupling_matrix', 'transfer_entropy_matrix',
                           'granger_causality_values', 'cross_correlation_matrix', 'heart_brain_coherence',
                           'phase_amplitude_coupling', 'mutual_information_matrix', 'graph_efficiency_heart_brain',
                           'network_synchrony_index']:
                features[hrv_feat] = np.zeros((1, 1)) if 'matrix' in hrv_feat else 0.0
        
    except Exception as e:
        st.warning(f"Error computing network features for {patient_id}: {str(e)}")
    
    return features


def get_cache_key(project_name, patient_id, pickle_file, sleep_stage, min_duration_min):
    """
    Generate a cache key based on parameters that affect the results.
    
    Parameters:
    -----------
    project_name : str
        Name of the project
    patient_id : str
        Patient identifier
    pickle_file : str
        Name of the pickle file
    sleep_stage : str or None
        Sleep stage to extract
    min_duration_min : int
        Minimum duration in minutes
    
    Returns:
    --------
    str : Cache key string
    """
    import hashlib
    
    # Create a unique key based on all parameters
    cache_params = {
        'project_name': project_name,
        'patient_id': patient_id,
        'pickle_file': pickle_file,
        'sleep_stage': sleep_stage if sleep_stage else 'full',
        'min_duration_min': min_duration_min,
        'cache_version': '1.0'  # Increment if feature computation changes
    }
    
    # Create hash from sorted parameters
    param_string = '_'.join([f"{k}:{v}" for k, v in sorted(cache_params.items())])
    cache_key = hashlib.md5(param_string.encode()).hexdigest()
    return cache_key


def get_cache_path(project_name, cache_key):
    """
    Get the cache file path for a given cache key.
    
    Parameters:
    -----------
    project_name : str
        Name of the project
    cache_key : str
        Cache key
    
    Returns:
    --------
    str : Path to cache file
    """
    cache_dir = os.path.join('cache', project_name)
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{cache_key}.pkl")


def load_from_cache(cache_path):
    """
    Load cached results if they exist.
    
    Parameters:
    -----------
    cache_path : str
        Path to cache file
    
    Returns:
    --------
    dict or None : Cached results or None if not found
    """
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            # If cache is corrupted, return None
            return None
    return None


def save_to_cache(cache_path, data):
    """
    Save results to cache.
    
    Parameters:
    -----------
    cache_path : str
        Path to cache file
    data : dict
        Data to cache (should contain 'patient_features' and 'feature_data_entries')
    """
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        # If caching fails, just continue without cache
        pass


def extract_systole_diastole_segments(raw, sfreq):
    """
    Extract systole (RST) and diastole (PQR) segments from ECG data.
    
    Parameters:
    -----------
    raw : mne.io.Raw
        MNE Raw object with EEG/ECG data
    sfreq : float
        Sampling frequency
    
    Returns:
    --------
    dict : Dictionary with 'systole' and 'diastole' keys, each containing:
        - 'eeg_data': EEG data during that phase (n_channels, n_samples)
        - 'ecg_data': ECG data during that phase (n_samples,)
        - 'indices': Sample indices for the segments
    """
    import neurokit2 as nk
    import numpy as np
    
    # Get channel names and data
    ch_names = raw.ch_names
    data = raw.get_data()
    
    # Find ECG channel
    ch_lower = [ch.lower() for ch in ch_names]
    ecg_indices = [i for i, ch in enumerate(ch_lower) if 'ecg' in ch or 'ekg' in ch]
    
    if not ecg_indices:
        return None
    
    ecg_ch_idx = ecg_indices[0]
    ecg_signal = data[ecg_ch_idx, :]
    
    # Clean ECG signal
    try:
        ecg_clean = nk.ecg_clean(ecg_signal, sampling_rate=sfreq)
    except:
        ecg_clean = ecg_signal
    
    # Detect R-peaks
    try:
        _, rpk = nk.ecg_peaks(ecg_clean, sampling_rate=sfreq)
        rpeaks = rpk['ECG_R_Peaks']
    except:
        # Fallback: simple peak detection
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(ecg_clean, distance=int(sfreq * 0.3))
        rpeaks = peaks
    
    if len(rpeaks) < 2:
        return None
    
    # Find EEG channels
    eeg_ch_idx = [i for i, ch in enumerate(ch_names) 
                  if 'eeg' in ch.lower() or any(elec in ch.upper() for elec in ['FP', 'F', 'C', 'P', 'O', 'T', 'A'])]
    
    if not eeg_ch_idx:
        return None
    
    eeg_data = data[eeg_ch_idx, :]
    
    # Extract segments
    systole_segments_eeg = []
    systole_segments_ecg = []
    systole_indices = []
    
    diastole_segments_eeg = []
    diastole_segments_ecg = []
    diastole_indices = []
    
    # For each R-R interval
    for i in range(len(rpeaks) - 1):
        r_start = rpeaks[i]
        r_end = rpeaks[i + 1]
        rr_length = r_end - r_start
        
        if rr_length < 10:  # Skip very short intervals
            continue
        
        # Systole: R to T (RST) - approximately first 40% of R-R interval (R peak to T wave)
        systole_end = r_start + int(0.4 * rr_length)
        if systole_end > r_start and systole_end < len(ecg_signal):
            systole_segments_eeg.append(eeg_data[:, r_start:systole_end])
            systole_segments_ecg.append(ecg_signal[r_start:systole_end])
            systole_indices.append((r_start, systole_end))
        
        # Diastole: P to R (PQR) - approximately last 30% of previous R-R interval (P wave to R peak)
        # For the first R peak, we need to look at the previous interval
        if i > 0:
            prev_r_start = rpeaks[i - 1]
            prev_rr_length = r_start - prev_r_start
            # P wave typically starts around 70-80% into the previous R-R interval
            diastole_start = prev_r_start + int(0.7 * prev_rr_length)
            diastole_end = r_start
            if diastole_start < diastole_end and diastole_end < len(ecg_signal) and diastole_start >= 0:
                diastole_segments_eeg.append(eeg_data[:, diastole_start:diastole_end])
                diastole_segments_ecg.append(ecg_signal[diastole_start:diastole_end])
                diastole_indices.append((diastole_start, diastole_end))
    
    # Concatenate all segments
    if systole_segments_eeg and diastole_segments_eeg:
        # Pad segments to same length for concatenation
        max_systole_len = max(seg.shape[1] for seg in systole_segments_eeg) if systole_segments_eeg else 0
        max_diastole_len = max(seg.shape[1] for seg in diastole_segments_eeg) if diastole_segments_eeg else 0
        
        # For feature extraction, we'll use all segments separately
        return {
            'systole': {
                'eeg_segments': systole_segments_eeg,
                'ecg_segments': systole_segments_ecg,
                'indices': systole_indices
            },
            'diastole': {
                'eeg_segments': diastole_segments_eeg,
                'ecg_segments': diastole_segments_ecg,
                'indices': diastole_indices
            }
        }
    
    return None


def compute_autoencoder_features(data, feature_name_prefix="", variance_threshold=0.95):
    """
    Compute autoencoder features from data using PCA-based autoencoder.
    
    Parameters:
    -----------
    data : np.ndarray
        Data array. For EEG: (n_channels, n_timepoints) or list of segments
        For ECG: (n_timepoints,) or list of segments
    feature_name_prefix : str
        Prefix for feature names (e.g., "eeg", "ecg")
    variance_threshold : float
        Variance threshold for PCA (default: 0.95)
    
    Returns:
    --------
    dict : Dictionary with autoencoder features
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    features = {}
    
    try:
        # Handle list of segments - concatenate them
        if isinstance(data, list):
            if len(data) == 0:
                return features
            # Concatenate segments
            if isinstance(data[0], np.ndarray):
                if data[0].ndim == 1:  # ECG segments
                    data_concatenated = np.concatenate(data)
                    data_for_ae = data_concatenated.reshape(-1, 1).T  # Shape: (1, n_samples)
                else:  # EEG segments (n_channels, n_samples)
                    # Concatenate along time axis
                    data_concatenated = np.concatenate(data, axis=1)  # Shape: (n_channels, total_samples)
                    data_for_ae = data_concatenated.T  # Shape: (total_samples, n_channels)
            else:
                return features
        else:
            # Single array
            if data.ndim == 1:  # ECG
                data_for_ae = data.reshape(-1, 1).T  # Shape: (1, n_samples)
            else:  # EEG (n_channels, n_samples)
                data_for_ae = data.T  # Shape: (n_samples, n_channels)
        
        if data_for_ae.size == 0:
            return features
        
        # Standardize
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_for_ae)
        
        # Find components for variance threshold
        pca_ae = PCA()
        pca_ae.fit(data_scaled)
        cumsum_variance = np.cumsum(pca_ae.explained_variance_ratio_)
        n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
        n_components = min(n_components, data_scaled.shape[1], data_scaled.shape[0])
        
        if n_components == 0:
            n_components = 1
        
        # Encode
        pca_encoder = PCA(n_components=n_components)
        encoded = pca_encoder.fit_transform(data_scaled)
        
        # Calculate metrics
        decoded = pca_encoder.inverse_transform(encoded)
        reconstruction_error = np.mean((data_scaled - decoded) ** 2)
        variance_explained = np.sum(pca_encoder.explained_variance_ratio_)
        
        # Store features
        # For feature matrix, we'll use mean of encoded features across time
        features[f'{feature_name_prefix}_autoencoder_features_mean'] = np.mean(encoded, axis=0)
        features[f'{feature_name_prefix}_autoencoder_features_std'] = np.std(encoded, axis=0)
        features[f'{feature_name_prefix}_autoencoder_n_components'] = n_components
        features[f'{feature_name_prefix}_autoencoder_variance_explained'] = variance_explained
        features[f'{feature_name_prefix}_autoencoder_reconstruction_error'] = reconstruction_error
        
        # Also store the full encoded features (flattened) for more detailed analysis
        features[f'{feature_name_prefix}_autoencoder_features_full'] = encoded.flatten()
        
    except Exception as e:
        # Return empty features on error
        pass
    
    return features


def systole_diastole_comparison_run(project_name, sleep_stage=None, min_duration_min=1):
    """
    Extract systole and diastole segments from ECG and compute autoencoder features
    for EEG and ECG data in both phases.
    
    Parameters:
    -----------
    project_name : str
        Name of the project (used to locate pickle files)
    sleep_stage : str, optional
        Sleep stage to extract (N1, N2, N3, R, or W). If None, uses full recording (default: None)
    min_duration_min : int
        Minimum duration of stage segment in minutes (default: 1)
        For W stage, this is ignored and 10 minutes is used
    """
    import sys
    sys.modules['mne.io.array.array'] = mne.io.array
    
    st.title("Systole/Diastole Comparison: Autoencoder Features")
    
    # Get pickle directory
    pickles_dir = f'pickles/{project_name}'
    
    if not os.path.exists(pickles_dir):
        st.error(f"Directory not found: {pickles_dir}")
        return
    
    # Get all pickle files
    pickle_files = [f for f in os.listdir(pickles_dir) if f.endswith('.pkl')]
    
    if not pickle_files:
        st.warning(f"No pickle files found in {pickles_dir}")
        return
    
    st.info(f"Found {len(pickle_files)} pickle files in {pickles_dir}")
    
    # Group files by patient ID
    patient_files = {}
    for pickle_file in pickle_files:
        if '.edf' in pickle_file:
            patient_id = pickle_file.split('.edf')[0]
        else:
            patient_id = pickle_file.replace('.pkl', '').split('_')[0] if '_' in pickle_file else pickle_file.replace('.pkl', '')
        
        if patient_id not in patient_files:
            patient_files[patient_id] = []
        patient_files[patient_id].append(pickle_file)
    
    # Sort files for each patient
    for patient_id in patient_files:
        patient_files[patient_id].sort()
    
    # Select files to load: if patient has multiple files, use the second one (index 1)
    files_to_load = []
    for patient_id, files in patient_files.items():
        if len(files) > 1:
            files_to_load.append((patient_id, files[1]))
        else:
            files_to_load.append((patient_id, files[0]))
    
    st.info(f"Loading {len(files_to_load)} files (one per patient)")
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Store feature matrices
    all_features = []
    patient_ids_list = []
    
    for idx, (patient_id, pickle_file) in enumerate(files_to_load):
        progress_bar.progress((idx + 1) / len(files_to_load))
        status_text.text(f"Processing {pickle_file} (Patient: {patient_id})...")
        
        try:
            file_path = os.path.join(pickles_dir, pickle_file)
            with open(file_path, 'rb') as f:
                raw = pickle.load(f)
            
            # Extract sleep stage segment if specified
            if sleep_stage is not None:
                raw_stage = extract_sleep_stage_segment(raw, stage=sleep_stage, min_duration_min=min_duration_min)
                if raw_stage is not None:
                    raw = raw_stage
                else:
                    st.warning(f"No {sleep_stage} stage segment found for {patient_id}. Skipping.")
                    continue
            
            sfreq = raw.info['sfreq']
            
            # Extract systole and diastole segments
            segments = extract_systole_diastole_segments(raw, sfreq)
            
            if segments is None:
                st.warning(f"Could not extract systole/diastole segments for {patient_id}. Skipping.")
                continue
            
            # Compute autoencoder features for systole
            systole_features = {}
            
            # EEG features in systole
            if segments['systole']['eeg_segments']:
                eeg_systole_features = compute_autoencoder_features(
                    segments['systole']['eeg_segments'],
                    feature_name_prefix="eeg_systole"
                )
                systole_features.update(eeg_systole_features)
            
            # ECG features in systole
            if segments['systole']['ecg_segments']:
                ecg_systole_features = compute_autoencoder_features(
                    segments['systole']['ecg_segments'],
                    feature_name_prefix="ecg_systole"
                )
                systole_features.update(ecg_systole_features)
            
            # Compute autoencoder features for diastole
            diastole_features = {}
            
            # EEG features in diastole
            if segments['diastole']['eeg_segments']:
                eeg_diastole_features = compute_autoencoder_features(
                    segments['diastole']['eeg_segments'],
                    feature_name_prefix="eeg_diastole"
                )
                diastole_features.update(eeg_diastole_features)
            
            # ECG features in diastole
            if segments['diastole']['ecg_segments']:
                ecg_diastole_features = compute_autoencoder_features(
                    segments['diastole']['ecg_segments'],
                    feature_name_prefix="ecg_diastole"
                )
                diastole_features.update(ecg_diastole_features)
            
            # Combine features into a single row
            patient_feature_dict = {
                'patient_id': patient_id,
                'phase': 'systole'
            }
            patient_feature_dict.update(systole_features)
            
            # Flatten array features
            flattened_features = {}
            for key, value in patient_feature_dict.items():
                if isinstance(value, np.ndarray):
                    if value.ndim == 1:
                        # Create separate columns for each element
                        for i, v in enumerate(value):
                            flattened_features[f'{key}_{i}'] = v
                    else:
                        # Flatten multi-dimensional arrays
                        flat = value.flatten()
                        for i, v in enumerate(flat):
                            flattened_features[f'{key}_{i}'] = v
                else:
                    flattened_features[key] = value
            
            all_features.append(flattened_features)
            patient_ids_list.append(f"{patient_id}_systole")
            
            # Add diastole features
            patient_feature_dict_diastole = {
                'patient_id': patient_id,
                'phase': 'diastole'
            }
            patient_feature_dict_diastole.update(diastole_features)
            
            flattened_features_diastole = {}
            for key, value in patient_feature_dict_diastole.items():
                if isinstance(value, np.ndarray):
                    if value.ndim == 1:
                        for i, v in enumerate(value):
                            flattened_features_diastole[f'{key}_{i}'] = v
                    else:
                        flat = value.flatten()
                        for i, v in enumerate(flat):
                            flattened_features_diastole[f'{key}_{i}'] = v
                else:
                    flattened_features_diastole[key] = value
            
            all_features.append(flattened_features_diastole)
            patient_ids_list.append(f"{patient_id}_diastole")
            
        except Exception as e:
            st.warning(f"Error processing {pickle_file}: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    if not all_features:
        st.error("No features could be extracted from any files")
        return
    
    st.success(f"Successfully processed {len(patient_ids_list) // 2} patients")
    
    # Create DataFrame
    try:
        # Get all unique keys
        all_keys = set()
        for feat_dict in all_features:
            all_keys.update(feat_dict.keys())
        
        # Create DataFrame with all keys
        feature_matrix = []
        for feat_dict in all_features:
            row = {key: feat_dict.get(key, np.nan) for key in all_keys}
            feature_matrix.append(row)
        
        df_features = pd.DataFrame(feature_matrix)
        
        st.subheader("Feature Matrix")
        st.dataframe(df_features, use_container_width=True)
        
        # Display summary statistics
        st.subheader("Summary Statistics")
        st.dataframe(df_features.describe(), use_container_width=True)
        
        # Display feature comparison between systole and diastole
        st.subheader("Systole vs Diastole Comparison")
        
        # Separate by phase
        df_systole = df_features[df_features['phase'] == 'systole']
        df_diastole = df_features[df_features['phase'] == 'diastole']
        
        if not df_systole.empty and not df_diastole.empty:
            # Compare mean values
            numeric_cols = df_features.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'phase']
            
            comparison_data = []
            for col in numeric_cols:
                if col in df_systole.columns and col in df_diastole.columns:
                    systole_mean = df_systole[col].mean()
                    diastole_mean = df_diastole[col].mean()
                    comparison_data.append({
                        'feature': col,
                        'systole_mean': systole_mean,
                        'diastole_mean': diastole_mean,
                        'difference': systole_mean - diastole_mean,
                        'percent_change': ((systole_mean - diastole_mean) / abs(diastole_mean) * 100) if diastole_mean != 0 else np.nan
                    })
            
            if comparison_data:
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating feature matrix: {str(e)}")
        import traceback
        st.error(traceback.format_exc())


def ecg_analysis_run(project_name, sleep_stage=None, min_duration_min=1):
    """
    Read all pickle files from pickles/{project_name} folder and plot 
    ECG/EEG network measurements and their correlations.
    
    Parameters:
    -----------
    project_name : str
        Name of the project (used to locate pickle files)
    sleep_stage : str, optional
        Sleep stage to extract (N1, N2, N3, R, or W). If None, uses full recording (default: None)
    min_duration_min : int
        Minimum duration of stage segment in minutes (default: 1)
        For W stage, this is ignored and 10 minutes is used
    """
    import sys
    sys.modules['mne.io.array.array'] = mne.io.array
    
    st.title("ECG Analysis: Network Features and Correlations")
    
    # Define feature categories
    eeg_network_features = [
        "coherence_matrix",
        # "wPLI_matrix",
        # "PLI_matrix",
        # "imaginary_coherence",
        "amplitude_envelope_corr",
        "phase_locking_value",
        "graph_degree",
        "graph_strength",
        "clustering_coefficient",
        "global_efficiency",
        "characteristic_path_length",
        "modularity",
        "betweenness_centrality",
        "eigenvector_centrality",
        "participation_coefficient",
        "small_worldness",
        "assortativity",
        "network_entropy",
        "dynamic_connectivity_entropy",
        "functional_connectivity_density",
        "pca_components_95",
        "pca_explained_variance_ratio",
        "pca_n_components_95",
        "pca_total_variance_explained",
        "autoencoder_features_95",
        "autoencoder_n_components_95",
        "autoencoder_variance_explained",
        "autoencoder_reconstruction_error"
    ]
    
    hrv_network_features = [
        "time_domain_matrix",
        "frequency_coupling_matrix",
        "transfer_entropy_matrix",
        "granger_causality_values",
        "cross_correlation_matrix",
        "heart_brain_coherence",
        "phase_amplitude_coupling",
        "mutual_information_matrix",
        "graph_efficiency_heart_brain",
        "network_synchrony_index"
    ]
    
    all_features = eeg_network_features + hrv_network_features
    
    # Get pickle directory
    pickles_dir = f'pickles/{project_name}'
    
    if not os.path.exists(pickles_dir):
        st.error(f"Directory not found: {pickles_dir}")
        return
    
    # Get all pickle files
    pickle_files = [f for f in os.listdir(pickles_dir) if f.endswith('.pkl')]
    
    if not pickle_files:
        st.warning(f"No pickle files found in {pickles_dir}")
        return
    
    st.info(f"Found {len(pickle_files)} pickle files in {pickles_dir}")
    
    # Group files by patient ID (extract part before .edf)
    patient_files = {}
    for pickle_file in pickle_files:
        # Extract patient ID - everything before .edf
        if '.edf' in pickle_file:
            patient_id = pickle_file.split('.edf')[0]
        else:
            # Fallback: use part before first underscore
            patient_id = pickle_file.replace('.pkl', '').split('_')[0] if '_' in pickle_file else pickle_file.replace('.pkl', '')
        
        if patient_id not in patient_files:
            patient_files[patient_id] = []
        patient_files[patient_id].append(pickle_file)
    
    # Sort files for each patient (by filename to ensure consistent ordering)
    for patient_id in patient_files:
        patient_files[patient_id].sort()
    
    # Select files to load: if patient has multiple files, use the second one (index 1)
    # If patient has only one file, use that one
    files_to_load = []
    for patient_id, files in patient_files.items():
        if len(files) > 1:
            # Multiple files - use the second one (index 1)
            files_to_load.append((patient_id, files[1]))
        else:
            # Single file - use it
            files_to_load.append((patient_id, files[0]))
    
    st.info(f"Loading {len(files_to_load)} files (one per patient)")
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Load data from pickle files
    all_data = {}
    feature_data = {feature: [] for feature in all_features}
    patient_ids = []
    
    # Track cache statistics
    cache_hits = 0
    cache_misses = 0
    
    for idx, (patient_id, pickle_file) in enumerate(files_to_load):
        progress_bar.progress((idx + 1) / len(files_to_load))
        status_text.text(f"Loading {pickle_file} (Patient: {patient_id})...")
        
        try:
            # Generate cache key
            cache_key = get_cache_key(project_name, patient_id, pickle_file, sleep_stage, min_duration_min)
            cache_path = get_cache_path(project_name, cache_key)
            
            # Try to load from cache
            cached_data = load_from_cache(cache_path)
            
            if cached_data is not None:
                # Load from cache
                cache_hits += 1
                status_text.text(f"Loading from cache: {pickle_file} (Patient: {patient_id})...")
                patient_features = cached_data.get('patient_features')
                feature_data_entries = cached_data.get('feature_data_entries', {})
                raw = cached_data.get('raw')
                
                # Restore feature_data entries
                for feature, entries in feature_data_entries.items():
                    if feature in feature_data:
                        feature_data[feature].extend(entries)
                
                patient_ids.append(patient_id)
                all_data[patient_id] = {
                    'raw': raw,
                    'features': patient_features
                }
                continue
            
            # Cache miss - compute features
            cache_misses += 1
            status_text.text(f"Computing features: {pickle_file} (Patient: {patient_id})...")
            
            file_path = os.path.join(pickles_dir, pickle_file)
            with open(file_path, 'rb') as f:
                raw = pickle.load(f)
            
            # Extract sleep stage segment if specified
            if sleep_stage is not None:
                raw_stage = extract_sleep_stage_segment(raw, stage=sleep_stage, min_duration_min=min_duration_min)
                if raw_stage is not None:
                    raw = raw_stage
                else:
                    st.warning(f"No {sleep_stage} stage segment found for {patient_id}. Skipping.")
                    continue
            
            patient_ids.append(patient_id)
            
            # Compute all network features from raw data
            patient_features = compute_all_network_features(raw, patient_id)
            if patient_features is None:
                st.warning(f"Error computing network features for {patient_id}")
                continue
            
            # Process computed features and extract summary statistics
            feature_data_entries = {}
            for feature in all_features:
                if feature in patient_features:
                    value = patient_features[feature]
                    entry = None
                    # Handle matrix/array features - extract summary statistics
                    if isinstance(value, np.ndarray):
                        if value.ndim == 2:  # Matrix
                            # Extract mean, std, max, min of matrix
                            entry = {
                                'patient_id': patient_id,
                                'mean': np.mean(value),
                                'std': np.std(value),
                                'max': np.max(value),
                                'min': np.min(value),
                                'median': np.median(value)
                            }
                        elif value.ndim == 1:  # 1D array
                            entry = {
                                'patient_id': patient_id,
                                'mean': np.mean(value),
                                'std': np.std(value),
                                'max': np.max(value),
                                'min': np.min(value),
                                'median': np.median(value)
                            }
                        else:
                            # Multi-dimensional array - flatten and compute stats
                            flat_value = value.flatten()
                            entry = {
                                'patient_id': patient_id,
                                'mean': np.mean(flat_value),
                                'std': np.std(flat_value),
                                'max': np.max(flat_value),
                                'min': np.min(flat_value),
                                'median': np.median(flat_value)
                            }
                    elif isinstance(value, (int, float, np.number)):
                        entry = {
                            'patient_id': patient_id,
                            'value': float(value)
                        }
                    
                    if entry is not None:
                        feature_data[feature].append(entry)
                        if feature not in feature_data_entries:
                            feature_data_entries[feature] = []
                        feature_data_entries[feature].append(entry)
            
            all_data[patient_id] = {
                'raw': raw,
                'features': patient_features
            }
            
            # Save to cache
            cache_data = {
                'patient_features': patient_features,
                'feature_data_entries': feature_data_entries,
                'raw': raw
            }
            save_to_cache(cache_path, cache_data)
            
        except Exception as e:
            st.warning(f"Error loading {pickle_file}: {str(e)}")
            continue
    
    # Display cache statistics
    if cache_hits > 0 or cache_misses > 0:
        total = cache_hits + cache_misses
        cache_hit_rate = (cache_hits / total * 100) if total > 0 else 0
        st.info(f"Cache: {cache_hits} hits, {cache_misses} misses ({cache_hit_rate:.1f}% hit rate)")
    
    progress_bar.empty()
    status_text.empty()
    
    if not all_data:
        st.error("No data could be loaded from pickle files")
        return
    
    st.success(f"Successfully loaded data from {len(all_data)} files")
    
    # Create DataFrames for each feature type
    st.subheader("Feature Measurements")
    
    # Process feature data into DataFrames
    feature_dfs = {}
    for feature in all_features:
        if feature_data[feature]:
            # Convert list of dicts to DataFrame
            df = pd.DataFrame(feature_data[feature])
            if not df.empty:
                feature_dfs[feature] = df
    
    if not feature_dfs:
        st.warning("No feature data could be extracted. The pickle files may not contain the expected network features.")
        st.info("Note: This function expects pickle files to contain MNE Raw objects with network features stored in the info dictionary.")
        return
    
    # Display feature selection
    st.sidebar.header("ECG Analysis Options")
    feature_category = st.sidebar.selectbox(
        "Select feature category:",
        ["EEG Network Features", "HRV Network Features", "All Features"]
    )
    
    if feature_category == "EEG Network Features":
        available_features = [f for f in eeg_network_features if f in feature_dfs]
    elif feature_category == "HRV Network Features":
        available_features = [f for f in hrv_network_features if f in feature_dfs]
    else:
        available_features = list(feature_dfs.keys())
    
    if not available_features:
        st.warning(f"No {feature_category} found in the data")
        return
    
    # Plot 1: Summary statistics for selected features
    st.subheader("Feature Summary Statistics")
    
    # Create a combined DataFrame for correlation analysis
    correlation_data = {}
    
    for feature in available_features[:10]:  # Limit to first 10 for performance
        df = feature_dfs[feature]
        # Get numeric columns (excluding patient_id)
        numeric_cols = [col for col in df.columns if col != 'patient_id' and pd.api.types.is_numeric_dtype(df[col])]
        
        if numeric_cols:
            # Use mean if available, otherwise use value
            if 'mean' in numeric_cols:
                correlation_data[feature] = df.set_index('patient_id')['mean']
            elif 'value' in numeric_cols:
                correlation_data[feature] = df.set_index('patient_id')['value']
    
    if correlation_data:
        # Create correlation matrix
        correlation_df = pd.DataFrame(correlation_data)
        correlation_df = correlation_df.dropna()
        
        if not correlation_df.empty and len(correlation_df.columns) > 1:
            # Plot correlation heatmap
            st.subheader("Feature Correlation Matrix")
            fig, ax = plt.subplots(figsize=(12, 10))
            corr_matrix = correlation_df.corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
            ax.set_title("Correlation Matrix of Network Features")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # Display correlation table
            st.dataframe(corr_matrix, use_container_width=True)
        
        # Plot individual feature distributions
        st.subheader("Feature Distributions")
        
        n_features = min(len(available_features), 6)  # Show up to 6 features
        cols = st.columns(2)
        
        for idx, feature in enumerate(available_features[:n_features]):
            df = feature_dfs[feature]
            numeric_cols = [col for col in df.columns if col != 'patient_id' and pd.api.types.is_numeric_dtype(df[col])]
            
            if numeric_cols:
                col_idx = idx % 2
                with cols[col_idx]:
                    # Plot distribution
                    fig, ax = plt.subplots(figsize=(8, 5))
                    
                    # Use mean if available
                    if 'mean' in numeric_cols:
                        data_to_plot = df['mean'].dropna()
                        ax.hist(data_to_plot, bins=20, edgecolor='black', alpha=0.7)
                        ax.set_xlabel('Mean Value')
                        ax.set_ylabel('Frequency')
                        ax.set_title(f'{feature}\n(n={len(data_to_plot)})')
                    elif 'value' in numeric_cols:
                        data_to_plot = df['value'].dropna()
                        ax.hist(data_to_plot, bins=20, edgecolor='black', alpha=0.7)
                        ax.set_xlabel('Value')
                        ax.set_ylabel('Frequency')
                        ax.set_title(f'{feature}\n(n={len(data_to_plot)})')
                    
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
        
        # Pairwise scatter plots for top correlated features
        if len(correlation_df.columns) >= 2:
            st.subheader("Top Correlated Feature Pairs")
            
            # Get top correlations
            corr_pairs = []
            for i, col1 in enumerate(correlation_df.columns):
                for j, col2 in enumerate(correlation_df.columns):
                    if i < j:
                        corr_val = correlation_df[col1].corr(correlation_df[col2])
                        if not np.isnan(corr_val):
                            corr_pairs.append((col1, col2, abs(corr_val)))
            
            corr_pairs.sort(key=lambda x: x[2], reverse=True)
            top_pairs = corr_pairs[:4]  # Show top 4 pairs
            
            if top_pairs:
                cols = st.columns(2)
                for idx, (col1, col2, corr_val) in enumerate(top_pairs):
                    col_idx = idx % 2
                    with cols[col_idx]:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.scatter(correlation_df[col1], correlation_df[col2], alpha=0.6, s=50)
                        ax.set_xlabel(col1)
                        ax.set_ylabel(col2)
                        ax.set_title(f'{col1} vs {col2}\n(r={corr_val:.3f})')
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
    
    # Display raw feature data tables
    st.subheader("Feature Data Tables")
    
    selected_feature_display = st.selectbox(
        "Select a feature to view detailed data:",
        available_features
    )
    
    if selected_feature_display in feature_dfs:
        st.dataframe(feature_dfs[selected_feature_display], use_container_width=True)
        
        # Show statistics
        df = feature_dfs[selected_feature_display]
        numeric_cols = [col for col in df.columns if col != 'patient_id' and pd.api.types.is_numeric_dtype(df[col])]
        
        if numeric_cols:
            st.write("**Summary Statistics:**")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)


def forest_plot_all_features(df_wnv3, selected_feature, target_features, analysis_type="Full"):
    """
    Create a forest plot with 2 columns (one for each group) showing median, std, and p-values
    for each target feature (EEG features) grouped by the selected feature.
    
    Parameters:
    - df_wnv3: DataFrame with the data
    - selected_feature: The feature being analyzed (used for grouping)
    - target_features: List of EEG features to compare
    - analysis_type: "Full" or "Significant" to filter results
    """
    from scipy.stats import mannwhitneyu, ttest_ind
    
    # Prepare data for analysis
    df_clean = df_wnv3[df_wnv3[selected_feature].notna()].copy()
    
    # Determine if binary and get threshold
    n_unique = df_clean[selected_feature].nunique()
    unique_vals = sorted(df_clean[selected_feature].dropna().unique())
    
    threshold = None
    if n_unique == 2:
        # Binary - use 0 and 1 groups
        threshold = 0.5  # For binary comparison
        df_clean['Group'] = df_clean[selected_feature].apply(lambda x: 1 if x == unique_vals[1] else 0)
        group_name_0 = f"{selected_feature} = {unique_vals[0]}"
        group_name_1 = f"{selected_feature} = {unique_vals[1]}"
    else:
        # Ask user for threshold
        median_val = float(df_clean[selected_feature].median())
        threshold = st.number_input(
            f"Enter threshold for {selected_feature} to split into groups:",
            value=median_val,
            key=f"threshold_{selected_feature}_forest"
        )
        df_clean['Group'] = df_clean[selected_feature].apply(lambda x: 1 if x > threshold else 0)
        group_name_0 = f"{selected_feature} <= {threshold}"
        group_name_1 = f"{selected_feature} > {threshold}"
    
    # Calculate statistics for each target feature
    results = []
    for feature in target_features:
        if feature not in df_clean.columns:
            continue
        
        # Get data for both groups
        group0_data = df_clean[df_clean['Group'] == 0][feature].dropna()
        group1_data = df_clean[df_clean['Group'] == 1][feature].dropna()
        
        if len(group0_data) < 2 or len(group1_data) < 2:
            continue
        
        # Calculate median and std for each group
        median0 = group0_data.median()
        std0 = group0_data.std()
        median1 = group1_data.median()
        std1 = group1_data.std()
        
        # Statistical test
        try:
            # Normality test
            _, normal_p0 = stats.normaltest(group0_data)
            _, normal_p1 = stats.normaltest(group1_data)
            
            if normal_p0 < 0.05 or normal_p1 < 0.05:  # Non-parametric
                stat, p_value = mannwhitneyu(group0_data, group1_data, alternative='two-sided')
                test_used = "Mann-Whitney U"
            else:  # Parametric
                stat, p_value = ttest_ind(group0_data, group1_data)
                test_used = "T-test"
        except Exception:
            p_value = np.nan
            test_used = "Failed"
        
        results.append({
            'Feature': feature,
            'Group0_Median': median0,
            'Group0_Std': std0,
            'Group1_Median': median1,
            'Group1_Std': std1,
            'P_value': p_value,
            'Test': test_used,
            'N0': len(group0_data),
            'N1': len(group1_data)
        })
    
    if not results:
        st.warning("No valid comparisons could be made.")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Filter by significance if requested
    if analysis_type == "Significant":
        results_df = results_df[results_df['P_value'] < 0.05]
    
    if results_df.empty:
        st.warning("No significant results found.")
        return
    
    # Sort by p-value (most significant first)
    results_df = results_df.sort_values('P_value', ascending=True)
    
    # Create forest plot with 2 columns
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, max(8, len(results_df) * 0.5)), sharey=True)
    
    y_pos = np.arange(len(results_df))
    
    # Calculate common x-axis range for both plots
    all_medians = np.concatenate([results_df['Group0_Median'].values, results_df['Group1_Median'].values])
    all_stds = np.concatenate([results_df['Group0_Std'].values, results_df['Group1_Std'].values])
    # Calculate range including error bars (median ± std)
    all_min = np.nanmin(all_medians - all_stds)
    all_max = np.nanmax(all_medians + all_stds)
    range_val = all_max - all_min
    x_min = all_min - range_val * 0.1
    x_max = all_max + range_val * 0.1
    # Ensure range is valid
    if x_min == x_max or np.isnan(x_min) or np.isnan(x_max):
        x_min, x_max = None, None  # Let matplotlib auto-scale
    
    # Column 0: Group 0
    medians0 = results_df['Group0_Median'].values
    stds0 = results_df['Group0_Std'].values
    
    # Plot median points
    ax0.scatter(medians0, y_pos, s=100, color='blue', alpha=0.7, label='Median', zorder=3)
    
    # Plot error bars (std)
    for i, (median, std) in enumerate(zip(medians0, stds0)):
        ax0.plot([median - std, median + std], [i, i], 'b-', linewidth=2, alpha=0.7)
        ax0.plot([median - std, median - std], [i-0.1, i+0.1], 'b-', linewidth=2, alpha=0.7)
        ax0.plot([median + std, median + std], [i-0.1, i+0.1], 'b-', linewidth=2, alpha=0.7)
        # Add median value text
        ax0.text(median, i, f' {median:.2f}', va='center', fontsize=9)
        # Add std text
        ax0.text(median + std + (ax0.get_xlim()[1] - ax0.get_xlim()[0]) * 0.05, i, f'σ={std:.2f}', va='center', fontsize=8, alpha=0.7)
    
    ax0.set_yticks(y_pos)
    ax0.set_yticklabels(results_df['Feature'], fontsize=10)
    ax0.set_xlabel(f'{group_name_0}\nMedian ± Std', fontsize=12)
    ax0.set_title(f'Group 0: {group_name_0}\n(N={results_df["N0"].sum()})', fontsize=12, fontweight='bold')
    if x_min is not None and x_max is not None:
        ax0.set_xlim(x_min, x_max)
    ax0.grid(True, alpha=0.3, axis='x')
    ax0.invert_yaxis()  # Top to bottom
    
    # Column 1: Group 1
    medians1 = results_df['Group1_Median'].values
    stds1 = results_df['Group1_Std'].values
    
    # Plot median points
    ax1.scatter(medians1, y_pos, s=100, color='red', alpha=0.7, label='Median', zorder=3)
    
    # Plot error bars (std)
    for i, (median, std) in enumerate(zip(medians1, stds1)):
        ax1.plot([median - std, median + std], [i, i], 'r-', linewidth=2, alpha=0.7)
        ax1.plot([median - std, median - std], [i-0.1, i+0.1], 'r-', linewidth=2, alpha=0.7)
        ax1.plot([median + std, median + std], [i-0.1, i+0.1], 'r-', linewidth=2, alpha=0.7)
        # Add median value text
        ax1.text(median, i, f' {median:.2f}', va='center', fontsize=9)
        # Add std text
        ax1.text(median + std + (ax1.get_xlim()[1] - ax1.get_xlim()[0]) * 0.05, i, f'σ={std:.2f}', va='center', fontsize=8, alpha=0.7)
        # Add p-value text
        p_val = results_df.iloc[i]['P_value']
        p_color = 'red' if p_val < 0.05 else 'gray'
        p_text = f'p={p_val:.3f}' if not np.isnan(p_val) else 'p=NaN'
        ax1.text(ax1.get_xlim()[1] * 0.98, i, p_text, va='center', ha='right', fontsize=9, 
                color=p_color, fontweight='bold' if p_val < 0.05 else 'normal')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(results_df['Feature'], fontsize=10)
    ax1.set_xlabel(f'{group_name_1}\nMedian ± Std', fontsize=12)
    ax1.set_title(f'Group 1: {group_name_1}\n(N={results_df["N1"].sum()})', fontsize=12, fontweight='bold')
    if x_min is not None and x_max is not None:
        ax1.set_xlim(x_min, x_max)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()  # Top to bottom
    
    # Overall title
    fig.suptitle(f'Forest Plot: {selected_feature} vs All EEG Features\n'
                 f'Median ± Std (Group 0 vs Group 1)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Display in Streamlit
    st.pyplot(fig)
    plt.close(fig)
    
    # Display summary statistics table
    st.subheader("Summary Statistics")
    display_df = results_df[['Feature', 'Group0_Median', 'Group0_Std', 'Group1_Median', 'Group1_Std', 'P_value', 'Test', 'N0', 'N1']].copy()
    display_df['Group0_Median'] = display_df['Group0_Median'].round(3)
    display_df['Group0_Std'] = display_df['Group0_Std'].round(3)
    display_df['Group1_Median'] = display_df['Group1_Median'].round(3)
    display_df['Group1_Std'] = display_df['Group1_Std'].round(3)
    display_df['P_value'] = display_df['P_value'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
    display_df.columns = ['Feature', f'{group_name_0} Median', f'{group_name_0} Std', 
                          f'{group_name_1} Median', f'{group_name_1} Std', 'P-value', 'Test', 'N0', 'N1']
    st.dataframe(display_df, use_container_width=True)

def longitudinal_analysis(project_name):
    """
    Longitudinal analysis function that reads clinical data with date columns
    and raw parquet files to create time-series plots showing changes over time.
    """
    st.header("Longitudinal Analysis")
    
    # Load clinical data with dates
    clinical_df = get_clinical_data(project_name)
    
    if clinical_df.empty:
        raise ValueError(f"No clinical data found for project {project_name}")
    
    # Load raw parquet files
    parquet_df = load_raw_parquet_files(project_name)
    
    if parquet_df.empty:
        raise ValueError(f"No parquet files found for project {project_name}")
    
    # Display available date columns
    date_columns = find_date_columns(clinical_df)
    if not date_columns:
        raise ValueError("No date columns found in clinical data")
    
    # Let user select date column
    date_col = st.sidebar.selectbox("Select date column:", date_columns)
    
    # Get numeric columns for y-axis from both clinical and parquet data
    numeric_columns = get_numeric_columns(clinical_df, parquet_df)
    if not numeric_columns:
        raise ValueError("No numeric columns found for plotting")
    
    # Let user select y-axis column
    y_col = st.sidebar.selectbox("Select numeric column for y-axis:", numeric_columns)
    
    # Patient selection
    if 'ID' in clinical_df.columns:
        available_patients = clinical_df['ID'].unique()
        selected_patients = st.sidebar.multiselect(
            "Select patients (leave empty for all):", 
            available_patients
        )
        if selected_patients:
            clinical_df = clinical_df[clinical_df['ID'].isin(selected_patients)]
            parquet_df = parquet_df[parquet_df['ID'].isin(selected_patients)]
    
    # Create the longitudinal plot
    create_longitudinal_plot(clinical_df, parquet_df, date_col, y_col, project_name)

def load_clinical_data_with_dates(project_name):
    """
    Load clinical data from EDF_Format folder for the given project.
    """
    import os
    import pandas as pd
    from pathlib import Path
    
    clinical_df = pd.DataFrame()
    edf_proj_dir = os.path.join('EDF_Format', project_name)
    
    if os.path.isdir(edf_proj_dir):
        # Look for Excel/CSV files in the directory and subdirs
        data_files = []
        for root, dirs, files in os.walk(edf_proj_dir):
            for f in files:
                if f.lower().endswith(('.xls', '.xlsx', '.csv')):
                    data_files.append(os.path.join(root, f))
        
        if data_files:
            # Try to read the first available file
            for file_path in data_files:
                try:
                    if file_path.lower().endswith('.csv'):
                        clinical_df = pd.read_csv(file_path)
                    else:
                        # For Excel files, try different sheets
                        if project_name == 'Seeg':
                            clinical_df = pd.read_excel(file_path, sheet_name='SEEG_PATIENTS')
                        else:
                            clinical_df = pd.read_excel(file_path)
                    
                    if not clinical_df.empty:
                        break
                except Exception as e:
                    st.warning(f"Could not read {file_path}: {e}")
                    continue
    
    return clinical_df

def load_raw_parquet_files(project_name):
    """
    Load raw parquet files from parquet_results/{project_name} directory.
    """
    import os
    import pandas as pd
    import glob
    from pathlib import Path
    
    parquet_dir = os.path.join('parquet_results', project_name)
    df_list = []
    
    if os.path.isdir(parquet_dir):
        for fname in sorted(os.listdir(parquet_dir)):
            if not fname.lower().endswith('.parquet'):
                continue
            fpath = os.path.join(parquet_dir, fname)
            try:
                df = pd.read_parquet(fpath)
                # Ensure there's a file_name column - use the parquet filename as fallback
                if 'file_name' not in df.columns:
                    base = fname.replace('.parquet', '')
                    # remove trailing .edf if present
                    if base.lower().endswith('.edf'):
                        base = base[:-4]
                    df['file_name'] = base
                
                # Create ID column from file_name
                df['ID'] = df['file_name'].astype(str).apply(lambda x: os.path.basename(x).replace('.edf', '').lower().strip())
                
                df_list.append(df)
            except Exception as e:
                st.warning(f"Warning: failed reading {fpath}: {e}")
                continue
    
    if len(df_list) == 0:
        return pd.DataFrame()
    else:
        return pd.concat(df_list, ignore_index=True, sort=False)

def find_date_columns(df):
    """
    Find columns that contain date information.
    """
    date_columns = []
    
    for col in df.columns:
        # Check if column name suggests it's a date
        if any(keyword in col.lower() for keyword in ['date', 'time', 'visit', 'onset', 'final', 'eeg']):
            date_columns.append(col)
        # Check if column contains date-like data
        elif df[col].dtype == 'object':
            # Sample a few non-null values to check if they look like dates
            sample_values = df[col].dropna().head(5)
            if len(sample_values) > 0:
                try:
                    pd.to_datetime(sample_values.iloc[0])
                    date_columns.append(col)
                except:
                    pass
    
    return date_columns

def get_numeric_columns(clinical_df, parquet_df):
    """
    Get numeric columns from both clinical and parquet data.
    """
    numeric_cols = []
    
    # From clinical data
    for col in clinical_df.columns:
        if pd.api.types.is_numeric_dtype(clinical_df[col]):
            numeric_cols.append(f"clinical_{col}")
    
    # From parquet data
    for col in parquet_df.columns:
        if pd.api.types.is_numeric_dtype(parquet_df[col]) and col not in ['ID', 'file_name']:
            numeric_cols.append(f"parquet_{col}")
    
    return sorted(numeric_cols)

def create_average_trajectory_plot(df_plot_normalized, x_col, actual_y_col, project_name):
    """
    Create a plot showing the average trajectory across all patients with statistical analysis.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    import statsmodels.api as sm
    
    # Calculate average trajectory
    avg_trajectory = df_plot_normalized.groupby(x_col)[actual_y_col].agg(['mean', 'std', 'count']).reset_index()
    avg_trajectory = avg_trajectory[avg_trajectory['count'] > 0]  # Remove time points with no data
    
    if len(avg_trajectory) < 2:
        st.warning("Not enough data points for average trajectory analysis")
        return
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot average trajectory with error bars
    ax.errorbar(avg_trajectory[x_col], avg_trajectory['mean'], 
                yerr=avg_trajectory['std'], 
                marker='o', capsize=5, capthick=2, 
                label='Average ± 1 SD', alpha=0.8, linewidth=2)
    
    # Add individual patient trajectories in background (lighter)
    for patient_id in df_plot_normalized['ID'].unique():
        patient_data = df_plot_normalized[df_plot_normalized['ID'] == patient_id].sort_values(x_col)
        if len(patient_data) > 1:
            ax.plot(patient_data[x_col], patient_data[actual_y_col], 
                   alpha=0.2, linewidth=1, color='gray')
    
    # Statistical analysis: test if slope is significantly different from zero
    X = sm.add_constant(avg_trajectory[x_col])
    model = sm.OLS(avg_trajectory['mean'], X).fit()
    r_squared = model.rsquared
    p_value = model.pvalues[1]  # p-value for the slope
    slope = model.params[1]
    
    # Add regression line
    x_range = np.linspace(avg_trajectory[x_col].min(), avg_trajectory[x_col].max(), 100)
    y_pred = model.predict(sm.add_constant(x_range))
    ax.plot(x_range, y_pred, 'r--', linewidth=2, alpha=0.8, label=f'Regression Line (slope={slope:.2e})')
    
    # Customize plot
    ax.set_xlabel('Days from First Visit', fontsize=12)
    ax.set_ylabel(f'{actual_y_col}', fontsize=12)
    ax.set_title(f'Average Trajectory: {actual_y_col} over Time\n'
                f'R² = {r_squared:.3f}, p-value = {p_value:.2e}', fontsize=14)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc='best')
    
    # Add statistical significance annotation
    if p_value < 0.001:
        sig_text = "p < 0.001"
    elif p_value < 0.01:
        sig_text = "p < 0.01"
    elif p_value < 0.05:
        sig_text = "p < 0.05"
    else:
        sig_text = "p ≥ 0.05 (not significant)"
    
    plt.tight_layout()
    
    # Display the plot
    st.subheader("Average Trajectory Analysis")
    st_pyplot_func(fig, filename=f'average_trajectory_{project_name}_{actual_y_col}.png')
    
    # Display statistical summary
    st.subheader("Statistical Analysis")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R²", f"{r_squared:.3f}")
    
    with col2:
        st.metric("Slope", f"{slope:.3f}")
    
    with col3:
        st.metric("P-value", f"{p_value:.3e}")
    
    with col4:
        significance = "Significant" if p_value < 0.05 else "Not Significant"
        st.metric("Slope Significance", significance)
    
    # Interpretation
    if p_value < 0.05:
        direction = "increasing" if slope > 0 else "decreasing"
        st.success(f"The average trajectory shows a statistically significant {direction} trend (p < 0.05)")
    else:
        st.info("The average trajectory does not show a statistically significant trend (p ≥ 0.05)")
    
    plt.close(fig)

def create_longitudinal_plot(clinical_df, parquet_df, date_col, y_col='overall_pswe_median_percentage', project_name='Seeg'):
    """
    Create a longitudinal plot with date on x-axis and selected column on y-axis.
    """
    import matplotlib.pyplot as plt
    
    # Determine if we're using clinical or parquet data for the y-axis
    if y_col.startswith('clinical_'):
        # Use clinical data
        df_plot = clinical_df.copy()
        actual_y_col = y_col.replace('clinical_', '')
        
        # Parse dates from clinical data
        try:
            df_plot[date_col] = pd.to_datetime(df_plot[date_col], errors='coerce')
        except Exception as e:
            raise ValueError(f"Error parsing dates: {e}")
        
        # Remove rows with invalid dates
        df_plot = df_plot.dropna(subset=[date_col])
        
    elif y_col.startswith('parquet_'):
        # Use parquet data - need to merge with clinical data for dates
        actual_y_col = y_col.replace('parquet_', '')
        
        # Check if y column exists in parquet data
        if actual_y_col not in parquet_df.columns:
            raise ValueError(f"Column {actual_y_col} not found in parquet data")
        
        # Merge parquet data with clinical data to get dates
        if 'ID' in clinical_df.columns and 'ID' in parquet_df.columns:
            df_plot = parquet_df.merge(clinical_df[['ID', date_col]], on='ID', how='left')
        else:
            raise ValueError("Cannot merge parquet and clinical data - missing ID columns")
        
        # Parse dates
        try:
            df_plot[date_col] = pd.to_datetime(df_plot[date_col], errors='coerce')
        except Exception as e:
            raise ValueError(f"Error parsing dates: {e}")
        
        # Remove rows with invalid dates
        df_plot = df_plot.dropna(subset=[date_col])
    
    else:
        raise ValueError("Invalid column selection")
    
    if df_plot.empty:
        raise ValueError("No valid data found after merging and filtering")
    
    # Convert y column to numeric
    try:
        df_plot[actual_y_col] = pd.to_numeric(df_plot[actual_y_col], errors='coerce')
    except Exception as e:
        raise ValueError(f"Error converting {actual_y_col} to numeric: {e}")
    
    # Remove rows with invalid y values
    df_plot = df_plot.dropna(subset=[actual_y_col])
    
    if df_plot.empty:
        raise ValueError("No valid numeric data found")
    
    # Normalize dates to start from day 0 for each patient
    df_plot_normalized = df_plot.copy()
    if 'ID' in df_plot.columns:
        # Calculate days from first visit for each patient
        df_plot_normalized['days_from_start'] = df_plot_normalized.groupby('ID')[date_col].transform(
            lambda x: (x - x.min()).dt.days
        )
        x_col = 'days_from_start'
        x_label = 'Days from First Visit'
    else:
        # If no ID column, normalize all dates to start from 0
        min_date = df_plot[date_col].min()
        df_plot_normalized['days_from_start'] = (df_plot[date_col] - min_date).dt.days
        x_col = 'days_from_start'
        x_label = 'Days from First Visit'
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # If there's an ID column, plot lines for each patient
    if 'ID' in df_plot_normalized.columns:
        for patient_id in df_plot_normalized['ID'].unique():
            patient_data = df_plot_normalized[df_plot_normalized['ID'] == patient_id].sort_values(x_col)
            if len(patient_data) > 1:
                ax.plot(patient_data[x_col], patient_data[actual_y_col], 
                       marker='o', label=f'Patient {patient_id}', alpha=0.7)
            else:
                ax.scatter(patient_data[x_col], patient_data[actual_y_col], 
                          label=f'Patient {patient_id}', alpha=0.7)
    else:
        # Plot all points
        ax.scatter(df_plot_normalized[x_col], df_plot_normalized[actual_y_col], alpha=0.7)
    
    # Customize plot
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(f'{actual_y_col}', fontsize=12)
    ax.set_title(f'Longitudinal Analysis: {actual_y_col} over Time (Normalized)', fontsize=14)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add legend if there are multiple patients
    if 'ID' in df_plot.columns and df_plot['ID'].nunique() > 1:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Display the plot
    st_pyplot_func(fig, filename=f'longitudinal_{project_name}_{actual_y_col}.png')
    
    # Create average trajectory plot
    if 'ID' in df_plot_normalized.columns and df_plot_normalized['ID'].nunique() > 1:
        create_average_trajectory_plot(df_plot_normalized, x_col, actual_y_col, project_name)
    
    # Display summary statistics
    st.subheader("Summary Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(df_plot))
    
    with col2:
        if 'ID' in df_plot.columns:
            st.metric("Unique Patients", df_plot['ID'].nunique())
        else:
            st.metric("Data Points", len(df_plot))
    
    with col3:
        if 'days_from_start' in df_plot_normalized.columns:
            max_days = df_plot_normalized['days_from_start'].max()
            st.metric("Max Days from Start", max_days)
        else:
            date_range = (df_plot[date_col].max() - df_plot[date_col].min()).days
            st.metric("Date Range (days)", date_range)
    
    # Show data table
    st.subheader("Data Preview")
    display_cols = [x_col, actual_y_col]
    if 'ID' in df_plot_normalized.columns:
        display_cols = ['ID'] + display_cols
    
    st.dataframe(df_plot_normalized[display_cols].head(20), use_container_width=True)
    
    plt.close(fig)

def nilearn_plotting(df_wnv3):
    # leave all columns that say 'EEG' and group column
    df_eeg_group = df_wnv3.filter(like='EEG')
    df_eeg_group = df_eeg_group.assign(Group=df_wnv3['Group'])
    # remove columns any NaN
    df_eeg_group = df_eeg_group.dropna(axis=1)
    # save eeg_group to csv
    df_eeg_group.to_csv("eeg_group.csv", index=False)
    # Now you can use df_eeg_group for your Nilearn plotting
    # Example: plot brain maps, connectivity matrices, etc.
    pass

if __name__ == "__main__":
    main()
