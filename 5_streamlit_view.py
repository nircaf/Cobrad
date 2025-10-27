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
def vs_controls_run(project_name,df_wnv2,controls,boxplot_columns,analysis_type):
    df_wnv2['Group'] = project_name
    controls['Group'] = 'Controls'
    combined_df = pd.concat([df_wnv2, controls], ignore_index=True,axis=0)
    controls_dir = f'temps_Controls_EDF'
    try:
        st.header('Controls Demographics')
        # get controls demographic from get_controls_ages_genders(controls_dir)
        controls_demographics_df = get_controls_ages_genders(controls_dir)
        cobrad_ages = df_wnv2['clinical_age_at_visit']
        cobrad_sexes = df_wnv2['clinical_sex, 1=male']
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
        boxplot_plot_dabest(results_df,curr_data, col, 'vs_controls',is_streamlit=True,analysis_type=analysis_type)

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


def HEP_plots(project_name, df_wnv3, controls, boxplot_columns, analysis_type, selected_feature=None, size_feature=None):
    # if size_feature is None:
    #     size_feature = 'clinical_moca'
    if project_name == 'COBRAD':
        # Define both directories
        edf_hep_dir = 'parquets_HEP/EDF_N1' #'parquets_HEP/EDF_N1'
    else:
        return
    # Run over power bands
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
        pair_clustering_analysis(results_df, df_wnv3_clean, n_clusters=n_clusters)
        
        st.divider()


def pair_clustering_analysis(results_df, df_wnv3_clean, n_clusters=2):
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
    selected_classifier = st.selectbox('Select classifier', list(classifier_options.keys()), key='pair_clustering_classifier')
    
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
    
    # Limit to first 2 pairs unless checkbox is checked
    if not run_all_pairs:
        pairs = all_pairs[:2]
        st.info(f"Running first 2 pairs only. Check 'Run all pairs' to analyze all {total_pairs} pairs.")
    else:
        pairs = all_pairs
    
    # Create columns for displaying results
    col1, col2 = st.columns(2)
    
    results_summary = []

    for i, (col1_name, col2_name) in enumerate(pairs):
        with st.expander(f"Pair {i+1}: {col1_name} vs {col2_name}", expanded=False):
            
            # Prepare data for this pair
            pair_data = results_df[[col1_name, col2_name, 'Group']].dropna()
            
            if len(pair_data) < n_clusters:
                st.warning(f"Not enough data points for pair {col1_name} vs {col2_name}")
                continue
            
            # Perform K-means clustering
            X_pair = pair_data[[col1_name, col2_name]]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_pair)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
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
                
                # Store results for summary
                results_summary.append({
                    'pair': f"{col1_name} vs {col2_name}",
                    'n_samples': len(X_clinical),
                    'accuracy': clf.score(X_test, y_test),
                    'top_clinical_feature': importance_df.iloc[0]['feature'],
                    'top_importance': importance_df.iloc[0]['importance']
                })
                
            except Exception as e:
                st.error(f"Error in feature importance analysis: {str(e)}")
    
    # Create comprehensive pairgrid for all features
    st.subheader("Comprehensive Feature Pairgrid")
    
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
    
    if len(pairgrid_data) > 0:
        # Create pairgrid
        g = sns.PairGrid(pairgrid_data, hue='Group', diag_sharey=False)
        
        def plot_upper(x, y, **kwargs):
            """Plot upper triangle with group-colored scatter plots."""
            ax = kwargs.get('ax', plt.gca())
            
            # Get unique groups and colors - use vibrant palette
            unique_groups = pairgrid_data['Group'].unique()
            color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
            group_colors = {group: color_palette[i % len(color_palette)] for i, group in enumerate(unique_groups)}
            
            # Create scatter plot with group colors
            for group in unique_groups:
                mask = pairgrid_data['Group'] == group
                if mask.sum() > 0:
                    ax.scatter(x[mask], y[mask], 
                              c=[group_colors[group]], 
                              label=group, 
                              alpha=0.7, s=50)
            
            # Add regression line
            try:
                from scipy.stats import linregress
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                line_x = np.linspace(x.min(), x.max(), 100)
                line_y = slope * line_x + intercept
                ax.plot(line_x, line_y, 'r--', alpha=0.8, linewidth=2)
                ax.text(0.05, 0.95, f'R² = {r_value**2:.3f}', 
                       transform=ax.transAxes, fontsize=8, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except:
                pass
        
        def plot_lower(x, y, **kwargs):
            """Plot lower triangle with correlation heatmap."""
            ax = kwargs.get('ax', plt.gca())
            
            # Calculate correlation
            try:
                from scipy.stats import pearsonr
                corr, p_val = pearsonr(x, y)
                
                # Create scatter plot
                ax.scatter(x, y, alpha=0.6, s=30)
                
                # Add correlation text
                ax.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
                       transform=ax.transAxes, fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except:
                ax.scatter(x, y, alpha=0.6, s=30)
        
        def plot_diag(x, **kwargs):
            """Plot diagonal with histograms for each group."""
            ax = kwargs.get('ax', plt.gca())
            
            # Get unique groups and colors
            unique_groups = pairgrid_data['Group'].unique()
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))
            group_colors = {group: colors[i] for i, group in enumerate(unique_groups)}
            
            for group in unique_groups:
                mask = pairgrid_data['Group'] == group
                if mask.sum() > 0:
                    ax.hist(x[mask], alpha=0.6, color=group_colors[group], 
                           label=group, bins=15, density=True)
            
            ax.legend(fontsize=8)
        
        # Apply the plotting functions
        g.map_upper(plot_upper)
        g.map_lower(plot_lower)
        g.map_diag(plot_diag)
        
        # Set title
        g.fig.suptitle(f'Comprehensive Feature Analysis - All Pairs\nN={len(pairgrid_data)}', 
                       fontsize=16, y=1.02)
        
        plt.tight_layout()
        st.pyplot(g.fig)
        plt.close(g.fig)
    
    # Create clustering results pairgrid with classifier performance groups
    st.subheader("Clustering Results Pairgrid - Classifier Performance Groups")
    
    if len(pairgrid_data) > 0:
        # Perform clustering on all features for the pairgrid using the ordered labels
        X_all = pairgrid_data[available_labels]
        scaler_all = StandardScaler()
        X_scaled_all = scaler_all.fit_transform(X_all)
        
        kmeans_all = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels_all = kmeans_all.fit_predict(X_scaled_all)
        
        # Add cluster labels to data
        pairgrid_data_with_clusters = pairgrid_data.copy()
        pairgrid_data_with_clusters['cluster'] = cluster_labels_all
        
        # Get classifier predictions for the data that has clinical features
        if len(X_clinical) > 0:
            classifier_predictions = classifier.predict(X_clinical)
            
            # Create performance groups based on classifier vs K-means clusters
            performance_groups = []
            for i, (true_cluster, pred_cluster) in enumerate(zip(cluster_labels_all, classifier_predictions)):
                if true_cluster == 1 and pred_cluster == 1:
                    performance_groups.append('TP')
                elif true_cluster == 0 and pred_cluster == 0:
                    performance_groups.append('TN')
                elif true_cluster == 0 and pred_cluster == 1:
                    performance_groups.append('FP')
                else:  # true_cluster == 1 and pred_cluster == 0
                    performance_groups.append('FN')
            pairgrid_data_with_clusters['classifier_pred'] = classifier_predictions
            pairgrid_data_with_clusters['performance_group'] =  performance_groups
        # remove column cluster and classifier_pred
        pairgrid_data_with_clusters = pairgrid_data_with_clusters.drop(columns=['cluster', 'classifier_pred'])
        # Create pairgrid with performance groups
        g_cluster = sns.PairGrid(pairgrid_data_with_clusters, hue='performance_group', diag_sharey=False)
        
        def plot_upper_cluster(x, y, **kwargs):
            """Plot upper triangle with performance group-colored scatter plots."""
            ax = kwargs.get('ax', plt.gca())
            
            # Get unique performance groups and colors
            unique_groups = pairgrid_data_with_clusters['performance_group'].unique()
            # Use specific colors for TP, TN, FP, FN
            performance_colors = {
                'TP': '#2E8B57',    # Green
                'TN': '#4169E1',    # Blue  
                'FP': '#FF6347',    # Red
                'FN': '#FFD700'     # Gold
            }
            
            # Create scatter plot with performance group colors
            for group in unique_groups:
                mask = pairgrid_data_with_clusters['performance_group'] == group
                if mask.sum() > 0:
                    color = performance_colors.get(group, '#808080')  # Default gray
                    ax.scatter(x[mask], y[mask], 
                              c=[color], 
                              alpha=0.7, s=50)
            
            # Add cluster centers if this is a 2D plot
            if len(np.unique(x)) > 1 and len(np.unique(y)) > 1:
                try:
                    # Find the centers for this specific pair
                    col1_name = x.name
                    col2_name = y.name
                    if col1_name in available_labels and col2_name in available_labels:
                        # Get the centers for this specific pair
                        pair_idx = [available_labels.index(col1_name), available_labels.index(col2_name)]
                        centers_2d = kmeans_all.cluster_centers_[:, pair_idx]
                        ax.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                                 c='red', marker='x', s=200, linewidths=3, 
                                 label='Centroids')
                except:
                    pass
        
        def plot_lower_cluster(x, y, **kwargs):
            """Plot lower triangle with performance group-colored scatter plots."""
            ax = kwargs.get('ax', plt.gca())
            
            # Get unique performance groups and colors
            unique_groups = pairgrid_data_with_clusters['performance_group'].unique()
            performance_colors = {
                'TP': '#2E8B57',    # Green
                'TN': '#4169E1',    # Blue  
                'FP': '#FF6347',    # Red
                'FN': '#FFD700'     # Gold
            }
            
            # Create scatter plot with performance group colors
            for group in unique_groups:
                mask = pairgrid_data_with_clusters['performance_group'] == group
                if mask.sum() > 0:
                    color = performance_colors.get(group, '#808080')  # Default gray
                    ax.scatter(x[mask], y[mask], 
                              c=[color], 
                              alpha=0.6, s=30)
        
        def plot_diag_cluster(x, **kwargs):
            """Plot diagonal with histograms for each performance group."""
            ax = kwargs.get('ax', plt.gca())
            
            # Get unique performance groups and colors
            unique_groups = pairgrid_data_with_clusters['performance_group'].unique()
            performance_colors = {
                'TP': '#2E8B57',    # Green
                'TN': '#4169E1',    # Blue  
                'FP': '#FF6347',    # Red
                'FN': '#FFD700'     # Gold
            }
            
            for group in unique_groups:
                mask = pairgrid_data_with_clusters['performance_group'] == group
                if mask.sum() > 0:
                    color = performance_colors.get(group, '#808080')  # Default gray
                    ax.hist(x[mask], alpha=0.6, color=color, 
                           bins=15, density=True)
        
        # Apply the plotting functions
        g_cluster.map_upper(plot_upper_cluster)
        g_cluster.map_lower(plot_lower_cluster)
        g_cluster.map_diag(plot_diag_cluster)
        
        # Set title
        g_cluster.fig.suptitle(f'Classifier Performance Groups - All Features\nN={len(pairgrid_data)}, Performance Groups: TP, TN, FP, FN', 
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
        
        # Show performance group summary
        st.subheader("Performance Group Summary")
        performance_summary = pairgrid_data_with_clusters['performance_group'].value_counts()
        st.dataframe(performance_summary.to_frame('Count'))
    
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


# Streamlit App
def main():
    # options to choose from folders in pickles
    default_options = ["COBRAD", "WNV"] +  [f for f in os.listdir('parquet_results') if os.path.isdir(os.path.join('parquet_results', f))] 
    # allow multiple selection for project
    # default_options = ["COBRAD", "WNV"]
    # Deduplicate while preserving order
    seen = set()
    opts = [x for x in default_options if not (x in seen or seen.add(x))]
    selected_projects = st.sidebar.pills("Select Project(s)", opts, default=[x for x in ["COBRAD"] if x in opts],selection_mode ="multi")
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
    feature_types = ( 'HEP', 'All', 'Clinical Feature', "EEG Feature", "ml_plots", "vs_Controls", "Pair Plot",'Raw','Spectrogram') # 'Longitudinal',
    feature_type = st.sidebar.selectbox("Select feature type to plot against the other type:", feature_types)
    
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
            vs_controls_run(project_name,df_wnv2,controls,boxplot_columns,analysis_type)
        elif current_feature_type == "HEP":
            HEP_plots(project_name,df_wnv2,controls,boxplot_columns,analysis_type,size_feature=size_feature)
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
            st.title("Spectrogram")
            # ask user for win_sec
            win_sec = st.sidebar.slider("Select window size in seconds", 1, 30, 5)
            st.subheader(f"{current_feature_type} {cases_group_name}")
            spectrogram_run(cases_group_name,win_sec=win_sec)
            st.divider()
            st.subheader("Spectrogram Controls")
            spectrogram_run(f'Controls',win_sec=win_sec)
        elif current_feature_type == "Raw":
            st.title("Raw Data")
            raw_run(cases_group_name)
        elif current_feature_type == "Longitudinal":
            longitudinal_analysis(project_name)
        elif current_feature_type == "EEG Feature":
            forest_plot_eeg = st.sidebar.checkbox("Show Forest Plot vs All Clinical Features", value=False)
            eeg_feature_options =  eeg_features+ ["All Features"] 
            selected_feature = st.sidebar.selectbox("Select an EEG feature:", eeg_feature_options)
            if selected_feature == "All Features" or feature_type == "All":
                all_feat_list = eeg_features
                selected_feature = eeg_features[0]
            plot_title = f"Plots of {selected_feature} vs All Clinical Features"
            boxplot_columns = clinical_features_numeric
            
            # Add forest plot if requested
            if forest_plot_eeg:
                st.subheader("Forest Plot Analysis")
                forest_plot_all_features(df_wnv2, selected_feature, clinical_features_numeric, analysis_type)
        elif current_feature_type == "Clinical Feature":
            clinical_features_correlation = st.sidebar.checkbox("Show Clinical Features Correlation", value=False)
            forest_plot_clinical = st.sidebar.checkbox("Show Forest Plot vs All EEG Features", value=False)
            marked_clinical_features_w_all = clinical_features + ["All Features"]
            selected_feature = st.sidebar.selectbox("Select a Clinical feature:", marked_clinical_features_w_all)
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
                    else:
                        # group values based on band if =1, else f'not {band}'
                        org_selected_feature_for_plot = org_selected_feature(selected_feature)
                        df_wnv3['Group'] = df_wnv3[selected_feature].apply(lambda x: f'{org_selected_feature_for_plot}+' if x == 1 else f'{org_selected_feature_for_plot}-')
                    # run over df_wnv2_others
                    for idx, (project_name_other, df_wnv2_other) in enumerate(df_wnv2_others):
                        if idx ==0:
                            # concat with controls, df_wnv3
                            df_wnv3 = pd.concat([controls, df_wnv3], ignore_index=True)
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
                                    # st checkbox if HEP
                    if hep_checkbox is None:
                        hep_checkbox = st.checkbox("Show HEP & TSNE Plots", value=False, key=f"hep_checkbox_{selected_feature}")
                    if hep_checkbox:
                        st.subheader("HEP Plots")
                        HEP_plots(project_name, df_wnv3, controls, boxplot_columns, analysis_type, selected_feature, size_feature)
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
                
                for feature in all_feat_list:
                    clinical_features, boxplot_columns = get_clinical_and_boxplot_cols(df_wnv2=df_wnv2)
                    selected_feature = feature
                    st.write(f"## Analyzing Feature: {selected_feature}")
                    run_selected_feature(boxplot_columns, hep_checkbox)
                    # Add forest plot if requested
                    if forest_plot_clinical:
                        st.subheader("Forest Plot Analysis")
                        forest_plot_all_features(df_wnv2, selected_feature, eeg_features, analysis_type)
                    st.divider()
            else:
                run_selected_feature(boxplot_columns)
                # Add forest plot if requested
                if forest_plot_clinical:
                    st.subheader("Forest Plot Analysis")
                    forest_plot_all_features(df_wnv2, selected_feature, eeg_features, analysis_type)
        
        # Add divider between feature types when running "All"
        if feature_type == "All" and current_feature_type != feature_types_to_run[-1]:
            st.divider()
    download_pptx_button()


def forest_plot_all_features(df_wnv3, selected_feature, target_features, analysis_type="Full"):
    """
    Create a forest plot showing effect sizes (Cohen's d) for the selected feature
    against all target features (either all EEG or all Clinical features).
    
    Parameters:
    - df_wnv3: DataFrame with the data
    - selected_feature: The feature being analyzed
    - target_features: List of features to compare against
    - analysis_type: "Full" or "Significant" to filter results
    """
    from scipy.stats import mannwhitneyu, ttest_ind
    from statsmodels.stats.multitest import multipletests
    import matplotlib.patches as patches
    
    # Prepare data for analysis
    df_clean = df_wnv3[df_wnv3[selected_feature].notna()].copy()
    
    # Convert selected feature to binary if it's not already
    if df_clean[selected_feature].nunique() == 2:
        # Already binary
        unique_vals = sorted(df_clean[selected_feature].unique())
        df_clean['Group'] = df_clean[selected_feature].apply(lambda x: f"{selected_feature}+" if x == unique_vals[1] else f"{selected_feature}-")
    else:
        # Convert to binary using median split
        median_val = df_clean[selected_feature].median()
        df_clean['Group'] = df_clean[selected_feature].apply(lambda x: f"{selected_feature}+" if x >= median_val else f"{selected_feature}-")
    
    # Calculate effect sizes for each target feature
    results = []
    for feature in target_features:
        if feature not in df_clean.columns:
            continue
            
        # Get data for both groups
        group1_data = df_clean[df_clean['Group'] == f"{selected_feature}+"][feature].dropna()
        group2_data = df_clean[df_clean['Group'] == f"{selected_feature}-"][feature].dropna()
        
        if len(group1_data) < 2 or len(group2_data) < 2:
            continue
            
        # Calculate effect size (Cohen's d)
        n1, n2 = len(group1_data), len(group2_data)
        mean1, mean2 = group1_data.mean(), group2_data.mean()
        std1, std2 = group1_data.std(), group2_data.std()
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        # Statistical test
        try:
            # Normality test
            _, normal_p1 = stats.normaltest(group1_data)
            _, normal_p2 = stats.normaltest(group2_data)
            
            if normal_p1 < 0.05 or normal_p2 < 0.05:  # Non-parametric
                stat, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                test_used = "Mann-Whitney U"
            else:  # Parametric
                stat, p_value = ttest_ind(group1_data, group2_data)
                test_used = "T-test"
        except:
            p_value = np.nan
            test_used = "Failed"
        
        results.append({
            'Feature': feature,
            'Cohen_d': cohens_d,
            'P_value': p_value,
            'Test': test_used,
            'N1': n1,
            'N2': n2,
            'Mean1': mean1,
            'Mean2': mean2
        })
    
    if not results:
        st.warning("No valid comparisons could be made.")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Multiple comparison correction
    valid_pvals = results_df['P_value'].dropna()
    if len(valid_pvals) > 1:
        _, pvals_corrected, _, _ = multipletests(valid_pvals, method='fdr_bh')
        results_df.loc[valid_pvals.index, 'P_corrected'] = pvals_corrected
    else:
        results_df['P_corrected'] = results_df['P_value']
    
    # Filter by significance if requested
    if analysis_type == "Significant":
        results_df = results_df[results_df['P_corrected'] < 0.05]
    
    if results_df.empty:
        st.warning("No significant results found.")
        return
    
    # Sort by effect size
    results_df = results_df.sort_values('Cohen_d', key=abs, ascending=False)
    
    # Create forest plot
    fig, ax = plt.subplots(figsize=(12, max(6, len(results_df) * 0.4)))
    
    # Colors based on significance
    colors = ['red' if p < 0.05 else 'blue' for p in results_df['P_corrected']]
    
    # Plot effect sizes
    y_pos = np.arange(len(results_df))
    bars = ax.barh(y_pos, results_df['Cohen_d'], color=colors, alpha=0.7)
    
    # Add vertical line at 0
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add confidence intervals (simplified)
    for i, (idx, row) in enumerate(results_df.iterrows()):
        # Simple 95% CI approximation
        se = 1 / np.sqrt(row['N1'] + row['N2'] - 2)  # Simplified standard error
        ci_lower = row['Cohen_d'] - 1.96 * se
        ci_upper = row['Cohen_d'] + 1.96 * se
        
        ax.plot([ci_lower, ci_upper], [i, i], 'k-', linewidth=2)
        ax.plot([ci_lower, ci_lower], [i-0.1, i+0.1], 'k-', linewidth=2)
        ax.plot([ci_upper, ci_upper], [i-0.1, i+0.1], 'k-', linewidth=2)
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_df['Feature'], fontsize=10)
    ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12)
    ax.set_title(f"Forest Plot: {selected_feature} vs All {len(target_features)} Features\n"
                f"Effect Sizes (Cohen's d) with 95% Confidence Intervals", fontsize=14)
    
    # Add legend
    red_patch = patches.Patch(color='red', alpha=0.7, label='Significant (p < 0.05)')
    blue_patch = patches.Patch(color='blue', alpha=0.7, label='Non-significant')
    ax.legend(handles=[red_patch, blue_patch], loc='upper right')
    
    # Add effect size interpretation
    ax.text(0.02, 0.98, "Effect Size Interpretation:\n"
                        "|d| < 0.2: Small\n"
                        "0.2 ≤ |d| < 0.5: Medium\n"
                        "0.5 ≤ |d| < 0.8: Large\n"
                        "|d| ≥ 0.8: Very Large",
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Display in Streamlit
    st.subheader(f"Forest Plot: {selected_feature} vs All Features")
    st_pyplot_func(fig, filename=f'forest_plot_{selected_feature}_vs_all.png')
    
    # Display summary statistics table
    st.subheader("Summary Statistics")
    display_df = results_df[['Feature', 'Cohen_d', 'P_value', 'P_corrected', 'Test', 'N1', 'N2']].copy()
    display_df['Cohen_d'] = display_df['Cohen_d'].round(3)
    display_df['P_value'] = display_df['P_value'].apply(lambda x: f"{x:.3e}" if not pd.isna(x) else "N/A")
    display_df['P_corrected'] = display_df['P_corrected'].apply(lambda x: f"{x:.3e}" if not pd.isna(x) else "N/A")
    display_df.columns = ['Feature', "Cohen's d", 'P-value', 'P-corrected', 'Test', 'N+', 'N-']
    st.dataframe(display_df, use_container_width=True)
    
    plt.close(fig)

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
