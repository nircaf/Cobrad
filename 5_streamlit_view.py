import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from utils.eeg_utils import *
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

def HEP_plots(project_name, df_wnv3, controls, boxplot_columns, analysis_type,selected_feature=None):
    if project_name == 'COBRAD':
        HEP_dir = 'parquet_HEP/EDF_HEP'
    else:
        return
    # Run over power bands
    for band_name, band_range in power_bands.items():
        st.write(f"Analyzing power band: {band_name}")
        # glob all files in HEP_dir that match f"*_{band_name}.parquet"
        band_files = [f for f in os.listdir(HEP_dir) if f.endswith(f"_{band_name}.parquet")]
        dfs = []
        hue = None
        for file in band_files:
            file_path = os.path.join(HEP_dir, file)
            if selected_feature:
                ID = file.split('_')[0][1:]
                id_group = df_wnv3[df_wnv3['ID'] == ID]['Group'].values
            try:
                df = pd.read_parquet(file_path)
                if selected_feature:
                    # Attach group info to each df
                    if len(id_group) > 0 and not df.empty:
                        df['Group'] = id_group[0]
                        dfs.append(df)
            except Exception as e:
                st.warning(f"Could not read {file}: {e}")
        if dfs:
            if selected_feature:
                # Group by 'Group' and compute mean for each group
                group_dfs = []
                for group in df_wnv3['Group'].unique():
                    group_patients2 = []
                    for df in dfs:
                        if 'Group' in df.columns and df['Group'].iloc[0] == group:
                            group_patients2.append(df)
                    # group_patients = pd.concat(group_patients2, ignore_index=False, axis=0)
                    # Filter out dfs shorter than 50 rows
                    group_patients = [df for df in group_patients2 if df.shape[0] >= 50]
                    if group_patients:
                        min_len = min(df.shape[0] for df in group_patients)
                        if min_len < 50:
                            continue
                        # Truncate all dfs to min_len rows
                        truncated = [df.drop(columns=['Group'], errors='ignore').iloc[:min_len] for df in group_patients]
                        # Get mean of each df (returns a Series) and concat to DataFrame
                        group_mean_df = pd.concat([t.mean() for t in truncated], axis=1).T
                        group_mean_df['Group'] = group
                        group_dfs.append(group_mean_df)
                results_df = pd.concat(group_dfs, ignore_index=True)
                # remove cols more than 50% NaN
                results_df = results_df.loc[:, results_df.isnull().mean() < 0.5]
                # remove rows with any NaN
                results_df = results_df.dropna()
                hue = results_df['Group']
                st.write(f"Loaded {len(results_df)} group means for band {band_name}")
            else:
                # Compute the mean of all DataFrames (row-wise, column-wise)
                results_df = pd.concat(dfs, ignore_index=True).groupby(level=0).mean()
                st.write(f"Loaded {len(results_df)} rows (mean of all files) for band {band_name}")
        else:
            st.write(f"No data found for band {band_name}")
        #  apply zscore each column (only numeric columns)
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns
        results_df[numeric_cols] = results_df[numeric_cols].apply(zscore)
        # if selected_feature not None. 
        # run only_plots on results_df
        only_plots(results_df, save_plot='', save_dir='', edf_pickle_name="plot", band=band_name, step_sec=5,is_streamlit=True,hue=hue)
        st.divider()




# Streamlit App
def main():
    # options to choose from folders in pickles
    default_options = ["COBRAD", "WNV"] +  [f for f in os.listdir('parquet_results') if os.path.isdir(os.path.join('parquet_results', f))] 
    # allow multiple selection for project
    # default_options = ["COBRAD", "WNV"]
    # Deduplicate while preserving order
    seen = set()
    opts = [x for x in default_options if not (x in seen or seen.add(x))]
    selected_projects = st.sidebar.pills("Select Project(s)", opts, default=[x for x in ["Seeg"] if x in opts],selection_mode ="multi")
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
    feature_types = ('All', 'Clinical Feature', "EEG Feature", "ml_plots", "vs_Controls", "Pair Plot",'Raw','Spectrogram')
    feature_type = st.sidebar.selectbox("Select feature type to plot against the other type:", feature_types)
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
        if current_feature_type == "vs_Controls":
            vs_controls_run(project_name,df_wnv2,controls,boxplot_columns,analysis_type)
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
        elif current_feature_type == "EEG Feature":
            forest_plot_eeg = st.sidebar.checkbox("Show Forest Plot vs All Clinical Features", value=False)
            eeg_feature_options =  eeg_features+ ["All Features"] 
            selected_feature = st.sidebar.selectbox("Select an EEG feature:", eeg_feature_options)
            if selected_feature == "All Features" or feature_type == "All":
                all_feat_list = eeg_features
                selected_feature = eeg_features[0]
                bool_all_features = True
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
                bool_all_features = True
            plot_title = f"Plots of {selected_feature} vs All EEG Features"
            st.header(plot_title)
            # keep only rows that can be safely converted to float
            feature_data = (
                df_wnv2[selected_feature]
                .apply(pd.to_numeric, errors='coerce')  # turn invalids into NaN
                .dropna()
                .astype(float)
            )
            if feature_data.empty:
                st.warning(f"No valid numeric data available for the selected feature: {selected_feature}")
                continue  # Skip to next feature type
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
            def run_selected_feature():
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
                    hep_checkbox = st.checkbox("Show HEP & TSNE Plots", value=False)
                    if hep_checkbox:
                        st.subheader("HEP Plots")
                        HEP_plots(project_name, df_wnv3, controls, boxplot_columns, analysis_type,selected_feature)
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
                for feature in all_feat_list:
                    selected_feature = feature
                    st.write(f"## Analyzing Feature: {selected_feature}")
                    run_selected_feature()
                    # Add forest plot if requested
                    if forest_plot_clinical:
                        st.subheader("Forest Plot Analysis")
                        forest_plot_all_features(df_wnv2, selected_feature, eeg_features, analysis_type)
                    st.divider()
            else:
                run_selected_feature()
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
