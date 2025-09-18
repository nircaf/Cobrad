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
    scatterplots_dir = f"{project_name}_figures/topomaps_p_values/vs_controls"
    boxplots_dir = f"{project_name}_figures/boxplots/vs_controls"
    controls_dir = f'temps_Controls_EDF' if project_name == 'COBRAD' else ''
    try:
        st.header('COBRAD Controls Demographics')
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
        st.error(f"Error occurred while processing controls demographics: {e}")
    for col in boxplot_columns:
        curr_data = combined_df[[col, 'Group']].dropna()
        num_groups = curr_data['Group'].nunique()
        if num_groups < 2:
            continue
        results_df = analyze_and_correct(curr_data, [col], groups=curr_data['Group'].unique())
        boxplot_plot_sns(results_df,curr_data, col, 'vs_controls',is_streamlit=True,analysis_type=analysis_type)
    # Display scatterplots
    st.header("Scatterplots vs Controls")
    if os.path.exists(scatterplots_dir):
        scatterplot_files = [f for f in os.listdir(scatterplots_dir) if f.endswith('.png')]
        for file in scatterplot_files:
            st.image(os.path.join(scatterplots_dir, file), caption=file)
    else:
        st.write(f"No scatterplots found in {scatterplots_dir}")

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
        HEP_dir = 'temps_EDF_HEP'
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
    # have user choose COBRAD or WNV
    # allow multiple selection for project
    selected_projects = st.sidebar.multiselect("Select Project(s)", ["COBRAD", "WNV", "TGA"], default=["COBRAD","TGA"])
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
    
    boxplots_folder = f"{project_name}_figures/boxplots"
    scatterplots_folder = f"{project_name}_figures/scatterplots"
    
    # Sidebar for feature selection
    st.sidebar.header("Feature Selection")
    feature_types = ("Clinical Feature", "EEG Feature", "ml_plots", "vs_Controls", "Pair Plot",'Spectogram','Raw',"HEP")
    feature_type = st.sidebar.selectbox("Select feature type to plot against the other type:", feature_types)
    if feature_type == "Clinical Feature" or feature_type == "EEG Feature" or feature_type == "vs_Controls":
        # ask user if they want only significant, or full.
        st.sidebar.header("Select Analysis Type")
        analysis_type = st.sidebar.selectbox("Select Analysis Type", [ "Full","Significant"])
    else:
        analysis_type = "Full"
        
    marked_clinical_features = []
    dict_features = {}
    bool_all_features = False
    cols_to_skip = ['ID','annotations','bad_channels','Group','patient_number','size','n_samples']
    clinical_features = [feature for feature in clinical_features if feature not in cols_to_skip]
    for feature in clinical_features:
        if os.path.exists(os.path.join(boxplots_folder, feature)) or os.path.exists(os.path.join(scatterplots_folder, feature)):
            feature_name = f"**{feature}**".upper()
            marked_clinical_features.append(feature_name)
            dict_features[feature] = feature_name
        else:
            feature_name = f"_{feature}_".lower()
            if analysis_type == 'Full':
                marked_clinical_features.append(feature_name)
            dict_features[feature] = feature_name
    
    if not clinical_features or not eeg_features:
        st.error("Could not identify clinical or EEG features based on the 'overall_' separator.")
        return
    if feature_type == "vs_Controls":
        vs_controls_run(project_name,df_wnv2,controls,boxplot_columns,analysis_type)
        return
    elif feature_type == "HEP":
        HEP_plots(project_name, df_wnv2, controls, boxplot_columns, analysis_type)
        return
    elif feature_type == "Pair Plot":
        pairplot_columns(df_wnv2, clinical_features, eeg_features)
        return
    elif feature_type == "ml_plots":
        # get the names of folders that are in {figures_dir}/ml_plots
        # ml_plots_features = [f for f in os.listdir(f"{project_name}_figures/ml_plots") if os.path.isdir(os.path.join(f"{project_name}_figures/ml_plots", f))]
        sorted_files = find_and_sort_ml_plots(f"{project_name}_figures/ml_plots")
        ml_plots_features = [f.split('/')[2] for f in sorted_files]
        selected_feature = st.sidebar.radio("Select a feature for ML plots:", ml_plots_features)
        if selected_feature:
            ml_plots_get_images(project_name, selected_feature)
        return
    elif feature_type == "Spectogram":
        st.title("Spectrogram")
        # ask user for win_sec
        win_sec = st.sidebar.slider("Select window size in seconds", 1, 30, 5)
        spectogram_run(cases_group_name,win_sec=win_sec)
        st.subheader("Spectrogram Controls")
        spectogram_run(f'Controls',win_sec=win_sec)

        return
    elif feature_type == "Raw":
        st.title("Raw Data")
        raw_run(cases_group_name)
        return
    elif feature_type == "EEG Feature":
        eeg_feature_options =  eeg_features+ ["All Features"] 
        selected_feature = st.sidebar.radio("Select an EEG feature:", eeg_feature_options)
        if selected_feature == "All Features":
            all_feat_list = eeg_features
            selected_feature = eeg_features[0]
            bool_all_features = True
        plot_title = f"Plots of {selected_feature} vs All Clinical Features"
        boxplot_columns = clinical_features_numeric
    else: # Clinical Feature
        clinical_features_correlation = st.sidebar.checkbox("Show Clinical Features Correlation", value=False)
        marked_clinical_features_w_all = marked_clinical_features + ["All Features"]
        selected_feature = st.sidebar.radio("Select a Clinical feature:", marked_clinical_features_w_all)
        if selected_feature == "All Features":
            all_feat_list = marked_clinical_features
            selected_feature = marked_clinical_features[0]
            bool_all_features = True
        # map back to key of dict_features
        selected_feature = [key for key, value in dict_features.items() if value == selected_feature][0]
        plot_title = f"Plots of {selected_feature} vs All EEG Features"
    st.header(plot_title)
    # remove ID from selected_feature
    feature_data = df_wnv2[selected_feature].dropna().astype(float)
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
    # if EEG Fearture
    if feature_type == "Clinical Feature":
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
                # st checkbox if HEP
                hep_checkbox = st.sidebar.checkbox("Show HEP Plots", value=True)
                if hep_checkbox:
                    # new container
                    with st.container():
                        st.subheader("HEP Plots")
                        HEP_plots(project_name, df_wnv3, controls, boxplot_columns, analysis_type,selected_feature)
                plot_tsne_by_group(df_wnv3)
                st.divider()
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
                for band in boxplot_columns:
                    results_df = analyze_and_correct(df_wnv3, [band], groups=df_wnv3['Group'].unique())
                    boxplot_plot(results_df, df_wnv3, band, f'{selected_feature}',is_streamlit=True,analysis_type=analysis_type)
                    # boxplot_plot_sns(results_df, df_wnv3, band, f'{selected_feature}',is_streamlit=True,analysis_type=analysis_type)
                # if frequency band is contained in the column name
                # group_data = {}
                # for value in unique_values:
                #     group = selected_feature if value == 1 else f'not {selected_feature}'
                #     run_df = df_wnv3[df_wnv3[selected_feature] == value]
                #     group_data = process_group_data(group, run_df, frequency_bands, eeg_dict_convertion, eeg_channels, montage, group_data)
            # If numeric non-binary
                # if col name has ( and )
            elif '(' in selected_feature and ')' in selected_feature:
                for band in boxplot_columns:
                    df_wnv3['Group'] = df_wnv3[selected_feature].astype(str)
                    df_wnv3[selected_feature] = df_wnv3[selected_feature].astype(float)
                    # do boxplot for each band
                    results_df = analyze_and_correct(df_wnv3, [band], groups=df_wnv3['Group'].unique())
                    boxplot_plot(results_df, df_wnv3, band, f'{selected_feature}',is_streamlit=True,analysis_type=analysis_type)
            elif selected_feature in numeric_colunms:
                for band in boxplot_columns:
                    scatter_plot_with_regression({}, df_wnv3, selected_feature, band, f'{selected_feature}',is_streamlit=True,analysis_type=analysis_type)
    # if selected_feature_w_all is not None
    if bool_all_features:
        for feature in all_feat_list:
            selected_feature = [key for key, value in dict_features.items() if value == feature][0]
            st.write(f"## Analyzing Feature: {selected_feature}")
            run_selected_feature()
            st.divider()
    else:
        run_selected_feature()

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
