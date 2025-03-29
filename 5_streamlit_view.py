import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os
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


def pairplot_columns(df, columns, hue=None):
    """
    Creates a pairplot for the specified columns in the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        columns (list): List of column names to include in the pairplot.
        hue (str, optional): Column name to use for color encoding (e.g., 'Group').
        output_dir (str, optional): Directory to save the pairplot image. If None, the plot is displayed in Streamlit.

    Returns:
        None
    """
    # Ensure the columns exist in the DataFrame
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"The following columns are missing in the DataFrame: {missing_columns}")

    # Drop rows with NaN values in the specified columns
    df_cleaned = df[columns + ([hue] if hue else [])].dropna()

    # Create the pairplot
    pairplot = sns.pairplot(df_cleaned, hue=hue, diag_kind='kde', corner=True)
    pairplot.fig.suptitle("Pairplot of Selected Columns", y=1.02)
    st.pyplot(pairplot)

def vs_controls_run(project_name):
    scatterplots_dir = f"{project_name}_figures/topomaps_p_values/vs_controls"
    boxplots_dir = f"{project_name}_figures/boxplots/vs_controls"
    # Display boxplots
    st.header("Boxplots vs Controls")
    if os.path.exists(boxplots_dir):
        boxplot_files = [f for f in os.listdir(boxplots_dir) if f.endswith('.png')]
        for file in boxplot_files:
            st.image(os.path.join(boxplots_dir, file), caption=file)
    else:
        st.write(f"No boxplots found in {boxplots_dir}")
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

# Streamlit App
def main():
    # have user choose COBRAD or WNV
    project_name = st.sidebar.selectbox("Select Project", ["COBRAD", "WNV"])
    if project_name == "COBRAD":
        # slider from 1 to 12
        num_samples_per_patient = st.sidebar.slider("Select number of samples per patient", 0, 12, 0)
        # Load COBRAD data
        df_wnv, patients_folder, control_folder, controls, df_wnv2, cases_group_name = cobrad_get_files(num_samples_per_patient)
    else:
        # Load WNV data
        df_wnv, patients_folder, control_folder, controls, df_wnv2, cases_group_name = wnv_get_files()
    # ask user if they want only significant, or full.
    st.sidebar.header("Select Analysis Type")
    analysis_type = st.sidebar.selectbox("Select Analysis Type", ["Significant", "Full"])
    st.title("EEG vs Clinical Features")
    # Iterate over each frequency band and plot the topomap
    frequency_bands = ['delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power','pswe_events_per_minute','pswe_avg_length','mean_mpf','dfv_std','dfv_mean']
    # remove from df_wnv2 columns that contain dates
    df_wnv2 = df_wnv2.drop(columns=[col for col in df_wnv2.columns if 'date' in col.lower()])
    #%% clinical data analysis
    clinical_columns, boxplot_columns = get_clinical_and_boxplot_cols(df_wnv2=df_wnv2)
    # Identify the separation point between clinical and EEG features
    separator_index = next((i for i, col in enumerate(df_wnv2.columns) if 'overall_' in col), None)
    if separator_index is None:
        st.error("No column with 'overall_' found to separate clinical and EEG features.")
        return
    
    # Split columns into clinical and EEG features
    clinical_features = [col for col in df_wnv2.columns[:separator_index] if col != 'Group']
    eeg_features = [col for col in df_wnv2.columns[separator_index:] if col != 'Group']
    clinical_features_numeric = [col for col in clinical_features if pd.api.types.is_numeric_dtype(df_wnv2[col])]
    
    boxplots_folder = f"{project_name}_figures/boxplots"
    scatterplots_folder = f"{project_name}_figures/scatterplots"
    
    marked_clinical_features = []
    dict_features = {}
    cols_to_skip = ['ID','annotations','bad_channels','Group','patient_number']
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
    
    # Sidebar for feature selection
    st.sidebar.header("Feature Selection")
    feature_type = st.sidebar.selectbox("Select feature type to plot against the other type:", ("Clinical Feature", "EEG Feature","ml_plots", "vs_Controls","Pair Plot"))
    
    if feature_type == "vs_Controls":
        vs_controls_run(project_name)
        return
    elif feature_type == "Pair Plot":
        selected_features = st.sidebar.multiselect("Select features for pairplot:", clinical_features + eeg_features)
        if selected_features:
            pairplot_columns(df_wnv2, selected_features)
        return
    elif feature_type == "ml_plots":
        # get the names of folders that are in {figures_dir}/ml_plots
        ml_plots_features = [f for f in os.listdir(f"{project_name}_figures/ml_plots") if os.path.isdir(os.path.join(f"{project_name}_figures/ml_plots", f))]
        selected_feature = st.sidebar.radio("Select a feature for ML plots:", ml_plots_features)
        if selected_feature:
            ml_plots_get_images(project_name, selected_feature)
        return
    elif feature_type == "EEG Feature":
        selected_feature = st.sidebar.radio("Select an EEG feature:", eeg_features)
        plot_title = f"Plots of {selected_feature} vs All Clinical Features"
        boxplot_columns = clinical_features_numeric
    else:
        selected_feature = st.sidebar.radio("Select a Clinical feature:", marked_clinical_features)
        # map back to key of dict_features
        selected_feature = [key for key, value in dict_features.items() if value == selected_feature][0]
        plot_title = f"Plots of {selected_feature} vs All EEG Features"
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
    numeric_colunms = df_wnv2.select_dtypes(include=[np.number]).columns
    # Display selected feature and plots
    if selected_feature:
        st.header(plot_title)
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
            for band in boxplot_columns:
                if selected_feature == 'sex':
                    # if 1 'f' else 'm'
                    df_wnv3['Group'] = df_wnv3[selected_feature].apply(lambda x: 'f' if x == 1 else 'm')
                elif selected_feature == 'sex, 1=male':
                    df_wnv3['Group'] = df_wnv3[selected_feature].apply(lambda x: 'm' if x == 1 else 'f')
                else:
                    # group values based on band if =1, else f'not {band}'
                    df_wnv3['Group'] = df_wnv3[selected_feature].apply(lambda x: selected_feature if x == 1 else f'not {selected_feature}')
                results_df = analyze_and_correct(df_wnv3, [band], groups=df_wnv3['Group'].unique())
                boxplot_plot(results_df, df_wnv3, band, f'{selected_feature}',is_streamlit=True)
            # if frequency band is contained in the column name
            # group_data = {}
            # for value in unique_values:
            #     group = selected_feature if value == 1 else f'not {selected_feature}'
            #     run_df = df_wnv3[df_wnv3[selected_feature] == value]
            #     group_data = process_group_data(group, run_df, frequency_bands, eeg_dict_convertion, eeg_channels, montage, group_data)
        # If numeric non-binary
        elif selected_feature in numeric_colunms:
            for band in boxplot_columns:
                scatter_plot_with_regression({}, df_wnv3, selected_feature, band, f'{selected_feature}',is_streamlit=True)

if __name__ == "__main__":
    main()