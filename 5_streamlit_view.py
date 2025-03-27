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

def boxplot_plot(results_df, df_wnv2, col, output_dir):
    # Function to remove outliers based on 5 standard deviations
    def remove_outliers(df, col, group_col, threshold=5):
        def filter_group(group):
            mean = group[col].mean()
            std = group[col].std()
            return group[np.abs(group[col] - mean) <= threshold * std]
        
        return df.groupby(group_col).apply(filter_group).reset_index(drop=True)

    # Remove outliers from each group
    cleaned_df = remove_outliers(df_wnv2, col, 'Group')

    # Plot the cleaned data
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Group', y=col, data=cleaned_df, showfliers=False)
    # Add stripplot
    sns.stripplot(x='Group', y=col, data=cleaned_df, alpha=0.5, jitter=True, color='black')
    # Add significance markers
    filtered_df = results_df[results_df['Variable'] == col]
    if filtered_df.empty:
        return
    row = results_df[results_df['Variable'] == col].iloc[0]
    if row['adj_p_value'] < 0.001:
        sig_symbol = '***'
    elif row['adj_p_value'] < 0.01:
        sig_symbol = '**'
    elif row['adj_p_value'] < 0.05:
        sig_symbol = '*'
    else:
        sig_symbol = 'ns'
    title_text = (
        f"{row['Test']}\n"
        f"p = {row['adj_p_value']:.3e} ({sig_symbol})\n"
        f"Cohen's d = {row['Cohen_d']:.2f}\n"
        f"{col} Comparison"
    )
    if sig_symbol != 'ns':
        plt.title(title_text, ha='center')
        # Add sample size to x-axis labels
        group_counts = df_wnv2['Group'].value_counts()
        ax = plt.gca()
        ax.set_xticklabels([f"{label.get_text()}\nn={group_counts[label.get_text()]}" for label in ax.get_xticklabels()])
        plt.tight_layout()
        st.write(f"Boxplot of {col} by Group")
        st.pyplot(plt)
        plt.close()
        
        # Plot histograms for each group and both groups together
        plt.figure(figsize=(10, 6))
        sns.histplot(data=cleaned_df, x=col, hue='Group', element='step', stat='density', common_norm=False)
        for i, group in enumerate(cleaned_df['Group'].unique()):
            group_data = cleaned_df[cleaned_df['Group'] == group][col]
            stats_text = stat_text_get(group_data)
            plt.annotate(stats_text, xy=(0.25, 0.95 - i * 0.1), xycoords='axes fraction', fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
        plt.title(f"{col} Histogram by Group")
        st.write(f"Histogram of {col} by Group")
        st.pyplot(plt)
        plt.close()
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data=cleaned_df, x=col, element='step', stat='density')
        combined_data = cleaned_df[col]
        stats_text = stat_text_get(combined_data)
        plt.annotate(stats_text, xy=(0.25, 0.95), xycoords='axes fraction', fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
        plt.title(f"{col} Histogram Combined")
        st.write(f"Histogram of {col} Combined")
        st.pyplot(plt)
        plt.close()

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
        
def scatter_plot_with_regression(results_df, df_wnv2, x_col, y_col, output_dir):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_col, y=y_col, data=df_wnv2, alpha=0.5, color='black')
    sns.regplot(x=x_col, y=y_col, data=df_wnv2, scatter=False, color='blue')
    # Perform linear regression
    X = sm.add_constant(df_wnv2[x_col])
    y = df_wnv2[y_col]
    model = sm.OLS(y, X).fit()
    p_value = model.pvalues[1]  # p-value for the slope
    r_squared = model.rsquared  # R-squared value
    # Determine significance symbol
    if (p_value < 0.001):
        sig_symbol = '***'
    elif (p_value < 0.01):
        sig_symbol = '**'
    elif (p_value < 0.05):
        sig_symbol = '*'
    else:
        sig_symbol = 'ns'
    if sig_symbol != 'ns':

        # Add stats results to the title
        plt.title(
            f"{x_col} vs {y_col} Regression\n"
            f"Slope p = {p_value:.3e} ({sig_symbol}), R^2 = {r_squared:.2f}, n = {len(df_wnv2)}",
            ha='center'
        )
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.tight_layout()
        st.write(f"Scatterplot of {x_col} vs {y_col}")
        st.pyplot(plt)
        plt.close()

        # Plot histogram of X
        plt.figure(figsize=(10, 6))
        sns.histplot(df_wnv2[x_col], color='blue', kde=True, stat='density', element='step')
        x_stats = stat_text_get(df_wnv2, x_col)
        plt.annotate(x_stats, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
        plt.title(f"Histogram of {x_col}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.tight_layout()
        st.write(f"Histogram of {x_col}")
        st.pyplot(plt)
        plt.close()

        # Plot histogram of Y
        plt.figure(figsize=(10, 6))
        sns.histplot(df_wnv2[y_col], color='red', kde=True, stat='density', element='step')
        y_stats = stat_text_get(df_wnv2, y_col) 
        plt.annotate(y_stats, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
        plt.title(f"Histogram of {y_col}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.tight_layout()
        st.write(f"Histogram of {y_col}")
        st.pyplot(plt)
        plt.close()
    
def analyze_and_correct(df_wnv2, columns_to_analyze, groups=['Control', 'WNV']):
    def analyze_groups(df_wnv2, col, groups):
        # Split data
        control_data = df_wnv2[df_wnv2['Group'] == groups[0]][col].dropna()
        case_data = df_wnv2[df_wnv2['Group'] == groups[1]][col].dropna()
        if len(control_data) < 2 or len(case_data) < 2:
            return None
        # Normality test
        _, normal_p = stats.normaltest(df_wnv2[col].dropna())
        # Choose appropriate test
        if normal_p < 0.05:  # Non-parametric
            stat, p = stats.mannwhitneyu(control_data, case_data)
            test_used = "Mann-Whitney U"
        else:  # Parametric
            stat, p = stats.ttest_ind(control_data, case_data)
            test_used = "T-test"
        # Effect size
        cohen_d = (case_data.mean() - control_data.mean()) / np.sqrt((
            (len(case_data)-1)*case_data.std()**2 + 
            (len(control_data)-1)*control_data.std()**2) / 
            (len(case_data) + len(control_data) - 2))
        return {
            'Variable': col,
            'Test': test_used,
            'Statistic': stat,
            'p_value': p,
            'Cohen_d': cohen_d
        }

    results = []    
    # Statistical analysis
    for col in columns_to_analyze:
        result = analyze_groups(df_wnv2, col, groups)
        if result is not None:
            results.append(result)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Multiple testing correction
    if not results_df.empty:
        rej, adj_p, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
        results_df['adj_p_value'] = adj_p
        results_df['Significant'] = adj_p < 0.05

    return results_df

def topomap_group_data( band, montage,control_data,wnv_data,output_dir):
    # Calculate p-values for each channel
    common_channels = control_data.columns.intersection(wnv_data.columns)
    dict_p_values = {}
    for common_channel in common_channels:
        control_channel = control_data[common_channel].dropna()
        wnv_channel = wnv_data[common_channel].dropna()
        if len(control_channel) < 2 or len(wnv_channel) < 2:
            continue
        _, p = stats.mannwhitneyu(control_channel, wnv_channel)
        dict_p_values[common_channel] = p
    df_p_values = pd.DataFrame(dict_p_values, index=[0]).T
    reject, pvals_corrected, _, _ = smm.multipletests(df_p_values[0].values, alpha=0.05, method='fdr_bh')
    df_p_values['pvals_corrected'] = pvals_corrected
    if any(df_p_values['pvals_corrected'] < 0.05):
        ch_names = df_p_values.index.tolist()
        # Create an info object
        info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types='eeg')
        info.set_montage(montage)
        # Create an EvokedArray object for p-values
        p_evoked = mne.EvokedArray(df_p_values['pvals_corrected'].values.reshape(-1, 1), info)
        # Plot the topomap of p-values
        fig, ax = plt.subplots()
        vlim_max = min(0.05, df_p_values['pvals_corrected'].max())
        im, cm = mne.viz.plot_topomap(p_evoked.data[:, 0], p_evoked.info, axes=ax, show=False, cmap='jet_r', vlim=[0, vlim_max])
        fig.colorbar(im, ax=ax)
        plt.title(f"{band} {output_dir} P-Value Topomap")
        st.write(f"{band} {output_dir} P-Value Topomap")
        st.pyplot(plt)
        plt.close()

def process_group_data(group, run_df, frequency_bands, eeg_dict_convertion, eeg_channels, montage,group_data):
    # get only columns that say EEG
    eeg_df = run_df.filter(like='EEG')
    group_data[group] = {}
    for band in frequency_bands:
        # get the columns which have the band name
        power_df = eeg_df.filter(like=band)
        # column name split ' '[-1]
        power_df.columns = power_df.columns.str.split(' ').str[-1]
        power_df.columns = [eeg_dict_convertion[col] if col in eeg_dict_convertion else col for col in power_df.columns]
        # leave only the eeg channels
        # Get the channels that exist in the DataFrame
        existing_channels = [ch for ch in eeg_channels if ch in power_df.columns]
        # Filter the DataFrame to include only the existing channels
        power_df = power_df[existing_channels]
        if power_df.empty:
            continue
        # drop columns more than half of the values are NaN
        power_df = power_df.dropna(axis=1, thresh=power_df.shape[0]//2)
        # change column names to only be the channel. if not
        # Extract the power values for the current band
        # power_values = power_df[band].values
        ch_names = power_df.columns.tolist()
        # Create an info object
        info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types='eeg')
        info.set_montage(montage)
        power_values = power_df.T.values
        group_data[group][band] = power_df

        # Create an EvokedArray object
        evoked = mne.EvokedArray(power_values, info)
        # Plot the topomap
        fig, ax = plt.subplots()
        im, cm = mne.viz.plot_topomap(evoked.data[:, 0], evoked.info, axes=ax, show=False)
        fig.colorbar(im, ax=ax)
        st.write(f"{band} {group} Topomap")
        st.pyplot(plt)
        plt.close()
    return group_data  

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
        # Load COBRAD data
        df_wnv, patients_folder, control_folder, controls, df_wnv2, cases_group_name = cobrad_get_files()
    else:
        # Load WNV data
        df_wnv, patients_folder, control_folder, controls, df_wnv2, cases_group_name = wnv_get_files()
    st.title("EEG vs Clinical Features Visualization")
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
    for feature in clinical_features:
        if os.path.exists(os.path.join(boxplots_folder, feature)) or os.path.exists(os.path.join(scatterplots_folder, feature)):
            feature_name = f"**{feature}**".upper()
            marked_clinical_features.append(feature_name)
            dict_features[feature] = feature_name
        else:
            feature_name = f"_{feature}_".lower()
            # marked_clinical_features.append(feature_name)
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

    feature_data = df_wnv2[selected_feature].dropna()
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean", f"{feature_data.mean():.2f}")
    col2.metric("Median", f"{feature_data.median():.2f}")
    col3.metric("Std Dev", f"{feature_data.std():.2f}")
    col4, col5 ,col6 = st.columns(3)
    col4.metric("Minimum", f"{feature_data.min():.2f}")
    col5.metric("Maximum", f"{feature_data.max():.2f}")
    # col 6 is N with dropna
    col6.metric("N", f"{feature_data.dropna().count()}")
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
                boxplot_plot(results_df, df_wnv3, band, f'{selected_feature}')
            # if frequency band is contained in the column name
            # group_data = {}
            # for value in unique_values:
            #     group = selected_feature if value == 1 else f'not {selected_feature}'
            #     run_df = df_wnv3[df_wnv3[selected_feature] == value]
            #     group_data = process_group_data(group, run_df, frequency_bands, eeg_dict_convertion, eeg_channels, montage, group_data)
        # If numeric non-binary
        elif df_wnv3[selected_feature].dtype in [np.float64, np.int64]:
            for band in boxplot_columns:
                scatter_plot_with_regression({}, df_wnv3, selected_feature, band, f'{selected_feature}')

if __name__ == "__main__":
    main()