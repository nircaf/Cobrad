import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency
import warnings
warnings.filterwarnings('ignore')
from utils.eeg_utils import cobrad_get_files

# Page configuration
st.set_page_config(
    page_title="EEG Data Analysis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better presentation
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stExpander > div > div {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the data"""
    try:
        # Get data from cobrad_get_files function
        df_wnv, patients_folder, temp_controls, temp_df_wnv2, temp_cases_group_name = cobrad_get_files(sample_window_size=0, only_awake=False)
        
        # Use the returned data
        edf_patient_means = temp_df_wnv2.copy()
        controls_data = temp_controls.copy()
        
        # Add group labels
        edf_patient_means['Group'] = 'Patients'
        controls_data['Group'] = 'Controls'
        
        # Print channels from columns that contain EEG and split by ' '[-1]
        p_ch = set([col.split(' ')[-1] for col in edf_patient_means.columns if 'EEG' in col])
        c_ch = set([col.split(' ')[-1] for col in controls_data.columns if 'EEG' in col])
        print(f'Patients channels: {p_ch}')
        print(f"Controls channels: {c_ch}")
        print(f"Patients channels - Controls channels: {p_ch - c_ch}")
        print(f"Controls channels - Patients channels: {c_ch - p_ch}")
        
        # Combine datasets for comparison - only include mutual columns
        mutual_columns = list(set(edf_patient_means.columns) & set(controls_data.columns))
        combined_data = pd.concat([
            edf_patient_means[mutual_columns], 
            controls_data[mutual_columns]
        ], ignore_index=True)

        return edf_patient_means, controls_data, combined_data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

def get_eeg_measurements():
    """Define EEG measurement categories and their meanings"""
    measurements = {
        'Basic Statistics': {
            'mean': 'Mean amplitude - Average signal strength across the recording period',
            'median': 'Median amplitude - Middle value of signal distribution, less affected by outliers',
            'std': 'Standard deviation - Variability in signal amplitude',
            'min': 'Minimum amplitude - Lowest signal value recorded',
            'max': 'Maximum amplitude - Highest signal value recorded'
        },
        'Power Spectral Density': {
            'psd_mean': 'PSD Mean - Average power across all frequencies',
            'psd_std': 'PSD Standard Deviation - Variability in power distribution',
            'delta_power': 'Delta Power (0.5-4 Hz) - Associated with deep sleep and unconsciousness',
            'theta_power': 'Theta Power (4-8 Hz) - Related to drowsiness, meditation, and memory',
            'alpha_power': 'Alpha Power (8-13 Hz) - Present during relaxed wakefulness, eyes closed',
            'beta_power': 'Beta Power (13-30 Hz) - Associated with active concentration and alertness',
            'gamma_power': 'Gamma Power (30-100 Hz) - Related to consciousness and cognitive processing'
        },
        'Frequency Analysis': {
            'fft_mean': 'FFT Mean - Average frequency content',
            'fft_std': 'FFT Standard Deviation - Variability in frequency distribution',
            'mean_mpf': 'Mean Peak Frequency - Average frequency of highest power',
            'median_mpf': 'Median Peak Frequency - Middle frequency of peak power',
            'df_mean': 'Dominant Frequency Mean - Most prominent frequency component',
            'dfv_std': 'Dominant Frequency Variability - Consistency of dominant frequency'
        },
        'Signal Characteristics': {
            'skewness': 'Skewness - Asymmetry of signal distribution (positive = right-skewed)',
            'kurtosis': 'Kurtosis - Peakness of distribution (high = sharp peaks, low = flat)',
            'overall_pswe_median_percentage': 'PSWE Percentage - Percentage of time with pathological slow wave events',
            'overall_pswe_events_per_minute': 'PSWE Events/Min - Frequency of pathological slow wave events',
            'overall_pswe_avg_length': 'PSWE Average Length - Mean duration of pathological events'
        }
    }
    return measurements

def create_summary_stats(df, group_name):
    """Create summary statistics for a group"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = df[numeric_cols].describe()
    
    # Add additional statistics
    summary.loc['skewness'] = df[numeric_cols].skew()
    summary.loc['kurtosis'] = df[numeric_cols].kurtosis()
    summary.loc['missing_pct'] = (df[numeric_cols].isnull().sum() / len(df)) * 100
    
    return summary

def perform_statistical_tests(patients_df, controls_df, measurement):
    """Perform statistical tests between groups"""
    try:
        # Get data for both groups
        patients_data = patients_df[measurement].dropna()
        controls_data = controls_df[measurement].dropna()
        
        if len(patients_data) == 0 or len(controls_data) == 0:
            return None, None, None, None
        
        # Normality test
        _, p_normal_patients = stats.shapiro(patients_data)
        _, p_normal_controls = stats.shapiro(controls_data)
        
        # Choose appropriate test
        if p_normal_patients > 0.05 and p_normal_controls > 0.05:
            # Both groups normal - use t-test
            statistic, p_value = ttest_ind(patients_data, controls_data)
            test_name = "Independent t-test"
        else:
            # Non-normal - use Mann-Whitney U
            statistic, p_value = mannwhitneyu(patients_data, controls_data, alternative='two-sided')
            test_name = "Mann-Whitney U test"
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(patients_data) - 1) * patients_data.std()**2 + 
                             (len(controls_data) - 1) * controls_data.std()**2) / 
                            (len(patients_data) + len(controls_data) - 2))
        cohens_d = (patients_data.mean() - controls_data.mean()) / pooled_std
        
        return test_name, statistic, p_value, cohens_d
        
    except Exception as e:
        st.error(f"Error in statistical test for {measurement}: {str(e)}")
        return None, None, None, None

def create_distribution_plot(df, measurement, group_col='Group'):
    """Create distribution plot for a measurement"""
    fig = px.histogram(
        df, 
        x=measurement, 
        color=group_col,
        marginal="box",
        title=f"Distribution of {measurement}",
        nbins=50,
        opacity=0.7
    )
    fig.update_layout(
        xaxis_title=measurement,
        yaxis_title="Count",
        showlegend=True
    )
    return fig

def create_comparison_plot(df, measurement, group_col='Group'):
    """Create comparison box plot for a measurement"""
    fig = px.box(
        df, 
        x=group_col, 
        y=measurement,
        title=f"Comparison of {measurement} between Groups",
        color=group_col
    )
    fig.update_layout(
        xaxis_title="Group",
        yaxis_title=measurement,
        showlegend=False
    )
    return fig

def create_multi_histogram_plot(df, columns, color_by_binary=None):
    """Create multiple histograms in one figure"""
    if len(columns) == 1:
        # Single plot - check if binary or continuous
        data = df[columns[0]].dropna()
        unique_values = data.nunique()
        
        if unique_values == 2:
            # Create pie chart for binary data
            value_counts = data.value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=value_counts.index.tolist(),
                values=value_counts.values.tolist(),
                textinfo='label+value'
            )])
            fig.update_layout(title=f"Distribution of {columns[0]}")
        else:
            # Create histogram for continuous data
            if color_by_binary and color_by_binary in df.columns:
                # Create colored histogram by binary groups
                combined_data = df[[columns[0], color_by_binary]].dropna()
                fig = go.Figure()
                
                for binary_value in combined_data[color_by_binary].unique():
                    group_data = combined_data[combined_data[color_by_binary] == binary_value][columns[0]]
                    fig.add_trace(go.Histogram(
                        x=group_data,
                        name=f"{binary_value}",
                        nbinsx=50,
                        opacity=0.7
                    ))
                
                fig.update_layout(
                    title=f"Distribution of {columns[0]} by {color_by_binary}",
                    xaxis_title=columns[0],
                    yaxis_title="Count",
                    barmode='overlay'
                )
            else:
                # Regular histogram without color grouping
                fig = px.histogram(
                    df, 
                    x=columns[0], 
                    title=f"Distribution of {columns[0]}",
                    nbins=50,
                    opacity=0.7
                )
                fig.update_layout(
                    xaxis_title=columns[0],
                    yaxis_title="Count"
                )
    else:
        # Separate binary and continuous columns
        binary_columns = []
        continuous_columns = []
        
        for col in columns:
            data = df[col].dropna()
            if len(data) > 0:
                unique_values = data.nunique()
                if unique_values == 2:
                    binary_columns.append(col)
                else:
                    continuous_columns.append(col)
        
        # Create separate figures for binary and continuous data
        if binary_columns and continuous_columns:
            # Both types exist - create two separate figures
            st.markdown("### Binary/Categorical Variables")
            if len(binary_columns) == 1:
                # Single pie chart
                data = df[binary_columns[0]].dropna()
                value_counts = data.value_counts()
                fig_binary = go.Figure(data=[go.Pie(
                    labels=value_counts.index.tolist(),
                    values=value_counts.values.tolist(),
                    textinfo='label+percent',
                    textposition='outside',
                    textfont_size=14,
                    rotation=90
                )])
                fig_binary.update_layout(
                    title=f"Distribution of {binary_columns[0]}",
                    margin=dict(t=50, b=50, l=50, r=50),
                    showlegend=False
                )
                st.plotly_chart(fig_binary, use_container_width=True)
            else:
                # Multiple pie charts in subplots
                from plotly.subplots import make_subplots
                n_cols = min(2, len(binary_columns))
                n_rows = (len(binary_columns) + n_cols - 1) // n_cols
                
                fig_binary = make_subplots(
                    rows=n_rows, 
                    cols=n_cols,
                    specs=[[{"type": "pie"}] * n_cols for _ in range(n_rows)],
                    subplot_titles=binary_columns,
                    vertical_spacing=0.1
                )
                
                for i, col in enumerate(binary_columns):
                    row = (i // n_cols) + 1
                    col_idx = (i % n_cols) + 1
                    data = df[col].dropna()
                    value_counts = data.value_counts()
                    fig_binary.add_trace(
                        go.Pie(
                            labels=value_counts.index.tolist(),
                            values=value_counts.values.tolist(),
                            textinfo='label+value',
                            textposition='outside',
                            textfont_size=14,
                            rotation=90
                        ),
                        row=row, col=col_idx
                    )
                
                fig_binary.update_layout(
                    title="Binary/Categorical Variables",
                    showlegend=False,
                    height=300 * n_rows,
                    margin=dict(t=50, b=50, l=50, r=50)
                )
                st.plotly_chart(fig_binary, use_container_width=True)
            
            st.markdown("### Continuous Variables")
            # Create histogram subplots for continuous data
            from plotly.subplots import make_subplots
            n_cols = min(2, len(continuous_columns))
            n_rows = (len(continuous_columns) + n_cols - 1) // n_cols
            
            fig_continuous = make_subplots(
                rows=n_rows, 
                cols=n_cols,
                subplot_titles=continuous_columns,
                vertical_spacing=0.1
            )
            
            for i, col in enumerate(continuous_columns):
                row = (i // n_cols) + 1
                col_idx = (i % n_cols) + 1
                data = df[col].dropna()
                
                if color_by_binary and color_by_binary in df.columns:
                    # Create colored histogram by binary groups
                    binary_data = df[color_by_binary].dropna()
                    combined_data = df[[col, color_by_binary]].dropna()
                    
                    for binary_value in combined_data[color_by_binary].unique():
                        group_data = combined_data[combined_data[color_by_binary] == binary_value][col]
                        fig_continuous.add_trace(
                            go.Histogram(
                                x=group_data,
                                name=f"{col} ({binary_value})",
                                nbinsx=50,
                                opacity=0.7,
                                legendgroup=col
                            ),
                            row=row, col=col_idx
                        )
                else:
                    # Regular histogram without color grouping
                    fig_continuous.add_trace(
                        go.Histogram(
                            x=data,
                            name=col,
                            nbinsx=50,
                            opacity=0.7
                        ),
                        row=row, col=col_idx
                    )
            
            fig_continuous.update_layout(
                title="Continuous Variables",
                showlegend=color_by_binary is not None,
                height=300 * n_rows
            )
            
            # Update x-axis labels
            for i in range(1, n_rows + 1):
                for j in range(1, n_cols + 1):
                    fig_continuous.update_xaxes(title_text="Value", row=i, col=j)
                    fig_continuous.update_yaxes(title_text="Count", row=i, col=j)
            
            st.plotly_chart(fig_continuous, use_container_width=True)
            return None  # Return None since we've already displayed the charts
            
        elif binary_columns:
            # Only binary columns
            if len(binary_columns) == 1:
                data = df[binary_columns[0]].dropna()
                value_counts = data.value_counts()
                fig = go.Figure(data=[go.Pie(
                    labels=value_counts.index.tolist(),
                    values=value_counts.values.tolist(),
                    textinfo='label+value',
                    textposition='outside',
                    textfont_size=14,
                    rotation=90
                )])
                fig.update_layout(
                    title=f"Distribution of {binary_columns[0]}",
                    margin=dict(t=50, b=50, l=50, r=50),
                    showlegend=False
                )
            else:
                from plotly.subplots import make_subplots
                n_cols = min(2, len(binary_columns))
                n_rows = (len(binary_columns) + n_cols - 1) // n_cols
                
                fig = make_subplots(
                    rows=n_rows, 
                    cols=n_cols,
                    specs=[[{"type": "pie"}] * n_cols for _ in range(n_rows)],
                    subplot_titles=binary_columns,
                    vertical_spacing=0.1
                )
                
                for i, col in enumerate(binary_columns):
                    row = (i // n_cols) + 1
                    col_idx = (i % n_cols) + 1
                    data = df[col].dropna()
                    value_counts = data.value_counts()
                    fig.add_trace(
                        go.Pie(
                            labels=value_counts.index.tolist(),
                            values=value_counts.values.tolist(),
                            textinfo='label+value',
                            textposition='outside',
                            textfont_size=14,
                            rotation=90
                        ),
                        row=row, col=col_idx
                    )
                
                fig.update_layout(
                    title="Binary/Categorical Variables",
                    showlegend=False,
                    height=300 * n_rows,
                    margin=dict(t=50, b=50, l=50, r=50)
                )
        
        else:
            # Only continuous columns - create histogram subplots
            from plotly.subplots import make_subplots
            n_cols = min(2, len(continuous_columns))
            n_rows = (len(continuous_columns) + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows, 
                cols=n_cols,
                subplot_titles=continuous_columns,
                vertical_spacing=0.1
            )
            
            for i, col in enumerate(continuous_columns):
                row = (i // n_cols) + 1
                col_idx = (i % n_cols) + 1
                data = df[col].dropna()
                
                if color_by_binary and color_by_binary in df.columns:
                    # Create colored histogram by binary groups
                    combined_data = df[[col, color_by_binary]].dropna()
                    
                    for binary_value in combined_data[color_by_binary].unique():
                        group_data = combined_data[combined_data[color_by_binary] == binary_value][col]
                        fig.add_trace(
                            go.Histogram(
                                x=group_data,
                                name=f"{col} ({binary_value})",
                                nbinsx=50,
                                opacity=0.7,
                                legendgroup=col
                            ),
                            row=row, col=col_idx
                        )
                else:
                    # Regular histogram without color grouping
                    fig.add_trace(
                        go.Histogram(
                            x=data,
                            name=col,
                            nbinsx=50,
                            opacity=0.7
                        ),
                        row=row, col=col_idx
                    )
            
            fig.update_layout(
                title="Patient Data Distributions",
                showlegend=color_by_binary is not None,
                height=300 * n_rows
            )
            
            # Update x-axis labels
            for i in range(1, n_rows + 1):
                for j in range(1, n_cols + 1):
                    fig.update_xaxes(title_text="Value", row=i, col=j)
                    fig.update_yaxes(title_text="Count", row=i, col=j)
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 EEG Data Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Patient-Level Analysis of EEG Measurements: Patients vs Controls")
    st.info("📊 **Data Level**: Patient means (averaged across all sessions per patient)")
    
    # Load data
    with st.spinner("Loading data..."):
        patients_df, controls_df, combined_df = load_data()
    
    if patients_df is None:
        st.error("Failed to load data. Please check if the CSV files exist.")
        return
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    
    # Data overview
    st.sidebar.markdown("### 📈 Data Overview")
    st.sidebar.metric("Patients", len(patients_df))
    st.sidebar.metric("Controls", len(controls_df))
    st.sidebar.metric("Total Subjects", len(combined_df))
    st.sidebar.markdown("**Note**: Patient data shows means across all sessions")
    
    # Get measurement categories
    measurements = get_eeg_measurements()
    
    # -- Begin single-page layout (previously tabs) --
    
    # Section: Data Overview
    st.markdown('<h2 class="section-header">Data Overview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Patients Group Summary (Patient Means)")
        patients_summary = create_summary_stats(patients_df, "Patients")
        st.dataframe(patients_summary.round(4), use_container_width=True)
    
    with col2:
        st.markdown("### Controls Group Summary")
        controls_summary = create_summary_stats(controls_df, "Controls")
        st.dataframe(controls_summary.round(4), use_container_width=True)
    
    # Key metrics comparison
    st.markdown("### Key Metrics Comparison")
    
    # Select key measurements for comparison
    key_measurements = ['overall_pswe_median_percentage', 'overall_pswe_events_per_minute', 
                      'overall_delta_power', 'overall_alpha_power', 'overall_beta_power']
    
    comparison_data = []
    for measurement in key_measurements:
        if measurement in patients_df.columns and measurement in controls_df.columns:
            patients_mean = patients_df[measurement].mean()
            controls_mean = controls_df[measurement].mean()
            comparison_data.append({
                'Measurement': measurement,
                'Patients Mean': patients_mean,
                'Controls Mean': controls_mean,
                'Difference': patients_mean - controls_mean,
                'Difference %': ((patients_mean - controls_mean) / controls_mean * 100) if controls_mean != 0 else 0
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df.round(4), use_container_width=True)
    
    st.markdown("---")
    
    # Section: Data Distributions
    st.markdown('<h2 class="section-header">Patient Data Distributions</h2>', unsafe_allow_html=True)
    
    # Get numeric columns from patients data
    numeric_columns = patients_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove 'Group' column if it exists
    if 'Group' in numeric_columns:
        numeric_columns.remove('Group')
    
    # Multi-select for columns
    selected_columns = st.multiselect(
        "Select columns to visualize (multi-select enabled)",
        numeric_columns,
        default=numeric_columns[:3] if len(numeric_columns) >= 3 else numeric_columns,
        help="Select one or more columns to display histograms. Multiple selections will be shown in subplots."
    )
    
    # Binary column selection for color grouping
    binary_columns_for_color = [col for col in numeric_columns if patients_df[col].nunique() == 2]
    color_by_binary = None
    if binary_columns_for_color:
        color_by_binary = st.selectbox(
            "Color histograms by binary group (optional)",
            ["None"] + binary_columns_for_color,
            help="Select a binary column to color the histograms by different groups"
        )
        if color_by_binary == "None":
            color_by_binary = None
    
    if selected_columns:
        # Create multi-histogram plot
        fig = create_multi_histogram_plot(patients_df, selected_columns, color_by_binary)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        
        # Show summary statistics for selected columns
        st.markdown("### Summary Statistics for Selected Columns")
        summary_stats = patients_df[selected_columns].describe()
        st.dataframe(summary_stats.round(4), use_container_width=True)

    
    st.markdown("---")
    
    # Section: Statistical Analysis
    st.markdown('<h2 class="section-header">Statistical Analysis</h2>', unsafe_allow_html=True)
    
    # Select measurements for statistical analysis
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
    overall_measurements = [col for col in numeric_cols if col.startswith('overall_')]
    
    selected_measurements = st.multiselect(
        "Select measurements for statistical analysis",
        overall_measurements,
        default=overall_measurements[:5]
    )
    
    if selected_measurements:
        # Perform statistical tests
        results = []
        
        for measurement in selected_measurements:
            test_name, statistic, p_value, cohens_d = perform_statistical_tests(
                patients_df, controls_df, measurement
            )
            
            if p_value is not None:
                # Determine significance
                if p_value < 0.001:
                    significance = "***"
                elif p_value < 0.01:
                    significance = "**"
                elif p_value < 0.05:
                    significance = "*"
                else:
                    significance = "ns"
                
                results.append({
                    'Measurement': measurement,
                    'Test': test_name,
                    'Statistic': statistic,
                    'P-value': p_value,
                    'Effect Size (Cohen\'s d)': cohens_d,
                    'Significance': significance
                })
        
        if results:
            results_df = pd.DataFrame(results)
            st.dataframe(results_df.round(4), use_container_width=True)
            
            # Summary of significant differences
            significant_results = results_df[results_df['P-value'] < 0.05]
            if len(significant_results) > 0:
                st.markdown("### Significant Differences (p < 0.05)")
                st.dataframe(significant_results[['Measurement', 'P-value', 'Effect Size (Cohen\'s d)']].round(4), 
                           use_container_width=True)
            else:
                st.info("No statistically significant differences found between groups.")
    
    st.markdown("---")
    
    # Section: Raw Data
    st.markdown('<h2 class="section-header">Raw Data</h2>', unsafe_allow_html=True)
    
    # Data selection
    data_choice = st.radio("Select dataset to view", ["Patients (Means)", "Controls", "Combined"])
    
    if data_choice == "Patients (Means)":
        display_df = patients_df
    elif data_choice == "Controls":
        display_df = controls_df
    else:
        display_df = combined_df
    
    # Column selection
    all_columns = display_df.columns.tolist()
    selected_columns = st.multiselect(
        "Select columns to display",
        all_columns,
        default=all_columns[:20]  # Show first 20 columns by default
    )
    
    if selected_columns:
        st.dataframe(display_df[selected_columns], use_container_width=True)
        
        # Download option
        csv = display_df[selected_columns].to_csv(index=False)
        file_name = f"{data_choice.lower().replace(' ', '_').replace('(', '').replace(')', '')}_data.csv"
        st.download_button(
            label="Download selected data as CSV",
            data=csv,
            file_name=file_name,
            mime="text/csv"
        )
    
    st.markdown("---")
    
    # Section: Measurement Guide
    st.markdown('<h2 class="section-header">Measurement Guide</h2>', unsafe_allow_html=True)
    st.markdown("### Understanding EEG Measurements")
    
    for category, measures in measurements.items():
        with st.expander(f"📊 {category}", expanded=False):
            for measure, description in measures.items():
                st.markdown(f"**{measure}**: {description}")
    
    # Additional information
    st.markdown("### Additional Information")
    st.info("""
    **Data Collection Notes:**
    - Sampling frequency: 256 Hz
    - High-pass filter: 0.15-1 Hz
    - Low-pass filter: 67-70 Hz
    - Recording duration: 60 minutes per session
    - Electrode placement: Standard 10-20 system
    - **Patient data**: Averaged across all sessions per patient
    - **Control data**: Individual session measurements
    """)
    
    st.warning("""
    **Interpretation Guidelines:**
    - Statistical significance (p < 0.05) indicates reliable differences between groups
    - Effect size (Cohen's d) indicates practical significance:
      - Small: 0.2
      - Medium: 0.5
      - Large: 0.8
    - PSWE (Pathological Slow Wave Events) are markers of neurological dysfunction
    """)

if __name__ == "__main__":
    main()
