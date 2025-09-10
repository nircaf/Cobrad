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

# Optional LightningChart 3D helper
def plot_lightning_3d(vals_pat, vals_ctrl, pos, channels_names):
    """
    Attempt to render a simple 3D chart using LightningChart (v0.9.3).
    Inputs:
      - vals_pat: array-like of patient values (per channel)
      - vals_ctrl: array-like of control values (per channel)
      - pos: array-like of (x,y) positions for each channel (shape: n_channels x 2)
      - channels_names: list of channel names in the same order as pos/vals
    Returns:
      - (chart_3d, brain_model) on success, (None, None) on failure
    Notes:
      - This function is defensive: it will catch import/usage errors and return (None, None).
      - The LightningChart Python API and available primitives can vary; this function uses a minimal approach
        and falls back gracefully if the exact API is not available in the runtime.
    """
    try:
        # Import LightningChart dashboard module. This requires lightningchart==0.9.3 to be installed.
        import lightningchart as lc
        import streamlit.components.v1 as components

        lc.set_license('P001-ZqGqPDvXFDmhqXa92cd6KHJyCoswDwCnqkxCo3xWz6x2jOv0Ths=-MEQCIGOG4bzJwFvTBOAvYndMWc3cTt1zfL6+Y1C3GXN7lk6gAiAXwpJuBgm74Fz6F7slZbGbeym+gnL2+qClV6GZe8em8A==')
        
        dashboard = lc.Dashboard(rows=1, columns=1)

        chart_3d = dashboard.Chart3D(column_index=1,row_index=1)
        # Create a mesh model placeholder on the chart
        brain_model = chart_3d.add_mesh_model()
        
        # Prepare data
        vals_pat = np.asarray(vals_pat, dtype=float)
        vals_ctrl = np.asarray(vals_ctrl, dtype=float)
        diff = vals_pat - vals_ctrl
        pos_arr = np.asarray(pos, dtype=float)
        
        # If there is a point-series API, use it to add colored points at electrode positions.
        # We'll attempt a point-series first; if unavailable, we leave the mesh_model as a placeholder.
        try:
            # Many lightningchart versions expose add_point_series3d / add_point_series. Try common names.
            if hasattr(chart_3d, "add_point_series3d"):
                series = chart_3d.add_point_series3d()
            elif hasattr(chart_3d, "add_point_series"):
                series = chart_3d.add_point_series()
            else:
                series = None

            if series is not None:
                # Add each electrode as a small point with the difference value as an attribute (for coloring)
                for i, name in enumerate(channels_names):
                    if i >= len(pos_arr):
                        break
                    x, y = pos_arr[i, 0], pos_arr[i, 1]
                    z = 0.0  # flat on z plane; LightningChart supports 3D points
                    value = float(diff[i]) if i < len(diff) else 0.0
                    # The exact method to add a point differs across API versions; try common ones.
                    if hasattr(series, "add"):
                        # (x, y, z, value) or (x, y, z)
                        try:
                            series.add(x, y, z, value)
                        except TypeError:
                            # fallback to add(x,y,z)
                            series.add(x, y, z)
                    elif hasattr(series, "add_point"):
                        try:
                            series.add_point(x, y, z, value)
                        except TypeError:
                            series.add_point(x, y, z)
                # Optionally configure series coloring by value if API supports it
                try:
                    if hasattr(series, "set_color_map"):
                        series.set_color_map('RdBu')
                except Exception:
                    pass
        except Exception:
            # If point series cannot be created, keep the mesh placeholder
            pass

        return chart_3d, brain_model
    except Exception as e:
        # If LightningChart is not installed or any error occurs, warn and return None
        st.warning(f"LightningChart 3D unavailable or failed to initialize: {e}")
        return None, None

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
        # Load main datasets
        edf_data = pd.read_csv('EDF.csv')
        controls_data = pd.read_csv('EDF_controls.csv')
        
        # Group EDF data by patient_number and calculate mean
        print("Grouping EDF data by patient...")
        numeric_cols = edf_data.select_dtypes(include=[np.number]).columns.tolist()
        # If 'patient_number' was detected as numeric, remove it so reset_index() won't try to insert a duplicate column
        if 'patient_number' in numeric_cols:
            numeric_cols.remove('patient_number')
        edf_patient_means = edf_data.groupby('patient_number')[numeric_cols].mean().reset_index()
        
        # Add group labels
        edf_patient_means['Group'] = 'Patients'
        controls_data['Group'] = 'Controls'
        # print channels from columns that contain EEG and split by ' '[-1]
        p_ch = set([col.split(' ')[-1] for col in edf_patient_means.columns if 'EEG' in col])
        c_ch = set([col.split(' ')[-1] for col in controls_data.columns if 'EEG' in col])
        print(f'Patients channels: {p_ch}')
        print(f"Controls channels: {c_ch}")
        print(f"Patients channels - Controls channels: {p_ch - c_ch}")
        print(f"Controls channels - Patients channels: {c_ch - p_ch}")
        
        
        
        # Combine datasets for comparison
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
    st.markdown('<h2 class="section-header">Data Distributions</h2>', unsafe_allow_html=True)
    
    # Measurement selection
    measurement_categories = list(measurements.keys())
    selected_category = st.selectbox("Select Measurement Category", measurement_categories)
    
    if selected_category:
        category_measurements = list(measurements[selected_category].keys())
        selected_measurement = st.selectbox("Select Measurement", category_measurements)
        
        if selected_measurement:
            # Check if measurement exists in data
            available_measurements = [col for col in combined_df.columns if selected_measurement in col]
            
            if available_measurements:
                # Show overall measurement
                if f'overall_{selected_measurement}' in combined_df.columns:
                    measurement_col = f'overall_{selected_measurement}'
                else:
                    measurement_col = available_measurements[0]
                
                # Create plots
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(
                        create_distribution_plot(combined_df, measurement_col),
                        use_container_width=True
                    )
                
                with col2:
                    st.plotly_chart(
                        create_comparison_plot(combined_df, measurement_col),
                        use_container_width=True
                    )
                
                # Show channel-specific measurements if available
                if len(available_measurements) > 1:
                    st.markdown("### Channel-Specific Measurements")
                    channel_measurements = [m for m in available_measurements if 'EEG' in m]
                    
                    if channel_measurements:
                        # electrode_options.split ' '[0]
                        measurement_options = [ch.split(' ')[0] for ch in channel_measurements]
                        # Allow user to choose a specific electrode or the overall measurement
                        measurement_col = st.selectbox("Choose electrode", measurement_options, index=0)
                        # get the vals of the columns that is in measurement_col
                        p_measurement_cols = [col for col in patients_df.columns if measurement_col in col]
                        patients_vals = patients_df[p_measurement_cols].dropna()

                        c_measurement_cols = [col for col in controls_df.columns if measurement_col in col]
                        controls_vals = controls_df[c_measurement_cols].dropna()
                        
                        # Build summary table (mean, std, n) for both groups
                        summary_table = pd.DataFrame([{
                            'Group': 'Patients',
                            'N': len(patients_vals),
                            'Mean': patients_vals.mean() if len(patients_vals) > 0 else np.nan,
                            'Std': patients_vals.std() if len(patients_vals) > 0 else np.nan
                        },{
                            'Group': 'Controls',
                            'N': len(controls_vals),
                            'Mean': controls_vals.mean() if len(controls_vals) > 0 else np.nan,
                            'Std': controls_vals.std() if len(controls_vals) > 0 else np.nan
                        }])
                        
                        # Compute statistical test and effect size from the raw arrays (robust to temp column)
                        def _compute_test(a, b):
                            if len(a) == 0 or len(b) == 0:
                                return None, None, None, None
                            try:
                                p_normal_a = stats.shapiro(a)[1] if len(a) >= 3 else 0.0
                                p_normal_b = stats.shapiro(b)[1] if len(b) >= 3 else 0.0
                            except Exception:
                                p_normal_a, p_normal_b = 0.0, 0.0
                            
                            if (p_normal_a > 0.05) and (p_normal_b > 0.05):
                                statistic, p_value = ttest_ind(a, b, nan_policy='omit')
                                test_name = "Independent t-test"
                            else:
                                try:
                                    statistic, p_value = mannwhitneyu(a, b, alternative='two-sided')
                                except Exception:
                                    statistic, p_value = None, None
                                test_name = "Mann-Whitney U test"
                            
                            # Cohen's d (pooled std)
                            try:
                                pooled_std = np.sqrt(((len(a) - 1) * a.std()**2 + (len(b) - 1) * b.std()**2) / (len(a) + len(b) - 2))
                                cohens_d = (a.mean() - b.mean()) / pooled_std if pooled_std != 0 else np.nan
                            except Exception:
                                cohens_d = np.nan
                            
                            return test_name, statistic, p_value, cohens_d
                        
                        test_name, statistic, p_value, cohens_d = _compute_test(patients_vals, controls_vals)
                        
                        stats_row = {
                            'Measurement': measurement_col,
                            'Test': test_name,
                            'Statistic': statistic,
                            'P-value': p_value,
                            "Effect Size (Cohen's d)": cohens_d
                        }
                        
                        # Display summary and statistical comparison
                        st.markdown(f"#### {measurement_col} — Summary")
                        st.dataframe(summary_table.set_index('Group').round(4))
                        
                        st.markdown("#### Statistical Comparison (Patients vs Controls)")
                        stats_df = pd.DataFrame([stats_row])
                        st.dataframe(stats_df.round(6), use_container_width=True)
                        
                        channels_names = set([ch.split(' ')[-1] for ch in channel_measurements])
                        # Create topomap(s) of channel measurements (averaged by group)
                        channel_data = combined_df[['Group'] + channel_measurements]
                        channel_pivot = channel_data.groupby('Group')[channel_measurements].mean()
                        
                        # Attempt to plot with MNE topomap; fall back to heatmap if MNE is not available
                        try:
                            import mne
                            
                            # Use standard 10-20 montage to get channel positions
                            montage = mne.channels.make_standard_montage('standard_1020')
                            ch_pos = montage.get_positions()['ch_pos']
                            # Build position array and values aligned to ch_names (only keep channels with known positions)
                            pos_list = []
                            vals_pat = []
                            vals_ctrl = []
                            ch_names_order = []
                            for ch in channels_names:
                                ch_and_measurement = f'{measurement_col} {ch}'
                                # lookup position in montage (returns (x,y,z))
                                p = ch_pos.get(ch)
                                if p is None:
                                    # skip channels without known montage position
                                    continue
                                # keep only x, y for topomap
                                pos_list.append([p[0], p[1]])
                                # safely get values; fall back to NaN when missing
                                try:
                                    vals_pat.append(channel_pivot.loc['Patients', ch_and_measurement])
                                except Exception:
                                    vals_pat.append(np.nan)
                                try:
                                    vals_ctrl.append(channel_pivot.loc['Controls', ch_and_measurement])
                                except Exception:
                                    vals_ctrl.append(np.nan)
                                ch_names_order.append(ch)
                            
                            # Convert to numpy arrays with proper shapes
                            if len(pos_list) == 0:
                                raise RuntimeError("No matching channel positions found for topomap.")
                            pos = np.array(pos_list, dtype=float)  # shape (n_channels, 2)
                            vals_pat = np.array(vals_pat, dtype=float)
                            vals_ctrl = np.array(vals_ctrl, dtype=float)
                            
                            # Plot Patients & Controls side-by-side in one figure, Difference in a separate figure below
                            # Top figure: Patients (left) and Controls (right)
                            top_fig, top_axes = plt.subplots(1, 2, figsize=(12, 5))
                            # Determine common scale for Patients/Controls
                            try:
                                combined_vals = np.concatenate([vals_pat, vals_ctrl])
                                if np.all(np.isnan(combined_vals)):
                                    vmin_all, vmax_all = None, None
                                else:
                                    vmin_all = np.nanmin(combined_vals)
                                    vmax_all = np.nanmax(combined_vals)
                            except Exception:
                                vmin_all, vmax_all = None, None
                            
                            top_im, _ = mne.viz.plot_topomap(
                                vals_pat, pos, axes=top_axes[0], show=False, names=ch_names_order,
                                vlim = (vmin_all, vmax_all)
                            )
                            top_axes[0].set_title('Patients')
                            
                            top_im2, _ = mne.viz.plot_topomap(
                                vals_ctrl, pos, axes=top_axes[1], show=False, names=ch_names_order,
                                vlim = (vmin_all, vmax_all)
                            )
                            top_axes[1].set_title('Controls')
                            
                            # Add shared colorbar for top figure (Patients/Controls) at far right outside axes
                            try:
                                if (vmin_all is not None) and (vmax_all is not None):
                                    # create a dedicated colorbar axis on the right side of the figure
                                    # [left, bottom, width, height] in figure coordinates
                                    cax = top_fig.add_axes([0.92, 0.15, 0.02, 0.7])
                                    cbar_top = top_fig.colorbar(top_im, cax=cax, orientation='vertical')
                                    cbar_top.ax.set_ylabel(measurement_col)
                            except Exception:
                                pass
                            
                            plt.suptitle(f"Topomap: {measurement_col} (Patients / Controls)")
                            plt.tight_layout()
                            st.pyplot(top_fig)
                            
                            # Bottom figure: Difference (full width)
                            diff = vals_pat - vals_ctrl
                            vmax_diff = np.nanmax(np.abs(diff)) if np.any(~np.isnan(diff)) else None
                            
                            diff_fig, diff_ax = plt.subplots(1, 1, figsize=(6, 4))
                            diff_im, _ = mne.viz.plot_topomap(
                                diff, pos, axes=diff_ax, show=False, cmap='RdBu_r',
                                vlim = (-vmax_diff if vmax_diff is not None else None, vmax_diff if vmax_diff is not None else None),
                                names=ch_names_order
                            )
                            diff_ax.set_title('Patients - Controls')
                            
                            # Colorbar for difference map
                            try:
                                if vmax_diff is not None:
                                    cbar_diff = diff_fig.colorbar(diff_im, ax=diff_ax, orientation='vertical')
                                    cbar_diff.ax.set_ylabel('Difference')
                            except Exception:
                                pass
                            
                            plt.suptitle(f"Topomap: Difference — Average {measurement_col}")
                            plt.tight_layout()
                            st.pyplot(diff_fig)
                        except Exception as e:
                            # Fallback to heatmap if mne is not installed or plotting fails
                            fig = px.imshow(
                                channel_pivot.T,
                                title=f"Average {measurement_col} by Channel and Group",
                                color_continuous_scale='RdBu_r'
                            )
                            fig.update_layout(
                                xaxis_title="Group",
                                yaxis_title="EEG Channel"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.warning(f"Topomap unavailable (MNE missing or error): {e}")
                        
                        # Clean up temporary column if created
                        if '_temp_channel_mean' in combined_df.columns:
                            combined_df.drop(columns=['_temp_channel_mean'], inplace=True)
                        # plot_lightning_3d
                        # plot_lightning_3d(vals_pat,vals_ctrl, pos, ch_names_order)

            else:
                st.warning(f"Measurement '{measurement_col}' not found in the data.")
    
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
