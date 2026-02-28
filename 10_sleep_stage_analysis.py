import streamlit as st
import os
st.set_page_config(layout="wide")
import pickle
import numpy as np
import mne
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import pandas as pd
from scipy.signal import welch

# Optional pptx support
try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False


st.title("Sleep Stage Novel Analysis (Berkeley vs EDF)")
st.markdown("Analyzing Single Scan EEG, Heart-Brain Interaction, and AI Clustering between Healthy and Demented Groups.")

# Set up Matplotlib Customizations
_original_st_pyplot = st.pyplot
def custom_st_pyplot(fig=None, clear_figure=None, **kwargs):
    if fig is None:
        fig = plt.gcf()
    fig.patch.set_facecolor('#0e1117')
    for ax in fig.get_axes():
        ax.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        
        legend = ax.get_legend()
        if legend is not None:
            plt.setp(legend.get_texts(), color='white')
            frame = legend.get_frame()
            frame.set_facecolor('#0e1117')
            frame.set_edgecolor('white')
            frame.set_alpha(0.8)
    _original_st_pyplot(fig, clear_figure, **kwargs)

st.pyplot = custom_st_pyplot

# Paths
BASE_PATH = "/storage/pblab_shared_data/Nir/Cobrad/pickles_sleep_stage"
GROUPS = ["Berkeley_data", "EDF"]
STAGES = ["N1", "N2", "N3", "W"]

st.sidebar.header("Data Selection")
selected_group = st.sidebar.selectbox("Select Group", ["Both"] + GROUPS, index=0)
selected_stage = st.sidebar.selectbox("Select Sleep Stage", STAGES, index=0)

# Optional dependency imports
try:
    from autoreject import AutoReject
except ImportError:
    AutoReject = None
try:
    from mne.time_frequency import psd_array_welch
except ImportError:
    pass

@st.cache_data
def get_file_list(group, stage):
    group_path = os.path.join(BASE_PATH, group, stage)
    if not os.path.exists(group_path):
        return []
    return [os.path.join(group_path, f) for f in os.listdir(group_path) if f.endswith('.pkl')]

def load_pkl(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# --- Feature Extraction Functions ---
def compute_pswe(psd_values):
    # Power Spectral Wavelet Entropy approximation
    total_power = float(np.sum(psd_values))
    if total_power <= 0: return 0.0
    prob = psd_values / (total_power + 1e-12)
    n_bins = len(prob)
    if n_bins <= 1: return 0.0
    entropy = -np.sum(prob * np.log(prob + 1e-12))
    return float(entropy / np.log(n_bins))

def extract_features(raw):
    # Helper to extract features from a loaded raw object
    sfreq = raw.info["sfreq"]
    data = raw.get_data()
    ch_names = raw.ch_names
    
    # Frequency domain features
    psds, freqs = psd_array_welch(data, sfreq=sfreq, fmin=0.5, fmax=40.0, average="mean", verbose=False)
    
    # Calculate band powers
    bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 40)}
    features = {}
    
    eeg_ch_indices = []
    ecg_ch_idx = None
    
    for ch_idx, ch_name in enumerate(ch_names):
        if "ECG" in ch_name.upper() or "EKG" in ch_name.upper():
            ecg_ch_idx = ch_idx
            continue
        eeg_ch_indices.append(ch_idx)
        psd = psds[ch_idx]
        for band, (fmin, fmax) in bands.items():
            idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
            if np.any(idx_band):
                bp = np.sum(psd[idx_band])
            else:
                bp = 0
            features[f"{ch_name}_{band}_power"] = bp
        features[f"{ch_name}_pswe"] = compute_pswe(psd)
        
    # Heart-Brain Interaction Metrics
    if ecg_ch_idx is not None:
        try:
            ecg_signal = data[ecg_ch_idx]
            # Clean ECG and detect peaks
            ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sfreq)
            _, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sfreq)
            rpeaks = info["ECG_R_Peaks"]
            
            # HRV metrics (Time domain RMSSD)
            hrv_time = nk.hrv_time(rpeaks, sampling_rate=sfreq)
            if not hrv_time.empty:
                features["HRV_RMSSD"] = float(hrv_time["HRV_RMSSD"].iloc[0])
                features["HRV_MeanNN"] = float(hrv_time["HRV_MeanNN"].iloc[0])
                
            # HEP (Heartbeat Evoked Potential) on EEG channels
            # Create events from rpeaks
            events = np.column_stack((rpeaks, np.zeros_like(rpeaks), np.ones_like(rpeaks)))
            # Epoch from -0.2 to 0.6 seconds around R-peak
            epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.6, baseline=(-0.2, 0), preload=True, verbose=False)
            evoked = epochs.average()
            
            # Extract HEP amplitude (mean amplitude between 200-600ms)
            hep_data = evoked.get_data()
            times = evoked.times
            hep_window = np.logical_and(times >= 0.2, times <= 0.6)
            
            for idx in eeg_ch_indices:
                ch_name = ch_names[idx]
                hep_amp = np.mean(hep_data[idx, hep_window])
                features[f"{ch_name}_HEP_amplitude"] = hep_amp
                
        except Exception as e:
            # If ECG processing fails, silently pass to avoid breaking the whole pipeline
            features["HRV_RMSSD"] = 0.0
            features["HRV_MeanNN"] = 0.0
            for idx in eeg_ch_indices:
                features[f"{ch_names[idx]}_HEP_amplitude"] = 0.0
                
    return features


def clean_eeg_super_res(raw):
    """Super resolution cleaning using CSD (Current Source Density) if applicable and filtering"""
    raw_cleaned = raw.copy()
    raw_cleaned.notch_filter(50.0, verbose=False)
    raw_cleaned.filter(0.5, 40.0, verbose=False)
    
    # Try CSD
    try:
        raw_cleaned = mne.preprocessing.compute_current_source_density(raw_cleaned)
    except Exception as e:
        pass # Not all montages support CSD without proper coordinates
        
    return raw_cleaned

# --- Streamlit UI App ---
def main():
    st.write(f"### Analyzing Stage: {selected_stage}")
    
    groups_to_run = GROUPS if selected_group == "Both" else [selected_group]
    
    all_files = []
    for g in groups_to_run:
        all_files.extend(get_file_list(g, selected_stage))
        
    st.write(f"Found {len(all_files)} files.")
    
    # Dashboard Tabs
    tab1, tab2, tab3 = st.tabs(["Single Scan Viewer", "Group Level Differential Analysis", "Network AI Clustering"])
    
    with tab1:
        st.subheader("Single Scan EEG & ECG Viewer")
        if all_files:
            target_file = st.selectbox("Select Patient File", all_files)
            if st.button("Process & View"):
                with st.spinner("Processing..."):
                    raw = load_pkl(target_file)
                    st.write(raw.info)
                    
                    # Clean
                    raw_cleaned = clean_eeg_super_res(raw)
                    
                    st.write("#### Raw vs Cleaned Visualization")
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6))
                    
                    time_idx = slice(0, int(raw.info["sfreq"] * 10)) # first 10 seconds
                    times = raw.times[time_idx]
                    
                    # Look for first EEG channel
                    eeg_ch_idx = 0
                    for i, ch in enumerate(raw.ch_names):
                        if "ECG" not in ch.upper() and "EKG" not in ch.upper():
                            eeg_ch_idx = i
                            break
                    
                    eeg_ch_name = raw.ch_names[eeg_ch_idx]
                    
                    ax1.plot(times, raw.get_data()[eeg_ch_idx, time_idx], label="Raw")
                    ax1.legend(); ax1.set_title(f"Raw EEG ({eeg_ch_name})")
                    
                    ax2.plot(times, raw_cleaned.get_data()[eeg_ch_idx, time_idx], label="Cleaned (Super Res)", color='orange')
                    ax2.legend(); ax2.set_title(f"Cleaned EEG ({eeg_ch_name})")
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.write("#### Extracted Features")
                    feats = extract_features(raw_cleaned)
                    st.dataframe(pd.DataFrame([feats]))
                    
    with tab2:
        st.subheader("Group Level Differential Analysis")
        st.write("Process all individuals and compare features via T-Test.")
        if st.button("Run Group Analysis", key="btn_group"):
            if len(groups_to_run) < 2:
                st.warning("Please select 'Both' groups to run comparative analysis.")
            else:
                with st.spinner("Extracting features for all patients... this may take a while."):
                    group_data = []
                    for g in groups_to_run:
                        files = get_file_list(g, selected_stage)
                        for f in files:
                            try:
                                raw = load_pkl(f)
                                raw_c = clean_eeg_super_res(raw)
                                feats = extract_features(raw_c)
                                feats["Group"] = g
                                feats["Patient"] = os.path.basename(f)
                                group_data.append(feats)
                            except Exception as e:
                                st.error(f"Failed processing {f}: {e}")
                    
                    if group_data:
                        df = pd.DataFrame(group_data)
                        st.session_state["group_df"] = df
                        st.success("Feature extraction complete!")
                        st.dataframe(df.head())
        
        if "group_df" in st.session_state:
            df = st.session_state["group_df"]
            st.write("#### Statistical Comparison (Berkeley vs EDF)")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            results = []
            from scipy.stats import ttest_ind
            
            g1 = df[df["Group"] == "Berkeley_data"]
            g2 = df[df["Group"] == "EDF"]
            
            for col in numeric_cols:
                stat, p = ttest_ind(g1[col].dropna(), g2[col].dropna(), equal_var=False)
                results.append({"Feature": col, "T-Statistic": stat, "P-Value": p})
            
            res_df = pd.DataFrame(results).sort_values("P-Value")
            st.dataframe(res_df)
            
            # Plot top feature
            top_feat = res_df.iloc[0]["Feature"]
            st.write(f"#### Top Feature Distribution: {top_feat}")
            fig, ax = plt.subplots(figsize=(8, 5))
            import seaborn as sns
            sns.boxplot(data=df, x="Group", y=top_feat, ax=ax, palette="Set2")
            st.pyplot(fig)
                        
    with tab3:
        st.subheader("Network Level AI Clustering")
        st.write("Unsupervised clustering (K-Means/GMM) and dimensionality reduction (PCA) on extracted features.")
        if "group_df" in st.session_state:
            df = st.session_state["group_df"]
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            from sklearn.cluster import KMeans
            import seaborn as sns
            
            X = df[numeric_cols].fillna(0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            df["PCA1"] = X_pca[:, 0]
            df["PCA2"] = X_pca[:, 1]
            
            n_clusters = st.slider("Number of Clusters (K-Means)", 2, 5, 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df["Cluster"] = kmeans.fit_predict(X_scaled)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Group", palette="Set1", ax=ax1, s=100)
            ax1.set_title("True Labels (Groups)")
            
            sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="viridis", ax=ax2, s=100)
            ax2.set_title(f"K-Means Clustering (K={n_clusters})")
            
            st.pyplot(fig)
            st.write(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
        else:
            st.warning("Please run Group Analysis in Tab 2 first to extract features.")

if __name__ == "__main__":
    main()
