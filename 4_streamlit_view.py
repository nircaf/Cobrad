import streamlit as st
import os
import pandas as pd

# streamlit run 4_streamlit_view.py
# Define the root directory where figures are stored
root_dir = "COBRAD_figures"

# Define plot types with their properties
# - name: folder name
# - has_subfolder: True if figures are in subfolders named after features (Type A), False if feature is in filename (Type B)
# - prefix: static prefix before EEG feature in filename (for Type A)
# - prefix_template: template for prefix including feature (for Type B)
# - suffix: suffix after EEG feature in filename
plot_types = [
    {"name": "boxplots", "has_subfolder": True, "prefix": "overall_", "suffix": "_comparison.png"},
    {"name": "hist", "has_subfolder": True, "prefix": "overall_", "suffix": "_hist_combined.png"},
    {"name": "hist", "has_subfolder": True, "prefix": "overall_", "suffix": "_hist_by_group.png"},
    {"name": "topomaps", "has_subfolder": False, "prefix_template": "{feature}_", "suffix": "_topomap.png"},
    {"name": "topomaps_p_values", "has_subfolder": True, "prefix": "p_values_", "suffix": "_topomap.png"},
    {"name": "topomaps_p_values_vs_controls", "has_subfolder": True, "prefix": "p_values_", "suffix": "_topomap.png"},
    {"name": "scatterplots", "has_subfolder": True, "prefix": "overall_", "suffix": "_regression.png"},
    {"name": "scatterplots", "has_subfolder": True, "prefix": "overall_", "suffix": "_histogram.png"},


    # Add new plot types here as needed, following the same structure
]

# Function to get list of features from Type A plot types
def get_features():
    features = set()
    for plot_type in plot_types:
        if plot_type["has_subfolder"]:
            folder = os.path.join(root_dir, plot_type["name"])
            if os.path.exists(folder):
                subfolders = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
                features.update(subfolders)
    return sorted(features)

# Function to get available EEG features for a selected feature
def get_eeg_features(selected_feature):
    eeg_features = set()
    for plot_type in plot_types:
        if plot_type["has_subfolder"]:
            folder = os.path.join(root_dir, plot_type["name"], selected_feature)
        else:
            folder = os.path.join(root_dir, plot_type["name"])
        
        if os.path.exists(folder):
            for file in os.listdir(folder):
                # Determine prefix based on plot type
                if "prefix" in plot_type:
                    prefix = plot_type["prefix"]
                elif "prefix_template" in plot_type:
                    prefix = plot_type["prefix_template"].format(feature=selected_feature)
                else:
                    continue
                suffix = plot_type["suffix"]
                
                if file.startswith(prefix) and file.endswith(suffix):
                    eeg_feature = file[len(prefix):-len(suffix)]
                    eeg_features.add(eeg_feature)
    return sorted(eeg_features)

# Function to load raw data for a given feature
def load_raw_data(feature):
    raw_data_file = os.path.join(root_dir, "raw_data", f"{feature}_raw_data.csv")
    if os.path.exists(raw_data_file):
        return pd.read_csv(raw_data_file)
    else:
        return None

# Streamlit app
st.title("COBRAD Figures Dashboard")

# Get list of features
features = get_features()
if not features:
    st.error("No features found in the specified directory structure.")
    st.stop()

# User selects a feature
selected_feature = st.selectbox("Select Feature", features)

# Get available EEG features for the selected feature
eeg_features = get_eeg_features(selected_feature)
if not eeg_features:
    st.warning(f"No EEG features found for {selected_feature}.")
else:
    # User selects an EEG feature
    selected_eeg_feature = st.selectbox("Select EEG Feature", eeg_features)
    

    # Display figures
    st.header(f"Figures for {selected_feature} - {selected_eeg_feature}")
    
    figures_found = False
    for plot_type in plot_types:
        # Determine the folder to look in
        if plot_type["has_subfolder"]:
            folder = os.path.join(root_dir, plot_type["name"], selected_feature)
        else:
            folder = os.path.join(root_dir, plot_type["name"])
        
        # Determine prefix
        if "prefix" in plot_type:
            prefix = plot_type["prefix"]
        elif "prefix_template" in plot_type:
            prefix = plot_type["prefix_template"].format(feature=selected_feature)
        else:
            continue
        suffix = plot_type["suffix"]
        
        # Construct expected filename
        figure_file = f"{prefix}{selected_eeg_feature}{suffix}"
        figure_path = os.path.join(folder, figure_file)
        
        st.subheader(plot_type["name"].capitalize())
        st.image(figure_path, caption=figure_file, use_column_width=True)
        figures_found = True
    
    if not figures_found:
        st.info(f"No figures found for {selected_feature} with EEG feature {selected_eeg_feature}.")

    # Load and display raw data
    raw_data = load_raw_data(selected_feature)
    if raw_data is not None:
        st.header(f"Raw Data for {selected_feature}")
        st.markdown(raw_data.to_html(escape=False), unsafe_allow_html=True)
    else:
        st.warning(f"No raw data found for {selected_feature}.")