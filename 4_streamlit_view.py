import streamlit as st
import os
import pandas as pd
import json
# streamlit run 4_streamlit_view.py
# Define the root directory where figures are stored

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
def load_raw_data():
    raw_data_file = os.path.join(root_dir, "raw_data", f"raw_data.csv")
    if os.path.exists(raw_data_file):
        return pd.read_csv(raw_data_file)
    else:
        return None
# ask user to choose if COBRAD or WNV
user_choice = st.selectbox("Select Dataset", ["COBRAD", "WNV"])
if user_choice == "COBRAD":
    root_dir = "COBRAD_figures"
else:
    root_dir = "WNV_figures"
# Streamlit app
st.title("COBRAD Figures Dashboard")
# Get list of features
features = get_features()
if not features:
    st.error("No features found in the specified directory structure.")
    st.stop()

# User selects a feature
selected_feature = st.selectbox("Select Feature", features)
raw_data = load_raw_data()
# load utils/cobrad_clinical.json
with open("utils/cobrad_clinical.json", "r") as f:
    clinical_columns_data = json.load(f)
    clinical_columns_data = pd.DataFrame(clinical_columns_data, index=["description"]).T

# Get available EEG features for the selected feature
eeg_features = get_eeg_features(selected_feature)
if not eeg_features:
    st.warning(f"No EEG features found for {selected_feature}.")
else:
    # User selects an EEG feature
    selected_eeg_feature = st.selectbox("Select EEG Feature", eeg_features)
    
    # Descriptive data on the selected feature
    # Load and display raw data
    feature_data = raw_data[selected_feature].dropna()
    # show N with dropna
    selected_feature_str = selected_feature.replace("_", " ")
    st.markdown(f"### **{selected_feature_str}**")
    st.write(clinical_columns_data.loc[selected_feature, "description"])
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean", f"{feature_data.mean():.2f}")
    col2.metric("Median", f"{feature_data.median():.2f}")
    col3.metric("Std Dev", f"{feature_data.std():.2f}")
    col4, col5 ,col6 = st.columns(3)
    col4.metric("Minimum", f"{feature_data.min():.2f}")
    col5.metric("Maximum", f"{feature_data.max():.2f}")
    # col 6 is N with dropna
    col6.metric("N", f"{feature_data.dropna().count()}")
    # Display figures
    st.header(f"Figures for {selected_feature_str} - {selected_eeg_feature.replace('_', ' ').capitalize()}")

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
        # if figure exists, display it
        if os.path.exists(figure_path):
            st.subheader(plot_type["name"].capitalize())
            st.image(figure_path, caption=figure_file, use_column_width=True)
            figures_found = True
    
    if not figures_found:
        st.info(f"No figures found for {selected_feature} with EEG feature {selected_eeg_feature}.")
    if raw_data is not None:
        st.header(f"Raw Data for {selected_feature}")
        # show raw data[selected_feature] 
        st.write(raw_data[selected_feature])

    else:
        st.warning(f"No raw data found for {selected_feature}.")