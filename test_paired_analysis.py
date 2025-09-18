#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import glob
from scipy import stats

def load_pre_post_data():
    """Load pre and post data from parquet files and determine group membership from EDF directory structure."""
    
    # Load all parquet files
    parquet_dir = "parquet_results/VNS_PRE_POST_25"
    parquet_files = glob.glob(os.path.join(parquet_dir, "*.parquet"))
    
    pre_data = []
    post_data = []
    
    # EDF directory structure for determining pre/post
    edf_base_dir = "EDF Format/VNS_PRE_POST_25"
    
    for parquet_file in parquet_files:
        try:
            # Extract filename without extension
            filename = os.path.basename(parquet_file).replace('.edf.parquet', '')
            
            # Load parquet data
            df = pd.read_parquet(parquet_file)
            
            # Determine if this is pre or post based on EDF directory structure
            is_post = False
            for vns_dir in glob.glob(os.path.join(edf_base_dir, "VNS*")):
                post_dir = os.path.join(vns_dir, "POST")
                if os.path.exists(post_dir):
                    post_files = glob.glob(os.path.join(post_dir, f"{filename}*"))
                    if post_files:
                        is_post = True
                        break
            
            # Add group label
            df['Group'] = 'POST' if is_post else 'PRE'
            df['Subject_ID'] = filename
            
            if is_post:
                post_data.append(df)
            else:
                pre_data.append(df)
                
        except Exception as e:
            print(f"Could not load {parquet_file}: {e}")
            continue
    
    if not pre_data and not post_data:
        print("No data found!")
        return None, None, None
    
    pre_df = pd.concat(pre_data, ignore_index=True) if pre_data else pd.DataFrame()
    post_df = pd.concat(post_data, ignore_index=True) if post_data else pd.DataFrame()
    
    # Find patients with both PRE and POST data
    pre_subjects = set(pre_df['Subject_ID'].unique()) if len(pre_df) > 0 else set()
    post_subjects = set(post_df['Subject_ID'].unique()) if len(post_df) > 0 else set()
    paired_subjects = pre_subjects.intersection(post_subjects)
    
    # Filter to only include paired subjects
    if len(paired_subjects) > 0:
        pre_paired = pre_df[pre_df['Subject_ID'].isin(paired_subjects)].copy()
        post_paired = post_df[post_df['Subject_ID'].isin(paired_subjects)].copy()
        
        # Sort by Subject_ID to ensure proper pairing
        pre_paired = pre_paired.sort_values('Subject_ID').reset_index(drop=True)
        post_paired = post_paired.sort_values('Subject_ID').reset_index(drop=True)
        
        return pre_paired, post_paired, paired_subjects
    else:
        print("No patients found with both PRE and POST data!")
        return None, None, None

def get_eeg_features(df):
    """Extract EEG features from the dataframe."""
    # Get columns that contain EEG data (typically start with 'overall_' or contain frequency band names)
    eeg_columns = []
    for col in df.columns:
        if any(band in col.lower() for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']) or 'overall_' in col:
            eeg_columns.append(col)
    return eeg_columns

def test_paired_analysis(pre_df, post_df, feature):
    """Test paired statistical analysis for a specific feature."""
    
    # Ensure we have the same subjects in both datasets
    pre_subjects = set(pre_df['Subject_ID'].unique()) if len(pre_df) > 0 else set()
    post_subjects = set(post_df['Subject_ID'].unique()) if len(post_df) > 0 else set()
    common_subjects = pre_subjects.intersection(post_subjects)
    
    if len(common_subjects) < 2:
        print(f"Insufficient paired data for {feature}")
        return None
    
    # Prepare paired data
    pre_vals = []
    post_vals = []
    for subject in common_subjects:
        pre_val = pre_df[pre_df['Subject_ID'] == subject][feature].iloc[0] if len(pre_df[pre_df['Subject_ID'] == subject]) > 0 else np.nan
        post_val = post_df[post_df['Subject_ID'] == subject][feature].iloc[0] if len(post_df[post_df['Subject_ID'] == subject]) > 0 else np.nan
        
        if not np.isnan(pre_val) and not np.isnan(post_val):
            pre_vals.append(pre_val)
            post_vals.append(post_val)
    
    if len(pre_vals) < 2:
        print(f"Insufficient paired data for {feature}")
        return None
    
    # Wilcoxon signed-rank test for paired data
    try:
        stat, p_value = stats.wilcoxon(post_vals, pre_vals, alternative='two-sided')
    except:
        p_value = np.nan
    
    # Cohen's d for paired data (using difference scores)
    try:
        differences = np.array(post_vals) - np.array(pre_vals)
        cohen_d = np.mean(differences) / np.std(differences, ddof=1)
    except:
        cohen_d = np.nan
    
    print(f"Feature: {feature}")
    print(f"Paired N: {len(pre_vals)}")
    print(f"PRE Mean ± SD: {np.mean(pre_vals):.3f} ± {np.std(pre_vals):.3f}")
    print(f"POST Mean ± SD: {np.mean(post_vals):.3f} ± {np.std(post_vals):.3f}")
    print(f"Change Mean ± SD: {np.mean(differences):.3f} ± {np.std(differences):.3f}")
    print(f"P-value (Wilcoxon): {p_value:.3e}")
    print(f"Cohen's d (Paired): {cohen_d:.3f}")
    print("-" * 50)
    
    return p_value, cohen_d

if __name__ == "__main__":
    print("Loading pre/post data...")
    pre_df, post_df, paired_subjects = load_pre_post_data()
    
    print(f"PRE data shape: {pre_df.shape if pre_df is not None else 'None'}")
    print(f"POST data shape: {post_df.shape if post_df is not None else 'None'}")
    print(f"Paired subjects: {len(paired_subjects) if paired_subjects else 0}")
    
    if paired_subjects:
        print(f"Paired subject IDs: {sorted(paired_subjects)}")
    
    if pre_df is not None and len(pre_df) > 0:
        eeg_features = get_eeg_features(pre_df)
        print(f"Found {len(eeg_features)} EEG features")
        
        # Test analysis on first few features
        print("\nTesting paired analysis on first 5 features:")
        for feature in eeg_features[:5]:
            test_paired_analysis(pre_df, post_df, feature)
    else:
        print("No PRE data available")
    
    if post_df is not None and len(post_df) > 0:
        eeg_features = get_eeg_features(post_df)
        print(f"Found {len(eeg_features)} EEG features in POST data")
    else:
        print("No POST data available")
