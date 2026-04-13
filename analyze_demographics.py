import os
import pandas as pd
import numpy as np
import glob

def analyze_assy(folder_path):
    """
    Analyzes the ASSY dataset in the given folder.
    Expects a CSV file named 'ASSy_demographics.csv'.
    """
    print("\n--- Analyzing Project: ASSY ---")
    
    # search for the specific csv file
    search_path = os.path.join(folder_path, "ASSy_demographics.csv")
    files = glob.glob(search_path)
    
    if not files:
        print(f"No demographics file found in {folder_path}")
        return

    file_path = files[0]
    try:
        # Load CSV without header based on previous inspection
        # Columns: ID, Age, Sex
        df = pd.read_csv(file_path, header=None, names=['ID', 'Age', 'Sex'])
        
        # Select numeric columns for analysis
        # ID is skipped. Age and Sex (if 0/1) are numeric-ish.
        # Although Sex is categorical, calculating mean gives the proportion.
        # But standard deviation for Sex is not very descriptive in common clinical tables, 
        # but the request asks for Mean and Std.
        
        numeric_cols = ['Age', 'Sex']
        
        stats = []
        for col in numeric_cols:
            if col in df.columns:
                mu = df[col].mean()
                sigma = df[col].std()
                stats.append({'Feature': col, 'Mean': mu, 'Std': sigma, 'N': df[col].count()})
        
        stats_df = pd.DataFrame(stats)
        print_stats(stats_df)
        
    except Exception as e:
        print(f"Error analyzing ASSY data: {e}")

def analyze_cobrad(folder_path):
    """
    Analyzes the COBRAD dataset (EDF folder).
    Expects an Excel file containing 'clinical'.
    """
    print("\n--- Analyzing Project: COBRAD (EDF) ---")
    
    # Search for xlsx files, prioritizing the one we saw earlier
    search_pattern = os.path.join(folder_path, "*clinical*.xlsx")
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"No clinical/demographics Excel file found in {folder_path}")
        return

    # Use the first match
    file_path = files[0]
    try:
        df = pd.read_excel(file_path)
        
        # Adjust 'sex' column to be 0 and 1 instead of 1 and 2
        if 'sex' in df.columns:
            df['sex'] = df['sex'] - 1

        # Identify numeric columns
        # We also want to filter out IDs or unstructured data if possible, but 
        # simplest is to select numbers.
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Exclude 'record_id' or 'Has_eeg' if they are just identifiers/flags
        # 'Has_eeg' seems like a flag (all 1?), but 'record_id' implies ID.
        # Let's inspect columns to exclude obvious IDs if they are numeric
        exclude_cols = ['record_id', 'Has_eeg', 'ID'] 
        
        stats = []
        for col in numeric_df.columns:
            if col not in exclude_cols:
                mu = numeric_df[col].mean()
                sigma = numeric_df[col].std()
                stats.append({'Feature': col, 'Mean': mu, 'Std': sigma, 'N': numeric_df[col].count()})
        
        stats_df = pd.DataFrame(stats)
        print_stats(stats_df)
        
    except Exception as e:
        print(f"Error analyzing COBRAD data: {e}")

def print_stats(stats_df):
    """
    Prints the statistics in a formatted way.
    """
    if stats_df.empty:
        print("No numeric features found.")
        return

    # Determine padding for alignment
    max_len = stats_df['Feature'].apply(str).map(len).max()
    padding = max(max_len, 10) + 2
    
    print(f"{'Feature'.ljust(padding)} | {'Mean':>10} | {'Std Dev':>10} | {'Count':>5}")
    print("-" * (padding + 35))
    
    for _, row in stats_df.iterrows():
        feature = str(row['Feature']).ljust(padding)
        mean_val = f"{row['Mean']:.2f}"
        std_val = f"{row['Std']:.2f}"
        count_val = str(int(row['N']))
        print(f"{feature} | {mean_val:>10} | {std_val:>10} | {count_val:>5}")

def main():
    base_dir = "."  # Current directory, or absolute path if needed
    
    # Define paths based on user request
    assy_path = os.path.join(base_dir, "EDF_Format", "ASSY")
    edf_path = os.path.join(base_dir, "EDF_Format", "EDF")
    
    # Analyze ASSY
    if os.path.exists(assy_path):
        analyze_assy(assy_path)
    else:
        print(f"Directory not found: {assy_path}")
        
    # Analyze EDF (COBRAD)
    if os.path.exists(edf_path):
        analyze_cobrad(edf_path)
    else:
        print(f"Directory not found: {edf_path}")

if __name__ == "__main__":
    main()
