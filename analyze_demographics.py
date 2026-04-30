import os
import pandas as pd
import numpy as np
import glob
from scipy import stats

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

def load_excluded_ids(base_dir):
    excl_path = os.path.join(base_dir, "pickles_sleep_stage", "excluded_patients.csv")
    if not os.path.exists(excl_path):
        print(f"Warning: excluded_patients.csv not found at {excl_path}")
        return set()
    excl = pd.read_csv(excl_path)
    return set(excl['patient_id'].str.upper())


def load_controls(base_dir, excluded_ids):
    path = os.path.join(base_dir, "EDF_Format", "Berkeley_data", "df_all_demographics_young.xlsx")
    df = pd.read_excel(path)
    df = df[~df['Subj'].str.upper().isin(excluded_ids)].copy()
    # Normalize sex: 1=female, 0=male
    df['sex_binary'] = (df['Gender'] == 'F').astype(int)
    df = df.rename(columns={'Age': 'age'})
    df['group'] = 'Control'
    return df[['Subj', 'age', 'sex_binary', 'group']]


def load_patients(base_dir, excluded_ids):
    path = os.path.join(base_dir, "EDF_Format", "EDF", "COBRAD_clinical_24022025.xlsx")
    df = pd.read_excel(path)
    # COBRAD record_ids are "345-xxx", excluded uses "0345-xxx" — strip leading zero
    df['id_upper'] = df['record_id'].str.upper()
    excl_cobrad = {x.lstrip('0') for x in excluded_ids if '345' in x}
    df = df[~df['id_upper'].isin({x.upper() for x in excl_cobrad})].copy()
    # sex: 1=male, 2=female → binary: 1=female, 0=male
    df['sex_binary'] = (df['sex'] == 2).astype(int)
    df['group'] = 'COBRAD'
    return df[['record_id', 'age', 'sex_binary', 'group',
               'education_by_diploma', 'mmse', 'moca', 'cdr_sum_of_boxes']]


def make_table_one(base_dir):
    print("\n" + "=" * 65)
    print("TABLE 1 — Demographics: Controls vs COBRAD Patients")
    print("=" * 65)

    excluded_ids = load_excluded_ids(base_dir)
    ctrl = load_controls(base_dir, excluded_ids)
    pat  = load_patients(base_dir, excluded_ids)

    # ── helper to format continuous variables ──────────────────────
    def cont_row(label, ctrl_vals, pat_vals):
        ctrl_vals = ctrl_vals.dropna()
        pat_vals  = pat_vals.dropna()
        _, p = stats.ttest_ind(ctrl_vals, pat_vals, equal_var=False)
        return {
            'Variable':  label,
            'Controls':  f"{ctrl_vals.mean():.1f} ± {ctrl_vals.std():.1f}",
            'Patients':  f"{pat_vals.mean():.1f} ± {pat_vals.std():.1f}",
            'p-value':   f"{p:.3f}" if p >= 0.001 else "<0.001",
        }

    # ── helper to format categorical variables ─────────────────────
    def cat_row(label, ctrl_vals, pat_vals):
        ctrl_vals = ctrl_vals.dropna()
        pat_vals  = pat_vals.dropna()
        c_n = int(ctrl_vals.sum()); c_tot = len(ctrl_vals)
        p_n = int(pat_vals.sum());  p_tot = len(pat_vals)
        table = [[c_n, c_tot - c_n], [p_n, p_tot - p_n]]
        _, p, _, _ = stats.chi2_contingency(table)
        return {
            'Variable': label,
            'Controls': f"{c_n} ({100*c_n/c_tot:.1f}%)",
            'Patients': f"{p_n} ({100*p_n/p_tot:.1f}%)",
            'p-value':  f"{p:.3f}" if p >= 0.001 else "<0.001",
        }

    # ── patients-only helper ───────────────────────────────────────
    def pat_only_row(label, pat_vals):
        pat_vals = pat_vals.dropna()
        return {
            'Variable': label,
            'Controls': '—',
            'Patients': f"{pat_vals.mean():.1f} ± {pat_vals.std():.1f}",
            'p-value':  'N/A',
        }

    rows = [
        {'Variable': 'N', 'Controls': str(len(ctrl)), 'Patients': str(len(pat)), 'p-value': ''},
        cont_row('Age (years)',      ctrl['age'],        pat['age']),
        cat_row( 'Female, n (%)',    ctrl['sex_binary'], pat['sex_binary']),
        pat_only_row('Education (diploma level)', pat['education_by_diploma']),
        pat_only_row('MMSE',         pat['mmse']),
        pat_only_row('MoCA',         pat['moca']),
        pat_only_row('CDR Sum of Boxes', pat['cdr_sum_of_boxes']),
    ]

    tbl = pd.DataFrame(rows, columns=['Variable', 'Controls', 'Patients', 'p-value'])

    # ── print formatted table ──────────────────────────────────────
    col_w = [28, 20, 20, 10]
    header = (f"{'Variable':<{col_w[0]}} {'Controls':>{col_w[1]}} "
              f"{'Patients':>{col_w[2]}} {'p-value':>{col_w[3]}}")
    print(header)
    print("-" * sum(col_w + [3]))
    for _, row in tbl.iterrows():
        print(f"{row['Variable']:<{col_w[0]}} {row['Controls']:>{col_w[1]}} "
              f"{row['Patients']:>{col_w[2]}} {row['p-value']:>{col_w[3]}}")
    print("-" * sum(col_w + [3]))
    print("Continuous: mean ± SD (unpaired t-test); Categorical: n (%) (χ²)")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    assy_path = os.path.join(base_dir, "EDF_Format", "Berkeley_data")
    edf_path  = os.path.join(base_dir, "EDF_Format", "EDF")

    if os.path.exists(edf_path):
        analyze_cobrad(edf_path)
    else:
        print(f"Directory not found: {edf_path}")

    make_table_one(base_dir)


if __name__ == "__main__":
    main()
