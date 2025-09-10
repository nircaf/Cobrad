import pandas as pd
import numpy as np

# Load the data
print("Loading EDF.csv...")
df = pd.read_csv('EDF.csv')

print(f"Original data shape: {df.shape}")
print(f"Number of unique patients: {df['patient_number'].nunique()}")

# Group by patient_number and calculate mean of all numeric columns
print("\nGrouping by patient_number and calculating means...")

# Select only numeric columns for mean calculation
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Group by patient_number and calculate mean
patient_means = df.groupby('patient_number')[numeric_cols].mean()

print(f"Grouped data shape: {patient_means.shape}")
print(f"Columns included: {len(numeric_cols)}")

# Display basic info about the grouped data
print("\nFirst few rows of grouped data:")
print(patient_means.head())

print("\nSummary statistics of grouped data:")
print(patient_means.describe())

# Save the grouped data
output_file = 'EDF_patient_means.csv'
patient_means.to_csv(output_file)
print(f"\nGrouped data saved to: {output_file}")

# Show which columns were excluded (non-numeric)
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    print(f"\nNon-numeric columns excluded from mean calculation: {non_numeric_cols}")

# Show some key statistics
print(f"\nKey statistics:")
print(f"Number of patients: {len(patient_means)}")
print(f"Number of measurements per patient: {len(numeric_cols)}")
print(f"Missing values in grouped data: {patient_means.isnull().sum().sum()}")

# Show some example measurements
print(f"\nExample measurements (first 10 columns):")
example_cols = numeric_cols[:10]
print(patient_means[example_cols].head())
