# EEG Data Analysis Dashboard

A comprehensive Streamlit dashboard for analyzing EEG data comparing patients and controls.

## Features

### 📊 Data Overview
- Summary statistics for both patient and control groups
- Key metrics comparison with percentage differences
- Data quality indicators

### 📈 Interactive Visualizations
- Distribution plots (histograms with box plots)
- Group comparison box plots
- Channel-specific heatmaps
- Interactive measurement selection

### 🔬 Statistical Analysis
- Automated statistical testing (t-test or Mann-Whitney U based on normality)
- Effect size calculations (Cohen's d)
- Significance testing with multiple comparison correction
- Summary of significant differences

### 📋 Raw Data Access
- Filterable data tables
- Column selection
- CSV export functionality
- Group-specific data views

### 📚 Measurement Guide
- Detailed explanations of all EEG measurements
- Clinical interpretation guidelines
- Data collection specifications

## Installation

1. Install required packages:
```bash
pip install -r dashboard_requirements.txt
```

2. Ensure your data files are in the same directory:
   - `EDF.csv` (patient data)
   - `EDF_controls.csv` (control data)

## Running the Dashboard

```bash
streamlit run eeg_dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Data Structure

The dashboard expects CSV files with the following structure:
- Patient data: `EDF.csv` (1442 records, 525 columns)
- Control data: `EDF_controls.csv` (197 records, 406 columns)

### Key Measurements Included:
- **Basic Statistics**: mean, median, std, min, max
- **Power Spectral Density**: delta, theta, alpha, beta, gamma power
- **Frequency Analysis**: FFT, peak frequencies, dominant frequencies
- **Signal Characteristics**: skewness, kurtosis, PSWE metrics

## Usage Tips

1. **Overview Tab**: Start here to understand your data structure and key differences
2. **Distributions Tab**: Explore individual measurements with interactive plots
3. **Statistical Analysis Tab**: Identify significant differences between groups
4. **Raw Data Tab**: Access and export specific data subsets
5. **Measurement Guide Tab**: Understand what each measurement means

## Statistical Methods

- **Normality Testing**: Shapiro-Wilk test
- **Group Comparison**: Independent t-test (normal data) or Mann-Whitney U test (non-normal)
- **Effect Size**: Cohen's d for practical significance
- **Significance Levels**: * p<0.05, ** p<0.01, *** p<0.001

## Presentation Features

- Professional styling with custom CSS
- Responsive design for different screen sizes
- Interactive elements for engaging presentations
- Export capabilities for reports and publications

## Troubleshooting

If you encounter issues:
1. Ensure all required packages are installed
2. Check that CSV files are in the correct location
3. Verify data file formats match expected structure
4. Check console for error messages

## Contact

For questions or issues with the dashboard, please refer to the measurement guide within the application or check the data structure requirements.
