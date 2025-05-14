import streamlit as st
import pandas as pd
import numpy as np
import glob
import re
from scipy import stats
import matplotlib.pyplot as plt


def analyze_eeg_sessions(file_objs):
    """
    Load EEG CSVs, identify earliest and latest sessions, compute stats, and generate plots.
    """
    # Parse files and dates
    data_dict = {}
    for f in file_objs:
        name = f.name if hasattr(f, 'name') else getattr(f, 'filename', 'unknown.csv')
        m = re.search(r"(\d{8})", name)
        date = pd.to_datetime(m.group(1), format="%Y%m%d") if m else pd.NaT
        df = pd.read_csv(f)
        data_dict[date] = df

    # Identify before and after sessions
    dates = sorted(d for d in data_dict.keys() if not pd.isna(d))
    before_date, after_date = dates[0], dates[-1]
    df_before, df_after = data_dict[before_date], data_dict[after_date]

    # Display session info
    st.write(f"**Before session**: {before_date.date()}, **After session**: {after_date.date()}")

    # Select numeric columns present in both
    common_cols = [
        col for col in df_before.columns if col in df_after.columns
        and np.issubdtype(df_before[col].dtype, np.number)
        and np.issubdtype(df_after[col].dtype, np.number)
    ]

    results = []
    for col in common_cols:
        a = df_before[col].dropna().reset_index(drop=True)
        b = df_after[col].dropna().reset_index(drop=True)
        n = min(len(a), len(b))
        a, b = a.iloc[:n], b.iloc[:n]
        diff = b - a

        # Statistical computations
        t_stat, p_val = stats.ttest_rel(b, a)
        mean_before, mean_after = a.mean(), b.mean()
        mean_diff = mean_after - mean_before
        perc_change = (mean_diff / mean_before * 100) if mean_before != 0 else np.nan
        cohen_d = mean_diff / diff.std(ddof=1) if diff.std(ddof=1) != 0 else np.nan

        # Record results
        results.append({
            'Channel': col,
            'Mean Before': mean_before,
            'Mean After': mean_after,
            'Mean Difference': mean_diff,
            '% Change': perc_change,
            't-statistic': t_stat,
            'p-value': p_val,
            "Cohen's d": cohen_d
        })

        # Plot boxplot
        fig, ax = plt.subplots()
        ax.boxplot([a, b], labels=["Before", "After"])
        ax.set_title(f"{col}: Before vs After")
        ax.set_ylabel(col)
        st.pyplot(fig)
        st.markdown(f"**Figure**: Distribution of **{col}** values before and after.")

    # Percent change bar chart
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        st.subheader("Percent Change by Channel")
        fig2, ax2 = plt.subplots()
        ax2.bar(res_df["Channel"], res_df["% Change"])
        ax2.axhline(0, linewidth=0.8)
        ax2.set_ylabel("% Change")
        ax2.set_title("Percent Change in Mean Values")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig2)
        st.markdown("**Figure**: Bar chart showing percent change in mean EEG values.")

    # Display and interpret results
    st.subheader("Statistical Results")
    st.dataframe(res_df)

    st.subheader("Summary Interpretation")
    for _, row in res_df.iterrows():
        significance = "significant (p < 0.05)" if row['p-value'] < 0.05 else "not significant"
        st.markdown(
            f"- **{row['Channel']}**: mean before = {row['Mean Before']:.3f}, "
            f"mean after = {row['Mean After']:.3f}, p = {row['p-value']:.4f} ({significance}), "
            f"Cohen's d = {row['Cohen d']:.2f}"
        )


def main():
    st.title("EEG Before-and-After Analysis")
    st.markdown(
        """
        Upload EEG CSV files with dates in filenames; the app auto-detects files in the working directory if none are uploaded.
        Sessions are compared using paired statistical tests and visualized.
        """
    )

    # File input
    uploaded = st.file_uploader(
        "Upload EEG CSV files (multiple)", type="csv", accept_multiple_files=True
    )
    file_objs = uploaded if uploaded else [open(fn, 'rb') for fn in glob.glob("*.csv")]

    if len(file_objs) < 2:
        st.warning("Please provide at least two EEG CSV files.")
    else:
        analyze_eeg_sessions(file_objs)


if __name__ == "__main__":
    main()