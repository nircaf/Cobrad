"""Shared channel canonicalisation for per-electrode figures.

Channel labels in this corpus mix monopolar ("F3", case-inconsistent as
"Fp2"/"FP2") and mastoid/ear-referenced bipolar ("F3-M2", "F8-A1")
derivations. Canonicalising to the scalp-side 10-20 site, case-insensitively,
avoids two bugs: (1) case duplicates ("Fp2-A1" and "FP2-A1" counted as
different channels), and (2) the reference electrode itself (M1/M2/A1/A2)
being treated as a scalp recording site when it appears alone.
"""
import pandas as pd

CHANNEL_ORDER = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "T7", "C3", "Cz", "C4", "T4", "T8",
    "T5", "P7", "P3", "Pz", "P4", "T6", "P8",
    "O1", "Oz", "O2",
]
ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
CANON = {name.upper(): name for name in CHANNEL_ORDER}
for _legacy, _modern in ALIASES.items():
    CANON[_legacy.upper()] = _modern


def canonicalize(channel_series: pd.Series) -> pd.Series:
    """Map raw channel labels to their canonical scalp-side 10-20 site.

    Non-scalp labels (bare references like "M1"/"A1"/"A2", or anything not
    in the 10-20/10-10 set) map to NaN and should be dropped by the caller.
    """
    return channel_series.str.split("-").str[0].str.upper().map(CANON)


def filter_min_coverage(df: pd.DataFrame, patient_col: str, canon_col: str, min_frac: float = 0.5) -> pd.DataFrame:
    """Keep only rows whose canonical channel appears in >= min_frac of all
    patients present in df -- so a channel recorded in a handful of patients
    doesn't get plotted alongside ones covering the whole cohort."""
    total_patients = df[patient_col].nunique()
    coverage = df.groupby(canon_col)[patient_col].nunique() / total_patients
    keep = coverage[coverage >= min_frac].index
    return df[df[canon_col].isin(keep)].copy()
