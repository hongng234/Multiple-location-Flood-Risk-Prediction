"""Cleaning, chronological split, train-only labelling and the sklearn preprocessor."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate (location, date) rows and sort. No imputation here (done inside the model pipeline).

    Why no imputation: filling NaN with statistics of the whole dataset would leak test-period
    information into training; the pipeline's SimpleImputer learns medians from train folds only.
    """
    df = df.drop_duplicates(subset=["location", "date"]).copy()
    return df.sort_values(["location", "date"]).reset_index(drop=True)


def time_split(df: pd.DataFrame, test_start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological hold-out split.

    Train keeps only rows whose TARGET day is before `test_start`, so no train label
    peeks into the test period (a gap of `horizon` days is created automatically).

    How: compare `target_date` (train) and `date` (test) with the cut-off timestamp.

    Returns:
        (train, test) DataFrames.
    """
    test_start = pd.Timestamp(test_start)
    train = df[df["target_date"] < test_start].reset_index(drop=True)
    test = df[df["date"] >= test_start].reset_index(drop=True)
    # A cut-off outside the data range would silently produce an empty side: stop with a clear message
    if train.empty or test.empty:
        raise ValueError(f"test_start={test_start.date()} leaves an empty train or test set")
    return train, test


def compute_thresholds(train_df: pd.DataFrame, percentile: float, col: str = "q_t") -> dict:
    """Per-location flood threshold = discharge percentile computed on TRAIN rows only.

    How: group train rows by location and take the given percentile of discharge for each
    (nanpercentile ignores missing values).

    Returns:
        {location_name: threshold_m3s}
    """
    return {
        loc: float(np.nanpercentile(group[col], percentile))
        for loc, group in train_df.groupby("location")
    }


def apply_labels(df: pd.DataFrame, thresholds: dict, target: str = "flood_risk") -> pd.DataFrame:
    """Label = 1 if discharge at t + horizon exceeds the location's train threshold.

    How: look up each row's threshold by location, compare it with `q_future`, store 0/1.
    """
    df = df.copy()
    thr = df["location"].map(thresholds)
    # A location without a threshold would get NaN and be silently labelled 0: raise instead
    if thr.isna().any():
        missing = sorted(df.loc[thr.isna(), "location"].unique())
        raise KeyError(f"No threshold for locations: {missing}")
    df[target] = (df["q_future"] > thr).astype(int)
    return df


def build_preprocessor(numeric_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """Impute + scale numeric columns, one-hot encode categoricals (fit happens inside each CV fold).

    How: a ColumnTransformer routes numeric columns through median imputation -> standard
    scaling, and categorical columns (e.g. location) through one-hot encoding.
    `handle_unknown="ignore"` turns an unseen location into an all-zero vector instead of crashing.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    transformers = [("num", numeric, numeric_cols)]
    # Add the one-hot step only when there are categorical columns to encode
    if cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        )
    return ColumnTransformer(transformers)


def time_series_cv_splits(dates: pd.Series, n_splits: int, gap_days: int = 1) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window CV folds split on unique DATES (all locations of a day stay in one fold).

    `gap_days` skips days between train and validation so a train label never overlaps validation.

    How: run sklearn's TimeSeriesSplit on the sorted list of unique days (not on rows), then
    translate each fold's days back to the row positions that carry those days.

    Returns:
        List of (train_row_indices, validation_row_indices), usable as `cv=` in sklearn.
    """
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    unique_days = np.sort(dates.unique())
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap_days)
    folds = []
    # Loop over folds: each gives day positions; isin() maps them to every row (all locations) on those days
    for tr_days, va_days in splitter.split(unique_days):
        tr_idx = np.flatnonzero(dates.isin(unique_days[tr_days]).to_numpy())
        va_idx = np.flatnonzero(dates.isin(unique_days[va_days]).to_numpy())
        folds.append((tr_idx, va_idx))
    return folds
