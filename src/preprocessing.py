"""Cleaning, chronological split, train-only labelling and the sklearn preprocessor."""
import numpy as np    # percentiles and index arrays
import pandas as pd   # DataFrames and timestamps
from sklearn.compose import ColumnTransformer          # applies different steps to different columns
from sklearn.impute import SimpleImputer               # fills missing values (NaN), e.g. with the median
from sklearn.model_selection import TimeSeriesSplit    # forward-in-time CV folds (no shuffling)
from sklearn.pipeline import Pipeline                  # chains steps: impute -> scale -> ...
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # text category -> 0/1 columns; rescale to mean 0, std 1


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate (location, date) rows and sort. No imputation here (done inside the model pipeline).

    Why no imputation: filling NaN with statistics of the whole dataset would leak test-period
    information into training; the pipeline's SimpleImputer learns medians from train folds only.
    """
    # Keep only the first row for each (location, day) pair
    df = df.drop_duplicates(subset=["location", "date"]).copy()
    # Order rows by location, then by day, and renumber them 0..n-1
    return df.sort_values(["location", "date"]).reset_index(drop=True)


def time_split(df: pd.DataFrame, test_start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological hold-out split.

    Train keeps only rows whose TARGET day is before `test_start`, so no train label
    peeks into the test period (a gap of `horizon` days is created automatically).

    How: compare `target_date` (train) and `date` (test) with the cut-off timestamp.

    Returns:
        (train, test) DataFrames.
    """
    test_start = pd.Timestamp(test_start)                               # "2024-01-01" text -> timestamp
    train = df[df["target_date"] < test_start].reset_index(drop=True)   # forecasts whose answer day is before the cut-off
    test = df[df["date"] >= test_start].reset_index(drop=True)          # forecasts issued on/after the cut-off
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
    # Dict comprehension: loop over each location's train rows and store its percentile value
    return {
        loc: float(np.nanpercentile(group[col], percentile))   # e.g. P90 = 90% of train days are below this
        for loc, group in train_df.groupby("location")
    }


def apply_labels(df: pd.DataFrame, thresholds: dict, target: str = "flood_risk") -> pd.DataFrame:
    """Label = 1 if discharge at t + horizon exceeds the location's train threshold.

    How: look up each row's threshold by location, compare it with `q_future`, store 0/1.
    """
    df = df.copy()                            # don't modify the caller's table
    thr = df["location"].map(thresholds)      # threshold for each row, looked up by location name
    # A location without a threshold would get NaN and be silently labelled 0: raise instead
    if thr.isna().any():
        missing = sorted(df.loc[thr.isna(), "location"].unique())   # names of the locations lacking a threshold
        raise KeyError(f"No threshold for locations: {missing}")
    # True/False comparison converted to 1/0: 1 = discharge on the target day is above flood level
    df[target] = (df["q_future"] > thr).astype(int)
    return df


def build_preprocessor(numeric_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """Impute + scale numeric columns, one-hot encode categoricals (fit happens inside each CV fold).

    How: a ColumnTransformer routes numeric columns through median imputation -> standard
    scaling, and categorical columns (e.g. location) through one-hot encoding.
    `handle_unknown="ignore"` turns an unseen location into an all-zero vector instead of crashing.
    """
    # Steps for numeric columns, applied in order
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),   # replace NaN with the column median (learned from train data)
        ("scale", StandardScaler()),                    # put all features on the same scale (mean 0, std 1)
    ])
    # List of (name, steps, columns) triples for the ColumnTransformer
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
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)   # row i of the data -> its date
    unique_days = np.sort(dates.unique())                             # each calendar day once, oldest first
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap_days)       # each fold: older days train, newer days validate
    folds = []                                                        # collected (train_rows, val_rows) pairs
    # Loop over folds: each gives day positions; isin() maps them to every row (all locations) on those days
    for tr_days, va_days in splitter.split(unique_days):
        tr_idx = np.flatnonzero(dates.isin(unique_days[tr_days]).to_numpy())
        va_idx = np.flatnonzero(dates.isin(unique_days[va_days]).to_numpy())
        folds.append((tr_idx, va_idx))
    return folds
