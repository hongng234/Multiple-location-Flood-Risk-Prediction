"""Leakage-safe time-series feature engineering.

Rule: every feature for row (location, day t) uses ONLY values observed on day t or earlier,
and is computed within its own location (groupby) so series never bleed into each other.
The only forward-looking column is `q_future` (discharge at t + horizon), which is the label
SOURCE — it is never used as a feature.
"""
import numpy as np
import pandas as pd

# Columns that must never be fed to a model (identifiers, label source, label)
NON_FEATURE_COLUMNS = {"date", "location", "q_future", "target_date"}


def build_timeseries_features(
    df: pd.DataFrame,
    horizon: int = 1,
    lags: list[int] = (1, 2, 3, 7),
    rain_windows: list[int] = (3, 7, 14),
    api_decay: float = 0.9,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """Create per-location backward-looking features plus the forward target source `q_future`.

    What: turns raw daily rows into model inputs for "predict day t + horizon from data up to day t".

    How:
      1. Sort by (location, date) and group by location, so every rolling/shift stays inside one river.
      2. Rainfall: today's rain, rolling sums over `rain_windows`, and the Antecedent Precipitation Index.
      3. Discharge: today's value, past lags, rolling means and the 1-day change (rising/falling).
      4. Temperature + day-of-year encoded as sin/cos (season).
      5. `q_future` = discharge shifted BACK by `horizon` (the value `horizon` days ahead) -> label source.

    Args:
        df: raw data with RAW_COLUMNS.
        horizon: forecast lead time in days.
        lags: discharge lags (days) to include as features.
        rain_windows: window lengths (days) for rolling rainfall sums.
        api_decay: daily decay factor k of the API (0.9 = 10% of "wetness" drains per day).
        drop_incomplete: True drops rows lacking full lag history or a future value (training);
            use False at inference time, where `q_future` is unknown by definition.

    Returns:
        DataFrame with date, location, all feature columns, q_future and target_date.
    """
    # Sort so that "previous row" == "previous day of the same location", then group by location
    df = df.sort_values(["location", "date"]).reset_index(drop=True).copy()
    g = df.groupby("location", group_keys=False)
    out = df[["date", "location"]].copy()

    # --- Rainfall (observed up to day t) ---
    out["rain_1d"] = df["precipitation_mm"]
    # Loop over window lengths: one rolling-sum column per window (e.g. rain_3d, rain_7d);
    # `w=w` freezes the current window inside the lambda; min_periods=1 avoids NaN at series start
    for w in rain_windows:
        out[f"rain_{w}d"] = g["precipitation_mm"].transform(lambda s, w=w: s.rolling(w, min_periods=1).sum())
    # Antecedent Precipitation Index: API_t = P_t + k * API_{t-1}  (ewm trick, alpha = 1 - k)
    # (ewm with adjust=False computes y_t = k*y_{t-1} + (1-k)*P_t; dividing by (1-k) gives exactly API_t)
    out["api"] = g["precipitation_mm"].transform(
        lambda s: s.fillna(0).ewm(alpha=1 - api_decay, adjust=False).mean() / (1 - api_decay)
    )

    # --- Discharge observed up to day t (legit for forecasting t + horizon) ---
    out["q_t"] = df["river_discharge_m3s"]
    # Loop over lags: shift(k) moves each location's series down k rows -> value from k days ago
    for k in lags:
        out[f"q_lag_{k}"] = g["river_discharge_m3s"].shift(k)
    out["q_3d_mean"] = g["river_discharge_m3s"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    out["q_7d_mean"] = g["river_discharge_m3s"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    out["q_diff_1d"] = out["q_t"] - g["river_discharge_m3s"].shift(1)    # rising / falling limb

    # --- Weather + seasonality ---
    out["temperature_mean_c"] = df["temperature_mean_c"]
    # Encode day-of-year on a circle so Dec 31 and Jan 1 are close (a raw 1..365 number would not be)
    doy = df["date"].dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # --- Target source: discharge `horizon` days ahead (NOT a feature) ---
    # shift(-horizon) pulls the future value up to row t; target_date records which day it belongs to
    out["q_future"] = g["river_discharge_m3s"].shift(-horizon)
    out["target_date"] = out["date"] + pd.Timedelta(days=horizon)

    # For training, drop the first rows of each location (incomplete lag history) and the last
    # `horizon` rows (no future value to label); inference keeps them because q_future is unknown anyway
    if drop_incomplete:
        lag_cols = [f"q_lag_{k}" for k in lags]
        out = out.dropna(subset=lag_cols + ["q_t", "q_future"]).reset_index(drop=True)
    return out


def add_threshold_features(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Add scale-free discharge features: discharge divided by the location's TRAIN threshold.

    This lets one model compare rivers of very different size (e.g. Mekong vs a small stream).

    How: map each row's location to its threshold, then divide today's discharge and the
    7-day mean by it. A value > 1 means "already above flood level".
    """
    df = df.copy()
    thr = df["location"].map(thresholds)
    df["q_t_rel"] = df["q_t"] / thr
    df["q_7d_rel"] = df["q_7d_mean"] / thr
    return df


def feature_columns(df: pd.DataFrame, target: str = "flood_risk") -> list[str]:
    """List numeric model-input columns (excludes ids, the label and its source).

    How: take every numeric column and remove NON_FEATURE_COLUMNS plus the target, so a new
    feature added in `build_timeseries_features` is picked up automatically.
    """
    excluded = NON_FEATURE_COLUMNS | {target}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in excluded]
