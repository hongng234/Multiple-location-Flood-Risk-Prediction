"""Feature engineering is per-location and strictly backward-looking."""
import numpy as np    # numeric comparisons with tolerance
import pandas as pd   # DataFrames and date arithmetic
import pytest         # pytest.approx: equality with a small floating-point tolerance

from src.features import build_timeseries_features, feature_columns

# Feature settings shared by the tests that rebuild features themselves
KW = dict(horizon=1, lags=[1, 2, 3, 7], rain_windows=[3, 7], api_decay=0.9)


def test_q_future_is_shifted_discharge(raw_df, feature_df):
    """q_future at (loc, t) must equal raw discharge at (loc, t+1)."""
    # Index raw discharge by (location, date) for direct lookups
    raw = raw_df.set_index(["location", "date"])["river_discharge_m3s"]
    sample = feature_df.sample(50, random_state=0)   # 50 random rows (fixed seed -> same rows every run)
    # For every sampled row, look up the raw value one day later at the same location
    expected = [raw[(loc, d + pd.Timedelta(days=1))] for loc, d in zip(sample["location"], sample["date"])]
    np.testing.assert_allclose(sample["q_future"], expected)


def test_lags_match_raw(raw_df, feature_df):
    """q_lag_7 equals the raw discharge exactly 7 days earlier at the same location."""
    raw = raw_df.set_index(["location", "date"])["river_discharge_m3s"]   # lookup: (location, date) -> discharge
    row = feature_df.iloc[100]                                            # any one feature row
    assert row["q_lag_7"] == pytest.approx(raw[(row["location"], row["date"] - pd.Timedelta(days=7))])


def test_features_ignore_the_future(raw_df):
    """Changing data AFTER a cutoff must not change any feature at or before the cutoff."""
    cutoff = pd.Timestamp("2021-06-01")
    base = build_timeseries_features(raw_df, **KW)   # features from the unchanged data
    # Multiply every value after the cutoff by 10 (a drastic change to the "future")
    perturbed_raw = raw_df.copy()
    future = perturbed_raw["date"] > cutoff   # True for rows after the cutoff
    perturbed_raw.loc[future, ["river_discharge_m3s", "precipitation_mm"]] *= 10
    pert = build_timeseries_features(perturbed_raw, **KW)   # features from the changed data

    cols = feature_columns(base)   # compare only real model inputs
    # Compare rows whose day t is strictly before the cutoff (q_future of the cutoff day may change; it's not a feature)
    a = base[base["date"] < cutoff][cols].reset_index(drop=True)
    b = pert[pert["date"] < cutoff][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)   # identical -> features never looked past day t


def test_no_cross_location_bleed(raw_df, feature_df):
    """The first rows of each location must not use the previous location's values."""
    # Check every location separately
    for loc, g in feature_df.groupby("location"):
        first_day = raw_df[raw_df["location"] == loc]["date"].min()   # first raw day of this location
        # lag-7 requires 7 days of the SAME location history -> first kept row is day 8
        assert g["date"].min() == first_day + pd.Timedelta(days=7)


def test_feature_columns_exclude_ids_and_label_source(feature_df):
    """feature_columns never returns ids, the label or the label source, but keeps real features."""
    cols = feature_columns(feature_df.assign(flood_risk=0))   # add a dummy label column to check it is excluded
    # None of the forbidden columns may appear
    for bad in ["date", "location", "q_future", "target_date", "flood_risk"]:
        assert bad not in cols
    assert {"q_t", "rain_7d", "api", "doy_sin"} <= set(cols)   # `<=` on sets = "is a subset of"


def test_inference_mode_keeps_last_day(raw_df):
    """With drop_incomplete=False the newest day is kept (needed for live forecasts) and its q_future is NaN."""
    out = build_timeseries_features(raw_df, drop_incomplete=False, **KW)   # inference mode
    last = out.groupby("location")["date"].max()                          # newest day kept per location
    assert (last == raw_df["date"].max()).all()                           # equals the newest raw day
    assert out.groupby("location").tail(1)["q_future"].isna().all()       # its future value is unknown
