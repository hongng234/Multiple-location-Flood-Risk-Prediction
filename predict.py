"""Live next-day flood-risk forecast for every configured location.

Usage:
    python predict.py                 # fetch the latest ~30 days and forecast
    python predict.py --config other.yaml
"""
import argparse        # reads the --config command-line flag

import joblib          # loads the trained model bundle saved by main.py
import pandas as pd    # DataFrames

from src.config import load_config                 # reads config.yaml
from src.data_loader import fetch_recent           # downloads the last ~30 days of real data
from src.features import add_threshold_features, build_timeseries_features   # same feature code as training


def predict_latest(bundle: dict, recent: pd.DataFrame) -> pd.DataFrame:
    """Build features from recent raw data and score the LAST complete day of each location.

    What: turns ~30 days of raw observations into one forecast per location for day t + horizon.
    How: rebuild features with the SAME settings used in training (stored in the bundle),
    add threshold-relative features with the TRAIN thresholds, keep the newest row that has a
    full lag history for each location, and ask the saved pipeline for P(flood).

    Args:
        bundle: dict saved by main.py (model, feature_columns, thresholds, horizon_days, features_config).
        recent: raw rows with RAW_COLUMNS for one or more locations.

    Returns:
        One row per location: issue date, forecast date, today's discharge, threshold, probability, alert.
    """
    fc = bundle["features_config"]   # feature settings (lags, rain windows, API decay) used during training
    feats = build_timeseries_features(
        recent, horizon=bundle["horizon_days"], lags=fc["lags"],
        rain_windows=fc["rain_windows"], api_decay=fc["api_decay"],
        drop_incomplete=False,                       # q_future is unknown at inference time
    )
    feats = add_threshold_features(feats, bundle["thresholds"])   # q_t_rel / q_7d_rel with TRAIN thresholds
    # Rows without all discharge lags can't be scored; of the rest, keep the newest per location
    lag_cols = [c for c in feats.columns if c.startswith("q_lag_")]   # e.g. q_lag_1 ... q_lag_7
    latest = feats.dropna(subset=lag_cols + ["q_t"]).groupby("location").tail(1)   # last full day per location

    # The saved Pipeline applies the same imputation/scaling/one-hot as in training
    proba = bundle["model"].predict_proba(latest[bundle["feature_columns"]])[:, 1]   # column 1 = P(flood)
    # Build the human-readable forecast table (one row per location)
    return pd.DataFrame({
        "location": latest["location"].to_numpy(),
        "issued_on": latest["date"].dt.date.to_numpy(),           # last observed day (day t)
        "forecast_for": latest["target_date"].dt.date.to_numpy(), # the day being forecast (t + horizon)
        "discharge_today_m3s": latest["q_t"].round(2).to_numpy(),
        "threshold_m3s": latest["location"].map(bundle["thresholds"]).round(2).to_numpy(),
        "flood_probability": proba.round(3),
        "alert": proba >= 0.5,                                    # True = raise a flood alert
    })


def main(argv=None) -> None:
    """CLI entry point: load config + trained bundle, download recent data, print forecasts."""
    # Read the optional --config flag (defaults to config.yaml)
    parser = argparse.ArgumentParser(description="Forecast tomorrow's flood risk per location")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)

    config = load_config(args.config)                       # settings (locations, model path)
    bundle = joblib.load(config["output"]["model_path"])    # trained model + thresholds saved by main.py

    # Download recent data for every location the model has a threshold for
    frames = []   # one DataFrame of recent days per location
    for loc in config["data"]["locations"]:
        # A location added to config after training has no threshold yet: retrain before forecasting it
        if loc["name"] not in bundle["thresholds"]:
            print(f"Skipping {loc['name']}: not in trained model")
            continue
        df = fetch_recent(loc["latitude"], loc["longitude"], past_days=30)   # needs internet
        frames.append(df.assign(location=loc["name"]))                       # tag rows with the location name

    # Score all locations together and print the forecast table
    result = predict_latest(bundle, pd.concat(frames, ignore_index=True))
    print(f"Model: {bundle['model_name']} (trained until {bundle['trained_until']})")
    print(result.to_string(index=False))


# Run only when executed as a script (not when imported by tests)
if __name__ == "__main__":
    main()
