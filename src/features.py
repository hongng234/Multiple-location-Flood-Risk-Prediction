import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Perform feature engineering; automatically skip a feature if the input column is missing."""
    df = df.copy()

    # --- Rainfall features (có ở cả 2 nguồn) ---
    if "rainfall_24h_mm" in df.columns and "rainfall_72h_mm" in df.columns:
        df["rainfall_ratio_72_24"] = (
            df["rainfall_72h_mm"] / (df["rainfall_24h_mm"] + 1e-6)
        ).clip(0, 10)

    # --- Discharge features (có ở cả 2 nguồn) ---
    if "river_discharge_m3s" in df.columns and "rainfall_72h_mm" in df.columns:
        df["discharge_per_rain"] = (
            df["river_discharge_m3s"] / (df["rainfall_72h_mm"] + 1e-6)
        ).clip(0, 50)

    # --- Water level: Create only if a water column is present, or use a proxy based on discharge ---
    if "water_level_m" in df.columns:
        df["water_level_excess"] = (df["water_level_m"] - 3.0).clip(lower=0)
    elif "river_discharge_m3s" in df.columns:
        # Proxy: Standardization of discharge on a 0–6 scale (simulated water level)
        df["water_level_m"] = (
            df["river_discharge_m3s"] / df["river_discharge_m3s"].max() * 6
        )
        df["water_level_excess"] = (df["water_level_m"] - 3.0).clip(lower=0)

    # --- Composite index ---
    required_for_index = ["rainfall_24h_mm", "water_level_m", "soil_moisture", "urbanization_index"]
    if all(col in df.columns for col in required_for_index):
        df["flood_pressure_index"] = (
            0.35 * (df["rainfall_24h_mm"] / df["rainfall_24h_mm"].max())
            + 0.30 * (df["water_level_m"] / df["water_level_m"].max())
            + 0.20 * df["soil_moisture"]
            + 0.15 * df["urbanization_index"]
        )

    # --- Drainage deficit ---
    if "river_discharge_m3s" in df.columns and "drainage_capacity" in df.columns:
        df["drainage_deficit"] = (
            df["river_discharge_m3s"] / (df["drainage_capacity"] + 1e-6)
        ).clip(0, 20)

    # --- Low elevation flag ---
    if "elevation_m" in df.columns:
        df["low_elevation"] = (df["elevation_m"] < 20).astype(int)

    return df