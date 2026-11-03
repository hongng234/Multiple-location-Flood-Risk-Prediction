import numpy as np
import pandas as pd

from src.data_loader import fetch_open_meteo_weather, fetch_open_meteo_discharge


# ------------------------------------------------------------------
# 1. SYNTHETIC DATA
# ------------------------------------------------------------------
def generate_synthetic_flood_data(
    n_samples: int = 5000, random_state: int = 42
) -> pd.DataFrame:
    """Generate synthetic flood data for training the pipeline."""
    rng = np.random.default_rng(random_state)

    rainfall_24h = rng.gamma(shape=2.0, scale=25.0, size=n_samples)
    rainfall_72h = rainfall_24h * rng.uniform(1.2, 3.5, n_samples)
    river_discharge = 50 + rainfall_72h * rng.uniform(0.8, 2.0, n_samples)
    water_level = 1.0 + river_discharge / rng.uniform(400, 900, n_samples)
    soil_moisture = np.clip(rng.normal(0.45, 0.15, n_samples), 0.05, 0.95)
    elevation = rng.uniform(2, 120, n_samples)
    slope = rng.uniform(0.5, 25, n_samples)
    drainage_capacity = rng.uniform(20, 200, n_samples)
    urbanization = rng.uniform(0, 1, n_samples)

    risk_score = (
        0.9 * (rainfall_24h / 100)
        + 1.1 * (rainfall_72h / 250)
        + 1.3 * (water_level / 6)
        + 0.8 * soil_moisture
        + 0.5 * urbanization
        - 0.8 * (elevation / 120)
        - 0.6 * (drainage_capacity / 200)
        - 0.3 * (slope / 25)
    )

    prob = 1 / (1 + np.exp(-(risk_score - 1.6) * 2.2))
    flood_risk = rng.binomial(1, prob)

    return pd.DataFrame(
        {
            "rainfall_24h_mm": rainfall_24h.round(2),
            "rainfall_72h_mm": rainfall_72h.round(2),
            "river_discharge_m3s": river_discharge.round(2),
            "water_level_m": water_level.round(3),
            "soil_moisture": soil_moisture.round(3),
            "elevation_m": elevation.round(2),
            "slope_deg": slope.round(2),
            "drainage_capacity": drainage_capacity.round(2),
            "urbanization_index": urbanization.round(3),
            "flood_risk": flood_risk,
        }
    )


# ------------------------------------------------------------------
# 2. OPEN-METEO DATA (real rainfall + discharge → label by threshold)
# ------------------------------------------------------------------
def build_dataset_from_open_meteo(
    locations: list[dict],
    start_date: str,
    end_date: str,
    discharge_threshold_percentile: int = 90,
) -> pd.DataFrame:
    """
    Load data for MULTIPLE locations, add the location column,
    and merge everything into a single DataFrame.
    """
    all_dfs = []

    for loc in locations:
        print(f"Fetching data for: {loc['name']} "
              f"({loc['latitude']}, {loc['longitude']})")

        weather = fetch_open_meteo_weather(
            loc["latitude"], loc["longitude"], start_date, end_date
        )
        discharge = fetch_open_meteo_discharge(
            loc["latitude"], loc["longitude"], start_date, end_date
        )

        df = weather.merge(discharge, on="date", how="inner")
        df = df.sort_values("date").reset_index(drop=True)
        df["location"] = loc["name"]          # ← gắn tên địa điểm

        # Rolling features
        df["rainfall_24h_mm"] = df["precipitation_mm"]
        df["rainfall_72h_mm"] = (
            df["precipitation_mm"].rolling(window=3, min_periods=1).sum()
        )
        df["rainfall_7d_mm"] = (
            df["precipitation_mm"].rolling(window=7, min_periods=1).sum()
        )
        df["discharge_3d_mean"] = (
            df["river_discharge_m3s"].rolling(window=3, min_periods=1).mean()
        )
        df["discharge_7d_mean"] = (
            df["river_discharge_m3s"].rolling(window=7, min_periods=1).mean()
        )

        # Label based on the INDIVIDUAL percentile for each location
        threshold = np.percentile(
            df["river_discharge_m3s"].dropna(), discharge_threshold_percentile
        )
        df["flood_risk"] = (df["river_discharge_m3s"] > threshold).astype(int)

        # Topographic Features (placeholder — can be replaced with an actual DEM)
        df["elevation_m"] = 15.0
        df["slope_deg"] = 5.0
        df["drainage_capacity"] = 100.0
        df["urbanization_index"] = 0.5
        df["soil_moisture"] = 0.5

        all_dfs.append(df)
        print(f"  → {len(df)} rows, flood_risk=1: {df['flood_risk'].sum()} "
              f"({df['flood_risk'].mean():.1%}), threshold={threshold:.2f}")

    # Combine all locations into a single dataframe
    combined = pd.concat(all_dfs, ignore_index=True)

    cols = [
        "date", "location",
        "rainfall_24h_mm", "rainfall_72h_mm", "rainfall_7d_mm",
        "river_discharge_m3s", "discharge_3d_mean", "discharge_7d_mean",
        "temperature_mean_c",
        "elevation_m", "slope_deg", "drainage_capacity",
        "urbanization_index", "soil_moisture",
        "flood_risk",
    ]
    combined = combined[cols].dropna().reset_index(drop=True)

    print(f"\nCombined dataset: {len(combined)} rows across "
          f"{combined['location'].nunique()} locations")

    return combined


# ------------------------------------------------------------------
# 3. Dispatcher — Select the source based on the configuration
# ------------------------------------------------------------------
def generate_flood_data(config: dict) -> pd.DataFrame:
    source = config["data"]["source"]

    if source == "synthetic":
        return generate_synthetic_flood_data(
            n_samples=config["data"]["n_samples"],
            random_state=config["data"]["random_state"],
        )
    elif source == "open_meteo":
        om = config["data"]["open_meteo"]
        return build_dataset_from_open_meteo(
            locations=om["locations"],          
            start_date=om["start_date"],
            end_date=om["end_date"],
            discharge_threshold_percentile=om["discharge_threshold_percentile"],
        )
    else:
        raise ValueError(f"Unknown data source: {source}")