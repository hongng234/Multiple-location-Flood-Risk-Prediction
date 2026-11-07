"""Build the RAW daily multi-location dataset (real Open-Meteo data or offline synthetic data).

Both sources return the same schema (see `RAW_COLUMNS`). No labels or features are created here:
labels need a threshold computed on the TRAIN period only, which happens after the time split.
"""
import numpy as np    # fast numeric arrays + random number generation for the synthetic data
import pandas as pd   # DataFrames (tables) and date ranges

# Shared raw column list + the two API download functions (real data source)
from src.data_loader import RAW_COLUMNS, fetch_open_meteo_discharge, fetch_open_meteo_weather


# ------------------------------------------------------------------
# 1. SYNTHETIC DATA (offline; simple rainfall-runoff simulation)
# ------------------------------------------------------------------
def generate_synthetic_flood_data(
    location_names: list[str],
    start_date: str,
    end_date: str,
    random_state: int = 42,
) -> pd.DataFrame:
    """Simulate daily rain / temperature / discharge per location with a linear-reservoir model.

    What: a fake but physically plausible dataset with the same columns as the real one,
    so the whole pipeline (and the test suite) can run without internet.

    How:
      1. A sine wave gives each day a seasonal "wetness" between 0 (dry) and 1 (wet).
      2. Rain occurs randomly (more often when wet); its amount is drawn from a gamma distribution.
      3. Discharge follows a linear reservoir: it decays back to a base flow and jumps after rain.

    Args:
        location_names: one simulated river per name.
        start_date, end_date: inclusive date range ("YYYY-MM-DD"), one row per day.
        random_state: seed, so the same inputs always produce the same data.

    Returns:
        Long DataFrame with columns RAW_COLUMNS, one row per (location, day).
    """
    # Shared setup for all locations: random generator, the daily calendar and day-of-year (1..366)
    rng = np.random.default_rng(random_state)
    dates = pd.date_range(start_date, end_date, freq="D")
    doy = dates.dayofyear.to_numpy()
    frames = []   # one DataFrame per location, concatenated at the end

    # Loop over locations: each iteration builds an independent daily series for one river.
    # `i` (the location index) is used to shift the rainy season so locations behave differently.
    for i, name in enumerate(location_names):
        # Seasonal wetness (0..1), phase-shifted per location so regions differ
        wet = 0.5 + 0.5 * np.sin(2 * np.pi * (doy - 200 - 20 * i) / 365.25)
        # Does it rain today? Bernoulli draw with probability 15% (dry season) .. 65% (wet season)
        rain_occurs = rng.binomial(1, 0.15 + 0.5 * wet)
        # How much rain (mm)? Gamma draw (many small, few large events), heavier in the wet season;
        # multiplied by 0/1 so dry days get exactly 0 mm
        rain = rain_occurs * rng.gamma(0.8, 8 + 30 * wet)
        # Temperature: yearly cosine cycle around 26 °C (coolest mid-January) + daily noise
        temp = 26 - 5 * np.cos(2 * np.pi * (doy - 15) / 365.25) + rng.normal(0, 1, len(dates))

        # Each location has its own river size -> thresholds must be per location
        base = rng.uniform(20, 1500)   # dry-weather base flow (m3/s): small stream .. large river
        gain = base / 60               # how strongly 1 mm of rain raises discharge (scaled to river size)
        q = np.empty(len(dates))       # discharge series to fill day by day
        q[0] = base                    # start the river at base flow

        # Loop over days (starting at day 1): a Python loop is needed because each day's
        # discharge depends on the previous day's value (recursive, cannot be vectorised simply).
        for t in range(1, len(dates)):
            # Reservoir recession + response to yesterday's and today's rain + multiplicative noise
            #   0.85 * (q[t-1] - base): 85% of yesterday's excess flow remains (recession curve)
            #   gain * (0.7 * rain[t-1] + 0.3 * rain[t]): runoff mostly arrives with a 1-day delay
            q[t] = base + 0.85 * (q[t - 1] - base) + gain * (0.7 * rain[t - 1] + 0.3 * rain[t])
            # Random ±3% measurement/model noise; lognormal keeps discharge strictly positive
            q[t] *= rng.lognormal(0, 0.03)

        # Store this location's series in long format (one row per day, tagged with its name)
        frames.append(pd.DataFrame({
            "date": dates,
            "location": name,
            "precipitation_mm": rain.round(2),
            "temperature_mean_c": temp.round(2),
            "river_discharge_m3s": q.round(2),
        }))

    # Stack all locations into one table and enforce the shared raw column order
    return pd.concat(frames, ignore_index=True)[RAW_COLUMNS]


# ------------------------------------------------------------------
# 2. OPEN-METEO DATA (real rainfall + GloFAS discharge)
# ------------------------------------------------------------------
def build_dataset_from_open_meteo(locations: list[dict], start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch weather + discharge for every location and stack them into one long DataFrame.

    What: the real dataset — daily rain/temperature (Open-Meteo Archive) and daily river
    discharge (GloFAS via Open-Meteo Flood API) for each configured location.

    How: for each location, call both APIs (network required), inner-join the two results
    on the date, tag rows with the location name, then concatenate all locations.

    Args:
        locations: list of {"name", "latitude", "longitude"} dicts from config.yaml.
        start_date, end_date: inclusive date range ("YYYY-MM-DD").

    Returns:
        Long DataFrame with columns RAW_COLUMNS, one row per (location, day).
    """
    all_dfs = []   # one DataFrame per location

    # Loop over locations: 2 API calls each -> merge -> tag -> keep only the raw columns
    for loc in locations:
        print(f"Fetching data for: {loc['name']} ({loc['latitude']}, {loc['longitude']})")
        weather = fetch_open_meteo_weather(loc["latitude"], loc["longitude"], start_date, end_date)
        discharge = fetch_open_meteo_discharge(loc["latitude"], loc["longitude"], start_date, end_date)

        # Join both sources on calendar day and tag rows with the location name
        # (inner join: a day missing in either API is dropped, so every row has all values)
        df = weather.merge(discharge, on="date", how="inner").sort_values("date")
        df["location"] = loc["name"]
        all_dfs.append(df[RAW_COLUMNS])

        # Print the grid cell the API actually used, so wrong coordinates are easy to spot
        # (attrs["grid"] is set by data_loader from the API response; a tiny mean discharge
        #  for a big river means the coordinates hit a small stream instead of the main channel)
        print(f"  -> {len(df)} rows | weather grid={weather.attrs.get('grid')} "
              f"| GloFAS grid={discharge.attrs.get('grid')} "
              f"| mean discharge={df['river_discharge_m3s'].mean():.2f} m3/s")

    # After the loop: stack all locations vertically into one long table
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nCombined raw dataset: {len(combined)} rows across {combined['location'].nunique()} locations")
    return combined


# ------------------------------------------------------------------
# 3. Dispatcher — select the source based on the configuration
# ------------------------------------------------------------------
def generate_flood_data(config: dict) -> pd.DataFrame:
    """Return the raw daily dataset from the source named in config['data']['source'].

    What: the single entry point main.py calls; it hides which source is used.
    How: read `source` from config and forward the relevant config values to the matching builder.
    """
    data_cfg = config["data"]       # the "data:" section of config.yaml
    source = data_cfg["source"]     # "synthetic" or "open_meteo"

    # Synthetic source: simulate data offline; only location NAMES are used (coordinates are ignored)
    if source == "synthetic":
        return generate_synthetic_flood_data(
            location_names=[loc["name"] for loc in data_cfg["locations"]],
            start_date=data_cfg["start_date"],
            end_date=data_cfg["end_date"],
            random_state=data_cfg["random_state"],
        )
    # Open-Meteo source: download real data for every configured location (needs internet)
    if source == "open_meteo":
        return build_dataset_from_open_meteo(
            locations=data_cfg["locations"],
            start_date=data_cfg["start_date"],
            end_date=data_cfg["end_date"],
        )
    # Any other source name is a config mistake: fail fast with a clear error instead of returning nothing
    raise ValueError(f"Unknown data source: {source}")
