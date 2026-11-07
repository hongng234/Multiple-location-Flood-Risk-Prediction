"""I/O helpers: CSV read/write and Open-Meteo API clients (historical + recent data).

Open-Meteo is a free weather service (no API key). We call three of its endpoints over HTTP
and turn each JSON answer into a pandas DataFrame with clear column names.
"""
from pathlib import Path   # file paths, used to create missing folders

import pandas as pd        # tables (DataFrames): the main data structure of the project
import requests            # simple HTTP client used to call the Open-Meteo web APIs

# API endpoints
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"   # historical weather (ERA5-based)
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"         # recent weather (for live prediction)
FLOOD_URL = "https://flood-api.open-meteo.com/v1/flood"         # GloFAS river discharge

# Raw columns every data source must provide (one row per location per day)
RAW_COLUMNS = ["date", "location", "precipitation_mm", "temperature_mean_c", "river_discharge_m3s"]


def load_csv(path: str) -> pd.DataFrame:
    """Read a CSV; parse the 'date' column as datetime when present.

    How: first read only the header (nrows=0, very cheap) to check whether a 'date' column
    exists, then read the full file with `parse_dates` only if it does. This way the same
    helper works for both dated (raw/processed) and undated CSVs.
    """
    # Column names only (no data rows are loaded)
    header = pd.read_csv(path, nrows=0).columns
    # Read the whole file; convert the 'date' text column to real datetimes when it exists
    return pd.read_csv(path, parse_dates=["date"]) if "date" in header else pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to CSV, creating the parent folder if needed (index is not written)."""
    # Make sure the target folder exists (no error if it already does)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # index=False: don't write pandas' 0,1,2,... row numbers as an extra column
    df.to_csv(path, index=False)


def _get_daily(url: str, params: dict) -> tuple[dict, dict]:
    """GET an Open-Meteo endpoint; return (daily block, full JSON payload).

    How: send the request with a 60 s timeout, raise an HTTPError on any 4xx/5xx status
    (so failures are loud, not silent empty data), then split out the "daily" section.
    The full payload is also returned because it holds the snapped grid latitude/longitude.
    """
    # Send the HTTP GET request; `params` become the ?latitude=...&daily=... query string
    response = requests.get(url, params=params, timeout=60)
    # Stop with an error if the server answered with a failure code (e.g. 400 bad request, 500 server error)
    response.raise_for_status()
    # Decode the JSON body into a Python dict
    payload = response.json()
    # "daily" holds parallel lists: {"time": [...], "<variable>": [...], ...}
    return payload["daily"], payload


def _weather_frame(daily: dict, payload: dict) -> pd.DataFrame:
    """Convert an Open-Meteo daily weather block into a DataFrame (keeps snapped grid coords in attrs).

    How: map the API's parallel lists (time, temperature, precipitation) to named columns,
    then store the grid cell coordinates in `df.attrs` so callers can print/verify them.
    """
    # One row per day; rename the API's variable names to the project's column names
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]),             # "2024-01-01" text -> datetime
            "temperature_mean_c": daily["temperature_2m_mean"],  # mean air temperature at 2 m (°C)
            "precipitation_mm": daily["precipitation_sum"],      # total rain of the day (mm)
        }
    )
    # Remember which grid cell the API actually used (may differ slightly from the requested point)
    df.attrs["grid"] = (payload.get("latitude"), payload.get("longitude"))
    return df


def _discharge_frame(daily: dict, payload: dict) -> pd.DataFrame:
    """Convert an Open-Meteo flood daily block into a DataFrame (keeps snapped grid coords in attrs).

    How: same as `_weather_frame`, but for the single `river_discharge` series (m3/s).
    """
    # One row per day: the date and the river discharge (volume of water flowing per second)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]),
            "river_discharge_m3s": daily["river_discharge"],   # cubic metres per second
        }
    )
    # Remember which GloFAS grid cell answered (used to spot coordinates that miss the main river)
    df.attrs["grid"] = (payload.get("latitude"), payload.get("longitude"))
    return df


def fetch_open_meteo_weather(latitude: float, longitude: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Download historical daily rainfall + mean temperature from the Open-Meteo Archive API.

    How: request the daily variables `temperature_2m_mean` and `precipitation_sum` for the
    date range; `timezone="auto"` makes "a day" mean the local calendar day of the location.

    Returns:
        DataFrame with columns date, temperature_mean_c, precipitation_mm.
    """
    # Query parameters understood by the Archive API
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean,precipitation_sum",   # which daily variables to return
        "timezone": "auto",                                  # use the location's own time zone
    }
    # Call the API, then convert the answer; `*` unpacks the (daily, payload) tuple into two arguments
    return _weather_frame(*_get_daily(ARCHIVE_URL, params))


def fetch_open_meteo_discharge(latitude: float, longitude: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Download historical daily river discharge (GloFAS) from the Open-Meteo Flood API.

    How: request the daily `river_discharge` variable; the API returns the value of the
    ~5 km GloFAS grid cell nearest to the given coordinates.

    Returns:
        DataFrame with columns date, river_discharge_m3s.
    """
    # Query parameters understood by the Flood API
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "river_discharge",   # daily mean discharge of the nearest GloFAS river cell
    }
    # Call the API, then convert the answer into a DataFrame
    return _discharge_frame(*_get_daily(FLOOD_URL, params))


def fetch_recent(latitude: float, longitude: float, past_days: int = 30) -> pd.DataFrame:
    """Download the last `past_days` of weather + discharge (used by predict.py for live forecasts).

    What: the archive API lags a few days behind real time, so live prediction uses the
    forecast API (which also serves recent past days) plus the flood API.
    How: call both APIs with `past_days`, inner-join on date, then drop today and later,
    because those values are still partial or forecasted (not yet observed).

    Returns:
        DataFrame with columns date, temperature_mean_c, precipitation_mm, river_discharge_m3s.
    """
    # Recent weather (past days + today's forecast, which is removed below)
    weather = _weather_frame(*_get_daily(FORECAST_URL, {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_mean,precipitation_sum",
        "past_days": past_days,
        "forecast_days": 1,
        "timezone": "auto",
    }))
    # Recent discharge for the same window
    discharge = _discharge_frame(*_get_daily(FLOOD_URL, {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "river_discharge",
        "past_days": past_days,
        "forecast_days": 1,
    }))
    # Keep only days that are fully observed (drop today's / future partial values)
    df = weather.merge(discharge, on="date", how="inner")   # one row per day with all three variables
    today = pd.Timestamp.today().normalize()                # today at 00:00 (time of day removed)
    return df[df["date"] < today].reset_index(drop=True)    # keep strictly past days, renumber rows 0..n-1
