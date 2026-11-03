from pathlib import Path
import pandas as pd
import requests


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"]) if "date" in pd.read_csv(path, nrows=0).columns else pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def fetch_open_meteo_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download daily weather (rain, temperature) from Open-Meteo Archive API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean,precipitation_sum",
        "timezone": "auto",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    daily = response.json()["daily"]

    return pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]),
            "temperature_mean_c": daily["temperature_2m_mean"],
            "precipitation_mm": daily["precipitation_sum"],
        }
    )


def fetch_open_meteo_discharge(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download daily river discharge from Open-Meteo Flood API (GloFAS)."""
    url = "https://flood-api.open-meteo.com/v1/flood"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "river_discharge",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    daily = response.json()["daily"]

    return pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]),
            "river_discharge_m3s": daily["river_discharge"],
        }
    )