"""Open-Meteo clients parse responses correctly (HTTP is mocked) and CSV helpers round-trip."""
import pandas as pd

from src import data_loader


class FakeResponse:
    """Minimal stand-in for `requests.Response`: only the two methods data_loader uses."""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        # Pretend every request succeeded (HTTP 200)
        pass

    def json(self):
        return self._payload


def _fake_get(url, params=None, timeout=None):
    """Return a canned JSON payload depending on which API is called.

    How: the flood API URL contains "flood" -> return discharge; anything else -> weather.
    The shape mirrors the real Open-Meteo response (lat/lon + "daily" lists).
    """
    if "flood" in url:
        daily = {"time": ["2024-01-01", "2024-01-02"], "river_discharge": [10.0, 12.5]}
    else:
        daily = {"time": ["2024-01-01", "2024-01-02"],
                 "temperature_2m_mean": [25.0, 26.0], "precipitation_sum": [0.0, 5.2]}
    return FakeResponse({"latitude": 1.0, "longitude": 2.0, "daily": daily})


def test_fetch_weather_parses_payload(monkeypatch):
    """Weather JSON becomes the expected columns, dates are datetimes, grid coords are kept."""
    # Replace the real network call with the fake one for this test only
    monkeypatch.setattr(data_loader.requests, "get", _fake_get)
    df = data_loader.fetch_open_meteo_weather(1, 2, "2024-01-01", "2024-01-02")
    assert list(df.columns) == ["date", "temperature_mean_c", "precipitation_mm"]
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["precipitation_mm"].tolist() == [0.0, 5.2]
    assert df.attrs["grid"] == (1.0, 2.0)


def test_fetch_discharge_parses_payload(monkeypatch):
    """Flood-API JSON becomes a river_discharge_m3s column with the original values."""
    monkeypatch.setattr(data_loader.requests, "get", _fake_get)
    df = data_loader.fetch_open_meteo_discharge(1, 2, "2024-01-01", "2024-01-02")
    assert df["river_discharge_m3s"].tolist() == [10.0, 12.5]


def test_csv_roundtrip_parses_dates(tmp_path):
    """save_csv creates missing folders; load_csv reads 'date' back as datetime."""
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "x": [1]})
    path = tmp_path / "sub" / "f.csv"   # "sub" does not exist yet
    data_loader.save_csv(df, str(path))
    back = data_loader.load_csv(str(path))
    assert pd.api.types.is_datetime64_any_dtype(back["date"])
    assert back["x"].tolist() == [1]
