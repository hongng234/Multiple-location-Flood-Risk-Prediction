"""Raw dataset builders return the shared raw schema."""
import pandas as pd

from src import data_generator
from src.data_loader import RAW_COLUMNS


def test_synthetic_schema_and_size(raw_df):
    """Synthetic data has the raw columns, one row per (location, day), and physically valid values."""
    assert list(raw_df.columns) == RAW_COLUMNS
    n_days = len(pd.date_range("2020-01-01", "2022-06-30"))
    assert len(raw_df) == 3 * n_days                    # 3 locations × every day
    assert (raw_df["river_discharge_m3s"] > 0).all()    # discharge is always positive
    assert (raw_df["precipitation_mm"] >= 0).all()      # rain is never negative


def test_synthetic_is_reproducible():
    """Same seed -> identical data (needed for reproducible tests and benchmarks)."""
    a = data_generator.generate_synthetic_flood_data(["A"], "2020-01-01", "2020-03-01", random_state=1)
    b = data_generator.generate_synthetic_flood_data(["A"], "2020-01-01", "2020-03-01", random_state=1)
    pd.testing.assert_frame_equal(a, b)


def test_open_meteo_builder_stacks_locations(monkeypatch):
    """Fetchers are mocked; builder must tag rows with location and keep raw columns only."""
    dates = pd.date_range("2024-01-01", periods=5)

    # Fake fetchers returning 5 days of constant values (discharge differs per latitude)
    def fake_weather(lat, lon, s, e):
        return pd.DataFrame({"date": dates, "temperature_mean_c": 25.0, "precipitation_mm": 1.0})

    def fake_discharge(lat, lon, s, e):
        return pd.DataFrame({"date": dates, "river_discharge_m3s": 100.0 + lat})

    # Patch the names as imported inside data_generator (that's where they are looked up)
    monkeypatch.setattr(data_generator, "fetch_open_meteo_weather", fake_weather)
    monkeypatch.setattr(data_generator, "fetch_open_meteo_discharge", fake_discharge)
    locs = [{"name": "X", "latitude": 1, "longitude": 0}, {"name": "Y", "latitude": 2, "longitude": 0}]
    df = data_generator.build_dataset_from_open_meteo(locs, "2024-01-01", "2024-01-05")

    assert list(df.columns) == RAW_COLUMNS
    assert df.groupby("location").size().to_dict() == {"X": 5, "Y": 5}
    assert "flood_risk" not in df.columns   # labels are created later, after the split
