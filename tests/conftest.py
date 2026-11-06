"""Shared pytest fixtures. All tests are offline (synthetic data / mocked HTTP).

pytest loads this file automatically; any test can request a fixture by naming it as an argument.
"""
import sys
from pathlib import Path

import pytest

# Make `src` and `main` importable when pytest runs from any folder
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generator import generate_synthetic_flood_data  # noqa: E402
from src.features import add_threshold_features, build_timeseries_features  # noqa: E402
from src.preprocessing import apply_labels, compute_thresholds, time_split  # noqa: E402

# Small, fixed test setup: 3 fake rivers, test period = the last 6 months of data
LOCATIONS = ["RiverA", "RiverB", "RiverC"]
TEST_START = "2022-01-01"


@pytest.fixture(scope="session")
def raw_df():
    """~2.5 years of synthetic daily data for 3 locations.

    scope="session": built once and shared by all tests (fast); tests must not modify it in place.
    """
    return generate_synthetic_flood_data(LOCATIONS, "2020-01-01", "2022-06-30", random_state=0)


@pytest.fixture(scope="session")
def feature_df(raw_df):
    """Features + q_future for the synthetic data (horizon = 1)."""
    return build_timeseries_features(raw_df, horizon=1, lags=[1, 2, 3, 7], rain_windows=[3, 7], api_decay=0.9)


@pytest.fixture(scope="session")
def labelled_split(feature_df):
    """(train, test, thresholds) built exactly as main.py does.

    How: chronological split -> thresholds from train -> label + threshold features on both sides.
    """
    train, test = time_split(feature_df, TEST_START)
    thr = compute_thresholds(train, 90)
    train = add_threshold_features(apply_labels(train, thr), thr)
    test = add_threshold_features(apply_labels(test, thr), thr)
    return train, test, thr


@pytest.fixture
def tmp_config(tmp_path):
    """A small synthetic-source config that writes everything under tmp_path.

    How: mirrors config.yaml's structure, but every path points into pytest's per-test temp
    folder, so tests never touch the real data/ or outputs/ folders.
    """
    return {
        "data": {
            "source": "synthetic", "random_state": 0,
            "raw_path": str(tmp_path / "raw.csv"),
            "processed_path": str(tmp_path / "processed.csv"),
            "start_date": "2020-01-01", "end_date": "2022-06-30",
            "locations": [{"name": n, "latitude": 0.0, "longitude": 0.0} for n in LOCATIONS],
        },
        "task": {"target": "flood_risk", "horizon_days": 1, "threshold_percentile": 90},
        "features": {"lags": [1, 2, 3, 7], "rain_windows": [3, 7], "api_decay": 0.9},
        "split": {"test_start": TEST_START, "cv_folds": 3},
        "models": {"random_state": 0, "selection_metric": "average_precision"},
        "output": {
            "figures_dir": str(tmp_path / "figures"),
            "metrics_dir": str(tmp_path / "metrics"),
            "model_path": str(tmp_path / "models" / "best_model.joblib"),
        },
    }
