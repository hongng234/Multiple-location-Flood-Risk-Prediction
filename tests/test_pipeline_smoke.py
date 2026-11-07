"""End-to-end: main.run() on synthetic data produces outputs and a usable model bundle;
predict.predict_latest() scores the newest day per location."""
from pathlib import Path

import joblib     # reads the saved model bundle

import main       # training script (its run() function)
import predict    # live-forecast script (its predict_latest() function)
from src.data_generator import generate_synthetic_flood_data
from tests.conftest import LOCATIONS   # the 3 fake location names


def test_run_end_to_end(tmp_config):
    """Full training run writes all metrics/figures and a bundle that can predict on raw recent data."""
    # Act: run the whole pipeline into the temp folder (refresh=True -> build synthetic raw data)
    result = main.run(tmp_config, refresh=True)

    # All metric files are written
    out = tmp_config["output"]   # output folders inside the temp directory
    for f in ["cv_results.csv", "cv_folds.csv", "test_results.csv", "per_location_results.csv", "thresholds.json"]:
        assert (Path(out["metrics_dir"]) / f).exists(), f
    # Key charts exist, including one time-series chart per location
    figs = {p.name for p in Path(out["figures_dir"]).glob("*.png")}   # file names of all saved charts
    assert {"roc_curves.png", "pr_curves.png", "cv_folds.png", "per_location_f1.png"} <= figs
    assert {f"timeseries_{loc}.png" for loc in LOCATIONS} <= figs

    # The chosen model is a real ML model, never a baseline
    assert result["best_model_name"] not in {"DummyBaseline", "PersistenceBaseline"}

    # Saved bundle is self-contained: preprocessing + model + thresholds
    bundle = joblib.load(out["model_path"])
    assert set(bundle["thresholds"]) == set(LOCATIONS)   # one threshold per location
    proba = bundle["model"].predict_proba(result["test"][bundle["feature_columns"]])[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()           # valid probabilities

    # Live-style inference from raw recent data (30 fake "recent" days)
    recent = generate_synthetic_flood_data(LOCATIONS, "2022-06-01", "2022-06-30", random_state=5)
    pred = predict.predict_latest(bundle, recent)
    assert sorted(pred["location"]) == sorted(LOCATIONS)   # exactly one forecast per location
    assert pred["flood_probability"].between(0, 1).all()


def test_cached_raw_data_is_reused(tmp_config, capsys):
    """A second load without refresh reads the cached CSV instead of rebuilding it."""
    main.load_raw(tmp_config, refresh=True)    # builds + caches
    main.load_raw(tmp_config, refresh=False)   # should hit the cache
    # capsys captures everything printed; the cache message proves the CSV was reused
    assert "Loaded cached raw data" in capsys.readouterr().out
