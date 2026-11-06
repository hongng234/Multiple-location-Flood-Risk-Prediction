"""Train + benchmark the multi-location next-day flood-risk forecaster.

Usage:
    python main.py                # use cached raw data if its schema is current
    python main.py --refresh      # re-download data from Open-Meteo
    python main.py --config other.yaml
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.benchmark import cross_validate_models, evaluate_on_test, evaluate_per_location, select_best_model
from src.config import ensure_dirs, load_config
from src.data_generator import generate_flood_data
from src.data_loader import RAW_COLUMNS, load_csv, save_csv
from src.features import add_threshold_features, build_timeseries_features, feature_columns
from src.models import get_models
from src.preprocessing import (
    apply_labels, build_preprocessor, clean_data, compute_thresholds,
    time_series_cv_splits, time_split,
)
from src import visualize as viz


def load_raw(config: dict, refresh: bool = False) -> pd.DataFrame:
    """Return the raw daily dataset: cached CSV if valid, otherwise fetch/generate and cache it.

    What: avoids re-downloading 5 years × 5 locations on every run.
    How: reuse data/raw/flood_data.csv only if it exists, `refresh` is off, its columns match
    RAW_COLUMNS and it holds exactly the configured locations; otherwise rebuild and save it.
    """
    raw_path = config["data"]["raw_path"]
    # Try the cache first (skipped entirely when --refresh is given or the file doesn't exist)
    if not refresh and Path(raw_path).exists():
        df = load_csv(raw_path)
        # Re-fetch when the cache comes from an older schema or a different location list
        expected_locs = {loc["name"] for loc in config["data"]["locations"]}
        # Cache matches the current config: use it as-is
        if list(df.columns) == RAW_COLUMNS and set(df["location"]) == expected_locs:
            print(f"Loaded cached raw data: {raw_path}")
            return df
        print("Cached raw data is stale (schema/locations changed) -> rebuilding")

    # No usable cache: build from the configured source and save it for next time
    df = generate_flood_data(config)
    save_csv(df, raw_path)
    print(f"Built raw data from source='{config['data']['source']}': {raw_path}")
    return df


def run(config: dict, refresh: bool = False) -> dict:
    """Execute the full pipeline and return the key artefacts (used by tests).

    How (in order): raw data -> features -> chronological split -> train-only thresholds and
    labels -> EDA charts -> time-series CV -> test evaluation -> best-model selection by CV ->
    save metrics, charts and the model bundle.

    Args:
        config: dict from `load_config`.
        refresh: force rebuilding the raw data instead of using the cache.

    Returns:
        dict with result tables, thresholds, the best model name, the saved bundle and train/test frames.
    """
    # Shortcuts to frequently used config sections and output folders
    ensure_dirs(config)
    fig_dir = Path(config["output"]["figures_dir"])
    met_dir = Path(config["output"]["metrics_dir"])
    task, feat_cfg, split_cfg = config["task"], config["features"], config["split"]
    target, horizon = task["target"], task["horizon_days"]

    # 1. Raw data -> clean
    raw = clean_data(load_raw(config, refresh))

    # 2. Backward-looking features + forward target source (q_future)
    df = build_timeseries_features(
        raw, horizon=horizon, lags=feat_cfg["lags"],
        rain_windows=feat_cfg["rain_windows"], api_decay=feat_cfg["api_decay"],
    )

    # 3. Chronological split FIRST, then thresholds from TRAIN only, then labels
    #    (the same train thresholds label the test set, exactly as they would in production)
    train, test = time_split(df, split_cfg["test_start"])
    thresholds = compute_thresholds(train, task["threshold_percentile"])
    train = add_threshold_features(apply_labels(train, thresholds, target), thresholds)
    test = add_threshold_features(apply_labels(test, thresholds, target), thresholds)
    # Save the labelled data with a "split" column for inspection
    save_csv(pd.concat([train.assign(split="train"), test.assign(split="test")]), config["data"]["processed_path"])
    print(f"\nTrain: {train['date'].min().date()} -> {train['date'].max().date()} ({len(train)} rows, "
          f"flood rate {train[target].mean():.1%})")
    print(f"Test : {test['date'].min().date()} -> {test['date'].max().date()} ({len(test)} rows, "
          f"flood rate {test[target].mean():.1%})")

    # 4. Feature matrix: numeric features + location (one-hot inside the pipeline)
    num_cols = feature_columns(train, target)
    X_cols = num_cols + ["location"]
    train = train.sort_values(["date", "location"]).reset_index(drop=True)   # chronological for CV
    X_train, y_train = train[X_cols], train[target]
    X_test, y_test = test[X_cols], test[target]

    # 5. EDA charts (data overview before any modelling)
    viz.plot_discharge_overview(raw, thresholds, split_cfg["test_start"], fig_dir / "discharge_overview.png")
    labelled = pd.concat([train, test])
    viz.plot_class_distribution(labelled, target, fig_dir / "class_distribution.png")
    viz.plot_monthly_flood_frequency(labelled, target, fig_dir / "monthly_flood_frequency.png")
    viz.plot_correlation_heatmap(train[num_cols + [target]], fig_dir / "correlation_heatmap.png")

    # 6. Time-series cross-validation on the train period
    #    (folds split on dates with a gap of `horizon` days so labels never overlap validation)
    models = get_models(build_preprocessor(num_cols, ["location"]), config["models"]["random_state"])
    cv = time_series_cv_splits(train["date"], n_splits=split_cfg["cv_folds"], gap_days=horizon)
    cv_results, cv_folds = cross_validate_models(models, X_train, y_train, cv)
    print("\n=== Time-series CV (train period) ===")
    print(cv_results.to_string(index=False, float_format="%.3f"))

    # 7. Hold-out test evaluation (all models refit on full train)
    test_results, fitted, probas = evaluate_on_test(models, X_train, y_train, X_test, y_test)
    per_loc = evaluate_per_location(fitted, X_test, y_test, test["location"])
    print("\n=== Test set (hold-out) ===")
    print(test_results.to_string(index=False, float_format="%.3f"))

    # 8. Select best model by CV (not test) and report it
    #    (the per-location table compares it with the persistence baseline it has to beat)
    metric = config["models"]["selection_metric"]
    best_name = select_best_model(cv_results, metric)
    best = fitted[best_name]
    print(f"\nBest model by CV {metric}: {best_name}")
    print(per_loc[per_loc["model"].isin([best_name, "PersistenceBaseline"])]
          .to_string(index=False, float_format="%.3f"))

    # 9. Save metrics (CSV tables + the train thresholds as JSON)
    cv_results.to_csv(met_dir / "cv_results.csv", index=False)
    cv_folds.to_csv(met_dir / "cv_folds.csv", index=False)
    test_results.to_csv(met_dir / "test_results.csv", index=False)
    per_loc.to_csv(met_dir / "per_location_results.csv", index=False)
    (met_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    # 10. Benchmark charts
    viz.plot_model_comparison(test_results, fig_dir / "model_comparison.png")
    viz.plot_roc_curves(probas, y_test, fig_dir / "roc_curves.png")
    viz.plot_pr_curves(probas, y_test, fig_dir / "pr_curves.png")
    viz.plot_cv_folds(cv_folds, metric, fig_dir / "cv_folds.png")
    viz.plot_per_location_metrics(per_loc, [best_name, "PersistenceBaseline"], "f1", fig_dir / "per_location_f1.png")
    viz.plot_confusion_matrix(y_test, best.predict(X_test), best_name, fig_dir / "confusion_matrix.png")
    viz.plot_feature_importance(best, fig_dir / "feature_importance.png")
    # Loop over locations: one discharge-vs-forecast time-series chart per location
    for loc in thresholds:
        viz.plot_location_timeseries(test, probas[best_name], thresholds[loc], loc, best_name,
                                     fig_dir / f"timeseries_{loc}.png")

    # 11. Save the model bundle (pipeline + everything needed to rebuild features at inference)
    bundle = {
        "model": best,
        "model_name": best_name,
        "feature_columns": X_cols,
        "thresholds": thresholds,
        "horizon_days": horizon,
        "features_config": feat_cfg,
        "trained_until": str(train["target_date"].max().date()),
    }
    joblib.dump(bundle, config["output"]["model_path"])
    print(f"Saved model bundle: {config['output']['model_path']}")

    return {
        "cv_results": cv_results, "test_results": test_results, "per_location": per_loc,
        "thresholds": thresholds, "best_model_name": best_name, "bundle": bundle,
        "train": train, "test": test,
    }


def parse_args(argv=None) -> argparse.Namespace:
    """Command-line flags.

    How: `argv=None` makes argparse read sys.argv; tests can pass a list instead.
    """
    parser = argparse.ArgumentParser(description="Multi-location flood risk forecasting pipeline")
    parser.add_argument("--config", default="config.yaml", help="path to YAML config")
    parser.add_argument("--refresh", action="store_true", help="re-download/rebuild raw data")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """CLI entry point: parse flags, load the config, run the pipeline."""
    args = parse_args(argv)
    run(load_config(args.config), refresh=args.refresh)


# Run only when executed as a script (not when imported by tests)
if __name__ == "__main__":
    main()
