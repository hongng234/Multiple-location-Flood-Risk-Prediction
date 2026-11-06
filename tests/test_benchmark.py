"""Models, CV, evaluation tables and best-model selection."""
import numpy as np
import pandas as pd

from src.benchmark import (
    SCORING, cross_validate_models, evaluate_on_test, evaluate_per_location, select_best_model,
)
from src.features import feature_columns
from src.models import PersistenceBaseline, get_models
from src.preprocessing import build_preprocessor, time_series_cv_splits


def test_persistence_baseline_rule():
    """Persistence predicts flood only when today's relative discharge is strictly above 1."""
    X = pd.DataFrame({"q_t_rel": [0.5, 1.0, 1.5]})
    m = PersistenceBaseline().fit(X, [0, 1, 1])
    assert m.predict(X).tolist() == [0, 0, 1]
    np.testing.assert_allclose(m.predict_proba(X).sum(axis=1), 1)   # valid probability rows


def test_full_benchmark_on_synthetic(labelled_split):
    """CV + test + per-location evaluation run end-to-end and return well-formed tables."""
    # Arrange: chronological train set and the same feature columns as main.py
    train, test, _ = labelled_split
    train = train.sort_values(["date", "location"]).reset_index(drop=True)
    num = feature_columns(train)
    cols = num + ["location"]
    models = get_models(build_preprocessor(num, ["location"]), random_state=0)
    models = {k: models[k] for k in ["DummyBaseline", "PersistenceBaseline", "LogisticRegression"]}  # keep fast

    # Cross-validation: summary has every metric, folds table has one row per (model, fold)
    cv = time_series_cv_splits(train["date"], n_splits=3)
    summary, folds = cross_validate_models(models, train[cols], train["flood_risk"], cv)
    assert set(SCORING) <= set(summary.columns)
    assert len(folds) == 3 * len(models)

    # Test evaluation: one row per model, one probability per test row
    res, fitted, probas = evaluate_on_test(models, train[cols], train["flood_risk"], test[cols], test["flood_risk"])
    assert set(res["model"]) == set(models)
    assert all(len(p) == len(test) for p in probas.values())

    # A real model must beat the "always no flood" baseline on PR-AUC
    ap = res.set_index("model")["average_precision"]
    assert ap["LogisticRegression"] > ap["DummyBaseline"]

    # Per-location table: one row per (model, location)
    per_loc = evaluate_per_location(fitted, test[cols], test["flood_risk"], test["location"])
    assert len(per_loc) == len(models) * test["location"].nunique()


def test_select_best_model_ignores_baselines_and_uses_cv():
    """Even if a baseline has the top score, the best ML model is selected."""
    cv = pd.DataFrame({
        "model": ["PersistenceBaseline", "RandomForest", "LogisticRegression"],
        "average_precision": [0.99, 0.70, 0.80],
    })
    assert select_best_model(cv, "average_precision") == "LogisticRegression"
