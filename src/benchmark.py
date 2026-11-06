"""Cross-validation, hold-out evaluation, per-location metrics and model selection."""
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import cross_validate

from src.models import BASELINES

# Metrics computed in cross-validation (sklearn scorer names); average_precision = PR-AUC
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]


def _metrics(y_true, y_pred, y_proba) -> dict:
    """Compute all classification metrics; ranking metrics are NaN when only one class is present.

    How: threshold-based metrics (accuracy/precision/recall/F1) use hard predictions;
    ranking metrics (ROC-AUC, PR-AUC) use probabilities. ROC/PR-AUC are undefined when a
    subset (e.g. one location) has no flood day, so NaN is returned there instead of crashing.
    """
    both_classes = len(np.unique(y_true)) == 2
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if both_classes else np.nan,
        "average_precision": average_precision_score(y_true, y_proba) if both_classes else np.nan,
    }


def cross_validate_models(models: dict, X, y, cv) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-series CV for every model. Returns (mean summary, per-fold scores).

    How: for each model, sklearn's `cross_validate` clones it, fits on each fold's train rows
    and scores on its validation rows (the preprocessor is refit per fold -> no leakage).

    Args:
        models: {name: estimator} from `get_models`.
        X, y: train features/labels, sorted by date.
        cv: list of (train_idx, val_idx) from `time_series_cv_splits`.

    Returns:
        summary: one row per model with mean metrics, sorted by PR-AUC.
        folds: one row per (model, fold) — used for the CV stability chart.
    """
    summary, folds = [], []
    # Loop over models: run CV for each and collect mean + per-fold scores
    for name, model in models.items():
        # Silence expected warnings only inside this block (they would flood the console)
        with warnings.catch_warnings():
            # Early folds can lack positives -> sklearn warns and scores NaN; we nan-average instead
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            warnings.simplefilter("ignore", UserWarning)
            scores = cross_validate(model, X, y, cv=cv, scoring=SCORING, error_score=np.nan)

        # Average each metric over folds, ignoring NaN folds
        row = {"model": name}
        for m in SCORING:
            row[m] = np.nanmean(scores[f"test_{m}"])
        row["fit_time_s"] = scores["fit_time"].mean()
        summary.append(row)

        # Keep the raw score of every fold (fold numbers start at 1 for readability)
        for i in range(len(scores["fit_time"])):
            folds.append({"model": name, "fold": i + 1, **{m: scores[f"test_{m}"][i] for m in SCORING}})

    summary_df = pd.DataFrame(summary).sort_values("average_precision", ascending=False)
    return summary_df.reset_index(drop=True), pd.DataFrame(folds)


def evaluate_on_test(models: dict, X_train, y_train, X_test, y_test) -> tuple[pd.DataFrame, dict, dict]:
    """Fit each model on the full train set and score it on the hold-out test set.

    How: clone each model (so the originals stay unfitted), fit on all train rows, predict
    probabilities + labels on the test rows and compute `_metrics`.

    Returns:
        (metrics table, fitted models, test probabilities per model).
    """
    rows, fitted, probas = [], {}, {}
    # Loop over models: fit once on full train, evaluate once on test
    for name, model in models.items():
        est = clone(model).fit(X_train, y_train)
        y_proba = est.predict_proba(X_test)[:, 1]   # column 1 = P(flood)
        y_pred = est.predict(X_test)
        rows.append({"model": name, **_metrics(y_test, y_pred, y_proba)})
        fitted[name], probas[name] = est, y_proba
    return pd.DataFrame(rows), fitted, probas


def evaluate_per_location(fitted: dict, X_test, y_test, locations) -> pd.DataFrame:
    """Test metrics broken down by location (a model can be good in Hanoi and bad in Hue).

    How: predict once per model on the whole test set, then slice predictions with a boolean
    mask per location and compute metrics on each slice.

    Returns:
        Long table: one row per (model, location) with n_days, n_floods and all metrics.
    """
    # Plain positional Series so boolean masks line up with the prediction arrays
    locations = pd.Series(np.asarray(locations))
    y_test = pd.Series(np.asarray(y_test))
    rows = []
    # Outer loop over fitted models: predict the full test set once per model
    for name, est in fitted.items():
        y_proba = est.predict_proba(X_test)[:, 1]
        y_pred = est.predict(X_test)
        # Inner loop over locations: score only the rows belonging to that location
        for loc in sorted(locations.unique()):
            mask = (locations == loc).to_numpy()
            rows.append({
                "model": name, "location": loc,
                "n_days": int(mask.sum()), "n_floods": int(y_test[mask].sum()),
                **_metrics(y_test[mask], y_pred[mask], y_proba[mask]),
            })
    return pd.DataFrame(rows)


def select_best_model(cv_results: pd.DataFrame, metric: str = "average_precision") -> str:
    """Pick the best NON-baseline model by its CV score (never by test score -> no selection bias).

    How: drop baseline rows, sort the rest by `metric` (descending) and return the top name.
    """
    candidates = cv_results[~cv_results["model"].isin(BASELINES)]
    return candidates.sort_values(metric, ascending=False).iloc[0]["model"]
