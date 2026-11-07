"""Every plotting function writes a non-empty PNG.

These tests don't check how the charts look, only that each one is drawn without errors.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression   # a fast model to feed the model-based charts
from sklearn.pipeline import Pipeline

from src import visualize as viz   # module under test, short alias
from src.features import feature_columns
from src.preprocessing import build_preprocessor


def _ok(path):
    """A chart counts as written if the file exists and is bigger than an empty image (>1 KB)."""
    return path.exists() and path.stat().st_size > 1000


def test_eda_plots(tmp_path, raw_df, labelled_split):
    """The four EDA charts render from the labelled synthetic data."""
    train, test, thr = labelled_split
    df = pd.concat([train, test])   # all labelled rows
    # Draw the four charts into the temp folder as a.png ... d.png
    viz.plot_discharge_overview(raw_df, thr, "2022-01-01", tmp_path / "a.png")
    viz.plot_class_distribution(df, "flood_risk", tmp_path / "b.png")
    viz.plot_monthly_flood_frequency(df, "flood_risk", tmp_path / "c.png")
    viz.plot_correlation_heatmap(train[feature_columns(train) + ["flood_risk"]], tmp_path / "d.png")
    assert all(_ok(tmp_path / f"{n}.png") for n in "abcd")


def test_benchmark_plots(tmp_path, labelled_split):
    """All eight benchmark charts render from a quickly fitted model and small fake result tables."""
    # Arrange: fit a small real pipeline so feature-importance/timeseries charts get real inputs
    train, test, thr = labelled_split
    num = feature_columns(train)
    model = Pipeline([("prep", build_preprocessor(num, ["location"])), ("clf", LogisticRegression(max_iter=500))])
    model.fit(train[num + ["location"]], train["flood_risk"])
    p = model.predict_proba(test[num + ["location"]])[:, 1]   # P(flood) per test row
    y = test["flood_risk"]                                    # true labels
    # Two "models": a constant guess (flood rate) and the fitted logistic regression
    probas = {"Dummy": np.full(len(y), y.mean()), "LR": p}

    # Fake result tables with the same columns the pipeline produces
    results = pd.DataFrame([{"model": m, "accuracy": .9, "precision": .5, "recall": .5, "f1": .5,
                             "roc_auc": .8, "average_precision": .6} for m in probas])
    folds = pd.DataFrame([{"model": m, "fold": f, "average_precision": .5 + f / 10} for m in probas for f in (1, 2, 3)])
    per_loc = pd.DataFrame([{"model": m, "location": l, "f1": .6} for m in probas for l in thr])
    loc = next(iter(thr))   # any one location for the timeseries chart

    # Act: draw every chart
    viz.plot_model_comparison(results, tmp_path / "1.png")
    viz.plot_roc_curves(probas, y, tmp_path / "2.png")
    viz.plot_pr_curves(probas, y, tmp_path / "3.png")
    viz.plot_cv_folds(folds, "average_precision", tmp_path / "4.png")
    viz.plot_per_location_metrics(per_loc, ["LR", "Dummy"], "f1", tmp_path / "5.png")
    viz.plot_confusion_matrix(y, (p > .5).astype(int), "LR", tmp_path / "6.png")
    viz.plot_feature_importance(model, tmp_path / "7.png")
    viz.plot_location_timeseries(test, p, thr[loc], loc, "LR", tmp_path / "8.png")
    # Assert: every PNG exists and is non-trivial
    assert all(_ok(tmp_path / f"{i}.png") for i in range(1, 9))
