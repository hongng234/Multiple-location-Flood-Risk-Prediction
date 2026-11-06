"""Guards against data leakage — these invariants must never break."""
import numpy as np
import pandas as pd

from src.features import feature_columns
from src.preprocessing import compute_thresholds, time_series_cv_splits, time_split
from tests.conftest import TEST_START


def test_train_targets_end_before_test_starts(labelled_split):
    """No train label refers to a test-period day, and train/test do not overlap in time."""
    train, test, _ = labelled_split
    assert train["target_date"].max() < pd.Timestamp(TEST_START)   # no train label inside test period
    assert test["date"].min() >= pd.Timestamp(TEST_START)
    assert train["date"].max() < test["date"].min()


def test_thresholds_use_train_only(feature_df):
    """Changing test-period discharge must not change the thresholds."""
    train, _ = time_split(feature_df, TEST_START)
    thr = compute_thresholds(train, 90)

    # Inflate only test-period discharge 100×; thresholds must stay identical
    tampered = feature_df.copy()
    tampered.loc[tampered["date"] >= TEST_START, "q_t"] *= 100
    train2, _ = time_split(tampered, TEST_START)
    assert compute_thresholds(train2, 90) == thr


def test_thresholds_are_per_location(labelled_split):
    """Each location's threshold equals the P90 of its OWN train discharge."""
    train, _, thr = labelled_split
    for loc, g in train.groupby("location"):
        assert thr[loc] == np.percentile(g["q_t"], 90)


def test_label_and_its_source_are_not_features(labelled_split):
    """flood_risk and q_future are never part of the model inputs."""
    train, _, _ = labelled_split
    cols = feature_columns(train)
    assert "flood_risk" not in cols and "q_future" not in cols


def test_no_feature_is_a_copy_of_the_label_source(labelled_split):
    """A near-perfect correlation with q_future would signal a leaked future value."""
    train, _, _ = labelled_split
    # For every feature, count how often it is numerically equal to q_future
    for c in feature_columns(train):
        same_day = np.isclose(train[c], train["q_future"]).mean()
        assert same_day < 0.5, f"{c} equals q_future on {same_day:.0%} of rows"


def test_cv_folds_are_chronological_with_gap(labelled_split):
    """Every CV fold validates on days strictly after its training days, with a gap, never splitting a day."""
    train, _, _ = labelled_split
    train = train.sort_values(["date", "location"]).reset_index(drop=True)
    # Check each fold: train period ends, then at least one empty day, then validation starts
    for tr, va in time_series_cv_splits(train["date"], n_splits=3, gap_days=1):
        tr_max, va_min = train.loc[tr, "date"].max(), train.loc[va, "date"].min()
        assert tr_max < va_min
        assert (va_min - tr_max).days >= 2   # gap of 1 full day between folds
        # every day is entirely in one side (all locations together)
        assert not set(train.loc[tr, "date"]) & set(train.loc[va, "date"])


def test_preprocessing_is_fit_inside_pipeline(tmp_config):
    """ML models must carry their own preprocessor so CV refits it per fold."""
    from sklearn.pipeline import Pipeline

    from src.models import get_models
    from src.preprocessing import build_preprocessor

    models = get_models(build_preprocessor(["a"], ["location"]))
    # Each ML model must be a Pipeline whose first step is the preprocessor
    for name in ["LogisticRegression", "RandomForest", "GradientBoosting"]:
        assert isinstance(models[name], Pipeline)
        assert models[name].steps[0][0] == "prep"
