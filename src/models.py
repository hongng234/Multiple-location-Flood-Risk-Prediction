"""Model zoo: baselines + ML classifiers, each wrapped with its own preprocessor."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Reference models that are reported but never selected as "best model"
BASELINES = {"DummyBaseline", "PersistenceBaseline"}


class PersistenceBaseline(ClassifierMixin, BaseEstimator):
    """Hydrology's naive forecast: "tomorrow floods if today is already above the threshold".

    Reads the scale-free column `q_t_rel` (= today's discharge / train threshold). An ML model
    is only useful if it beats this rule.

    How: implements the sklearn classifier interface (fit / predict_proba / predict), so it can be
    cross-validated and evaluated exactly like the real models.
    """

    def __init__(self, column: str = "q_t_rel"):
        # Name of the column holding "discharge / threshold" (stored as-is, sklearn convention)
        self.column = column

    def fit(self, X, y):
        # Nothing to learn; only record the class labels sklearn expects
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        # Hard 0/1 "probability" from the persistence rule
        # (above threshold today -> P(flood)=1, otherwise 0; columns are [P(no flood), P(flood)])
        values = pd.DataFrame(X)[self.column].to_numpy(dtype=float)
        p = (values > 1.0).astype(float)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        # Class label = 1 when P(flood) >= 0.5 (here: exactly when the rule fires)
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def get_models(preprocessor, random_state: int = 42) -> dict:
    """Return {name: estimator}. ML models are Pipelines (preprocessing + classifier) so the
    scaler/imputer are fit inside every CV fold and are saved together with the model.

    How: each ML classifier is wrapped with its own fresh copy of the preprocessor; baselines
    are used bare because they read raw columns directly.
    `class_weight="balanced"` up-weights the rare flood class (~10% of days).

    Args:
        preprocessor: unfitted ColumnTransformer from `build_preprocessor`.
        random_state: seed for reproducible models.
    """
    def wrap(clf):
        # clone() gives each pipeline its own unfitted preprocessor (no shared state between models)
        return Pipeline([("prep", clone(preprocessor)), ("clf", clf)])

    return {
        "DummyBaseline": DummyClassifier(strategy="prior"),     # always predicts the base rate
        "PersistenceBaseline": PersistenceBaseline(),
        "LogisticRegression": wrap(LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=random_state)),
        "RandomForest": wrap(RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=3, class_weight="balanced",
            random_state=random_state, n_jobs=-1)),
        "GradientBoosting": wrap(GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.08, max_depth=3, random_state=random_state)),
    }
