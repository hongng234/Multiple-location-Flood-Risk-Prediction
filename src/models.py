"""Model zoo: baselines + ML classifiers, each wrapped with its own preprocessor.

A "baseline" is a very simple rule used as a yardstick: a machine-learning model is only
worth using if it scores better than these rules.
"""
import numpy as np    # arrays for the baseline's probability output
import pandas as pd   # read a named column from the input table
from sklearn.base import BaseEstimator, ClassifierMixin, clone   # base classes for a custom model; clone = fresh unfitted copy
from sklearn.dummy import DummyClassifier                         # "always guess the most common answer" model
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # tree-ensemble models
from sklearn.linear_model import LogisticRegression              # linear model that outputs probabilities
from sklearn.pipeline import Pipeline                            # chains the preprocessor and the classifier

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
        values = pd.DataFrame(X)[self.column].to_numpy(dtype=float)   # today's discharge / threshold, per row
        p = (values > 1.0).astype(float)                              # 1.0 if above flood level, else 0.0
        return np.column_stack([1 - p, p])                            # two columns, like every sklearn classifier

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
        "PersistenceBaseline": PersistenceBaseline(),           # "tomorrow = today" rule (see class above)
        # Linear model; max_iter = enough optimisation steps to converge
        "LogisticRegression": wrap(LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=random_state)),
        # 300 independent decision trees that vote; depth/leaf limits keep each tree from memorising noise;
        # n_jobs=-1 uses all CPU cores
        "RandomForest": wrap(RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=3, class_weight="balanced",
            random_state=random_state, n_jobs=-1)),
        # 200 small trees built one after another, each correcting the previous ones' errors;
        # learning_rate = how much each new tree is allowed to change the prediction
        "GradientBoosting": wrap(GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.08, max_depth=3, random_state=random_state)),
    }
