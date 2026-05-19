from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression


class TwoStageMagnitudeModel(BaseEstimator, RegressorMixin):
    """Predict event occurrence first, then expected maximum magnitude.

    Stage 1 estimates P(event in next 7 days). Stage 2 estimates max magnitude
    conditional on event occurrence. The default prediction is the expected
    value: probability * conditional magnitude.
    """

    def __init__(
        self,
        classifier=None,
        regressor=None,
        threshold: float = 0.5,
        mode: str = "expected",
    ):
        self.classifier = classifier
        self.regressor = regressor
        self.threshold = threshold
        self.mode = mode

    def fit(self, X, y):
        y_array = np.asarray(y, dtype=float)
        event_target = (y_array > 0).astype(int)

        classifier = self.classifier
        if classifier is None:
            classifier = RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )

        regressor = self.regressor
        if regressor is None:
            regressor = LinearRegression()

        if len(np.unique(event_target)) < 2:
            classifier = DummyClassifier(strategy="constant", constant=int(event_target[0]))

        self.classifier_ = clone(classifier)
        self.classifier_.fit(X, event_target)

        event_mask = event_target == 1
        if event_mask.any():
            self.regressor_ = clone(regressor)
            self.regressor_.fit(np.asarray(X)[event_mask], y_array[event_mask])
        else:
            self.regressor_ = DummyRegressor(strategy="constant", constant=0.0)
            self.regressor_.fit(X, y_array)

        return self

    def event_probability(self, X) -> np.ndarray:
        if hasattr(self.classifier_, "predict_proba"):
            proba = self.classifier_.predict_proba(X)
            classes = list(getattr(self.classifier_, "classes_", []))
            if 1 in classes:
                return proba[:, classes.index(1)]
            return np.zeros(len(proba), dtype=float)
        return np.asarray(self.classifier_.predict(X), dtype=float)

    def predict(self, X):
        event_proba = self.event_probability(X)
        conditional_magnitude = np.clip(np.asarray(self.regressor_.predict(X), dtype=float), 0.0, None)

        if self.mode == "expected":
            return event_proba * conditional_magnitude
        if self.mode == "gated":
            return np.where(event_proba >= self.threshold, conditional_magnitude, 0.0)
        raise ValueError("mode must be either 'expected' or 'gated'")
