"""LightGBM fault classifier on windowed features, with SHAP attribution."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np

from . import config
from .features import STATS, feature_names, window_matrix

DEFAULT_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.5,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 0,
}


@dataclass
class GBMDetector:
    booster: lgb.Booster
    sensors: list[str] = field(default_factory=lambda: list(config.SENSORS))
    window: int = config.WINDOW
    _explainer: object = field(default=None, repr=False, compare=False)

    name = "lightgbm"

    @classmethod
    def fit(cls, x_train: np.ndarray, y_train: np.ndarray, x_tune: np.ndarray, y_tune: np.ndarray,
            sensors=None, window: int = config.WINDOW, params: dict | None = None,
            num_rounds: int = 600, early_stopping: int = 30, log_every: int = 50) -> "GBMDetector":
        """Train normal-versus-fault, stopping when the tuning runs stop improving."""
        sensors = list(sensors or config.SENSORS)
        names = feature_names(sensors)
        train = lgb.Dataset(x_train, y_train, feature_name=names, free_raw_data=False)
        tune = lgb.Dataset(x_tune, y_tune, feature_name=names, reference=train)
        booster = lgb.train(
            {**DEFAULT_PARAMS, **(params or {})}, train, num_boost_round=num_rounds, valid_sets=[tune],
            valid_names=["tune"],
            callbacks=[lgb.early_stopping(early_stopping, verbose=False), lgb.log_evaluation(log_every)],
        )
        return cls(booster, sensors, window)

    def score(self, features: np.ndarray) -> np.ndarray:
        """Probability that the fault is active, one per feature row."""
        return self.booster.predict(np.atleast_2d(features))

    def score_run(self, values: np.ndarray) -> np.ndarray:
        """Probability per sample of a run (n, V); NaN until the first full window."""
        scores = np.full(len(values), np.nan)
        features = window_matrix(values, self.window)
        if len(features):
            scores[self.window - 1:] = self.score(features)
        return scores

    def shap_values(self, features: np.ndarray) -> np.ndarray:
        """SHAP values (log-odds of "fault") per feature, shape (n, n_features)."""
        if self._explainer is None:
            import shap

            self._explainer = shap.TreeExplainer(self.booster)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)   # shap notes its output format per call
            values = self._explainer.shap_values(np.atleast_2d(features))
        if isinstance(values, list):            # older shap: one array per class
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:                    # (n, n_features, n_classes)
            values = values[..., -1]
        return values

    def contributions(self, window_values: np.ndarray) -> dict:
        """Per-sensor SHAP for the latest window of (>= window, V) raw samples.

        A sensor's value is the sum of the SHAP values of its mean, std and
        slope features. Positive pushes towards "fault".
        """
        features = window_matrix(np.asarray(window_values)[-self.window:], self.window)
        if len(features) == 0:
            raise ValueError(f"need at least {self.window} samples to explain a window")
        shap_row = self.shap_values(features[-1:])[0]
        per_stat = shap_row.reshape(len(STATS), len(self.sensors))
        return {"values": per_stat.sum(axis=0), "by_stat": dict(zip(STATS, per_stat))}

    def save(self, path: Path) -> None:
        self.booster.save_model(str(path))

    @classmethod
    def load(cls, path: Path, sensors=None, window: int = config.WINDOW) -> "GBMDetector":
        return cls(lgb.Booster(model_file=str(path)), list(sensors or config.SENSORS), window)
