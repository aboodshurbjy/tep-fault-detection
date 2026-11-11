"""Trained detectors plus their alarm thresholds, saved and loaded as one bundle.

The API, the dashboard and the evaluation all score through this module,
so there is one definition of "score", "alarm" and "which sensors".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from . import config
from .alarms import alarm_state
from .attribution import explain
from .gbm import GBMDetector
from .pca_monitor import PCAMonitor

MODELS = ("pca", "lightgbm")


class ModelNotFound(FileNotFoundError):
    pass


@dataclass
class Scorer:
    pca: PCAMonitor
    gbm: GBMDetector
    thresholds: dict[str, float]
    window: int = config.WINDOW
    persist: int = config.PERSIST
    sample_minutes: float = config.SAMPLE_MINUTES
    sensors: list[str] = field(default_factory=lambda: list(config.SENSORS))
    info: dict = field(default_factory=dict)

    @property
    def min_samples(self) -> int:
        """Fewest samples that produce one score from both models."""
        return self.window

    @property
    def samples_for_alarm(self) -> int:
        """Fewest samples that let the debounced alarm be evaluated."""
        return self.window + self.persist - 1

    def detector(self, model: str):
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")
        return self.pca if model == "pca" else self.gbm

    def score_run(self, model: str, values: np.ndarray) -> np.ndarray:
        """One anomaly score per sample of (n, V); NaN where the model has no full window yet."""
        return self.detector(model).score_run(np.asarray(values, dtype=np.float64))

    def alarms(self, model: str, scores: np.ndarray) -> np.ndarray:
        return alarm_state(scores, self.thresholds[model], self.persist)

    def explain(self, model: str, values: np.ndarray, k: int = 5) -> dict:
        """Top sensors and unit to check for the latest sample of (n, V)."""
        values = np.asarray(values, dtype=np.float64)
        if model == "pca":
            detail = self.pca.contributions(values[-1])
            return {**explain(detail["values"], self.sensors, k), "statistic": detail["statistic"]}
        return explain(self.gbm.contributions(values)["values"], self.sensors, k)

    def score_window(self, values: np.ndarray, k: int = 5) -> dict:
        """Score the most recent sample given a window of readings (n, V), oldest first.

        `alarm` applies the same persistence rule as the evaluation: the
        score must be over the threshold for the last `persist` samples.
        With too few samples to judge that, `alarm` is None and only
        `over_threshold` (the latest sample alone) is reported.
        """
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.sensors):
            raise ValueError(f"expected rows of {len(self.sensors)} sensor values, got shape {values.shape}")
        if len(values) < self.min_samples:
            raise ValueError(f"need at least {self.min_samples} samples, got {len(values)}")
        if not np.isfinite(values).all():
            raise ValueError("readings contain missing or non-finite values")

        result = {}
        for model in MODELS:
            scores = self.score_run(model, values)
            threshold = self.thresholds[model]
            valid = scores[~np.isnan(scores)]
            over = bool(valid[-1] > threshold)
            alarm = bool(self.alarms(model, scores)[-1]) if len(valid) >= self.persist else None
            entry = {
                "score": float(valid[-1]),
                "threshold": float(threshold),
                "over_threshold": over,
                "alarm": alarm,
                "attribution": self.explain(model, values, k) if over else None,
            }
            if model == "pca":
                t2, spe = self.pca.statistics(values[-1])
                entry["t2_ratio"] = float(t2[0] / self.pca.t2_limit)
                entry["spe_ratio"] = float(spe[0] / self.pca.spe_limit)
            result[model] = entry
        return result

    # ------------------------------------------------------------ persistence

    def save(self, model_dir: Path | None = None) -> Path:
        model_dir = Path(model_dir or config.MODEL_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pca, model_dir / "pca.joblib")
        self.gbm.save(model_dir / "lightgbm.txt")
        (model_dir / "meta.json").write_text(json.dumps({
            "thresholds": self.thresholds, "window": self.window, "persist": self.persist,
            "sample_minutes": self.sample_minutes, "sensors": self.sensors, "info": self.info,
        }, indent=2))
        return model_dir

    @classmethod
    def load(cls, model_dir: Path | None = None) -> "Scorer":
        model_dir = Path(model_dir or config.MODEL_DIR)
        meta_path = model_dir / "meta.json"
        if not meta_path.exists():
            raise ModelNotFound(f"no trained models in {model_dir}. Run `python -m tep.train` first.")
        meta = json.loads(meta_path.read_text())
        return cls(
            pca=joblib.load(model_dir / "pca.joblib"),
            gbm=GBMDetector.load(model_dir / "lightgbm.txt", meta["sensors"], meta["window"]),
            thresholds=meta["thresholds"], window=meta["window"], persist=meta["persist"],
            sample_minutes=meta["sample_minutes"], sensors=meta["sensors"], info=meta.get("info", {}),
        )


def score_frame(scorer: Scorer, model: str, frame) -> dict[tuple[int, int], np.ndarray]:
    """Score every run in a frame in one batch: (fault, run) -> one score per sample."""
    from .features import build_features

    sizes = frame.groupby(config.RUN_KEYS, sort=False).size()
    if model == "pca":
        flat = scorer.pca.score(frame[scorer.sensors].to_numpy())
        bounds = np.concatenate([[0], np.cumsum(sizes.to_numpy())])
        return {key: flat[bounds[i]:bounds[i + 1]] for i, key in enumerate(sizes.index)}

    features, meta = build_features(frame, scorer.window, sensors=scorer.sensors)
    flat = scorer.gbm.score(features) if len(features) else np.empty(0)
    out, start = {}, 0
    for key, size in sizes.items():
        scores = np.full(size, np.nan)
        n = max(size - scorer.window + 1, 0)
        scores[scorer.window - 1:] = flat[start:start + n]
        start += n
        out[key] = scores
    return out
