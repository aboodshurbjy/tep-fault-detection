"""Fixtures.

Unit tests run on small generated arrays shaped like the dataset, so they
need no download and finish in seconds. They test mechanics only (windows,
splits, alarm logic, the API contract); every reported result comes from
the real dataset. Tests marked `real_data` run only when it is present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tep import config
from tep.alarms import calibrate_threshold
from tep.data import table_path
from tep.features import build_features
from tep.gbm import GBMDetector
from tep.pca_monitor import PCAMonitor
from tep.scoring import MODELS, Scorer, score_frame

SHIFTED = ("xmeas_9", "xmv_10")        # sensors the generated fault moves; both belong to the reactor
ONSET = 20

real_data = pytest.mark.skipif(
    not table_path("fault_free_training").exists(), reason="dataset not downloaded (python -m tep.download)")


def make_runs(n_runs: int, length: int = 60, fault: int = 0, onset: int = ONSET, seed: int = 0,
              first_run: int = 1) -> pd.DataFrame:
    """Correlated noise in the dataset's column layout; a fault shifts two reactor sensors."""
    rng = np.random.default_rng(seed + 1000 * fault)
    loadings = np.random.default_rng(99).normal(size=(5, len(config.SENSORS)))
    frames = []
    for run in range(first_run, first_run + n_runs):
        values = rng.normal(size=(length, 5)) @ loadings + rng.normal(scale=0.5, size=(length, len(config.SENSORS)))
        values += 50.0
        if fault:
            for sensor in SHIFTED:
                values[onset:, config.SENSORS.index(sensor)] += 12.0
        frame = pd.DataFrame(values.astype(np.float32), columns=config.SENSORS)
        frame.insert(0, "sample", np.arange(1, length + 1, dtype=np.int16))
        frame.insert(0, "simulationRun", np.int16(run))
        frame.insert(0, "faultNumber", np.int16(fault))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def tiny_scorer() -> Scorer:
    """Both detectors trained on generated runs: enough to exercise scoring end to end."""
    normal_train = make_runs(12, seed=1)
    faulty_train = make_runs(12, fault=1, seed=2)
    tune = pd.concat([make_runs(3, seed=3, first_run=100), make_runs(3, fault=1, seed=4, first_run=100)])
    calibrate = make_runs(8, seed=5, first_run=200)

    pca = PCAMonitor.fit(normal_train[config.SENSORS].to_numpy())
    x_train, meta_train = build_features(pd.concat([normal_train, faulty_train]), config.WINDOW, ONSET)
    x_tune, meta_tune = build_features(tune, config.WINDOW, ONSET)
    gbm = GBMDetector.fit(x_train, meta_train["label"].to_numpy(), x_tune, meta_tune["label"].to_numpy(),
                          params={"min_data_in_leaf": 5, "num_leaves": 7}, num_rounds=40, log_every=0)
    scorer = Scorer(pca, gbm, thresholds={model: float("inf") for model in MODELS})
    for model in MODELS:
        runs = list(score_frame(scorer, model, calibrate).values())
        scorer.thresholds[model] = calibrate_threshold(runs, target_per_24h=0.0)
    return scorer
