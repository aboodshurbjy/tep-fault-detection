"""Fit both detectors and calibrate their alarm thresholds.

    python -m tep.train

Runs from the training files are split by run number into train, tune and
calibrate sets (see data.py). The testing files are not opened here.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from . import config
from .alarms import calibrate_threshold, false_alarms_per_24h
from .data import assert_complete_runs, assert_disjoint_runs, load_runs, split_runs
from .features import build_features
from .gbm import GBMDetector
from .pca_monitor import PCAMonitor
from .scoring import MODELS, Scorer, score_frame


def train(runs_per_fault: int = 80, tune_runs_per_fault: int = 15, seed: int = 0,
          target_per_24h: float = config.TARGET_FALSE_ALARMS_PER_24H,
          data_dir=None, model_dir=None, save: bool = True) -> Scorer:
    started = time.time()
    onset = config.ONSET_SAMPLES["training"]

    normal = load_runs("fault_free_training", data_dir=data_dir)
    split = split_runs(normal["simulationRun"].unique(), seed=seed)
    by_run = normal["simulationRun"]
    normal_train = normal[by_run.isin(split.train)]
    normal_tune = normal[by_run.isin(split.tune)]
    normal_calibrate = normal[by_run.isin(split.calibrate)]

    faulty_train = load_runs("faulty_training", runs=split.train[:runs_per_fault], data_dir=data_dir)
    faulty_tune = load_runs("faulty_training", runs=split.tune[:tune_runs_per_fault], data_dir=data_dir)

    train_frame = pd.concat([normal_train, faulty_train], ignore_index=True)
    tune_frame = pd.concat([normal_tune, faulty_tune], ignore_index=True)
    assert_disjoint_runs(train=train_frame, tune=tune_frame, calibrate=normal_calibrate)
    for frame in (train_frame, tune_frame, normal_calibrate):
        assert_complete_runs(frame)
    print(f"runs: train {len(split.train)} normal + {faulty_train.groupby(config.RUN_KEYS).ngroups} faulty | "
          f"tune {len(split.tune)} + {faulty_tune.groupby(config.RUN_KEYS).ngroups} | "
          f"calibrate {len(split.calibrate)} normal")

    print("fitting PCA on fault-free training runs")
    pca = PCAMonitor.fit(normal_train[config.SENSORS].to_numpy(), config.SENSORS)
    print(f"  {pca.n_components} components, T2 limit {pca.t2_limit:.1f}, SPE limit {pca.spe_limit:.1f}")

    print("building window features")
    x_train, meta_train = build_features(train_frame, config.WINDOW, onset)
    x_tune, meta_tune = build_features(tune_frame, config.WINDOW, onset)
    print(f"  train {x_train.shape}, {meta_train['label'].mean():.0%} fault | tune {x_tune.shape}")

    print("fitting LightGBM")
    gbm = GBMDetector.fit(x_train, meta_train["label"].to_numpy(), x_tune, meta_tune["label"].to_numpy(),
                          config.SENSORS, config.WINDOW)
    print(f"  {gbm.booster.best_iteration} trees")

    scorer = Scorer(pca, gbm, thresholds={m: float("inf") for m in MODELS})
    calibration_hours = len(normal_calibrate) * config.SAMPLE_MINUTES / 60
    print(f"calibrating thresholds on {calibration_hours:,.0f} h of fault-free operation "
          f"(target {target_per_24h:.3f} false alarms per 24 h)")
    achieved = {}
    for model in MODELS:
        runs = list(score_frame(scorer, model, normal_calibrate).values())
        scorer.thresholds[model] = calibrate_threshold(runs, target_per_24h)
        achieved[model] = false_alarms_per_24h(runs, scorer.thresholds[model])
        print(f"  {model}: threshold {scorer.thresholds[model]:.4f}, "
              f"{achieved[model]:.3f} false alarms per 24 h on the calibration runs")

    scorer.info = {
        "dataset": config.DATASET_DOI,
        "seed": seed,
        "split": split.as_dict(),
        "faulty_train_runs_per_fault": min(runs_per_fault, len(split.train)),
        "faulty_tune_runs_per_fault": min(tune_runs_per_fault, len(split.tune)),
        "train_rows": int(len(x_train)),
        "pca_components": pca.n_components,
        "lightgbm_trees": int(gbm.booster.best_iteration),
        "target_false_alarms_per_24h": target_per_24h,
        "calibration_hours": calibration_hours,
        "calibration_false_alarms_per_24h": achieved,
    }
    if save:
        print(f"saved to {scorer.save(model_dir)}")
    print(f"done in {time.time() - started:.0f} s")
    return scorer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-per-fault", type=int, default=80, help="faulty training runs per fault type")
    parser.add_argument("--tune-runs-per-fault", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(args.runs_per_fault, args.tune_runs_per_fault, args.seed)


if __name__ == "__main__":
    main()
