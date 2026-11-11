"""Windowed features: rolling mean, standard deviation and slope per sensor.

Windows are trailing (causal): the feature row for sample t uses samples
t-window+1 .. t and nothing later, so the same code can score a live
stream. Windows never cross a run boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from . import config

STATS = ("mean", "std", "slope")


def feature_names(sensors=None) -> list[str]:
    sensors = list(sensors or config.SENSORS)
    return [f"{sensor}__{stat}" for stat in STATS for sensor in sensors]


def window_matrix(values: np.ndarray, window: int = config.WINDOW) -> np.ndarray:
    """Features for every full trailing window of one run.

    values: (T, V) samples in time order.
    Returns (T - window + 1, 3V): all means, then all standard deviations,
    then all slopes. Row i describes the window ending at input row
    i + window - 1. Slope is the least-squares slope in units per sample.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("values must be a 2-D array of shape (samples, sensors)")
    if window < 2:
        raise ValueError("window must be at least 2 samples")
    n_samples, n_sensors = values.shape
    if n_samples < window:
        return np.empty((0, 3 * n_sensors), dtype=np.float32)

    windows = sliding_window_view(values, window, axis=0)      # (T - window + 1, V, window)
    mean = windows.mean(axis=-1)
    std = windows.std(axis=-1)
    time = np.arange(window) - (window - 1) / 2                # centred, so the slope needs no intercept
    slope = windows @ time / (time @ time)
    return np.concatenate([mean, std, slope], axis=1).astype(np.float32)


def build_features(frame: pd.DataFrame, window: int = config.WINDOW, onset_samples: int | None = None,
                   sensors=None) -> tuple[np.ndarray, pd.DataFrame]:
    """Window features for every run in a frame.

    Returns (X, meta). meta has one row per feature row: faultNumber,
    simulationRun, sample (the last sample in the window) and, when
    onset_samples is given, label = 1 where the fault is active at that
    sample. A window that straddles the onset is labelled by its last
    sample, because that is the moment the operator would be alerted.
    """
    sensors = list(sensors or config.SENSORS)
    blocks, metas = [], []
    for (fault, run), group in frame.groupby(config.RUN_KEYS, sort=False):
        samples = group["sample"].to_numpy()
        if not np.array_equal(samples, np.arange(samples[0], samples[0] + len(samples))):
            raise ValueError(f"run (fault {fault}, run {run}) is not sorted, contiguous samples")
        block = window_matrix(group[sensors].to_numpy(), window)
        if len(block) == 0:
            continue
        blocks.append(block)
        metas.append(pd.DataFrame({
            "faultNumber": np.int16(fault),
            "simulationRun": np.int16(run),
            "sample": samples[window - 1:],
        }))
    if not blocks:
        columns = ["faultNumber", "simulationRun", "sample"] + (["label"] if onset_samples is not None else [])
        return np.empty((0, 3 * len(sensors)), dtype=np.float32), pd.DataFrame(columns=columns)
    meta = pd.concat(metas, ignore_index=True)
    if onset_samples is not None:
        meta["label"] = ((meta["faultNumber"] > 0) & (meta["sample"] > onset_samples)).astype(np.int8)
    return np.concatenate(blocks), meta
