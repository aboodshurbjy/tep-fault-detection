"""Loading simulation runs and splitting them without leakage.

The unit of independence in this dataset is the simulation run: 500 (or
960) consecutive, strongly autocorrelated samples from one random seed.
Splitting individual rows would put near-identical neighbours on both
sides of the split, so every split here is made on whole runs.

Three disjoint sets are used:

    train        fit the models                      } runs from the
    tune         early stopping for the classifier   } *training* files,
    calibrate    set alarm thresholds                } split by run number
    test         report results                        the *testing* files

A run number is assigned to exactly one of train / tune / calibrate and
that assignment is applied to every fault, so run 17 of fault 4 and run 17
of the fault-free data always land on the same side.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

TABLES = ("fault_free_training", "faulty_training", "fault_free_testing", "faulty_testing")


class DataNotFound(FileNotFoundError):
    pass


def table_path(table: str, data_dir: Path | None = None) -> Path:
    if table not in TABLES:
        raise ValueError(f"unknown table {table!r}; expected one of {TABLES}")
    return Path(data_dir or config.DATA_DIR) / "processed" / f"{table}.parquet"


def load_runs(table: str, runs=None, faults=None, data_dir: Path | None = None) -> pd.DataFrame:
    """Rows of one table, optionally limited to some run numbers and fault numbers.

    Returned rows are ordered by fault, run and sample.
    """
    path = table_path(table, data_dir)
    if not path.exists():
        raise DataNotFound(f"{path} not found. Run `python -m tep.download` first.")
    filters = []
    if runs is not None:
        filters.append(("simulationRun", "in", [int(r) for r in runs]))
    if faults is not None:
        filters.append(("faultNumber", "in", [int(f) for f in faults]))
    frame = pd.read_parquet(path, filters=filters or None)
    return frame.sort_values(config.RUN_KEYS + ["sample"], kind="stable").reset_index(drop=True)


@dataclass(frozen=True)
class RunSplit:
    train: tuple[int, ...]
    tune: tuple[int, ...]
    calibrate: tuple[int, ...]

    def as_dict(self) -> dict[str, list[int]]:
        return {"train": list(self.train), "tune": list(self.tune), "calibrate": list(self.calibrate)}


def split_runs(run_numbers, tune_fraction: float = 0.1, calibrate_fraction: float = 0.3,
               seed: int = 0) -> RunSplit:
    """Assign each run number to exactly one of train / tune / calibrate."""
    unique = np.unique(np.asarray(list(run_numbers), dtype=int))
    if len(unique) < 3:
        raise ValueError("need at least three runs to make three disjoint sets")
    if not (0 < tune_fraction and 0 < calibrate_fraction and tune_fraction + calibrate_fraction < 1):
        raise ValueError("fractions must be positive and leave room for a training set")
    shuffled = np.random.default_rng(seed).permutation(unique)
    n_tune = max(1, round(len(unique) * tune_fraction))
    n_calibrate = max(1, round(len(unique) * calibrate_fraction))
    if n_tune + n_calibrate >= len(unique):
        raise ValueError("not enough runs left for training")
    tune = shuffled[:n_tune]
    calibrate = shuffled[n_tune:n_tune + n_calibrate]
    train = shuffled[n_tune + n_calibrate:]
    return RunSplit(*(tuple(sorted(int(r) for r in part)) for part in (train, tune, calibrate)))


def run_numbers(frame: pd.DataFrame) -> set[int]:
    return set(int(r) for r in frame["simulationRun"].unique())


def assert_disjoint_runs(**frames: pd.DataFrame) -> None:
    """Raise if any simulation run number appears in more than one of the given frames."""
    names = list(frames)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            shared = run_numbers(frames[first]) & run_numbers(frames[second])
            if shared:
                raise AssertionError(
                    f"run leakage between {first} and {second}: runs {sorted(shared)[:10]} appear in both")


def assert_complete_runs(frame: pd.DataFrame) -> None:
    """Raise unless every run is whole: samples 1..n with no gaps or repeats."""
    for key, samples in frame.groupby(config.RUN_KEYS, sort=False)["sample"]:
        values = samples.to_numpy()
        if not np.array_equal(values, np.arange(1, len(values) + 1)):
            raise AssertionError(f"run {key} is not a complete, ordered sequence of samples")
