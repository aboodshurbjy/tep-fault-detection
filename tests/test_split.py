"""The split is by simulation run, with no run on more than one side."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_runs, real_data
from tep import config
from tep.data import (assert_complete_runs, assert_disjoint_runs, load_runs, run_numbers, split_runs, table_path)


def test_split_is_disjoint_and_covers_every_run():
    split = split_runs(range(1, 501))
    train, tune, calibrate = set(split.train), set(split.tune), set(split.calibrate)
    assert not (train & tune) and not (train & calibrate) and not (tune & calibrate)
    assert train | tune | calibrate == set(range(1, 501))
    assert (len(train), len(tune), len(calibrate)) == (300, 50, 150)


def test_split_is_reproducible_and_seeded():
    assert split_runs(range(1, 101), seed=3) == split_runs(range(1, 101), seed=3)
    assert split_runs(range(1, 101), seed=3) != split_runs(range(1, 101), seed=4)


def test_split_ignores_repeated_run_numbers():
    repeated = list(range(1, 51)) * 20                      # as in a frame holding many rows per run
    assert split_runs(repeated) == split_runs(range(1, 51))


@pytest.mark.parametrize("kwargs", [
    {"tune_fraction": 0.0}, {"calibrate_fraction": 0.0}, {"tune_fraction": 0.5, "calibrate_fraction": 0.5},
])
def test_split_rejects_fractions_that_leave_a_set_empty(kwargs):
    with pytest.raises(ValueError):
        split_runs(range(1, 101), **kwargs)


def test_split_needs_enough_runs():
    with pytest.raises(ValueError):
        split_runs([1, 2])


def test_whole_runs_stay_together_and_the_same_run_number_stays_on_one_side_for_every_fault():
    frame = pd.concat([make_runs(20, length=30, fault=fault, seed=fault) for fault in (0, 1, 2)], ignore_index=True)
    split = split_runs(frame["simulationRun"])
    parts = {name: frame[frame["simulationRun"].isin(runs)] for name, runs in split.as_dict().items()}

    assert sum(len(part) for part in parts.values()) == len(frame)          # every row used exactly once
    assert_disjoint_runs(**parts)
    for part in parts.values():
        assert_complete_runs(part)                                          # no run was cut in two
        runs_by_fault = [set(group["simulationRun"]) for _, group in part.groupby("faultNumber")]
        assert all(runs == runs_by_fault[0] for runs in runs_by_fault)


def test_a_row_level_split_would_be_caught():
    frame = make_runs(10, length=30)
    shuffled = frame.sample(frac=1.0, random_state=0)
    first, second = shuffled.iloc[:150], shuffled.iloc[150:]
    with pytest.raises(AssertionError, match="run leakage"):
        assert_disjoint_runs(train=first, test=second)
    with pytest.raises(AssertionError, match="not a complete"):
        assert_complete_runs(first.sort_values(["faultNumber", "simulationRun", "sample"]))


def test_load_runs_filters_by_run_and_fault(tmp_path):
    (tmp_path / "processed").mkdir()
    frame = pd.concat([make_runs(6, length=20, fault=fault, seed=fault) for fault in (1, 2, 3)], ignore_index=True)
    frame.sample(frac=1.0, random_state=1).to_parquet(table_path("faulty_training", tmp_path), index=False)

    loaded = load_runs("faulty_training", runs=[2, 5], faults=[1, 3], data_dir=tmp_path)
    assert run_numbers(loaded) == {2, 5}
    assert set(loaded["faultNumber"]) == {1, 3}
    assert len(loaded) == 2 * 2 * 20
    assert_complete_runs(loaded)                             # returned in run and sample order
    assert len(load_runs("faulty_training", data_dir=tmp_path)) == len(frame)


def test_load_runs_reports_missing_data_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="tep.download"):
        load_runs("fault_free_training", data_dir=tmp_path)
    with pytest.raises(ValueError):
        load_runs("not_a_table", data_dir=tmp_path)


@real_data
def test_real_data_same_run_number_shares_a_seed_across_faults():
    """Why the split is by run number: run k starts identically for every fault."""
    onset = config.ONSET_SAMPLES["training"]
    normal = load_runs("fault_free_training", runs=[1])
    faulty = load_runs("faulty_training", runs=[1], faults=[1, 4])
    reference = normal[config.SENSORS].to_numpy()[:onset]
    for _, group in faulty.groupby("faultNumber"):
        np.testing.assert_array_equal(group[config.SENSORS].to_numpy()[:onset], reference)


@real_data
def test_real_data_training_and_testing_files_are_different_simulations():
    train = load_runs("fault_free_training", runs=[1, 2, 3])[config.SENSORS].to_numpy()
    test = load_runs("fault_free_testing", runs=[1, 2, 3])
    for _, group in test.groupby("simulationRun"):
        head = group[config.SENSORS].to_numpy()[:20]
        assert not any(np.array_equal(head, train[start:start + 20]) for start in (0, 500, 1000))


@real_data
def test_real_data_saved_model_split_is_leak_free():
    meta_path = config.MODEL_DIR / "meta.json"
    if not meta_path.exists():
        pytest.skip("no trained model")
    split = json.loads(meta_path.read_text())["info"]["split"]
    train, tune, calibrate = (set(split[name]) for name in ("train", "tune", "calibrate"))
    assert not (train & tune) and not (train & calibrate) and not (tune & calibrate)
    assert train | tune | calibrate == set(range(1, 501))
