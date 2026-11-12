"""Mean shift and variability ratio of a fault against fault-free runs."""

from __future__ import annotations

import numpy as np

from conftest import ONSET, SHIFTED, make_runs
from tep import config
from tep.signature import signature, signatures_markdown


def test_a_shifted_sensor_stands_out_and_the_rest_do_not():
    normal = make_runs(30, seed=1)
    table = signature(normal, make_runs(30, fault=1, seed=2), ONSET)
    assert list(table.index) == config.SENSORS
    others = table.drop(index=list(SHIFTED))
    assert (table.loc[list(SHIFTED), "mean_shift_sd"] > 3).all()
    assert others["mean_shift_sd"].abs().max() < 0.3
    assert np.allclose(table["variability_ratio"], 1.0, atol=0.1)      # the generated fault adds no noise


def test_noisier_runs_raise_the_variability_ratio_without_a_mean_shift():
    normal = make_runs(30, seed=1)
    noisy = make_runs(30, seed=1)
    column = "xmeas_22"
    centre = noisy[column].mean()
    noisy[column] = centre + (noisy[column] - centre) * 2.0
    table = signature(normal, noisy, ONSET)
    assert abs(table.loc[column, "variability_ratio"] - 2.0) < 0.05
    assert abs(table.loc[column, "mean_shift_sd"]) < 0.05


def test_samples_before_onset_are_ignored():
    normal = make_runs(10, seed=1)
    faulty = make_runs(10, seed=1)
    faulty.loc[faulty["sample"] <= ONSET, "xmeas_1"] += 100.0
    assert abs(signature(normal, faulty, ONSET).loc["xmeas_1", "mean_shift_sd"]) < 1e-6


def test_table_names_the_sensor_or_says_none():
    normal = make_runs(10, seed=1)
    text = signatures_markdown({3: signature(normal, make_runs(10, seed=1), ONSET),
                                4: signature(normal, make_runs(10, fault=4, seed=2), ONSET)})
    rows = text.splitlines()[2:]
    assert "none (under 0.05)" in rows[0] and "none (under 1%)" in rows[0]
    assert "Reactor" in rows[1]
