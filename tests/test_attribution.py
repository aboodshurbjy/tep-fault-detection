from __future__ import annotations

import numpy as np
import pytest

from conftest import SHIFTED, make_runs
from tep import config
from tep.attribution import FAULT_INFO, SENSOR_INFO, UNIT_CHECKS, explain, suspect_unit, top_sensors
from tep.pca_monitor import PCAMonitor


def test_lookup_table_covers_every_sensor_and_fault():
    assert list(SENSOR_INFO) == config.SENSORS
    assert {unit for _, unit in SENSOR_INFO.values()} == set(UNIT_CHECKS)
    assert {"Reactor", "Separator", "Stripper", "Compressor"} <= set(UNIT_CHECKS)
    assert set(FAULT_INFO) == set(config.FAULTS)
    assert SENSOR_INFO["xmeas_9"] == ("Reactor temperature", "Reactor")
    assert SENSOR_INFO["xmv_5"][1] == "Compressor"


def test_top_sensors_are_ranked_and_ignore_negative_contributions():
    contributions = np.zeros(len(config.SENSORS))
    contributions[config.SENSORS.index("xmeas_9")] = 6.0
    contributions[config.SENSORS.index("xmv_10")] = 3.0
    contributions[config.SENSORS.index("xmeas_15")] = 1.0
    contributions[config.SENSORS.index("xmeas_1")] = -50.0
    top = top_sensors(contributions, k=5)
    assert [entry["sensor"] for entry in top] == ["xmeas_9", "xmv_10", "xmeas_15"]
    assert [entry["share"] for entry in top] == [0.6, 0.3, 0.1]
    assert top[0]["unit"] == "Reactor" and top[0]["description"] == "Reactor temperature"


def test_suspect_unit_sums_shares_across_sensors():
    contributions = np.zeros(len(config.SENSORS))
    contributions[config.SENSORS.index("xmeas_15")] = 4.0          # one strong stripper sensor
    contributions[config.SENSORS.index("xmeas_9")] = 3.0           # two reactor sensors that add up to more
    contributions[config.SENSORS.index("xmv_10")] = 3.0
    result = explain(contributions)
    assert result["unit"] == "Reactor"
    assert result["check"] == UNIT_CHECKS["Reactor"]


def test_no_positive_contribution_gives_no_suspect():
    assert top_sensors(np.zeros(len(config.SENSORS))) == []
    assert suspect_unit([]) is None
    assert explain(-np.ones(len(config.SENSORS)))["unit"] is None


@pytest.fixture(scope="module")
def monitor() -> PCAMonitor:
    return PCAMonitor.fit(make_runs(15, seed=7)[config.SENSORS].to_numpy())


def test_pca_contributions_add_up_to_their_statistic(monitor):
    faulty = make_runs(1, fault=1, seed=8)[config.SENSORS].to_numpy()
    for sample in (faulty[5], faulty[-1]):
        detail = monitor.contributions(sample)
        total = detail["spe"] if detail["statistic"] == "SPE" else detail["t2"]
        assert detail["values"].sum() == pytest.approx(total, rel=1e-6)
        t2, spe = monitor.statistics(sample)
        assert (detail["t2"], detail["spe"]) == (pytest.approx(t2[0]), pytest.approx(spe[0]))


def test_pca_scores_rise_after_a_fault_and_point_at_the_shifted_sensors(monitor):
    faulty = make_runs(1, fault=1, seed=9)[config.SENSORS].to_numpy()
    scores = monitor.score(faulty)
    assert np.median(scores[25:]) > 5 * np.median(scores[:20])
    top = explain(monitor.contributions(faulty[-1])["values"])
    assert {entry["sensor"] for entry in top["top_sensors"][:2]} == set(SHIFTED)
    assert top["unit"] == "Reactor"


def test_pca_limits_hold_on_fresh_normal_data(monitor):
    fresh = make_runs(10, seed=10, first_run=50)[config.SENSORS].to_numpy()
    assert np.mean(monitor.score(fresh) > 1) < 0.1
