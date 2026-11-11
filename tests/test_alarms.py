from __future__ import annotations

import numpy as np
import pytest

from tep.alarms import (alarm_state, calibrate_threshold, count_onsets, debounce, detection_delay,
                        false_alarms_per_24h, summarize_delays)


def test_debounce_needs_consecutive_samples():
    over = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1], dtype=bool)
    assert debounce(over, 3).tolist() == [False, False, False, False, False, True, True, False, False]
    assert debounce(over, 1).tolist() == over.tolist()
    assert debounce(np.array([True, True]), 3).tolist() == [False, False]


def test_missing_scores_never_alarm():
    scores = np.array([np.nan, np.nan, 5.0, 5.0, 5.0])
    assert alarm_state(scores, threshold=1.0, persist=3).tolist() == [False, False, False, False, True]


def test_count_onsets_counts_each_alarm_once():
    assert count_onsets(np.array([0, 1, 1, 0, 0, 1, 0, 1, 1], dtype=bool)) == 3
    assert count_onsets(np.array([1, 1, 0], dtype=bool)) == 1
    assert count_onsets(np.array([], dtype=bool)) == 0


def test_false_alarm_rate_is_per_24_hours():
    # two runs of 240 samples at 3 minutes = 24 hours in total, three alarm onsets
    run_a = np.zeros(240); run_a[10:20] = 9; run_a[100:105] = 9
    run_b = np.zeros(240); run_b[50:60] = 9
    assert false_alarms_per_24h([run_a, run_b], threshold=1.0, persist=3, sample_minutes=3) == pytest.approx(3.0)


def test_calibrated_threshold_meets_the_target_and_is_not_needlessly_high():
    rng = np.random.default_rng(0)
    runs = [rng.normal(size=480) for _ in range(40)]            # 960 hours
    target = 0.5
    threshold = calibrate_threshold(runs, target, persist=3, sample_minutes=3)
    assert false_alarms_per_24h(runs, threshold, 3, 3) <= target
    assert false_alarms_per_24h(runs, threshold - 0.05, 3, 3) > target


def test_calibration_with_zero_target_silences_every_fault_free_run():
    rng = np.random.default_rng(1)
    runs = [rng.normal(size=200) for _ in range(5)]
    threshold = calibrate_threshold(runs, target_per_24h=0.0)
    assert false_alarms_per_24h(runs, threshold) == 0


def test_detection_delay_is_measured_from_the_fault_onset():
    scores = np.zeros(100)
    scores[40:] = 9                                           # fault at sample 41, score responds at once
    assert detection_delay(scores, 1.0, onset_samples=40, persist=3, sample_minutes=3) == 9   # earliest possible
    scores = np.zeros(100)
    scores[50:] = 9                                           # responds ten samples late
    assert detection_delay(scores, 1.0, onset_samples=40, persist=3, sample_minutes=3) == (10 + 3) * 3


def test_an_alarm_already_ringing_before_the_fault_is_not_a_detection():
    scores = np.zeros(100)
    scores[30:42] = 9                                         # false alarm that straddles the onset
    assert detection_delay(scores, 1.0, onset_samples=40, persist=3) is None
    scores[70:] = 9
    assert detection_delay(scores, 1.0, onset_samples=40, persist=3, sample_minutes=3) == (30 + 3) * 3


def test_undetected_runs_count_against_the_median():
    summary = summarize_delays([9.0, 12.0, None, None, None])
    assert summary["detection_rate"] == pytest.approx(0.4)
    assert summary["median_delay_min"] == np.inf
    assert summary["mean_delay_detected_min"] == pytest.approx(10.5)
    assert summarize_delays([9.0, 12.0, 30.0])["median_delay_min"] == 12.0
