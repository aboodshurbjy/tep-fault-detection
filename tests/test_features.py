"""Feature windows: correct values, strictly causal, never across runs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_runs
from tep import config
from tep.features import STATS, build_features, feature_names, window_matrix


def test_shape_and_feature_names():
    values = np.random.default_rng(0).normal(size=(30, 4))
    features = window_matrix(values, window=10)
    assert features.shape == (21, 12)
    assert features.dtype == np.float32
    names = feature_names(["a", "b", "c", "d"])
    assert names[:4] == ["a__mean", "b__mean", "c__mean", "d__mean"]
    assert names[4] == "a__std" and names[-1] == "d__slope"
    assert len(feature_names()) == len(STATS) * len(config.SENSORS) == 156


def test_values_match_an_independent_rolling_calculation():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(40, 3)).cumsum(axis=0)
    window = 8
    features = window_matrix(values, window)
    frame = pd.DataFrame(values)
    expected_mean = frame.rolling(window).mean().dropna().to_numpy()
    expected_std = frame.rolling(window).std(ddof=0).dropna().to_numpy()
    expected_slope = frame.rolling(window).apply(lambda w: np.polyfit(np.arange(window), w, 1)[0]).dropna().to_numpy()
    np.testing.assert_allclose(features[:, 0:3], expected_mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(features[:, 3:6], expected_std, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(features[:, 6:9], expected_slope, rtol=1e-5, atol=1e-5)


def test_ramp_and_constant_signals():
    ramp = 2.5 * np.arange(20) + 7
    constant = np.full(20, 3.0)
    features = window_matrix(np.column_stack([ramp, constant]), window=5)
    mean, std, slope = features[:, 0:2], features[:, 2:4], features[:, 4:6]
    np.testing.assert_allclose(slope[:, 0], 2.5, rtol=1e-6)        # slope is per sample
    np.testing.assert_allclose(slope[:, 1], 0.0, atol=1e-9)
    np.testing.assert_allclose(std[:, 1], 0.0, atol=1e-9)
    np.testing.assert_allclose(mean[:, 1], 3.0)
    np.testing.assert_allclose(mean[0, 0], ramp[:5].mean())


def test_row_i_describes_the_window_ending_at_sample_i_plus_window_minus_1():
    values = np.arange(12, dtype=float).reshape(-1, 1)
    features = window_matrix(values, window=4)
    assert features[0, 0] == pytest.approx(values[0:4].mean())
    assert features[-1, 0] == pytest.approx(values[8:12].mean())


def test_windows_are_causal():
    """Changing the future must not change any feature already computed."""
    rng = np.random.default_rng(2)
    values = rng.normal(size=(50, 3))
    changed = values.copy()
    changed[30:] += 100.0
    window = 10
    before, after = window_matrix(values, window), window_matrix(changed, window)
    last_unaffected_row = 29 - (window - 1)          # window ending at sample index 29
    np.testing.assert_array_equal(before[:last_unaffected_row + 1], after[:last_unaffected_row + 1])
    assert not np.allclose(before[last_unaffected_row + 1], after[last_unaffected_row + 1])


def test_too_few_samples_gives_no_rows():
    assert window_matrix(np.zeros((9, 4)), window=10).shape == (0, 12)
    assert window_matrix(np.zeros((10, 4)), window=10).shape == (1, 12)


def test_invalid_input_is_rejected():
    with pytest.raises(ValueError):
        window_matrix(np.zeros(10), window=5)
    with pytest.raises(ValueError):
        window_matrix(np.zeros((10, 2)), window=1)


def test_windows_never_cross_a_run_boundary():
    window = 10
    frame = pd.concat([make_runs(2, length=30, seed=1), make_runs(1, length=25, fault=3, seed=2)], ignore_index=True)
    features, meta = build_features(frame, window)
    assert len(features) == len(meta) == 2 * (30 - window + 1) + (25 - window + 1)
    # no feature row exists for the first window-1 samples of any run
    assert meta.groupby(["faultNumber", "simulationRun"])["sample"].min().eq(window).all()
    # each run's block equals that run featurised entirely on its own
    for (fault, run), group in frame.groupby(["faultNumber", "simulationRun"], sort=False):
        mask = (meta["faultNumber"] == fault) & (meta["simulationRun"] == run)
        np.testing.assert_array_equal(features[mask.to_numpy()], window_matrix(group[config.SENSORS].to_numpy(), window))


def test_meta_sample_is_the_last_sample_in_the_window():
    frame = make_runs(1, length=30)
    features, meta = build_features(frame, window=10)
    assert meta["sample"].tolist() == list(range(10, 31))
    np.testing.assert_allclose(features[0, 0], frame["xmeas_1"].iloc[:10].mean(), rtol=1e-5)


def test_labels_follow_the_fault_onset():
    frame = pd.concat([make_runs(1, length=40), make_runs(1, length=40, fault=4)], ignore_index=True)
    _, meta = build_features(frame, window=10, onset_samples=20)
    normal, faulty = meta[meta["faultNumber"] == 0], meta[meta["faultNumber"] == 4]
    assert normal["label"].eq(0).all()
    assert faulty.loc[faulty["sample"] <= 20, "label"].eq(0).all()      # fault not yet switched on
    assert faulty.loc[faulty["sample"] > 20, "label"].eq(1).all()
    assert "label" not in build_features(frame, window=10)[1].columns


def test_unsorted_or_gapped_runs_are_rejected():
    frame = make_runs(1, length=30)
    with pytest.raises(ValueError, match="contiguous"):
        build_features(frame.iloc[::-1], window=10)
    with pytest.raises(ValueError, match="contiguous"):
        build_features(frame.drop(index=15), window=10)
