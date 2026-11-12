"""Alarm logic and the two numbers an operator judges a detector by.

A detector produces one anomaly score per sample. The alarm is raised when
the score has been above its threshold for `persist` consecutive samples;
this is ordinary alarm debouncing and it applies identically to every
model, so the models are compared on equal terms.

    false alarms per 24 h   alarm onsets during normal operation
    detection delay         minutes from the fault starting to the alarm
"""

from __future__ import annotations

import numpy as np

from . import config


def debounce(over: np.ndarray, persist: int = config.PERSIST) -> np.ndarray:
    """alarm[t] is True when over[t-persist+1 .. t] are all True."""
    over = np.asarray(over, dtype=bool)
    if persist <= 1:
        return over.copy()
    alarm = np.zeros(len(over), dtype=bool)
    if len(over) < persist:
        return alarm
    streak = np.lib.stride_tricks.sliding_window_view(over, persist).all(axis=1)
    alarm[persist - 1:] = streak
    return alarm


def alarm_state(scores: np.ndarray, threshold: float, persist: int = config.PERSIST) -> np.ndarray:
    """Debounced alarm for a score series. Missing scores (NaN) never count as over."""
    scores = np.asarray(scores, dtype=float)
    return debounce(np.nan_to_num(scores, nan=-np.inf) > threshold, persist)


def count_onsets(alarm: np.ndarray) -> int:
    """Number of times the alarm switches on."""
    alarm = np.asarray(alarm, dtype=bool)
    if len(alarm) == 0:
        return 0
    return int(alarm[0]) + int(np.sum(alarm[1:] & ~alarm[:-1]))


def false_alarms_per_24h(score_runs: list[np.ndarray], threshold: float, persist: int = config.PERSIST,
                         sample_minutes: float = config.SAMPLE_MINUTES) -> float:
    """Alarm onsets per 24 hours across fault-free runs."""
    onsets = sum(count_onsets(alarm_state(scores, threshold, persist)) for scores in score_runs)
    hours = sum(len(scores) for scores in score_runs) * sample_minutes / 60
    return onsets / hours * 24 if hours else float("nan")


def calibrate_threshold(score_runs: list[np.ndarray], target_per_24h: float = config.TARGET_FALSE_ALARMS_PER_24H,
                        persist: int = config.PERSIST, sample_minutes: float = config.SAMPLE_MINUTES) -> float:
    """Lowest threshold that keeps false alarms on fault-free runs at or under the target.

    The alarm at sample t depends only on the minimum score over the last
    `persist` samples, so those minima are the only thresholds worth
    trying. The false alarm rate falls as the threshold rises, which
    allows a binary search over them.
    """
    floors = []
    for scores in score_runs:
        clean = np.nan_to_num(np.asarray(scores, dtype=float), nan=-np.inf)
        if len(clean) >= persist:
            floors.append(np.lib.stride_tricks.sliding_window_view(clean, persist).min(axis=1))
    if not floors:
        raise ValueError("no fault-free run is long enough to calibrate on")
    candidates = np.unique(np.concatenate(floors))
    candidates = candidates[np.isfinite(candidates)]          # ascending
    if len(candidates) == 0:
        raise ValueError("fault-free runs contain no finite scores")

    def meets_target(threshold: float) -> bool:
        return false_alarms_per_24h(score_runs, threshold, persist, sample_minutes) <= target_per_24h

    # The alarm needs score > threshold, so a threshold equal to the largest
    # candidate never alarms and always meets the target.
    low, high = 0, len(candidates) - 1
    while low < high:
        middle = (low + high) // 2
        if meets_target(float(candidates[middle])):
            high = middle
        else:
            low = middle + 1
    return float(candidates[low])


def detection_delay(scores: np.ndarray, threshold: float, onset_samples: int, persist: int = config.PERSIST,
                    sample_minutes: float = config.SAMPLE_MINUTES) -> float | None:
    """Minutes from the fault starting to the alarm, or None if the run ends without one.

    `onset_samples` is the number of fault-free samples at the start of the
    run. Only scores after the fault starts are considered, so an alarm
    that happened to be ringing beforehand is not credited as a detection.
    The earliest possible delay is persist * sample_minutes.
    """
    after = np.asarray(scores, dtype=float)[onset_samples:]
    alarm = alarm_state(after, threshold, persist)
    if not alarm.any():
        return None
    return float((int(np.argmax(alarm)) + 1) * sample_minutes)


def summarize_delays(delays: list[float | None]) -> dict:
    """Detection rate and delay statistics over the runs of one fault."""
    detected = np.array([d for d in delays if d is not None], dtype=float)
    n = len(delays)
    rate = len(detected) / n if n else float("nan")
    # undetected runs count as "never" in the median
    as_inf = np.array([np.inf if d is None else d for d in delays], dtype=float)
    median = float(np.median(as_inf)) if n else float("nan")
    return {
        "runs": n,
        "detection_rate": rate,
        "median_delay_min": median,
        "mean_delay_detected_min": float(detected.mean()) if len(detected) else float("nan"),
        "p90_delay_min": float(np.quantile(as_inf, 0.9, method="higher")) if n else float("nan"),
        "within_1h_rate": float(np.mean(as_inf <= 60)) if n else float("nan"),
    }


def alarm_episodes(alarm: np.ndarray, clear_samples: int = 20) -> list[tuple[int, int]]:
    """Group a debounced alarm series into episodes an operator would see.

    An episode starts when the alarm switches on and stays open until the
    alarm has been off for `clear_samples` consecutive samples, so a
    flickering alarm counts as one event. Returns (start, end)
    index pairs with `end` exclusive.
    """
    alarm = np.asarray(alarm, dtype=bool)
    episodes: list[tuple[int, int]] = []
    start, last_on = None, None
    for index, on in enumerate(alarm):
        if on:
            if start is None:
                start = index
            last_on = index
        elif start is not None and index - last_on >= clear_samples:
            episodes.append((start, index))
            start = None
    if start is not None:
        episodes.append((start, len(alarm)))
    return episodes
