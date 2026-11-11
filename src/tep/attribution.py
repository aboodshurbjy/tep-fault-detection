"""Which sensors drove an alarm, and which part of the plant they belong to."""

from __future__ import annotations

import numpy as np

from . import config

FEED, REACTOR, SEPARATOR, STRIPPER, COMPRESSOR = "Feed", "Reactor", "Separator", "Stripper", "Compressor"

# sensor -> (description, unit operation). Names follow Downs and Vogel (1993).
SENSOR_INFO: dict[str, tuple[str, str]] = {
    "xmeas_1": ("A feed flow (stream 1)", FEED),
    "xmeas_2": ("D feed flow (stream 2)", FEED),
    "xmeas_3": ("E feed flow (stream 3)", FEED),
    "xmeas_4": ("A and C feed flow (stream 4)", FEED),
    "xmeas_5": ("Recycle flow (stream 8)", COMPRESSOR),
    "xmeas_6": ("Reactor feed rate (stream 6)", REACTOR),
    "xmeas_7": ("Reactor pressure", REACTOR),
    "xmeas_8": ("Reactor level", REACTOR),
    "xmeas_9": ("Reactor temperature", REACTOR),
    "xmeas_10": ("Purge rate (stream 9)", SEPARATOR),
    "xmeas_11": ("Separator temperature", SEPARATOR),
    "xmeas_12": ("Separator level", SEPARATOR),
    "xmeas_13": ("Separator pressure", SEPARATOR),
    "xmeas_14": ("Separator underflow (stream 10)", SEPARATOR),
    "xmeas_15": ("Stripper level", STRIPPER),
    "xmeas_16": ("Stripper pressure", STRIPPER),
    "xmeas_17": ("Stripper underflow (stream 11)", STRIPPER),
    "xmeas_18": ("Stripper temperature", STRIPPER),
    "xmeas_19": ("Stripper steam flow", STRIPPER),
    "xmeas_20": ("Compressor work", COMPRESSOR),
    "xmeas_21": ("Reactor cooling water outlet temperature", REACTOR),
    "xmeas_22": ("Condenser cooling water outlet temperature", SEPARATOR),
    **{f"xmeas_{23 + i}": (f"Reactor feed composition, {c}", REACTOR) for i, c in enumerate("ABCDEF")},
    **{f"xmeas_{29 + i}": (f"Purge gas composition, {c}", SEPARATOR) for i, c in enumerate("ABCDEFGH")},
    **{f"xmeas_{37 + i}": (f"Product composition, {c}", STRIPPER) for i, c in enumerate("DEFGH")},
    "xmv_1": ("D feed valve", FEED),
    "xmv_2": ("E feed valve", FEED),
    "xmv_3": ("A feed valve", FEED),
    "xmv_4": ("A and C feed valve", FEED),
    "xmv_5": ("Compressor recycle valve", COMPRESSOR),
    "xmv_6": ("Purge valve", SEPARATOR),
    "xmv_7": ("Separator liquid flow valve", SEPARATOR),
    "xmv_8": ("Stripper product flow valve", STRIPPER),
    "xmv_9": ("Stripper steam valve", STRIPPER),
    "xmv_10": ("Reactor cooling water valve", REACTOR),
    "xmv_11": ("Condenser cooling water valve", SEPARATOR),
}

# What an operator would look at first in each area.
UNIT_CHECKS: dict[str, str] = {
    FEED: "Feed headers: supply pressure, feed composition and the four feed control valves.",
    REACTOR: "Reactor: temperature and pressure trends, cooling water supply and valve, level control.",
    SEPARATOR: "Condenser and separator: condenser cooling water, separator level and pressure, purge valve.",
    STRIPPER: "Stripper: steam supply and valve, level control, product composition.",
    COMPRESSOR: "Recycle compressor: recycle valve position, compressor load, recycle flow.",
}

# Where each fault physically starts, for the faults Downs and Vogel describe.
FAULT_INFO: dict[int, tuple[str, str | None]] = {
    1: ("A/C feed ratio step, B constant (stream 4)", FEED),
    2: ("B composition step, A/C ratio constant (stream 4)", FEED),
    3: ("D feed temperature step (stream 2)", FEED),
    4: ("Reactor cooling water inlet temperature step", REACTOR),
    5: ("Condenser cooling water inlet temperature step", SEPARATOR),
    6: ("A feed loss (stream 1)", FEED),
    7: ("C header pressure loss (stream 4)", FEED),
    8: ("A, B, C feed composition random variation (stream 4)", FEED),
    9: ("D feed temperature random variation (stream 2)", FEED),
    10: ("C feed temperature random variation (stream 4)", FEED),
    11: ("Reactor cooling water inlet temperature random variation", REACTOR),
    12: ("Condenser cooling water inlet temperature random variation", SEPARATOR),
    13: ("Reaction kinetics slow drift", REACTOR),
    14: ("Reactor cooling water valve sticking", REACTOR),
    15: ("Condenser cooling water valve sticking", SEPARATOR),
    16: ("Unknown", None),
    17: ("Unknown", None),
    18: ("Unknown", None),
    19: ("Unknown", None),
    20: ("Unknown", None),
}

assert list(SENSOR_INFO) == config.SENSORS, "lookup table must cover every sensor, in dataset order"


def describe(sensor: str) -> str:
    return SENSOR_INFO[sensor][0]


def unit_of(sensor: str) -> str:
    return SENSOR_INFO[sensor][1]


def top_sensors(contributions: np.ndarray, sensors=None, k: int = 5) -> list[dict]:
    """The k sensors pushing hardest towards "fault", with their unit operation.

    contributions: one value per sensor; larger means more responsible.
    Negative values (evidence for normal operation) are ignored. `share`
    is each sensor's fraction of all positive contribution.
    """
    sensors = list(sensors or config.SENSORS)
    values = np.clip(np.asarray(contributions, dtype=float), 0, None)
    total = values.sum()
    if not np.isfinite(total) or total <= 0:
        return []
    order = np.argsort(values)[::-1][:k]
    return [{
        "sensor": sensors[i],
        "description": describe(sensors[i]),
        "unit": unit_of(sensors[i]),
        "share": round(float(values[i] / total), 4),
    } for i in order if values[i] > 0]


def suspect_unit(top: list[dict]) -> str | None:
    """The unit operation carrying the most contribution among the top sensors."""
    if not top:
        return None
    totals: dict[str, float] = {}
    for entry in top:
        totals[entry["unit"]] = totals.get(entry["unit"], 0.0) + entry["share"]
    return max(totals, key=totals.get)


def explain(contributions: np.ndarray, sensors=None, k: int = 5) -> dict:
    """Top sensors, the unit to check and what to check there."""
    top = top_sensors(contributions, sensors, k)
    unit = suspect_unit(top)
    return {"unit": unit, "check": UNIT_CHECKS.get(unit), "top_sensors": top}
