"""How much each fault moves the measured variables, against fault-free operation.

    python -m tep.signature

For the three hard faults (3, 9, 15) and an easily detected fault of the
same kind for each (4, 11, 14), reports the two changes a detector could
pick up, using every sample after the fault begins in the testing files:

  * mean shift: the change in a variable's mean, in units of its
    fault-free standard deviation
  * variability ratio: a variable's average within-run standard deviation
    under the fault, divided by the same figure for fault-free runs

Writes reports/signatures.md.
"""

from __future__ import annotations

import argparse

import pandas as pd

from . import config
from .attribution import FAULT_INFO, SENSOR_INFO
from .data import load_runs

# hard fault -> a detected fault of the same kind
PAIRS = {3: 4, 9: 11, 15: 14}


def signature(normal: pd.DataFrame, faulty: pd.DataFrame, onset: int) -> pd.DataFrame:
    """Mean shift and variability ratio of every sensor, for one fault.

    Only samples after `onset` are used, in both frames.
    """
    normal, faulty = normal[normal["sample"] > onset], faulty[faulty["sample"] > onset]
    sensors = config.SENSORS
    shift = (faulty[sensors].mean() - normal[sensors].mean()) / normal[sensors].std()
    within = lambda frame: frame.groupby(config.RUN_KEYS)[sensors].std().mean()
    return pd.DataFrame({"mean_shift_sd": shift, "variability_ratio": within(faulty) / within(normal)})


def _largest(table: pd.DataFrame) -> tuple[str, float, str, float]:
    by_shift = table["mean_shift_sd"].abs().idxmax()
    by_ratio = table["variability_ratio"].idxmax()
    return (by_shift, float(table.loc[by_shift, "mean_shift_sd"]),
            by_ratio, float(table.loc[by_ratio, "variability_ratio"]))


def signatures_markdown(tables: dict[int, pd.DataFrame]) -> str:
    lines = ["| Fault | Description | Largest mean shift (fault-free SD) | Largest rise in variability |",
             "|---|---|---|---|"]
    for fault, table in tables.items():
        shift_sensor, shift, ratio_sensor, ratio = _largest(table)
        shift_cell = "none (under 0.05)" if abs(shift) < 0.05 else f"{shift:+.2f}, {SENSOR_INFO[shift_sensor][0]}"
        ratio_cell = "none (under 1%)" if ratio < 1.01 else f"×{ratio:.2f}, {SENSOR_INFO[ratio_sensor][0]}"
        lines.append(f"| {fault} | {FAULT_INFO[fault][0]} | {shift_cell} | {ratio_cell} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=500, help="test runs per fault and fault-free")
    args = parser.parse_args()
    onset = config.ONSET_SAMPLES["testing"]
    runs = range(1, args.runs + 1)
    normal = load_runs("fault_free_testing", runs=runs)
    tables = {}
    for hard, detected in PAIRS.items():
        for fault in (hard, detected):
            tables[fault] = signature(normal, load_runs("faulty_testing", runs=runs, faults=[fault]), onset)
    text = signatures_markdown(tables)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "signatures.md").write_text(
        "## What each fault does to the measured variables\n\n" + text + "\n")
    print(text)


if __name__ == "__main__":
    main()
