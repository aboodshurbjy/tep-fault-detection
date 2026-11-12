"""Evaluate both detectors on the held-out testing files.

    python -m tep.evaluate

Reports, per model:
  * false alarms per 24 hours on fault-free runs
  * for each of the 20 faults: share of runs detected and the detection delay
  * a chance baseline: how often a fault-free run would be "detected" by a
    false alarm over the same 40 hours, so weak results are not mistaken
    for detection
  * which unit operation each model points at when it first alarms

Writes reports/results.csv, reports/results.md, reports/attribution.md and
reports/metrics.json.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import numpy as np
import pandas as pd

from . import config
from .alarms import alarm_state, count_onsets, detection_delay, false_alarms_per_24h, summarize_delays
from .attribution import FAULT_INFO
from .data import load_runs
from .scoring import MODELS, Scorer, score_frame

HARD_FAULTS = (3, 9, 15)
MODEL_LABEL = {"pca": "PCA (T²/SPE)", "lightgbm": "LightGBM"}


def evaluate_normal(scorer: Scorer, frame: pd.DataFrame) -> dict:
    """False alarm behaviour on fault-free test runs, plus the chance detection baseline."""
    onset = config.ONSET_SAMPLES["testing"]
    out = {}
    for model in MODELS:
        runs = score_frame(scorer, model, frame)
        threshold = scorer.thresholds[model]
        alarms = [alarm_state(s, threshold, scorer.persist) for s in runs.values()]
        samples = sum(len(a) for a in alarms)
        out[model] = {
            "runs": len(runs),
            "hours": samples * config.SAMPLE_MINUTES / 60,
            "false_alarms": int(sum(count_onsets(a) for a in alarms)),
            "false_alarms_per_24h": false_alarms_per_24h(list(runs.values()), threshold, scorer.persist),
            "time_in_alarm": float(sum(a.sum() for a in alarms) / samples),
            # the same question asked of faulty runs, asked of runs with no fault
            "chance": summarize_delays([detection_delay(s, threshold, onset, scorer.persist) for s in runs.values()]),
        }
    return out


def evaluate_fault(scorer: Scorer, frame: pd.DataFrame, attribution_runs: int = 25) -> dict:
    """Detection delay and first-alarm attribution for the runs of one fault."""
    onset = config.ONSET_SAMPLES["testing"]
    groups = {key: group[scorer.sensors].to_numpy() for key, group in frame.groupby(config.RUN_KEYS, sort=False)}
    out = {}
    for model in MODELS:
        runs = score_frame(scorer, model, frame)
        threshold = scorer.thresholds[model]
        delays = {key: detection_delay(s, threshold, onset, scorer.persist) for key, s in runs.items()}
        units, sensors = Counter(), Counter()
        explained = 0
        for key, delay in delays.items():
            if delay is None or explained >= attribution_runs:
                continue
            alarm_index = onset + int(delay / config.SAMPLE_MINUTES) - 1
            attribution = scorer.explain(model, groups[key][:alarm_index + 1])
            if attribution["unit"]:
                units[attribution["unit"]] += 1
                sensors.update(entry["sensor"] for entry in attribution["top_sensors"][:3])
                explained += 1
        out[model] = {
            **summarize_delays(list(delays.values())),
            "attribution_runs": explained,
            "unit_votes": dict(units.most_common()),
            "top_sensors": [name for name, _ in sensors.most_common(3)],
        }
    return out


def _percent(value: float) -> str:
    return "—" if np.isnan(value) else f"{value:.0%}"


def _minutes(value: float) -> str:
    return "not detected" if not np.isfinite(value) else f"{value:,.0f}"


def results_table(metrics: dict) -> pd.DataFrame:
    rows = []
    for fault in config.FAULTS:
        entry = metrics["faults"][str(fault)]
        row = {"fault": fault, "description": FAULT_INFO[fault][0]}
        for model in MODELS:
            for key, value in entry[model].items():
                if key not in ("unit_votes", "top_sensors", "attribution_runs"):
                    row[f"{model}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def results_markdown(metrics: dict) -> str:
    lines = [
        "| Fault | Description | PCA detected | PCA median delay (min) | LightGBM detected | LightGBM median delay (min) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for fault in config.FAULTS:
        entry = metrics["faults"][str(fault)]
        mark = " \\*" if fault in HARD_FAULTS else ""
        cells = [f"{fault}{mark}", FAULT_INFO[fault][0]]
        for model in MODELS:
            cells += [_percent(entry[model]["detection_rate"]), _minutes(entry[model]["median_delay_min"])]
        lines.append("| " + " | ".join(cells) + " |")
    chance = ["—", "**No fault (chance level)**"]
    for model in MODELS:
        baseline = metrics["normal"][model]["chance"]
        chance += [f"**{_percent(baseline['detection_rate'])}**", _minutes(baseline["median_delay_min"])]
    lines.append("| " + " | ".join(chance) + " |")
    return "\n".join(lines)


def false_alarm_markdown(metrics: dict) -> str:
    lines = ["| Model | False alarms per 24 h | False alarms | Fault-free hours | Time in alarm |",
             "|---|---:|---:|---:|---:|"]
    for model in MODELS:
        entry = metrics["normal"][model]
        lines.append(f"| {MODEL_LABEL[model]} | {entry['false_alarms_per_24h']:.2f} | {entry['false_alarms']} | "
                     f"{entry['hours']:,.0f} | {entry['time_in_alarm']:.2%} |")
    return "\n".join(lines)


def attribution_markdown(metrics: dict) -> str:
    lines = ["| Fault | Where it starts | PCA points at | PCA top sensors | LightGBM points at | LightGBM top sensors |",
             "|---|---|---|---|---|---|"]
    for fault in config.FAULTS:
        entry = metrics["faults"][str(fault)]
        cells = [str(fault), FAULT_INFO[fault][1] or "unknown"]
        for model in MODELS:
            votes, n = entry[model]["unit_votes"], entry[model]["attribution_runs"]
            if not votes:
                cells += ["no alarms", ""]
                continue
            unit, count = next(iter(votes.items()))
            cells += [f"{unit} ({count}/{n})", ", ".join(entry[model]["top_sensors"])]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def evaluate(normal_runs: int = 500, runs_per_fault: int = 500, data_dir=None, model_dir=None,
             report_dir=None) -> dict:
    started = time.time()
    scorer = Scorer.load(model_dir)
    report_dir = report_dir or config.REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"fault-free test runs: {normal_runs}")
    normal = load_runs("fault_free_testing", runs=range(1, normal_runs + 1), data_dir=data_dir)
    metrics = {
        "settings": {
            "window_samples": scorer.window, "persist_samples": scorer.persist,
            "sample_minutes": scorer.sample_minutes, "thresholds": scorer.thresholds,
            "target_false_alarms_per_24h": scorer.info.get("target_false_alarms_per_24h"),
            "normal_test_runs": normal_runs, "faulty_test_runs_per_fault": runs_per_fault,
            "post_fault_hours": (960 - config.ONSET_SAMPLES["testing"]) * config.SAMPLE_MINUTES / 60,
        },
        "normal": evaluate_normal(scorer, normal),
        "faults": {},
    }
    for model in MODELS:
        entry = metrics["normal"][model]
        print(f"  {model}: {entry['false_alarms_per_24h']:.3f} false alarms per 24 h over {entry['hours']:,.0f} h; "
              f"chance detection {entry['chance']['detection_rate']:.0%}")

    for fault in config.FAULTS:
        frame = load_runs("faulty_testing", runs=range(1, runs_per_fault + 1), faults=[fault], data_dir=data_dir)
        metrics["faults"][str(fault)] = evaluate_fault(scorer, frame)
        summary = " | ".join(
            f"{model} {metrics['faults'][str(fault)][model]['detection_rate']:.0%} "
            f"{_minutes(metrics['faults'][str(fault)][model]['median_delay_min'])} min" for model in MODELS)
        print(f"  fault {fault:>2}: {summary}")

    results_table(metrics).to_csv(report_dir / "results.csv", index=False)
    (report_dir / "results.md").write_text(
        "## Detection by fault\n\n" + results_markdown(metrics)
        + "\n\n## False alarms on fault-free test runs\n\n" + false_alarm_markdown(metrics) + "\n")
    (report_dir / "attribution.md").write_text("## Unit flagged at the first alarm\n\n" + attribution_markdown(metrics) + "\n")
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"wrote reports to {report_dir} in {time.time() - started:.0f} s")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--normal-runs", type=int, default=500, help="fault-free test runs (48 h each)")
    parser.add_argument("--runs-per-fault", type=int, default=500, help="faulty test runs per fault type")
    args = parser.parse_args()
    evaluate(args.normal_runs, args.runs_per_fault)


if __name__ == "__main__":
    main()
