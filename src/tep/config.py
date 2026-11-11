"""Paths and the constants that describe the dataset and the alarm policy."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("TEP_HOME", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(os.environ.get("TEP_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = Path(os.environ.get("TEP_MODEL_DIR", ROOT / "models"))
REPORT_DIR = ROOT / "reports"

# --- dataset -----------------------------------------------------------------
# Rieth et al. (2017), "Additional Tennessee Eastman Process Simulation Data
# for Anomaly Detection Evaluation", Harvard Dataverse.
DATASET_DOI = "doi:10.7910/DVN/6C3JR1"

MEASUREMENTS = [f"xmeas_{i}" for i in range(1, 42)]
MANIPULATED = [f"xmv_{i}" for i in range(1, 12)]
SENSORS = MEASUREMENTS + MANIPULATED            # 52 process variables
RUN_KEYS = ["faultNumber", "simulationRun"]
FAULTS = list(range(1, 21))

SAMPLE_MINUTES = 3                              # one sample every 3 minutes
# Fault-free samples at the start of each faulty run: the fault is switched
# on after 1 hour in the training files and after 8 hours in the testing files.
ONSET_SAMPLES = {"training": 20, "testing": 160}

# --- alarm policy ------------------------------------------------------------
WINDOW = 10                                     # samples per feature window (30 minutes)
PERSIST = 3                                     # consecutive samples over threshold to alarm
TARGET_FALSE_ALARMS_PER_24H = 1 / 7             # at most one false alarm a week
