# TEP Fault Detection

Early fault detection on the Tennessee Eastman Process benchmark, built as a predictive maintenance tool for a plant operator: raise an alarm soon after something goes wrong, rarely when nothing has, and say which part of the plant to look at.

Two detectors are compared on equal terms:

- **PCA with T² and SPE**, the classical multivariate process monitoring method.
- **LightGBM** on windowed features (30-minute rolling mean, standard deviation and slope of every sensor).

## The process in plain English

The Tennessee Eastman Process is a simulation of a real chemical plant, published by Downs and Vogel of Eastman Chemical in 1993 as a test problem for control and monitoring.

- **Feed.** Four gas streams bring in the reactants (A, C, D, E) and a little inert (B).
- **Reactor.** The gases react over a catalyst to form two liquid products, G and H, plus an unwanted byproduct F. The reactions give off heat, so the reactor is cooled with cooling water. If cooling falls behind, temperature and pressure climb, and the plant trips at high pressure.
- **Condenser and separator.** The reactor's outlet gas is cooled so the products condense. A separator drum splits liquid from uncondensed gas.
- **Compressor and purge.** The uncondensed gas is compressed and recycled to the reactor. A small purge stream is vented so the inert and the byproduct do not build up.
- **Stripper.** The liquid goes to a stripping column, where steam and fresh feed strip out leftover reactants. Product leaves from the bottom.

The plant runs under closed-loop control, which matters for monitoring: when a disturbance hits, the controllers move valves to hold temperatures, levels and pressures steady. A fault often shows up in a *valve position* rather than in the measurement the valve is protecting, and some faults are hidden almost completely.

There are 52 variables: 41 measurements (flows, temperatures, pressures, levels and delayed composition analysers) and 11 valve positions. The benchmark defines 20 faults, including step changes, random variation, a slow drift, sticking valves and five whose cause was never published.

## Data

Rieth, Amsel, Tran and Cook (2017), *Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation*, Harvard Dataverse, [doi:10.7910/DVN/6C3JR1](https://doi.org/10.7910/DVN/6C3JR1).

| File | Runs | Length | Fault starts |
|---|---|---|---|
| Fault-free training | 500 | 25 h (500 samples) | — |
| Faulty training | 500 per fault × 20 faults | 25 h | after 1 h |
| Fault-free testing | 500 | 48 h (960 samples) | — |
| Faulty testing | 500 per fault × 20 faults | 48 h | after 8 h |

One sample is 3 minutes. `python -m tep.download` reads the file list and checksums from the Dataverse API, downloads about 1.4 GB, verifies every file against its published MD5, and converts to Parquet. If anything fails it stops with an error; it never falls back to other data.

## Quick start

Requires Python 3.12. On macOS, LightGBM also needs `brew install libomp`.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m tep.download      # fetch and verify the dataset (about 1.4 GB, plus 1.6 GB of Parquet)
python -m tep.train         # fit both detectors, calibrate alarm thresholds (under a minute)
python -m tep.evaluate      # held-out results -> reports/
pytest

uvicorn tep.api:app                    # scoring API on :8000
streamlit run dashboard/app.py         # operator dashboard on :8501
```

## How it works

### Split by simulation run, never by row

A run is 25 or 48 hours of strongly autocorrelated samples from one random seed. Splitting rows at random would put near-identical neighbours on both sides and inflate every score. All splits here are on whole runs.

There is a second, less obvious trap. In this dataset **run number *k* uses the same random seed for every fault and for the fault-free file**: the first hour of run 17 under fault 4 is bit-for-bit identical to the first hour of fault-free run 17. Splitting by (fault, run) pairs would therefore still leak. So a run number is assigned to exactly one set, and that assignment is applied across all faults. A test checks this property on the real files.

| Set | Source | Used for |
|---|---|---|
| Train | 300 run numbers from the training files (300 fault-free runs, 80 runs of each fault) | Fitting PCA and LightGBM |
| Tune | 50 run numbers from the training files | Early stopping for LightGBM |
| Calibrate | 150 run numbers from the training files, fault-free only (3,750 h) | Setting alarm thresholds |
| Test | The separate testing files, generated from different seeds | Every number reported below |

### The two detectors

**PCA (T² and SPE).** A PCA model is fitted to standardised fault-free data, keeping 31 components (90% of variance). T² measures how far a sample sits from normal operation inside the model; SPE measures how badly the usual relationships between sensors are broken. Each is divided by its control limit and the larger ratio is the anomaly score. Limits are empirical 99th percentiles, because the textbook F and chi-squared limits assume independent Gaussian samples and closed-loop process data are neither.

**LightGBM.** A binary normal-versus-fault classifier over 156 features: the mean, standard deviation and least-squares slope of each of the 52 variables over a trailing 30-minute window. Windows are causal (they use only past samples) and never cross a run boundary. The anomaly score is the predicted probability of a fault.

### One alarm rule for both

Each detector produces one score per sample. The alarm is raised when the score has been over its threshold for **3 consecutive samples** (9 minutes), which is ordinary alarm debouncing. Each threshold is calibrated on the fault-free calibration runs to the same budget: **at most one false alarm per week** (0.14 per 24 h). Because both models are held to the same false alarm budget, their detection delays can be compared directly. The fastest possible detection under this rule is 9 minutes.

## Results

All numbers are from the held-out testing files: 100 runs of each fault (40 hours of faulty operation per run) and 200 fault-free runs (9,600 hours).

The testing files hold 500 fault-free runs and 500 runs of each fault (10,000 faulty runs), every run 48 hours long. The evaluation reads the first 200 fault-free runs and the first 100 runs of each fault, which are the defaults of `python -m tep.evaluate` (`--normal-runs` and `--runs-per-fault`). The subset is there only to keep evaluation runtime and memory down: the seed-sharing split does not call for it, because that split applies to the training files and every testing run is held out, so `--normal-runs 500 --runs-per-fault 500` scores the full set.

### False alarms in normal operation

| Model | False alarms per 24 h | False alarms | Fault-free hours | Time in alarm |
|---|---:|---:|---:|---:|
| PCA (T²/SPE) | 0.18 | 73 | 9,600 | 0.06% |
| LightGBM | 0.12 | 48 | 9,600 | 0.06% |

LightGBM stays inside the one-per-week budget on unseen data. PCA overshoots it (about 1.3 per week), which is typical when a threshold set on one batch of runs meets another.

### Detection delay by fault

"Detected" is the share of runs in which the alarm was raised at some point in the 40 hours after the fault began. The median counts undetected runs as never, so it reads "not detected" when fewer than half the runs were caught.

| Fault | Description | PCA detected | PCA median delay (min) | LightGBM detected | LightGBM median delay (min) |
|---|---|---:|---:|---:|---:|
| 1 | A/C feed ratio step, B constant (stream 4) | 100% | 15 | 100% | 18 |
| 2 | B composition step, A/C ratio constant (stream 4) | 100% | 39 | 100% | 48 |
| 3 \* | D feed temperature step (stream 2) | 27% | not detected | 42% | not detected |
| 4 | Reactor cooling water inlet temperature step | 100% | 9 | 100% | 9 |
| 5 | Condenser cooling water inlet temperature step | 100% | 9 | 100% | 12 |
| 6 | A feed loss (stream 1) | 100% | 9 | 100% | 15 |
| 7 | C header pressure loss (stream 4) | 100% | 9 | 100% | 9 |
| 8 | A, B, C feed composition random variation (stream 4) | 100% | 63 | 100% | 68 |
| 9 \* | D feed temperature random variation (stream 2) | 30% | not detected | 37% | not detected |
| 10 | C feed temperature random variation (stream 4) | 100% | 147 | 100% | 99 |
| 11 | Reactor cooling water inlet temperature random variation | 100% | 27 | 100% | 27 |
| 12 | Condenser cooling water inlet temperature random variation | 100% | 30 | 100% | 30 |
| 13 | Reaction kinetics slow drift | 100% | 117 | 100% | 136 |
| 14 | Reactor cooling water valve sticking | 100% | 9 | 100% | 12 |
| 15 \* | Condenser cooling water valve sticking | 32% | not detected | 40% | not detected |
| 16 | Unknown | 100% | 105 | 100% | 52 |
| 17 | Unknown | 100% | 96 | 100% | 93 |
| 18 | Unknown | 100% | 147 | 100% | 140 |
| 19 | Unknown | 100% | 206 | 100% | 15 |
| 20 | Unknown | 100% | 123 | 100% | 122 |
| — | **No fault (chance level)** | **26%** | not detected | **15%** | not detected |

\* Known to be hard to detect.

**Reading the table**

- **Seventeen of the twenty faults are detected in every test run by both models.** Step faults in the cooling water, feed loss and header pressure (4, 5, 6, 7, 14) are caught at or near the 9-minute floor.
- **The classical baseline is hard to beat.** On most faults the two models are within a few minutes of each other, and PCA is slightly faster on several step faults because it reacts to single samples while the classifier waits for its 30-minute window to fill with faulty data.
- **LightGBM is clearly better where the signature is in the dynamics rather than the level:** fault 19 (15 minutes against 206), fault 16 (52 against 105) and fault 10 (99 against 147). Rolling standard deviation and slope see changes in variability that a snapshot method averages away.
- **Slow faults are slow for both.** The kinetics drift (13) and unknown faults 17, 18 and 20 take 1.5 to 2.5 hours to be flagged by either model.

### Faults 3, 9 and 15: not detected

These three are the benchmark's known blind spots, and this project does not solve them.

The last row of the table is the key to reading them. It asks the same question of fault-free runs: how often does an alarm go off in the same 40-hour window when *nothing is wrong*? At one false alarm per week that happens by chance in 26% of runs for PCA and 15% for LightGBM.

| Fault | PCA detected | LightGBM detected | LightGBM mean delay when it does alarm | Caught within 1 hour (either model) |
|---|---:|---:|---:|---:|
| 3 | 27% | 42% | 20.4 h | 1% |
| 9 | 30% | 37% | 18.2 h | 1% or less |
| 15 | 32% | 40% | 20.3 h | 1% |
| No fault (chance) | 26% | 15% | 21.5 h | under 1% |

- **PCA is at chance.** Its 27 to 32% is indistinguishable from the 26% it scores on runs with no fault at all. Those "detections" are false alarms that happened to fall after the fault began.
- **LightGBM is above chance but not useful.** It flags 37 to 42% of runs against a chance level of 15%, so it has found some real signal. But it still misses about six runs in ten, and when it does alarm it is on average 18 to 20 hours after the fault began, barely sooner than a random false alarm (21.5 hours). About 1 run in 100 is caught in the first hour.
- **Why they are hard.** Faults 3 and 9 disturb the temperature of one feed stream, which carries little heat compared with the reactor. Fault 15 is a sticking condenser cooling water valve. In all three the control system absorbs the disturbance and the measured variables stay within their normal noise. Published methods that report high detection rates on these faults usually do so at false alarm rates no operator would accept.
- **What this means in practice.** An operator should treat this tool as blind to these three faults. Lowering the threshold to "catch" them would trade a small gain for many more false alarms across the whole plant.

With 100 runs per fault, a detection rate near 35% carries a sampling uncertainty of roughly ±9 percentage points, so the differences between the three hard faults are not meaningful.

## Fault attribution

When an alarm is raised, each model reports the sensors most responsible and the unit operation they belong to.

- **PCA contribution plots.** If SPE is the statistic over its limit, each sensor's contribution is its squared residual. If T² is, the contribution is that sensor's share of the T² sum.
- **SHAP for LightGBM.** TreeSHAP values for the alarming window, with each sensor's mean, standard deviation and slope features added together. Positive values push the prediction towards "fault".

The top five sensors are mapped to a unit through a lookup table (`src/tep/attribution.py`), and the unit carrying the most contribution is the one to check first.

| Unit | Sensors | Check first |
|---|---|---|
| Feed | Feed flows and feed valves (XMEAS 1–4, XMV 1–4) | Supply pressure, feed composition, feed control valves |
| Reactor | Reactor pressure, level, temperature, feed rate and feed analysis; cooling water outlet temperature and valve (XMEAS 6–9, 21, 23–28; XMV 10) | Temperature and pressure trends, cooling water supply and valve, level control |
| Separator | Separator temperature, level, pressure, underflow; condenser cooling water; purge rate, valve and analysis (XMEAS 10–14, 22, 29–36; XMV 6, 7, 11) | Condenser cooling water, separator level and pressure, purge valve |
| Stripper | Stripper level, pressure, temperature, underflow, steam; product analysis (XMEAS 15–19, 37–41; XMV 8, 9) | Steam supply and valve, level control, product composition |
| Compressor | Recycle flow, compressor work, recycle valve (XMEAS 5, 20; XMV 5) | Recycle valve position, compressor load, recycle flow |

### Does it point at the right place?

For up to 25 detected test runs per fault, the unit each model named at its first alarm, with how many of those runs agreed, next to where the fault physically starts.

| Fault | Where it starts | PCA points at | PCA top sensors | LightGBM points at | LightGBM top sensors |
|---|---|---|---|---|---|
| 1 | Feed | Compressor (20/25) | xmeas_16, xmeas_20, xmv_5 | Compressor (22/25) | xmeas_16, xmv_5, xmeas_7 |
| 2 | Feed | Separator (19/25) | xmeas_30, xmeas_10, xmeas_24 | Separator (25/25) | xmeas_30, xmeas_10, xmeas_24 |
| 3 | Feed | Stripper (12/25) | xmeas_41, xmeas_40, xmeas_23 | Reactor (16/25) | xmeas_21, xmv_1, xmeas_2 |
| 4 | Reactor | Reactor (25/25) | xmv_10, xmeas_9, xmeas_25 | Reactor (25/25) | xmv_10, xmeas_9, xmeas_2 |
| 5 | Separator | Separator (23/25) | xmeas_11, xmeas_22, xmeas_21 | Stripper (22/25) | xmeas_18, xmeas_22, xmeas_21 |
| 6 | Feed | Feed (25/25) | xmv_3, xmeas_1, xmeas_21 | Feed (19/25) | xmv_3, xmeas_1, xmv_5 |
| 7 | Feed | Reactor (14/25) | xmeas_16, xmeas_25, xmeas_31 | Feed (18/25) | xmeas_22, xmv_4, xmeas_18 |
| 8 | Feed | Compressor (11/25) | xmeas_16, xmeas_20, xmv_5 | Compressor (12/25) | xmeas_16, xmeas_20, xmv_5 |
| 9 | Feed | Stripper (12/25) | xmeas_37, xmeas_23, xmeas_41 | Reactor (15/25) | xmeas_21, xmeas_2, xmv_1 |
| 10 | Feed | Stripper (22/25) | xmeas_18, xmeas_19, xmeas_38 | Stripper (24/25) | xmeas_18, xmeas_2, xmeas_20 |
| 11 | Reactor | Reactor (25/25) | xmv_10, xmeas_9, xmeas_4 | Reactor (25/25) | xmv_10, xmeas_9, xmeas_2 |
| 12 | Separator | Separator (21/25) | xmeas_11, xmeas_22, xmeas_21 | Stripper (17/25) | xmeas_18, xmeas_22, xmeas_11 |
| 13 | Reactor | Reactor (10/25) | xmeas_16, xmeas_25, xmeas_7 | Stripper (11/25) | xmeas_16, xmeas_21, xmeas_7 |
| 14 | Reactor | Reactor (25/25) | xmeas_21, xmeas_2, xmv_1 | Reactor (25/25) | xmeas_9, xmv_10, xmeas_21 |
| 15 | Separator | Separator (10/25) | xmeas_39, xmeas_40, xmeas_23 | Separator (10/25) | xmeas_11, xmeas_18, xmeas_22 |
| 16 | unknown | Stripper (24/25) | xmv_9, xmeas_19, xmeas_38 | Stripper (25/25) | xmeas_19, xmeas_18, xmv_9 |
| 17 | unknown | Reactor (25/25) | xmeas_21, xmv_1, xmeas_2 | Reactor (25/25) | xmeas_21, xmeas_2, xmv_1 |
| 18 | unknown | Separator (25/25) | xmeas_11, xmeas_22, xmeas_31 | Separator (25/25) | xmeas_22, xmv_1, xmeas_20 |
| 19 | unknown | Compressor (10/25) | xmeas_5, xmv_5, xmeas_33 | Compressor (25/25) | xmv_5, xmeas_20, xmv_1 |
| 20 | unknown | Compressor (21/25) | xmv_5, xmeas_33, xmeas_7 | Compressor (24/25) | xmv_5, xmeas_20, xmeas_11 |

- **Faults that start inside a unit are located well.** Reactor cooling faults (4, 11, 14) are attributed to the reactor in every run by both models, led by the cooling water valve, reactor temperature and cooling water outlet temperature. Feed loss (6) is attributed to the feed.
- **Attribution shows where the symptoms are, not the root cause.** Feed composition faults (1, 8) are flagged at the compressor, and fault 10 at the stripper, because that is where the control system's response first becomes visible. This is a starting point for the operator's own diagnosis, not a verdict.
- **The two models can disagree.** For condenser cooling faults (5, 12), PCA names the separator, where the fault is, while LightGBM names the stripper, which is downstream.
- **For faults 3, 9 and 15 the attribution is noise.** The alarms are mostly false, the votes are split, and the named unit should be ignored.

## Scoring API

`POST /score` takes recent readings, oldest first, one row of 52 values per sample in the order given by `GET /schema`. It needs at least 10 rows (one feature window) and 12 to apply the 3-sample alarm rule.

```bash
curl -s localhost:8000/score -H 'content-type: application/json' \
  -d '{"readings": [[0.25, 3664.0, 4509.3, ...], ...], "top_k": 3}'
```

```json
{
  "n_samples": 12,
  "alarm": true,
  "models": {
    "lightgbm": {
      "score": 0.99999, "threshold": 0.9031, "over_threshold": true, "alarm": true,
      "attribution": {
        "unit": "Reactor",
        "check": "Reactor: temperature and pressure trends, cooling water supply and valve, level control.",
        "top_sensors": [
          {"sensor": "xmv_10", "description": "Reactor cooling water valve", "unit": "Reactor", "share": 0.7461},
          {"sensor": "xmeas_9", "description": "Reactor temperature", "unit": "Reactor", "share": 0.2195}
        ]
      }
    },
    "pca": {"...": "same fields, plus t2_ratio, spe_ratio and which statistic alarmed"}
  }
}
```

`alarm` is `null` for a model when too few samples were sent to judge three in a row; `over_threshold` then reports the latest sample alone.

## Dashboard

`streamlit run dashboard/app.py` replays any faulty test run in simulated real time. Pick a fault, a run and a detector, press Start, and the anomaly score is drawn sample by sample. When the alarm rule is met the page shows the unit to check, the contributing sensors and their recent trends. The moment the fault was introduced can be shown or hidden, since a real operator would not know it.

![Dashboard screenshot](docs/dashboard.png)

## What would change in a real plant

The benchmark is a clean, stationary simulation. A real unit is neither, and the gap is mostly on the side that hurts.

**Sensor drift and bad data.** Real transmitters drift, foul, freeze and get recalibrated; analysers go offline for hours. Both models here assume the "normal" they were trained on is permanent, so slow drift would look like a fault and be alarmed. A deployment needs sensor validation in front of the models (range, rate-of-change and stuck-value checks), a defined behaviour when a reading is missing, and a way to tell "this instrument is wrong" from "the process is wrong". PCA's SPE statistic is naturally sensitive to single-sensor failures, which is useful if it is routed to the instrument technician rather than the board operator.

**Operating modes.** This data covers one operating point. Real plants change rate, grade and feedstock, start up, shut down and run with equipment out of service. Each change shifts normal behaviour and would trip a single-mode model. Options are a model per mode, features expressed relative to setpoints, or suppressing monitoring during known transitions.

**Retraining.** Catalyst ages, exchangers foul, valves are replaced, control loops are retuned. The model of "normal" has to follow, which means scheduled refits on recent data that operators have confirmed as fault-free, recalibrating thresholds every time, and versioning models so an alarm can be traced to the model that raised it. The danger is retraining on a period that contained an undetected slow fault, which teaches the model that the fault is normal. Monitoring the score distribution and the false alarm rate between refits shows when a model has gone stale.

**Labelled faults are rare.** The classifier here was trained on 80 examples of each fault. A real plant has a handful of documented incidents, rarely labelled to the minute and never covering every failure. In practice the unsupervised method (PCA or similar) carries the monitoring, and a supervised model is added only for the few fault types with enough history. A classifier also says nothing reliable about a fault it has never seen.

**Alarm fatigue.** Operators already manage hundreds of alarms, and guidance such as ISA-18.2 and EEMUA 191 puts a manageable load at around one alarm per ten minutes across the whole console. A detector that cries wolf gets ignored or disabled. The one-per-week budget used here is per model and would be divided among every monitored unit. A deployment would also add what this prototype leaves out: shelving and acknowledgement, deadbands so a score hovering at the threshold does not chatter, grouping related alarms into one event, and an advisory priority below safety alarms. The right threshold is set with operations, by weighing the cost of a missed fault against the cost of a wasted investigation, and the false alarm rate should be tracked as a live performance figure.

**Attribution is a hint.** As the results show, contributions point at where symptoms appear. Operators need to know that, and the lookup table should be reviewed by a process engineer for the actual plant.

**Integration.** Scoring would run against the plant historian or an OPC UA feed, on the plant side of the network boundary, with the tool strictly advisory: it informs the operator and never writes to the control system. Alarms and operator responses should be logged so the tool can be audited and improved.

## Tests

`pytest` covers the feature windows (values checked against an independent rolling calculation, causality, no window across a run boundary), the leakage-free split, the alarm logic and the scoring endpoint. Unit tests run on small generated arrays so they need no download; they test mechanics only. When the dataset and trained models are present, extra tests check the shared-seed property, the saved split and a real fault through the API. Every result in this README comes from the real dataset.

## Layout

```
src/tep/
  download.py      fetch, verify and convert the dataset
  data.py          load runs, split by simulation run
  features.py      rolling mean, std and slope
  pca_monitor.py   PCA, T², SPE and contributions
  gbm.py           LightGBM detector and SHAP
  alarms.py        alarm rule, false alarm rate, detection delay
  attribution.py   sensor and fault lookup tables
  scoring.py       both detectors and thresholds as one bundle
  train.py         fit and calibrate
  evaluate.py      held-out evaluation and reports
  api.py           FastAPI scoring endpoint
dashboard/app.py   Streamlit replay
reports/           results from the last evaluation
tests/
```

## References

- J. J. Downs and E. F. Vogel, "A plant-wide industrial process control problem", *Computers & Chemical Engineering* 17(3), 1993.
- C. A. Rieth, B. D. Amsel, R. Tran and M. B. Cook, "Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation", Harvard Dataverse, 2017.
