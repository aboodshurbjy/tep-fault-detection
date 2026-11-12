# What would change in a real plant

The benchmark is a clean, stationary simulation and a real unit is neither. These are the things I would expect to change, moved here from the [README](../README.md).

## Sensor drift and bad data

Real transmitters drift, foul, freeze and get recalibrated, and analysers go offline for hours. Both models here assume the "normal" they were trained on is permanent, so slow drift would look like a fault and be alarmed. A deployment needs sensor validation in front of the models (range, rate-of-change and stuck-value checks), a defined behaviour when a reading is missing, and a way to tell "this instrument is wrong" from "the process is wrong". PCA's SPE statistic is sensitive to single-sensor failures. That is useful if those alarms go to the instrument technician and not to the board operator.

## Operating modes

This data covers one operating point. Real plants change rate, grade and feedstock, start up, shut down and run with equipment out of service. Each change shifts normal behaviour and would trip a single-mode model. Options are a model per mode, features expressed relative to setpoints, or suppressing monitoring during known transitions.

## Retraining

Catalyst ages, exchangers foul, valves are replaced, control loops are retuned. The model of "normal" has to follow. That means scheduled refits on recent data that operators have confirmed as fault-free, recalibrating thresholds every time, and versioning models so an alarm can be traced to the model that raised it. The danger is retraining on a period that contained an undetected slow fault, which teaches the model that the fault is normal. Monitoring the score distribution and the false alarm rate between refits shows when a model has gone stale.

## Few labelled faults

The classifier here was trained on 80 examples of each fault. A real plant has a handful of documented incidents, rarely labelled to the minute and never covering every failure. I would expect the unsupervised method (PCA or similar) to carry the monitoring, with a supervised model added only for the few fault types with enough history. A classifier also says nothing reliable about a fault it has never seen.

## Alarm fatigue

Operators already manage hundreds of alarms, and guidance such as ISA-18.2 and EEMUA 191 puts a manageable load at around one alarm per ten minutes across the whole console. A detector with too many false alarms gets ignored or disabled. The one-per-week budget used here is per model and would be divided among every monitored unit. A deployment would also add what this prototype leaves out: shelving and acknowledgement, deadbands so a score hovering at the threshold does not chatter, grouping related alarms into one event, and an advisory priority below safety alarms. The threshold should be set with operations, by weighing the cost of a missed fault against the cost of a wasted investigation, and the false alarm rate should be tracked as a live performance figure.

## Attribution

Contributions point at where symptoms appear (see [attribution.md](attribution.md)). Operators need to know that, and the lookup table should be reviewed by a process engineer for the actual plant.

## Integration

Scoring would run against the plant historian or an OPC UA feed, on the plant side of the network boundary. The tool would be advisory only: it informs the operator and never writes to the control system. Alarms and operator responses should be logged so the tool can be audited.
