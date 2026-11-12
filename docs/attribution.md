# Attribution accuracy

This table backs the fault attribution section of the [README](../README.md). For up to 25 detected test runs per fault, it lists the unit each model named at its first alarm and how many of those runs agreed, next to where the fault physically starts.

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

- Faults that start inside a unit are located well. Reactor cooling faults (4, 11, 14) are attributed to the reactor in every run by both models, led by the cooling water valve, reactor temperature and cooling water outlet temperature. Feed loss (6) is attributed to the feed.
- Attribution shows where the symptoms are. Feed composition faults (1, 8) are flagged at the compressor, and fault 10 at the stripper, because that is where the control system's response first becomes visible. The operator still has to trace the cause from there.
- The two models can disagree. For condenser cooling faults (5, 12), PCA names the separator, where the fault is. LightGBM names the stripper, which is downstream.
- For faults 3, 9 and 15 the attribution is noise. The alarms are mostly false and the votes are split, so the named unit should be ignored.
