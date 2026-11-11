## Detection by fault

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

## False alarms on fault-free test runs

| Model | False alarms per 24 h | False alarms | Fault-free hours | Time in alarm |
|---|---:|---:|---:|---:|
| PCA (T²/SPE) | 0.18 | 73 | 9,600 | 0.06% |
| LightGBM | 0.12 | 48 | 9,600 | 0.06% |
