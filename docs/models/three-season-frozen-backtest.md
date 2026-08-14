---
last_updated: "2026-08-13"
---

# Three-Season Frozen Leaderboard

Every candidate forecasts 2023-24, 2024-25, and 2025-26 from information
available before each target season begins. Target-season lineup allocation is
an oracle input; target outcomes never enter the frozen forecast. Bold values
are pooled leaders. Lower is better except skill and winner accuracy.

## Regular Season

Pooled over 584,970 eligible possessions from 3,284 games. Full-game and team
metrics cover 3,511 reconstructed games.

| Model | Poss. RMSE | Poss. MAE | Poss. skill | Eligible game RMSE | Eligible game skill | Full-game RMSE | Winner accuracy | Team NetRtg RMSE | Pythagorean-win RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HPM v2.2 usage allocation | **1.198004 (1)** | **1.141380 (1)** | **0.1182% (1)** | 14.1464 (4) | 16.7730% (4) | 14.4050 (4) | 67.47% (5) | 3.4802 (4) | 7.4793 (4) |
| HPM v2.1 empirical rebound capacity | 1.198010 (2) | 1.141403 (3) | 0.1171% (2) | **14.1340 (1)** | **16.9193% (1)** | 14.3869 (2) | 67.67% (3) | 3.4365 (2) | 7.3881 (2) |
| Value-Conditioned Aging HPM | 1.198010 (3) | 1.141402 (2) | 0.1171% (3) | 14.1342 (2) | 16.9169% (2) | **14.3802 (1)** | **67.84% (1)** | **3.4290 (1)** | **7.3532 (1)** |
| HPM v2 shooting composition | 1.198022 (4) | 1.141404 (4) | 0.1151% (4) | 14.1409 (3) | 16.8378% (3) | 14.3908 (3) | 67.70% (2) | 3.4592 (3) | 7.4282 (3) |
| Complete player-prior RAPM, no context or box score | 1.198061 (5) | 1.141675 (5) | 0.1086% (5) | 14.2105 (5) | 16.0175% (5) | 14.4756 (5) | 67.64% (4) | 3.6754 (5) | 7.7561 (5) |
| Forward 1-year RAPM-prior baseline | 1.198196 (6) | 1.141757 (6) | 0.0861% (6) | 14.5265 (6) | 12.2406% (6) | 14.8154 (6) | 65.39% (6) | 4.2374 (6) | 9.1066 (6) |
| Forward 3-year RAPM-prior baseline | 1.198246 (7) | 1.141950 (7) | 0.0778% (7) | 14.6559 (7) | 10.6699% (7) | 14.9714 (7) | 64.68% (7) | 4.5326 (7) | 9.7789 (7) |

## Playoffs

Pooled over 39,967 eligible possessions from 238 games. Each playoff cohort
uses its matching frozen pre-season player prior and prior-year context state;
postseason outcomes are evaluation-only.

| Model | Poss. RMSE | Poss. MAE | Poss. skill | Eligible game RMSE | Eligible game skill |
| --- | ---: | ---: | ---: | ---: | ---: |
| HPM v2 shooting composition | **1.192759 (1)** | **1.137652 (1)** | **0.0564% (1)** | 16.6913 (4) | 6.6559% (4) |
| Complete player-prior RAPM, no context or box score | 1.192761 (2) | 1.137741 (5) | 0.0559% (2) | **16.6068 (1)** | **7.5991% (1)** |
| HPM v2.1 empirical rebound capacity | 1.192772 (3) | 1.137654 (2) | 0.0542% (3) | 16.6701 (2) | 6.8932% (2) |
| Value-Conditioned Aging HPM | 1.192792 (4) | 1.137701 (4) | 0.0508% (4) | 16.7155 (5) | 6.3851% (5) |
| HPM v2.2 usage allocation | 1.192807 (5) | 1.137656 (3) | 0.0482% (5) | 16.6806 (3) | 6.7759% (3) |
| Forward 3-year RAPM-prior baseline | 1.193041 (6) | 1.137974 (7) | 0.0091% (6) | 17.1808 (6) | 1.1016% (6) |
| Forward 1-year RAPM-prior baseline | 1.193123 (7) | 1.137869 (6) | -0.0047% (7) | 17.2902 (7) | -0.1627% (7) |

See [Forward RAPM Memory Baselines](forward-rapm-memory-baselines.md) and
[Complete Player-Prior RAPM Baseline](complete-player-prior-baseline.md) for
model specifications, annual lambda selections, and immutable artifacts.
