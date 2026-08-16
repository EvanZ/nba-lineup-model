---
last_updated: "2026-08-15"
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
| [NAIL-RAPM v1.0](hpm-x3-linear-ridge-without-uncertainty.md) | **1.197978 (1)** | **1.141349 (1)** | **0.1225% (1)** | **14.1051 (1)** | **17.2578% (1)** | **14.3529 (1)** | **68.13% (1)** | 3.4348 (3) | 7.3731 (3) |
| [HPM x3 ORB claim rebound replacement](hpm-x3.md) | 1.198000 (2) | 1.141400 (6) | 0.1188% (2) | 14.1309 (2) | 16.9555% (2) | 14.3774 (2) | 67.73% (6) | **3.4229 (1)** | **7.3269 (1)** |
| [Compiled-additive HPM x3 plus quadratic side context](linear-hpm-x3-quadratic-side-context.md) | 1.198002 (3) | 1.141366 (3) | 0.1185% (3) | 14.1311 (3) | 16.9525% (3) | 14.3922 (6) | 67.36% (12) | 3.5139 (8) | 7.5889 (9) |
| HPM v2.2 usage allocation | 1.198004 (4) | 1.141380 (4) | 0.1182% (4) | 14.1464 (7) | 16.7730% (7) | 14.4050 (7) | 67.47% (11) | 3.4802 (6) | 7.4793 (7) |
| HPM v2.1 empirical rebound capacity | 1.198010 (5) | 1.141403 (8) | 0.1171% (5) | 14.1340 (4) | 16.9193% (4) | 14.3869 (4) | 67.67% (9) | 3.4365 (4) | 7.3881 (4) |
| Value-Conditioned Aging HPM | 1.198010 (6) | 1.141402 (7) | 0.1171% (6) | 14.1342 (5) | 16.9169% (5) | 14.3802 (3) | 67.84% (3) | 3.4290 (2) | 7.3532 (2) |
| HPM v2 shooting composition | 1.198022 (7) | 1.141404 (9) | 0.1151% (7) | 14.1409 (6) | 16.8378% (6) | 14.3908 (5) | 67.70% (7) | 3.4592 (5) | 7.4282 (5) |
| [HPM v2.3 shot portfolio](hpm-v23.md) | 1.198025 (8) | 1.141396 (5) | 0.1147% (8) | 14.1815 (8) | 16.3594% (8) | 14.4288 (8) | 67.70% (7) | 3.5037 (7) | 7.5274 (8) |
| [Additive prior plus linear non-additive context](additive-profile-linear-shape-context.md) | 1.198026 (9) | 1.141366 (2) | 0.1144% (9) | 14.2154 (12) | 15.9592% (12) | 14.4542 (9) | 67.76% (5) | 3.5492 (9) | 7.4793 (6) |
| [HPM x1 ORB claim context](hpm-x1.md) | 1.198047 (10) | 1.141658 (10) | 0.1110% (10) | 14.2081 (10) | 16.0449% (10) | 14.4665 (11) | 67.87% (2) | 3.6653 (11) | 7.7498 (11) |
| [HPM x2 raw OREB/100 context](hpm-x2.md) | 1.198048 (11) | 1.141660 (11) | 0.1108% (11) | 14.2073 (9) | 16.0552% (9) | 14.4641 (10) | 67.79% (4) | 3.6558 (10) | 7.7389 (10) |
| Complete player-prior RAPM, no context or box score | 1.198061 (12) | 1.141675 (12) | 0.1086% (12) | 14.2105 (11) | 16.0175% (11) | 14.4756 (12) | 67.64% (10) | 3.6754 (12) | 7.7561 (12) |
| Forward 1-year RAPM-prior baseline | 1.198196 (13) | 1.141757 (13) | 0.0861% (13) | 14.5265 (13) | 12.2406% (13) | 14.8154 (13) | 65.39% (13) | 4.2374 (13) | 9.1066 (13) |
| Forward 3-year RAPM-prior baseline | 1.198246 (14) | 1.141950 (14) | 0.0778% (14) | 14.6559 (14) | 10.6699% (14) | 14.9714 (14) | 64.68% (14) | 4.5326 (14) | 9.7789 (14) |

## Playoffs

Pooled over 39,967 eligible possessions from 238 games. Each playoff cohort
uses its matching frozen pre-season player prior and prior-year context state;
postseason outcomes are evaluation-only.

| Model | Poss. RMSE | Poss. MAE | Poss. skill | Eligible game RMSE | Eligible game skill |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Compiled-additive HPM x3 plus quadratic side context](linear-hpm-x3-quadratic-side-context.md) | **1.192719 (1)** | **1.137554 (1)** | **0.0630% (1)** | **16.5978 (1)** | **7.6986% (1)** |
| HPM v2 shooting composition | 1.192759 (2) | 1.137652 (5) | 0.0564% (2) | 16.6913 (10) | 6.6559% (10) |
| [HPM x2 raw OREB/100 context](hpm-x2.md) | 1.192760 (3) | 1.137716 (11) | 0.0562% (3) | 16.6382 (4) | 7.2857% (4) |
| Complete player-prior RAPM, no context or box score | 1.192761 (4) | 1.137741 (12) | 0.0559% (4) | 16.6068 (2) | 7.5991% (2) |
| [HPM x1 ORB claim context](hpm-x1.md) | 1.192763 (5) | 1.137713 (10) | 0.0557% (5) | 16.6315 (3) | 7.3242% (3) |
| [NAIL-RAPM v1.0](hpm-x3-linear-ridge-without-uncertainty.md) | 1.192766 (6) | 1.137639 (3) | 0.0552% (6) | 16.6817 (9) | 6.7642% (9) |
| HPM v2.1 empirical rebound capacity | 1.192772 (7) | 1.137654 (6) | 0.0542% (7) | 16.6701 (6) | 6.8932% (6) |
| [HPM v2.3 shot portfolio](hpm-v23.md) | 1.192777 (8) | 1.137641 (4) | 0.0534% (8) | 16.6546 (5) | 7.0659% (5) |
| [HPM x3 ORB claim rebound replacement](hpm-x3.md) | 1.192781 (9) | 1.137688 (8) | 0.0526% (9) | 16.6792 (7) | 6.7912% (7) |
| Value-Conditioned Aging HPM | 1.192792 (10) | 1.137701 (9) | 0.0508% (10) | 16.7155 (11) | 6.3851% (11) |
| HPM v2.2 usage allocation | 1.192807 (11) | 1.137656 (7) | 0.0482% (11) | 16.6806 (8) | 6.7759% (8) |
| [Additive prior plus linear non-additive context](additive-profile-linear-shape-context.md) | 1.192814 (12) | 1.137613 (2) | 0.0470% (12) | 16.7224 (12) | 6.3076% (12) |
| Forward 3-year RAPM-prior baseline | 1.193041 (13) | 1.137974 (14) | 0.0091% (13) | 17.1808 (13) | 1.1016% (13) |
| Forward 1-year RAPM-prior baseline | 1.193123 (14) | 1.137869 (13) | -0.0047% (14) | 17.2902 (14) | -0.1627% (14) |

See [Forward RAPM Memory Baselines](forward-rapm-memory-baselines.md) and
[Complete Player-Prior RAPM Baseline](complete-player-prior-baseline.md) for
model specifications, annual lambda selections, and immutable artifacts.
