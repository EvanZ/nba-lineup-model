---
last_updated: "2026-08-21"
---

# Three-Season Frozen Leaderboard

Every candidate forecasts 2023-24, 2024-25, and 2025-26 from information
available before each target season begins. Target-season lineup allocation is
an oracle input; target outcomes never enter the frozen forecast. Bold values
are pooled leaders. Lower is better except skill and winner accuracy.

## Regular Season

Pooled over 584,970 eligible possessions from 3,284 games. Full-game and team
metrics cover 3,511 reconstructed games. Models are ordered by the median of
their displayed metric ranks. Ties are ordered by mean rank, then game RMSE.

| Model | Median rank | Poss. RMSE | Poss. MAE | Poss. skill | Eligible game RMSE | Eligible game skill | Full-game RMSE | Winner accuracy | Team NetRtg RMSE | Pythagorean-win RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [NAIL-RAPM v1.2 gap-returner priors](nail-rapm-v12-gap-returners.md) | **1** | **1.197958 (1)** | **1.141344 (1)** | **0.1258% (1)** | **14.0414 (1)** | **18.0039% (1)** | **14.2660 (1)** | 68.16% (4) | **3.2908 (1)** | **7.0899 (1)** |
| [NAIL-RAPM v1.1 stat-specific padding](nail-rapm-v11-profile-padding.md) | 2 | 1.197979 (3) | 1.141391 (6) | 0.1224% (3) | 14.0864 (2) | 17.4775% (2) | 14.3236 (2) | **68.53% (1)** | 3.3898 (2) | 7.2757 (2) |
| [NAIL-RAPM fixed context alpha=20,000](nail-context-regularization.md) | 4 | 1.197980 (4) | 1.141411 (13) | 0.1221% (4) | 14.0907 (3) | 17.4268% (3) | 14.3296 (3) | 68.24% (3) | 3.4095 (4) | 7.2966 (5) |
| [NAIL-RAPM fixed context alpha=1,000](nail-context-regularization.md) | 5 | 1.197999 (9) | 1.141439 (16) | 0.1190% (9) | 14.1017 (5) | 17.2975% (5) | 14.3389 (5) | 67.99% (6) | 3.4009 (3) | 7.2776 (3) |
| [NAIL-RAPM fixed context alpha=5,000](nail-context-regularization.md) | 5 | 1.197994 (7) | 1.141463 (17) | 0.1199% (7) | 14.0949 (4) | 17.3778% (4) | 14.3300 (4) | 67.84% (10) | 3.4156 (5) | 7.2821 (4) |
| [NAIL-RAPM v1.0](nail-rapm-v1.md) | 6 | 1.197977 (2) | 1.141387 (5) | 0.1226% (2) | 14.1119 (6) | 17.1779% (6) | 14.3516 (6) | 68.33% (2) | 3.4307 (8) | 7.3460 (7) |
| [NAIL token-MLP residual](nail-token-residual.md) | 7 | 1.197986 (5) | 1.141411 (13) | 0.1211% (5) | 14.1207 (7) | 17.0756% (7) | 14.3698 (7) | 67.99% (6) | 3.4669 (11) | 7.3841 (9) |
| [NAIL-RAPM additive-only context](nail-additive-only-context.md) | 8 | 1.197989 (6) | 1.141430 (15) | 0.1206% (6) | 14.1286 (8) | 16.9824% (8) | 14.3754 (8) | 67.93% (8) | 3.4783 (12) | 7.4728 (13) |
| [HPM x3 ORB claim rebound replacement](hpm-x3.md) | 9 | 1.198000 (10) | 1.141400 (9) | 0.1188% (10) | 14.1309 (9) | 16.9555% (9) | 14.3774 (9) | 67.73% (14) | 3.4229 (6) | 7.3269 (6) |
| Value-Conditioned Aging HPM | 10 | 1.198010 (14) | 1.141402 (10) | 0.1171% (14) | 14.1342 (13) | 16.9169% (13) | 14.3802 (10) | 67.84% (10) | 3.4290 (7) | 7.3532 (8) |
| [NAIL-RAPM normalized context penalty](nail-context-regularization.md) | 11 | 1.198002 (11) | 1.141533 (18) | 0.1185% (11) | 14.1339 (11) | 16.9203% (11) | 14.3822 (11) | 68.01% (5) | 3.5234 (17) | 7.4597 (12) |
| [Compiled-additive HPM x3 plus quadratic side context](linear-hpm-x3-quadratic-side-context.md) | 11 | 1.198002 (11) | 1.141366 (2) | 0.1185% (11) | 14.1311 (10) | 16.9525% (10) | 14.3922 (14) | 67.36% (21) | 3.5139 (16) | 7.5889 (18) |
| HPM v2.1 empirical rebound capacity | 12 | 1.198010 (14) | 1.141403 (11) | 0.1171% (14) | 14.1340 (12) | 16.9193% (12) | 14.3869 (12) | 67.67% (17) | 3.4365 (9) | 7.3881 (10) |
| HPM v2 shooting composition | 14 | 1.198022 (16) | 1.141404 (12) | 0.1151% (16) | 14.1409 (14) | 16.8378% (14) | 14.3908 (13) | 67.70% (15) | 3.4592 (10) | 7.4282 (11) |
| HPM v2.2 usage allocation | 14 | 1.198004 (13) | 1.141380 (4) | 0.1182% (13) | 14.1464 (16) | 16.7730% (16) | 14.4050 (15) | 67.47% (20) | 3.4802 (13) | 7.4793 (14) |
| [NAIL Set Attention residual](nail-token-residual.md) | 15 | 1.197997 (8) | 1.141396 (7) | 0.1192% (8) | 14.1458 (15) | 16.7796% (15) | 14.4098 (16) | 67.64% (18) | 3.5075 (15) | 7.5605 (17) |
| [HPM v2.3 shot portfolio](hpm-v23.md) | 17 | 1.198025 (17) | 1.141396 (7) | 0.1147% (17) | 14.1815 (17) | 16.3594% (17) | 14.4288 (17) | 67.70% (15) | 3.5037 (14) | 7.5274 (16) |
| [Additive prior plus linear non-additive context](additive-profile-linear-shape-context.md) | 18 | 1.198026 (18) | 1.141366 (2) | 0.1144% (18) | 14.2154 (21) | 15.9592% (21) | 14.4542 (18) | 67.76% (13) | 3.5492 (18) | 7.4793 (14) |
| [HPM x1 ORB claim context](hpm-x1.md) | 19 | 1.198047 (19) | 1.141658 (19) | 0.1110% (19) | 14.2081 (19) | 16.0449% (19) | 14.4665 (20) | 67.87% (9) | 3.6653 (20) | 7.7498 (20) |
| [HPM x2 raw OREB/100 context](hpm-x2.md) | 19 | 1.198048 (20) | 1.141660 (20) | 0.1108% (20) | 14.2073 (18) | 16.0552% (18) | 14.4641 (19) | 67.79% (12) | 3.6558 (19) | 7.7389 (19) |
| Complete player-prior RAPM, no context or box score | 21 | 1.198061 (21) | 1.141675 (21) | 0.1086% (21) | 14.2105 (20) | 16.0175% (20) | 14.4756 (21) | 67.64% (18) | 3.6754 (21) | 7.7561 (21) |
| Forward 1-year RAPM-prior baseline | 22 | 1.198196 (22) | 1.141757 (22) | 0.0861% (22) | 14.5265 (22) | 12.2406% (22) | 14.8154 (22) | 65.39% (22) | 4.2374 (22) | 9.1066 (22) |
| Forward 3-year RAPM-prior baseline | 23 | 1.198246 (23) | 1.141950 (23) | 0.0778% (23) | 14.6559 (23) | 10.6699% (23) | 14.9714 (23) | 64.68% (23) | 4.5326 (23) | 9.7789 (23) |

## Playoffs

Pooled over 39,967 eligible possessions from 238 games. Each playoff cohort
uses its matching frozen pre-season player prior and prior-year context state;
postseason outcomes are evaluation-only. Models use the same median-rank
ordering and tie-breakers as the regular-season table.

| Model | Median rank | Poss. RMSE | Poss. MAE | Poss. skill | Eligible game RMSE | Eligible game skill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [NAIL-RAPM additive-only context](nail-additive-only-context.md) | **1** | **1.192713 (1)** | 1.137590 (3) | **0.0640% (1)** | **16.5724 (1)** | **7.9812% (1)** |
| [Compiled-additive HPM x3 plus quadratic side context](linear-hpm-x3-quadratic-side-context.md) | 2 | 1.192719 (2) | **1.137554 (1)** | 0.0630% (2) | 16.5978 (2) | 7.6986% (2) |
| [NAIL-RAPM normalized context penalty](nail-context-regularization.md) | 3 | 1.192726 (3) | 1.137680 (14) | 0.0618% (3) | 16.5979 (3) | 7.6979% (3) |
| [NAIL-RAPM fixed context alpha=5,000](nail-context-regularization.md) | 5 | 1.192727 (4) | 1.137753 (21) | 0.0617% (4) | 16.6090 (5) | 7.5741% (5) |
| [NAIL-RAPM v1.2 gap-returner priors](nail-rapm-v12-gap-returners.md) | 6 | 1.192734 (5) | 1.137641 (7) | 0.0605% (5) | 16.6148 (6) | 7.5097% (6) |
| [NAIL-RAPM fixed context alpha=20,000](nail-context-regularization.md) | 7 | 1.192736 (6) | 1.137655 (11) | 0.0602% (6) | 16.6187 (7) | 7.4664% (7) |
| [NAIL-RAPM fixed context alpha=1,000](nail-context-regularization.md) | 8 | 1.192745 (8) | 1.137743 (20) | 0.0587% (8) | 16.6197 (8) | 7.4553% (8) |
| [NAIL-RAPM v1.1 stat-specific padding](nail-rapm-v11-profile-padding.md) | 9 | 1.192740 (7) | 1.137655 (11) | 0.0595% (7) | 16.6267 (9) | 7.3771% (9) |
| [NAIL token-MLP residual](nail-token-residual.md) | 9 | 1.192747 (9) | 1.137607 (4) | 0.0583% (9) | 16.6393 (12) | 7.2373% (12) |
| [NAIL-RAPM v1.0](nail-rapm-v1.md) | 10 | 1.192752 (10) | 1.137640 (6) | 0.0574% (10) | 16.6636 (14) | 6.9658% (14) |
| HPM v2 shooting composition | 11 | 1.192759 (11) | 1.137652 (9) | 0.0564% (11) | 16.6913 (18) | 6.6559% (18) |
| [HPM x2 raw OREB/100 context](hpm-x2.md) | 12 | 1.192760 (12) | 1.137716 (18) | 0.0562% (12) | 16.6382 (11) | 7.2857% (11) |
| Complete player-prior RAPM, no context or box score | 13 | 1.192761 (13) | 1.137741 (19) | 0.0559% (13) | 16.6068 (4) | 7.5991% (4) |
| [HPM v2.3 shot portfolio](hpm-v23.md) | 13 | 1.192777 (17) | 1.137641 (7) | 0.0534% (17) | 16.6546 (13) | 7.0659% (13) |
| [HPM x1 ORB claim context](hpm-x1.md) | 14 | 1.192763 (14) | 1.137713 (17) | 0.0557% (14) | 16.6315 (10) | 7.3242% (10) |
| HPM v2.1 empirical rebound capacity | 15 | 1.192772 (16) | 1.137654 (10) | 0.0542% (16) | 16.6701 (15) | 6.8932% (15) |
| [NAIL Set Attention residual](nail-token-residual.md) | 15 | 1.192768 (15) | 1.137571 (2) | 0.0547% (15) | 16.7235 (21) | 6.2956% (21) |
| [HPM x3 ORB claim rebound replacement](hpm-x3.md) | 16 | 1.192781 (18) | 1.137688 (15) | 0.0526% (18) | 16.6792 (16) | 6.7912% (16) |
| HPM v2.2 usage allocation | 17 | 1.192807 (20) | 1.137656 (13) | 0.0482% (20) | 16.6806 (17) | 6.7759% (17) |
| Value-Conditioned Aging HPM | 19 | 1.192792 (19) | 1.137701 (16) | 0.0508% (19) | 16.7155 (19) | 6.3851% (19) |
| [Additive prior plus linear non-additive context](additive-profile-linear-shape-context.md) | 20 | 1.192814 (21) | 1.137613 (5) | 0.0470% (21) | 16.7224 (20) | 6.3076% (20) |
| Forward 3-year RAPM-prior baseline | 22 | 1.193041 (22) | 1.137974 (23) | 0.0091% (22) | 17.1808 (22) | 1.1016% (22) |
| Forward 1-year RAPM-prior baseline | 23 | 1.193123 (23) | 1.137869 (22) | -0.0047% (23) | 17.2902 (23) | -0.1627% (23) |

See [Forward RAPM Memory Baselines](forward-rapm-memory-baselines.md) and
[Complete Player-Prior RAPM Baseline](complete-player-prior-baseline.md) for
model specifications, annual lambda selections, and immutable artifacts.
