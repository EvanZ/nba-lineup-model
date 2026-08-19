---
last_updated: "2026-08-18"
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
| [NAIL-RAPM v1.1 stat-specific padding](nail-rapm-v11-profile-padding.md) | **1** | 1.197979 (2) | 1.141391 (5) | 0.1224% (2) | **14.0864 (1)** | **17.4775% (1)** | **14.3236 (1)** | **68.53% (1)** | **3.3898 (1)** | **7.2757 (1)** |
| [NAIL-RAPM fixed context alpha=20,000](nail-context-regularization.md) | 3 | 1.197980 (3) | 1.141411 (12) | 0.1221% (3) | 14.0907 (2) | 17.4268% (2) | 14.3296 (2) | 68.24% (3) | 3.4095 (3) | 7.2966 (4) |
| [NAIL-RAPM fixed context alpha=5,000](nail-context-regularization.md) | 4 | 1.197994 (6) | 1.141463 (16) | 0.1199% (6) | 14.0949 (3) | 17.3778% (3) | 14.3300 (3) | 67.84% (9) | 3.4156 (4) | 7.2821 (3) |
| [NAIL-RAPM fixed context alpha=1,000](nail-context-regularization.md) | 4 | 1.197999 (8) | 1.141439 (15) | 0.1190% (8) | 14.1017 (4) | 17.2975% (4) | 14.3389 (4) | 67.99% (6) | 3.4009 (2) | 7.2776 (2) |
| [NAIL-RAPM v1.0](nail-rapm-v1.md) | 5 | **1.197977 (1)** | 1.141387 (4) | **0.1226% (1)** | 14.1119 (5) | 17.1779% (5) | 14.3516 (5) | 68.33% (2) | 3.4307 (7) | 7.3460 (6) |
| [NAIL token-MLP residual](nail-token-residual.md) | 6 | 1.197986 (4) | 1.141411 (12) | 0.1211% (4) | 14.1207 (6) | 17.0756% (6) | 14.3698 (6) | 67.99% (5) | 3.4669 (10) | 7.3841 (8) |
| [NAIL-RAPM additive-only context](nail-additive-only-context.md) | 7 | 1.197989 (5) | 1.141430 (14) | 0.1206% (5) | 14.1286 (7) | 16.9824% (7) | 14.3754 (7) | 67.93% (7) | 3.4783 (11) | 7.4728 (12) |
| [HPM x3 ORB claim rebound replacement](hpm-x3.md) | 8 | 1.198000 (9) | 1.141400 (8) | 0.1188% (9) | 14.1309 (8) | 16.9555% (8) | 14.3774 (8) | 67.73% (13) | 3.4229 (5) | 7.3269 (5) |
| Value-Conditioned Aging HPM | 10 | 1.198010 (13) | 1.141402 (9) | 0.1171% (13) | 14.1342 (12) | 16.9169% (12) | 14.3802 (9) | 67.84% (10) | 3.4290 (6) | 7.3532 (7) |
| [NAIL-RAPM normalized context penalty](nail-context-regularization.md) | 10 | 1.198002 (10) | 1.141533 (17) | 0.1185% (10) | 14.1339 (10) | 16.9203% (10) | 14.3822 (10) | 68.01% (4) | 3.5234 (16) | 7.4597 (11) |
| [Compiled-additive HPM x3 plus quadratic side context](linear-hpm-x3-quadratic-side-context.md) | 10 | 1.198002 (10) | **1.141366 (1)** | 0.1185% (10) | 14.1311 (9) | 16.9525% (9) | 14.3922 (13) | 67.36% (20) | 3.5139 (15) | 7.5889 (17) |
| HPM v2.1 empirical rebound capacity | 11 | 1.198010 (13) | 1.141403 (10) | 0.1171% (13) | 14.1340 (11) | 16.9193% (11) | 14.3869 (11) | 67.67% (16) | 3.4365 (8) | 7.3881 (9) |
| HPM v2 shooting composition | 13 | 1.198022 (15) | 1.141404 (11) | 0.1151% (15) | 14.1409 (13) | 16.8378% (13) | 14.3908 (12) | 67.70% (14) | 3.4592 (9) | 7.4282 (10) |
| HPM v2.2 usage allocation | 13 | 1.198004 (12) | 1.141380 (3) | 0.1182% (12) | 14.1464 (15) | 16.7730% (15) | 14.4050 (14) | 67.47% (19) | 3.4802 (12) | 7.4793 (13) |
| [NAIL Set Attention residual](nail-token-residual.md) | 14 | 1.197997 (7) | 1.141396 (6) | 0.1192% (7) | 14.1458 (14) | 16.7796% (14) | 14.4098 (15) | 67.64% (17) | 3.5075 (14) | 7.5605 (16) |
| [HPM v2.3 shot portfolio](hpm-v23.md) | 16 | 1.198025 (16) | 1.141396 (6) | 0.1147% (16) | 14.1815 (16) | 16.3594% (16) | 14.4288 (16) | 67.70% (14) | 3.5037 (13) | 7.5274 (15) |
| [Additive prior plus linear non-additive context](additive-profile-linear-shape-context.md) | 17 | 1.198026 (17) | **1.141366 (1)** | 0.1144% (17) | 14.2154 (20) | 15.9592% (20) | 14.4542 (17) | 67.76% (12) | 3.5492 (17) | 7.4793 (13) |
| [HPM x1 ORB claim context](hpm-x1.md) | 18 | 1.198047 (18) | 1.141658 (18) | 0.1110% (18) | 14.2081 (18) | 16.0449% (18) | 14.4665 (19) | 67.87% (8) | 3.6653 (19) | 7.7498 (19) |
| [HPM x2 raw OREB/100 context](hpm-x2.md) | 18 | 1.198048 (19) | 1.141660 (19) | 0.1108% (19) | 14.2073 (17) | 16.0552% (17) | 14.4641 (18) | 67.79% (11) | 3.6558 (18) | 7.7389 (18) |
| Complete player-prior RAPM, no context or box score | 20 | 1.198061 (20) | 1.141675 (20) | 0.1086% (20) | 14.2105 (19) | 16.0175% (19) | 14.4756 (20) | 67.64% (17) | 3.6754 (20) | 7.7561 (20) |
| Forward 1-year RAPM-prior baseline | 21 | 1.198196 (21) | 1.141757 (21) | 0.0861% (21) | 14.5265 (21) | 12.2406% (21) | 14.8154 (21) | 65.39% (21) | 4.2374 (21) | 9.1066 (21) |
| Forward 3-year RAPM-prior baseline | 22 | 1.198246 (22) | 1.141950 (22) | 0.0778% (22) | 14.6559 (22) | 10.6699% (22) | 14.9714 (22) | 64.68% (22) | 4.5326 (22) | 9.7789 (22) |

## Playoffs

Pooled over 39,967 eligible possessions from 238 games. Each playoff cohort
uses its matching frozen pre-season player prior and prior-year context state;
postseason outcomes are evaluation-only. Models use the same median-rank
ordering and tie-breakers as the regular-season table.

| Model | Median rank | Poss. RMSE | Poss. MAE | Poss. skill | Eligible game RMSE | Eligible game skill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [NAIL-RAPM additive-only context](nail-additive-only-context.md) | **1** | **1.192713 (1)** | 1.137590 (3) | **0.0640% (1)** | **16.5724 (1)** | **7.9812% (1)** |
| [Compiled-additive HPM x3 plus quadratic side context](linear-hpm-x3-quadratic-side-context.md) | 2 | 1.192719 (2) | **1.137554 (1)** | 0.0630% (2) | 16.5978 (2) | 7.6986% (2) |
| [NAIL-RAPM normalized context penalty](nail-context-regularization.md) | 3 | 1.192726 (3) | 1.137680 (13) | 0.0618% (3) | 16.5979 (3) | 7.6979% (3) |
| [NAIL-RAPM fixed context alpha=5,000](nail-context-regularization.md) | 5 | 1.192727 (4) | 1.137753 (20) | 0.0617% (4) | 16.6090 (5) | 7.5741% (5) |
| [NAIL-RAPM fixed context alpha=20,000](nail-context-regularization.md) | 6 | 1.192736 (5) | 1.137655 (10) | 0.0602% (5) | 16.6187 (6) | 7.4664% (6) |
| [NAIL-RAPM fixed context alpha=1,000](nail-context-regularization.md) | 7 | 1.192745 (7) | 1.137743 (19) | 0.0587% (7) | 16.6197 (7) | 7.4553% (7) |
| [NAIL-RAPM v1.1 stat-specific padding](nail-rapm-v11-profile-padding.md) | 8 | 1.192740 (6) | 1.137655 (10) | 0.0595% (6) | 16.6267 (8) | 7.3771% (8) |
| [NAIL token-MLP residual](nail-token-residual.md) | 8 | 1.192747 (8) | 1.137607 (4) | 0.0583% (8) | 16.6393 (11) | 7.2373% (11) |
| [NAIL-RAPM v1.0](nail-rapm-v1.md) | 9 | 1.192752 (9) | 1.137640 (6) | 0.0574% (9) | 16.6636 (13) | 6.9658% (13) |
| HPM v2 shooting composition | 10 | 1.192759 (10) | 1.137652 (8) | 0.0564% (10) | 16.6913 (17) | 6.6559% (17) |
| [HPM x2 raw OREB/100 context](hpm-x2.md) | 11 | 1.192760 (11) | 1.137716 (17) | 0.0562% (11) | 16.6382 (10) | 7.2857% (10) |
| Complete player-prior RAPM, no context or box score | 12 | 1.192761 (12) | 1.137741 (18) | 0.0559% (12) | 16.6068 (4) | 7.5991% (4) |
| [HPM v2.3 shot portfolio](hpm-v23.md) | 12 | 1.192777 (16) | 1.137641 (7) | 0.0534% (16) | 16.6546 (12) | 7.0659% (12) |
| [HPM x1 ORB claim context](hpm-x1.md) | 13 | 1.192763 (13) | 1.137713 (16) | 0.0557% (13) | 16.6315 (9) | 7.3242% (9) |
| HPM v2.1 empirical rebound capacity | 14 | 1.192772 (15) | 1.137654 (9) | 0.0542% (15) | 16.6701 (14) | 6.8932% (14) |
| [NAIL Set Attention residual](nail-token-residual.md) | 14 | 1.192768 (14) | 1.137571 (2) | 0.0547% (14) | 16.7235 (20) | 6.2956% (20) |
| [HPM x3 ORB claim rebound replacement](hpm-x3.md) | 15 | 1.192781 (17) | 1.137688 (14) | 0.0526% (17) | 16.6792 (15) | 6.7912% (15) |
| HPM v2.2 usage allocation | 16 | 1.192807 (19) | 1.137656 (12) | 0.0482% (19) | 16.6806 (16) | 6.7759% (16) |
| Value-Conditioned Aging HPM | 18 | 1.192792 (18) | 1.137701 (15) | 0.0508% (18) | 16.7155 (18) | 6.3851% (18) |
| [Additive prior plus linear non-additive context](additive-profile-linear-shape-context.md) | 19 | 1.192814 (20) | 1.137613 (5) | 0.0470% (20) | 16.7224 (19) | 6.3076% (19) |
| Forward 3-year RAPM-prior baseline | 21 | 1.193041 (21) | 1.137974 (22) | 0.0091% (21) | 17.1808 (21) | 1.1016% (21) |
| Forward 1-year RAPM-prior baseline | 22 | 1.193123 (22) | 1.137869 (21) | -0.0047% (22) | 17.2902 (22) | -0.1627% (22) |

See [Forward RAPM Memory Baselines](forward-rapm-memory-baselines.md) and
[Complete Player-Prior RAPM Baseline](complete-player-prior-baseline.md) for
model specifications, annual lambda selections, and immutable artifacts.
