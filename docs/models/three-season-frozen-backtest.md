---
last_updated: "2026-08-22"
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
| [NAIL-RAPM v1.2.1 pruned non-additive context](nail-rapm-v121-pruned-nonadditive.md) | 2 | **1.197952 (1)** | 1.141355 (4) | **0.1268% (1)** | 14.0245 (2) | 18.2005% (2) | **14.2521 (1)** | 68.24% (8) | **3.2706 (1)** | 7.0351 (2) |
| [NAIL-RAPM v1.2.2 defensive-rebound profile (not promoted)](nail-rapm-v122-defensive-rebound-profile.md) | 2 | 1.197952 (2) | 1.141364 (5) | 0.1268% (2) | **14.0238 (1)** | **18.2094% (1)** | 14.2529 (2) | 68.39% (5) | 3.2723 (2) | 7.0434 (3) |
| [NAIL-RAPM v1.2 gap-returner priors](nail-rapm-v12-gap-returners.md) | 3 | 1.197958 (3) | 1.141344 (3) | 0.1258% (3) | 14.0414 (3) | 18.0039% (3) | 14.2660 (3) | 68.16% (11) | 3.2908 (3) | 7.0899 (7) |
| [NAIL-RAPM v1.2.3 free-throw profile (not promoted)](nail-rapm-v123-free-throw-profile.md) | 4 | 1.197964 (4) | 1.141341 (2) | 0.1248% (4) | 14.0456 (4) | 17.9547% (4) | 14.2726 (4) | 68.47% (2) | 3.2974 (5) | 7.0511 (5) |
| [NAIL-RAPM v1.2.4 free-throw replacement (not promoted)](nail-rapm-v124-free-throw-replacement.md) | 5 | 1.197966 (5) | **1.141328 (1)** | 0.1244% (5) | 14.0457 (5) | 17.9541% (5) | 14.2733 (5) | 68.19% (10) | 3.2933 (4) | 7.0459 (4) |
| [NAIL-RAPM v1.4 Kalman additive profiles (not promoted)](nail-rapm-v14-filtered-additive-profiles.md) | 6 | 1.197972 (6) | 1.141432 (19) | 0.1235% (6) | 14.0630 (6) | 17.7513% (6) | 14.2750 (6) | 68.39% (4) | 3.3174 (6) | **7.0088 (1)** |
| [NAIL-RAPM v1.3 additive profiles (not promoted)](nail-rapm-v13-additive-profiles.md) | 7 | 1.197987 (11) | 1.141423 (17) | 0.1211% (10) | 14.0811 (7) | 17.5398% (7) | 14.2981 (7) | 68.44% (3) | 3.3320 (7) | 7.0619 (6) |
| [NAIL-RAPM v1.1 stat-specific padding](nail-rapm-v11-profile-padding.md) | 8 | 1.197979 (8) | 1.141391 (9) | 0.1224% (8) | 14.0864 (8) | 17.4775% (8) | 14.3236 (9) | **68.53% (1)** | 3.3898 (9) | 7.2757 (9) |
| [NAIL-RAPM v1.3.1 pruned additive profile](nail-rapm-v131-pruned-additive-profiles.md) | 9 | 1.197990 (13) | 1.141406 (15) | 0.1205% (12) | 14.0891 (9) | 17.4455% (9) | 14.3072 (8) | 68.36% (6) | 3.3446 (8) | 7.1012 (8) |
| [NAIL-RAPM fixed context alpha=20,000](nail-context-regularization.md) | 10 | 1.197980 (9) | 1.141411 (16) | 0.1221% (9) | 14.0907 (10) | 17.4268% (10) | 14.3296 (10) | 68.24% (9) | 3.4095 (11) | 7.2966 (12) |
| [NAIL-RAPM fixed context alpha=5,000](nail-context-regularization.md) | 12 | 1.197994 (14) | 1.141463 (21) | 0.1199% (13) | 14.0949 (11) | 17.3778% (11) | 14.3300 (11) | 67.84% (16) | 3.4156 (12) | 7.2821 (11) |
| [NAIL-RAPM fixed context alpha=1,000](nail-context-regularization.md) | 12 | 1.197999 (16) | 1.141439 (20) | 0.1190% (15) | 14.1017 (12) | 17.2975% (12) | 14.3389 (12) | 67.99% (13) | 3.4009 (10) | 7.2776 (10) |
| [NAIL-RAPM v1.0](nail-rapm-v1.md) | 13 | 1.197977 (7) | 1.141387 (8) | 0.1226% (7) | 14.1119 (13) | 17.1779% (13) | 14.3516 (13) | 68.33% (7) | 3.4307 (15) | 7.3460 (14) |
| [NAIL token-MLP residual](nail-token-residual.md) | 14 | 1.197986 (10) | 1.141411 (16) | 0.1211% (10) | 14.1207 (14) | 17.0756% (14) | 14.3698 (14) | 67.99% (13) | 3.4669 (18) | 7.3841 (16) |
| [NAIL-RAPM additive-only context](nail-additive-only-context.md) | 15 | 1.197989 (12) | 1.141430 (18) | 0.1206% (11) | 14.1286 (15) | 16.9824% (15) | 14.3754 (15) | 67.93% (14) | 3.4783 (19) | 7.4728 (20) |
| [HPM x3 ORB claim rebound replacement](hpm-x3.md) | 16 | 1.198000 (17) | 1.141400 (11) | 0.1188% (16) | 14.1309 (16) | 16.9555% (16) | 14.3774 (16) | 67.73% (19) | 3.4229 (13) | 7.3269 (13) |
| Value-Conditioned Aging HPM | 17 | 1.198010 (20) | 1.141402 (12) | 0.1171% (19) | 14.1342 (20) | 16.9169% (20) | 14.3802 (17) | 67.84% (16) | 3.4290 (14) | 7.3532 (15) |
| [NAIL-RAPM normalized context penalty](nail-context-regularization.md) | 18 | 1.198002 (18) | 1.141533 (22) | 0.1185% (17) | 14.1339 (18) | 16.9203% (18) | 14.3822 (18) | 68.01% (12) | 3.5234 (24) | 7.4597 (19) |
| [Compiled-additive HPM x3 plus quadratic side context](linear-hpm-x3-quadratic-side-context.md) | 18 | 1.198002 (18) | 1.141366 (6) | 0.1185% (17) | 14.1311 (17) | 16.9525% (17) | 14.3922 (21) | 67.36% (24) | 3.5139 (23) | 7.5889 (24) |
| HPM v2.1 empirical rebound capacity | 19 | 1.198010 (20) | 1.141403 (13) | 0.1171% (19) | 14.1340 (19) | 16.9193% (19) | 14.3869 (19) | 67.67% (21) | 3.4365 (16) | 7.3881 (17) |
| HPM v2 shooting composition | 20 | 1.198022 (21) | 1.141404 (14) | 0.1151% (20) | 14.1409 (21) | 16.8378% (21) | 14.3908 (20) | 67.70% (20) | 3.4592 (17) | 7.4282 (18) |
| HPM v2.2 usage allocation | 21 | 1.198004 (19) | 1.141380 (7) | 0.1182% (18) | 14.1464 (23) | 16.7730% (23) | 14.4050 (22) | 67.47% (23) | 3.4802 (20) | 7.4793 (21) |
| [NAIL Set Attention residual](nail-token-residual.md) | 22 | 1.197997 (15) | 1.141396 (10) | 0.1192% (14) | 14.1458 (22) | 16.7796% (22) | 14.4098 (23) | 67.64% (22) | 3.5075 (22) | 7.5605 (23) |
| [HPM v2.3 shot portfolio](hpm-v23.md) | 22 | 1.198025 (22) | 1.141396 (10) | 0.1147% (21) | 14.1815 (24) | 16.3594% (24) | 14.4288 (24) | 67.70% (20) | 3.5037 (21) | 7.5274 (22) |
| [Additive prior plus linear non-additive context](additive-profile-linear-shape-context.md) | 23 | 1.198026 (23) | 1.141366 (6) | 0.1144% (22) | 14.2154 (28) | 15.9592% (28) | 14.4542 (25) | 67.76% (18) | 3.5492 (25) | 7.4793 (21) |
| [HPM x2 raw OREB/100 context](hpm-x2.md) | 25 | 1.198048 (25) | 1.141660 (24) | 0.1108% (24) | 14.2073 (25) | 16.0552% (25) | 14.4641 (26) | 67.79% (17) | 3.6558 (26) | 7.7389 (25) |
| [HPM x1 ORB claim context](hpm-x1.md) | 26 | 1.198047 (24) | 1.141658 (23) | 0.1110% (23) | 14.2081 (26) | 16.0449% (26) | 14.4665 (27) | 67.87% (15) | 3.6653 (27) | 7.7498 (26) |
| Complete player-prior RAPM, no context or box score | 27 | 1.198061 (26) | 1.141675 (25) | 0.1086% (25) | 14.2105 (27) | 16.0175% (27) | 14.4756 (28) | 67.64% (22) | 3.6754 (28) | 7.7561 (27) |
| Forward 1-year RAPM-prior baseline | 28 | 1.198196 (27) | 1.141757 (26) | 0.0861% (26) | 14.5265 (29) | 12.2406% (29) | 14.8154 (29) | 65.39% (25) | 4.2374 (29) | 9.1066 (28) |
| Forward 3-year RAPM-prior baseline | 29 | 1.198246 (28) | 1.141950 (27) | 0.0778% (27) | 14.6559 (30) | 10.6699% (30) | 14.9714 (30) | 64.68% (26) | 4.5326 (30) | 9.7789 (29) |

NAIL-RAPM v1.3.1 is the preferred, parsimonious version of the non-promoted
v1.3 branch: it passed its direct no-material-harm gate versus v1.3, but it
does not supersede v1.2.1 as the global regular-season leader. See its
[experiment record](nail-rapm-v131-pruned-additive-profiles.md).

## Playoffs

Pooled over 39,967 eligible possessions from 238 games. Each playoff cohort
uses its matching frozen pre-season player prior and prior-year context state;
postseason outcomes are evaluation-only. Models use the same median-rank
ordering and tie-breakers as the regular-season table.

| Model | Median rank | Poss. RMSE | Poss. MAE | Poss. skill | Eligible game RMSE | Eligible game skill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [NAIL-RAPM v1.3 additive profiles (not promoted)](nail-rapm-v13-additive-profiles.md) | **1** | **1.192630 (1)** | 1.137623 (10) | **0.0780% (1)** | **16.5034 (1)** | **8.7464% (1)** |
| [NAIL-RAPM v1.3.1 pruned additive profile](nail-rapm-v131-pruned-additive-profiles.md) | 2 | 1.192646 (2) | 1.137625 (11) | 0.0753% (2) | 16.5248 (2) | 8.5093% (2) |
| [NAIL-RAPM v1.4 Kalman additive profiles (not promoted)](nail-rapm-v14-filtered-additive-profiles.md) | 4 | 1.192674 (3) | 1.137670 (18) | 0.0706% (3) | 16.5836 (4) | 7.8566% (4) |
| [NAIL-RAPM additive-only context](nail-additive-only-context.md) | 5 | 1.192713 (7) | 1.137590 (5) | 0.0640% (7) | 16.5724 (3) | 7.9812% (3) |
| [NAIL-RAPM v1.2.2 defensive-rebound profile (not promoted)](nail-rapm-v122-defensive-rebound-profile.md) | 5 | 1.192702 (4) | 1.137609 (8) | 0.0659% (4) | 16.5875 (5) | 7.8132% (5) |
| [NAIL-RAPM v1.2.4 free-throw replacement (not promoted)](nail-rapm-v124-free-throw-replacement.md) | 5 | 1.192706 (5) | 1.137567 (2) | 0.0652% (5) | 16.5988 (9) | 7.6880% (9) |
| [NAIL-RAPM v1.2.1 pruned non-additive context](nail-rapm-v121-pruned-nonadditive.md) | 6 | 1.192709 (6) | 1.137608 (7) | 0.0647% (6) | 16.5942 (6) | 7.7392% (6) |
| [Compiled-additive HPM x3 plus quadratic side context](linear-hpm-x3-quadratic-side-context.md) | 7 | 1.192719 (9) | **1.137554 (1)** | 0.0630% (9) | 16.5978 (7) | 7.6986% (7) |
| [NAIL-RAPM v1.2.3 free-throw profile (not promoted)](nail-rapm-v123-free-throw-profile.md) | 8 | 1.192714 (8) | 1.137583 (4) | 0.0638% (8) | 16.6157 (13) | 7.5004% (13) |
| [NAIL-RAPM normalized context penalty](nail-context-regularization.md) | 10 | 1.192726 (10) | 1.137680 (19) | 0.0618% (10) | 16.5979 (8) | 7.6979% (8) |
| [NAIL-RAPM fixed context alpha=5,000](nail-context-regularization.md) | 11 | 1.192727 (11) | 1.137753 (26) | 0.0617% (11) | 16.6090 (11) | 7.5741% (11) |
| [NAIL-RAPM v1.2 gap-returner priors](nail-rapm-v12-gap-returners.md) | 12 | 1.192734 (12) | 1.137641 (13) | 0.0605% (12) | 16.6148 (12) | 7.5097% (12) |
| [NAIL-RAPM fixed context alpha=20,000](nail-context-regularization.md) | 14 | 1.192736 (13) | 1.137655 (16) | 0.0602% (13) | 16.6187 (14) | 7.4664% (14) |
| [NAIL-RAPM fixed context alpha=1,000](nail-context-regularization.md) | 15 | 1.192745 (15) | 1.137743 (25) | 0.0587% (15) | 16.6197 (15) | 7.4553% (15) |
| [NAIL-RAPM v1.1 stat-specific padding](nail-rapm-v11-profile-padding.md) | 16 | 1.192740 (14) | 1.137655 (16) | 0.0595% (14) | 16.6267 (16) | 7.3771% (16) |
| [NAIL token-MLP residual](nail-token-residual.md) | 16 | 1.192747 (16) | 1.137607 (6) | 0.0583% (16) | 16.6393 (19) | 7.2373% (19) |
| [NAIL-RAPM v1.0](nail-rapm-v1.md) | 17 | 1.192752 (17) | 1.137640 (12) | 0.0574% (17) | 16.6636 (21) | 6.9658% (21) |
| HPM v2 shooting composition | 18 | 1.192759 (18) | 1.137652 (14) | 0.0564% (18) | 16.6913 (25) | 6.6559% (25) |
| [HPM x2 raw OREB/100 context](hpm-x2.md) | 19 | 1.192760 (19) | 1.137716 (23) | 0.0562% (19) | 16.6382 (18) | 7.2857% (18) |
| Complete player-prior RAPM, no context or box score | 20 | 1.192761 (20) | 1.137741 (24) | 0.0559% (20) | 16.6068 (10) | 7.5991% (10) |
| [HPM v2.3 shot portfolio](hpm-v23.md) | 20 | 1.192777 (24) | 1.137641 (13) | 0.0534% (24) | 16.6546 (20) | 7.0659% (20) |
| [HPM x1 ORB claim context](hpm-x1.md) | 21 | 1.192763 (21) | 1.137713 (22) | 0.0557% (21) | 16.6315 (17) | 7.3242% (17) |
| [NAIL Set Attention residual](nail-token-residual.md) | 22 | 1.192768 (22) | 1.137571 (3) | 0.0547% (22) | 16.7235 (28) | 6.2956% (28) |
| HPM v2.1 empirical rebound capacity | 22 | 1.192772 (23) | 1.137654 (15) | 0.0542% (23) | 16.6701 (22) | 6.8932% (22) |
| [HPM x3 ORB claim rebound replacement](hpm-x3.md) | 23 | 1.192781 (25) | 1.137688 (20) | 0.0526% (25) | 16.6792 (23) | 6.7912% (23) |
| HPM v2.2 usage allocation | 24 | 1.192807 (27) | 1.137656 (17) | 0.0482% (27) | 16.6806 (24) | 6.7759% (24) |
| Value-Conditioned Aging HPM | 26 | 1.192792 (26) | 1.137701 (21) | 0.0508% (26) | 16.7155 (26) | 6.3851% (26) |
| [Additive prior plus linear non-additive context](additive-profile-linear-shape-context.md) | 27 | 1.192814 (28) | 1.137613 (9) | 0.0470% (28) | 16.7224 (27) | 6.3076% (27) |
| Forward 3-year RAPM-prior baseline | 29 | 1.193041 (29) | 1.137974 (28) | 0.0091% (29) | 17.1808 (29) | 1.1016% (29) |
| Forward 1-year RAPM-prior baseline | 30 | 1.193123 (30) | 1.137869 (27) | -0.0047% (30) | 17.2902 (30) | -0.1627% (30) |

NAIL-RAPM v1.3.1 is the preferred, parsimonious version of the non-promoted
v1.3 branch: it passed its direct no-material-harm gate versus v1.3, but it
does not supersede v1.2.1 as the global regular-season leader. See its
[experiment record](nail-rapm-v131-pruned-additive-profiles.md).

See [Forward RAPM Memory Baselines](forward-rapm-memory-baselines.md) and
[Complete Player-Prior RAPM Baseline](complete-player-prior-baseline.md) for
model specifications, annual lambda selections, and immutable artifacts.
