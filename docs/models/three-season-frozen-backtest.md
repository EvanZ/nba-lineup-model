---
last_updated: "2026-08-23"
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
| [NAIL-RAPM v1.2.1.1 standard USG% (not promoted)](nail-rapm-v1211-standard-usage.md) | 2 | **1.197951 (1)** | 1.141332 (2) | 0.1269% (2) | 14.0264 (3) | 18.1792% (3) | **14.2516 (1)** | 68.10% (12) | **3.2658 (1)** | 7.0279 (2) |
| [NAIL-RAPM v1.2.1.1 standard USG%](nail-rapm-v1211-standard-usage.md) | 2 | **1.197951 (1)** | 1.141332 (2) | **0.1269% (1)** | 14.0264 (4) | 18.1792% (3) | 14.2516 (2) | 68.10% (13) | 3.2658 (2) | 7.0279 (3) |
| [NAIL-RAPM v1.2.1 pruned non-additive context](nail-rapm-v121-pruned-nonadditive.md) | 3 | 1.197952 (2) | 1.141355 (5) | 0.1268% (3) | 14.0245 (2) | 18.2005% (2) | 14.2521 (3) | 68.24% (8) | 3.2706 (3) | 7.0351 (4) |
| [NAIL-RAPM v1.2.2 defensive-rebound profile (not promoted)](nail-rapm-v122-defensive-rebound-profile.md) | 4 | 1.197952 (3) | 1.141364 (6) | 0.1268% (4) | **14.0238 (1)** | **18.2094% (1)** | 14.2529 (4) | 68.39% (5) | 3.2723 (4) | 7.0434 (5) |
| [NAIL-RAPM v1.2 gap-returner priors](nail-rapm-v12-gap-returners.md) | 5 | 1.197958 (4) | 1.141344 (4) | 0.1258% (5) | 14.0414 (5) | 18.0039% (4) | 14.2660 (5) | 68.16% (11) | 3.2908 (5) | 7.0899 (9) |
| [NAIL-RAPM v1.2.3 free-throw profile (not promoted)](nail-rapm-v123-free-throw-profile.md) | 6 | 1.197964 (5) | 1.141341 (3) | 0.1248% (6) | 14.0456 (6) | 17.9547% (5) | 14.2726 (6) | 68.47% (2) | 3.2974 (7) | 7.0511 (7) |
| [NAIL-RAPM v1.2.4 free-throw replacement (not promoted)](nail-rapm-v124-free-throw-replacement.md) | 6 | 1.197966 (6) | **1.141328 (1)** | 0.1244% (7) | 14.0457 (7) | 17.9541% (6) | 14.2733 (7) | 68.19% (10) | 3.2933 (6) | 7.0459 (6) |
| [NAIL-RAPM v1.4 Kalman additive profiles (not promoted)](nail-rapm-v14-filtered-additive-profiles.md) | 8 | 1.197972 (7) | 1.141432 (20) | 0.1235% (8) | 14.0630 (8) | 17.7513% (7) | 14.2750 (8) | 68.39% (4) | 3.3174 (8) | **7.0088 (1)** |
| [NAIL-RAPM v1.3 additive profiles (not promoted)](nail-rapm-v13-additive-profiles.md) | 9 | 1.197987 (12) | 1.141423 (18) | 0.1211% (12) | 14.0811 (9) | 17.5398% (8) | 14.2981 (9) | 68.44% (3) | 3.3320 (9) | 7.0619 (8) |
| [NAIL-RAPM v1.1 stat-specific padding](nail-rapm-v11-profile-padding.md) | 10 | 1.197979 (9) | 1.141391 (10) | 0.1224% (10) | 14.0864 (10) | 17.4775% (9) | 14.3236 (11) | **68.53% (1)** | 3.3898 (11) | 7.2757 (11) |
| [NAIL-RAPM v1.3.1 pruned additive profile](nail-rapm-v131-pruned-additive-profiles.md) | 10 | 1.197990 (14) | 1.141406 (16) | 0.1205% (14) | 14.0891 (11) | 17.4455% (10) | 14.3072 (10) | 68.36% (6) | 3.3446 (10) | 7.1012 (10) |
| [NAIL-RAPM fixed context alpha=20,000](nail-context-regularization.md) | 12 | 1.197980 (10) | 1.141411 (17) | 0.1221% (11) | 14.0907 (12) | 17.4268% (11) | 14.3296 (12) | 68.24% (9) | 3.4095 (13) | 7.2966 (14) |
| [NAIL-RAPM v1.0](nail-rapm-v1.md) | 14 | 1.197977 (8) | 1.141387 (9) | 0.1226% (9) | 14.1119 (15) | 17.1779% (14) | 14.3516 (15) | 68.33% (7) | 3.4307 (17) | 7.3460 (16) |
| [NAIL-RAPM fixed context alpha=5,000](nail-context-regularization.md) | 14 | 1.197994 (15) | 1.141463 (22) | 0.1199% (15) | 14.0949 (13) | 17.3778% (12) | 14.3300 (13) | 67.84% (18) | 3.4156 (14) | 7.2821 (13) |
| [NAIL-RAPM fixed context alpha=1,000](nail-context-regularization.md) | 14 | 1.197999 (17) | 1.141439 (21) | 0.1190% (17) | 14.1017 (14) | 17.2975% (13) | 14.3389 (14) | 67.99% (15) | 3.4009 (12) | 7.2776 (12) |
| [NAIL token-MLP residual](nail-token-residual.md) | 16 | 1.197986 (11) | 1.141411 (17) | 0.1211% (12) | 14.1207 (16) | 17.0756% (15) | 14.3698 (16) | 67.99% (15) | 3.4669 (20) | 7.3841 (18) |
| [NAIL-RAPM additive-only context](nail-additive-only-context.md) | 17 | 1.197989 (13) | 1.141430 (19) | 0.1206% (13) | 14.1286 (17) | 16.9824% (16) | 14.3754 (17) | 67.93% (16) | 3.4783 (21) | 7.4728 (22) |
| [HPM x3 ORB claim rebound replacement](hpm-x3.md) | 18 | 1.198000 (18) | 1.141400 (12) | 0.1188% (18) | 14.1309 (18) | 16.9555% (17) | 14.3774 (18) | 67.73% (21) | 3.4229 (15) | 7.3269 (15) |
| Value-Conditioned Aging HPM | 19 | 1.198010 (21) | 1.141402 (13) | 0.1171% (21) | 14.1342 (22) | 16.9169% (21) | 14.3802 (19) | 67.84% (18) | 3.4290 (16) | 7.3532 (17) |
| [Compiled-additive HPM x3 plus quadratic side context](linear-hpm-x3-quadratic-side-context.md) | 19 | 1.198002 (19) | 1.141366 (7) | 0.1185% (19) | 14.1311 (19) | 16.9525% (18) | 14.3922 (23) | 67.36% (26) | 3.5139 (25) | 7.5889 (26) |
| [NAIL-RAPM normalized context penalty](nail-context-regularization.md) | 20 | 1.198002 (19) | 1.141533 (23) | 0.1185% (19) | 14.1339 (20) | 16.9203% (19) | 14.3822 (20) | 68.01% (14) | 3.5234 (26) | 7.4597 (21) |
| HPM v2.1 empirical rebound capacity | 21 | 1.198010 (21) | 1.141403 (14) | 0.1171% (21) | 14.1340 (21) | 16.9193% (20) | 14.3869 (21) | 67.67% (23) | 3.4365 (18) | 7.3881 (19) |
| HPM v2 shooting composition | 22 | 1.198022 (22) | 1.141404 (15) | 0.1151% (22) | 14.1409 (23) | 16.8378% (22) | 14.3908 (22) | 67.70% (22) | 3.4592 (19) | 7.4282 (20) |
| HPM v2.2 usage allocation | 23 | 1.198004 (20) | 1.141380 (8) | 0.1182% (20) | 14.1464 (25) | 16.7730% (24) | 14.4050 (24) | 67.47% (25) | 3.4802 (22) | 7.4793 (23) |
| [HPM v2.3 shot portfolio](hpm-v23.md) | 23 | 1.198025 (23) | 1.141396 (11) | 0.1147% (23) | 14.1815 (26) | 16.3594% (25) | 14.4288 (26) | 67.70% (22) | 3.5037 (23) | 7.5274 (24) |
| [NAIL Set Attention residual](nail-token-residual.md) | 24 | 1.197997 (16) | 1.141396 (11) | 0.1192% (16) | 14.1458 (24) | 16.7796% (23) | 14.4098 (25) | 67.64% (24) | 3.5075 (24) | 7.5605 (25) |
| [Additive prior plus linear non-additive context](additive-profile-linear-shape-context.md) | 24 | 1.198026 (24) | 1.141366 (7) | 0.1144% (24) | 14.2154 (30) | 15.9592% (29) | 14.4542 (27) | 67.76% (20) | 3.5492 (27) | 7.4793 (23) |
| [HPM x2 raw OREB/100 context](hpm-x2.md) | 26 | 1.198048 (26) | 1.141660 (25) | 0.1108% (26) | 14.2073 (27) | 16.0552% (26) | 14.4641 (28) | 67.79% (19) | 3.6558 (28) | 7.7389 (27) |
| [HPM x1 ORB claim context](hpm-x1.md) | 27 | 1.198047 (25) | 1.141658 (24) | 0.1110% (25) | 14.2081 (28) | 16.0449% (27) | 14.4665 (29) | 67.87% (17) | 3.6653 (29) | 7.7498 (28) |
| Complete player-prior RAPM, no context or box score | 28 | 1.198061 (27) | 1.141675 (26) | 0.1086% (27) | 14.2105 (29) | 16.0175% (28) | 14.4756 (30) | 67.64% (24) | 3.6754 (30) | 7.7561 (29) |
| Forward 1-year RAPM-prior baseline | 30 | 1.198196 (28) | 1.141757 (27) | 0.0861% (28) | 14.5265 (31) | 12.2406% (30) | 14.8154 (31) | 65.39% (27) | 4.2374 (31) | 9.1066 (30) |
| Forward 3-year RAPM-prior baseline | 31 | 1.198246 (29) | 1.141950 (28) | 0.0778% (29) | 14.6559 (32) | 10.6699% (31) | 14.9714 (32) | 64.68% (28) | 4.5326 (32) | 9.7789 (31) |

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
| [NAIL-RAPM v1.3 additive profiles (not promoted)](nail-rapm-v13-additive-profiles.md) | **1** | **1.192630 (1)** | 1.137623 (11) | **0.0780% (1)** | **16.5034 (1)** | **8.7464% (1)** |
| [NAIL-RAPM v1.3.1 pruned additive profile](nail-rapm-v131-pruned-additive-profiles.md) | 2 | 1.192646 (2) | 1.137625 (12) | 0.0753% (2) | 16.5248 (2) | 8.5093% (2) |
| [NAIL-RAPM v1.4 Kalman additive profiles (not promoted)](nail-rapm-v14-filtered-additive-profiles.md) | 4 | 1.192674 (3) | 1.137670 (19) | 0.0706% (3) | 16.5836 (4) | 7.8566% (4) |
| [NAIL-RAPM additive-only context](nail-additive-only-context.md) | 5 | 1.192713 (8) | 1.137590 (5) | 0.0640% (8) | 16.5724 (3) | 7.9812% (3) |
| [NAIL-RAPM v1.2.2 defensive-rebound profile (not promoted)](nail-rapm-v122-defensive-rebound-profile.md) | 5 | 1.192702 (4) | 1.137609 (9) | 0.0659% (4) | 16.5875 (5) | 7.8132% (5) |
| [NAIL-RAPM v1.2.4 free-throw replacement (not promoted)](nail-rapm-v124-free-throw-replacement.md) | 5 | 1.192706 (5) | 1.137567 (2) | 0.0652% (5) | 16.5988 (9) | 7.6880% (9) |
| [NAIL-RAPM v1.2.1.1 standard USG% (not promoted)](nail-rapm-v1211-standard-usage.md) | 6 | 1.192708 (6) | 1.137593 (6) | 0.0649% (6) | 16.5989 (10) | 7.6873% (10) |
| [NAIL-RAPM v1.2.1.1 standard USG%](nail-rapm-v1211-standard-usage.md) | 6 | 1.192708 (6) | 1.137593 (6) | 0.0649% (6) | 16.5989 (11) | 7.6873% (10) |
| [NAIL-RAPM v1.2.1 pruned non-additive context](nail-rapm-v121-pruned-nonadditive.md) | 7 | 1.192709 (7) | 1.137608 (8) | 0.0647% (7) | 16.5942 (6) | 7.7392% (6) |
| [Compiled-additive HPM x3 plus quadratic side context](linear-hpm-x3-quadratic-side-context.md) | 7 | 1.192719 (10) | **1.137554 (1)** | 0.0630% (10) | 16.5978 (7) | 7.6986% (7) |
| [NAIL-RAPM v1.2.3 free-throw profile (not promoted)](nail-rapm-v123-free-throw-profile.md) | 9 | 1.192714 (9) | 1.137583 (4) | 0.0638% (9) | 16.6157 (15) | 7.5004% (14) |
| [NAIL-RAPM normalized context penalty](nail-context-regularization.md) | 11 | 1.192726 (11) | 1.137680 (20) | 0.0618% (11) | 16.5979 (8) | 7.6979% (8) |
| [NAIL-RAPM fixed context alpha=5,000](nail-context-regularization.md) | 12 | 1.192727 (12) | 1.137753 (27) | 0.0617% (12) | 16.6090 (13) | 7.5741% (12) |
| [NAIL-RAPM v1.2 gap-returner priors](nail-rapm-v12-gap-returners.md) | 13 | 1.192734 (13) | 1.137641 (14) | 0.0605% (13) | 16.6148 (14) | 7.5097% (13) |
| [NAIL-RAPM fixed context alpha=20,000](nail-context-regularization.md) | 15 | 1.192736 (14) | 1.137655 (17) | 0.0602% (14) | 16.6187 (16) | 7.4664% (15) |
| [NAIL-RAPM fixed context alpha=1,000](nail-context-regularization.md) | 16 | 1.192745 (16) | 1.137743 (26) | 0.0587% (16) | 16.6197 (17) | 7.4553% (16) |
| [NAIL-RAPM v1.1 stat-specific padding](nail-rapm-v11-profile-padding.md) | 17 | 1.192740 (15) | 1.137655 (17) | 0.0595% (15) | 16.6267 (18) | 7.3771% (17) |
| [NAIL token-MLP residual](nail-token-residual.md) | 17 | 1.192747 (17) | 1.137607 (7) | 0.0583% (17) | 16.6393 (21) | 7.2373% (20) |
| [NAIL-RAPM v1.0](nail-rapm-v1.md) | 18 | 1.192752 (18) | 1.137640 (13) | 0.0574% (18) | 16.6636 (23) | 6.9658% (22) |
| HPM v2 shooting composition | 19 | 1.192759 (19) | 1.137652 (15) | 0.0564% (19) | 16.6913 (27) | 6.6559% (26) |
| [HPM x2 raw OREB/100 context](hpm-x2.md) | 20 | 1.192760 (20) | 1.137716 (24) | 0.0562% (20) | 16.6382 (20) | 7.2857% (19) |
| Complete player-prior RAPM, no context or box score | 21 | 1.192761 (21) | 1.137741 (25) | 0.0559% (21) | 16.6068 (12) | 7.5991% (11) |
| [HPM x1 ORB claim context](hpm-x1.md) | 22 | 1.192763 (22) | 1.137713 (23) | 0.0557% (22) | 16.6315 (19) | 7.3242% (18) |
| [HPM v2.3 shot portfolio](hpm-v23.md) | 22 | 1.192777 (25) | 1.137641 (14) | 0.0534% (25) | 16.6546 (22) | 7.0659% (21) |
| [NAIL Set Attention residual](nail-token-residual.md) | 23 | 1.192768 (23) | 1.137571 (3) | 0.0547% (23) | 16.7235 (30) | 6.2956% (29) |
| HPM v2.1 empirical rebound capacity | 24 | 1.192772 (24) | 1.137654 (16) | 0.0542% (24) | 16.6701 (24) | 6.8932% (23) |
| [HPM x3 ORB claim rebound replacement](hpm-x3.md) | 25 | 1.192781 (26) | 1.137688 (21) | 0.0526% (26) | 16.6792 (25) | 6.7912% (24) |
| HPM v2.2 usage allocation | 26 | 1.192807 (28) | 1.137656 (18) | 0.0482% (28) | 16.6806 (26) | 6.7759% (25) |
| Value-Conditioned Aging HPM | 27 | 1.192792 (27) | 1.137701 (22) | 0.0508% (27) | 16.7155 (28) | 6.3851% (27) |
| [Additive prior plus linear non-additive context](additive-profile-linear-shape-context.md) | 29 | 1.192814 (29) | 1.137613 (10) | 0.0470% (29) | 16.7224 (29) | 6.3076% (28) |
| Forward 3-year RAPM-prior baseline | 30 | 1.193041 (30) | 1.137974 (29) | 0.0091% (30) | 17.1808 (31) | 1.1016% (30) |
| Forward 1-year RAPM-prior baseline | 31 | 1.193123 (31) | 1.137869 (28) | -0.0047% (31) | 17.2902 (32) | -0.1627% (31) |

NAIL-RAPM v1.3.1 is the preferred, parsimonious version of the non-promoted
v1.3 branch: it passed its direct no-material-harm gate versus v1.3, but it
does not supersede v1.2.1 as the global regular-season leader. See its
[experiment record](nail-rapm-v131-pruned-additive-profiles.md).

See [Forward RAPM Memory Baselines](forward-rapm-memory-baselines.md) and
[Complete Player-Prior RAPM Baseline](complete-player-prior-baseline.md) for
model specifications, annual lambda selections, and immutable artifacts.
