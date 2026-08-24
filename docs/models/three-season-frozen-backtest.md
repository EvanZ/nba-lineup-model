---
last_updated: "2026-08-24"
---

# Three-Season Frozen Leaderboard

Every candidate forecasts 2023-24, 2024-25, and 2025-26 from information
available before each target season begins. Target-season lineup allocation is
an oracle input; target outcomes never enter the frozen forecast. Bold values
are pooled leaders. Lower is better except skill and winner accuracy.

> **Current website production model:** [NAIL-RAPM v1.2.1.2 back-to-back schedule
> control](nail-rapm-v1212-back-to-back.md). Its table row is marked
> **(Production)**. All other rows are evaluated experiments unless explicitly
> promoted and deployed.

## Regular Season

Pooled over 584,970 eligible possessions from 3,284 games. Full-game and team
metrics cover 3,511 reconstructed games. Models are ordered by the median of
their displayed metric ranks. Ties are ordered by mean rank, then game RMSE.

| Model | Median rank | Poss. RMSE | Poss. MAE | Poss. skill | Eligible game RMSE | Eligible game skill | Full-game RMSE | Winner accuracy | Team NetRtg RMSE | Pythagorean-win RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [NAIL-RAPM v1.2.1.2 back-to-back schedule control](nail-rapm-v1212-back-to-back.md) **(Production)** | **1** | **1.197946 (1)** | **1.141313 (1)** | **0.1279% (1)** | **14.0107 (1)** | **18.3623% (1)** | **14.2330 (1)** | 67.96% (16) | 3.2847 (6) | 7.0551 (8) |
| [NAIL-RAPM v1.2.1.1 standard USG% (not promoted)](nail-rapm-v1211-standard-usage.md) | 2 | 1.197951 (2) | 1.141332 (4) | 0.1269% (2) | 14.0264 (5) | 18.1792% (5) | 14.2516 (2) | 68.10% (13) | **3.2658 (1)** | 7.0279 (2) |
| [NAIL-RAPM v1.2.1 pruned non-additive context](nail-rapm-v121-pruned-nonadditive.md) | 3 | 1.197952 (3) | 1.141355 (8) | 0.1268% (3) | 14.0245 (4) | 18.2005% (4) | 14.2521 (3) | 68.24% (9) | 3.2706 (2) | 7.0351 (3) |
| [NAIL-RAPM v1.2.2 defensive-rebound profile (not promoted)](nail-rapm-v122-defensive-rebound-profile.md) | 4 | 1.197952 (4) | 1.141364 (9) | 0.1268% (4) | 14.0238 (2) | 18.2094% (2) | 14.2529 (4) | 68.39% (6) | 3.2723 (3) | 7.0434 (4) |
| [NAIL Critical Spacing candidate](nail-critical-spacing.md) | 5 | 1.197954 (5) | 1.141347 (7) | 0.1264% (6) | 14.0241 (3) | 18.2063% (3) | 14.2532 (5) | 68.47% (3) | 3.2805 (5) | 7.0601 (9) |
| [NAIL quartile Critical Spacing plus standard USG% (not promoted)](nail-critical-spacing-quartile-standard-usage.md) | 6 | 1.197954 (5) | 1.141327 (2) | 0.1265% (5) | 14.0304 (6) | 18.1317% (6) | 14.2584 (6) | 68.10% (13) | 3.2787 (4) | 7.0460 (6) |
| [NAIL-RAPM v1.2 gap-returner priors](nail-rapm-v12-gap-returners.md) | 7 | 1.197958 (6) | 1.141344 (6) | 0.1258% (7) | 14.0414 (7) | 18.0039% (7) | 14.2660 (7) | 68.16% (12) | 3.2908 (7) | 7.0899 (11) |
| [NAIL-RAPM v1.2.3 free-throw profile (not promoted)](nail-rapm-v123-free-throw-profile.md) | 8 | 1.197964 (7) | 1.141341 (5) | 0.1248% (8) | 14.0456 (8) | 17.9547% (8) | 14.2726 (8) | 68.47% (2) | 3.2974 (9) | 7.0511 (7) |
| [NAIL-RAPM v1.2.4 free-throw replacement (not promoted)](nail-rapm-v124-free-throw-replacement.md) | 9 | 1.197966 (8) | 1.141328 (3) | 0.1244% (9) | 14.0457 (9) | 17.9541% (9) | 14.2733 (9) | 68.19% (11) | 3.2933 (8) | 7.0459 (5) |
| [NAIL-RAPM v1.4 Kalman additive profiles (not promoted)](nail-rapm-v14-filtered-additive-profiles.md) | 10 | 1.197972 (9) | 1.141432 (23) | 0.1235% (10) | 14.0630 (10) | 17.7513% (10) | 14.2750 (10) | 68.39% (5) | 3.3174 (10) | **7.0088 (1)** |
| [NAIL-RAPM v1.3 additive profiles (not promoted)](nail-rapm-v13-additive-profiles.md) | 11 | 1.197987 (14) | 1.141423 (21) | 0.1211% (14) | 14.0811 (11) | 17.5398% (11) | 14.2981 (11) | 68.44% (4) | 3.3320 (11) | 7.0619 (10) |
| [NAIL-RAPM v1.1 stat-specific padding](nail-rapm-v11-profile-padding.md) | 12 | 1.197979 (11) | 1.141391 (13) | 0.1224% (12) | 14.0864 (12) | 17.4775% (12) | 14.3236 (13) | **68.53% (1)** | 3.3898 (13) | 7.2757 (13) |
| [NAIL-RAPM v1.3.1 pruned additive profile](nail-rapm-v131-pruned-additive-profiles.md) | 13 | 1.197990 (16) | 1.141406 (19) | 0.1205% (16) | 14.0891 (13) | 17.4455% (13) | 14.3072 (12) | 68.36% (7) | 3.3446 (12) | 7.1012 (12) |
| [NAIL-RAPM fixed context alpha=20,000](nail-context-regularization.md) | 14 | 1.197980 (12) | 1.141411 (20) | 0.1221% (13) | 14.0907 (14) | 17.4268% (14) | 14.3296 (14) | 68.24% (10) | 3.4095 (15) | 7.2966 (16) |
| [NAIL-RAPM fixed context alpha=1,000](nail-context-regularization.md) | 16 | 1.197999 (19) | 1.141439 (24) | 0.1190% (19) | 14.1017 (16) | 17.2975% (16) | 14.3389 (16) | 67.99% (15) | 3.4009 (14) | 7.2776 (14) |
| [NAIL-RAPM fixed context alpha=5,000](nail-context-regularization.md) | 16 | 1.197994 (17) | 1.141463 (25) | 0.1199% (17) | 14.0949 (15) | 17.3778% (15) | 14.3300 (15) | 67.84% (19) | 3.4156 (16) | 7.2821 (15) |
| [NAIL-RAPM v1.0](nail-rapm-v1.md) | 17 | 1.197977 (10) | 1.141387 (12) | 0.1226% (11) | 14.1119 (17) | 17.1779% (17) | 14.3516 (17) | 68.33% (8) | 3.4307 (19) | 7.3460 (18) |
| [NAIL token-MLP residual](nail-token-residual.md) | 18 | 1.197986 (13) | 1.141411 (20) | 0.1211% (14) | 14.1207 (18) | 17.0756% (18) | 14.3698 (18) | 67.99% (15) | 3.4669 (22) | 7.3841 (20) |
| [NAIL-RAPM additive-only context](nail-additive-only-context.md) | 19 | 1.197989 (15) | 1.141430 (22) | 0.1206% (15) | 14.1286 (19) | 16.9824% (19) | 14.3754 (19) | 67.93% (17) | 3.4783 (23) | 7.4728 (24) |
| [HPM x3 ORB claim rebound replacement](hpm-x3.md) | 20 | 1.198000 (20) | 1.141400 (15) | 0.1188% (20) | 14.1309 (20) | 16.9555% (20) | 14.3774 (20) | 67.73% (22) | 3.4229 (17) | 7.3269 (17) |
| Value-Conditioned Aging HPM | 21 | 1.198010 (23) | 1.141402 (16) | 0.1171% (23) | 14.1342 (24) | 16.9169% (24) | 14.3802 (21) | 67.84% (19) | 3.4290 (18) | 7.3532 (19) |
| [Compiled-additive HPM x3 plus quadratic side context](linear-hpm-x3-quadratic-side-context.md) | 21 | 1.198002 (21) | 1.141366 (10) | 0.1185% (21) | 14.1311 (21) | 16.9525% (21) | 14.3922 (25) | 67.36% (27) | 3.5139 (27) | 7.5889 (28) |
| [NAIL-RAPM normalized context penalty](nail-context-regularization.md) | 22 | 1.198002 (21) | 1.141533 (26) | 0.1185% (21) | 14.1339 (22) | 16.9203% (22) | 14.3822 (22) | 68.01% (14) | 3.5234 (28) | 7.4597 (23) |
| HPM v2.1 empirical rebound capacity | 23 | 1.198010 (23) | 1.141403 (17) | 0.1171% (23) | 14.1340 (23) | 16.9193% (23) | 14.3869 (23) | 67.67% (24) | 3.4365 (20) | 7.3881 (21) |
| HPM v2 shooting composition | 24 | 1.198022 (24) | 1.141404 (18) | 0.1151% (24) | 14.1409 (25) | 16.8378% (25) | 14.3908 (24) | 67.70% (23) | 3.4592 (21) | 7.4282 (22) |
| HPM v2.2 usage allocation | 25 | 1.198004 (22) | 1.141380 (11) | 0.1182% (22) | 14.1464 (27) | 16.7730% (27) | 14.4050 (26) | 67.47% (26) | 3.4802 (24) | 7.4793 (25) |
| [HPM v2.3 shot portfolio](hpm-v23.md) | 25 | 1.198025 (25) | 1.141396 (14) | 0.1147% (25) | 14.1815 (28) | 16.3594% (28) | 14.4288 (28) | 67.70% (23) | 3.5037 (25) | 7.5274 (26) |
| [NAIL Set Attention residual](nail-token-residual.md) | 26 | 1.197997 (18) | 1.141396 (14) | 0.1192% (18) | 14.1458 (26) | 16.7796% (26) | 14.4098 (27) | 67.64% (25) | 3.5075 (26) | 7.5605 (27) |
| [Additive prior plus linear non-additive context](additive-profile-linear-shape-context.md) | 26 | 1.198026 (26) | 1.141366 (10) | 0.1144% (26) | 14.2154 (32) | 15.9592% (32) | 14.4542 (29) | 67.76% (21) | 3.5492 (29) | 7.4793 (25) |
| [HPM x2 raw OREB/100 context](hpm-x2.md) | 29 | 1.198048 (28) | 1.141660 (28) | 0.1108% (28) | 14.2073 (29) | 16.0552% (29) | 14.4641 (30) | 67.79% (20) | 3.6558 (30) | 7.7389 (29) |
| [HPM x1 ORB claim context](hpm-x1.md) | 30 | 1.198047 (27) | 1.141658 (27) | 0.1110% (27) | 14.2081 (30) | 16.0449% (30) | 14.4665 (31) | 67.87% (18) | 3.6653 (31) | 7.7498 (30) |
| Complete player-prior RAPM, no context or box score | 31 | 1.198061 (29) | 1.141675 (29) | 0.1086% (29) | 14.2105 (31) | 16.0175% (31) | 14.4756 (32) | 67.64% (25) | 3.6754 (32) | 7.7561 (31) |
| Forward 1-year RAPM-prior baseline | 32 | 1.198196 (30) | 1.141757 (30) | 0.0861% (30) | 14.5265 (33) | 12.2406% (33) | 14.8154 (33) | 65.39% (28) | 4.2374 (33) | 9.1066 (32) |
| Forward 3-year RAPM-prior baseline | 33 | 1.198246 (31) | 1.141950 (31) | 0.0778% (31) | 14.6559 (34) | 10.6699% (34) | 14.9714 (34) | 64.68% (29) | 4.5326 (34) | 9.7789 (33) |

NAIL-RAPM v1.3.1 is the preferred, parsimonious version of the non-promoted
v1.3 branch: it passed its direct no-material-harm gate versus v1.3, but it
does not supersede the production v1.2.1.2 model as the global regular-season
leader. See its
[experiment record](nail-rapm-v131-pruned-additive-profiles.md).

## Playoffs

Pooled over 39,967 eligible possessions from 238 games. Each playoff cohort
uses its matching frozen pre-season player prior and prior-year context state;
postseason outcomes are evaluation-only. Models use the same median-rank
ordering and tie-breakers as the regular-season table.

| Model | Median rank | Poss. RMSE | Poss. MAE | Poss. skill | Eligible game RMSE | Eligible game skill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [NAIL-RAPM v1.3 additive profiles (not promoted)](nail-rapm-v13-additive-profiles.md) | **1** | **1.192630 (1)** | 1.137623 (14) | **0.0780% (1)** | **16.5034 (1)** | **8.7464% (1)** |
| [NAIL-RAPM v1.3.1 pruned additive profile](nail-rapm-v131-pruned-additive-profiles.md) | 2 | 1.192646 (2) | 1.137625 (15) | 0.0753% (2) | 16.5248 (2) | 8.5093% (2) |
| [NAIL-RAPM v1.4 Kalman additive profiles (not promoted)](nail-rapm-v14-filtered-additive-profiles.md) | 4 | 1.192674 (3) | 1.137670 (22) | 0.0706% (3) | 16.5836 (4) | 7.8566% (4) |
| [NAIL-RAPM v1.2.2 defensive-rebound profile (not promoted)](nail-rapm-v122-defensive-rebound-profile.md) | 5 | 1.192702 (4) | 1.137609 (11) | 0.0659% (4) | 16.5875 (5) | 7.8132% (5) |
| [NAIL-RAPM additive-only context](nail-additive-only-context.md) | 5 | 1.192713 (10) | 1.137590 (5) | 0.0640% (10) | 16.5724 (3) | 7.9812% (3) |
| [NAIL-RAPM v1.2.4 free-throw replacement (not promoted)](nail-rapm-v124-free-throw-replacement.md) | 5 | 1.192706 (5) | 1.137567 (2) | 0.0652% (5) | 16.5988 (10) | 7.6880% (10) |
| [NAIL Critical Spacing candidate](nail-critical-spacing.md) | 6 | 1.192706 (6) | 1.137609 (12) | 0.0652% (6) | 16.5896 (6) | 7.7902% (6) |
| [NAIL-RAPM v1.2.1.1 standard USG% (not promoted)](nail-rapm-v1211-standard-usage.md) | 7 | 1.192708 (7) | 1.137593 (6) | 0.0649% (7) | 16.5989 (11) | 7.6873% (11) |
| [NAIL-RAPM v1.2.1 pruned non-additive context](nail-rapm-v121-pruned-nonadditive.md) | 8 | 1.192709 (8) | 1.137608 (10) | 0.0647% (8) | 16.5942 (7) | 7.7392% (7) |
| [Compiled-additive HPM x3 plus quadratic side context](linear-hpm-x3-quadratic-side-context.md) | 8 | 1.192719 (12) | **1.137554 (1)** | 0.0630% (12) | 16.5978 (8) | 7.6986% (8) |
| [NAIL-RAPM v1.2.1.2 back-to-back schedule control](nail-rapm-v1212-back-to-back.md) **(Production)** | 9 | 1.192710 (9) | 1.137604 (8) | 0.0645% (9) | 16.6032 (12) | 7.6393% (12) |
| [NAIL quartile Critical Spacing plus standard USG% (not promoted)](nail-critical-spacing-quartile-standard-usage.md) | 10 | 1.192713 (10) | 1.137596 (7) | 0.0640% (10) | 16.6145 (15) | 7.5130% (15) |
| [NAIL-RAPM v1.2.3 free-throw profile (not promoted)](nail-rapm-v123-free-throw-profile.md) | 11 | 1.192714 (11) | 1.137583 (4) | 0.0638% (11) | 16.6157 (17) | 7.5004% (17) |
| [NAIL-RAPM normalized context penalty](nail-context-regularization.md) | 13 | 1.192726 (13) | 1.137680 (23) | 0.0618% (13) | 16.5979 (9) | 7.6979% (9) |
| [NAIL-RAPM fixed context alpha=5,000](nail-context-regularization.md) | 14 | 1.192727 (14) | 1.137753 (30) | 0.0617% (14) | 16.6090 (14) | 7.5741% (14) |
| [NAIL-RAPM v1.2 gap-returner priors](nail-rapm-v12-gap-returners.md) | 16 | 1.192734 (15) | 1.137641 (17) | 0.0605% (15) | 16.6148 (16) | 7.5097% (16) |
| [NAIL-RAPM fixed context alpha=20,000](nail-context-regularization.md) | 18 | 1.192736 (16) | 1.137655 (20) | 0.0602% (16) | 16.6187 (18) | 7.4664% (18) |
| [NAIL token-MLP residual](nail-token-residual.md) | 19 | 1.192747 (19) | 1.137607 (9) | 0.0583% (19) | 16.6393 (23) | 7.2373% (23) |
| [NAIL-RAPM fixed context alpha=1,000](nail-context-regularization.md) | 19 | 1.192745 (18) | 1.137743 (29) | 0.0587% (18) | 16.6197 (19) | 7.4553% (19) |
| [NAIL-RAPM v1.1 stat-specific padding](nail-rapm-v11-profile-padding.md) | 20 | 1.192740 (17) | 1.137655 (20) | 0.0595% (17) | 16.6267 (20) | 7.3771% (20) |
| [NAIL-RAPM v1.0](nail-rapm-v1.md) | 20 | 1.192752 (20) | 1.137640 (16) | 0.0574% (20) | 16.6636 (25) | 6.9658% (25) |
| HPM v2 shooting composition | 21 | 1.192759 (21) | 1.137652 (18) | 0.0564% (21) | 16.6913 (29) | 6.6559% (29) |
| [HPM x2 raw OREB/100 context](hpm-x2.md) | 22 | 1.192760 (22) | 1.137716 (27) | 0.0562% (22) | 16.6382 (22) | 7.2857% (22) |
| Complete player-prior RAPM, no context or box score | 23 | 1.192761 (23) | 1.137741 (28) | 0.0559% (23) | 16.6068 (13) | 7.5991% (13) |
| [HPM x1 ORB claim context](hpm-x1.md) | 24 | 1.192763 (24) | 1.137713 (26) | 0.0557% (24) | 16.6315 (21) | 7.3242% (21) |
| [HPM v2.3 shot portfolio](hpm-v23.md) | 24 | 1.192777 (27) | 1.137641 (17) | 0.0534% (27) | 16.6546 (24) | 7.0659% (24) |
| [NAIL Set Attention residual](nail-token-residual.md) | 25 | 1.192768 (25) | 1.137571 (3) | 0.0547% (25) | 16.7235 (32) | 6.2956% (32) |
| HPM v2.1 empirical rebound capacity | 26 | 1.192772 (26) | 1.137654 (19) | 0.0542% (26) | 16.6701 (26) | 6.8932% (26) |
| [HPM x3 ORB claim rebound replacement](hpm-x3.md) | 27 | 1.192781 (28) | 1.137688 (24) | 0.0526% (28) | 16.6792 (27) | 6.7912% (27) |
| HPM v2.2 usage allocation | 28 | 1.192807 (30) | 1.137656 (21) | 0.0482% (30) | 16.6806 (28) | 6.7759% (28) |
| Value-Conditioned Aging HPM | 29 | 1.192792 (29) | 1.137701 (25) | 0.0508% (29) | 16.7155 (30) | 6.3851% (30) |
| [Additive prior plus linear non-additive context](additive-profile-linear-shape-context.md) | 31 | 1.192814 (31) | 1.137613 (13) | 0.0470% (31) | 16.7224 (31) | 6.3076% (31) |
| Forward 3-year RAPM-prior baseline | 32 | 1.193041 (32) | 1.137974 (32) | 0.0091% (32) | 17.1808 (33) | 1.1016% (33) |
| Forward 1-year RAPM-prior baseline | 33 | 1.193123 (33) | 1.137869 (31) | -0.0047% (33) | 17.2902 (34) | -0.1627% (34) |

NAIL-RAPM v1.3.1 is the preferred, parsimonious version of the non-promoted
v1.3 branch: it passed its direct no-material-harm gate versus v1.3, but it
does not supersede the production v1.2.1.2 model as the global regular-season
leader. See its
[experiment record](nail-rapm-v131-pruned-additive-profiles.md).

See [Forward RAPM Memory Baselines](forward-rapm-memory-baselines.md) and
[Complete Player-Prior RAPM Baseline](complete-player-prior-baseline.md) for
model specifications, annual lambda selections, and immutable artifacts.
