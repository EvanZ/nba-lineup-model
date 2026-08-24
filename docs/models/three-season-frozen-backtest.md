---
last_updated: "2026-08-24"
---

# Three-Season Frozen Leaderboard

Every candidate forecasts 2023-24, 2024-25, and 2025-26 from information
available before each target season begins. Target-season lineup allocation is
an oracle input; target outcomes never enter the frozen forecast. Bold values
are pooled leaders. Lower is better except skill and winner accuracy.

> **Current website production model:** [NAIL-RAPM v1.2.1 pruned non-additive
> context](nail-rapm-v121-pruned-nonadditive.md). Its table row is marked
> **(Production)**. All other rows are evaluated experiments unless explicitly
> promoted and deployed.

## Regular Season

Pooled over 584,970 eligible possessions from 3,284 games. Full-game and team
metrics cover 3,511 reconstructed games. Models are ordered by the median of
their displayed metric ranks. Ties are ordered by mean rank, then game RMSE.

| Model | Median rank | Poss. RMSE | Poss. MAE | Poss. skill | Eligible game RMSE | Eligible game skill | Full-game RMSE | Winner accuracy | Team NetRtg RMSE | Pythagorean-win RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [NAIL-RAPM v1.2.1.1 standard USG% (not promoted)](nail-rapm-v1211-standard-usage.md) | 2 | **1.197951 (1)** | 1.141332 (3) | **0.1269% (1)** | 14.0264 (4) | 18.1792% (4) | **14.2516 (1)** | 68.10% (13) | **3.2658 (1)** | 7.0279 (2) |
| [NAIL-RAPM v1.2.2 defensive-rebound profile (not promoted)](nail-rapm-v122-defensive-rebound-profile.md) | 3 | 1.197952 (3) | 1.141364 (8) | 0.1268% (3) | **14.0238 (1)** | **18.2094% (1)** | 14.2529 (3) | 68.39% (6) | 3.2723 (3) | 7.0434 (4) |
| [NAIL-RAPM v1.2.1 pruned non-additive context](nail-rapm-v121-pruned-nonadditive.md) **(Production)** | 3 | 1.197952 (2) | 1.141355 (7) | 0.1268% (2) | 14.0245 (3) | 18.2005% (3) | 14.2521 (2) | 68.24% (9) | 3.2706 (2) | 7.0351 (3) |
| [NAIL Critical Spacing candidate](nail-critical-spacing.md) | 4 | 1.197954 (4) | 1.141347 (6) | 0.1264% (5) | 14.0241 (2) | 18.2063% (2) | 14.2532 (4) | 68.47% (3) | 3.2805 (5) | 7.0601 (8) |
| [NAIL quartile Critical Spacing plus standard USG% (not promoted)](nail-critical-spacing-quartile-standard-usage.md) | 5 | 1.197954 (4) | **1.141327 (1)** | 0.1265% (4) | 14.0304 (5) | 18.1317% (5) | 14.2584 (5) | 68.10% (13) | 3.2787 (4) | 7.0460 (6) |
| [NAIL-RAPM v1.2 gap-returner priors](nail-rapm-v12-gap-returners.md) | 6 | 1.197958 (5) | 1.141344 (5) | 0.1258% (6) | 14.0414 (6) | 18.0039% (6) | 14.2660 (6) | 68.16% (12) | 3.2908 (6) | 7.0899 (10) |
| [NAIL-RAPM v1.2.3 free-throw profile (not promoted)](nail-rapm-v123-free-throw-profile.md) | 7 | 1.197964 (6) | 1.141341 (4) | 0.1248% (7) | 14.0456 (7) | 17.9547% (7) | 14.2726 (7) | 68.47% (2) | 3.2974 (8) | 7.0511 (7) |
| [NAIL-RAPM v1.2.4 free-throw replacement (not promoted)](nail-rapm-v124-free-throw-replacement.md) | 8 | 1.197966 (7) | 1.141328 (2) | 0.1244% (8) | 14.0457 (8) | 17.9541% (8) | 14.2733 (8) | 68.19% (11) | 3.2933 (7) | 7.0459 (5) |
| [NAIL-RAPM v1.4 Kalman additive profiles (not promoted)](nail-rapm-v14-filtered-additive-profiles.md) | 9 | 1.197972 (8) | 1.141432 (22) | 0.1235% (9) | 14.0630 (9) | 17.7513% (9) | 14.2750 (9) | 68.39% (5) | 3.3174 (9) | **7.0088 (1)** |
| [NAIL-RAPM v1.3 additive profiles (not promoted)](nail-rapm-v13-additive-profiles.md) | 10 | 1.197987 (13) | 1.141423 (20) | 0.1211% (13) | 14.0811 (10) | 17.5398% (10) | 14.2981 (10) | 68.44% (4) | 3.3320 (10) | 7.0619 (9) |
| [NAIL-RAPM v1.1 stat-specific padding](nail-rapm-v11-profile-padding.md) | 11 | 1.197979 (10) | 1.141391 (12) | 0.1224% (11) | 14.0864 (11) | 17.4775% (11) | 14.3236 (12) | **68.53% (1)** | 3.3898 (12) | 7.2757 (12) |
| [NAIL-RAPM v1.3.1 pruned additive profile](nail-rapm-v131-pruned-additive-profiles.md) | 12 | 1.197990 (15) | 1.141406 (18) | 0.1205% (15) | 14.0891 (12) | 17.4455% (12) | 14.3072 (11) | 68.36% (7) | 3.3446 (11) | 7.1012 (11) |
| [NAIL-RAPM fixed context alpha=20,000](nail-context-regularization.md) | 13 | 1.197980 (11) | 1.141411 (19) | 0.1221% (12) | 14.0907 (13) | 17.4268% (13) | 14.3296 (13) | 68.24% (10) | 3.4095 (14) | 7.2966 (15) |
| [NAIL-RAPM fixed context alpha=5,000](nail-context-regularization.md) | 15 | 1.197994 (16) | 1.141463 (24) | 0.1199% (16) | 14.0949 (14) | 17.3778% (14) | 14.3300 (14) | 67.84% (18) | 3.4156 (15) | 7.2821 (14) |
| [NAIL-RAPM fixed context alpha=1,000](nail-context-regularization.md) | 15 | 1.197999 (18) | 1.141439 (23) | 0.1190% (18) | 14.1017 (15) | 17.2975% (15) | 14.3389 (15) | 67.99% (15) | 3.4009 (13) | 7.2776 (13) |
| [NAIL-RAPM v1.0](nail-rapm-v1.md) | 16 | 1.197977 (9) | 1.141387 (11) | 0.1226% (10) | 14.1119 (16) | 17.1779% (16) | 14.3516 (16) | 68.33% (8) | 3.4307 (18) | 7.3460 (17) |
| [NAIL token-MLP residual](nail-token-residual.md) | 17 | 1.197986 (12) | 1.141411 (19) | 0.1211% (13) | 14.1207 (17) | 17.0756% (17) | 14.3698 (17) | 67.99% (15) | 3.4669 (21) | 7.3841 (19) |
| [NAIL-RAPM additive-only context](nail-additive-only-context.md) | 18 | 1.197989 (14) | 1.141430 (21) | 0.1206% (14) | 14.1286 (18) | 16.9824% (18) | 14.3754 (18) | 67.93% (16) | 3.4783 (22) | 7.4728 (23) |
| [HPM x3 ORB claim rebound replacement](hpm-x3.md) | 19 | 1.198000 (19) | 1.141400 (14) | 0.1188% (19) | 14.1309 (19) | 16.9555% (19) | 14.3774 (19) | 67.73% (21) | 3.4229 (16) | 7.3269 (16) |
| Value-Conditioned Aging HPM | 20 | 1.198010 (22) | 1.141402 (15) | 0.1171% (22) | 14.1342 (23) | 16.9169% (23) | 14.3802 (20) | 67.84% (18) | 3.4290 (17) | 7.3532 (18) |
| [Compiled-additive HPM x3 plus quadratic side context](linear-hpm-x3-quadratic-side-context.md) | 20 | 1.198002 (20) | 1.141366 (9) | 0.1185% (20) | 14.1311 (20) | 16.9525% (20) | 14.3922 (24) | 67.36% (26) | 3.5139 (26) | 7.5889 (27) |
| [NAIL-RAPM normalized context penalty](nail-context-regularization.md) | 21 | 1.198002 (20) | 1.141533 (25) | 0.1185% (20) | 14.1339 (21) | 16.9203% (21) | 14.3822 (21) | 68.01% (14) | 3.5234 (27) | 7.4597 (22) |
| HPM v2.1 empirical rebound capacity | 22 | 1.198010 (22) | 1.141403 (16) | 0.1171% (22) | 14.1340 (22) | 16.9193% (22) | 14.3869 (22) | 67.67% (23) | 3.4365 (19) | 7.3881 (20) |
| HPM v2 shooting composition | 23 | 1.198022 (23) | 1.141404 (17) | 0.1151% (23) | 14.1409 (24) | 16.8378% (24) | 14.3908 (23) | 67.70% (22) | 3.4592 (20) | 7.4282 (21) |
| HPM v2.2 usage allocation | 24 | 1.198004 (21) | 1.141380 (10) | 0.1182% (21) | 14.1464 (26) | 16.7730% (26) | 14.4050 (25) | 67.47% (25) | 3.4802 (23) | 7.4793 (24) |
| [HPM v2.3 shot portfolio](hpm-v23.md) | 24 | 1.198025 (24) | 1.141396 (13) | 0.1147% (24) | 14.1815 (27) | 16.3594% (27) | 14.4288 (27) | 67.70% (22) | 3.5037 (24) | 7.5274 (25) |
| [NAIL Set Attention residual](nail-token-residual.md) | 25 | 1.197997 (17) | 1.141396 (13) | 0.1192% (17) | 14.1458 (25) | 16.7796% (25) | 14.4098 (26) | 67.64% (24) | 3.5075 (25) | 7.5605 (26) |
| [Additive prior plus linear non-additive context](additive-profile-linear-shape-context.md) | 25 | 1.198026 (25) | 1.141366 (9) | 0.1144% (25) | 14.2154 (31) | 15.9592% (31) | 14.4542 (28) | 67.76% (20) | 3.5492 (28) | 7.4793 (24) |
| [HPM x2 raw OREB/100 context](hpm-x2.md) | 28 | 1.198048 (27) | 1.141660 (27) | 0.1108% (27) | 14.2073 (28) | 16.0552% (28) | 14.4641 (29) | 67.79% (19) | 3.6558 (29) | 7.7389 (28) |
| [HPM x1 ORB claim context](hpm-x1.md) | 29 | 1.198047 (26) | 1.141658 (26) | 0.1110% (26) | 14.2081 (29) | 16.0449% (29) | 14.4665 (30) | 67.87% (17) | 3.6653 (30) | 7.7498 (29) |
| Complete player-prior RAPM, no context or box score | 30 | 1.198061 (28) | 1.141675 (28) | 0.1086% (28) | 14.2105 (30) | 16.0175% (30) | 14.4756 (31) | 67.64% (24) | 3.6754 (31) | 7.7561 (30) |
| Forward 1-year RAPM-prior baseline | 31 | 1.198196 (29) | 1.141757 (29) | 0.0861% (29) | 14.5265 (32) | 12.2406% (32) | 14.8154 (32) | 65.39% (27) | 4.2374 (32) | 9.1066 (31) |
| Forward 3-year RAPM-prior baseline | 32 | 1.198246 (30) | 1.141950 (30) | 0.0778% (30) | 14.6559 (33) | 10.6699% (33) | 14.9714 (33) | 64.68% (28) | 4.5326 (33) | 9.7789 (32) |

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
| [NAIL-RAPM v1.3 additive profiles (not promoted)](nail-rapm-v13-additive-profiles.md) | **1** | **1.192630 (1)** | 1.137623 (13) | **0.0780% (1)** | **16.5034 (1)** | **8.7464% (1)** |
| [NAIL-RAPM v1.3.1 pruned additive profile](nail-rapm-v131-pruned-additive-profiles.md) | 2 | 1.192646 (2) | 1.137625 (14) | 0.0753% (2) | 16.5248 (2) | 8.5093% (2) |
| [NAIL-RAPM v1.4 Kalman additive profiles (not promoted)](nail-rapm-v14-filtered-additive-profiles.md) | 4 | 1.192674 (3) | 1.137670 (21) | 0.0706% (3) | 16.5836 (4) | 7.8566% (4) |
| [NAIL-RAPM v1.2.2 defensive-rebound profile (not promoted)](nail-rapm-v122-defensive-rebound-profile.md) | 5 | 1.192702 (4) | 1.137609 (10) | 0.0659% (4) | 16.5875 (5) | 7.8132% (5) |
| [NAIL-RAPM additive-only context](nail-additive-only-context.md) | 5 | 1.192713 (9) | 1.137590 (5) | 0.0640% (9) | 16.5724 (3) | 7.9812% (3) |
| [NAIL-RAPM v1.2.4 free-throw replacement (not promoted)](nail-rapm-v124-free-throw-replacement.md) | 5 | 1.192706 (5) | 1.137567 (2) | 0.0652% (5) | 16.5988 (10) | 7.6880% (10) |
| [NAIL Critical Spacing candidate](nail-critical-spacing.md) | 6 | 1.192706 (6) | 1.137609 (11) | 0.0652% (6) | 16.5896 (6) | 7.7902% (6) |
| [NAIL-RAPM v1.2.1.1 standard USG% (not promoted)](nail-rapm-v1211-standard-usage.md) | 7 | 1.192708 (7) | 1.137593 (6) | 0.0649% (7) | 16.5989 (11) | 7.6873% (11) |
| [NAIL-RAPM v1.2.1 pruned non-additive context](nail-rapm-v121-pruned-nonadditive.md) **(Production)** | 8 | 1.192709 (8) | 1.137608 (9) | 0.0647% (8) | 16.5942 (7) | 7.7392% (7) |
| [Compiled-additive HPM x3 plus quadratic side context](linear-hpm-x3-quadratic-side-context.md) | 8 | 1.192719 (11) | **1.137554 (1)** | 0.0630% (11) | 16.5978 (8) | 7.6986% (8) |
| [NAIL quartile Critical Spacing plus standard USG% (not promoted)](nail-critical-spacing-quartile-standard-usage.md) | 9 | 1.192713 (9) | 1.137596 (7) | 0.0640% (9) | 16.6145 (14) | 7.5130% (14) |
| [NAIL-RAPM v1.2.3 free-throw profile (not promoted)](nail-rapm-v123-free-throw-profile.md) | 10 | 1.192714 (10) | 1.137583 (4) | 0.0638% (10) | 16.6157 (16) | 7.5004% (16) |
| [NAIL-RAPM normalized context penalty](nail-context-regularization.md) | 12 | 1.192726 (12) | 1.137680 (22) | 0.0618% (12) | 16.5979 (9) | 7.6979% (9) |
| [NAIL-RAPM fixed context alpha=5,000](nail-context-regularization.md) | 13 | 1.192727 (13) | 1.137753 (29) | 0.0617% (13) | 16.6090 (13) | 7.5741% (13) |
| [NAIL-RAPM v1.2 gap-returner priors](nail-rapm-v12-gap-returners.md) | 15 | 1.192734 (14) | 1.137641 (16) | 0.0605% (14) | 16.6148 (15) | 7.5097% (15) |
| [NAIL-RAPM fixed context alpha=20,000](nail-context-regularization.md) | 17 | 1.192736 (15) | 1.137655 (19) | 0.0602% (15) | 16.6187 (17) | 7.4664% (17) |
| [NAIL token-MLP residual](nail-token-residual.md) | 18 | 1.192747 (18) | 1.137607 (8) | 0.0583% (18) | 16.6393 (22) | 7.2373% (22) |
| [NAIL-RAPM fixed context alpha=1,000](nail-context-regularization.md) | 18 | 1.192745 (17) | 1.137743 (28) | 0.0587% (17) | 16.6197 (18) | 7.4553% (18) |
| [NAIL-RAPM v1.1 stat-specific padding](nail-rapm-v11-profile-padding.md) | 19 | 1.192740 (16) | 1.137655 (19) | 0.0595% (16) | 16.6267 (19) | 7.3771% (19) |
| [NAIL-RAPM v1.0](nail-rapm-v1.md) | 19 | 1.192752 (19) | 1.137640 (15) | 0.0574% (19) | 16.6636 (24) | 6.9658% (24) |
| HPM v2 shooting composition | 20 | 1.192759 (20) | 1.137652 (17) | 0.0564% (20) | 16.6913 (28) | 6.6559% (28) |
| [HPM x2 raw OREB/100 context](hpm-x2.md) | 21 | 1.192760 (21) | 1.137716 (26) | 0.0562% (21) | 16.6382 (21) | 7.2857% (21) |
| Complete player-prior RAPM, no context or box score | 22 | 1.192761 (22) | 1.137741 (27) | 0.0559% (22) | 16.6068 (12) | 7.5991% (12) |
| [HPM x1 ORB claim context](hpm-x1.md) | 23 | 1.192763 (23) | 1.137713 (25) | 0.0557% (23) | 16.6315 (20) | 7.3242% (20) |
| [HPM v2.3 shot portfolio](hpm-v23.md) | 23 | 1.192777 (26) | 1.137641 (16) | 0.0534% (26) | 16.6546 (23) | 7.0659% (23) |
| [NAIL Set Attention residual](nail-token-residual.md) | 24 | 1.192768 (24) | 1.137571 (3) | 0.0547% (24) | 16.7235 (31) | 6.2956% (31) |
| HPM v2.1 empirical rebound capacity | 25 | 1.192772 (25) | 1.137654 (18) | 0.0542% (25) | 16.6701 (25) | 6.8932% (25) |
| [HPM x3 ORB claim rebound replacement](hpm-x3.md) | 26 | 1.192781 (27) | 1.137688 (23) | 0.0526% (27) | 16.6792 (26) | 6.7912% (26) |
| HPM v2.2 usage allocation | 27 | 1.192807 (29) | 1.137656 (20) | 0.0482% (29) | 16.6806 (27) | 6.7759% (27) |
| Value-Conditioned Aging HPM | 28 | 1.192792 (28) | 1.137701 (24) | 0.0508% (28) | 16.7155 (29) | 6.3851% (29) |
| [Additive prior plus linear non-additive context](additive-profile-linear-shape-context.md) | 30 | 1.192814 (30) | 1.137613 (12) | 0.0470% (30) | 16.7224 (30) | 6.3076% (30) |
| Forward 3-year RAPM-prior baseline | 31 | 1.193041 (31) | 1.137974 (31) | 0.0091% (31) | 17.1808 (32) | 1.1016% (32) |
| Forward 1-year RAPM-prior baseline | 32 | 1.193123 (32) | 1.137869 (30) | -0.0047% (32) | 17.2902 (33) | -0.1627% (33) |

NAIL-RAPM v1.3.1 is the preferred, parsimonious version of the non-promoted
v1.3 branch: it passed its direct no-material-harm gate versus v1.3, but it
does not supersede v1.2.1 as the global regular-season leader. See its
[experiment record](nail-rapm-v131-pruned-additive-profiles.md).

See [Forward RAPM Memory Baselines](forward-rapm-memory-baselines.md) and
[Complete Player-Prior RAPM Baseline](complete-player-prior-baseline.md) for
model specifications, annual lambda selections, and immutable artifacts.
