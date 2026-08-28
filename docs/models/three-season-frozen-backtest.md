---
last_updated: "2026-08-27"
---

# Three-Season Frozen Leaderboard

Every candidate forecasts 2023-24, 2024-25, and 2025-26 from information
available before each target season begins. Target-season lineup allocation is
an oracle input; target outcomes never enter the frozen forecast. Bold values
are pooled leaders. Lower is better except skill and winner accuracy.

> **Current production model:** [NAIL-RAPM v1.2.1.3 residualized-target lambda
> CV](nail-rapm-v1213-residualized-lambda.md). It selected each source-season
> player penalty directly on the source residualized target and won the agreed
> frozen comparison. All table rows are retained regardless of promotion status.

## Regular Season

Pooled over 584,970 eligible possessions from 3,284 games. Full-game and team
metrics cover 3,511 reconstructed games. Models are ordered by the median of
their displayed metric ranks. Ties are ordered by mean rank, then game RMSE.
Every row is retained regardless of promotion status.

| Model | Median rank | Poss. RMSE | Poss. MAE | Poss. skill | Eligible game RMSE | Eligible game skill | Full-game RMSE | Winner accuracy | Team NetRtg RMSE | Pythagorean-win RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [NAIL-RAPM v1.2.1.3 residualized-target lambda CV](nail-rapm-v1213-residualized-lambda.md) **(Production)** | 2 | **1.197946 (1)** | **1.141306 (1)** | 0.1278% (2) | 13.9939 (2) | 18.5574% (2) | **14.2166 (1)** | 68.30% (9) | **3.2351 (1)** | 6.9423 (2) |
| [NAIL-RAPM v1.2.1.2 back-to-back schedule control](nail-rapm-v1212-back-to-back.md) **(Prior production)** | 3 | **1.197946 (1)** | 1.141313 (2) | **0.1279% (1)** | 14.0107 (3) | 18.3623% (3) | 14.2330 (2) | 67.96% (17) | 3.2847 (7) | 7.0551 (10) |
| [NAIL-RAPM v1.2.1.1 standard USG% (not promoted)](nail-rapm-v1211-standard-usage.md) | 4 | 1.197951 (2) | 1.141332 (5) | 0.1269% (3) | 14.0264 (8) | 18.1792% (8) | 14.2516 (3) | 68.10% (14) | 3.2658 (2) | 7.0279 (4) |
| [NAIL-RAPM v1.2.2 defensive-rebound profile (not promoted)](nail-rapm-v122-defensive-rebound-profile.md) | 5 | 1.197952 (4) | 1.141364 (10) | 0.1268% (5) | 14.0238 (4) | 18.2094% (4) | 14.2529 (5) | 68.39% (6) | 3.2723 (4) | 7.0434 (6) |
| [NAIL-RAPM v1.2.1 pruned non-additive context](nail-rapm-v121-pruned-nonadditive.md) | 5 | 1.197952 (3) | 1.141355 (9) | 0.1268% (4) | 14.0245 (6) | 18.2005% (6) | 14.2521 (4) | 68.24% (10) | 3.2706 (3) | 7.0351 (5) |
| [NAIL Critical Spacing candidate](nail-critical-spacing.md) | 6 | 1.197954 (5) | 1.141347 (8) | 0.1264% (7) | 14.0241 (5) | 18.2063% (5) | 14.2532 (6) | 68.47% (3) | 3.2805 (6) | 7.0601 (11) |
| [NAIL quartile Critical Spacing plus standard USG% (not promoted)](nail-critical-spacing-quartile-standard-usage.md) | 7 | 1.197954 (5) | 1.141327 (3) | 0.1265% (6) | 14.0304 (9) | 18.1317% (9) | 14.2584 (7) | 68.10% (14) | 3.2787 (5) | 7.0460 (8) |
| [NAIL-RAPM v1.2 gap-returner priors](nail-rapm-v12-gap-returners.md) | 8 | 1.197958 (6) | 1.141344 (7) | 0.1258% (8) | 14.0414 (10) | 18.0039% (10) | 14.2660 (8) | 68.16% (13) | 3.2908 (8) | 7.0899 (13) |
| [NAIL-RAPM v1.2.3 free-throw profile (not promoted)](nail-rapm-v123-free-throw-profile.md) | 9 | 1.197964 (7) | 1.141341 (6) | 0.1248% (9) | 14.0456 (11) | 17.9547% (11) | 14.2726 (10) | 68.47% (2) | 3.2974 (10) | 7.0511 (9) |
| [NAIL-RAPM v1.2.4 free-throw replacement (not promoted)](nail-rapm-v124-free-throw-replacement.md) | 10 | 1.197966 (8) | 1.141328 (4) | 0.1244% (10) | 14.0457 (12) | 17.9541% (12) | 14.2733 (11) | 68.19% (12) | 3.2933 (9) | 7.0459 (7) |
| [Split NAIL-RAPM constrained O/D decomposition (not promoted)](split-nail-rapm.md) | 11 | 1.198026 (28) | 1.142568 (34) | 0.1144% (28) | **13.9679 (1)** | **18.8598% (1)** | 14.2668 (9) | 66.56% (30) | 3.3020 (11) | **6.9130 (1)** |
| [NAIL-RAPM v1.4 Kalman additive profiles (not promoted)](nail-rapm-v14-filtered-additive-profiles.md) | 12 | 1.197972 (10) | 1.141432 (24) | 0.1235% (12) | 14.0630 (13) | 17.7513% (13) | 14.2750 (12) | 68.39% (5) | 3.3174 (12) | 7.0088 (3) |
| [State-Precision NAIL (posterior uncertainty, no forgetting; not promoted)](state-precision-no-forgetting.md) | 13 | 1.197970 (9) | 1.141453 (26) | 0.1239% (11) | 14.0255 (7) | 18.1896% (7) | 14.2783 (13) | 67.84% (20) | 3.4051 (17) | 7.1644 (15) |
| [NAIL-RAPM v1.3 additive profiles (not promoted)](nail-rapm-v13-additive-profiles.md) | 14 | 1.197987 (15) | 1.141423 (22) | 0.1211% (16) | 14.0811 (14) | 17.5398% (14) | 14.2981 (14) | 68.44% (4) | 3.3320 (13) | 7.0619 (12) |
| [NAIL-RAPM v1.1 stat-specific padding](nail-rapm-v11-profile-padding.md) | 15 | 1.197979 (12) | 1.141391 (14) | 0.1224% (14) | 14.0864 (15) | 17.4775% (15) | 14.3236 (16) | **68.53% (1)** | 3.3898 (15) | 7.2757 (16) |
| [NAIL-RAPM v1.3.1 pruned additive profile](nail-rapm-v131-pruned-additive-profiles.md) | 16 | 1.197990 (17) | 1.141406 (20) | 0.1205% (18) | 14.0891 (16) | 17.4455% (16) | 14.3072 (15) | 68.36% (7) | 3.3446 (14) | 7.1012 (14) |
| [NAIL-RAPM fixed context alpha=20,000](nail-context-regularization.md) | 17 | 1.197980 (13) | 1.141411 (21) | 0.1221% (15) | 14.0907 (17) | 17.4268% (17) | 14.3296 (17) | 68.24% (11) | 3.4095 (18) | 7.2966 (19) |
| [NAIL-RAPM fixed context alpha=5,000](nail-context-regularization.md) | 18 | 1.197994 (18) | 1.141463 (27) | 0.1199% (19) | 14.0949 (18) | 17.3778% (18) | 14.3300 (18) | 67.84% (21) | 3.4156 (19) | 7.2821 (18) |
| [NAIL-RAPM fixed context alpha=1,000](nail-context-regularization.md) | 19 | 1.197999 (20) | 1.141439 (25) | 0.1190% (21) | 14.1017 (19) | 17.2975% (19) | 14.3389 (19) | 67.99% (16) | 3.4009 (16) | 7.2776 (17) |
| [NAIL-RAPM v1.0](nail-rapm-v1.md) | 20 | 1.197977 (11) | 1.141387 (13) | 0.1226% (13) | 14.1119 (20) | 17.1779% (20) | 14.3516 (20) | 68.33% (8) | 3.4307 (22) | 7.3460 (21) |
| [NAIL token-MLP residual](nail-token-residual.md) | 21 | 1.197986 (14) | 1.141411 (21) | 0.1211% (16) | 14.1207 (21) | 17.0756% (21) | 14.3698 (21) | 67.99% (16) | 3.4669 (25) | 7.3841 (23) |
| [HPM x3 ORB claim rebound replacement](hpm-x3.md) | 22 | 1.198000 (21) | 1.141400 (16) | 0.1188% (22) | 14.1309 (23) | 16.9555% (23) | 14.3774 (23) | 67.73% (24) | 3.4229 (20) | 7.3269 (20) |
| [NAIL-RAPM additive-only context](nail-additive-only-context.md) | 22 | 1.197989 (16) | 1.141430 (23) | 0.1206% (17) | 14.1286 (22) | 16.9824% (22) | 14.3754 (22) | 67.93% (18) | 3.4783 (26) | 7.4728 (27) |
| Value-Conditioned Aging HPM | 24 | 1.198010 (24) | 1.141402 (17) | 0.1171% (25) | 14.1342 (27) | 16.9169% (27) | 14.3802 (24) | 67.84% (21) | 3.4290 (21) | 7.3532 (22) |
| [Compiled-additive HPM x3 plus quadratic side context](linear-hpm-x3-quadratic-side-context.md) | 24 | 1.198002 (22) | 1.141366 (11) | 0.1185% (23) | 14.1311 (24) | 16.9525% (24) | 14.3922 (28) | 67.36% (29) | 3.5139 (30) | 7.5889 (31) |
| HPM v2.1 empirical rebound capacity | 25 | 1.198010 (24) | 1.141403 (18) | 0.1171% (25) | 14.1340 (26) | 16.9193% (26) | 14.3869 (26) | 67.67% (26) | 3.4365 (23) | 7.3881 (24) |
| [NAIL-RAPM normalized context penalty](nail-context-regularization.md) | 25 | 1.198002 (22) | 1.141533 (28) | 0.1185% (23) | 14.1339 (25) | 16.9203% (25) | 14.3822 (25) | 68.01% (15) | 3.5234 (31) | 7.4597 (26) |
| HPM v2 shooting composition | 25 | 1.198022 (25) | 1.141404 (19) | 0.1151% (26) | 14.1409 (28) | 16.8378% (28) | 14.3908 (27) | 67.70% (25) | 3.4592 (24) | 7.4282 (25) |
| HPM v2.2 usage allocation | 28 | 1.198004 (23) | 1.141380 (12) | 0.1182% (24) | 14.1464 (30) | 16.7730% (30) | 14.4050 (29) | 67.47% (28) | 3.4802 (27) | 7.4793 (28) |
| [HPM v2.3 shot portfolio](hpm-v23.md) | 28 | 1.198025 (26) | 1.141396 (15) | 0.1147% (27) | 14.1815 (31) | 16.3594% (31) | 14.4288 (31) | 67.70% (25) | 3.5037 (28) | 7.5274 (29) |
| [NAIL Set Attention residual](nail-token-residual.md) | 29 | 1.197997 (19) | 1.141396 (15) | 0.1192% (20) | 14.1458 (29) | 16.7796% (29) | 14.4098 (30) | 67.64% (27) | 3.5075 (29) | 7.5605 (30) |
| [Additive prior plus linear non-additive context](additive-profile-linear-shape-context.md) | 29 | 1.198026 (27) | 1.141366 (11) | 0.1144% (29) | 14.2154 (35) | 15.9592% (35) | 14.4542 (32) | 67.76% (23) | 3.5492 (32) | 7.4793 (28) |
| [HPM x2 raw OREB/100 context](hpm-x2.md) | 32 | 1.198048 (30) | 1.141660 (30) | 0.1108% (31) | 14.2073 (32) | 16.0552% (32) | 14.4641 (33) | 67.79% (22) | 3.6558 (33) | 7.7389 (32) |
| [HPM x1 ORB claim context](hpm-x1.md) | 33 | 1.198047 (29) | 1.141658 (29) | 0.1110% (30) | 14.2081 (33) | 16.0449% (33) | 14.4665 (34) | 67.87% (19) | 3.6653 (34) | 7.7498 (33) |
| Complete player-prior RAPM, no context or box score | 34 | 1.198061 (31) | 1.141675 (31) | 0.1086% (32) | 14.2105 (34) | 16.0175% (34) | 14.4756 (35) | 67.64% (27) | 3.6754 (35) | 7.7561 (34) |
| Forward 1-year RAPM-prior baseline | 35 | 1.198196 (32) | 1.141757 (32) | 0.0861% (33) | 14.5265 (36) | 12.2406% (36) | 14.8154 (36) | 65.39% (31) | 4.2374 (36) | 9.1066 (35) |
| Forward 3-year RAPM-prior baseline | 36 | 1.198246 (33) | 1.141950 (33) | 0.0778% (34) | 14.6559 (37) | 10.6699% (37) | 14.9714 (37) | 64.68% (32) | 4.5326 (37) | 9.7789 (36) |

NAIL-RAPM v1.3.1 is the preferred, parsimonious version of the non-promoted
v1.3 branch: it passed its direct no-material-harm gate versus v1.3, but it
does not supersede the selected v1.2.1.3 model as the global regular-season
release. See its
[experiment record](nail-rapm-v131-pruned-additive-profiles.md).

## Playoffs

Pooled over 39,967 eligible possessions from 238 games. Each playoff cohort
uses its matching frozen pre-season player prior and prior-year context state;
postseason outcomes are evaluation-only. Models use the same median-rank
ordering and tie-breakers as the regular-season table.

| Model | Median rank | Poss. RMSE | Poss. MAE | Poss. skill | Eligible game RMSE | Eligible game skill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [NAIL-RAPM v1.3 additive profiles (not promoted)](nail-rapm-v13-additive-profiles.md) | **1** | **1.192630 (1)** | 1.137623 (14) | **0.0780% (1)** | **16.5034 (1)** | **8.7464% (1)** |
| [State-Precision NAIL (posterior uncertainty, no forgetting; not promoted)](state-precision-no-forgetting.md) | 2 | 1.192641 (2) | 1.137743 (30) | 0.0760% (2) | 16.5191 (2) | 8.5725% (2) |
| [NAIL-RAPM v1.3.1 pruned additive profile](nail-rapm-v131-pruned-additive-profiles.md) | 3 | 1.192646 (3) | 1.137625 (16) | 0.0753% (3) | 16.5248 (3) | 8.5093% (3) |
| [NAIL-RAPM additive-only context](nail-additive-only-context.md) | 5 | 1.192713 (11) | 1.137590 (5) | 0.0640% (11) | 16.5724 (5) | 7.9812% (5) |
| [NAIL-RAPM v1.2.4 free-throw replacement (not promoted)](nail-rapm-v124-free-throw-replacement.md) | 6 | 1.192706 (6) | 1.137567 (2) | 0.0652% (6) | 16.5988 (13) | 7.6880% (13) |
| [NAIL-RAPM v1.4 Kalman additive profiles (not promoted)](nail-rapm-v14-filtered-additive-profiles.md) | 7 | 1.192674 (4) | 1.137670 (23) | 0.0706% (4) | 16.5836 (7) | 7.8566% (7) |
| [NAIL-RAPM v1.2.2 defensive-rebound profile (not promoted)](nail-rapm-v122-defensive-rebound-profile.md) | 8 | 1.192702 (5) | 1.137609 (11) | 0.0659% (5) | 16.5875 (8) | 7.8132% (8) |
| [NAIL-RAPM v1.2.1.1 standard USG% (not promoted)](nail-rapm-v1211-standard-usage.md) | 8 | 1.192708 (8) | 1.137593 (6) | 0.0649% (8) | 16.5989 (14) | 7.6873% (14) |
| [NAIL Critical Spacing candidate](nail-critical-spacing.md) | 9 | 1.192706 (7) | 1.137609 (12) | 0.0652% (7) | 16.5896 (9) | 7.7902% (9) |
| [NAIL-RAPM v1.2.1 pruned non-additive context](nail-rapm-v121-pruned-nonadditive.md) | 10 | 1.192709 (9) | 1.137608 (10) | 0.0647% (9) | 16.5942 (10) | 7.7392% (10) |
| [NAIL-RAPM v1.2.1.2 back-to-back schedule control](nail-rapm-v1212-back-to-back.md) **(Prior production)** | 10 | 1.192710 (10) | 1.137604 (8) | 0.0645% (10) | 16.6032 (15) | 7.6393% (15) |
| [Compiled-additive HPM x3 plus quadratic side context](linear-hpm-x3-quadratic-side-context.md) | 11 | 1.192719 (14) | **1.137554 (1)** | 0.0630% (14) | 16.5978 (11) | 7.6986% (11) |
| [NAIL quartile Critical Spacing plus standard USG% (not promoted)](nail-critical-spacing-quartile-standard-usage.md) | 11 | 1.192713 (11) | 1.137596 (7) | 0.0640% (11) | 16.6145 (18) | 7.5130% (18) |
| [NAIL-RAPM v1.2.3 free-throw profile (not promoted)](nail-rapm-v123-free-throw-profile.md) | 12 | 1.192714 (12) | 1.137583 (4) | 0.0638% (12) | 16.6157 (20) | 7.5004% (20) |
| [NAIL-RAPM v1.2.1.3 residualized-target lambda CV](nail-rapm-v1213-residualized-lambda.md) **(Production)** | 13 | 1.192717 (13) | 1.137624 (15) | 0.0633% (13) | 16.5784 (6) | 7.9150% (6) |
| [NAIL-RAPM normalized context penalty](nail-context-regularization.md) | 15 | 1.192726 (15) | 1.137680 (24) | 0.0618% (15) | 16.5979 (12) | 7.6979% (12) |
| [NAIL-RAPM fixed context alpha=5,000](nail-context-regularization.md) | 17 | 1.192727 (16) | 1.137753 (31) | 0.0617% (16) | 16.6090 (17) | 7.5741% (17) |
| [NAIL-RAPM v1.2 gap-returner priors](nail-rapm-v12-gap-returners.md) | 18 | 1.192734 (17) | 1.137641 (18) | 0.0605% (17) | 16.6148 (19) | 7.5097% (19) |
| [NAIL-RAPM fixed context alpha=20,000](nail-context-regularization.md) | 21 | 1.192736 (18) | 1.137655 (21) | 0.0602% (18) | 16.6187 (21) | 7.4664% (21) |
| [NAIL token-MLP residual](nail-token-residual.md) | 21 | 1.192747 (21) | 1.137607 (9) | 0.0583% (21) | 16.6393 (26) | 7.2373% (26) |
| [NAIL-RAPM v1.1 stat-specific padding](nail-rapm-v11-profile-padding.md) | 21 | 1.192740 (19) | 1.137655 (21) | 0.0595% (19) | 16.6267 (23) | 7.3771% (23) |
| [NAIL-RAPM fixed context alpha=1,000](nail-context-regularization.md) | 22 | 1.192745 (20) | 1.137743 (30) | 0.0587% (20) | 16.6197 (22) | 7.4553% (22) |
| [NAIL-RAPM v1.0](nail-rapm-v1.md) | 22 | 1.192752 (22) | 1.137640 (17) | 0.0574% (22) | 16.6636 (28) | 6.9658% (28) |
| HPM v2 shooting composition | 23 | 1.192759 (23) | 1.137652 (19) | 0.0564% (23) | 16.6913 (32) | 6.6559% (32) |
| Complete player-prior RAPM, no context or box score | 25 | 1.192761 (25) | 1.137741 (29) | 0.0559% (25) | 16.6068 (16) | 7.5991% (16) |
| [HPM x2 raw OREB/100 context](hpm-x2.md) | 25 | 1.192760 (24) | 1.137716 (28) | 0.0562% (24) | 16.6382 (25) | 7.2857% (25) |
| [HPM x1 ORB claim context](hpm-x1.md) | 26 | 1.192763 (26) | 1.137713 (27) | 0.0557% (26) | 16.6315 (24) | 7.3242% (24) |
| [NAIL Set Attention residual](nail-token-residual.md) | 27 | 1.192768 (27) | 1.137571 (3) | 0.0547% (27) | 16.7235 (35) | 6.2956% (35) |
| [HPM v2.3 shot portfolio](hpm-v23.md) | 27 | 1.192777 (29) | 1.137641 (18) | 0.0534% (29) | 16.6546 (27) | 7.0659% (27) |
| HPM v2.1 empirical rebound capacity | 28 | 1.192772 (28) | 1.137654 (20) | 0.0542% (28) | 16.6701 (29) | 6.8932% (29) |
| [HPM x3 ORB claim rebound replacement](hpm-x3.md) | 30 | 1.192781 (30) | 1.137688 (25) | 0.0526% (30) | 16.6792 (30) | 6.7912% (30) |
| HPM v2.2 usage allocation | 31 | 1.192807 (32) | 1.137656 (22) | 0.0482% (32) | 16.6806 (31) | 6.7759% (31) |
| Value-Conditioned Aging HPM | 31 | 1.192792 (31) | 1.137701 (26) | 0.0508% (31) | 16.7155 (33) | 6.3851% (33) |
| [Additive prior plus linear non-additive context](additive-profile-linear-shape-context.md) | 33 | 1.192814 (33) | 1.137613 (13) | 0.0470% (33) | 16.7224 (34) | 6.3076% (34) |
| [Split NAIL-RAPM constrained O/D decomposition (not promoted)](split-nail-rapm.md) | 34 | 1.194415 (36) | 1.140791 (34) | -0.2214% (36) | 16.5301 (4) | 8.4506% (4) |
| Forward 3-year RAPM-prior baseline | 34 | 1.193041 (34) | 1.137974 (33) | 0.0091% (34) | 17.1808 (36) | 1.1016% (36) |
| Forward 1-year RAPM-prior baseline | 35 | 1.193123 (35) | 1.137869 (32) | -0.0047% (35) | 17.2902 (37) | -0.1627% (37) |

NAIL-RAPM v1.3.1 is the preferred, parsimonious version of the non-promoted
v1.3 branch: it passed its direct no-material-harm gate versus v1.3, but it
does not supersede the selected v1.2.1.3 model as the global regular-season
release. See its
[experiment record](nail-rapm-v131-pruned-additive-profiles.md).

See [Forward RAPM Memory Baselines](forward-rapm-memory-baselines.md) and
[Complete Player-Prior RAPM Baseline](complete-player-prior-baseline.md) for
model specifications, annual lambda selections, and immutable artifacts.
