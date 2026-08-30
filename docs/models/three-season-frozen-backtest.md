---
last_updated: "2026-08-30"
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

The two teammate-continuity candidates were replayed after the latest coverage
recovery. Their possession and eligible-game entries apply paired deltas on
rows overlapping this legacy support snapshot to the production row;
full-game and team entries are direct 3,511-game results. The
[additive candidate](nail-teammate-continuity.md) and
[replacement candidate](nail-teammate-continuity-replacement.md) pages report
their direct 625,615-possession comparisons in full.

| Model | Median rank | Poss. RMSE | Poss. MAE | Poss. skill | Eligible game RMSE | Eligible game skill | Full-game RMSE | Winner accuracy | Team NetRtg RMSE | Pythagorean-win RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [NAIL-RAPM v1.2.1.3 residualized-target lambda CV](nail-rapm-v1213-residualized-lambda.md) **(Production)** | 2 | **1.197946 (1)** | **1.141306 (1)** | 0.1278% (2) | 13.9939 (3) | 18.5574% (3) | 14.2166 (3) | 68.30% (11) | **3.2351 (1)** | 6.9423 (2) |
| [NAIL teammate-continuity replacement candidate (not promoted)](nail-teammate-continuity-replacement.md) | 3 | 1.197948 (2) | 1.141388 (14) | 0.1275% (3) | 13.9919 (2) | 18.5812% (2) | **14.2119 (1)** | 68.41% (6) | 3.2853 (9) | 6.9815 (4) |
| [NAIL prior teammate-continuity candidate (not promoted)](nail-teammate-continuity.md) | 4 | 1.197955 (7) | 1.141392 (16) | 0.1263% (9) | 14.0010 (4) | 18.4755% (4) | 14.2146 (2) | 68.50% (2) | 3.2833 (7) | 6.9648 (3) |
| [NAIL-RAPM v1.2.1.2 back-to-back schedule control](nail-rapm-v1212-back-to-back.md) **(Prior production)** | 5 | **1.197946 (1)** | 1.141313 (2) | **0.1279% (1)** | 14.0107 (5) | 18.3623% (5) | 14.2330 (4) | 67.96% (19) | 3.2847 (8) | 7.0551 (12) |
| [NAIL-RAPM v1.2.1.1 standard USG% (not promoted)](nail-rapm-v1211-standard-usage.md) | 5 | 1.197951 (3) | 1.141332 (5) | 0.1269% (4) | 14.0264 (10) | 18.1792% (10) | 14.2516 (5) | 68.10% (16) | 3.2658 (2) | 7.0279 (6) |
| [NAIL-RAPM v1.2.2 defensive-rebound profile (not promoted)](nail-rapm-v122-defensive-rebound-profile.md) | 6 | 1.197952 (5) | 1.141364 (10) | 0.1268% (6) | 14.0238 (6) | 18.2094% (6) | 14.2529 (7) | 68.39% (8) | 3.2723 (4) | 7.0434 (8) |
| [NAIL-RAPM v1.2.1 pruned non-additive context](nail-rapm-v121-pruned-nonadditive.md) | 7 | 1.197952 (4) | 1.141355 (9) | 0.1268% (5) | 14.0245 (8) | 18.2005% (8) | 14.2521 (6) | 68.24% (12) | 3.2706 (3) | 7.0351 (7) |
| [NAIL Critical Spacing candidate](nail-critical-spacing.md) | 7 | 1.197954 (6) | 1.141347 (8) | 0.1264% (8) | 14.0241 (7) | 18.2063% (7) | 14.2532 (8) | 68.47% (4) | 3.2805 (6) | 7.0601 (13) |
| [NAIL quartile Critical Spacing plus standard USG% (not promoted)](nail-critical-spacing-quartile-standard-usage.md) | 9 | 1.197954 (6) | 1.141327 (3) | 0.1265% (7) | 14.0304 (11) | 18.1317% (11) | 14.2584 (9) | 68.10% (16) | 3.2787 (5) | 7.0460 (10) |
| [NAIL-RAPM v1.2 gap-returner priors](nail-rapm-v12-gap-returners.md) | 10 | 1.197958 (8) | 1.141344 (7) | 0.1258% (10) | 14.0414 (12) | 18.0039% (12) | 14.2660 (10) | 68.16% (15) | 3.2908 (10) | 7.0899 (15) |
| [NAIL-RAPM v1.2.3 free-throw profile (not promoted)](nail-rapm-v123-free-throw-profile.md) | 11 | 1.197964 (9) | 1.141341 (6) | 0.1248% (11) | 14.0456 (13) | 17.9547% (13) | 14.2726 (12) | 68.47% (3) | 3.2974 (12) | 7.0511 (11) |
| [NAIL-RAPM v1.2.4 free-throw replacement (not promoted)](nail-rapm-v124-free-throw-replacement.md) | 12 | 1.197966 (10) | 1.141328 (4) | 0.1244% (12) | 14.0457 (14) | 17.9541% (14) | 14.2733 (13) | 68.19% (14) | 3.2933 (11) | 7.0459 (9) |
| [Split NAIL-RAPM constrained O/D decomposition (not promoted)](split-nail-rapm.md) | 13 | 1.198026 (30) | 1.142568 (36) | 0.1144% (30) | **13.9679 (1)** | **18.8598% (1)** | 14.2668 (11) | 66.56% (32) | 3.3020 (13) | **6.9130 (1)** |
| [NAIL-RAPM v1.4 Kalman additive profiles (not promoted)](nail-rapm-v14-filtered-additive-profiles.md) | 14 | 1.197972 (12) | 1.141432 (26) | 0.1235% (14) | 14.0630 (15) | 17.7513% (15) | 14.2750 (14) | 68.39% (7) | 3.3174 (14) | 7.0088 (5) |
| [State-Precision NAIL (posterior uncertainty, no forgetting; not promoted)](state-precision-no-forgetting.md) | 15 | 1.197970 (11) | 1.141453 (28) | 0.1239% (13) | 14.0255 (9) | 18.1896% (9) | 14.2783 (15) | 67.84% (22) | 3.4051 (19) | 7.1644 (17) |
| [NAIL-RAPM v1.3 additive profiles (not promoted)](nail-rapm-v13-additive-profiles.md) | 16 | 1.197987 (17) | 1.141423 (24) | 0.1211% (18) | 14.0811 (16) | 17.5398% (16) | 14.2981 (16) | 68.44% (5) | 3.3320 (15) | 7.0619 (14) |
| [NAIL-RAPM v1.1 stat-specific padding](nail-rapm-v11-profile-padding.md) | 17 | 1.197979 (14) | 1.141391 (15) | 0.1224% (16) | 14.0864 (17) | 17.4775% (17) | 14.3236 (18) | **68.53% (1)** | 3.3898 (17) | 7.2757 (18) |
| [NAIL-RAPM v1.3.1 pruned additive profile](nail-rapm-v131-pruned-additive-profiles.md) | 18 | 1.197990 (19) | 1.141406 (22) | 0.1205% (20) | 14.0891 (18) | 17.4455% (18) | 14.3072 (17) | 68.36% (9) | 3.3446 (16) | 7.1012 (16) |
| [NAIL-RAPM fixed context alpha=20,000](nail-context-regularization.md) | 19 | 1.197980 (15) | 1.141411 (23) | 0.1221% (17) | 14.0907 (19) | 17.4268% (19) | 14.3296 (19) | 68.24% (13) | 3.4095 (20) | 7.2966 (21) |
| [NAIL-RAPM fixed context alpha=5,000](nail-context-regularization.md) | 20 | 1.197994 (20) | 1.141463 (29) | 0.1199% (21) | 14.0949 (20) | 17.3778% (20) | 14.3300 (20) | 67.84% (23) | 3.4156 (21) | 7.2821 (20) |
| [NAIL-RAPM fixed context alpha=1,000](nail-context-regularization.md) | 21 | 1.197999 (22) | 1.141439 (27) | 0.1190% (23) | 14.1017 (21) | 17.2975% (21) | 14.3389 (21) | 67.99% (18) | 3.4009 (18) | 7.2776 (19) |
| [NAIL-RAPM v1.0](nail-rapm-v1.md) | 22 | 1.197977 (13) | 1.141387 (13) | 0.1226% (15) | 14.1119 (22) | 17.1779% (22) | 14.3516 (22) | 68.33% (10) | 3.4307 (24) | 7.3460 (23) |
| [NAIL token-MLP residual](nail-token-residual.md) | 23 | 1.197986 (16) | 1.141411 (23) | 0.1211% (18) | 14.1207 (23) | 17.0756% (23) | 14.3698 (23) | 67.99% (18) | 3.4669 (27) | 7.3841 (25) |
| [HPM x3 ORB claim rebound replacement](hpm-x3.md) | 24 | 1.198000 (23) | 1.141400 (18) | 0.1188% (24) | 14.1309 (25) | 16.9555% (25) | 14.3774 (25) | 67.73% (26) | 3.4229 (22) | 7.3269 (22) |
| [NAIL-RAPM additive-only context](nail-additive-only-context.md) | 24 | 1.197989 (18) | 1.141430 (25) | 0.1206% (19) | 14.1286 (24) | 16.9824% (24) | 14.3754 (24) | 67.93% (20) | 3.4783 (28) | 7.4728 (29) |
| Value-Conditioned Aging HPM | 26 | 1.198010 (26) | 1.141402 (19) | 0.1171% (27) | 14.1342 (29) | 16.9169% (29) | 14.3802 (26) | 67.84% (23) | 3.4290 (23) | 7.3532 (24) |
| [Compiled-additive HPM x3 plus quadratic side context](linear-hpm-x3-quadratic-side-context.md) | 26 | 1.198002 (24) | 1.141366 (11) | 0.1185% (25) | 14.1311 (26) | 16.9525% (26) | 14.3922 (30) | 67.36% (31) | 3.5139 (32) | 7.5889 (33) |
| HPM v2.1 empirical rebound capacity | 27 | 1.198010 (26) | 1.141403 (20) | 0.1171% (27) | 14.1340 (28) | 16.9193% (28) | 14.3869 (28) | 67.67% (28) | 3.4365 (25) | 7.3881 (26) |
| [NAIL-RAPM normalized context penalty](nail-context-regularization.md) | 27 | 1.198002 (24) | 1.141533 (30) | 0.1185% (25) | 14.1339 (27) | 16.9203% (27) | 14.3822 (27) | 68.01% (17) | 3.5234 (33) | 7.4597 (28) |
| HPM v2 shooting composition | 27 | 1.198022 (27) | 1.141404 (21) | 0.1151% (28) | 14.1409 (30) | 16.8378% (30) | 14.3908 (29) | 67.70% (27) | 3.4592 (26) | 7.4282 (27) |
| HPM v2.2 usage allocation | 30 | 1.198004 (25) | 1.141380 (12) | 0.1182% (26) | 14.1464 (32) | 16.7730% (32) | 14.4050 (31) | 67.47% (30) | 3.4802 (29) | 7.4793 (30) |
| [HPM v2.3 shot portfolio](hpm-v23.md) | 30 | 1.198025 (28) | 1.141396 (17) | 0.1147% (29) | 14.1815 (33) | 16.3594% (33) | 14.4288 (33) | 67.70% (27) | 3.5037 (30) | 7.5274 (31) |
| [NAIL Set Attention residual](nail-token-residual.md) | 31 | 1.197997 (21) | 1.141396 (17) | 0.1192% (22) | 14.1458 (31) | 16.7796% (31) | 14.4098 (32) | 67.64% (29) | 3.5075 (31) | 7.5605 (32) |
| [Additive prior plus linear non-additive context](additive-profile-linear-shape-context.md) | 31 | 1.198026 (29) | 1.141366 (11) | 0.1144% (31) | 14.2154 (37) | 15.9592% (37) | 14.4542 (34) | 67.76% (25) | 3.5492 (34) | 7.4793 (30) |
| [HPM x2 raw OREB/100 context](hpm-x2.md) | 34 | 1.198048 (32) | 1.141660 (32) | 0.1108% (33) | 14.2073 (34) | 16.0552% (34) | 14.4641 (35) | 67.79% (24) | 3.6558 (35) | 7.7389 (34) |
| [HPM x1 ORB claim context](hpm-x1.md) | 35 | 1.198047 (31) | 1.141658 (31) | 0.1110% (32) | 14.2081 (35) | 16.0449% (35) | 14.4665 (36) | 67.87% (21) | 3.6653 (36) | 7.7498 (35) |
| Complete player-prior RAPM, no context or box score | 36 | 1.198061 (33) | 1.141675 (33) | 0.1086% (34) | 14.2105 (36) | 16.0175% (36) | 14.4756 (37) | 67.64% (29) | 3.6754 (37) | 7.7561 (36) |
| Forward 1-year RAPM-prior baseline | 37 | 1.198196 (34) | 1.141757 (34) | 0.0861% (35) | 14.5265 (38) | 12.2406% (38) | 14.8154 (38) | 65.39% (33) | 4.2374 (38) | 9.1066 (37) |
| Forward 3-year RAPM-prior baseline | 38 | 1.198246 (35) | 1.141950 (35) | 0.0778% (36) | 14.6559 (39) | 10.6699% (39) | 14.9714 (39) | 64.68% (34) | 4.5326 (39) | 9.7789 (38) |

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
| [NAIL-RAPM v1.3 additive profiles (not promoted)](nail-rapm-v13-additive-profiles.md) | 3 | **1.192630 (1)** | 1.137623 (14) | **0.0780% (1)** | 16.5034 (3) | 8.7464% (3) |
| [NAIL teammate-continuity replacement candidate (not promoted)](nail-teammate-continuity-replacement.md) | 4 | 1.192647 (4) | 1.137683 (25) | 0.0751% (4) | 16.4640 (2) | 9.1816% (2) |
| [State-Precision NAIL (posterior uncertainty, no forgetting; not promoted)](state-precision-no-forgetting.md) | 4 | 1.192641 (2) | 1.137743 (32) | 0.0760% (2) | 16.5191 (4) | 8.5725% (4) |
| [NAIL-RAPM v1.3.1 pruned additive profile](nail-rapm-v131-pruned-additive-profiles.md) | 5 | 1.192646 (3) | 1.137625 (16) | 0.0753% (3) | 16.5248 (5) | 8.5093% (5) |
| [NAIL prior teammate-continuity candidate (not promoted)](nail-teammate-continuity.md) | 5 | 1.192663 (5) | 1.137726 (30) | 0.0741% (5) | **16.4597 (1)** | **9.2313% (1)** |
| [NAIL-RAPM additive-only context](nail-additive-only-context.md) | 7 | 1.192713 (13) | 1.137590 (5) | 0.0640% (13) | 16.5724 (7) | 7.9812% (7) |
| [NAIL-RAPM v1.2.4 free-throw replacement (not promoted)](nail-rapm-v124-free-throw-replacement.md) | 8 | 1.192706 (8) | 1.137567 (2) | 0.0652% (8) | 16.5988 (15) | 7.6880% (15) |
| [NAIL-RAPM v1.4 Kalman additive profiles (not promoted)](nail-rapm-v14-filtered-additive-profiles.md) | 9 | 1.192674 (6) | 1.137670 (23) | 0.0706% (6) | 16.5836 (9) | 7.8566% (9) |
| [NAIL-RAPM v1.2.2 defensive-rebound profile (not promoted)](nail-rapm-v122-defensive-rebound-profile.md) | 10 | 1.192702 (7) | 1.137609 (11) | 0.0659% (7) | 16.5875 (10) | 7.8132% (10) |
| [NAIL-RAPM v1.2.1.1 standard USG% (not promoted)](nail-rapm-v1211-standard-usage.md) | 10 | 1.192708 (10) | 1.137593 (6) | 0.0649% (10) | 16.5989 (16) | 7.6873% (16) |
| [NAIL Critical Spacing candidate](nail-critical-spacing.md) | 11 | 1.192706 (9) | 1.137609 (12) | 0.0652% (9) | 16.5896 (11) | 7.7902% (11) |
| [NAIL-RAPM v1.2.1 pruned non-additive context](nail-rapm-v121-pruned-nonadditive.md) | 11 | 1.192709 (11) | 1.137608 (10) | 0.0647% (11) | 16.5942 (12) | 7.7392% (12) |
| [NAIL-RAPM v1.2.1.2 back-to-back schedule control](nail-rapm-v1212-back-to-back.md) **(Prior production)** | 12 | 1.192710 (12) | 1.137604 (8) | 0.0645% (12) | 16.6032 (17) | 7.6393% (17) |
| [Compiled-additive HPM x3 plus quadratic side context](linear-hpm-x3-quadratic-side-context.md) | 13 | 1.192719 (16) | **1.137554 (1)** | 0.0630% (16) | 16.5978 (13) | 7.6986% (13) |
| [NAIL quartile Critical Spacing plus standard USG% (not promoted)](nail-critical-spacing-quartile-standard-usage.md) | 13 | 1.192713 (13) | 1.137596 (7) | 0.0640% (13) | 16.6145 (20) | 7.5130% (20) |
| [NAIL-RAPM v1.2.3 free-throw profile (not promoted)](nail-rapm-v123-free-throw-profile.md) | 14 | 1.192714 (14) | 1.137583 (4) | 0.0638% (14) | 16.6157 (22) | 7.5004% (22) |
| [NAIL-RAPM v1.2.1.3 residualized-target lambda CV](nail-rapm-v1213-residualized-lambda.md) **(Production)** | 15 | 1.192717 (15) | 1.137624 (15) | 0.0633% (15) | 16.5784 (8) | 7.9150% (8) |
| [NAIL-RAPM normalized context penalty](nail-context-regularization.md) | 17 | 1.192726 (17) | 1.137680 (24) | 0.0618% (17) | 16.5979 (14) | 7.6979% (14) |
| [NAIL-RAPM v1.2 gap-returner priors](nail-rapm-v12-gap-returners.md) | 19 | 1.192734 (19) | 1.137641 (18) | 0.0605% (19) | 16.6148 (21) | 7.5097% (21) |
| [NAIL-RAPM fixed context alpha=5,000](nail-context-regularization.md) | 19 | 1.192727 (18) | 1.137753 (33) | 0.0617% (18) | 16.6090 (19) | 7.5741% (19) |
| [NAIL-RAPM fixed context alpha=20,000](nail-context-regularization.md) | 21 | 1.192736 (20) | 1.137655 (21) | 0.0602% (20) | 16.6187 (23) | 7.4664% (23) |
| [NAIL-RAPM v1.1 stat-specific padding](nail-rapm-v11-profile-padding.md) | 21 | 1.192740 (21) | 1.137655 (21) | 0.0595% (21) | 16.6267 (25) | 7.3771% (25) |
| [NAIL token-MLP residual](nail-token-residual.md) | 23 | 1.192747 (23) | 1.137607 (9) | 0.0583% (23) | 16.6393 (28) | 7.2373% (28) |
| [NAIL-RAPM fixed context alpha=1,000](nail-context-regularization.md) | 24 | 1.192745 (22) | 1.137743 (32) | 0.0587% (22) | 16.6197 (24) | 7.4553% (24) |
| [NAIL-RAPM v1.0](nail-rapm-v1.md) | 24 | 1.192752 (24) | 1.137640 (17) | 0.0574% (24) | 16.6636 (30) | 6.9658% (30) |
| HPM v2 shooting composition | 25 | 1.192759 (25) | 1.137652 (19) | 0.0564% (25) | 16.6913 (34) | 6.6559% (34) |
| Complete player-prior RAPM, no context or box score | 27 | 1.192761 (27) | 1.137741 (31) | 0.0559% (27) | 16.6068 (18) | 7.5991% (18) |
| [HPM x2 raw OREB/100 context](hpm-x2.md) | 27 | 1.192760 (26) | 1.137716 (29) | 0.0562% (26) | 16.6382 (27) | 7.2857% (27) |
| [HPM x1 ORB claim context](hpm-x1.md) | 28 | 1.192763 (28) | 1.137713 (28) | 0.0557% (28) | 16.6315 (26) | 7.3242% (26) |
| [NAIL Set Attention residual](nail-token-residual.md) | 29 | 1.192768 (29) | 1.137571 (3) | 0.0547% (29) | 16.7235 (37) | 6.2956% (37) |
| [HPM v2.3 shot portfolio](hpm-v23.md) | 29 | 1.192777 (31) | 1.137641 (18) | 0.0534% (31) | 16.6546 (29) | 7.0659% (29) |
| HPM v2.1 empirical rebound capacity | 30 | 1.192772 (30) | 1.137654 (20) | 0.0542% (30) | 16.6701 (31) | 6.8932% (31) |
| [HPM x3 ORB claim rebound replacement](hpm-x3.md) | 32 | 1.192781 (32) | 1.137688 (26) | 0.0526% (32) | 16.6792 (32) | 6.7912% (32) |
| HPM v2.2 usage allocation | 33 | 1.192807 (34) | 1.137656 (22) | 0.0482% (34) | 16.6806 (33) | 6.7759% (33) |
| Value-Conditioned Aging HPM | 33 | 1.192792 (33) | 1.137701 (27) | 0.0508% (33) | 16.7155 (35) | 6.3851% (35) |
| [Additive prior plus linear non-additive context](additive-profile-linear-shape-context.md) | 35 | 1.192814 (35) | 1.137613 (13) | 0.0470% (35) | 16.7224 (36) | 6.3076% (36) |
| [Split NAIL-RAPM constrained O/D decomposition (not promoted)](split-nail-rapm.md) | 36 | 1.194415 (38) | 1.140791 (36) | -0.2214% (38) | 16.5301 (6) | 8.4506% (6) |
| Forward 3-year RAPM-prior baseline | 36 | 1.193041 (36) | 1.137974 (35) | 0.0091% (36) | 17.1808 (38) | 1.1016% (38) |
| Forward 1-year RAPM-prior baseline | 37 | 1.193123 (37) | 1.137869 (34) | -0.0047% (37) | 17.2902 (39) | -0.1627% (39) |

NAIL-RAPM v1.3.1 is the preferred, parsimonious version of the non-promoted
v1.3 branch: it passed its direct no-material-harm gate versus v1.3, but it
does not supersede the selected v1.2.1.3 model as the global regular-season
release. See its
[experiment record](nail-rapm-v131-pruned-additive-profiles.md).

See [Forward RAPM Memory Baselines](forward-rapm-memory-baselines.md) and
[Complete Player-Prior RAPM Baseline](complete-player-prior-baseline.md) for
model specifications, annual lambda selections, and immutable artifacts.
