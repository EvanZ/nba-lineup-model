---
last_updated: "2026-08-29"
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
| [NAIL lead-handler allocation candidate](nail-lead-secondary-usage-gap.md) | **1** | **1.197938 (1)** | **1.141282 (1)** | **0.1292% (1)** | 13.9859 (2) | 18.6505% (2) | **14.2041 (1)** | 67.87% (21) | **3.2284 (1)** | 6.9251 (2) |
| [NAIL-RAPM v1.2.1.3 residualized-target lambda CV](nail-rapm-v1213-residualized-lambda.md) **(Production)** | 3 | 1.197946 (2) | 1.141306 (2) | 0.1278% (3) | 13.9939 (4) | 18.5574% (4) | 14.2166 (4) | 68.30% (11) | 3.2351 (2) | 6.9423 (3) |
| [NAIL teammate-continuity replacement candidate (not promoted)](nail-teammate-continuity-replacement.md) | 4 | 1.197948 (3) | 1.141388 (15) | 0.1275% (4) | 13.9919 (3) | 18.5812% (3) | 14.2119 (2) | 68.41% (6) | 3.2853 (10) | 6.9815 (5) |
| [NAIL prior teammate-continuity candidate (not promoted)](nail-teammate-continuity.md) | 5 | 1.197955 (8) | 1.141392 (17) | 0.1263% (10) | 14.0010 (5) | 18.4755% (5) | 14.2146 (3) | 68.50% (2) | 3.2833 (8) | 6.9648 (4) |
| [NAIL-RAPM v1.2.1.2 back-to-back schedule control](nail-rapm-v1212-back-to-back.md) **(Prior production)** | 6 | 1.197946 (2) | 1.141313 (3) | 0.1279% (2) | 14.0107 (6) | 18.3623% (6) | 14.2330 (5) | 67.96% (19) | 3.2847 (9) | 7.0551 (13) |
| [NAIL-RAPM v1.2.1.1 standard USG% (not promoted)](nail-rapm-v1211-standard-usage.md) | 6 | 1.197951 (4) | 1.141332 (6) | 0.1269% (5) | 14.0264 (11) | 18.1792% (11) | 14.2516 (6) | 68.10% (16) | 3.2658 (3) | 7.0279 (7) |
| [NAIL-RAPM v1.2.2 defensive-rebound profile (not promoted)](nail-rapm-v122-defensive-rebound-profile.md) | 7 | 1.197952 (6) | 1.141364 (11) | 0.1268% (7) | 14.0238 (7) | 18.2094% (7) | 14.2529 (8) | 68.39% (8) | 3.2723 (5) | 7.0434 (9) |
| [NAIL-RAPM v1.2.1 pruned non-additive context](nail-rapm-v121-pruned-nonadditive.md) | 8 | 1.197952 (5) | 1.141355 (10) | 0.1268% (6) | 14.0245 (9) | 18.2005% (9) | 14.2521 (7) | 68.24% (12) | 3.2706 (4) | 7.0351 (8) |
| [NAIL Critical Spacing candidate](nail-critical-spacing.md) | 8 | 1.197954 (7) | 1.141347 (9) | 0.1264% (9) | 14.0241 (8) | 18.2063% (8) | 14.2532 (9) | 68.47% (4) | 3.2805 (7) | 7.0601 (14) |
| [NAIL quartile Critical Spacing plus standard USG% (not promoted)](nail-critical-spacing-quartile-standard-usage.md) | 10 | 1.197954 (7) | 1.141327 (4) | 0.1265% (8) | 14.0304 (12) | 18.1317% (12) | 14.2584 (10) | 68.10% (16) | 3.2787 (6) | 7.0460 (11) |
| [NAIL-RAPM v1.2 gap-returner priors](nail-rapm-v12-gap-returners.md) | 11 | 1.197958 (9) | 1.141344 (8) | 0.1258% (11) | 14.0414 (13) | 18.0039% (13) | 14.2660 (11) | 68.16% (15) | 3.2908 (11) | 7.0899 (16) |
| [NAIL-RAPM v1.2.3 free-throw profile (not promoted)](nail-rapm-v123-free-throw-profile.md) | 12 | 1.197964 (10) | 1.141341 (7) | 0.1248% (12) | 14.0456 (14) | 17.9547% (14) | 14.2726 (13) | 68.47% (3) | 3.2974 (13) | 7.0511 (12) |
| [NAIL-RAPM v1.2.4 free-throw replacement (not promoted)](nail-rapm-v124-free-throw-replacement.md) | 13 | 1.197966 (11) | 1.141328 (5) | 0.1244% (13) | 14.0457 (15) | 17.9541% (15) | 14.2733 (14) | 68.19% (14) | 3.2933 (12) | 7.0459 (10) |
| [Split NAIL-RAPM constrained O/D decomposition (not promoted)](split-nail-rapm.md) | 14 | 1.198026 (31) | 1.142568 (37) | 0.1144% (31) | **13.9679 (1)** | **18.8598% (1)** | 14.2668 (12) | 66.56% (33) | 3.3020 (14) | **6.9130 (1)** |
| [NAIL-RAPM v1.4 Kalman additive profiles (not promoted)](nail-rapm-v14-filtered-additive-profiles.md) | 15 | 1.197972 (13) | 1.141432 (27) | 0.1235% (15) | 14.0630 (16) | 17.7513% (16) | 14.2750 (15) | 68.39% (7) | 3.3174 (15) | 7.0088 (6) |
| [State-Precision NAIL (posterior uncertainty, no forgetting; not promoted)](state-precision-no-forgetting.md) | 16 | 1.197970 (12) | 1.141453 (29) | 0.1239% (14) | 14.0255 (10) | 18.1896% (10) | 14.2783 (16) | 67.84% (23) | 3.4051 (20) | 7.1644 (18) |
| [NAIL-RAPM v1.3 additive profiles (not promoted)](nail-rapm-v13-additive-profiles.md) | 17 | 1.197987 (18) | 1.141423 (25) | 0.1211% (19) | 14.0811 (17) | 17.5398% (17) | 14.2981 (17) | 68.44% (5) | 3.3320 (16) | 7.0619 (15) |
| [NAIL-RAPM v1.1 stat-specific padding](nail-rapm-v11-profile-padding.md) | 18 | 1.197979 (15) | 1.141391 (16) | 0.1224% (17) | 14.0864 (18) | 17.4775% (18) | 14.3236 (19) | **68.53% (1)** | 3.3898 (18) | 7.2757 (19) |
| [NAIL-RAPM v1.3.1 pruned additive profile](nail-rapm-v131-pruned-additive-profiles.md) | 19 | 1.197990 (20) | 1.141406 (23) | 0.1205% (21) | 14.0891 (19) | 17.4455% (19) | 14.3072 (18) | 68.36% (9) | 3.3446 (17) | 7.1012 (17) |
| [NAIL-RAPM fixed context alpha=20,000](nail-context-regularization.md) | 20 | 1.197980 (16) | 1.141411 (24) | 0.1221% (18) | 14.0907 (20) | 17.4268% (20) | 14.3296 (20) | 68.24% (13) | 3.4095 (21) | 7.2966 (22) |
| [NAIL-RAPM fixed context alpha=5,000](nail-context-regularization.md) | 21 | 1.197994 (21) | 1.141463 (30) | 0.1199% (22) | 14.0949 (21) | 17.3778% (21) | 14.3300 (21) | 67.84% (24) | 3.4156 (22) | 7.2821 (21) |
| [NAIL-RAPM fixed context alpha=1,000](nail-context-regularization.md) | 22 | 1.197999 (23) | 1.141439 (28) | 0.1190% (24) | 14.1017 (22) | 17.2975% (22) | 14.3389 (22) | 67.99% (18) | 3.4009 (19) | 7.2776 (20) |
| [NAIL-RAPM v1.0](nail-rapm-v1.md) | 23 | 1.197977 (14) | 1.141387 (14) | 0.1226% (16) | 14.1119 (23) | 17.1779% (23) | 14.3516 (23) | 68.33% (10) | 3.4307 (25) | 7.3460 (24) |
| [NAIL token-MLP residual](nail-token-residual.md) | 24 | 1.197986 (17) | 1.141411 (24) | 0.1211% (19) | 14.1207 (24) | 17.0756% (24) | 14.3698 (24) | 67.99% (18) | 3.4669 (28) | 7.3841 (26) |
| [NAIL-RAPM additive-only context](nail-additive-only-context.md) | 25 | 1.197989 (19) | 1.141430 (26) | 0.1206% (20) | 14.1286 (25) | 16.9824% (25) | 14.3754 (25) | 67.93% (20) | 3.4783 (29) | 7.4728 (30) |
| [HPM x3 ORB claim rebound replacement](hpm-x3.md) | 25 | 1.198000 (24) | 1.141400 (19) | 0.1188% (25) | 14.1309 (26) | 16.9555% (26) | 14.3774 (26) | 67.73% (27) | 3.4229 (23) | 7.3269 (23) |
| Value-Conditioned Aging HPM | 27 | 1.198010 (27) | 1.141402 (20) | 0.1171% (28) | 14.1342 (30) | 16.9169% (30) | 14.3802 (27) | 67.84% (24) | 3.4290 (24) | 7.3532 (25) |
| [Compiled-additive HPM x3 plus quadratic side context](linear-hpm-x3-quadratic-side-context.md) | 27 | 1.198002 (25) | 1.141366 (12) | 0.1185% (26) | 14.1311 (27) | 16.9525% (27) | 14.3922 (31) | 67.36% (32) | 3.5139 (33) | 7.5889 (34) |
| HPM v2.1 empirical rebound capacity | 28 | 1.198010 (27) | 1.141403 (21) | 0.1171% (28) | 14.1340 (29) | 16.9193% (29) | 14.3869 (29) | 67.67% (29) | 3.4365 (26) | 7.3881 (27) |
| [NAIL-RAPM normalized context penalty](nail-context-regularization.md) | 28 | 1.198002 (25) | 1.141533 (31) | 0.1185% (26) | 14.1339 (28) | 16.9203% (28) | 14.3822 (28) | 68.01% (17) | 3.5234 (34) | 7.4597 (29) |
| HPM v2 shooting composition | 28 | 1.198022 (28) | 1.141404 (22) | 0.1151% (29) | 14.1409 (31) | 16.8378% (31) | 14.3908 (30) | 67.70% (28) | 3.4592 (27) | 7.4282 (28) |
| HPM v2.2 usage allocation | 31 | 1.198004 (26) | 1.141380 (13) | 0.1182% (27) | 14.1464 (33) | 16.7730% (33) | 14.4050 (32) | 67.47% (31) | 3.4802 (30) | 7.4793 (31) |
| [HPM v2.3 shot portfolio](hpm-v23.md) | 31 | 1.198025 (29) | 1.141396 (18) | 0.1147% (30) | 14.1815 (34) | 16.3594% (34) | 14.4288 (34) | 67.70% (28) | 3.5037 (31) | 7.5274 (32) |
| [NAIL Set Attention residual](nail-token-residual.md) | 32 | 1.197997 (22) | 1.141396 (18) | 0.1192% (23) | 14.1458 (32) | 16.7796% (32) | 14.4098 (33) | 67.64% (30) | 3.5075 (32) | 7.5605 (33) |
| [Additive prior plus linear non-additive context](additive-profile-linear-shape-context.md) | 32 | 1.198026 (30) | 1.141366 (12) | 0.1144% (32) | 14.2154 (38) | 15.9592% (38) | 14.4542 (35) | 67.76% (26) | 3.5492 (35) | 7.4793 (31) |
| [HPM x2 raw OREB/100 context](hpm-x2.md) | 35 | 1.198048 (33) | 1.141660 (33) | 0.1108% (34) | 14.2073 (35) | 16.0552% (35) | 14.4641 (36) | 67.79% (25) | 3.6558 (36) | 7.7389 (35) |
| [HPM x1 ORB claim context](hpm-x1.md) | 36 | 1.198047 (32) | 1.141658 (32) | 0.1110% (33) | 14.2081 (36) | 16.0449% (36) | 14.4665 (37) | 67.87% (22) | 3.6653 (37) | 7.7498 (36) |
| Complete player-prior RAPM, no context or box score | 37 | 1.198061 (34) | 1.141675 (34) | 0.1086% (35) | 14.2105 (37) | 16.0175% (37) | 14.4756 (38) | 67.64% (30) | 3.6754 (38) | 7.7561 (37) |
| Forward 1-year RAPM-prior baseline | 38 | 1.198196 (35) | 1.141757 (35) | 0.0861% (36) | 14.5265 (39) | 12.2406% (39) | 14.8154 (39) | 65.39% (34) | 4.2374 (39) | 9.1066 (38) |
| Forward 3-year RAPM-prior baseline | 39 | 1.198246 (36) | 1.141950 (36) | 0.0778% (37) | 14.6559 (40) | 10.6699% (40) | 14.9714 (40) | 64.68% (35) | 4.5326 (40) | 9.7789 (39) |

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
| [NAIL-RAPM v1.3 additive profiles (not promoted)](nail-rapm-v13-additive-profiles.md) | 4 | **1.192630 (1)** | 1.137623 (14) | **0.0780% (1)** | 16.5034 (4) | 8.7464% (4) |
| [NAIL teammate-continuity replacement candidate (not promoted)](nail-teammate-continuity-replacement.md) | 4 | 1.192647 (4) | 1.137683 (26) | 0.0751% (4) | 16.4640 (2) | 9.1816% (2) |
| [NAIL prior teammate-continuity candidate (not promoted)](nail-teammate-continuity.md) | 5 | 1.192663 (5) | 1.137726 (31) | 0.0741% (5) | **16.4597 (1)** | **9.2313% (1)** |
| [State-Precision NAIL (posterior uncertainty, no forgetting; not promoted)](state-precision-no-forgetting.md) | 5 | 1.192641 (2) | 1.137743 (33) | 0.0760% (2) | 16.5191 (5) | 8.5725% (5) |
| [NAIL-RAPM v1.3.1 pruned additive profile](nail-rapm-v131-pruned-additive-profiles.md) | 6 | 1.192646 (3) | 1.137625 (16) | 0.0753% (3) | 16.5248 (6) | 8.5093% (6) |
| [NAIL-RAPM additive-only context](nail-additive-only-context.md) | 8 | 1.192713 (13) | 1.137590 (5) | 0.0640% (13) | 16.5724 (8) | 7.9812% (8) |
| [NAIL-RAPM v1.2.4 free-throw replacement (not promoted)](nail-rapm-v124-free-throw-replacement.md) | 8 | 1.192706 (8) | 1.137567 (2) | 0.0652% (8) | 16.5988 (16) | 7.6880% (16) |
| [NAIL-RAPM v1.4 Kalman additive profiles (not promoted)](nail-rapm-v14-filtered-additive-profiles.md) | 10 | 1.192674 (6) | 1.137670 (24) | 0.0706% (6) | 16.5836 (10) | 7.8566% (10) |
| [NAIL-RAPM v1.2.1.1 standard USG% (not promoted)](nail-rapm-v1211-standard-usage.md) | 10 | 1.192708 (10) | 1.137593 (6) | 0.0649% (10) | 16.5989 (17) | 7.6873% (17) |
| [NAIL-RAPM v1.2.2 defensive-rebound profile (not promoted)](nail-rapm-v122-defensive-rebound-profile.md) | 11 | 1.192702 (7) | 1.137609 (11) | 0.0659% (7) | 16.5875 (11) | 7.8132% (11) |
| [NAIL-RAPM v1.2.1 pruned non-additive context](nail-rapm-v121-pruned-nonadditive.md) | 11 | 1.192709 (11) | 1.137608 (10) | 0.0647% (11) | 16.5942 (13) | 7.7392% (13) |
| [NAIL Critical Spacing candidate](nail-critical-spacing.md) | 12 | 1.192706 (9) | 1.137609 (12) | 0.0652% (9) | 16.5896 (12) | 7.7902% (12) |
| [NAIL-RAPM v1.2.1.2 back-to-back schedule control](nail-rapm-v1212-back-to-back.md) **(Prior production)** | 12 | 1.192710 (12) | 1.137604 (8) | 0.0645% (12) | 16.6032 (18) | 7.6393% (18) |
| [NAIL quartile Critical Spacing plus standard USG% (not promoted)](nail-critical-spacing-quartile-standard-usage.md) | 13 | 1.192713 (13) | 1.137596 (7) | 0.0640% (13) | 16.6145 (21) | 7.5130% (21) |
| [Compiled-additive HPM x3 plus quadratic side context](linear-hpm-x3-quadratic-side-context.md) | 14 | 1.192719 (16) | **1.137554 (1)** | 0.0630% (16) | 16.5978 (14) | 7.6986% (14) |
| [NAIL-RAPM v1.2.3 free-throw profile (not promoted)](nail-rapm-v123-free-throw-profile.md) | 14 | 1.192714 (14) | 1.137583 (4) | 0.0638% (14) | 16.6157 (23) | 7.5004% (23) |
| [NAIL-RAPM v1.2.1.3 residualized-target lambda CV](nail-rapm-v1213-residualized-lambda.md) **(Production)** | 15 | 1.192717 (15) | 1.137624 (15) | 0.0633% (15) | 16.5784 (9) | 7.9150% (9) |
| [NAIL lead-handler allocation candidate](nail-lead-secondary-usage-gap.md) | 17 | 1.192730 (19) | 1.137627 (17) | 0.0611% (19) | 16.4990 (3) | 8.7944% (3) |
| [NAIL-RAPM normalized context penalty](nail-context-regularization.md) | 17 | 1.192726 (17) | 1.137680 (25) | 0.0618% (17) | 16.5979 (15) | 7.6979% (15) |
| [NAIL-RAPM v1.2 gap-returner priors](nail-rapm-v12-gap-returners.md) | 20 | 1.192734 (20) | 1.137641 (19) | 0.0605% (20) | 16.6148 (22) | 7.5097% (22) |
| [NAIL-RAPM fixed context alpha=5,000](nail-context-regularization.md) | 20 | 1.192727 (18) | 1.137753 (34) | 0.0617% (18) | 16.6090 (20) | 7.5741% (20) |
| [NAIL-RAPM fixed context alpha=20,000](nail-context-regularization.md) | 22 | 1.192736 (21) | 1.137655 (22) | 0.0602% (21) | 16.6187 (24) | 7.4664% (24) |
| [NAIL-RAPM v1.1 stat-specific padding](nail-rapm-v11-profile-padding.md) | 22 | 1.192740 (22) | 1.137655 (22) | 0.0595% (22) | 16.6267 (26) | 7.3771% (26) |
| [NAIL token-MLP residual](nail-token-residual.md) | 24 | 1.192747 (24) | 1.137607 (9) | 0.0583% (24) | 16.6393 (29) | 7.2373% (29) |
| [NAIL-RAPM fixed context alpha=1,000](nail-context-regularization.md) | 25 | 1.192745 (23) | 1.137743 (33) | 0.0587% (23) | 16.6197 (25) | 7.4553% (25) |
| [NAIL-RAPM v1.0](nail-rapm-v1.md) | 25 | 1.192752 (25) | 1.137640 (18) | 0.0574% (25) | 16.6636 (31) | 6.9658% (31) |
| HPM v2 shooting composition | 26 | 1.192759 (26) | 1.137652 (20) | 0.0564% (26) | 16.6913 (35) | 6.6559% (35) |
| Complete player-prior RAPM, no context or box score | 28 | 1.192761 (28) | 1.137741 (32) | 0.0559% (28) | 16.6068 (19) | 7.5991% (19) |
| [HPM x2 raw OREB/100 context](hpm-x2.md) | 28 | 1.192760 (27) | 1.137716 (30) | 0.0562% (27) | 16.6382 (28) | 7.2857% (28) |
| [HPM x1 ORB claim context](hpm-x1.md) | 29 | 1.192763 (29) | 1.137713 (29) | 0.0557% (29) | 16.6315 (27) | 7.3242% (27) |
| [NAIL Set Attention residual](nail-token-residual.md) | 30 | 1.192768 (30) | 1.137571 (3) | 0.0547% (30) | 16.7235 (38) | 6.2956% (38) |
| [HPM v2.3 shot portfolio](hpm-v23.md) | 30 | 1.192777 (32) | 1.137641 (19) | 0.0534% (32) | 16.6546 (30) | 7.0659% (30) |
| HPM v2.1 empirical rebound capacity | 31 | 1.192772 (31) | 1.137654 (21) | 0.0542% (31) | 16.6701 (32) | 6.8932% (32) |
| [HPM x3 ORB claim rebound replacement](hpm-x3.md) | 33 | 1.192781 (33) | 1.137688 (27) | 0.0526% (33) | 16.6792 (33) | 6.7912% (33) |
| HPM v2.2 usage allocation | 34 | 1.192807 (35) | 1.137656 (23) | 0.0482% (35) | 16.6806 (34) | 6.7759% (34) |
| Value-Conditioned Aging HPM | 34 | 1.192792 (34) | 1.137701 (28) | 0.0508% (34) | 16.7155 (36) | 6.3851% (36) |
| [Additive prior plus linear non-additive context](additive-profile-linear-shape-context.md) | 36 | 1.192814 (36) | 1.137613 (13) | 0.0470% (36) | 16.7224 (37) | 6.3076% (37) |
| [Split NAIL-RAPM constrained O/D decomposition (not promoted)](split-nail-rapm.md) | 37 | 1.194415 (39) | 1.140791 (37) | -0.2214% (39) | 16.5301 (7) | 8.4506% (7) |
| Forward 3-year RAPM-prior baseline | 37 | 1.193041 (37) | 1.137974 (36) | 0.0091% (37) | 17.1808 (39) | 1.1016% (39) |
| Forward 1-year RAPM-prior baseline | 38 | 1.193123 (38) | 1.137869 (35) | -0.0047% (38) | 17.2902 (40) | -0.1627% (40) |

NAIL-RAPM v1.3.1 is the preferred, parsimonious version of the non-promoted
v1.3 branch: it passed its direct no-material-harm gate versus v1.3, but it
does not supersede the selected v1.2.1.3 model as the global regular-season
release. See its
[experiment record](nail-rapm-v131-pruned-additive-profiles.md).

See [Forward RAPM Memory Baselines](forward-rapm-memory-baselines.md) and
[Complete Player-Prior RAPM Baseline](complete-player-prior-baseline.md) for
model specifications, annual lambda selections, and immutable artifacts.
