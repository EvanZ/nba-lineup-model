---
last_updated: "2026-09-04"
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

The [v1.2.1.4 player-trend aging candidate](nail-rapm-v1214-player-trend-aging.md)
uses the same recovered support. Its possession and eligible-game cells are
the paired deltas normalized to the legacy production row; full-game and team
results are direct. It cleared the agreed bootstrap non-promotion gate, but is
not deployed pending release approval.

| Model | Median rank | Poss. RMSE | Poss. MAE | Poss. skill | Eligible game RMSE | Eligible game skill | Full-game RMSE | Winner accuracy | Team NetRtg RMSE | Pythagorean-win RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [NAIL lead-handler allocation candidate](nail-lead-secondary-usage-gap.md) | 2 | **1.197938 (1)** | **1.141282 (1)** | **0.1292% (1)** | 13.9859 (2) | 18.6505% (2) | **14.2041 (1)** | 67.87% (22) | 3.2284 (2) | 6.9251 (3) |
| [NAIL-RAPM v1.2.1.4 player-trend aging](nail-rapm-v1214-player-trend-aging.md) **(Promotion eligible; not deployed)** | 2 | 1.197944 (2) | 1.141291 (2) | 0.1262% (11) | 13.9885 (3) | 18.1370% (12) | 14.2080 (2) | **68.56% (1)** | **3.2206 (1)** | **6.8851 (1)** |
| [NAIL-RAPM v1.2.1.3 residualized-target lambda CV](nail-rapm-v1213-residualized-lambda.md) **(Production)** | 4 | 1.197946 (3) | 1.141306 (3) | 0.1278% (3) | 13.9939 (5) | 18.5574% (4) | 14.2166 (5) | 68.30% (12) | 3.2351 (3) | 6.9423 (4) |
| [NAIL teammate-continuity replacement candidate (not promoted)](nail-teammate-continuity-replacement.md) | 4 | 1.197948 (4) | 1.141388 (16) | 0.1275% (4) | 13.9919 (4) | 18.5812% (3) | 14.2119 (3) | 68.41% (7) | 3.2853 (11) | 6.9815 (6) |
| [NAIL prior teammate-continuity candidate (not promoted)](nail-teammate-continuity.md) | 6 | 1.197955 (9) | 1.141392 (18) | 0.1263% (10) | 14.0010 (6) | 18.4755% (5) | 14.2146 (4) | 68.50% (3) | 3.2833 (9) | 6.9648 (5) |
| [NAIL-RAPM v1.2.1.2 back-to-back schedule control](nail-rapm-v1212-back-to-back.md) **(Prior production)** | 6 | 1.197946 (3) | 1.141313 (4) | 0.1279% (2) | 14.0107 (7) | 18.3623% (6) | 14.2330 (6) | 67.96% (20) | 3.2847 (10) | 7.0551 (14) |
| [NAIL-RAPM v1.2.1.1 standard USG% (not promoted)](nail-rapm-v1211-standard-usage.md) | 7 | 1.197951 (5) | 1.141332 (7) | 0.1269% (5) | 14.0264 (12) | 18.1792% (11) | 14.2516 (7) | 68.10% (17) | 3.2658 (4) | 7.0279 (8) |
| [NAIL-RAPM v1.2.2 defensive-rebound profile (not promoted)](nail-rapm-v122-defensive-rebound-profile.md) | 8 | 1.197952 (7) | 1.141364 (12) | 0.1268% (7) | 14.0238 (8) | 18.2094% (7) | 14.2529 (9) | 68.39% (9) | 3.2723 (6) | 7.0434 (10) |
| [NAIL-RAPM v1.2.1 pruned non-additive context](nail-rapm-v121-pruned-nonadditive.md) | 9 | 1.197952 (6) | 1.141355 (11) | 0.1268% (6) | 14.0245 (10) | 18.2005% (9) | 14.2521 (8) | 68.24% (13) | 3.2706 (5) | 7.0351 (9) |
| [NAIL Critical Spacing candidate](nail-critical-spacing.md) | 9 | 1.197954 (8) | 1.141347 (10) | 0.1264% (9) | 14.0241 (9) | 18.2063% (8) | 14.2532 (10) | 68.47% (5) | 3.2805 (8) | 7.0601 (15) |
| [NAIL quartile Critical Spacing plus standard USG% (not promoted)](nail-critical-spacing-quartile-standard-usage.md) | 11 | 1.197954 (8) | 1.141327 (5) | 0.1265% (8) | 14.0304 (13) | 18.1317% (13) | 14.2584 (11) | 68.10% (17) | 3.2787 (7) | 7.0460 (12) |
| [NAIL-RAPM v1.2 gap-returner priors](nail-rapm-v12-gap-returners.md) | 12 | 1.197958 (10) | 1.141344 (9) | 0.1258% (12) | 14.0414 (14) | 18.0039% (14) | 14.2660 (12) | 68.16% (16) | 3.2908 (12) | 7.0899 (17) |
| [NAIL-RAPM v1.2.3 free-throw profile (not promoted)](nail-rapm-v123-free-throw-profile.md) | 13 | 1.197964 (11) | 1.141341 (8) | 0.1248% (13) | 14.0456 (15) | 17.9547% (15) | 14.2726 (14) | 68.47% (4) | 3.2974 (14) | 7.0511 (13) |
| [NAIL-RAPM v1.2.4 free-throw replacement (not promoted)](nail-rapm-v124-free-throw-replacement.md) | 14 | 1.197966 (12) | 1.141328 (6) | 0.1244% (14) | 14.0457 (16) | 17.9541% (16) | 14.2733 (15) | 68.19% (15) | 3.2933 (13) | 7.0459 (11) |
| [Split NAIL-RAPM constrained O/D decomposition (not promoted)](split-nail-rapm.md) | 15 | 1.198026 (32) | 1.142568 (38) | 0.1144% (32) | **13.9679 (1)** | **18.8598% (1)** | 14.2668 (13) | 66.56% (34) | 3.3020 (15) | 6.9130 (2) |
| [NAIL-RAPM v1.4 Kalman additive profiles (not promoted)](nail-rapm-v14-filtered-additive-profiles.md) | 16 | 1.197972 (14) | 1.141432 (28) | 0.1235% (16) | 14.0630 (17) | 17.7513% (17) | 14.2750 (16) | 68.39% (8) | 3.3174 (16) | 7.0088 (7) |
| [State-Precision NAIL (posterior uncertainty, no forgetting; not promoted)](state-precision-no-forgetting.md) | 17 | 1.197970 (13) | 1.141453 (30) | 0.1239% (15) | 14.0255 (11) | 18.1896% (10) | 14.2783 (17) | 67.84% (24) | 3.4051 (21) | 7.1644 (19) |
| [NAIL-RAPM v1.3 additive profiles (not promoted)](nail-rapm-v13-additive-profiles.md) | 18 | 1.197987 (19) | 1.141423 (26) | 0.1211% (20) | 14.0811 (18) | 17.5398% (18) | 14.2981 (18) | 68.44% (6) | 3.3320 (17) | 7.0619 (16) |
| [NAIL-RAPM v1.1 stat-specific padding](nail-rapm-v11-profile-padding.md) | 19 | 1.197979 (16) | 1.141391 (17) | 0.1224% (18) | 14.0864 (19) | 17.4775% (19) | 14.3236 (20) | 68.53% (2) | 3.3898 (19) | 7.2757 (20) |
| [NAIL-RAPM v1.3.1 pruned additive profile](nail-rapm-v131-pruned-additive-profiles.md) | 20 | 1.197990 (21) | 1.141406 (24) | 0.1205% (22) | 14.0891 (20) | 17.4455% (20) | 14.3072 (19) | 68.36% (10) | 3.3446 (18) | 7.1012 (18) |
| [NAIL-RAPM fixed context alpha=20,000](nail-context-regularization.md) | 21 | 1.197980 (17) | 1.141411 (25) | 0.1221% (19) | 14.0907 (21) | 17.4268% (21) | 14.3296 (21) | 68.24% (14) | 3.4095 (22) | 7.2966 (23) |
| [NAIL-RAPM fixed context alpha=5,000](nail-context-regularization.md) | 22 | 1.197994 (22) | 1.141463 (31) | 0.1199% (23) | 14.0949 (22) | 17.3778% (22) | 14.3300 (22) | 67.84% (25) | 3.4156 (23) | 7.2821 (22) |
| [NAIL-RAPM fixed context alpha=1,000](nail-context-regularization.md) | 23 | 1.197999 (24) | 1.141439 (29) | 0.1190% (25) | 14.1017 (23) | 17.2975% (23) | 14.3389 (23) | 67.99% (19) | 3.4009 (20) | 7.2776 (21) |
| [NAIL-RAPM v1.0](nail-rapm-v1.md) | 24 | 1.197977 (15) | 1.141387 (15) | 0.1226% (17) | 14.1119 (24) | 17.1779% (24) | 14.3516 (24) | 68.33% (11) | 3.4307 (26) | 7.3460 (25) |
| [NAIL token-MLP residual](nail-token-residual.md) | 25 | 1.197986 (18) | 1.141411 (25) | 0.1211% (20) | 14.1207 (25) | 17.0756% (25) | 14.3698 (25) | 67.99% (19) | 3.4669 (29) | 7.3841 (27) |
| [NAIL-RAPM additive-only context](nail-additive-only-context.md) | 26 | 1.197989 (20) | 1.141430 (27) | 0.1206% (21) | 14.1286 (26) | 16.9824% (26) | 14.3754 (26) | 67.93% (21) | 3.4783 (30) | 7.4728 (31) |
| [HPM x3 ORB claim rebound replacement](hpm-x3.md) | 26 | 1.198000 (25) | 1.141400 (20) | 0.1188% (26) | 14.1309 (27) | 16.9555% (27) | 14.3774 (27) | 67.73% (28) | 3.4229 (24) | 7.3269 (24) |
| Value-Conditioned Aging HPM | 28 | 1.198010 (28) | 1.141402 (21) | 0.1171% (29) | 14.1342 (31) | 16.9169% (31) | 14.3802 (28) | 67.84% (25) | 3.4290 (25) | 7.3532 (26) |
| [Compiled-additive HPM x3 plus quadratic side context](linear-hpm-x3-quadratic-side-context.md) | 28 | 1.198002 (26) | 1.141366 (13) | 0.1185% (27) | 14.1311 (28) | 16.9525% (28) | 14.3922 (32) | 67.36% (33) | 3.5139 (34) | 7.5889 (35) |
| HPM v2.1 empirical rebound capacity | 29 | 1.198010 (28) | 1.141403 (22) | 0.1171% (29) | 14.1340 (30) | 16.9193% (30) | 14.3869 (30) | 67.67% (30) | 3.4365 (27) | 7.3881 (28) |
| [NAIL-RAPM normalized context penalty](nail-context-regularization.md) | 29 | 1.198002 (26) | 1.141533 (32) | 0.1185% (27) | 14.1339 (29) | 16.9203% (29) | 14.3822 (29) | 68.01% (18) | 3.5234 (35) | 7.4597 (30) |
| HPM v2 shooting composition | 29 | 1.198022 (29) | 1.141404 (23) | 0.1151% (30) | 14.1409 (32) | 16.8378% (32) | 14.3908 (31) | 67.70% (29) | 3.4592 (28) | 7.4282 (29) |
| HPM v2.2 usage allocation | 32 | 1.198004 (27) | 1.141380 (14) | 0.1182% (28) | 14.1464 (34) | 16.7730% (34) | 14.4050 (33) | 67.47% (32) | 3.4802 (31) | 7.4793 (32) |
| [HPM v2.3 shot portfolio](hpm-v23.md) | 32 | 1.198025 (30) | 1.141396 (19) | 0.1147% (31) | 14.1815 (35) | 16.3594% (35) | 14.4288 (35) | 67.70% (29) | 3.5037 (32) | 7.5274 (33) |
| [NAIL Set Attention residual](nail-token-residual.md) | 33 | 1.197997 (23) | 1.141396 (19) | 0.1192% (24) | 14.1458 (33) | 16.7796% (33) | 14.4098 (34) | 67.64% (31) | 3.5075 (33) | 7.5605 (34) |
| [Additive prior plus linear non-additive context](additive-profile-linear-shape-context.md) | 33 | 1.198026 (31) | 1.141366 (13) | 0.1144% (33) | 14.2154 (39) | 15.9592% (39) | 14.4542 (36) | 67.76% (27) | 3.5492 (36) | 7.4793 (32) |
| [HPM x2 raw OREB/100 context](hpm-x2.md) | 36 | 1.198048 (34) | 1.141660 (34) | 0.1108% (35) | 14.2073 (36) | 16.0552% (36) | 14.4641 (37) | 67.79% (26) | 3.6558 (37) | 7.7389 (36) |
| [HPM x1 ORB claim context](hpm-x1.md) | 37 | 1.198047 (33) | 1.141658 (33) | 0.1110% (34) | 14.2081 (37) | 16.0449% (37) | 14.4665 (38) | 67.87% (23) | 3.6653 (38) | 7.7498 (37) |
| Complete player-prior RAPM, no context or box score | 38 | 1.198061 (35) | 1.141675 (35) | 0.1086% (36) | 14.2105 (38) | 16.0175% (38) | 14.4756 (39) | 67.64% (31) | 3.6754 (39) | 7.7561 (38) |
| Forward 1-year RAPM-prior baseline | 39 | 1.198196 (36) | 1.141757 (36) | 0.0861% (37) | 14.5265 (40) | 12.2406% (40) | 14.8154 (40) | 65.39% (35) | 4.2374 (40) | 9.1066 (39) |
| Forward 3-year RAPM-prior baseline | 40 | 1.198246 (37) | 1.141950 (37) | 0.0778% (38) | 14.6559 (41) | 10.6699% (41) | 14.9714 (41) | 64.68% (36) | 4.5326 (41) | 9.7789 (40) |

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
| [NAIL-RAPM v1.2.4 free-throw replacement (not promoted)](nail-rapm-v124-free-throw-replacement.md) | 8 | 1.192706 (8) | 1.137567 (2) | 0.0652% (8) | 16.5988 (17) | 7.6880% (17) |
| [NAIL-RAPM additive-only context](nail-additive-only-context.md) | 9 | 1.192713 (13) | 1.137590 (5) | 0.0640% (13) | 16.5724 (9) | 7.9812% (8) |
| [NAIL-RAPM v1.2.1.4 player-trend aging](nail-rapm-v1214-player-trend-aging.md) **(Promotion eligible; not deployed)** | 10 | 1.192716 (15) | 1.137608 (10) | 0.0639% (14) | 16.5704 (8) | 7.9251% (9) |
| [NAIL-RAPM v1.2.1.1 standard USG% (not promoted)](nail-rapm-v1211-standard-usage.md) | 10 | 1.192708 (10) | 1.137593 (6) | 0.0649% (10) | 16.5989 (18) | 7.6873% (18) |
| [NAIL-RAPM v1.2.2 defensive-rebound profile (not promoted)](nail-rapm-v122-defensive-rebound-profile.md) | 11 | 1.192702 (7) | 1.137609 (11) | 0.0659% (7) | 16.5875 (12) | 7.8132% (12) |
| [NAIL-RAPM v1.4 Kalman additive profiles (not promoted)](nail-rapm-v14-filtered-additive-profiles.md) | 11 | 1.192674 (6) | 1.137670 (24) | 0.0706% (6) | 16.5836 (11) | 7.8566% (11) |
| [NAIL-RAPM v1.2.1 pruned non-additive context](nail-rapm-v121-pruned-nonadditive.md) | 11 | 1.192709 (11) | 1.137608 (10) | 0.0647% (11) | 16.5942 (14) | 7.7392% (14) |
| [NAIL Critical Spacing candidate](nail-critical-spacing.md) | 12 | 1.192706 (9) | 1.137609 (12) | 0.0652% (9) | 16.5896 (13) | 7.7902% (13) |
| [NAIL-RAPM v1.2.1.2 back-to-back schedule control](nail-rapm-v1212-back-to-back.md) **(Prior production)** | 12 | 1.192710 (12) | 1.137604 (8) | 0.0645% (12) | 16.6032 (19) | 7.6393% (19) |
| [NAIL quartile Critical Spacing plus standard USG% (not promoted)](nail-critical-spacing-quartile-standard-usage.md) | 13 | 1.192713 (13) | 1.137596 (7) | 0.0640% (13) | 16.6145 (22) | 7.5130% (22) |
| [Compiled-additive HPM x3 plus quadratic side context](linear-hpm-x3-quadratic-side-context.md) | 15 | 1.192719 (17) | **1.137554 (1)** | 0.0630% (17) | 16.5978 (15) | 7.6986% (15) |
| [NAIL-RAPM v1.2.1.3 residualized-target lambda CV](nail-rapm-v1213-residualized-lambda.md) **(Production)** | 15 | 1.192717 (16) | 1.137624 (15) | 0.0633% (16) | 16.5784 (10) | 7.9150% (10) |
| [NAIL-RAPM v1.2.3 free-throw profile (not promoted)](nail-rapm-v123-free-throw-profile.md) | 15 | 1.192714 (14) | 1.137583 (4) | 0.0638% (15) | 16.6157 (24) | 7.5004% (24) |
| [NAIL lead-handler allocation candidate](nail-lead-secondary-usage-gap.md) | 17 | 1.192730 (20) | 1.137627 (17) | 0.0611% (20) | 16.4990 (3) | 8.7944% (3) |
| [NAIL-RAPM normalized context penalty](nail-context-regularization.md) | 18 | 1.192726 (18) | 1.137680 (25) | 0.0618% (18) | 16.5979 (16) | 7.6979% (16) |
| [NAIL-RAPM v1.2 gap-returner priors](nail-rapm-v12-gap-returners.md) | 21 | 1.192734 (21) | 1.137641 (19) | 0.0605% (21) | 16.6148 (23) | 7.5097% (23) |
| [NAIL-RAPM fixed context alpha=5,000](nail-context-regularization.md) | 21 | 1.192727 (19) | 1.137753 (34) | 0.0617% (19) | 16.6090 (21) | 7.5741% (21) |
| [NAIL-RAPM fixed context alpha=20,000](nail-context-regularization.md) | 22 | 1.192736 (22) | 1.137655 (22) | 0.0602% (22) | 16.6187 (25) | 7.4664% (25) |
| [NAIL-RAPM v1.1 stat-specific padding](nail-rapm-v11-profile-padding.md) | 23 | 1.192740 (23) | 1.137655 (22) | 0.0595% (23) | 16.6267 (27) | 7.3771% (27) |
| [NAIL token-MLP residual](nail-token-residual.md) | 25 | 1.192747 (25) | 1.137607 (9) | 0.0583% (25) | 16.6393 (30) | 7.2373% (30) |
| [NAIL-RAPM fixed context alpha=1,000](nail-context-regularization.md) | 26 | 1.192745 (24) | 1.137743 (33) | 0.0587% (24) | 16.6197 (26) | 7.4553% (26) |
| [NAIL-RAPM v1.0](nail-rapm-v1.md) | 26 | 1.192752 (26) | 1.137640 (18) | 0.0574% (26) | 16.6636 (32) | 6.9658% (32) |
| HPM v2 shooting composition | 27 | 1.192759 (27) | 1.137652 (20) | 0.0564% (27) | 16.6913 (36) | 6.6559% (36) |
| Complete player-prior RAPM, no context or box score | 29 | 1.192761 (29) | 1.137741 (32) | 0.0559% (29) | 16.6068 (20) | 7.5991% (20) |
| [HPM x2 raw OREB/100 context](hpm-x2.md) | 29 | 1.192760 (28) | 1.137716 (30) | 0.0562% (28) | 16.6382 (29) | 7.2857% (29) |
| [HPM x1 ORB claim context](hpm-x1.md) | 29 | 1.192763 (30) | 1.137713 (29) | 0.0557% (30) | 16.6315 (28) | 7.3242% (28) |
| [NAIL Set Attention residual](nail-token-residual.md) | 31 | 1.192768 (31) | 1.137571 (3) | 0.0547% (31) | 16.7235 (39) | 6.2956% (39) |
| [HPM v2.3 shot portfolio](hpm-v23.md) | 31 | 1.192777 (33) | 1.137641 (19) | 0.0534% (33) | 16.6546 (31) | 7.0659% (31) |
| HPM v2.1 empirical rebound capacity | 32 | 1.192772 (32) | 1.137654 (21) | 0.0542% (32) | 16.6701 (33) | 6.8932% (33) |
| [HPM x3 ORB claim rebound replacement](hpm-x3.md) | 34 | 1.192781 (34) | 1.137688 (27) | 0.0526% (34) | 16.6792 (34) | 6.7912% (34) |
| HPM v2.2 usage allocation | 35 | 1.192807 (36) | 1.137656 (23) | 0.0482% (36) | 16.6806 (35) | 6.7759% (35) |
| Value-Conditioned Aging HPM | 35 | 1.192792 (35) | 1.137701 (28) | 0.0508% (35) | 16.7155 (37) | 6.3851% (37) |
| [Split NAIL-RAPM constrained O/D decomposition (not promoted)](split-nail-rapm.md) | 37 | 1.194415 (40) | 1.140791 (37) | -0.2214% (40) | 16.5301 (7) | 8.4506% (7) |
| [Additive prior plus linear non-additive context](additive-profile-linear-shape-context.md) | 37 | 1.192814 (37) | 1.137613 (13) | 0.0470% (37) | 16.7224 (38) | 6.3076% (38) |
| Forward 3-year RAPM-prior baseline | 38 | 1.193041 (38) | 1.137974 (36) | 0.0091% (38) | 17.1808 (40) | 1.1016% (40) |
| Forward 1-year RAPM-prior baseline | 39 | 1.193123 (39) | 1.137869 (35) | -0.0047% (39) | 17.2902 (41) | -0.1627% (41) |

NAIL-RAPM v1.3.1 is the preferred, parsimonious version of the non-promoted
v1.3 branch: it passed its direct no-material-harm gate versus v1.3, but it
does not supersede the selected v1.2.1.3 model as the global regular-season
release. See its
[experiment record](nail-rapm-v131-pruned-additive-profiles.md).

See [Forward RAPM Memory Baselines](forward-rapm-memory-baselines.md) and
[Complete Player-Prior RAPM Baseline](complete-player-prior-baseline.md) for
model specifications, annual lambda selections, and immutable artifacts.
