---
last_updated: "2026-08-05"
---

# Exposure-Gated Cold-Start Prior

This frozen preseason prior applies only to first-NBA-season players. It keeps
the completed 2024-25 lagged RAPM state for returning players and zero for
other no-prior players. For every first-year player, it combines the
[draft RAPM rate](draft-prior.md) with the [cold-start exposure gate](cold-start-exposure.md)
and the historical pooled replacement-token estimate:

\[
\widehat R_i^{cold} =
p_i^{low}\,R^{replacement}
+(1-p_i^{low})\,\widehat R_i^{draft}.
\]

Here \(p_i^{low}\) is the probability of finishing under 5% of team possession
opportunities, \(R^{replacement}\) is the pooled low-exposure token RAPM, and
\(\widehat R_i^{draft}\) is the draft-profile rate estimate. This is a
continuous mixture, not a hard player classification.

## Frozen 2025-26 Inputs

The revised prior is
`artifacts/models/exposure_gated_cold_start/2025-26/exposure-gated-cold-start-2025-26-20260806T014656Z-cdafdec8/`.
All inputs end with 2024-25:

| Component | Frozen source | Value / role |
| --- | --- | --- |
| Draft rate | 1996-97 to 2024-25 first-year players | Player-specific \(\widehat R_i^{draft}\) |
| Exposure gate | Same historical cutoff | Player-specific \(p_i^{low}\) |
| Replacement token | 1996-97 to 2024-25 regular seasons | \(R^{replacement}=-4.740\) RAPM |

The frozen evaluation is
`artifacts/models/frozen_prior_evaluation/2025-26/frozen-exposure-gated-cold-start-prior-2025-26-20260806T015348Z-089d2a36/`.
It evaluates realized 2025-26 lineup exposure, but does not use target-season
scores, possessions, player coefficients, or playoff outcomes to form priors.

## Results

| Cohort | Possession RMSE | Game-margin RMSE | Team NetRtg RMSE | Pythagorean win RMSE |
| --- | ---: | ---: | ---: | ---: |
| Regular season | **1.198952** | **14.7413** | **4.6989** | **10.2466** |
| Playoffs | 1.192980 | 17.6725 | - | - |

Against the prior draft-cold-start ablation, the blend improves every listed
regular-season metric. It is slightly worse in the 85-game playoff sample, so
the [Frozen Preseason Leaderboard](preseason-leaderboard.md) continues to show
both cohorts separately.

## Revised 2025-26 Rookie Rankings

The sortable table ranks the blended preseason RAPM prior, not observed
2025-26 performance. It exposes both rate and exposure components so the
change from the original draft-only ranking is inspectable. The full 100-player
table is `revised_rookie_rankings.parquet` in the immutable prior run.

### Top 25 Exposure-Gated Rankings

| Rank | Player | Pos. | Pick | Draft rate | P(low exposure) | Blended RAPM prior |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | Collin Murray-Boyles | F | 9 | -0.60 | 5.6% | -0.83 |
| 2 | Cedric Coward | G | 11 | -0.56 | 6.9% | -0.85 |
| 3 | Dylan Harper | G | 2 | -0.85 | 2.2% | -0.94 |
| 4 | Derik Queen | C | 13 | -0.55 | 9.3% | -0.94 |
| 5 | Kon Knueppel | G-F | 4 | -0.83 | 3.0% | -0.95 |
| 6 | VJ Edgecombe | G | 3 | -0.90 | 2.5% | -1.00 |
| 7 | Walter Clayton Jr. | G | 18 | -0.50 | 12.0% | -1.01 |
| 8 | Tre Johnson | G | 6 | -0.87 | 3.8% | -1.02 |
| 9 | Egor Demin | G | 8 | -0.81 | 5.6% | -1.03 |
| 10 | Cooper Flagg | F | 1 | -1.00 | 2.4% | -1.08 |
| 11 | Carter Bryant | F | 14 | -0.74 | 8.8% | -1.09 |
| 12 | Khaman Maluach | C | 10 | -0.77 | 8.1% | -1.10 |
| 13 | Yang Hansen | C | 16 | -0.47 | 14.8% | -1.10 |
| 14 | Ace Bailey | F | 5 | -0.96 | 3.9% | -1.11 |
| 15 | Jeremiah Fears | G | 7 | -1.02 | 3.5% | -1.14 |
| 16 | Noa Essengue | F | 12 | -0.90 | 8.3% | -1.22 |
| 17 | Nolan Traore | G | 19 | -0.79 | 12.5% | -1.28 |
| 18 | Nique Clifford | G | 24 | -0.47 | 19.2% | -1.29 |
| 19 | Kasparas Jakucionis | G | 20 | -0.73 | 14.8% | -1.32 |
| 20 | Joan Beringer | F | 17 | -0.74 | 15.2% | -1.34 |
| 21 | Drake Powell | G-F | 22 | -0.72 | 17.5% | -1.42 |
| 22 | Jase Richardson | G | 25 | -0.75 | 18.1% | -1.48 |
| 23 | Will Riley | F | 21 | -0.72 | 20.9% | -1.56 |
| 24 | Asa Newell | F | 23 | -0.61 | 23.6% | -1.58 |
| 25 | Danny Wolf | F | 27 | -0.39 | 28.4% | -1.63 |

Rocco Zikarsky falls toward the bottom of the revised table because the gate
assigns him a 71.7% low-exposure probability. That is the intended correction:
draft-rate estimates no longer treat an exposure-risk player as a favorable
one-number cold start.

## Artifacts

| File | Contents |
| --- | --- |
| `revised_rookie_rankings.parquet` | Draft rate, gate probabilities, replacement value, and blended prior |
| `metadata.json` / `manifest.json` | Three source runs, temporal boundary, and integrity hashes |
| Frozen evaluation artifacts | Possession, game, team NetRtg, and win metrics with the exact prior vector |
