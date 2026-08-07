---
last_updated: "2026-08-05"
---

# Forward Exposure-Gated RAPM

This is the recursive one-number preseason RAPM state. Each completed regular
season is fit with prior-centered ridge RAPM. Its resulting player values form
the returning-player state for the next season. First-NBA-season players
receive the exposure-gated draft/replacement prior only when the necessary
components can be estimated from earlier completed seasons.

For target season \(t\), all three cold-start pieces use data through
\(t-1\): the draft-rate ridge, the first-year low-exposure logistic gate, and
the equal-season mean of tokenized replacement RAPM estimates. No future
season is used to form a historical prior.

## Current State

The completed run through 2025-26 is
`artifacts/models/forward_exposure_gated_rapm/2025-26/forward-exposure-gated-rapm-2025-26-20260806T030613Z-cddcd0ae/`.
It publishes 2026-27 returning-player priors from the completed 2025-26 fit.
The direct Draft History and active-roster pipeline supplies an external rookie
profile table before the new class enters the player-season panel.

The [2026-27 Exposure-Gated Player Rankings](2026-27-exposure-gated-rankings.md)
page publishes the sortable top 100 returning-player table. The direct NBA
Draft History ingestion now supports a separate sortable table for the drafted
class: [2026-27 Drafted Rookie Rankings](2026-27-draft-history-rankings.md).
That table is now scored from this completed forward state through 2025-26,
not from the earlier standalone cold-start artifact.
The frozen 2025-26 rookie branch is evaluated separately in
[Forward Cold-Start Validation](forward-cold-start-validation.md).

## Frozen 2025-26 Evaluation

The 2025-26 prior was frozen after 2024-25 and then evaluated on the same
regular-season and playoff cohorts as the Frozen Preseason Leaderboard.

| Cohort | Possession RMSE | Possession MAE | Game-margin RMSE | Team NetRtg RMSE | Pythagorean win RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Regular season | 1.198993 | 1.142055 | 14.8225 | **4.6680** | 10.2778 |
| Playoffs | 1.192821 | **1.136096** | 17.4188 | - | - |

The simpler frozen exposure-gated cold-start model remains better on the main
regular-season possession and game-margin metrics. This recursive model is
therefore the leading production-state candidate, not a promoted sole gold
standard yet. It needs additional target-season holdouts before that choice is
settled.

## Published Outputs

| File | Contents |
| --- | --- |
| `next_season_top_100_returning_rankings.parquet` | Top 100 completed-2025-26 RAPM values used as 2026-27 returning priors |
| `next_season_returning_rankings.parquet` | Full returning-player state, including 2025-26 possession exposure |
| `season_player_priors.parquet` | The strictly forward prior vector entering every historical season |
| `season_cold_start_metadata.parquet` | Per-season gate, draft-rate, and replacement-token settings |
| `season_replacement_tokens.parquet` | One completed-season token estimate per season |
| Frozen evaluation files | 2025-26 possession, game, team NetRtg, and Pythagorean-win results |
