---
last_updated: "2026-08-13"
---

# Complete Player-Prior RAPM Baseline

This is the recovered-coverage player-only control for HIPSTER PM. It preserves
the full forward player-prior pipeline but disables lineup context and excludes
all box-score prior features.

Returning players receive a regularized aging forecast built from prior RAPM,
prior exposure, age and experience, draft/entry information, physical profile,
and an age-by-prior-RAPM interaction. That final interaction is **value-
conditioned aging**: it allows the expected age transition to vary smoothly
with the player’s previously estimated value. The full preseason prior vector
is then centered using prior-season on-court possessions.

First-year players receive the forward exposure-gated cold-start prior:

\[
\mu_i^{cold}=p_i^{low}R^{replacement}+(1-p_i^{low})R_i^{draft}.
\]

For every season, the draft-rate ridge, exposure gate, replacement token, aging
transition model, and RAPM regularization selection use only earlier completed
recovered regular-season data. The target-season forecast is materialized
before that season's RAPM update. No player score from box-score residuals,
composition functions, or matchup functions enters this model.

<!-- complete-player-prior-results:start -->
## Recovered-Coverage Results

Artifact: `artifacts/models/forward_complete_player_prior_rapm/2023-24_to_2025-26/forward-complete-player-prior-rapm-2023-24-to-2025-26-20260813T210918Z-6520555d`.

| Possession RMSE | Eligible game RMSE | Full-game RMSE | Winner accuracy | Team NetRtg RMSE | Pythagorean-win RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.198061 | 14.2105 | 14.4756 | 67.64% | 3.6754 | 7.7561 |

### Per-Season Regular Results

| Season | Possession RMSE | Eligible game RMSE |
| --- | ---: | ---: |
| 2023-24 | 1.193156 | 13.8723 |
| 2024-25 | 1.202055 | 14.2200 |
| 2025-26 | 1.198795 | 14.4787 |

### Frozen Playoff Check

| Season | Possession RMSE | Eligible game RMSE |
| --- | ---: | ---: |
| 2023-24 | 1.191543 | 16.7687 |
| 2024-25 | 1.194370 | 16.8245 |
| 2025-26 | 1.192351 | 16.2592 |
<!-- complete-player-prior-results:end -->


