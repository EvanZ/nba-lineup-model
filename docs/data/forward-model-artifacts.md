---
last_updated: "2026-08-10"
---

# Forward Model Artifacts

Every recursive portable-contextual RAPM run is an immutable directory under
`artifacts/models/<model>/<target-season>/<run-id>/`. The manifest hashes each
file, so downstream studies can identify the precise seasonal state that
produced a rating, curve, or forecast.

## Annual Player Ratings

`player_season_ratings.parquet` has one row per fitted player-season. It joins
the fitted RAPM coefficient to the player-season panel and completed in-season
exposure.

| Column group | Examples | Meaning |
| --- | --- | --- |
| identity | `season`, `player_id`, `player_name` | Stable player-season key. |
| rating state | `rapm`, `prior_rapm`, `rapm_adjustment_from_prior`, `selected_lambda` | The fitted coefficient, frozen prior, update from that prior, and season penalty. |
| ranking | `rank_all_players`, `rank_exposure_eligible`, `percentile_all_players` | Deterministic season-local ranks; the exposure rank is null for players below the published threshold. |
| player context | `age`, `nba_experience_years`, `is_rookie` | Known player attributes from the season panel. |
| exposure | `on_court_possessions`, `exposure_share`, `team_count`, `rapm_exposure_eligible` | Completed participation used for interpretation, not a future-season input. |

The row for the terminal season is a completed-season refit. Earlier rows are
the recursive states that informed later priors. A forecast must use the row
from the season immediately preceding its target season.

## Seasonal Fit Metadata

`season_model_metadata.parquet` has one row per recursive seasonal fit. It
records the selected RAPM lambda, prior construction and centering metadata,
contextual penalties, and information boundary. Nested metadata is stored as
JSON text so the table remains portable across Parquet readers.

The `is_frozen_forecast_source_season` flag identifies the completed state used
to forecast the run's `frozen_forecast_target_season`; the terminal
`is_target_completed_refit` row is retained only for retrospective ranking and
diagnostic work.

## Aging Surfaces

Age-informed runs additionally publish:

| File | Grain | Purpose |
| --- | --- | --- |
| `season_aging_models.joblib` | fitted season | Exact fitted `scikit-learn` aging pipeline for reproducible inference. |
| `aging_curve_grid.parquet` | fitted season x age x prior-value profile | Precomputed population aging trajectories at prior-RAPM 25th, 50th, and 75th percentiles. |

Curve grids are diagnostic population surfaces, not individual player
histories. Each holds non-age features at a possession-weighted reference and
reports the partial age effect relative to its recorded reference age. In early
history where age 27 is outside observed training support, the reference age is
clamped to the nearest supported age rather than extrapolating a chart anchor.

`season_context_models.joblib`, `season_player_priors.parquet`, and the frozen
evaluation tables remain the authoritative inputs for replaying lineup scores.
