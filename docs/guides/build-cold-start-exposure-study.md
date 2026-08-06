# Build the Cold-Start Exposure Gate

This study predicts whether a first-NBA-season player will finish below a
realized regular-season team-possession share cutoff. It is only a diagnostic
gate: it does not change returning-player priors or fit a lineup model.

```bash
uv run nba-build-cold-start-exposure-study --season 2025-26
```

The default replacement-candidate definition is less than 5% of season-long
team possession opportunities. The command reconstructs exposure from the
existing regular-season RAPM stints, joins draft and player-bio data from the
player-season panel, tunes L2 logistic regularization with six expanding
historical validation seasons, then scores target profiles without retaining
their target outcomes.

Use a different retrospective definition only as a sensitivity analysis:

```bash
uv run nba-build-cold-start-exposure-study \
  --season 2025-26 \
  --replacement-share-cutoff 0.02
```

## Outputs

Every execution creates an immutable run:

```text
artifacts/models/cold_start_exposure/<season>/<run_id>/
```

| File | Purpose |
| --- | --- |
| `training_first_nba_season_exposure.parquet` | Historical labels and features |
| `target_first_nba_season_profiles.parquet` | Target profiles with outcomes removed |
| `cross_validation.parquet` | Regularization search by forward fold |
| `cross_validated_predictions.parquet` | Out-of-fold predictions |
| `calibration_deciles.parquet` | Calibration report |
| `target_exposure_predictions.parquet` | Preseason target scores |
| `metrics.parquet` | Profile model versus constant-rate baseline |
| `cold-start-exposure.svg` | Documentation chart |

The chart is copied to
`docs/assets/images/cold-start-exposure/cold-start-exposure.svg`. See
[Cold-Start Exposure Gate](../models/cold-start-exposure.md) for the model
contract and current results.
