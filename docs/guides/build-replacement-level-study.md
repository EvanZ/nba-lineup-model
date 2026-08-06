# Build the Replacement-Level Study

This command builds the descriptive all-player low-exposure study used to
assess a candidate replacement-level RAPM reference. It does not train or
modify a lineup model.

```bash
uv run nba-build-replacement-level-study --through-season 2025-26
```

The builder reads the validated player-season panel and the matching validated
regular-season RAPM stint partition for every included season. It derives a
player's on-court possession share from full team possession opportunities,
then validates that the resulting on-court totals agree with
`rapm_possessions` in the panel.

Use a different low-exposure reference cutoff only as an explicit sensitivity
analysis:

```bash
uv run nba-build-replacement-level-study \
  --through-season 2025-26 \
  --replacement-share-cutoff 0.05
```

## Outputs

Each run is immutable:

```text
artifacts/models/replacement_level/<through-season>/<run_id>/
```

| File | Purpose |
| --- | --- |
| `player_exposure_cohort.parquet` | All player-season RAPM outcomes joined to computed team-opportunity shares |
| `exposure_band_summary.parquet` | Fixed exposure-band statistics and bootstrap intervals |
| `low_exposure_season_estimates.parquet` | Season-balanced inputs to the candidate reference |
| `low_exposure_experience_summary.parquet` | Low-exposure pool composition and RAPM by career stage |
| `candidate_replacement_prior.json` | Candidate estimate and diagnostic-only status |
| `replacement-level-study.svg` | Documentation diagnostic chart |
| `metadata.json` / `manifest.json` | Reproducibility and integrity records |

The chart is copied to
`docs/assets/images/replacement-level/replacement-level-study.svg`. See
[Replacement-Level Exposure Study](../models/replacement-level.md) for its
interpretation and limitations.
