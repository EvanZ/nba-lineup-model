# Build The Draft-Prior Study

This diagnostic uses the validated player-season panel to estimate a simple,
interpretable first-NBA-season RAPM prior from draft profile fields. It is not
a lineup model and does not modify the frozen preseason Leaderboard.

```bash
uv run nba-build-draft-prior-study --season 2025-26
```

The default run trains on first-NBA-season players through 2024-25, chooses the
ridge regularization through expanding season-level folds, and scores the
2025-26 first-NBA-season cohort using profile fields only. It never passes
2025-26 RAPM or possession labels to the target profile table.

## Outputs

Each run is immutable:

```text
artifacts/models/draft_prior/<season>/<run_id>/
```

| File | Purpose |
| --- | --- |
| `training_first_nba_season_players.parquet` | Historical labels and model inputs through the cutoff |
| `target_first_nba_season_players.parquet` | Target-season profile inputs with RAPM and exposure columns removed |
| `cross_validation.parquet` | Expanding-fold ridge selection results |
| `regularization_stability.parquet` | Expanding-cutoff history of selected ridge regularization |
| `empirical_draft_curve.parquet` | Possession-weighted observed RAPM by draft tier |
| `adjusted_draft_curve.parquet` | Pick 1-60 partial curve with season-block bootstrap bands |
| `rookie_rankings.parquet` | Scores for the target first-NBA-season class |
| `model.joblib` | Fitted standardized ridge pipeline |
| `draft-prior-curve.svg` | Published diagnostic graphic |
| `metadata.json` / `manifest.json` | Cutoff, feature, hash, and artifact-integrity contract |

The current curve SVG is copied to
`docs/assets/images/draft-prior/draft-prior-curve.svg`; rebuilding the study
updates that documentation asset while retaining prior immutable runs.

See [Draft-Informed Cold Starts](../models/draft-prior.md) for the model,
conditioning caveat, and published rankings.
