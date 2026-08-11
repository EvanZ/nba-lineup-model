---
last_updated: "2026-08-11"
---

# Train Value HPM With Playoffs

This HIPSTER PM variant adds each completed historical season's playoff
stints to that season's regular-season fit. Those outcomes therefore update
the state used for the following season. The target season's regular season
and playoffs remain frozen evaluation cohorts and are never used to form their
own forecast state.

Only seasons with validated playoff stints are augmented. The resulting
`season_model_metadata.parquet` records the exact coverage, including every
season that remains regular-season-only because its playoff stints have not
yet been reconstructed to the same contract.

```bash
uv run nba-train-forward-centered-value-conditioned-aging-playoffs-bounded-hierarchical-portable-matchup-contextual-rapm \
  --through-season 2025-26
```

The model retains the value-conditioned aging prior, exposure-gated cold
starts, bounded hierarchical P-splines, and attributable portable-matchup
context from the regular-only HPM.

## Outputs

```text
artifacts/models/forward_centered_value_conditioned_aging_playoffs_bounded_hierarchical_portable_matchup_contextual_rapm/2025-26/
```

`season_model_metadata.parquet` records regular-season stint counts, included
playoff games, excluded playoff games, and the total fitting rows for every
season. The target row has zero included playoff games by design.
