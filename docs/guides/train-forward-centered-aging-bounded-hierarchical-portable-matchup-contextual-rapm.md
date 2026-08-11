---
last_updated: "2026-08-09"
---

# Train Centered Aging Portable Contextual RAPM

This candidate applies an explicit pre-season location constraint to the
age-informed bounded hierarchical portable-matchup contextual model:

```bash
uv run nba-train-forward-centered-aging-bounded-hierarchical-portable-matchup-contextual-rapm \
  --through-season 2025-26
```

## Centering Rule

For every target season (t), the model first builds the usual full prior vector
for returning and cold-start players. It then subtracts the prior-season
possession-weighted prior mean:

\[
\widetilde{\mu}_{i,t} = \mu_{i,t} -
\frac{\sum_j p_{j,t-1}\mu_{j,t}}{\sum_j p_{j,t-1}}.
\]

Here (p_{j,t-1}) is the completed prior-season on-court possession exposure.
Players without such exposure shift with the vector but receive zero weight in
the reference calculation. The initial historical season uses a uniform
reference because no completed NBA season precedes it.

The shift is common to every player, so it cancels from a five-versus-five RAPM
row. It establishes a meaningful reference for the recursive state without
using target-season outcomes.

## Outputs

The model writes the standard portable contextual artifact below:

```text
artifacts/models/forward_centered_aging_bounded_hierarchical_portable_matchup_contextual_rapm/2025-26/
```

`season_player_prior_metadata.parquet` records the centering method, offset,
and total reference exposure for each target season. Compare frozen metrics to
the uncentered aging candidate before promoting the centered state.
