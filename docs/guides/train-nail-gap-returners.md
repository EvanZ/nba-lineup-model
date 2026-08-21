---
title: Train NAIL Gap-Returner Priors
last_updated: "2026-08-21"
---

# Train NAIL Gap-Returner Priors

Train the NAIL-RAPM v1.2 gap-returner candidate through the final completed
season, then replay the three frozen evaluation seasons and run the paired
bootstrap non-promotion gate.

```sh
uv run nba-train-nail-gap-returners --through-season 2025-26
uv run nba-evaluate-nail-gap-returners
uv run nba-bootstrap-nail-gap-returners
```

The training command writes an immutable recursive run beneath
`artifacts/models/forward_nail_rapm_v12_gap_returner_priors/2025-26/`. It also
persists `gap_returner_projected_states.parquet`, which is an audit of the
internal annual bridges. The rows are not player-season RAPM observations.

The frozen evaluator compares the completed v1.1 state to the candidate on
2023-24 through 2025-26 without updating either model. The bootstrap command
uses game-block resampling within season and requires the candidate's upper 95%
full-game RMSE bound to be no worse than 0.5% of v1.1 in the pooled sample and
each season before it can be promoted.

See [NAIL-RAPM v1.2 Candidate: Gap-Returner Priors](../models/nail-rapm-v12-gap-returners.md)
for the forward contract and results.
