---
last_updated: "2026-08-12"
---

# Train Forward Box-Score Interaction HPM

Run the recursive interaction candidate through the frozen target season:

```bash
uv run nba-train-forward-box-score-interaction-hpm --through-season 2025-26
```

The command rebuilds leakage-safe lagged player box features, derives the six
declared interaction columns, and fits one HPM plus one annual residual ridge
state for each season. Target-season box scores are never part of that target
season's prior.

## Outputs

```text
artifacts/models/forward_box_score_interaction_hpm/2025-26/
```

The run retains the annual residual ridge models in
`season_box_score_residual_models.joblib` and their expanding-fold selection
details in `season_box_score_residual_selection.parquet`.
