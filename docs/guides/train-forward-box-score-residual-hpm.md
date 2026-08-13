---
last_updated: "2026-08-12"
---

# Train Forward Box-Score Residual HPM

Run the recursive candidate through the frozen target season:

```bash
uv run nba-train-forward-box-score-residual-hpm --through-season 2025-26
```

The command rebuilds leakage-safe lagged box-score features from the
player-season panel, then fits one HPM and one residual box-score state for each
season. It does not use target-season box scores to build that season's prior.

## Outputs

```text
artifacts/models/forward_box_score_residual_hpm/2025-26/
```

Alongside standard HPM outputs, each run records
`season_box_score_residual_models.joblib` and
`season_box_score_residual_selection.parquet`, which contain the annual
residual ridge models and their completed-season hyperparameter selections.
