---
last_updated: "2026-08-01"
---

# Train Blended Prior RAPM

This model chooses a convex combination of frozen lagged-RAPM and aging-model
prior means, jointly with the usual prior-centered RAPM penalty.

```bash
uv run nba-train-blended-prior-rapm \
  --season 2025-26 \
  --aging-run-id aging-2025-26-20260801T220356Z-4de5f001 \
  --lagged-run-id forward-lagged-rapm-2025-26-20260801T012439Z-3fe31ed6
```

The default lagged-weight grid is `0, 0.25, 0.5, 0.75, 1`. A weight of `1`
uses only lagged RAPM; `0` uses only the aging forecast. Selection pools
chronological validation MSE across the target-season folds. The final regular
holdout and playoff evaluation use the frozen first-1,044-game coefficient
state.

Runs are written to:

```text
artifacts/models/blended_prior_rapm/{season}/{run_id}/
```

`blend_selection.parquet` records every weight/lambda/fold result;
`blended_player_priors.parquet` records the selected player means; and the
frozen coefficient, holdout, playoff, metadata, and manifest artifacts follow
the age-informed prior RAPM contract.
