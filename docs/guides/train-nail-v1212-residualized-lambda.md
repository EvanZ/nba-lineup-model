---
last_updated: "2026-08-27"
---

# Train Residualized-Lambda NAIL

This controlled candidate changes only one production NAIL v1.2.1.2 decision:
the source of each season's player Ridge lambda. Production imports the lambda
schedule from the forward exposure-gated RAPM state. This candidate selects
each lambda on chronological folds of the season's already source-context- and
source-B2B-adjusted target.

It retains the production scalar prior, value-conditioned aging, cold starts,
gap-returner bridge, Medvedovsky padding contract, eight additive profile
features, two non-additive features, context alpha, B2B definition/alpha, fit
ordering, season type, and frozen evaluator. The positive grid is local to the
candidate and is written into the per-season CV artifact.

```bash
uv run python -m nba_lineup_model.modeling.forward_nail_v1212_residualized_lambda \
  --log-path artifacts/logs/nail-v1212-residualized-lambda.log
```

Follow the fit:

```bash
tail -f artifacts/logs/nail-v1212-residualized-lambda.log
```

Then run the shared three-season replay:

```bash
uv run python -m nba_lineup_model.modeling.nail_v1212_residualized_lambda_frozen_backtest \
  --log-path artifacts/logs/nail-v1212-residualized-lambda-frozen.log
```

The candidate cleared the agreed frozen comparison and was promoted as
NAIL-RAPM v1.2.1.3 after the ranking and coefficient-history review. Materialize
the same review before any future release-bundle migration:

```bash
uv run python -m nba_lineup_model.modeling.nail_v1213_promotion_review
```

This writes the completed-fit top 25, season-specific HCA and B2B trajectories,
and all ten standardized context coefficient histories. It never refits a
model or evaluates target-season outcomes.
