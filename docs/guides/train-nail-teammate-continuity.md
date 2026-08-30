---
last_updated: "2026-08-29"
---

# Train the NAIL Prior Teammate Continuity Candidate

This controlled candidate adds the mean log prior-season shared possessions
over a unit's ten player pairs to the production NAIL context block. Run the
full recursive fit through 2025-26:

```bash
uv run python -m nba_lineup_model.modeling.forward_nail_teammate_continuity \
  --through-season 2025-26 \
  --log-path artifacts/logs/nail-teammate-continuity.log
```

Follow progress:

```bash
tail -f artifacts/logs/nail-teammate-continuity.log
```

Evaluate the three frozen seasons on support shared by the incumbent and
candidate:

```bash
uv run python -m nba_lineup_model.modeling.nail_teammate_continuity_frozen_backtest \
  --log-path artifacts/logs/nail-teammate-continuity-frozen-shared-support.log
```

Render the complete non-additive coefficient audit:

```bash
uv run python -m nba_lineup_model.modeling.nail_teammate_continuity_weight_audit
```

Run the paired 10,000-draw game-block bootstrap gate:

```bash
uv run python -m nba_lineup_model.modeling.nail_teammate_continuity_bootstrap
```

The fit persists `season_prior_teammate_pair_exposures.parquet`, which records
the exact pair-exposure state used by each target season. The frozen evaluator
rebuilds each target's feature from the same strictly prior regular season.
