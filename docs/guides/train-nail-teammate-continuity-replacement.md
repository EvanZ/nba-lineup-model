---
last_updated: "2026-08-30"
---

# Train the NAIL Teammate-Continuity Replacement Candidate

This controlled candidate keeps the production NAIL-RAPM v1.2.1.3 contract
fixed except for one non-additive substitution:

\[
\{\text{usage concentration},\ \text{top-two assists}\}
\longrightarrow
\{\text{usage concentration},\ \text{prior teammate continuity}\}.
\]

Run the full recursive fit through 2025-26:

```bash
uv run python -m nba_lineup_model.modeling.forward_nail_teammate_continuity_replacement \
  --through-season 2025-26 \
  --log-path artifacts/logs/nail-teammate-continuity-replacement.log
```

Follow progress:

```bash
tail -f artifacts/logs/nail-teammate-continuity-replacement.log
```

Evaluate the three frozen seasons on support shared by the incumbent and
candidate:

```bash
uv run python -m nba_lineup_model.modeling.nail_teammate_continuity_replacement_frozen_backtest \
  --log-path artifacts/logs/nail-teammate-continuity-replacement-frozen.log
```

Render the complete two-feature coefficient audit:

```bash
uv run python -m nba_lineup_model.modeling.nail_teammate_continuity_replacement_weight_audit
```

Run the paired 10,000-draw game-block bootstrap gate:

```bash
uv run python -m nba_lineup_model.modeling.nail_teammate_continuity_replacement_bootstrap
```

The replacement fit persists `season_prior_teammate_pair_exposures.parquet`.
For target season \(t\), every continuity input is rebuilt exclusively from
same-unit shared possessions in regular season \(t-1\). The candidate never
uses target-season teammate exposure to predict that season.
