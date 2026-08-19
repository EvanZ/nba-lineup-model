---
title: Audit Frozen vs Completed NAIL Context
---

# Audit Frozen vs Completed NAIL Context

Build the retrospective allocation audit for [NAIL-RAPM v1.0](../models/nail-rapm-v1.md):

```bash
uv run nba-audit-nail-context-stability
```

The default audit scores every observed regular-season stint in 2023-24 through
2025-26 twice: once with the context state available before the target season,
and once with the completed target-season state.

Both scores use the same strictly lagged player profile. Actual target-season
lineups and possessions are used only to aggregate those scores into observed
lineup and player-exposure descriptions.

Use a narrower range or different display thresholds when needed:

```bash
uv run nba-audit-nail-context-stability \
  --seasons 2025-26 \
  --minimum-lineup-possessions 250 \
  --minimum-player-possessions 500
```

The artifact stores `coefficient_stability.parquet`, `stint_nale_stability.parquet`,
`lineup_nale_stability.parquet`, `player_nale_stability.parquet`, and `summary.parquet`.
See [Frozen vs Completed NALE Stability](../models/nail-context-stability.md) for the
published interpretation.
