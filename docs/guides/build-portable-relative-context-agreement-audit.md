---
last_updated: "2026-08-09"
---

# Audit Portable and Relative Context Agreement

This no-refit audit compares the completed bounded portable-matchup and
bounded relative-context fits on the same observed 2025-26 regular-season
RAPM stints. It separates the net player-rating edge from the net contextual
edge and weights every summary by stint possessions.

```bash
uv run --group docs nba-build-portable-relative-context-agreement-audit
```

The command writes immutable paired predictions and agreement metrics under
`artifacts/analysis/portable_relative_context_agreement/2025-26/`, updates the
latest pointer, renders aggregate and feature-level correlation figures, and
refreshes the audit section of [Forward Bounded Hierarchical Portable-Matchup
Contextual RAPM](../models/forward-bounded-hierarchical-portable-matchup-contextual-rapm.md).

It is a model-agreement diagnostic, not a predictive leaderboard evaluation:
both models have already fit the 2025-26 season before scoring its observed
lineup matchups.

Feature-level results compare each model's **total** attribution for the same
original contextual feature. In the portable model that is its composition
contribution plus its matchup-residual contribution; it is not a comparison of
the portable-only composition score, which has no counterpart in the relative
model.
