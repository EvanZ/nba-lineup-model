---
last_updated: "2026-08-07"
---

# Train Forward Contextual RAPM

```bash
MPLCONFIGDIR=/private/tmp uv run nba-train-forward-contextual-rapm
```

This command recursively fits player RAPM and contextual states through
2025-26. It uses the published season-specific forward-RAPM lambda schedule as
a fixed control and carries each completed contextual model forward as the next
season's lineup-level offset. The output is written to
`artifacts/models/forward_contextual_rapm/2025-26/<run_id>/`.

The regression suite includes a synthetic two-season leakage test. It asserts
that the frozen 2025-26 evaluator receives `g_2024-25`, while `g_2025-26` is
only retained as the completed state for the next forecast season:

```bash
MPLCONFIGDIR=/private/tmp uv run pytest tests/test_forward_contextual_rapm.py
```

## Publish The Next-Season Rankings

After a forward contextual run completes, create the immutable completed-state
ranking artifact and refresh the sortable top-100 table on the model page:

```bash
MPLCONFIGDIR=/private/tmp uv run nba-build-forward-contextual-rankings
```

For the current run, these are 2026-27 player priors from the completed
2025-26 additive state. The corresponding `g_2025-26` remains a lineup-level
term rather than a player ranking column.

## Publish The Frozen Lineup Context Case Study

This publishes established 2025-26 five-man units scored only with the frozen
`g_2024-25` contextual state. It writes an immutable artifact and regenerates
the positive and negative examples on the model page:

```bash
MPLCONFIGDIR=/private/tmp uv run nba-build-forward-contextual-case-study
```

The generated case study compares the frozen `g_(t-1)` forecast with the
completed additive and contextual state for the target season. The completed
columns are retrospective, in-sample explanations and are not leaderboard
evaluation metrics.

The default eligibility rule is at least 250 shared possessions and 20 games.
The standardized score uses a possession-weighted distribution of established
opponent units and is not an intrinsic player or lineup value.

The artifact also contains `top_lineup_attribution.parquet` and
`response_curves.parquet`. The latter records the frozen, orientation-
symmetrized spline responses for relative usage and defensive rebounding; the
command publishes its SVG to
`docs/assets/images/forward-contextual-rapm/context-response-curves.svg`.
