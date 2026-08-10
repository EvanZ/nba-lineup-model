---
last_updated: "2026-08-08"
---

# Train Forward Hierarchical P-spline Contextual RAPM

    MPLCONFIGDIR=/private/tmp uv run nba-train-forward-hierarchical-pspline-contextual-rapm

This experiment uses the same 1996-97 through 2025-26 forward player-prior
recursion and portable-matchup context contract as the published contextual
model. Its contextual state adds a P-spline curvature penalty and a prior
season function prior. The frozen 2025-26 evaluation is still scored only with
the completed 2024-25 state.

The default penalties are:

| Penalty | Value | Role |
| --- | ---: | --- |
| Ridge level | 10,000 | Shrinks every spline coefficient toward zero. |
| P-spline curvature | 1,000 | Penalizes second differences of each feature's unscaled B-spline coefficients. |
| Temporal hierarchy | 10,000 | Pulls each completed season toward the prior completed season after response-function projection onto the current basis. |

Override an individual value when running a sensitivity experiment:

    MPLCONFIGDIR=/private/tmp uv run nba-train-forward-hierarchical-pspline-contextual-rapm \
      --context-curvature-alpha 1000 \
      --context-temporal-alpha 10000

The immutable artifact is written below
`artifacts/models/forward_hierarchical_pspline_contextual_rapm/2025-26/`.
After a successful run, refresh the shared full-game table:

    MPLCONFIGDIR=/private/tmp uv run nba-report-frozen-game-outcomes --season 2025-26

See [Forward Hierarchical P-spline Contextual RAPM](../models/forward-hierarchical-pspline-contextual-rapm.md)
for the statistical construction and frozen boundary.
