---
last_updated: "2026-08-07"
---

# Evaluate Frozen No-Prior Window RAPM

Build the zero-prior controls used to compare a single completed season with a
pooled three-season window. Both commands fit regular-season stints only,
select lambda chronologically inside their completed window, and evaluate the
fixed coefficients on the entire 2025-26 regular season and playoffs.

```bash
uv run nba-evaluate-frozen-one-year-rapm --season 2025-26
uv run nba-evaluate-frozen-three-year-rapm --season 2025-26
```

The first command trains on 2024-25. The second trains on 2022-23 through
2024-25. Neither command accepts a player prior or uses 2025-26 outcomes for
training, lambda selection, or player-coefficient fitting.

The fit artifacts are written to `artifacts/models/frozen_one_year_rapm/` and
`artifacts/models/frozen_three_year_rapm/`. Their immutable evaluation outputs
are written to `artifacts/models/frozen_prior_evaluation/`, alongside the other
frozen preseason candidates. The completed run is also recorded in MLflow.

See [Frozen No-Prior Window RAPM](../models/frozen-window-rapm.md) for the
model contract and [Frozen Preseason Leaderboard](../models/preseason-leaderboard.md)
for the common evaluation table.
