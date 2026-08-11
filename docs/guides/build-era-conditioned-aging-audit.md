---
last_updated: "2026-08-10"
---

# Audit Era-Conditioned Aging

This no-RAPM-refit diagnostic explains the Era-Conditioned Aging HPM result.
It reconstructs the terminal value-conditioned aging pipeline from its
immutable historical coefficient state, loads the terminal era-conditioned
pipeline, and compares their population partial-age-effect curves.

```bash
uv run nba-build-era-conditioned-aging-audit --season 2025-26
```

The command writes an immutable audit under
`artifacts/analysis/era_conditioned_aging/2025-26/`, including the curve
comparison, possession-weighted veteran-cohort prior errors, and a rendered
curve-delta figure. It refreshes the analysis section of
[Era-Conditioned Aging HPM](../models/forward-centered-era-conditioned-aging-bounded-hierarchical-portable-matchup-contextual-rapm.md).

The cohort error table is diagnostic only: it compares each frozen player prior
with its own completed-season RAPM refit for returning players. It is not a new
preseason leaderboard measure and does not use 2025-26 outcomes to construct
the frozen priors.
