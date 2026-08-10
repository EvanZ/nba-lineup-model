---
last_updated: "2026-08-08"
---

# Train Forward Portable-Matchup Contextual RAPM

    MPLCONFIGDIR=/private/tmp uv run nba-train-forward-portable-matchup-contextual-rapm

This command fits the 1996-97 through 2025-26 recursive RAPM state. It uses
the same profile and cold-start inputs as Forward Contextual RAPM, but carries
an antisymmetric total context \(C\). For every completed season, it also
stores a possession-weighted empirical reference field that identifies
portable unit scores \(h\) and opponent-specific residuals \(q\).

The immutable run is written beneath
artifacts/models/forward_portable_matchup_contextual_rapm/2025-26/.
It contains season_context_models.joblib, forecast_reference_units.parquet,
context metadata, frozen player priors, and regular-season/playoff, team, and
win evaluation outputs.

Run the focused contracts with:

    MPLCONFIGDIR=/private/tmp uv run pytest \
      tests/test_matchup_contextual.py \
      tests/test_forward_portable_matchup_contextual_rapm.py

Refresh the shared full-game outcomes after a successful run:

    MPLCONFIGDIR=/private/tmp uv run nba-report-frozen-game-outcomes --season 2025-26

See [Forward Portable-Matchup Contextual RAPM](../models/forward-portable-matchup-contextual-rapm.md)
for the model equations, frozen information boundary, and result.
