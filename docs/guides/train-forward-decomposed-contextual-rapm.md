---
last_updated: "2026-08-08"
---

# Train Forward Decomposed Contextual RAPM

```bash
MPLCONFIGDIR=/private/tmp uv run nba-train-forward-decomposed-contextual-rapm
```

This command recursively fits the forward exposure-gated RAPM state and an
identifiable contextual side function through 2025-26. Each season's contextual
forecast is `h_(t-1)(home) - h_(t-1)(away)`, using exactly the same player
profiles and composition features as Forward Contextual RAPM. It writes an
immutable run under
`artifacts/models/forward_decomposed_contextual_rapm/2025-26/<run_id>/`.

The run retains `season_context_models.joblib`, context metadata, player
priors, historical coefficients, frozen target-season predictions, and all
frozen evaluation tables. The completed target-season context model is not used
for target-season evaluation.

Run the focused contract tests with:

```bash
MPLCONFIGDIR=/private/tmp uv run pytest \
  tests/test_contextual_features.py \
  tests/test_decomposed_contextual.py \
  tests/test_forward_decomposed_contextual_rapm.py
```

Refresh the shared full-game table after a successful run:

```bash
MPLCONFIGDIR=/private/tmp uv run nba-report-frozen-game-outcomes --season 2025-26
```

The model is currently an interpretability ablation rather than the promoted
predictive benchmark; see [Forward Decomposed Contextual
RAPM](../models/forward-decomposed-contextual-rapm.md) for the exact constraint
and frozen comparison.
