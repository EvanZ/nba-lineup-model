---
last_updated: "2026-08-11"
---

# Train Controlled No-Context RAPM

This is the controlled ablation for HIPSTER PM's recursive contextual state. It
retains the same annual data, published lambda schedule, possession-weighted
prior centering, value-conditioned aging prior, and exposure-gated cold-start
prior as Value-Conditioned Aging HPM. It disables only the prior-season lineup
context correction:

\[
C_{t-1}(U,V)=0.
\]

Each season is still fit recursively, so the resulting player state is the
proper no-context counterfactual rather than an after-the-fact translation of
an older RAPM artifact. The frozen 2025-26 evaluation also uses zero context.

```bash
uv run nba-train-forward-centered-value-conditioned-aging-no-context-rapm \
  --through-season 2025-26
```

## Outputs

```text
artifacts/models/forward_centered_value_conditioned_aging_no_context_rapm/2025-26/
```

The immutable artifact uses the usual frozen possession, game-margin, team
net-rating, and Pythagorean-win evaluation tables. Its `metadata.json` marks
`context_enabled: false`, and `season_context_models.joblib` is intentionally
an empty state.
