---
last_updated: "2026-08-10"
---

# Train Value-Conditioned Aging HPM

This HIPSTER PM candidate retains the centered aging prior and adds a learned,
ridge-penalized interaction between target age and the player's completed
prior-season RAPM:

\[
z_{i,t}=(\operatorname{age}_{i,t}-27)\operatorname{RAPM}_{i,t-1}.
\]

The interaction permits the strictly forward aging model to learn whether
players with different prior values age differently. It does not encode a
special rule for stars: the coefficient is selected jointly with the existing
aging features under expanding-season validation.

```bash
uv run nba-train-forward-centered-value-conditioned-aging-bounded-hierarchical-portable-matchup-contextual-rapm \
  --through-season 2025-26
```

The standard possession-weighted prior centering remains in place, so the new
feature changes relative player forecasts rather than only translating the
coefficient origin.

## Outputs

```text
artifacts/models/forward_centered_value_conditioned_aging_bounded_hierarchical_portable_matchup_contextual_rapm/2025-26/
```

`season_player_prior_metadata.parquet` records the selected aging
regularization and the enabled `age_by_prior_rapm` feature for every season.
