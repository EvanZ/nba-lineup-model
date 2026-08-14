---
last_updated: "2026-08-13"
---

# Train Forward RAPM Memory Baselines

Run both strict rolling RAPM-prior controls and evaluate their frozen forecasts
on 2023-24, 2024-25, and 2025-26:

```bash
uv run nba-train-forward-rapm-memory-baselines
```

The run writes annual player states, frozen target priors, possession and game
forecasts, team net-rating and Pythagorean-win tables, lambda-selection
records, and a manifest beneath
`artifacts/models/forward_rapm_memory_prior_baselines/`.

The command updates [Forward RAPM Memory Baselines](../models/forward-rapm-memory-baselines.md).
Use `--no-docs` for an artifact-only rerun.
