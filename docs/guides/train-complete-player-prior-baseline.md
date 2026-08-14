---
last_updated: "2026-08-13"
---

# Train Complete Player-Prior RAPM Baseline

Run the recovered-coverage, three-season player-only control:

```bash
uv run nba-train-complete-player-prior-baseline
```

It rebuilds seasonal replacement tokens, rookie draft-rate and exposure-gate
models, value-conditioned aging transitions, centered player priors, and
annual prior-centered RAPM states through 2025-26. It freezes forecasts for
2023-24, 2024-25, and 2025-26 before their respective annual updates.

Artifacts are stored beneath
`artifacts/models/forward_complete_player_prior_rapm/`. The command updates
[Complete Player-Prior RAPM Baseline](../models/complete-player-prior-baseline.md).
