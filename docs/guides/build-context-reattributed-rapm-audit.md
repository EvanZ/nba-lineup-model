---
last_updated: "2026-08-12"
---

# Build Context-Reattributed RAPM Audit

Project the frozen prior-season HPM context signal onto the realized target
season's signed player design, then retain the irreducible lineup-specific
remainder as residual synergy:

```bash
uv run nba-build-context-reattributed-rapm-audit --through-season 2025-26
```

The command writes an immutable artifact under
`artifacts/models/context_reattributed_rapm_audit/2025-26/`. It is an
interpretability audit, not a new leaderboard candidate: the target season's
realized lineups are used to understand an already frozen HPM context state.

The default is 25 possessions, balancing the sparsity of exact ten-player
matchups against interpretability. Use `--minimum-lineup-possessions 75` to
require more evidence before a lineup-pair residual appears in the output
table.
