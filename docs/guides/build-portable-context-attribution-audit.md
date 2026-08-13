---
last_updated: "2026-08-11"
---

# Audit Portable Context Attribution

Build exact player-level accounting for selected, realized 2025-26 units using
the frozen 2024-25 Value-Conditioned Aging HPM context state:

```bash
uv run nba-build-portable-context-attribution-audit --through-season 2025-26
```

The audit uses exact Shapley values rather than equal teammate splits. Five
composition contributions reconcile to each side's portable composition score.
Ten matchup contributions reconcile to the opponent-specific residual. The
published [audit page](../models/portable-context-attribution-audit.md)
defines the accounting origin and interpretation boundaries.
