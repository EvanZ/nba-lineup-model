---
last_updated: "2026-08-09"
---

# Train Bounded Portable Contextual RAPM

Run the full forward recursive fit:

```bash
uv run nba-train-forward-bounded-hierarchical-portable-matchup-contextual-rapm
```

To retain a progress log while it runs:

```bash
uv run nba-train-forward-bounded-hierarchical-portable-matchup-contextual-rapm \
  > /private/tmp/forward-bounded-portable-contextual-rapm.log 2>&1
tail -f /private/tmp/forward-bounded-portable-contextual-rapm.log
```

The command re-estimates every season from 1996-97 through the requested
target, carrying only completed player and bounded contextual states forward.
The default target is 2025-26.
