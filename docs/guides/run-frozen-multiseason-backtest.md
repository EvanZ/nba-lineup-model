---
last_updated: "2026-08-12"
---

# Run Frozen Multi-Season Backtest

Replay the persisted recursive states for 2023-24 through 2025-26 without
refitting the player or context models:

```bash
uv run nba-run-frozen-multiseason-backtest --seasons 2023-24 2024-25 2025-26
```

The command reads each target-season player-prior vector and the immediately
preceding completed context state from the model's immutable 2025-26 artifact.
It writes an independently versioned report under:

```text
artifacts/models/frozen_multiseason_backtest/2023-24_to_2025-26/
```

The report includes regular-season per-possession, eligible-game, full-game,
team NetRtg, and Pythagorean-win metrics for each replay season, plus pooled
summaries. Historical playoff possession partitions have not yet been
materialized for 2023-24 and 2024-25, so playoff results are intentionally out
of scope for this common comparison.
