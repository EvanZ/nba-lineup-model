---
last_updated: "2026-08-25"
---

# Evaluate L5-MMP

Evaluate the five-game median-minutes persistence baseline over regular-season
curated player box scores:

```bash
uv run nba-evaluate-l5-median-minutes-persistence \
  --seasons 2024-25 2025-26
```

The first five team-games for every team-season are excluded because L5-MMP
requires five completed historical allocations. Each run writes per-player
predictions, team-game metrics, aggregate season metrics, and metadata to an
immutable directory below `artifacts/rotation/l5_median_minutes_persistence/`.
