---
last_updated: "2026-08-13"
---

# Train HPM v2.1

HPM v2.1 retains HPM v2's depth-aware shooting features and replaces its
rebound-volume context with a learned mapping from prior player ORB%/DRB%
claims to realized team rebound rates. Each completed season stores the
calibration used by the next season's contextual state.

```bash
uv run nba-train-hpm-v21 --through-season 2025-26 \
  2>&1 | tee artifacts/logs/train-hpm-v21-2025-26.log
```

Follow a running job with:

```bash
tail -f artifacts/logs/train-hpm-v21-2025-26.log
```

The run derives percentage profiles from cached curated game box scores. It
does not fetch NBA endpoints. Each immutable artifact records the v2.1 feature
set and exact column list in `season_context_metadata.parquet`.

## Audit the Calibration

```bash
uv run nba-audit-rebound-capacity
```

The audit writes central-support response curves and per-season directional
diagnostics under `artifacts/analysis/rebound_capacity_audit/`.
