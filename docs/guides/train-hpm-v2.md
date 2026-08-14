---
last_updated: "2026-08-13"
---

# Train HPM v2

HPM v2 is the depth-aware shooting feature ablation of Value-Conditioned Aging
HIPSTER PM. It retains the full forward player-prior and bounded hierarchical
portable-matchup context contract while replacing the raw three-point lineup
totals with depth, capped capacity, and concentration features.

```bash
uv run nba-train-hpm-v2 --through-season 2025-26 \
  2>&1 | tee artifacts/logs/train-hpm-v2-2025-26.log
```

Follow a running job with:

```bash
tail -f artifacts/logs/train-hpm-v2-2025-26.log
```

The immutable output is written below:

```text
artifacts/models/forward_hpm_v2_depth_aware_shooting/2025-26/
```

Each run includes `season_context_metadata.parquet`, which records the feature
set and its exact column list for every seasonal context fit. The job writes
`cohort_metrics.parquet`, game and possession predictions, player ratings,
seasonal context models, and a manifest after the full recursive run succeeds.
