---
last_updated: "2026-08-05"
---

# Train Forward Exposure-Gated RAPM

Train the recursive regular-season-only state through the latest completed
season:

```bash
uv run nba-train-forward-exposure-gated-rapm --through-season 2025-26
```

The trainer checkpoints after each season at
`artifacts/models/forward_exposure_gated_rapm/<season>/.checkpoint.joblib`.
Re-run the same command after an interrupted process; it resumes at the next
uncompleted season. The checkpoint is removed only after the immutable run is
published.

For controlled short batches, use `--max-seasons 3`. See [Forward
Exposure-Gated RAPM](../models/forward-exposure-gated-rapm.md) for the
temporal contract and current evaluation.
