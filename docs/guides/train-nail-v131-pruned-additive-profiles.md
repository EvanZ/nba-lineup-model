---
last_updated: "2026-08-22"
---

# Train NAIL-RAPM v1.3.1 Pruned Profiles

This controlled v1.3 ablation removes only `three_pa_per_100` and
`usage_per_100` from the additive profile contract.

```bash
uv run nba-train-nail-v131-pruned-additive-profiles \
  --log-path artifacts/logs/nail-v131-pruned-additive-profiles.log
```

Follow recursive seasonal progress:

```bash
tail -f artifacts/logs/nail-v131-pruned-additive-profiles.log
```

Evaluate the three frozen seasons after training:

```bash
uv run nba-evaluate-nail-v131-pruned-additive-profiles \
  --log-path artifacts/logs/nail-v131-pruned-additive-profiles-frozen.log
```

Run the paired 10,000-draw bootstrap gate against v1.3:

```bash
uv run nba-bootstrap-nail-v131-pruned-additive-profiles
```

Render the all-season ten-panel retained-coefficient audit:

```bash
uv run nba-audit-nail-v131-pruned-additive-weights
```
