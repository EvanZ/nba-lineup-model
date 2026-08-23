---
last_updated: "2026-08-22"
---

# Train NAIL-RAPM v1.2.1 Pruned Non-Additive Context

v1.2.1 retains only `usage_concentration` and `top_two_assists` from the six
v1.2 non-additive context terms. All additive player-profile totals and player
prior behavior remain identical to v1.2.

```bash
uv run nba-train-nail-v121-pruned-nonadditive \
  --log-path artifacts/logs/nail-v121-pruned-nonadditive.log
```

Follow recursive seasonal progress:

```bash
tail -f artifacts/logs/nail-v121-pruned-nonadditive.log
```

Run the frozen replay and paired bootstrap against v1.2:

```bash
uv run nba-evaluate-nail-v121-pruned-nonadditive \
  --log-path artifacts/logs/nail-v121-pruned-nonadditive-frozen.log
uv run nba-bootstrap-nail-v121-pruned-nonadditive
```

Build the focused two-term coefficient audit used on the model page:

```bash
uv run nba-audit-nail-v121-pruned-nonadditive-weights
```
