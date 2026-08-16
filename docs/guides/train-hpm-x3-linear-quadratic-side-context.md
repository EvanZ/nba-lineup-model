---
last_updated: "2026-08-15"
---

# Train Linear HPM x3 Quadratic Side Context

Train the recursive linear-plus-quadratic candidate through the completed
2025-26 season:

```bash
uv run nba-train-hpm-x3-linear-quadratic-side-context --through-season 2025-26 \
  2>&1 | tee artifacts/logs/hpm-x3-linear-quadratic-side-context-2025-26.log
```

Follow progress:

```bash
tail -f artifacts/logs/hpm-x3-linear-quadratic-side-context-2025-26.log
```

Then regenerate the shared frozen 2023-24 through 2025-26 evaluation:

```bash
uv run nba-run-frozen-multiseason-backtest
```
