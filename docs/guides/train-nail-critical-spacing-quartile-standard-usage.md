---
last_updated: "2026-08-23"
---

# Train NAIL Quartile Critical Spacing Plus Standard USG%

This combined candidate changes two previously tested conventions relative to
NAIL-RAPM v1.2.1: it uses conventional game-level `usage_pct` in place of
`usage_per_100`, and adds a non-additive Critical Spacing indicator. The
indicator is one when two or more players are strictly below the forward-safe,
season-state lower quartile of shrunk `three_pm_per_100`.

```bash
uv run nba-train-nail-critical-spacing-quartile-standard-usage \
  --log-path artifacts/logs/nail-critical-spacing-quartile-standard-usage.log
```

Follow recursive seasonal progress:

```bash
tail -f artifacts/logs/nail-critical-spacing-quartile-standard-usage.log
```

After the recursive fit, run the three frozen seasons and publish the complete
three-term non-additive diagnostic chart:

```bash
uv run nba-evaluate-nail-critical-spacing-quartile-standard-usage \
  --log-path artifacts/logs/nail-critical-spacing-quartile-standard-usage-frozen.log
uv run nba-audit-nail-critical-spacing-quartile-standard-usage-weights
```
