---
last_updated: "2026-08-23"
---

# Train the NAIL Lower-Quintile Critical-Spacing Candidate

This controlled candidate differs from the lower-tercile Critical Spacing
experiment only in its forward-safe threshold. For each seasonal context state,
the model identifies the bottom 20% of shrunk `three_pm_per_100` profiles. A
unit receives `critical_spacing = 1` when at least two players fall strictly
below that lower-quintile cutoff.

```bash
uv run nba-train-nail-critical-spacing-quintile \
  --log-path artifacts/logs/nail-critical-spacing-quintile.log
```

Follow recursive seasonal progress:

```bash
tail -f artifacts/logs/nail-critical-spacing-quintile.log
```

After training, run the matched three-season frozen comparison:

```bash
uv run nba-evaluate-nail-critical-spacing-quintile \
  --log-path artifacts/logs/nail-critical-spacing-quintile-frozen.log
```

Publish the full three-term non-additive diagnostic chart after the recursive
fit. The audit includes `usage_concentration`, `top_two_assists`, and the new
quintile Critical Spacing coefficient, so it can detect displacement of either
retained term.

```bash
uv run nba-audit-nail-critical-spacing-quintile-weights
```
