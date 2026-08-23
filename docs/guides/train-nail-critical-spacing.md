---
last_updated: "2026-08-23"
---

# Train the NAIL Critical-Spacing Candidate

This controlled candidate starts with production NAIL-RAPM v1.2.1 and adds one
non-additive five-man feature, `critical_spacing`. No player-prior, additive
profile, padding, aging, cold-start, or Ridge regularization settings change.

For every seasonal context state, the model finds the lower tercile of shrunk
`three_pm_per_100` across the profiles available at that point in the recursive
history. A unit receives `critical_spacing = 1` when at least two players are
strictly below that cutoff; otherwise it receives `0`. Thus the cutoff is
season-specific and forward-safe, while the activation rule is fixed before
evaluation.

```bash
uv run nba-train-nail-critical-spacing \
  --log-path artifacts/logs/nail-critical-spacing.log
```

Follow recursive seasonal progress:

```bash
tail -f artifacts/logs/nail-critical-spacing.log
```

After the completed recursive run, replay the three frozen evaluation seasons
and run the paired bootstrap against v1.2.1:

```bash
uv run nba-evaluate-nail-critical-spacing \
  --log-path artifacts/logs/nail-critical-spacing-frozen.log
uv run nba-bootstrap-nail-critical-spacing
```
