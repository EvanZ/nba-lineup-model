---
last_updated: "2026-08-22"
---

# Train NAIL-RAPM v1.2.4 Free-Throw Replacement

v1.2.4 replaces the v1.2.1 additive `usage_per_100` coordinate with padded
`free_throw_attempts_per_100`; it does not add a ninth profile coordinate.

```bash
uv run nba-train-nail-v124-free-throw-replacement \
  --log-path artifacts/logs/nail-v124-free-throw-replacement.log
```

Follow the recursive fit:

```bash
tail -f artifacts/logs/nail-v124-free-throw-replacement.log
```

After the fit completes:

```bash
uv run nba-evaluate-nail-v124-free-throw-replacement \
  --log-path artifacts/logs/nail-v124-free-throw-replacement-frozen.log
uv run nba-bootstrap-nail-v124-free-throw-replacement --draws 2000
uv run nba-audit-nail-v124-free-throw-replacement-weights
```
