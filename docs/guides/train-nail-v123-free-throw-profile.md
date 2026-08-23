---
last_updated: "2026-08-22"
---

# Train NAIL-RAPM v1.2.3 Free-Throw Profile

v1.2.3 is the v1.2.1 contract plus the padded additive
`free_throw_attempts_per_100` total. It is a sibling of the rejected v1.2.2
DRB% experiment, not an extension of it.

```bash
uv run nba-train-nail-v123-free-throw-profile \
  --log-path artifacts/logs/nail-v123-free-throw-profile.log
```

Follow recursive seasonal progress:

```bash
tail -f artifacts/logs/nail-v123-free-throw-profile.log
```

After the recursive state is complete:

```bash
uv run nba-evaluate-nail-v123-free-throw-profile \
  --log-path artifacts/logs/nail-v123-free-throw-profile-frozen.log
uv run nba-bootstrap-nail-v123-free-throw-profile
```

Build the nine additive and two retained non-additive coefficient panels:

```bash
uv run nba-audit-nail-v123-free-throw-profile-weights
```
