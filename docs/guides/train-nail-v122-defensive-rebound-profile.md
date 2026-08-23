---
last_updated: "2026-08-22"
---

# Train NAIL-RAPM v1.2.2 Defensive-Rebound Profile

v1.2.2 is the strict v1.2.1 contract plus the padded additive
`defensive_rebound_pct` total. It must be compared directly with v1.2.1.

```bash
uv run nba-train-nail-v122-defensive-rebound-profile \
  --log-path artifacts/logs/nail-v122-defensive-rebound-profile.log
```

Follow recursive seasonal progress:

```bash
tail -f artifacts/logs/nail-v122-defensive-rebound-profile.log
```

After the recursive state is complete:

```bash
uv run nba-evaluate-nail-v122-defensive-rebound-profile \
  --log-path artifacts/logs/nail-v122-defensive-rebound-profile-frozen.log
uv run nba-bootstrap-nail-v122-defensive-rebound-profile
```

Build the nine additive and two retained non-additive coefficient panels:

```bash
uv run nba-audit-nail-v122-defensive-rebound-profile-weights
```
