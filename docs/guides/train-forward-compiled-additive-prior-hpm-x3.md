---
last_updated: "2026-08-16"
---

# Train Forward Compiled-Additive-Prior HPM x3

```bash
uv run nba-train-compiled-additive-prior-hpm-x3 --through-season 2025-26 \
  2>&1 | tee artifacts/logs/compiled-additive-prior-hpm-x3-2025-26.log
```

Follow the run:

```bash
tail -f artifacts/logs/compiled-additive-prior-hpm-x3-2025-26.log
```

The run performs one strictly forward seasonal update at a time. At season
\(t+1\), it uses only the season-\(t\) linear HPM x3 coefficients and lagged
player profiles. After training completes, evaluate it using the standard
three-season frozen comparison:

```bash
uv run nba-evaluate-compiled-additive-prior-hpm-x3 \
  2>&1 | tee artifacts/logs/compiled-additive-prior-hpm-x3-frozen.log
```
