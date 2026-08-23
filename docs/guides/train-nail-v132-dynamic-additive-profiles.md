---
last_updated: "2026-08-22"
---

# Train NAIL-RAPM v1.3.2 Dynamic Additive Profiles

v1.3.2 retains v1.3.1's ten additive profile terms. It replaces independent
seasonal Ridge coefficients with a forward feature-specific mean-reverting
state: a running long-run mean, empirical innovation variance, and an AR(1)
transition estimated only from completed prior seasons.

```bash
uv run nba-train-nail-v132-dynamic-additive-profiles \
  --log-path artifacts/logs/nail-v132-dynamic-additive-profiles.log
```

Follow recursive seasonal progress:

```bash
tail -f artifacts/logs/nail-v132-dynamic-additive-profiles.log
```

Run the three-season frozen replay:

```bash
uv run nba-evaluate-nail-v132-dynamic-additive-profiles \
  --log-path artifacts/logs/nail-v132-dynamic-additive-profiles-frozen.log
```

Evaluate the paired bootstrap gate against v1.3.1:

```bash
uv run nba-bootstrap-nail-v132-dynamic-additive-profiles
```

Render the completed 10-panel coefficient audit:

```bash
uv run nba-audit-nail-v132-dynamic-additive-weights
```
