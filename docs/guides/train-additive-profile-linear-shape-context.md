---
last_updated: "2026-08-15"
---

# Train Additive Prior Plus Linear Non-Additive Context

Fit the 30-season forward state through the completed 2025-26 season:

```bash
uv run nba-train-forward-additive-profile-linear-shape-context --through-season 2025-26 \
  2>&1 | tee artifacts/logs/additive-profile-linear-shape-context-2025-26.log
```

Follow the run:

```bash
tail -f artifacts/logs/additive-profile-linear-shape-context-2025-26.log
```

Then replay the controlled pair on the three frozen seasons:

```bash
uv run nba-evaluate-additive-profile-linear-shape-context \
  2>&1 | tee artifacts/logs/additive-profile-linear-shape-context-frozen.log
```
