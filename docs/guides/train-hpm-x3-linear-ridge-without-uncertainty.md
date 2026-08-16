---
last_updated: "2026-08-16"
---

# Train NAIL-RAPM v1.0

```bash
uv run nba-train-hpm-x3-linear-ridge-without-uncertainty --through-season 2025-26 \
  2>&1 | tee artifacts/logs/hpm-x3-linear-ridge-without-uncertainty-2025-26.log
```

Follow the run:

```bash
tail -f artifacts/logs/hpm-x3-linear-ridge-without-uncertainty-2025-26.log
```

Run the focused frozen comparison after training:

```bash
uv run nba-evaluate-hpm-x3-linear-ridge-without-uncertainty \
  2>&1 | tee artifacts/logs/hpm-x3-linear-ridge-without-uncertainty-frozen.log
```
