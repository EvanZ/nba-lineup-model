---
last_updated: "2026-08-21"
---

# Train NAIL-RAPM v1.3 Additive Profiles

Build the assisted-shot profile mart first:

```bash
uv run nba-build-assisted-shot-taxonomy --workers 4
```

Train the recursive candidate and mirror durable progress to a log:

```bash
uv run nba-train-nail-v13-additive-profiles \
  --log-path artifacts/logs/nail-v13-additive-profiles.log
```

Replay the three frozen seasons:

```bash
uv run nba-evaluate-nail-v13-additive-profiles \
  --log-path artifacts/logs/nail-v13-additive-profiles-frozen.log
```

Progress lines are intentionally append-only so they are readable with:

```bash
tail -f artifacts/logs/nail-v13-additive-profiles-frozen.log
```

Run the required 10,000-draw paired bootstrap gate after the frozen replay:

```bash
uv run nba-bootstrap-nail-v13-additive-profiles
```

Generate the all-season additive-weight trajectories and the accompanying
stability summary:

```bash
uv run nba-audit-nail-v13-additive-weights
```

The candidate is not eligible for promotion when the upper end of the paired
95% interval for full-game RMSE exceeds 0.5% of the incumbent's RMSE in the
pooled sample or any individual frozen season.
