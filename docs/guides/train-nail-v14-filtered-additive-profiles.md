---
last_updated: "2026-08-21"
---

# Train NAIL-RAPM v1.4 Kalman Profiles

Build the assisted-shot profile mart first:

```bash
uv run nba-build-assisted-shot-taxonomy --workers 4
```

Train the 30-season recursive candidate with an append-only log:

```bash
uv run nba-train-nail-v14-kalman-additive-profiles \
  --log-path artifacts/logs/nail-v14-kalman-additive-profiles.log
```

Each bar advances only after a seasonal context state has been stored. Follow
the run with:

```bash
tail -f artifacts/logs/nail-v14-kalman-additive-profiles.log
```

Replay the three frozen evaluation seasons after training:

```bash
uv run nba-evaluate-nail-v14-kalman-additive-profiles \
  --log-path artifacts/logs/nail-v14-kalman-additive-profiles-frozen.log
```

Then produce the required 10,000-draw paired-bootstrap non-promotion gate and
the all-season coefficient audit:

```bash
uv run nba-bootstrap-nail-v14-kalman-additive-profiles
uv run nba-audit-nail-v14-kalman-additive-weights
```

The initial Kalman experiment uses a diagonal random-walk process multiplier
of `1.0`: the next-season prior variance equals the prior posterior variance
plus one additional copy of that diagonal posterior variance. The current
season's weighted residual variance calibrates the measurement update.
