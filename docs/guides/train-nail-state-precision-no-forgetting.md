# Train State-Precision NAIL: No Forgetting

*Last updated: 2026-08-24*

This command runs the first uncertainty-aware NAIL candidate. It preserves
each completed player's posterior variance into the next season and adds no
offseason process variance.

```bash
uv run nba-train-nail-state-precision-no-forgetting \
  --through-season 2025-26 \
  --log-path artifacts/logs/nail-state-precision-no-forgetting.log
```

Monitor the run with:

```bash
tail -f artifacts/logs/nail-state-precision-no-forgetting.log
```

The output is isolated under
`artifacts/models/forward_nail_state_precision_no_forgetting/` and must be
evaluated against NAIL-RAPM v1.2.1.2 before promotion.
