# Train NAIL-RAPM v1.2.1.2 Back-to-Back Control

*Last updated: 2026-08-24*

This candidate holds the NAIL v1.2.1.1 profile contract fixed, including
conventional USG%, and adds one known-before-tipoff schedule feature:
home back-to-back minus away back-to-back.

Run the recursive fit through the current completed season:

```bash
uv run nba-train-nail-v1212-back-to-back \
  --through-season 2025-26 \
  --log-path artifacts/logs/nail-v1212-back-to-back.log
```

Follow its progress:

```bash
tail -f artifacts/logs/nail-v1212-back-to-back.log
```

The default schedule Ridge penalty matches the current context penalty. Test a
different fixed penalty only with an explicit value:

```bash
uv run nba-train-nail-v1212-back-to-back --schedule-alpha 10000
```

After training, run the strict three-season replay. It compares the candidate
to both the production v1.2.1 model and the standard-USG% v1.2.1.1 baseline:

```bash
uv run nba-backtest-nail-v1212-back-to-back \
  --log-path artifacts/logs/nail-v1212-back-to-back-frozen.log
```

Then run the paired bootstrap against the standard-USG% parent branch:

```bash
uv run nba-bootstrap-nail-v1212-back-to-back
```

Render the required 30-season coefficient trajectory:

```bash
uv run nba-audit-nail-v1212-back-to-back-weights
```

See [Schedule Controls](../data/schedule-controls.md) for the data contract.
