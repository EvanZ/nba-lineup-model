# Evaluate State-Precision NAIL Without Forgetting

This evaluator replays the three frozen target seasons from the completed
through-2025-26 State-Precision NAIL artifact. It does not retrain any seasons:
each target uses its persisted player-prior vector plus the immediately prior
completed context and schedule states.

```bash
uv run nba-evaluate-nail-state-precision-no-forgetting \
  --log-path artifacts/logs/nail-state-precision-no-forgetting-frozen.log
```

Follow the scoring-only run with:

```bash
tail -f artifacts/logs/nail-state-precision-no-forgetting-frozen.log
```

The report is written below
`artifacts/models/nail_state_precision_no_forgetting_frozen_backtest/` and
compares the candidate with the current NAIL-RAPM v1.2.1.2 production artifact.
