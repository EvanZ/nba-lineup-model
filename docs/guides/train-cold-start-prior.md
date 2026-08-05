# Train The Cold-Start RAPM Prior

The cold-start experiment forecasts players without an immediately preceding
NBA season. It uses only preseason age and experience, draft data, height,
weight, and listed position:

```bash
uv run nba-train-cold-start-prior --holdout-season 2025-26
```

It selects ridge regularization through expanding target-season folds and
publishes an immutable run in
`artifacts/models/cold_start_prior/<season>/<run-id>/`. Its required baselines
are zero RAPM and the possession-weighted mean target RAPM from the forward
training window; lagged-RAPM persistence is unavailable for this cohort.

This component is eligible for a complete box-score prior only if it improves
the possession-weighted objective over the forward mean. The first 2025-26
run did not clear that bar, so it is retained as a negative-result artifact and
is not used in a blend.
