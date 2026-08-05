# Evaluate A Frozen Preseason Prior

Evaluate the lagged-RAPM baseline with:

```bash
uv run nba-evaluate-frozen-lagged-prior --season 2025-26
```

The command resolves the latest regular-only forward RAPM run unless
`--prior-run-id` is supplied. It freezes the completed 2024-25 player state and
evaluates the full 2025-26 regular season and playoffs without fitting any
2025-26 player coefficient.

Evaluate the forward age/draft/physical candidate with:

```bash
uv run nba-evaluate-frozen-aging-prior --season 2025-26
```

This command resolves the latest validated aging model and regular-only lagged
RAPM run. The former supplies frozen player values; the latter supplies the
common 2024-25 mean and home-court term. Pass `--aging-run-id` or
`--reference-prior-run-id` to make either dependency explicit. Neither command
fits a 2025-26 player coefficient.

The immutable run is published under:

```text
artifacts/models/frozen_prior_evaluation/<season>/<run-id>/
```

The artifact includes:

- separate regular-season and playoff possession/game metrics;
- possession and game predictions;
- regular-season team net-rating predictions and metrics;
- Pythagorean expected wins from a forward historical NetRtg-to-win mapping;
- raw predicted game-winner counts retained as a diagnostic;
- the historical team-season panel used to fit the win mapping;
- the exact frozen player-prior table and source-state declaration;
- a hash-validated manifest and MLflow index.

Realized target-season lineups are an explicit oracle input. Target-season
outcomes appear only on the evaluation side of the artifact.
