# Build Exposure-Gated O/D Cold Starts

Build the frozen O/D cold-start model after the frozen O/D RAPM source and the
compatible cold-start exposure study exist:

```bash
uv run nba-build-exposure-gated-offense-defense --season 2025-26
```

The command resolves the latest compatible O/D and exposure-gate artifacts,
then verifies both stop at the prior regular season. It fits separate O/D
draft-rate ridge models and one tokenized O/D replacement fit for every
completed historical season through the source season.

Results are written under:

```text
artifacts/models/exposure_gated_offense_defense/<season>/<run_id>/
```

The command evaluates regular season and playoffs separately without a target
player refit. See [Exposure-Gated O/D Cold Starts](../models/exposure-gated-offense-defense.md)
for the formulas, source contract, and result.
