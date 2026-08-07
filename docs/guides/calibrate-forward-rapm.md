---
last_updated: "2026-08-04"
---

# Calibrate Forward RAPM To Wins

This command builds a retrospective, **usage-conditional** team-win calibration.
It uses only the frozen prior RAPM available before each season, then weights
those player priors by the lineup-stint seconds that the team actually used in
that season. It does not use age, box-score, draft, physical, or target-season
RAPM inputs.

Run it after training the regular-only forward lagged-prior RAPM exemplar:

```bash
uv run nba-calibrate-forward-rapm-wins \
  --forward-lagged-run-dir artifacts/models/prior_rapm/2025-26/forward-lagged-rapm-2025-26-20260803T203054Z-c627d89d
```

The command writes an immutable run under
`artifacts/models/forward_calibration/2025-26/` and updates `latest.json`.
It is also indexed in local MLflow when automatic tracking is enabled.

The first four seasons with usable frozen priors warm up the calibration. Each
subsequent season is evaluated using a weighted linear mapping fitted only on
earlier completed team-seasons. The 2025-26 target fit therefore uses
1997-98 through 2024-25 outcomes, never 2025-26 outcomes.

See [Forward RAPM Calibration](../models/forward-calibration.md) for the model
contract, equations, artifacts, and results.
