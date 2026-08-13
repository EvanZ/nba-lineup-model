---
last_updated: "2026-08-11"
---

# Controlled No-Context Value-Conditioned RAPM

This model is the direct player-only control for Value-Conditioned Aging HPM.
It preserves the same historical regular-season data, lambda schedule,
possession-weighted prior centering, value-conditioned aging prior, and
exposure-gated cold-start prior. It removes only the recursive contextual state:

\[
C_{t-1}(U,V)=0.
\]

Therefore it answers a narrow question: what does the HPM player-prior system
predict when no lineup-composition or opponent-matchup adjustment can be
allocated separately from player coefficients? Every season is retrained in
sequence, so this is a genuine counterfactual state rather than a translation
of an older uncentered RAPM artifact.

## Frozen 2025-26 Result

| Metric | No-context control | Value-Conditioned Aging HPM |
| --- | ---: | ---: |
| Regular possession RMSE | 1.198800 | 1.198763 |
| Regular game-margin RMSE | 14.4991 | 14.3469 |
| Team NetRtg RMSE | 4.0654 | 3.7568 |
| Pythagorean-win RMSE | 8.6225 | 8.0346 |
| Playoff possession RMSE | 1.192332 | 1.192460 |
| Playoff game-margin RMSE | 16.2971 | 16.4605 |

Context improves the primary regular-season game and team outcomes, while the
control is marginally stronger on the small target-season playoff cohort. The
result supports keeping context in HPM for preseason regular-season prediction,
but makes its effect on player attribution directly inspectable.

## Curry Diagnostic

The control materially changes Stephen Curry's 2015--19 player state:

| Season | No-context RAPM | Value-Conditioned Aging HPM |
| --- | ---: | ---: |
| 2014-15 | +7.49 | +5.06 |
| 2015-16 | +10.00 | +6.43 |
| 2016-17 | +9.54 | +5.35 |
| 2017-18 | +8.06 | +4.01 |
| 2018-19 | +7.91 | +4.30 |

That difference is consistent with a substantial part of the Warriors-era
unit environment being represented by the contextual state in HPM. It does not
establish a unique redistribution of credit among teammates: lineup context is
shared exposure, not a player-level causal attribution.

## Artifact

```text
artifacts/models/forward_centered_value_conditioned_aging_no_context_rapm/2025-26/
```

`metadata.json` records `context_enabled: false`, and all frozen-evaluation
context corrections are exactly zero. The [training guide](../guides/train-forward-centered-value-conditioned-aging-no-context-rapm.md)
contains the command and boundary contract.
