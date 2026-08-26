---
last_updated: "2026-08-25"
---

# Screen a Frozen Feature

Use this diagnostic before spending a full recursive training run on a proposed
lineup-context feature. It asks a narrow question: after the promoted frozen
NAIL-RAPM v1.2.1.2 prediction has accounted for player ratings, the retained
non-additive terms, home court, and back-to-back status, does the candidate
still align with the remaining stint-level residual?

## Run

```bash
uv run nba-screen-frozen-feature median_three_pm_per_100
```

The first registered candidate is the home-minus-away difference in the median
of each unit's five prior-season, shrinkage-adjusted `3PM / 100` values. With
five players, that is the third-ranked shooting profile, so it tests whether a
unit has a credible third shooter rather than simply one elite shooter.

## Contract

For target season `t`, the feature uses only player profiles available before
that season. The residual is:

\[
e_s = y_s - \widehat{y}^{\mathrm{NAIL\ v1.2.1.2}}_s.
\]

`y_s` is the observed home net rating for stint `s`; the prediction is replayed
from the persisted state through `t-1`. No target-season player, context, or
schedule parameter is refit. Target outcomes are used only to evaluate the
screen.

For a feature already present in the production context model, the diagnostic
uses a **leave-one-term-out** version of that same frozen prediction. It zeros
only that feature's home-minus-away context input before scoring, while holding
every other player, context, home-court, and back-to-back component fixed. That
makes production features valid positive controls: their own contribution is
left in the residual instead of being subtracted away by the full prediction.

Continuous candidates are divided into ten deterministic rank deciles. Binary
or otherwise low-cardinality candidates retain their natural value groups so a
one-hot feature is not artificially split across deciles. Each point is a
possession-weighted mean residual. The displayed 95% intervals are descriptive
normal approximations using effective stint sample size, not game-clustered
inference.

## Outputs

Each run writes an immutable artifact directory under
`artifacts/models/analysis/frozen_feature_screen/<feature>/<run-id>/`:

| Artifact | Purpose |
| --- | --- |
| `stint_residuals.parquet` | Frozen prediction, observed rating, residual, and candidate values for every regular-season stint. |
| `residual_bins.parquet` | Per-season and pooled bin means, descriptive intervals, and weighted residual correlations. |
| `metadata.json` | Candidate definition, promoted source-model run, seasons, and information-boundary contract. |

The command also renders the chart to
`docs/assets/images/frozen-feature-screens/<feature>-residual-screen.svg` by
default. A candidate advances to a full recursive training experiment only if
this screen shows a stable, basketball-plausible conditional pattern across the
three frozen seasons.

## Registering Candidates

Candidate definitions live in
`src/nba_lineup_model/modeling/frozen_feature_screen.py` as `FeatureCandidate`
entries. A new candidate provides a function mapping a five-player unit and its
frozen profile table to one scalar. The diagnostic forms the home-minus-away
contrast automatically, so every feature follows the same sign convention.

## Initial Result: Median Lineup 3PM / 100

The initial screen used the promoted v1.2.1.2 state and all three frozen
regular seasons. It covered 108,799 stints and 348,786 possessions. The
possession-weighted correlation between the median-shooting edge and the frozen
residual was effectively zero in every season:

| Target season | Weighted residual correlation |
| --- | ---: |
| 2023-24 | -0.0058 |
| 2024-25 | +0.0011 |
| 2025-26 | +0.0045 |
| Pooled | +0.0003 |

The residual deciles have no monotonic or stable conditional pattern, so median
lineup 3PM / 100 does **not** advance to a recursive model experiment.

![Frozen residual decile screen for median lineup 3PM per 100](../assets/images/frozen-feature-screens/median_three_pm_per_100-residual-screen.svg)

## Candidate Result: Rim-Protection Ceiling

The rim-protection candidate is the maximum prior-season,
shrinkage-adjusted `BLK / 100` profile among a unit's five players. It tests a
strictly non-additive hypothesis: a lineup without even one credible rim
protector may underperform what the sum of individual player ratings predicts.
Low values therefore represent a lineup that lacks rim protection.

The candidate does not clear the frozen screen. Its possession-weighted
residual correlations are effectively zero and change sign across seasons:

| Target season | Weighted residual correlation |
| --- | ---: |
| 2023-24 | -0.0120 |
| 2024-25 | +0.0039 |
| 2025-26 | +0.0099 |
| Pooled | +0.0009 |

The deciles are non-monotonic, with no stable low-ceiling penalty after the
frozen NAIL-RAPM v1.2.1.2 prediction. Maximum shrunken `BLK / 100` therefore
does **not** advance to a recursive model experiment.

![Frozen residual decile screen for rim-protection ceiling](../assets/images/frozen-feature-screens/rim_protection_ceiling-residual-screen.svg)

## Production Controls

The two retained production non-additive terms validate the screen's
leave-one-term-out control path. For each one, the frozen prediction retains
every other component but omits that term's own contribution. Both produce a
small but directionally stable positive residual relationship across all three
target seasons:

| Feature | 2023-24 | 2024-25 | 2025-26 | Pooled |
| --- | ---: | ---: | ---: | ---: |
| Usage concentration | +0.0166 | +0.0128 | +0.0142 | +0.0146 |
| Top-two assists / 100 | +0.0057 | +0.0096 | +0.0066 | +0.0073 |

The pooled usage-concentration deciles move from `-1.73` to `+2.84` residual
net-rating points from the lowest to highest feature edge. Top-two assists move
from `-1.34` to `+1.84`. Middle deciles remain noisy because stint outcomes are
noisy; these controls validate that the screen recovers a withheld production
term, not that it establishes a causal effect.

![Frozen residual decile screen for usage concentration](../assets/images/frozen-feature-screens/usage_concentration-residual-screen.svg)

![Frozen residual decile screen for top-two assists](../assets/images/frozen-feature-screens/top_two_assists-residual-screen.svg)
