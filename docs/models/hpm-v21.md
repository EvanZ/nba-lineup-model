---
last_updated: "2026-08-13"
---

# HIPSTER PM v2.1: Empirical Rebound Capacity

HIPSTER PM v2.1 is a feature release built on HPM v2's depth-aware shooting
representation. It preserves the value-conditioned aging player prior,
exposure-gated cold-start path, rolling information boundary, bounded
hierarchical P-spline penalties, and portable-matchup decomposition. Only the
rebounding context representation changes.

## Rebound Capacity

The prior feature for player \(i\) is standard box-score rebound percentage:

\[
\mathrm{DRB\%}_i = 100\frac{\mathrm{DRB}_i(\mathrm{TmMP}/5)}
{\mathrm{MP}_i(\mathrm{TmDRB}+\mathrm{OppORB})},
\]

with the analogous offensive formula using
\(\mathrm{TmORB}+\mathrm{OppDRB}\). Percentages are calculated game by game
from the prior regular season's cached player box scores, then minute-weighted
to the player-season profile. First-year profiles use the existing
draft-cohort/replacement blend of the same percentage traits.

For a rebound opportunity, let \(D\) be the defending unit and \(O\) the
offensive unit. v2.1 first sums their prior-season player claims:

\[
S_D(D)=\sum_{i\in D}\mathrm{DRB\%}_i,\qquad
S_O(O)=\sum_{i\in O}\mathrm{ORB\%}_i.
\]

It then fits a regularized logistic spline on the completed season's actual,
non-dead-ball rebound events:

\[
\Pr(\text{defensive rebound}) =
\operatorname{logit}^{-1}\{f_D(S_D(D)) + f_O(S_O(O))\}.
\]

The two smooth functions and their coefficients are empirical: they are
learned from observed rebound opportunities rather than chosen as a hard
25%/75% cap. A unit's portable features are its expected ORB% and DRB% against
the completed season's reference opponent-claim distribution. This gives
Kevin Love plus Steven Adams less than additive *realized* rebound credit when
their individual claims compete for the same opportunities, while leaving the
degree of saturation to the data.

## Feature Registry

| Rebound feature | HPM v2 | HPM v2.1 | Reason |
| --- | --- | --- | --- |
| Raw offensive rebounds per 100 | Active | Retired | Volume is not normalized to available opportunities. |
| Raw defensive rebounds per 100 | Active | Retired | Same opportunity problem. |
| Square-root offensive rebounds | Active | Retired | Generic concavity does not encode rebound competition. |
| Square-root defensive rebounds | Active | Retired | Same limitation. |
| Rebounding by usage | Active | Retired | Depends on the retired volume proxy. |
| Expected offensive rebound rate | None | Added | Empirical ORB% against reference defensive claims. |
| Expected defensive rebound rate | None | Added | Empirical DRB% against reference offensive claims. |

## Evaluation Contract

For a forecast of season \(t\), every player percentage and the rebound
realization calibration come from completed data through \(t-1\). The
contextual state fitted in \(t-1\) is then applied to target lineups.
Target-season regular and playoff outcomes are evaluation-only.

## Frozen Result

Across the frozen 2023-24 through 2025-26 regular-season evaluation, v2.1
records a possession RMSE of **1.198010** and an eligible game-margin RMSE of
**14.1340**. It is effectively tied with the incumbent HPM on possession error
while modestly improving eligible game-margin error. See the
[Three-Season Frozen Leaderboard](three-season-frozen-backtest.md) for the
complete regular-season and playoff comparison.

## Response-Curve Audit

The completed-history audit evaluates each stored rebound calibration across
its own 5th--95th percentile claim support. All 29 seasons have the expected
overall direction: higher defensive claims increase defensive-rebound
probability and higher offensive claims decrease it. The central-support
median effect is +4.73 percentage points for defensive claims and -6.28
points for offensive claims. Five seasons contain tiny local spline reversals;
the largest is only 0.020 percentage points over one of 100 grid intervals.
That is negligible relative to the learned season-level effect, but the audit
remains part of the release contract rather than assuming monotonicity.

Reproduce it with:

```bash
uv run nba-audit-rebound-capacity
```

## Reproduce

```bash
uv run nba-train-hpm-v21 --through-season 2025-26 \
  2>&1 | tee artifacts/logs/train-hpm-v21-2025-26.log
```

Follow progress with:

```bash
tail -f artifacts/logs/train-hpm-v21-2025-26.log
```

The immutable outputs are stored in:

```text
artifacts/models/forward_hpm_v21_empirical_rebound_capacity/2025-26/
```
