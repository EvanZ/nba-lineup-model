---
last_updated: "2026-08-08"
---

# Forward Hierarchical P-spline Contextual RAPM

This is a forward contextual RAPM candidate designed to address two limitations
of independently fit seasonal spline states: a scalar Ridge penalty does not
directly prefer smooth curves, and separate years can let weakly identified
features move too freely. It retains the portable-matchup total-context
contract:

\[
C(A,B)=h(x(A))-h(x(B))+q(x(A),x(B)).
\]

It uses the same player recursion, exposure-gated cold starts, side-level box
score profile features, and frozen evaluation boundary as [Forward
Portable-Matchup Contextual RAPM](forward-portable-matchup-contextual-rapm.md).
Only the seasonal context-function estimator changes.

## Seasonal Function Prior

For season \(s\), let \(\beta_s\) be the B-spline coefficient vector after the
current season's basis expansion and standardization. The context fit minimizes
weighted residual error with three regularizers:

\[
\sum_i w_i\left(y_i-X_i\beta_s\right)^2
+ \lambda_{\mathrm{level}}\lVert\beta_s\rVert_2^2
+ \lambda_{\mathrm{curve}}\lVert D^2\widetilde\beta_s\rVert_2^2
+ \lambda_{\mathrm{time}}\lVert\beta_s-\mu_{s-1\rightarrow s}\rVert_2^2.
\]

\(\widetilde\beta_s\) denotes coefficients in the unscaled B-spline basis,
and \(D^2\) takes adjacent second differences. The P-spline term is therefore
small for a locally linear feature function and larger for a sharp bend or
oscillation. It does not impose monotonicity: a stable, data-supported curved
effect remains possible.

\(\mu_{s-1\rightarrow s}\) is a temporal prior, not a reused coefficient
vector. Each seasonal fit learns its own knots and scaler from its own
completed data. Before fitting season \(s\), the completed season \(s-1\)
feature response is evaluated over the central observed support of season
\(s\), then least-squares projected onto season \(s\)'s standardized basis.
That projected response is the prior mean. The first contextual season has no
temporal predecessor and uses only level and curvature shrinkage.

This is a first-order Gaussian state hierarchy: completed states are connected
by \(\beta_s\mid\beta_{s-1}\), but no future-season profile, target, knot, or
coefficient is used to forecast an earlier season.

## Default Strengths

The initial exemplar fixes:

\[
\lambda_{\mathrm{level}}=10{,}000,\qquad
\lambda_{\mathrm{curve}}=1{,}000,\qquad
\lambda_{\mathrm{time}}=10{,}000.
\]

Those values deliberately make this a conservative function-family test. They
match the published level-shrinkage scale, penalize visible local artifacts,
and give the prior completed state material influence without making it a hard
constraint. The artifact records all three penalties for every run; later
work can tune them on expanding historical forecast seasons.

## What It Tests

The [Context Function Audit](context-function-audit.md) found little evidence
of repeated within-season turning points in central support, while many feature
contrasts scatter across seasons rather than follow a stable trend. This model
tests whether a smooth dynamic prior improves frozen prediction and produces
more defensible response functions. A visual improvement alone is not a win;
it must also preserve or improve the frozen regular-season and playoff metrics
in the [Frozen Preseason Leaderboard](preseason-leaderboard.md).

## Artifact And Evaluation

The completed run will be stored under
`artifacts/models/forward_hierarchical_pspline_contextual_rapm/2025-26/`.
It persists the same player states, seasonal context models, reference units,
context metadata, possession predictions, and full-game forecasts as the
portable-matchup model. The full-game report promotes the candidate
automatically after a valid artifact is present.

Use [the training guide](../guides/train-forward-hierarchical-pspline-contextual-rapm.md)
to reproduce the run.
