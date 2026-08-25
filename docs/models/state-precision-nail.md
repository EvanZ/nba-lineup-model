# State-Precision NAIL Foundation

*Last updated: 2026-08-24*

This is the implementation foundation for a future uncertainty-aware player
state model. It is not yet a fitted candidate and has no leaderboard result.
It replaces the invalid post-hoc Kalman audit with a state-space-compatible
RAPM objective.

## Objective

For player coefficient \(\beta_i\), forward prior mean \(m^-_{i,t}\), and
prior variance \(P^-_{i,t}\), State-Precision NAIL uses

\[
\min_{\beta}
\sum_s w_s(y_s - X_s\beta)^2 +
\lambda_{\mathrm{global}}
\sum_i
\underbrace{\frac{P_{\mathrm{ref}}}{P^-_{i,t}}}_{\lambda_i / \lambda_{\mathrm{global}}}
(\beta_i - m^-_{i,t})^2.
\]

`P_ref` is the median prior variance in the fitted season, so the median
relative precision is one. The existing globally selected `lambda` remains
the overall penalty scale. A player with high state uncertainty therefore has
a weaker pull toward their prior; a stable player has a stronger pull.

The implementation rescales player-design columns by
\(1 / \sqrt{P_{\mathrm{ref}} / P^-_{i,t}}\), then uses the existing
sample-size-normalized ridge solver. This preserves sparse-matrix behavior and
makes the uniform-precision case exactly equivalent to production
prior-centered ridge.

## State Contract

The completed NAIL fit produces a diagonal Laplace approximation to the player
posterior covariance:

\[
P^+_t \approx \hat{\sigma}^2
\operatorname{diag}\left(
X^\top W X + \alpha\,\operatorname{diag}(\lambda_i / \lambda_{\mathrm{global}})
\right)^{-1}.
\]

The existing forward aging and gap-returner model supplies the next mean. Its
variance advances without future outcomes:

\[
P^-_{i,t+1}=P^+_{i,t}+q\,\Delta t.
\]

The process variance \(q\) will be selected only with pre-target rolling
seasons. No completed ridge coefficient is reused as a second observation, so
the post-hoc double-shrinkage problem is avoided.

## Gates Passed

- **Uniform precision parity:** all relative precisions equal to one reproduce
  `PriorCenteredRidgeLineupModel` coefficients, adjustments, intercept, and
  predictions exactly.
- **Closed-form posterior:** a centered one-player test matches the analytic
  ridge posterior.
- **Uncertainty behavior:** posterior variance advances forward only and maps
  monotonically to relative precision, with median precision equal to one.
- **End-to-end replay:** a full 1996-97 through 2025-26 replay of the
  equal-variance path matched NAIL-RAPM v1.2.1.2 to floating-point solver
  tolerance: maximum absolute differences were `3.07e-06` for player
  coefficients and priors, `3.30e-09` for possession predictions, and
  `3.50e-07` for game predictions. The discrepancy was traced to multiplying
  the sparse design matrix by an all-ones diagonal; the final parity path now
  bypasses that no-op, and its unit test preserves the incumbent matrix route.

The tests live in `tests/test_kalman_player_prior.py`; the implementation is
`models/baselines.py` and `modeling/state_precision.py`.

## Next Gate

Select a forward-only process-variance policy on pre-target validation
seasons, then run the first non-uniform State-Precision NAIL candidate. It
must be compared with the frozen NAIL v1.2.1.2 baseline before it can be
considered for promotion.
