<p class="project-kicker">Player priors / forward filtering</p>

# RAPM Aging Model

<p class="project-lead">
The first player-history model predicts a target season's canonical RAPM from
information available before that season. It establishes the temporal contract
used later by prior-centered RAPM and neural player tokens.
</p>

## Estimand

For player \(i\) entering target season \(t\), the model estimates

\[
\mu_{i,t}
=
E\left[
  \widehat{\beta}^{\,0}_{i,t}
  \mid
  \widehat{\beta}^{\,0}_{i,t-1},
  a_{i,t},
  e_{i,t},
  n_{i,t-1}
\right],
\]

where:

- \(\widehat{\beta}^{\,0}\) is canonical zero-centered one-season RAPM;
- \(a\) is target-season age;
- \(e\) is NBA experience entering the target season;
- \(n\) is prior-season RAPM possession exposure.

The prediction \(\mu_{i,t}\) is a prior mean, not the final target-season RAPM
coefficient. [Prior-Centered RAPM](prior-rapm.md) is the first consumer of this
handoff.

## Initial feature model

The initial estimator is exposure-weighted ridge regression with:

- a quadratic B-spline expansion of target age;
- target NBA experience;
- prior RAPM;
- `log1p` prior RAPM possessions;
- explicit prior-season availability and rookie indicators.

Cold-start players retain `has_prior_season=false`. Their missing prior RAPM is
represented as zero only inside the fitted design; the published prior table
preserves the missing source value and includes the availability indicator.

The objective for training player-season rows \(j\) is

\[
\underset{\theta}{\operatorname{argmin}}
\quad
\sum_j
\widetilde{n}_j
\left(
  \widehat{\beta}^{\,0}_j - f_\theta(z_j)
\right)^2
+
N\lambda\lVert\theta\rVert_2^2,
\]

where \(\widetilde{n}_j\) is target RAPM exposure normalized to mean one and
\(N\lambda\) follows the project's sample-size-normalized ridge convention.

## Forward-only selection

Target seasons are the split unit. Suppose transition rows end in 2020-21
through 2025-26:

| Fold | Training targets | Validation target |
| --- | --- | --- |
| 0 | 2020-21 | 2021-22 |
| 1 | 2020-21 through 2021-22 | 2022-23 |
| 2 | 2020-21 through 2022-23 | 2023-24 |
| 3 | 2020-21 through 2023-24 | 2024-25 |
| Final | 2020-21 through 2024-25 | 2025-26 |

Regularization minimizes pooled exposure-weighted validation MSE. The age
spline, scaling parameters, and ridge coefficients are refitted inside every
fold. The latest target season is never used for preprocessing, selection, or
fitting.

Changing the latest season's RAPM outcomes therefore cannot change its
published priors. This is enforced by a regression test.

## Baselines and metrics

The untouched holdout compares:

| Model | Prediction |
| --- | --- |
| Zero | \(0\) |
| Training mean | Exposure-weighted mean training RAPM |
| Persistence | Prior RAPM for returning players, otherwise \(0\) |
| Aging ridge | Selected forward model |

RMSE, MAE, exposure-weighted RMSE and MAE, skill versus zero, and skill versus
persistence are reported for all players, exposure-eligible players, returning
players, and cold starts.

## Prior uncertainty

The model reports separate returning-player and cold-start error scales. Each
is the exposure-weighted RMSE from the selected model's expanding-fold
predictions:

\[
s_c
=
\sqrt{
\frac{
  \sum_{j\in c} n_j
  \left(
    \widehat{\beta}^{\,0}_j-\widehat{\mu}_j
  \right)^2
}{
  \sum_{j\in c} n_j
}
}.
\]

This is an out-of-time predictive error scale, not posterior parameter
uncertainty. A later prior-centered RAPM can use larger \(s_c\) to apply weaker
shrinkage.

## Leakage boundary

`player_priors.parquet` intentionally excludes:

- target RAPM;
- target RAPM exposure and seconds;
- target exposure eligibility;
- target-season box-score outcomes.

It contains only target identity/context, prior-season inputs, prior mean,
error scale, method, and the last target season used to train the model.

## Limitations

- Canonical RAPM is a noisy training label rather than latent player ability.
- A common age curve cannot represent every role or player archetype.
- The initial cold-start path uses age and experience but not yet draft or
  box-score information.
- The empirical error scale is cohort-level rather than player-specific.
- Missing an intervening season is treated as a cold start.

These limitations are deliberate. The first model establishes the temporal and
artifact contracts before adding box-score PM, draft priors, or a joint
hierarchical trajectory model.

## References

- Tashman, L. J. (2000). "Out-of-sample tests of forecasting accuracy: an
  analysis and review." *International Journal of Forecasting*, 16(4),
  437-450. [doi:10.1016/S0169-2070(00)00065-0](https://doi.org/10.1016/S0169-2070(00)00065-0)
- Hastie, T., Tibshirani, R., and Friedman, J. (2009).
  [*The Elements of Statistical Learning*](https://hastie.su.domains/ElemStatLearn/),
  second edition. See the spline, regularization, and model-assessment
  chapters.
- Sill, J. (2010). "Improved NBA Adjusted +/- Using Regularization and
  Out-of-Sample Testing." *MIT Sloan Sports Analytics Conference*.
