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

## Draft And Physical Feature Model

The initial estimator is exposure-weighted ridge regression with:

- a quadratic B-spline expansion of target age;
- target NBA experience;
- prior RAPM;
- `log1p` prior RAPM possessions;
- explicit prior-season availability and rookie indicators.
- estimated draft age, draft status and pick availability;
- draft pick, height, and weight with fold-local median imputation and
  missingness indicators;
- age-slope interactions for early-entry drafted players, late-entry drafted
  players, and undrafted players.

Season bios provide a listed age and draft year, not an exact birth date and
draft date. The model therefore uses an estimated draft age and retains only
plausible values from 17 through 30; other records are explicitly unknown.

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

Target seasons are the split unit. The full-history run has 27 expanding
validation folds: it fits 1997-98 to predict 1998-99, extends the training
window one season at a time through a 2023-24 to 2024-25 validation fold, then
fits 1997-98 through 2024-25 to publish 2025-26 priors.

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

## 2025-26 Holdout

The full-history run holds out 2025-26, trains on target seasons 1997-98
through 2024-25, and selects from the documented ridge grid using 27 expanding
folds. It uses 13,527 training player-seasons and 582 holdout players (462
returning and 120 cold starts). The selected normalized ridge regularization is
\(\lambda=0.001\).

| Cohort | Zero | Training mean | Persistence | Aging ridge | Aging skill vs. persistence |
| --- | ---: | ---: | ---: | ---: | ---: |
| All players | 1.959 | 1.920 | 2.013 | **1.728** | **26.3%** |
| Exposure eligible | 1.975 | 1.932 | 2.026 | **1.741** | **26.2%** |
| Returning | 1.989 | 1.927 | 2.049 | **1.743** | **27.6%** |
| Cold start | 1.712 | 1.862 | 1.712 | **1.611** | **11.4%** |

Values are target-RAPM exposure-weighted RMSE; skill is the reduction in
weighted MSE relative to persistence. The run is
`aging-2025-26-20260803T223755Z-51f6e707`, stored under
`artifacts/models/aging/2025-26/`. Its input panel covers 1996-97 through
2025-26 and is pinned by hash in the model manifest.

## Aging Curve Case Study

The fitted spline defines a shared, conditional age effect. To make that term
interpretable, the case study holds all non-age inputs fixed at an
exposure-weighted returning-player reference profile and reports

\[
g(a) - g(29),
\]

where \(g(a)\) is the model prediction when only target age changes. Because
the model has no age interactions, this difference is exactly the fitted age
spline contribution and does not depend on the particular valid reference
profile used for the calculation. It is not average player RAPM at age \(a\),
nor a causal biological effect of aging.

The published curve comes from
`aging-curve-2025-26-20260803T231802Z-3ab3d8c9`. It uses the 13,527
player-season rows from target seasons 1997-98 through 2024-25 that trained the
source aging model. The shaded bands are 5th to 95th percentiles from 250
target-season block bootstrap refits. The selected ridge penalty and spline
specification remain fixed in each refit, so the interval reflects training
season resampling, not hyperparameter-selection uncertainty.

### Fixed-Experience Background Effect

![Fixed-experience background effect centered at age 29](../assets/images/aging/2025-26/aging-curve.svg)

This curve holds NBA experience fixed while changing chronological age. It is a
background and selection comparison, not a within-player aging trajectory.

### Forward Aging Trajectory

![Forward aging trajectory](../assets/images/aging/2025-26/forward-aging-trajectory.svg)

This curve advances age and NBA experience together from age 29. It is the
appropriate conditional curve for a one-year forward player trajectory.

![Draft-adjusted aging curves](../assets/images/aging/2025-26/draft-adjusted-aging-curves.svg)

The cohort chart holds prior RAPM, exposure, experience, height, and weight at
the same reference profile. It contrasts early-entry drafted (estimated draft
age 20), late-entry drafted (estimated draft age 24), and undrafted profiles.
These are model counterfactuals, not separate observed cohort averages.

### Physical Profiles

![Forward physical-profile aging curves](../assets/images/aging/2025-26/physical-profile-aging-curves.svg)

This forward-trajectory comparison advances age and experience together for a
6'3", 190 lb small guard, 6'7", 220 lb wing, and 6'11", 255 lb center profile.

### Star RAPM Histories

<div class="star-rapm-chart" data-source="../../assets/images/aging/2025-26/star-observed-rapm.json"></div>

Lines and points are observed one-season RAPM for Shaquille O'Neal, Tim Duncan,
Kobe Bryant, LeBron James, Stephen Curry, Kevin Durant, and Nikola Jokić. This
is descriptive history rather than a model-implied aging trajectory.

The common age term rises rapidly through age 23, where the conditional effect
is +0.20 points per 100 possessions relative to age 27. It then declines from
the late twenties: the estimated partial effect is -0.17 at age 30 and -0.56
at age 35. These are model-based conditional effects, not direct within-player
aging estimates.

| Age | Partial effect vs. 27 | 90% interval | Change to next age | Training player-seasons |
| ---: | ---: | ---: | ---: | ---: |
| 19 | -0.49 | [-0.85, -0.21] | +0.32 | 87 |
| 21 | +0.05 | [-0.07, +0.17] | +0.12 | 600 |
| 23 | +0.20 | [+0.11, +0.28] | -0.08 | 1,290 |
| 27 | 0.00 | [-0.00, +0.00] | -0.03 | 1,050 |
| 30 | -0.17 | [-0.24, -0.11] | -0.09 | 749 |
| 33 | -0.42 | [-0.54, -0.32] | -0.07 | 470 |
| 35 | -0.56 | [-0.71, -0.42] | -0.06 | 273 |
| 40 | -0.78 | [-1.07, -0.50] | -0.02 | 31 |

![Observed age support](../assets/images/aging/2025-26/age-support.svg)

The chart intentionally shows the thin tails. Ages 41 through 44 have only 17
player-seasons combined, so the linear spline extension there should not
be used as a strong claim about late-career decline. The empirical result is
better read as a smoothed, exposure-weighted prior for the observed NBA
population.

Rebuild the report and images from the pinned aging run with:

```bash
uv run --group docs nba-build-aging-curve-case-study 2025-26 \
  --aging-run-id aging-2025-26-20260803T223755Z-51f6e707 \
  --bootstrap-samples 250 \
  --bootstrap-seed 20260801
```

The immutable report contains `curve.parquet`, source hashes, a complete age
table, and both SVGs under
`artifacts/reports/aging_curve/2025-26/`.

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
