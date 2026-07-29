<p class="project-kicker">Model contract / exact inference</p>

# Bayesian RAPM Methodology

<p class="project-lead">
The first Bayesian baseline is the exact probabilistic counterpart of
canonical ridge RAPM. It keeps the selected linear predictor and makes its
assumptions about noise, shrinkage, and uncertainty explicit.
</p>

## Likelihood

For stint \(i\), let \(x_i\) contain `+1` for each home player and `-1` for
each away player. Let \(\widetilde{w}_i\) be the stint possession weight
divided by the training-window mean weight. The model is

\[
y_i \mid \alpha, \beta, \sigma^2
\sim
\mathcal{N}\left(
\alpha + x_i^\mathsf{T}\beta,
\frac{\sigma^2}{\widetilde{w}_i}
\right).
\]

Thus a stint with twice the possession exposure has half the conditional
variance. The target remains home point differential per 100 average team
possessions. The intercept \(\alpha\) represents home-court advantage.

## Priors

The player prior is conditional Gaussian:

\[
\beta \mid \sigma^2
\sim
\mathcal{N}\left(
0,
\frac{\sigma^2}{n\lambda}I
\right),
\]

where \(n\) is the number of training stints and \(\lambda\) is the normalized
ridge penalty selected by chronological validation. The intercept is flat and
unpenalized. The residual variance uses the scale-invariant prior

\[
p(\sigma^2) \propto \frac{1}{\sigma^2}.
\]

This is a baseline prior, not the final prior design. In particular, its common
scale does not use age, position, draft information, season history, or
offensive and defensive roles.

## Exact posterior

Write \(Z=[\mathbf{1},X]\), \(W=\operatorname{diag}(\widetilde{w})\), and

\[
P =
\begin{bmatrix}
0 & 0 \\
0 & n\lambda I
\end{bmatrix}.
\]

The posterior precision and location are

\[
A = Z^\mathsf{T}WZ + P,
\qquad
m = A^{-1}Z^\mathsf{T}Wy.
\]

The player portion of \(m\) is the ridge solution. This equivalence is a
required numerical invariant in the implementation, not an empirical
coincidence.

After integrating out \(\sigma^2\), the parameter vector has a multivariate
Student-\(t\) posterior with \(n-1\) degrees of freedom. SciPy performs one
dense Cholesky factorization of the roughly 583-by-583 posterior precision
matrix. No MCMC, convergence diagnostic, or variational approximation is
needed.

## Reported uncertainty

The model reports:

- marginal posterior standard deviations and 90% credible intervals;
- posterior probability that each coefficient is positive;
- joint eligible-rank intervals and top-25/top-50 probabilities;
- held-out posterior predictive intervals and coverage;
- the full posterior location and precision Cholesky factor for future lineup
  contrasts and counterfactuals.

Rank summaries use joint posterior draws. Ranking marginal coefficient draws
independently would discard the strong covariance created by shared lineups.

## Bayesian posterior versus game bootstrap

The two distributions answer different questions:

| Distribution | Varies | Conditions on |
| --- | --- | --- |
| Bayesian posterior | latent coefficients and residual variance | observed design, lambda, likelihood, prior |
| Complete-game bootstrap | sampled games and refitted ridge estimates | estimator, lambda, allocation policy |

Regularization can make an estimator stable across game resamples while the
latent coefficients remain weakly separated under the posterior. Conversely,
the Bayesian posterior does not include uncertainty from changing lambda,
possession allocation, or model form. Both views belong in model review.

## Fit and reproduce

```bash
uv run nba-train-bayesian-rapm 2025-26

uv run --group docs nba-build-bayesian-rapm-case-study 2025-26
```

The trainer refuses a zero-lambda source because the all-player Gaussian prior
would be improper along unidentified directions. It validates the source ridge
manifest and the exact analytical stint hash before fitting.

## Next Bayesian model

The next Bayesian extension should estimate prior scales and partially pool
across seasons or player information. Separate offensive and defensive
effects, age curves, draft priors, and time-varying coefficients would move the
model beyond conjugacy. PyMC is the preferred framework at that point; this
SciPy implementation remains the exact reference case.

## References

- van Wieringen, W. N. (2015). [Lecture Notes on Ridge
  Regression](https://arxiv.org/abs/1509.09169). Includes the Gaussian-prior
  interpretation of ridge regression.
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., and
  Rubin, D. B. (2013). [Bayesian Data
  Analysis](https://sites.stat.columbia.edu/gelman/book/), third edition.
  Chapman & Hall/CRC.
- Hoerl, A. E., and Kennard, R. W. (1970). "Ridge Regression: Biased
  Estimation for Nonorthogonal Problems." *Technometrics*, 12(1), 55-67.
  [doi:10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634)
