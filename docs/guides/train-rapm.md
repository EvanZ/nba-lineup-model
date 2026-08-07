---
last_updated: "2026-07-29"
---

# Train the RAPM Baselines

The first training command runs three regular-season models:

1. a possession-weighted mean;
2. signed schedule-adjusted team strengths;
3. signed one-number player RAPM.

```bash
uv run nba-train-rapm 2025-26
```

The command first rebuilds and validates the RAPM stint mart. It then uses
three expanding chronological validation folds, holds out the final 15% of
games for one test evaluation, and refits the selected models on the complete
regular season for ratings and rankings.

## Sparse encoding

The player design matrix has one column per player appearing in the modeling
sample. A row contains `+1` for each home player and `-1` for each away player.
The team matrix uses the same signs with one column per team. Both are SciPy
CSR matrices consumed directly by Scikit-learn's sparse `lsqr` ridge solver.

The fitted intercept estimates home-court advantage. The unpenalized lambda
endpoint is available for the team baseline; the player model selects from the
same normalized grid.

## Chronological validation

Games, rather than stints, are the split unit. Every stint from one game stays
in one fold. By default:

- three expanding validation folds each hold out the next 10% of games;
- the final 15% of games remain untouched during lambda selection;
- lambda minimizes possession-weighted validation MSE.

The normalized objective is weighted mean squared error plus lambda times the
squared coefficient norm. This keeps lambda comparable as training folds grow.

Customize the experiment with:

```bash
uv run nba-train-rapm 2025-26 \
  --cv-folds 3 \
  --validation-fraction 0.10 \
  --test-fraction 0.15 \
  --minimum-ranking-possessions 500
```

## Outputs

The analytical input is written under `data/analytical/rapm_stints/`. Each
immutable run is written under:

```text
artifacts/models/rapm/{season}/{run_id}/
```

It contains CV results, final-test metrics and predictions, player rankings,
team ratings, game split assignments, sparse column mappings, model
parameters, and a hash-validated manifest. `latest.json` identifies the newest
successful run.

These outputs are ignored by Git. A ranking is published in the documentation
only after its run has been reviewed and explicitly promoted.

## Diagnose a ranking

Run the diagnostic suite against the latest successful model for a season:

```bash
uv run nba-diagnose-rapm 2025-26
```

The command validates the source model manifest and analytical-data hash before
running. It does not modify the trained model. Each diagnostic run is written
to an immutable directory under:

```text
artifacts/reports/rapm/{season}/{diagnostics_run_id}/
```

The season-level `latest.json` points to the newest validated report. Use
`--source-run-id` to diagnose an older model run explicitly.

### Ranking diagnostics

**Lambda sensitivity.** Refit full-season RAPM across a fixed ridge penalty
path. Coefficient correlation, Spearman rank correlation, mean absolute rank
change, and top-25/top-50 overlap show whether a player's apparent value
depends heavily on the selected penalty. This is sensitivity analysis, not a
second lambda-selection procedure. It is the tabular analogue of the ridge
trace introduced by [Hoerl and Kennard (1970)](#hoerl-kennard-1970), applied
to the basketball RAPM setting developed by [Sill (2010)](#sill-2010).

**Chronological stability.** Refit the selected lambda on each expanding
training window, the final training sample, and the full season. Wide
coefficient or eligible-rank ranges identify estimates that emerge late or
move substantially as more games arrive. The expanding-window design follows
the rolling-origin out-of-sample evaluation principles reviewed by
[Tashman (2000)](#tashman-2000); using it to inspect coefficient paths is our
diagnostic adaptation.

**Game-block bootstrap.** Sample complete games with replacement and refit
RAPM. Keeping every stint from a sampled game together respects the dependence
within games. The output includes coefficient intervals, positive-value
probability, top-25/top-50 probability, and eligible-rank intervals. Rank
statistics use the source model's fixed exposure-eligible player set. These
quantify sampling stability; they are not classical independent-row confidence
intervals. Resampling games as intact clusters follows the clustered-bootstrap
framework studied by [Field and Welsh (2007)](#field-welsh-2007).

**Context concentration.** Measure each player's teammate and lineup
diversity, most-common teammate and lineup shares, effective lineup count, and
share of exposure beside an eligible top-25 teammate. Highly concentrated
contexts make it harder for RAPM to separate the player from recurring
teammates. `effective_lineup_count` is the exponential of Shannon entropy
[Shannon (1948)](#shannon-1948), equivalent to the order-one effective
diversity described by [Hill (1973)](#hill-1973). Its use for lineup
identification is project-specific.

**Raw versus adjusted value.** Compare raw on-court net rating with the RAPM
coefficient and rank. Large adjustments show where opponent and teammate
control matters most. They can be legitimate schedule corrections or a signal
to inspect identification and data quality. This comparison reflects the
original motivation for adjusted plus-minus in
[Rosenbaum (2004)](#rosenbaum-2004) and its regularized extension in
[Sill (2010)](#sill-2010).

**Influence.** Rank player-stint and player-game observations using
ridge-leverage and residual-gradient screens. For the highest-ranked eligible
players, exact delete-one-game refits report how much the player's coefficient
changes. A large deletion effect means a small number of games materially
supports the estimate. The residual, leverage, and deletion logic descends from
[Cook (1977)](#cook-1977), with ridge-specific influence measures developed by
[Walker and Birch (1988)](#walker-birch-1988). Our game aggregation and
screen-then-refit procedure is a computational adaptation.

**Possession-allocation sensitivity.** Rebuild the modeling target under five
policies for possessions that cross lineup boundaries:

- `equal_segments` divides exposure across segments while retaining segment points;
- `starting_lineup` assigns the full possession to its first lineup;
- `terminal_lineup` assigns it to its last lineup;
- `boundary_split` divides the result equally between the first and last lineups;
- `exclude_multi_lineup` removes possessions observed under multiple lineups.

Compare coefficient and eligible-rank changes, plus each policy's RAPM skill
against its own mean baseline. Eligibility remains fixed to the source model.
Weighted RMSE levels are not directly comparable across policies because the
policies construct different stint targets. Treating multiple defensible data
constructions as a sensitivity set follows the multiverse-analysis principle
of [Steegen et al. (2016)](#steegen-et-al-2016); the five basketball allocation
policies are defined by this project.

### Diagnostic outputs

| File | Purpose |
| --- | --- |
| `player_diagnostics.parquet` | One-row-per-player review table combining the main signals |
| `lambda_coefficients.parquet` | Player coefficients and ranks at every tested lambda |
| `lambda_summary.parquet` | Aggregate agreement with the selected lambda |
| `chronological_coefficients.parquet` | Coefficients and exposure by expanding window |
| `chronological_summary.parquet` | Player-level time stability ranges |
| `bootstrap_coefficients.parquet` | Every player coefficient and rank in every resample |
| `bootstrap_summary.parquet` | Player intervals and stability probabilities |
| `context_concentration.parquet` | Teammate and lineup concentration measures |
| `raw_adjusted.parquet` | Raw on-court and adjusted RAPM comparison |
| `influential_stints.parquet` | Highest-influence stints for every player |
| `influential_games.parquet` | Highest-influence games for every player |
| `delete_game_influence.parquet` | Exact delete-one-game coefficient changes |
| `allocation_coefficients.parquet` | Player sensitivity to allocation policy |
| `allocation_metrics.parquet` | Mean and RAPM test metrics within each policy |
| `allocation_summary.parquet` | Possessions and exposure changed by each allocation policy |
| `manifest.json` | Configuration, source hashes, row counts, and artifact hashes |

Defaults use 200 bootstrap samples with seed `7`, five lambda values, exact
game deletions for the top 25 eligible players, and all five allocation
policies. For a faster development run:

```bash
uv run nba-diagnose-rapm 2025-26 \
  --bootstrap-samples 10 \
  --influence-player-count 5 \
  --delete-games-per-player 1
```

## Build a case study

Generate a committed documentation page and charts from one validated
diagnostics run:

```bash
uv run --group docs nba-build-rapm-case-study 2025-26 \
  --diagnostics-run-id diagnostics-2025-26-20260728T043406Z-32196bfa
```

The generator validates every diagnostics artifact before reading it. It
publishes `docs/models/{season}-rapm-case-study.md` and deterministic SVG
charts under `docs/assets/images/rapm/{season}/`. The generated page records
the diagnostics run, source model run, and manifest hash. Unlike large model
artifacts, this small reviewed documentation output belongs in Git.

The case-study review bands are deliberately editorial and remain separate
from the model. They summarize the initial eligible top 25 as a stable core,
qualified estimates, or fragile rank positions without changing RAPM
coefficients or claiming causal player value.

## Train the exact Bayesian baseline

After selecting and reviewing a positive ridge lambda, fit its conjugate
Bayesian counterpart:

```bash
uv run nba-train-bayesian-rapm 2025-26
```

By default, the command validates and uses the latest immutable ridge run. Pin
the source and posterior simulation explicitly for a published analysis:

```bash
uv run nba-train-bayesian-rapm 2025-26 \
  --source-run-id baseline-2025-26-20260727T230533Z-72eac627 \
  --posterior-draws 4000 \
  --posterior-seed 17 \
  --credible-interval 0.90
```

The Gaussian likelihood and prior are conjugate, so SciPy computes the exact
posterior location, covariance, marginal intervals, and independent joint
draws. PyMC and Pyro are intentionally not dependencies of this baseline.

Each immutable run is written under:

```text
artifacts/models/bayesian_rapm/{season}/{run_id}/
```

It contains:

- posterior coefficient and rank summaries for every player;
- ridge-equivalence and held-out point-prediction checks;
- held-out posterior predictive calibration;
- stint-level held-out predictive intervals;
- the player-column mapping and model parameters;
- a compressed posterior location and precision Cholesky factor;
- a hash-validated manifest linked to the exact source ridge run.

Generate the comparison case study from validated Bayesian and diagnostics
runs:

```bash
uv run --group docs nba-build-bayesian-rapm-case-study 2025-26
```

See [Bayesian RAPM methodology](../models/bayesian-rapm.md) for the equations
and [What Bayesian RAPM Adds to the Same Ridge Ranking](../models/2025-26-bayesian-rapm-case-study.md)
for the 2025-26 analysis.

## Automated test coverage

Run the focused diagnostics tests with:

```bash
uv run pytest tests/test_rapm_diagnostics.py
uv run pytest tests/test_rapm_case_study.py
uv run pytest tests/test_bayesian_rapm.py
uv run pytest tests/test_bayesian_case_study.py
```

The tests verify the point and exposure semantics of every allocation policy;
the selected-lambda identity and expanding-window output contracts;
deterministic game-block bootstrap results; context, raw-adjustment, leverage,
and exact game-deletion invariants; and the one-row-per-player consolidated
report. Case-study tests cover review-band classification, provenance
rendering, and deterministic chart output. Bayesian tests require exact
ridge-posterior location agreement, deterministic joint draws, predictive
coverage artifacts, provenance rendering, and deterministic comparison charts.
The full test suite remains:

```bash
uv run pytest
```

## Methodological references

<span id="hoerl-kennard-1970"></span>

- Hoerl, A. E., and Kennard, R. W. (1970). "Ridge Regression: Biased
  Estimation for Nonorthogonal Problems." *Technometrics*, 12(1), 55-67.
  [doi:10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634)

<span id="tashman-2000"></span>

- Tashman, L. J. (2000). "Out-of-Sample Tests of Forecasting Accuracy: An
  Analysis and Review." *International Journal of Forecasting*, 16(4),
  437-450.
  [doi:10.1016/S0169-2070(00)00065-0](https://doi.org/10.1016/S0169-2070(00)00065-0)

<span id="field-welsh-2007"></span>

- Field, C. A., and Welsh, A. H. (2007). "Bootstrapping Clustered Data."
  *Journal of the Royal Statistical Society: Series B*, 69(3), 369-390.
  [doi:10.1111/j.1467-9868.2007.00593.x](https://doi.org/10.1111/j.1467-9868.2007.00593.x)

<span id="shannon-1948"></span>

- Shannon, C. E. (1948). "A Mathematical Theory of Communication." *Bell
  System Technical Journal*, 27, 379-423 and 623-656.
  [Part I](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x);
  [Part II](https://doi.org/10.1002/j.1538-7305.1948.tb00917.x).

<span id="hill-1973"></span>

- Hill, M. O. (1973). "Diversity and Evenness: A Unifying Notation and Its
  Consequences." *Ecology*, 54(2), 427-432.
  [doi:10.2307/1934352](https://doi.org/10.2307/1934352)

<span id="rosenbaum-2004"></span>

- Rosenbaum, D. T. (2004). "Measuring How NBA Players Help Their Teams Win."
  *82games.com*.
  [Original article](https://www.82games.com/comm30.htm)

<span id="sill-2010"></span>

- Sill, J. (2010). "Improved NBA Adjusted +/- Using Regularization and
  Out-of-Sample Testing." *MIT Sloan Sports Analytics Conference*.
  [Conference paper page](https://www.sloansportsconference.com/research-papers/improved-nba-adjusted-using-regularization-and-out-of-sample-testing)

<span id="cook-1977"></span>

- Cook, R. D. (1977). "Detection of Influential Observation in Linear
  Regression." *Technometrics*, 19(1), 15-18.
  [doi:10.1080/00401706.1977.10489493](https://doi.org/10.1080/00401706.1977.10489493)

<span id="walker-birch-1988"></span>

- Walker, E., and Birch, J. B. (1988). "Influence Measures in Ridge
  Regression." *Technometrics*, 30(2), 221-227.
  [doi:10.1080/00401706.1988.10488370](https://doi.org/10.1080/00401706.1988.10488370)

<span id="steegen-et-al-2016"></span>

- Steegen, S., Tuerlinckx, F., Gelman, A., and Vanpaemel, W. (2016).
  "Increasing Transparency Through a Multiverse Analysis." *Perspectives on
  Psychological Science*, 11(5), 702-712.
  [doi:10.1177/1745691616658637](https://doi.org/10.1177/1745691616658637)
