---
last_updated: "2026-08-18"
---

# NAIL-RAPM Context Regularization

This controlled study tunes the inherited `alpha=10000` Ridge penalty in
[NAIL-RAPM v1.1](nail-rapm-v11-profile-padding.md). Nothing else changes: the
recursive player prior, statistic-specific profile padding, annual RAPM lambda,
14 linear context coordinates, and frozen evaluation contract are held fixed.

## Why Raw 10,000 Is Not Portable

For stint residual \(y_s\), relative lineup profile \(x_s\), possession weight
\(w_s\), and context coefficients \(\beta_t\), scikit-learn's weighted Ridge
objective is proportional to

\[
\sum_{s \in t} w_s
\left(y_s-x_s^\mathsf{T}\beta_t\right)^2
+\alpha\lVert\beta_t\rVert_2^2.
\]

A fixed \(\alpha\) becomes weaker when a season contributes more possession
weight and stronger in a shortened season. The candidate contract instead
normalizes the data loss:

\[
\mathcal L_t(\beta_t)
=
\frac{1}{W_t}\sum_{s \in t} w_s
\left(y_s-x_s^\mathsf{T}\beta_t\right)^2
+\lambda_C\lVert\beta_t\rVert_2^2,
\qquad W_t=\sum_{s \in t}w_s.
\]

The implementation adds the reversed signed orientation of every stint to
enforce exact antisymmetry. That doubles both SSE and weight without changing
the mean loss, so the ordinary scikit-learn Ridge fit uses
\(\alpha_t=2\lambda_C W_t=\lambda_C W_t^{\pm}\). Artifacts persist the
augmented weight (W_t^{\pm}), \(\lambda_C\), and \(\alpha_t\) for every
season.

## Selection Boundary

Candidate recursive states are trained only through 2022-23. Each candidate is
then replayed over 23 target seasons from 2000-01 through 2022-23, always using
the player prior and context state available before the target season.
Target-season realized lineup allocation remains the same oracle input used by
the public frozen leaderboard.

The primary selection score is the equal-season mean full-game margin squared
error. The exact minimum is reported, and the one-standard-error rule chooses
the strongest penalty whose paired season-by-season loss difference remains
within one standard error of that minimum. Pairing isolates uncertainty in the
candidate comparison instead of allowing variation in season difficulty to
inflate the tolerance. The 2023-24 through 2025-26 seasons remain untouched
until one value has been selected.

<!-- nail-context-regularization-results:start -->
## Results

The rolling selector evaluated 23 strictly forward target seasons from 2000-01
through 2022-23. Lower RMSE is better. The raw-α incumbent is retained as a
diagnostic but is not eligible for selection because it follows a different,
season-size-dependent regularization contract.

| Contract | Context penalty | Full-history RMSE | Recent-10 RMSE | Winner accuracy | Paired MSE delta ± SE | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Normalized | 0.00275 | 12.5398 | 13.0594 | 65.66% | 1.261 ± 0.577 | Outside one-SE band |
| Normalized | 0.022 | 12.5138 | 13.0249 | 65.63% | 0.609 ± 0.494 | Outside one-SE band |
| Raw α=10,000 control | season-dependent | 12.5243 | 13.0528 | 65.83% | n/a | Diagnostic only |
| Normalized | 0.044 | 12.5246 | 13.0522 | 65.85% | 0.880 ± 0.353 | Outside one-SE band |
| Normalized | 0.088 | 12.5136 | 13.0412 | 65.82% | 0.606 ± 0.272 | Outside one-SE band |
| Normalized | 0.176 | 12.4999 | 13.0272 | 65.96% | 0.263 ± 0.194 | Outside one-SE band |
| Normalized | 0.352 | 12.4910 | 13.0167 | 66.07% | 0.041 ± 0.101 | Inside one-SE band |
| **Normalized** | **0.704** | **12.4894** | 13.0117 | 66.12% | **0.000 ± 0.000** | **Selected; exact minimum** |
| Normalized | 1.408 | 12.4940 | **13.0117** | **66.15%** | 0.114 ± 0.096 | Outside one-SE band |

The selected value is interior to the tested grid. It is both the exact
full-history minimum and the strongest candidate within the paired one-SE
band, so the decision does not depend on an arbitrary search boundary.

### Frozen Evaluation

The selected penalty was then fitted through 2025-26 and compared with
published v1.1 over the untouched 2023-24 through 2025-26 cohorts.

| Cohort and metric | Published v1.1 | Normalized 0.704 | Difference |
| --- | ---: | ---: | ---: |
| Regular possession RMSE | **1.197979** | 1.198002 | +0.000023 |
| Regular eligible-game RMSE | **14.0864** | 14.1339 | +0.0475 |
| Regular full-game RMSE | **14.3236** | 14.3822 | +0.0587 |
| Regular winner accuracy | **68.53%** | 68.01% | -0.51 pp |
| Regular team NetRtg RMSE | **3.3898** | 3.5234 | +0.1336 |
| Regular Pythagorean-win RMSE | **7.2757** | 7.4597 | +0.1840 |
| Playoff possession RMSE | 1.192740 | **1.192726** | -0.000014 |
| Playoff eligible-game RMSE | 16.6267 | **16.5979** | -0.0288 |

The candidate improves the pooled playoff metrics slightly but loses every
reported regular-season metric. Because the project selects public models on
the regular-season frozen contract, the playoff movement is not sufficient for
promotion.

### Paired Bootstrap

The regular-season audit resamples games within each frozen season for 10,000
paired draws. Positive RMSE differences favor the incumbent.

| Metric | Challenger minus v1.1 | 95% interval | P(challenger better) |
| --- | ---: | ---: | ---: |
| Full-game RMSE | +0.0587 | [+0.0152, +0.1008] | 0.49% |
| Possession RMSE | +0.000023 | [+0.000002, +0.000045] | 1.67% |
| Possession MAE | +0.000142 | [+0.000118, +0.000167] | 0.00% |
| Winner accuracy | -0.51 pp | [-1.22 pp, +0.20 pp] | 7.45% |

The full-game interval excludes zero in the wrong direction. The normalized
contract is therefore a well-specified negative result, not an ambiguous tie.

### Decision

Retain the fixed raw `alpha=10000` model as **NAIL-RAPM v1.1**. Do not promote
the normalized `lambda_C=0.704` candidate. Normalization fixes the semantic
problem in the inherited hyperparameter, but the old season-size dependence
appears to have acted as useful implicit regularization for this particular
recursive pipeline.

## Fixed-Raw Sensitivity Audit

A follow-up audit tests raw `alpha` values of 1,000, 5,000, 10,000, and 20,000
under the original weighted-sum objective. This directly checks the local
sensitivity of the inherited 10,000 value without mixing raw and normalized
penalty semantics.

An exact-equivalence test also fits the same weighted design through both code
paths: raw `alpha=10000` and its normalized-lambda conversion produce identical
coefficients and predictions to numerical precision. The normalized result was
therefore not caused by an incorrect Ridge conversion.

### Pre-Frozen Selection

The four fixed values were replayed over the same 23 seasons ending in 2022-23.
Raw `alpha=5000` is the exact minimum, while `alpha=20000` is the strongest
penalty inside the paired one-standard-error band.

| Raw context alpha | Full-history RMSE | Recent-10 RMSE | Winner accuracy | Paired MSE delta +/- SE | Decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1,000 | 12.5342 | 13.0524 | 65.66% | 0.534 +/- 0.156 | Outside one-SE band |
| **5,000** | **12.5128** | **13.0246** | 65.61% | **0.000 +/- 0.000** | Exact minimum |
| 10,000 | 12.5243 | 13.0528 | **65.83%** | 0.286 +/- 0.253 | Outside one-SE band |
| 20,000 | 12.5142 | 13.0427 | 65.83% | 0.035 +/- 0.307 | Selected by one-SE rule |

### Frozen Sensitivity

All four fixed values were then scored on the untouched three-season frozen
window. Lower is better except winner accuracy.

| Raw alpha | Poss. RMSE | Eligible-game RMSE | Full-game RMSE | Winner accuracy | Team NetRtg RMSE | Pythagorean-win RMSE | Playoff poss. RMSE | Playoff game RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 1.197999 | 14.1017 | 14.3389 | 67.99% | 3.4009 | 7.2776 | 1.192745 | 16.6197 |
| 5,000 | 1.197994 | 14.0949 | 14.3300 | 67.84% | 3.4156 | 7.2821 | 1.192727 | **16.6090** |
| **10,000** | **1.197979** | **14.0864** | **14.3236** | **68.53%** | **3.3898** | **7.2757** | 1.192740 | 16.6267 |
| 20,000 | 1.197980 | 14.0907 | 14.3296 | 68.24% | 3.4095 | 7.2966 | **1.192736** | 16.6187 |

The regular-season result is locally flat, but `alpha=10000` remains the only
candidate that wins all six primary regular metrics. The pre-frozen-selected
`alpha=20000` is effectively tied on possession and full-game RMSE, yet it does
not improve either and loses 0.28 percentage points of winner accuracy.

### Fixed-Grid Bootstrap

The paired 10,000-draw regular-season bootstrap compares each challenger with
`alpha=10000`. Positive RMSE differences and negative accuracy differences
favor the incumbent.

| Challenger | Full-game RMSE delta (95% interval) | Poss. RMSE delta (95% interval) | Winner-accuracy delta (95% interval) |
| ---: | ---: | ---: | ---: |
| 1,000 | +0.0153 [-0.0135, +0.0432] | +0.000020 [+0.000004, +0.000036] | -0.54 pp [-1.20, +0.11] |
| 5,000 | +0.0064 [-0.0179, +0.0303] | +0.000015 [+0.000002, +0.000028] | -0.68 pp [-1.25, -0.11] |
| 20,000 | +0.0060 [-0.0020, +0.0141] | +0.000001 [-0.000003, +0.000006] | -0.28 pp [-0.63, +0.06] |

This confirms that 10,000 is not a sharply identified optimum, but no tested
fixed value supplies evidence for replacing it. NAIL-RAPM v1.1 therefore keeps
`alpha=10000`.

Artifacts:

- Rolling selection: `artifacts/models/nail_context_regularization_study/2000-01_to_2022-23/nail-context-regularization-2000-01-to-2022-23-20260818T233035Z-73b0dd4a`
- Selected recursion: `artifacts/models/forward_nail_rapm_v11_normalized_context_lambda_0p704/2025-26/forward-nail-rapm-v11-normalized-context-lambda-0p704-2025-26-20260819T002204Z-fa6a33f2`
- Frozen replay: `artifacts/models/nail_context_regularization_frozen_backtest/frozen_multiseason_backtest/2023-24_to_2025-26/frozen_multiseason_backtest-2023-24-to-2025-26-20260819T003135Z-affd58c8`
- Paired bootstrap: `artifacts/models/nail_context_regularization_bootstrap/2023-24_to_2025-26/nail-context-regularization-bootstrap-20260819T003228Z-48ad7562`
- Fixed-grid selection: `artifacts/models/nail_fixed_context_regularization_study/2000-01_to_2022-23/nail-fixed-context-2000-01-to-2022-23-20260819T032317Z-06a3ffb1`
- Fixed-grid frozen replay: `artifacts/models/nail_fixed_context_regularization_frozen/frozen_multiseason_backtest/2023-24_to_2025-26/frozen_multiseason_backtest-2023-24-to-2025-26-20260819T033440Z-8a79284e`
- Fixed-grid paired bootstrap: `artifacts/models/nail_fixed_context_regularization_bootstrap/2023-24_to_2025-26/nail-fixed-context-bootstrap-20260819T033523Z-eef47736`
<!-- nail-context-regularization-results:end -->

## Reproduction

See [Tune NAIL Context Regularization](../guides/tune-nail-context-regularization.md)
for the candidate training, rolling selection, frozen replay, and paired
bootstrap commands.
