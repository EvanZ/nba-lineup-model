# Models

The modeling program starts with transparent predictive baselines before adding
player priors, separate offensive and defensive effects, or nonlinear lineup
interactions.

## Baseline ladder

| Model | Information available |
| --- | --- |
| Mean | Training-window average home net rating |
| Team | Signed home and away team identities |
| RAPM | Signed identities of all ten players |
| Bayesian RAPM | RAPM posterior uncertainty under an explicit Gaussian model |

All four use the same regular-season stint target, chronological game splits,
possession weights, and test window. The first three compare predictive
information sets. Bayesian RAPM deliberately retains the ridge point estimate
and adds coefficient, rank, and predictive uncertainty.

See [Baseline methodology](baselines.md) for the model contract, the
[2025-26 RAPM case study](2025-26-rapm-case-study.md) for a worked diagnostic
review, the [Bayesian RAPM methodology](bayesian-rapm.md) and
[2025-26 Bayesian case study](2025-26-bayesian-rapm-case-study.md) for the
probabilistic baseline, and [Promoted rankings](rankings.md) for reviewed
public releases.
