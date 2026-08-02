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
| RAPM aging model | Forward player-season priors from prior RAPM and age |
| Age-informed prior RAPM | Lineup RAPM centered on the frozen aging forecast |
| Additive neural RAPM | Possession-level signed scalar player embeddings |
| Deep Sets | Nonlinear permutation-invariant lineup aggregation |
| CatBoost | Categorical player states and boosted-tree interactions |
| RAPM + Transformer | Frozen ridge prediction plus contextual player-player attention residual |

The first four use the same regular-season stint target, chronological game
splits, possession weights, and test window. The first three compare predictive
information sets. Bayesian RAPM deliberately retains the ridge point estimate
and adds coefficient, rank, and predictive uncertainty. Neural models move to
single-lineup possession rows while retaining chronological game boundaries.
RAPM + Transformer reconnects the two samples by converting a leakage-safe,
stint-weighted ridge prediction into the possession target and learning only
its nonlinear residual.

See [Baseline methodology](baselines.md) for the model contract, the
[2025-26 RAPM case study](2025-26-rapm-case-study.md) for a worked diagnostic
review, the [Bayesian RAPM methodology](bayesian-rapm.md) and
[2025-26 Bayesian case study](2025-26-bayesian-rapm-case-study.md) for the
probabilistic baseline, and [Promoted rankings](rankings.md) for reviewed
public releases. [Neural Networks](neural-networks.md) defines the staged
additive, Deep Sets, and RAPM + Transformer program. [Tree Models](tree-models.md)
defines the orthogonal CatBoost baseline. [Leaderboard](leaderboard.md) defines
the shared regular-holdout and playoff metrics and maintains the cross-model
scoreboard. [RAPM Aging Model](aging-model.md) defines the first temporal player
prior, while the [Modeling Roadmap](roadmap.md) records the approved joint
dynamic RAPM extension.
