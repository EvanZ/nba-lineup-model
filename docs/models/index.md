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

All three use the same regular-season stint target, chronological game splits,
possession weights, and test window. This makes incremental predictive value
measurable rather than inferred from a plausible leaderboard.

See [Baseline methodology](baselines.md) for the model contract, the
[2025-26 RAPM case study](2025-26-rapm-case-study.md) for a worked diagnostic
review, and [Promoted rankings](rankings.md) for reviewed public releases.
