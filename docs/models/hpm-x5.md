---
title: HPM x5
---

# HPM x5: Interaction-Only Creation

Last updated: 2026-08-15

HPM x5 builds on the promoted HPM x3 context contract. It removes the direct
lineup totals `assists_per_100` and `top_two_assists`, while retaining the
existing credible-shooter-by-top-two-assists interaction. This keeps the
creation signal that matters only when paired with lineup shooting depth,
rather than separately fitting three closely related passing response
functions.

The fixed-model audit found no material full-game reliance for raw assists
(`+0.0140`, paired 95% interval `[-0.0241, +0.0537]`) or standalone top-two
assists (`+0.0048`, `[-0.0349, +0.0459]`). It found clear reliance for the
shooter-passing interaction (`+0.0148`, `[+0.0025, +0.0268]`). The recursive
refit determines whether that simplification survives retraining.

## Run

```bash
uv run nba-train-hpm-x5 --through-season 2025-26
```

The artifact is written under
`artifacts/models/forward_hpm_x5_orb_claim_interaction_creation/` and will be
compared with x3 on the frozen 2023-24 through 2025-26 regular seasons.

## Frozen Result

| Metric | HPM x3 | HPM x5 | x5 minus x3 | Paired 95% interval | P(x5 better) |
| --- | ---: | ---: | ---: | --- | ---: |
| Full-game RMSE | 14.377365 | 14.380019 | +0.002655 | [-0.021433, +0.026680] | 42.29% |
| Winner accuracy | 67.73% | 67.99% | +0.26 pp | [-0.34 pp, +0.83 pp] | 79.37% |
| Possession RMSE | 1.198000 | 1.198006 | +0.000006 | [-0.000009, +0.000020] | 21.20% |
| Possession MAE | 1.141400 | 1.141429 | +0.000029 | [+0.000014, +0.000045] | 0.02% |

Although the point estimate is close, its full-game interval exceeds the
`+0.015` parsimony non-inferiority margin. HPM x3 remains the reference.
