---
title: HPM x4
---

# HPM x4: ORB Claims + Blocks

Last updated: 2026-08-15

HPM x4 is the defensive-event successor to the promoted HPM x3 feature
contract. It retains x3's single summed player ORB% claim feature in place of
the original five-feature rebounding bundle, then removes `steals_per_100`
from the remaining HPM v1 context signals. `blocks_per_100` is retained as the
sole defensive-event signal.

This is a simplification experiment, not a new defensive theory. In the
three-season fixed-model component audit, neutralizing blocks worsened
full-game margin RMSE by `+0.0237` (paired 95% interval `[+0.0028, +0.0446]`),
whereas neutralizing steals had an inconclusive `+0.0199` change
(`[-0.0206, +0.0603]`). The recursive refit tests whether that evidence holds
once the remaining contextual functions can adapt.

## Contract

| Feature family | HPM x3 | HPM x4 |
| --- | --- | --- |
| Rebounding | Summed ORB% player claims | Unchanged |
| Steals per 100 | Included | Removed |
| Blocks per 100 | Included | Retained |
| Shooting, passing, usage, turnovers, uncertainty | Included | Unchanged |

## Run

```bash
uv run nba-train-hpm-x4 --through-season 2025-26
```

The recursive artifact is written under
`artifacts/models/forward_hpm_x4_orb_claim_blocks_only/`. It will be evaluated
against HPM x3 on the frozen 2023-24 through 2025-26 regular seasons before
any promotion decision.

## Frozen Result

Artifact:
`artifacts/models/forward_hpm_x4_orb_claim_blocks_only/2025-26/forward-hpm-x4-orb-claim-blocks-only-2025-26-20260815T163858Z-229b4737`.

| Metric | HPM x3 | HPM x4 | x4 minus x3 | Paired 95% interval | P(x4 better) |
| --- | ---: | ---: | ---: | --- | ---: |
| Full-game RMSE | 14.377365 | 14.403557 | +0.026192 | [-0.010199, +0.063224] | 8.05% |
| Winner accuracy | 67.73% | 67.53% | -0.20 pp | [-0.85 pp, +0.46 pp] | 26.18% |
| Possession RMSE | 1.198000 | 1.198007 | +0.000007 | [-0.000012, +0.000025] | 23.98% |
| Possession MAE | 1.141400 | 1.141436 | +0.000036 | [+0.000017, +0.000055] | 0.01% |

The refit does not support removing steals. HPM x3 remains the published
reference: a fixed model can rely on blocks more than steals, while a full
recursive refit can still exploit the steals feature as a correlated defensive
proxy in combination with its re-estimated player prior and other context
functions.

Frozen replay artifact:
`artifacts/models/analysis/hpm_x4_frozen/frozen_multiseason_backtest/2023-24_to_2025-26/frozen_multiseason_backtest-2023-24-to-2025-26-20260815T164505Z-fad3f9ab`.
