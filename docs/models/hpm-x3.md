---
title: HPM x3
---

# HPM x3: ORB Claim Rebound Replacement

Last updated: 2026-08-14

HPM x3 is a conditional replacement experiment. It keeps the complete HPM v1
context contract, but replaces its entire rebounding bundle:

- `offensive_rebounds_per_100`
- `defensive_rebounds_per_100`
- `sqrt_offensive_rebounds`
- `sqrt_defensive_rebounds`
- `rebounding_usage_interaction`

with one forward-safe player-profile feature:

\[
\operatorname{ORBClaims}(U)=\sum_{i \in U}\operatorname{ORB\%}_i.
\]

All shooting, creation, usage, turnover, defensive-event, uncertainty, and
other HPM v1 context features remain. This directly tests whether summed ORB%
claims are a better rebounding representation when the rest of the incumbent
context model is held intact.

## Run

```bash
uv run nba-train-hpm-x3 --through-season 2025-26
```

The recursive artifact is written under
`artifacts/models/forward_hpm_x3_v1_orb_claim_replacement/`, then replayed on
the frozen 2023-24 through 2025-26 targets against HPM v1.

## Frozen Result

Artifact:
`artifacts/models/forward_hpm_x3_v1_orb_claim_replacement/2025-26/forward-hpm-x3-v1-orb-claim-replacement-2025-26-20260815T055927Z-d20c526b`.

| Metric | HPM v1 | HPM x3 | x3 minus v1 | Paired 95% interval | P(x3 better) |
| --- | ---: | ---: | ---: | --- | ---: |
| Full-game RMSE | 14.380249 | 14.377365 | -0.002884 | [-0.016831, +0.011267] | 65.17% |
| Winner accuracy | 67.84% | 67.73% | -0.11 pp | [-0.54 pp, +0.31 pp] | 28.37% |
| Possession RMSE | 1.198010 | 1.198000 | -0.000011 | [-0.000020, -0.000001] | 98.75% |
| Possession MAE | 1.141402 | 1.141400 | -0.000002 | [-0.000012, +0.000008] | 66.12% |
| Team NetRtg RMSE | 3.429015 | 3.422924 | -0.006091 | -- | -- |
| Pythagorean-win RMSE | 7.353235 | 7.326897 | -0.026338 | -- | -- |

x3 is the best point estimate on regular possession RMSE, eligible-game RMSE,
full-game RMSE, team NetRtg RMSE, and Pythagorean-win RMSE among the current
HPM candidates. Its full-game confidence interval still crosses zero, so it is
not statistically established as superior to HPM v1. It is nevertheless the
published production reference under the documented parsimony non-inferiority
rule: it removes the coherent five-term rebounding bundle while its `+0.0113`
upper full-game-RMSE interval remains below the `+0.015` practical margin.
