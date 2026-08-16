---
title: HPM x1
---

# HPM x1: ORB Claim Context

Last updated: 2026-08-14

HPM x1 is a deliberately narrow context experiment. It retains the complete
value-conditioned aging RAPM prior and exposure-gated cold-start procedure, but
its contextual model receives exactly one side feature:

\[
\operatorname{ORBClaims}(U) = \sum_{i \in U}\operatorname{ORB\%}_i.
\]

The individual `ORB%` values are forward-safe player profile estimates from
the prior season. The sum is measured in percentage points and is **not** a
predicted team offensive-rebound percentage: overlapping player claims mean it
is a claim total, not a realized rate.

No shooting, passing, usage, defensive-event, defensive-rebounding,
square-root-rebound, or rebound-by-usage context feature enters HPM x1. The
bounded hierarchical P-spline fit learns the response to the relative
home-minus-away ORB claim total with the same forward seasonal hierarchy as
HPM v1.

## Run

```bash
uv run nba-train-hpm-x1 --through-season 2025-26
```

The recursive artifact is written under
`artifacts/models/forward_hpm_x1_orb_claim_total/`. It will be evaluated on
the frozen 2023-24, 2024-25, and 2025-26 regular-season holdouts before any
promotion decision.

## Frozen Result

The relevant ablation comparison is the **Controlled No-Context
Value-Conditioned RAPM**, not the broader Complete Player-Prior RAPM baseline.
That control is being rebuilt from the corrected stint data and will be replayed
on the same frozen 2023-24 through 2025-26 targets before a standalone-context
conclusion is drawn.

Original HPM v1 remains materially stronger on the primary full-game and team
net-rating metrics: 14.3802 versus 14.4665 full-game RMSE and 3.4290 versus
3.6653 team NetRtg RMSE. See the [Frozen Model Tournament](frozen-model-tournament.md)
for the broader model comparison.
