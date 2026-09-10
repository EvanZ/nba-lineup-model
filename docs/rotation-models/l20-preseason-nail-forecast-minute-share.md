---
last_updated: "2026-09-08"
---

# L20-NAIL-MSP v0.2

This slice repeats [v0.1](l20-nail-minute-share.md) with the exact NAIL
forecast that a preseason product can display for the target season.

For target season \(t\), each player receives:

\[
R^{\mathrm{pre}}_{i,t} = \mu^{\mathrm{forward}}_{i,t}
+ \Delta^{\mathrm{additive}}_{i,t-1},
\]

where the first term is the value-conditioned forward prior and the second is
the frozen additive profile computed only from information through \(t-1\).
There is no target-season RAPM update or non-additive lineup observation.

The forecast applies the same rating tilt and expanding-window parameter
selection as v0.1. It is the only rating-aware allocation version eligible to
replace the Win Projections baseline, because it matches the public preseason
NAIL contract rather than using a completed rating as a proxy.

```bash
uv run nba-evaluate-l20-preseason-nail-minute-share
```

## Initial Frozen Result

The expanding-window replay covered ten frozen holdouts, `2016-17` through
`2025-26`. For each holdout, both the rating-aware candidate and the prior-only
control selected their own parameters from earlier seasons. The candidate won
seven of ten seasons on mean team allocation total variation:

| Metric | Independently tuned prior-minute control | Exact preseason NAIL | Difference |
| --- | ---: | ---: | ---: |
| Mean team allocation total variation | 0.22205 | **0.21980** | **-0.00226** |
| Mean Brier score | 0.02110 | **0.01977** | **-0.00133** |
| Player-share MAE | 0.02704 | **0.02677** | **-0.00027** |

The full completed history (`2015-16` through `2025-26`) selects
\(\epsilon=0.32\) and \(\beta=0.10\) for the next preseason. A one-standard-
deviation NAIL advantage therefore multiplies a player's relative allocation
score by \(\exp(0.10) \approx 1.105\) before the team total is normalized.

This is still an allocation model, not a claim that coaches mechanically play
the highest-rated roster. Prior minutes remain the dominant signal, while the
preseason NAIL forecast provides a measured within-roster adjustment. This
version is eligible for local Win Projections integration because the replay
uses the same forecast contract available in the product.

## Local Product Contract

Win Projections now uses this allocation for its local preseason baseline. The
website retains its practical active-15 roster gate: prior-season minutes choose
the initial 15 players, then the exact preseason NAIL forecast rebalances only
those 15 shares. This keeps the user-facing manual-only designation for players
outside the initial rotation. The gate itself was not part of the opening-roster
replay and remains a product constraint rather than a validated allocation term.
