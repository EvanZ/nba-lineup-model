---
last_updated: "2026-08-13"
---

# HIPSTER PM v2.2: Usage Allocation

HIPSTER PM v2.2 is a controlled context-feature release. It retains HPM v2.1's
depth-aware shooting, empirical rebound-capacity model, centered
value-conditioned aging player prior, exposure-gated cold-start path, bounded
hierarchical P-splines, and portable-matchup decomposition. It replaces only
the raw usage representation.

## Allocation Model

For a five-player unit with prior-season terminal-action claims
\(u_1,\ldots,u_5\), the completed season fits a conditional-logit allocation:

\[
\hat p_i = \frac{\exp(\tau u_i/s)}
{\sum_{j=1}^{5}\exp(\tau u_j/s)}.
\]

Here \(s\) is the observed cross-lineup standard deviation of the claims and
\(\tau\) is estimated from actual field-goal attempts, free throws, and
turnovers. Free throws receive the standard 0.44 possession weight. The
softmax is an intermediate description of how a lineup's historical claims
must be reallocated; it is not itself a value judgment about concentrated or
balanced usage.

For each unit, v2.2 supplies four continuous features to the bounded
contextual RAPM:

| Feature | Definition | Purpose |
| --- | --- | --- |
| Excess usage demand | \(\max(\sum_i u_i-B,0)\), with \(B\) learned from completed action lineups | Measures demand above the empirical lineup budget. |
| Allocation entropy | \(-\sum_i\hat p_i\log(\hat p_i)/\log(5)\) | Describes the fitted action-share distribution without a top-two cutoff. |
| Role reallocation JS | Jensen--Shannon divergence between \(u_i/\sum_j u_j\) and \(\hat p_i\) | Captures how strongly the allocation model reshapes the raw claims. |
| Allocation-weighted turnover burden | \(\sum_i\hat p_i\operatorname{TOV}_i\) | Couples the expected action allocation to the players expected to use possessions. |

## Feature Registry

| Usage feature | HPM v2.1 | HPM v2.2 | Reason |
| --- | --- | --- | --- |
| Raw usage events per 100 | Active | Retired | A single high-usage player can dominate a unit total without describing role conflict. |
| Raw turnovers per 100 | Active | Retired | Replaced by turnover burden under the predicted allocation. |
| Usage concentration | Active | Retired | A hand-built top-two cutoff. |
| Depth by usage interaction | Active | Retired | Depends on the retired concentration proxy. |
| Excess usage demand | None | Added | Empirical demand pressure above a learned budget. |
| Allocation entropy | None | Added | Smooth allocation shape. |
| Role reallocation JS | None | Added | Difference between raw claims and fitted shares. |
| Allocation-weighted turnover burden | None | Added | Turnover exposure under the fitted role allocation. |

Passing features remain unchanged: assists per 100, top-two assists, and the
shooter-passing interaction. This isolates the usage representation from the
rest of the context model.

## Information Boundary

For forecast season \(t\), player claims come from information through
\(t-1\). The allocation model is fit to actual season \(t-1\) actions using
those pre-season claims, stored in the completed contextual state, and used as
a context prior for \(t\). Neither target-season actions nor target-season
lineups are used in frozen evaluation.

## Frozen Result

Across the frozen 2023-24 through 2025-26 regular-season evaluation, v2.2
records the best possession RMSE (**1.198004**) and possession MAE
(**1.141380**) of the published comparison set. Its eligible game-margin RMSE
is **14.1464**, behind HPM v2.1's **14.1340**. The allocation representation
therefore improves the possession objective very slightly but does not yet
displace v2.1 on the more decision-relevant game and team aggregates.

On pooled playoffs, v2.2 has a possession RMSE of **1.192807** and eligible
game-margin RMSE of **16.6806**. See the
[Three-Season Frozen Leaderboard](three-season-frozen-backtest.md) for the
complete regular-season and playoff comparison.

## Reproduce

```bash
uv run nba-train-hpm-v22 --through-season 2025-26 \
  2>&1 | tee artifacts/logs/train-hpm-v22-2025-26.log
```

Follow a running job with:

```bash
tail -f artifacts/logs/train-hpm-v22-2025-26.log
```

The immutable outputs are written under:

```text
artifacts/models/forward_hpm_v22_usage_allocation/2025-26/
```
