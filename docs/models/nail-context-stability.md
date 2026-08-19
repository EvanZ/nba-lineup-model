---
title: Frozen vs Completed NALE Stability
---

# Frozen vs Completed NALE Stability

Last updated: 2026-08-17

This audit tests whether the non-additive context signal that improves frozen
lineup forecasting is stable enough to support retrospective attribution.

For target season \(t\), both scores use the same strictly lagged player profile
\(\phi_{t-1}\). The frozen state uses the context coefficients learned through
\(t-1\), while the completed state uses coefficients fitted after \(t\):

\[
C_t^{\mathrm{frozen}}(H,A)=\gamma_{t-1}^{\top}
  [g(\phi_{t-1}(H))-g(\phi_{t-1}(A))],
\]

\[
C_t^{\mathrm{completed}}(H,A)=\gamma_t^{\top}
  [g(\phi_{t-1}(H))-g(\phi_{t-1}(A))].
\]

Observed target-season stints are used only to calculate possession-weighted
lineup and player exposure. Thus frozen NALE is predictive conditional on
realized lineup allocation, while completed NALE is a retrospective description
of that same allocation.

## Result

The first audit uses all observed regular-season stints in the three frozen
leaderboard seasons. Lineups require at least 100 possessions; players require
at least 250 on-court possessions. All correlations below are possession
weighted except the coefficient rank correlation.

| Target season | Frozen context state | Stint Pearson | Stint Spearman | Lineup Pearson | Lineup Spearman | Player Pearson | Player Spearman | Coefficient rank correlation | Coefficient sign agreement |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2023-24 | 2022-23 | 0.804 | 0.795 | 0.795 | 0.797 | 0.779 | 0.766 | 0.257 | 2 / 6 |
| 2024-25 | 2023-24 | 0.782 | 0.759 | 0.814 | 0.754 | 0.785 | 0.702 | 0.943 | 6 / 6 |
| 2025-26 | 2024-25 | 0.738 | 0.722 | 0.666 | 0.663 | 0.729 | 0.695 | 0.829 | 4 / 6 |

The frozen non-additive signal is materially stable at the stint, lineup, and
player-exposure levels. It is not a one-for-one substitute for the completed
estimate: the 2025-26 lineup correlation is only 0.666, and several individual
feature coefficients change sign from one season to the next. The practical
interpretation is that frozen NALE is suitable as a **predictive contextual
companion** to player ratings, while completed NALE remains the appropriate
retrospective description of the season that actually occurred.

The coefficient pattern is deliberately shown separately from the score
correlations. With only six differently scaled features, raw-coefficient
Pearson correlation is dominated by the usage-concentration slope and is not a
useful stability summary. Rank correlation and sign agreement better describe
whether the individual feature functions themselves persisted.

## Interpretation

The audit reports stability at three levels:

1. **Coefficient stability** compares the six non-additive raw Ridge coefficients.
2. **Observed-lineup stability** compares each five-man unit's possession-weighted
   frozen and completed NALE against the opponents it actually faced.
3. **Player-exposure stability** averages those oriented stint effects over each
   player’s actual regular-season possession exposure.

A player NALE is not individual causal credit. It describes the non-additive
environments a player actually occupied. A high frozen/completed correlation
would support frozen NALE as a useful predictive companion to the completed
retrospective quantity; it would not make it a preseason forecast of teammates,
playing time, or lineup deployment.

## Reproduce

```bash
uv run nba-audit-nail-context-stability
```

Artifact: `artifacts/models/analysis/nail_context_stability/2023-24_to_2025-26/nail-context-stability-2023-24-to-2025-26-20260817T234036Z-45350c8c`.
