---
last_updated: "2026-09-26"
---

# Pregame Usage-Response Screen

This is a completed **negative experiment**. It tests whether a player's
pregame conventional-USG% change from their preseason role improves the
offense/defense scoring allocation on top of the fixed production Constrained
Split NAIL baseline. It does not alter scalar NAIL-RAPM, player ratings,
additive profiles, context, home court, or back-to-back controls.

## Pregame State

For each player game, conventional usage uses the production NAIL numerator

\[
E=\mathrm{FGA}+0.44\,\mathrm{FTA}+\mathrm{TOV}
\]

and the standard team-minute opportunity denominator. The preseason rate
\(\mu_{i,t}\) is the prior season's player rate shrunk to that season's league
rate using 300 usage opportunities. A player without a prior season begins at
the league rate.

Before game \(g\), the state is updated only with earlier target-season games:

\[
u_{i,g}^{\mathrm{pre}} =
\frac{300\mu_{i,t}+\sum_{h<g} E_{i,h}}
{300+\sum_{h<g} A_{i,h}},
\qquad
z_{i,g}=100\left(u_{i,g}^{\mathrm{pre}}-\mu_{i,t}\right),
\]

where \(A\) is usage opportunity. Thus the current game's FGA, FTA, and TOV
cannot influence its prediction. A player listed with zero box-score minutes
retains their pregame state but contributes no postgame exposure; this preserves
alignment with reconstructed lineup rows without inventing usage.

For each stint scoring side, the model sums \(z\) across the five offensive
players and separately across the five defenders. It adds only this residual
correction to the frozen Constrained Split prediction:

\[
\hat y = \hat y_{\mathrm{split}}
+ \beta_O\sum_{i\in\mathrm{offense}}z_i
- \beta_D\sum_{j\in\mathrm{defense}}z_j.
\]

The offense and defense coefficients are independently ridged. Therefore a
player could, in principle, improve or worsen on either side at higher usage;
the model does not impose a locked total response.

## Selection

The screen evaluated `0.0001`, `0.001`, `0.01`, `0.1`, `1`, and `10` under a
sample-size-normalized zero-intercept ridge penalty. Each development season
was predicted from all earlier completed seasons. Possession-weighted
scoring-side RMSE across 2019-20 through 2022-23 selected the interior value
`0.1`.

The development fit suggested a positive association on both sides. Through
2022-23, one standard deviation of an individual pregame usage shock (2.34
percentage points) implied approximately:

| Usage shock | Offense | Defense | Total |
| --- | ---: | ---: | ---: |
| -1 SD | -0.37 | -0.23 | -0.60 |
| +1 SD | +0.37 | +0.23 | +0.60 |
| +2 SD | +0.74 | +0.46 | +1.20 |

Those coefficients are descriptive associations in the source seasons. They
are not evidence that increasing an individual player's usage causes the same
change in value.

## Frozen Result

The selected contract was then replayed without retuning over 2023-24 through
2025-26. It failed the generalization check:

| Metric | Candidate minus fixed split | Paired 95% CI | Conclusion |
| --- | ---: | ---: | --- |
| Side-scoring RMSE | +0.0141 | [+0.0064, +0.0222] | Materially worse |
| Full-game margin RMSE | +0.0111 | [-0.0484, +0.0711] | No reliable difference |

The one-season results were inconsistent: 2023-24 improved, while 2024-25
became substantially worse and 2025-26 was also worse on side scoring. The
pooled, game-cluster bootstrap is decisive for the side target, so this model
is **not retained** as a production or descriptive O/D adjustment.

The immutable run is
`artifacts/models/usage_response_screen/usage-response-screen-20260927T051438Z-da5bfafd`.
It contains the player-game pregame states, exact lineup shock rows,
development selection, frozen side and full-game predictions, paired bootstrap
metrics, and the response curve.

## Reproduce

```bash
uv run nba-run-usage-response-screen \
  --log-path artifacts/logs/usage-response-screen.log
```
