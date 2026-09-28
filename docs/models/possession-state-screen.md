---
last_updated: "2026-09-28"
---

# Conditional Possession-State Screen

This screen asks a deliberately narrower question than production NAIL: once a
possession has started, does knowing whether the offense begins in a live-ball
or dead-ball state improve the fixed O/D Split NAIL scoring estimate? It is an
at-possession conditional evaluation, not a pregame model and not a frozen
leaderboard candidate.

## Inputs

The [possession start-state mart](../data/possession-start-states.md) labels
each possession using only the prior terminal event and same-clock handoff
events. The first screen pools that retained granular taxonomy into:

| Coarse state | Granular source states |
| --- | --- |
| Live ball | Defensive rebound; live-ball turnover with a credited steal; other live-ball turnover |
| Dead ball | Period start; made 2PT; made 3PT; final free throw; dead-ball turnover; held-ball recovery |

The mart's `audited_fallback` rows are excluded rather than imputed. Each row
is additionally restricted to a complete, single-lineup possession. Its
baseline is the exact frozen, prior-season O/D Split NAIL score for that
offensive lineup against that defensive lineup, including the production
player, additive-profile, retained non-additive, home-court, and back-to-back
terms.

## Nested Models

For possession outcome (y_p), baseline (b_p), and live-ball indicator
(L_p), the control model is

\[
y_p = b_p + a + \gamma L_p + \epsilon_p.
\]

The richer model retains that control and adds a player-specific live-ball
deviation on each O/D side:

\[
y_p = b_p + a + \gamma L_p
+ L_p\left(\sum_{i \in O_p}\delta_i^O
- \sum_{j \in D_p}\delta_j^D\right) + \epsilon_p.
\]

Every (\delta) has a zero prior and a relative ridge precision of `32`, so
the state-specific player terms are strongly pooled toward no deviation. Dead
ball is the reference state. The screen does not independently refit player
RAPM ratings.

## Evaluation

For each target season, the player deviations use the previous four complete
seasons only. Regularization is selected by 2019-20 through 2022-23
conditional possession RMSE and then frozen for 2023-24 through 2025-26. The
artifact reports fixed Split NAIL, state-control-only, and state-plus-player
RMSE, plus complete-game cluster bootstrap intervals, live-ball player
exposure, coefficient uncertainty, and player examples.

Because the realized state is known only when the possession begins, an
improvement here would justify an in-game conditional layer. It would not by
itself establish pregame predictive value. Modeling how teams and players
*create* those state mixes is a separate follow-on task.

## Result

The completed screen is
`artifacts/models/possession_state_screen/possession-state-screen-20260928T141513Z-295ddf6c`.
The development grid selected the interior player-deviation regularization
`0.0001`; its effective player penalty remains 32 times the baseline relative
precision. The frozen result pools 623,970 known-state, single-lineup
possessions from 2023-24 through 2025-26.

| Model | Conditional possession RMSE | Change vs. frozen Split NAIL |
| --- | ---: | ---: |
| Fixed O/D Split NAIL | 121.5274 | -- |
| Live/dead control only | 121.4560 | -0.0715 |
| Control plus pooled player live-ball deviations | 121.2808 | -0.2466 |

The game-cluster bootstrap for the full model gives a 95% interval of
`[-0.2686, -0.2230]` on the pooled possession-RMSE change, and every individual
frozen season improves. The realized live-ball control is positive: the
latest four-season descriptive fit estimates a `+5.43` point per 100
possessions shift relative to dead-ball starts after the fixed O/D baseline.

This is enough evidence to proceed to the hierarchically pooled fine-state
test in #22. It is **not** evidence to publish independent state-specific
player ratings or to add the realized state to pregame NAIL predictions: the
screen conditions on an event that has already occurred. Some residual
player-by-live-ball terms are also large, so they remain diagnostic
associations only. A later state-generation model must restore credit for
creating favorable next-possession starts before a total attribution can be
considered.

## Reproduce

```bash
uv run nba-run-possession-state-screen \
  --log-path artifacts/logs/possession-state-screen.log
```
