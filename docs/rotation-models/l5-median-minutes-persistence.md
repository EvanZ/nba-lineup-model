---
last_updated: "2026-08-25"
---

# L5-MMP v0.1

**Last-5 Median Minutes Persistence** is the second parameter-free Rotation
Models baseline. It tests whether a small robust history improves materially on
copying the prior team-game allocation.

## Forecast contract

For a target team-game \(g+1\), use exactly the five preceding completed
team-games for that team. Let \(m_{i,k}\) be a player's raw minutes in history
game \(k\), with zero assigned when the player was absent or did not play.

\[
\widetilde m_{i,g+1} =
\operatorname{median}(m_{i,g-4}, m_{i,g-3}, m_{i,g-2}, m_{i,g-1}, m_{i,g}).
\]

The raw median totals are normalized into a valid allocation:

\[
\widehat y_{i,g+1} =
\frac{\widetilde m_{i,g+1}}{\sum_j \widetilde m_{j,g+1}}.
\]

This is deliberately not a trailing average, not conditional on games played,
and not a NAIL-informed model. The median allows a stable rotation player to
survive a one-game rest or DNP, while repeated absences push the forecast toward
zero.

## Evaluation

Evaluation begins with a team's sixth completed game. L5-MMP reports the same
metrics as [L1-MSP](l1-minute-share-persistence.md): allocation total variation,
distributional Brier score, player-share MAE/RMSE, and strict cross-entropy.

Strict cross-entropy remains \(+\infty\) when a player receives positive target
minutes after a zero five-game median. The artifact separately reports:

- `zero_history_actual_share`: target share from a player absent from all five
  historical team-games;
- `zero_median_actual_share`: target share from a player assigned zero by the
  median rule, including players who were present but repeatedly inactive.

## Initial benchmark

L5-MMP is a control, not a promoted rotation baseline. It evaluates 120 fewer
team-games than L1-MSP in each season because five historical games are
required instead of one.

| Model | Season | Evaluated team-games | Mean TV | Brier | Player-share MAE | Player-share RMSE | Infinite CE rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L1-MSP | 2024-25 | 2,248 | 0.2038 | 0.0278 | 0.0287 | 0.0441 | 72.7% |
| L5-MMP | 2024-25 | 2,128 | 0.2174 | 0.0309 | 0.0276 | 0.0442 | 78.2% |
| L1-MSP | 2025-26 | 2,430 | 0.2025 | 0.0265 | 0.0229 | 0.0387 | 74.7% |
| L5-MMP | 2025-26 | 2,310 | 0.2139 | 0.0291 | 0.0237 | 0.0401 | 75.7% |

The median reduces sensitivity to one-game rests for established rotation
players, but it is too slow to adapt to real allocation changes. It loses to
L1-MSP on the two support-invariant allocation metrics, total variation and
Brier score, in both evaluated seasons.

This is not caused by L5-MMP's longer five-game warmup. On the identical 2,310
L5-eligible team-games in 2025-26, L1-MSP has Brier \(0.0271\) and TV
\(0.2046\), versus L5-MMP's Brier \(0.0291\) and TV \(0.2139\). L5-MMP wins
on Brier in 46.5% of those team-games, including rest-return cases, but it
loses slightly more often and by enough to lose on the average.

## Run

```bash
uv run nba-evaluate-l5-median-minutes-persistence \
  --seasons 2024-25 2025-26
```

The evaluator writes immutable artifacts beneath
`artifacts/rotation/l5_median_minutes_persistence/`.
