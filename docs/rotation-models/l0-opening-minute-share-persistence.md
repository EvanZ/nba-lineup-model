---
last_updated: "2026-09-08"
---

# L0-MSP v0.0

**Season-Opening Minute-Share Persistence** is the first preseason baseline
in the Rotation Models family. It predicts only a team's first regular-season
game, using information that existed before that season's opening date.

## Contract

For each team, take the player minute-share allocation in its final completed
regular-season game of season \(s-1\). Restrict that allocation to the
transaction-derived opening candidate roster \(C_{T,s}\), then renormalize it:

\[
p_i = \frac{y_{i,s-1}^{final}\,\mathbf{1}[i \in C_{T,s}]}
{\sum_{j \in C_{T,s}} y_{j,s-1}^{final}}.
\]

Players new to the candidate roster begin at zero. L0-MSP reserves uniform
cold-start mass \(\epsilon\) across all candidates:

\[
\hat y_i = (1 - \epsilon)p_i + \frac{\epsilon}{|C_{T,s}|}.
\]

The source season selects \(\epsilon\) by mean allocation total variation;
ties select the smaller value. The following season is then evaluated once with
that frozen value.

## Initial Evaluation

Using `2024-25` as the source season selected the interior value
\(\epsilon = 0.80\). The frozen `2025-26` result did not carry that advantage
forward:

| Allocation | 2024-25 source TV | 2024-25 source Brier | 2025-26 frozen TV | 2025-26 frozen Brier |
| --- | ---: | ---: | ---: | ---: |
| L0-MSP, frozen \(\epsilon=0.80\) | 0.3052 | 0.0359 | 0.4409 | 0.0596 |
| Uniform opening-candidate control, \(\epsilon=1\) | 0.3164 | 0.0370 | **0.4339** | **0.0534** |

The previous final-game allocation has weak source-season signal, but that
signal did not generalize to the next opening night. L0-MSP is therefore a
useful boundary and control, not a promoted forecast. The next candidate should
learn player-specific preseason propensity from prior-season availability and
role indicators, rather than increasing reliance on a single final game.

## Leakage Boundary

The [historical opening-roster mart](../data/opening-rosters.md) validates
transaction reconstruction against opening-game box scores. Those
after-the-fact reconciliation rows are intentionally excluded here: L0-MSP
uses only `membership_source = transaction_reconstruction`. An actual opening
player absent from that source state receives zero predicted share and is
reported as `outside_candidate_actual_share`.

This is a harsh but honest preseason baseline. It cannot infer injuries,
training-camp cuts, or a newcomer omitted by the transaction feed.

## Run

```bash
uv run nba-evaluate-l0-opening-minute-share-persistence \
  --tune-season 2024-25 \
  --holdout-season 2025-26
```

The evaluator writes its epsilon grid, 30 source opening forecasts, 30 frozen
holdout forecasts, team-level metrics, and metadata under
`artifacts/rotation/l0_opening_minute_share_persistence/`.
