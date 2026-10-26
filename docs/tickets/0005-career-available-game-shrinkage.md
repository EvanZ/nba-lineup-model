# Slice 7: career-available-game shrinkage for the minutes updater

GitHub: [#32](https://github.com/EvanZ/nba-lineup-model/issues/32)

## Objective

Replace the in-season conditional-minutes updater's single preseason weight for
every player with a simple, monotonic weight based on career **available NBA
games** known before the prediction date. The intended behavior is faster role
learning for cold-start and early-career players and greater retention of the
preseason role prior for established veterans.

This is a shadow-study slice. It must not change public minutes, game
projections, immutable historical snapshots, or production promotion status.

## Motivation

The current early-game role updater gives every player a preseason prior weight
of `0.2` equivalent games. A leakage-safe opening-week cohort audit found a
monotonic decline in the updater's played-game MAE improvement as experience
increased:

| Career stage | Frozen MAE | Updated MAE | Improvement |
| --- | ---: | ---: | ---: |
| First year | 6.99 | 5.67 | 1.33 (19.0%) |
| Second year | 6.21 | 5.28 | 0.94 (15.1%) |
| Years 3–4 | 5.95 | 5.09 | 0.85 (14.4%) |
| Veterans | 4.94 | 4.47 | 0.47 (9.6%) |

This is directional evidence, not a calibration sample. The 36-game retained
player-level artifact is too small to select weights, and the 43-game run
retained only aggregate metrics. The proposed rule must therefore be selected
and evaluated on historical seasons.

## Why career available games

Raw games played understate experience for healthy players who were available
but received a coach DNP. Calendar seasons or age overstate experience for
players who lost a season to injury. Blake Griffin and Chet Holmgren are the
canonical cases: their medically lost first seasons must not create 82 games of
role evidence before their NBA debuts.

For this slice, career available games for player `i` before game `t` is the
cumulative count of earlier NBA team-games classified as either:

- `played`; or
- `coach_dnp` / known healthy active DNP.

The count excludes:

- injury, illness, rest, personal, or other medical absence;
- inactive-list, assignment, suspension, or ineligible absence; and
- unknown availability states.

The feature is computed strictly before the target game. All games on the same
date are scored before that date contributes new availability observations.
The existing authoritative availability classification remains the source of
truth.

## Proposed updater

Keep the validated 44-minute normalized role target and three-eligible-game
lookback. Change only the preseason equivalent-game weight:

```text
career_available_games_i,t =
    count(prior played or healthy coach-DNP NBA team-games)

prior_weight_i,t = clip(
    w_min + beta * log1p(career_available_games_i,t / 82),
    w_min,
    w_max,
)

forecast_i,t =
    (prior_weight_i,t * preseason_role_i + sum(last K normalized roles_i))
    / (prior_weight_i,t + observed eligible games)
```

Constraints:

- `beta >= 0`, so additional available-game experience can never reduce prior
  weight;
- finite `w_min > 0` and `w_max >= w_min`;
- logarithmic growth, because the difference between 0 and 50 available games
  should matter more than the difference between 500 and 550; and
- a cap, so even a veteran can adapt to a genuine role change.

Career possessions are deliberately not used in this first slice. Possessions
are natural evidence for the NAIL rating updater, but available games are more
directly aligned with repeated opportunities to occupy an NBA rotation role.

## Historical study design

1. Build the as-of career-available-game feature from existing local historical
   availability data. Do not download or copy a large dataset from S3 without
   explicit user approval.
2. Preserve the current constant-`0.2` updater as the primary control.
3. Pre-register a compact grid for `w_min`, `beta`, and `w_max`. Do not select
   parameters from the 2026-27 opening-week sample.
4. Use rolling historical development seasons for parameter selection and lock
   the selected function before evaluating the final 2025-26 holdout.
5. Evaluate identical player-game rows for the control and candidate. Report
   played-only conditional minutes separately from played-plus-coach-DNP
   conditional minutes.
6. Report overall and career-exposure-segment MAE, RMSE, bias, paired
   game-bootstrap intervals, and representative player improvements and
   regressions.
7. Run the downstream four-cell game comparison with the same updated NAIL
   ratings in both minutes arms. Treat it as secondary evidence; conditional
   minute accuracy is the primary selection target.
8. Produce a browser-served report and state the verified findings directly to
   the user. Do not make the user inspect raw HTML or infer the conclusion from
   artifact files.

## Required audits

- Show the learned prior weight at 0, 10, 25, 50, 82, 164, 410, and 820 career
  available games.
- Confirm that a synthetic medically lost rookie season adds zero career
  available games.
- Confirm that a healthy coach DNP adds one career available game but a medical
  absence, inactive/assignment row, and unknown row add zero.
- Confirm strict pregame and same-date leakage protection.
- Preserve the row-level evaluation parquet in every shadow run so cohort
  diagnostics can be regenerated for the complete sample.
- Add career-exposure cohort metrics to the daily browser report.

## Acceptance criteria

- Unit tests cover all availability classifications, cumulative as-of feature
  construction, log transform, clipping, monotonicity, and same-day leakage.
- The candidate is selected without reading the locked holdout result.
- The final holdout compares identical rows against the constant-`0.2` control.
- The report includes played-only and played-plus-coach-DNP results overall and
  by career-exposure band, with sample sizes and uncertainty.
- The report explicitly identifies whether gains come from cold-start players,
  veterans, coach DNP classification, or a combination.
- A browser URL and a concise written conclusion are delivered to the user.
- No production projection is changed without a separate explicit approval.

## Non-goals

- Replacing the official injury-report availability assumptions.
- Reintroducing a probability-of-playing model for known injury-report states.
- Changing the NAIL ratings updater.
- Adding team continuity, coaching continuity, age, or role-volatility
  interactions in this slice. Those remain possible follow-ups only if the
  single-variable model leaves a documented residual problem.
