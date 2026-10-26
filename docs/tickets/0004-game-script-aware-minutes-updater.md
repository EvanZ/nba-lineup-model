# Slice 6: empirical early-game minutes updater

GitHub: [#31](https://github.com/EvanZ/nba-lineup-model/issues/31)

## Objective

Determine whether a player's rotation share from the competitive-looking early
portion of prior games predicts his next-game minutes better than the frozen
preseason role projection. This remains a shadow study: no public projection,
historical frozen snapshot, or site payload changes in this slice.

## Why the first implementation was rejected

The first implementation defined a score-based fourth-quarter garbage-time
boundary using a clock-dependent lead, an eventual-winner persistence rule,
and a separate projected-margin transform for pregame forecasts. That design
had three material problems:

1. The definition was too complicated to interpret and had not been reviewed
   with the user before implementation.
2. It classified 689 of 1,230 2025-26 games (56%) as containing garbage time,
   raising an immediate question about whether the boundary was too permissive.
3. The reported 2025-26 result was only a shrinkage calibration. It did not
   provide the frozen-versus-updated full-season backtest needed to evaluate
   the idea.

The score-based boundary is therefore superseded. Its code is not eligible for
deployment or promotion based on the prior evidence.

## Replacement idea

For each cutoff `C` in 36, 40, 44, and 48 elapsed minutes, reconstruct each
player's minutes before `C` from curated lineup stints and normalize them to a
48-minute role:

```text
normalized_role(C) = minutes_played_before_C / C * 48
```

When stint coverage is complete, the normalized player roles sum to exactly
240 minutes per team. The 48-minute arm is the ordinary full-regulation-game
control. The shorter cutoffs test whether excluding late-game rotation noise
improves the next-game forecast without making any score- or margin-based
garbage-time judgment.

## Chronological updater

For player `i` before game `t`, use only the last `K` eligible observations
strictly before the current game date:

```text
forecast_i,t =
    (w * preseason_role_i + sum(last K normalized roles_i))
    / (w + observed eligible games)
```

- `K`: 1, 2, 3, 5, 10, or all eligible games.
- `w`: 0.1, 0.2, 0.5, 1, 2, 4, or 8 preseason-equivalent games.
- Cutoff `C`: 36, 40, 44, or 48 minutes.
- Medical absence, inactive/assignment, and unknown rows do not update the
  role history or consume a lookback slot.
- A player officially available but receiving a coach DNP contributes a
  zero-minute role observation.
- Games on the same date are scored before any observation from that date is
  applied, preventing same-day leakage.

The complete grid contains 168 configurations.

## Study design

1. Reconstruct cutoff-normalized roles for 2024-25 and 2025-26 using only the
   local curated marts; do not copy training data from S3.
2. Select one configuration on 2024-25 using conditional-minute MAE as the
   primary metric and RMSE as the tie-breaker.
3. Lock that configuration before reading the 2025-26 result.
4. Evaluate it once on 2025-26 against the frozen preseason role baseline.
5. Report MAE, RMSE, sample size, improvement percentage, performance by month
   and prior-observation count, and representative player improvements and
   regressions.
6. Preserve the grid, selected configuration, input checksums, and holdout
   results as immutable local study artifacts.

The first study isolates conditional role accuracy for players whose official
availability state is known. It does not use the obsolete participation model,
and it does not claim a game-level win-probability improvement. A subsequent
game-level experiment is warranted only if the held-out player-minute evidence
supports the updater.

## Acceptance criteria

- Unit tests cover stint clipping at all four cutoffs, exact 240-minute team
  conservation, lookback behavior, preseason weighting, coach DNPs, medical
  exclusions, and same-day leakage prevention.
- Hyperparameters are selected on 2024-25 and evaluated untouched on 2025-26.
- The frozen baseline and selected updater are evaluated on identical holdout
  rows.
- The final response states the verified findings directly to the user,
  including evidence against the updater; a report artifact alone is not a
  completed deliverable.
- No production minutes or projections are promoted in this slice.

## Verified study result

Completed 2026-10-26 using the frozen 2024-25 preseason projection for tuning
and the frozen 2025-26 preseason projection for holdout. The former uses only
history completed through 2023-24; the latter uses only history completed
through 2024-25.

The 2024-25 grid selected:

- cutoff: first 44 elapsed minutes;
- lookback: last 3 eligible games;
- preseason weight: 0.2 equivalent games.

Locked 2025-26 holdout results across 1,230 games and 31,154 eligible
player-games:

| Metric | Frozen preseason | Updated | Change |
| --- | ---: | ---: | ---: |
| MAE | 7.912 | 5.028 | -2.884 (-36.4%) |
| RMSE | 10.227 | 6.944 | -3.283 |
| Bias | +0.031 | -0.096 | -0.127 |

The paired game bootstrap 95% interval for MAE improvement was 2.789 to 2.981
minutes (2,000 samples). The updater improved MAE both for actual appearances
(7.331 to 5.134, -30.0%) and known active/coach DNPs (11.349 to 4.401,
a 61.2% reduction), so the result is not solely a DNP effect.

The 44-minute observation beat the otherwise-identical 48-minute control by
only 0.070 MAE on holdout (95% paired bootstrap interval 0.051 to 0.088).
That supports excluding the last four minutes, but the practical edge is
small. The updater's larger value comes from rapid recent-role adaptation,
not from the cutoff itself.

Stint data is authoritative for participation and actual minutes. The study
overrode four incorrect 2024-25 absence labels where the lineup stints prove
the player entered the game; it found no such mismatches in the 2025-26
holdout. All 1,139 tuning games and 1,230 holdout games had complete coverage
at every cutoff, and every reconstructed team-cutoff conserved 240 minutes.

These results support continued shadow use. They do not yet support promotion
to the public game model because this study evaluates player-level conditional
role accuracy, not the final injury-conditioned, team-normalized 240-minute
game allocation or downstream margin probabilities.

## Shadow-report integration

The validated updater is now wired into the two-arm shadow report:

1. possession-updated NAIL ratings plus frozen pregame minutes; and
2. the same possession-updated NAIL ratings plus updated minutes.

The updated arm uses the first 44 minutes of each completed game, a three-game
rolling window, a 0.2-game preseason prior, no role clipping, the official
pregame injury assumptions, and an exact 240-minute team normalization. The
superseded score-based boundary and projected-blowout minutes transform are no
longer part of this comparison.

A local opening-sample replay through 2026-10-24 covered 36 games with no
exclusions or role-reconstruction fallbacks. Updated ratings plus updated
minutes beat updated ratings plus frozen minutes on every reported aggregate:

| Metric | Updated ratings + frozen minutes | Updated ratings + updated minutes |
| --- | ---: | ---: |
| Margin MAE | 11.150 | 10.971 |
| Margin RMSE | 15.427 | 15.394 |
| Brier | 0.19550 | 0.19459 |
| Log loss | 0.57206 | 0.57055 |
| Winner accuracy | 63.9% | 66.7% |
| ATS | 15-21 | 16-20 |

Both exported minute allocations reconcile to exactly 240 minutes for every
team-game. This small 36-game result is directional monitoring evidence, not
promotion evidence.

The deployed replay through 2026-10-25 expanded the sample to all 43 completed
games. It had zero exclusions, reconstructed all 43 early-game roles from CDN
play-by-play, and reconciled both arms to 240 minutes in all 86 team-games:

| Metric | Updated ratings + frozen minutes | Updated ratings + updated minutes |
| --- | ---: | ---: |
| Margin MAE | 11.497 | 11.288 |
| Margin RMSE | 15.373 | 15.372 |
| Brier | 0.19381 | 0.19215 |
| Log loss | 0.56835 | 0.56521 |
| Winner accuracy | 67.4% | 67.4% |
| ATS | 18-25 | 19-24 |

At the player level, conditional-minute MAE improved from 6.474 to 4.889 and
unconditional-minute MAE improved from 5.390 to 4.241. Updated minutes therefore
improved the current shadow sample's margin accuracy, probability calibration,
and ATS result without changing winner accuracy. Forty-three games remain far
too few for production promotion.
