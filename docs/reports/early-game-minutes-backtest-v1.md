# Early-game minutes backtest v1

Generated: 2026-10-26

Ticket: [#31](https://github.com/EvanZ/nba-lineup-model/issues/31)

## Finding

The simple empirical updater materially beat frozen preseason conditional
minutes on a locked full-season holdout. The grid was selected on 2024-25 and
evaluated once on 2025-26.

Selected configuration: first 44 minutes, last 3 eligible games, and a 0.2
preseason-equivalent-game prior.

| 2025-26 holdout | Frozen | Updated | Improvement |
| --- | ---: | ---: | ---: |
| MAE | 7.912 | 5.028 | 2.884 minutes / 36.4% |
| RMSE | 10.227 | 6.944 | 3.283 minutes |
| Bias | +0.031 | -0.096 | — |

Coverage: 1,230 games, 592 players, and 31,154 eligible player-games. The
paired game-bootstrap 95% interval for MAE improvement was 2.789 to 2.981
minutes, with all 2,000 bootstrap samples favoring the updater.

## What drove the result

| Holdout group | Rows | Frozen MAE | Updated MAE | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Played | 26,651 | 7.331 | 5.134 | 2.197 / 30.0% |
| Active or coach DNP | 4,503 | 11.349 | 4.401 | 6.948 / 61.2% |

The updater therefore improved actual appearances too; its gain was not just
learning that fringe active players often receive zero minutes.

The first observation produced an immediate holdout improvement: MAE fell from
7.137 to 4.497 for players with exactly one prior eligible observation. With
two observations it fell from 7.012 to 4.625. The no-history group is exactly
equal by construction because it uses the frozen preseason prior.

## Is the early cutoff useful?

With the selected 3-game lookback and 0.2 prior weight:

| Cutoff | Holdout MAE | Holdout RMSE |
| ---: | ---: | ---: |
| 36 | 5.138 | 7.139 |
| 40 | 5.050 | 6.999 |
| 44 | 5.028 | 6.944 |
| 48 | 5.098 | 6.940 |

The selected 44-minute arm beat the 48-minute control by 0.070 MAE (95% paired
game-bootstrap interval 0.051 to 0.088), while 48 minutes was better by 0.004
RMSE. The evidence says the final four minutes contain a small amount of
next-game MAE noise; it does not say that removing them is the main source of
the updater's success.

## Opening examples

Ryan Rollins had a frozen prior of 10.6 minutes. After one game, the updater
forecast 25.6 for his next game, where he played 29.7; after two, it forecast
28.7 for a 33.9-minute game. This is the type of rotation change the frozen
prior cannot capture.

Stephen Curry's frozen prior was 29.6. The updater moved to 30.4 after one
observation and 30.5 after two. His next actuals were 33.5 and 27.3, so the
update helped on the former and hurt on the latter. This illustrates the
variance introduced by the deliberately weak 0.2-game prior.

Anthony Davis shows the injury/in-game-event limitation. The updater improved
his second through fourth games, but forecast 36.4 before a game in which he
played only 6.6 minutes. Pregame minutes models cannot anticipate an in-game
injury or early exit.

## Verification and caveats

- All 1,139 tuning games and 1,230 holdout games had complete stint coverage
  at 36, 40, 44, and 48 minutes.
- Every team-cutoff reconstruction conserves exactly 240 normalized minutes.
- Same-date games are scored before applying any observations from that date.
- Medical, inactive, assignment, and unknown outcomes are excluded. Known
  active/coach DNPs contribute zero.
- Stints are authoritative for participation and actual minutes. They corrected
  four bad absence labels in 2024-25 and none in the holdout.
- This is a player-level conditional-role study. It does not yet test final
  injury-conditioned, team-normalized lineups or game win probabilities.

Immutable local outputs are under
`artifacts/analysis/early_game_minutes_backtest_v1/`; `summary.json` contains
the input SHA-256 lineage, full segment results, representative players, and
bootstrap settings.

## Current-season shadow integration

The validated configuration was integrated into the shadow report as a clean
two-arm comparison. Both arms use the same possession-updated NAIL ratings;
only frozen versus updated minutes changes. The former score-based role
boundary and pregame blowout transform are excluded.

On the locally available 36-game sample through 2026-10-24:

| Metric | Updated ratings + frozen minutes | Updated ratings + updated minutes |
| --- | ---: | ---: |
| Margin MAE | 11.150 | 10.971 |
| Margin RMSE | 15.427 | 15.394 |
| Brier | 0.19550 | 0.19459 |
| Log loss | 0.57206 | 0.57055 |
| Winner accuracy | 63.9% | 66.7% |
| ATS | 15-21 | 16-20 |

All 36 early-role targets reconstructed successfully and both exported minute
allocations sum to exactly 240 minutes for every team-game. The sample is too
small for promotion; it confirms the shadow integration is operating as
designed and currently points in the same direction as the historical study.

The first deployed replay through 2026-10-25 covered all 43 completed games,
with zero exclusions and 43 of 43 CDN play-by-play reconstructions:

| Metric | Updated ratings + frozen minutes | Updated ratings + updated minutes |
| --- | ---: | ---: |
| Margin MAE | 11.497 | 11.288 |
| Margin RMSE | 15.373 | 15.372 |
| Brier | 0.19381 | 0.19215 |
| Log loss | 0.56835 | 0.56521 |
| Winner accuracy | 67.4% | 67.4% |
| ATS | 18-25 | 19-24 |

Conditional-minute MAE improved from 6.474 to 4.889; unconditional-minute MAE
improved from 5.390 to 4.241. Both arms summed to exactly 240 minutes in all 86
team-games. This remains a monitoring result rather than promotion evidence.
