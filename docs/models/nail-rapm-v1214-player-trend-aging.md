---
last_updated: "2026-09-04"
---

# NAIL-RAPM v1.2.1.4 Player-Trend Aging

This candidate changes one part of the production
[NAIL-RAPM v1.2.1.3](nail-rapm-v1213-residualized-lambda.md) contract: the
forward aging prior may use a player's own latest completed change in RAPM.
All other production choices remain fixed, including residualized-target
player-lambda CV, exposure-gated cold starts, gap-returner priors, additive
profiles, the two retained non-additive features, and home-court and
back-to-back controls.

## Strict Forward Signal

For a target season \(t\), the feature is available only when the player was
observed in both completed seasons \(t-1\) and \(t-2\):

\[
\Delta_{i,t-1} = R_{i,t-1} - R_{i,t-2}.
\]

The aging Ridge receives the standardized change, an availability indicator,
and its interaction with source reliability:

\[
\Delta_{i,t-1}
\log\!\left(1 + \min\{E_{i,t-1}, E_{i,t-2}\}\right).
\]

\(E\) is modeled possession exposure. The Ridge therefore learns how much to
trust a recent improvement or decline; there is no fixed momentum coefficient.
Rookies, players without a consecutive two-season history, and gap returners
do not receive a fabricated trend. In particular, a gap returner continues to
use the production bridge rather than repeatedly applying their last observed
change.

## Frozen Evaluation

The immutable replay covers 2023-24 through 2025-26. It uses the same recovered
625,615 regular-season possession support as the production comparison and the
same 3,511 reconstructed full games. Possession and eligible-game entries in
the [global leaderboard](three-season-frozen-backtest.md) are normalized by
the paired candidate-minus-production delta to its legacy support row; the
full-game and team metrics below are direct.

| Metric | Production v1.2.1.3 | Player-trend aging |
| --- | ---: | ---: |
| Regular possession RMSE | 1.198147 | **1.198145** |
| Regular possession MAE | 1.141455 | **1.141440** |
| Regular eligible game RMSE | 14.005121 | **13.999691** |
| Regular full-game RMSE | 14.216588 | **14.207980** |
| Regular winner accuracy | 68.30% | **68.56%** |
| Team net-rating RMSE | 3.235071 | **3.220595** |
| Pythagorean-win RMSE | 6.942255 | **6.885142** |
| Playoff eligible game RMSE | 16.578243 | **16.570251** |

Unknown-player exposure is identical for production and candidate: 9,765,
2,289, and 3,998 regular-season possessions in the 2023-24, 2024-25, and
2025-26 frozen cohorts, respectively; playoff values are 1,238, 106, and 93.

## Bootstrap Gate

The 10,000-draw game-stratified paired bootstrap evaluates the primary
full-game margin RMSE. Candidate minus production is \(-0.008608\), with a
95% interval of \([-0.019011, +0.001890]\), and the candidate wins 94.98% of
draws. Each season also clears the agreed no-material-harm limit of +0.5%:

| Target season | Candidate minus production RMSE | 95% interval |
| --- | ---: | --- |
| Pooled | -0.008608 | [-0.019011, +0.001890] |
| 2023-24 | +0.008934 | [-0.008138, +0.026326] |
| 2024-25 | -0.012679 | [-0.031555, +0.006939] |
| 2025-26 | -0.020603 | [-0.037939, -0.003345] |

The result is **promotion eligible**, not a deployed release. It is retained in
the model tree and leaderboard so the release decision is explicit.

## Artifacts And Reproduction

- Completed fit: `artifacts/models/forward_nail_rapm_v1214_player_trend_aging/2025-26/forward-nail-rapm-v1214-player-trend-aging-2025-26-20260904T053025Z-287625b9`
- Frozen replay: `artifacts/models/nail_v1214_player_trend_aging_frozen_backtest/frozen_multiseason_backtest/2023-24_to_2025-26/frozen_multiseason-backtest-2023-24-to-2025-26-20260904T053614Z-1aeab0c9`
- Paired bootstrap: `artifacts/models/nail_v1214_player_trend_aging_bootstrap/2023-24_to_2025-26/nail-v1214-player-trend-aging-bootstrap-20260904T053813Z-d2414553`

```bash
uv run python -m nba_lineup_model.modeling.forward_nail_v1214_player_trend_aging \
  --log-path artifacts/logs/nail-v1214-player-trend-aging.log

uv run python -m nba_lineup_model.modeling.nail_v1214_player_trend_aging_frozen_backtest \
  --log-path artifacts/logs/nail-v1214-player-trend-aging-frozen.log

uv run python -m nba_lineup_model.modeling.nail_v1214_player_trend_aging_bootstrap
```
