---
last_updated: "2026-08-12"
---

# Three-Season Frozen Backtest

This backtest replays 2023-24, 2024-25, and 2025-26 from the completed
recursive states already retained by each model artifact. It does **not** train
three new HPM models.

For target season \(t\), the replay uses:

\[
\text{player prior } \mu_t, \qquad
\text{context state } C_{t-1}, \qquad
\text{target-season realized lineup allocation}.
\]

The player prior was built before \(t\); the context model was fitted only to
the immediately preceding completed season. Target-season scores, player
updates, and context refits are excluded. As with the Frozen Leaderboard,
realized lineup exposure is an oracle input. The common three-season report is
regular season only because historical playoff possession partitions are not
yet materialized for 2023-24 and 2024-25; 2025-26 playoffs remain separately
evaluated on the Frozen Preseason Leaderboard.

The initial comparison holds the shared recursive architecture fixed and tests
only the box-score prior branches: Value-Conditioned Aging HPM, additive
box-score residual HPM, and box-score interaction HPM.

Each model is scored on 584,970 regular-season possessions from 3,284 games.
The full-game and team summaries cover 3,280 games with complete reconstructed
stint allocations. Bold values are the pooled leaders; lower is better except
for winner accuracy.

<!-- frozen-multiseason-results:start -->
## Results

Artifact: `artifacts/models/frozen_multiseason_backtest/2023-24_to_2025-26/frozen_multiseason_backtest-2023-24-to-2025-26-20260813T000545Z-96b58af1`.

### Pooled Regular Season

| Model | Possession RMSE | Eligible game RMSE | Full-game RMSE | Winner accuracy | Team NetRtg RMSE | Pythagorean-win RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Value-Conditioned Aging HPM | **1.198035** | **14.1535** | 14.3850 | 67.38% | 3.5674 | 7.1120 |
| Forward box-score residual HPM | 1.198038 | 14.1574 | **14.3833** | **67.71%** | **3.5458** | **7.0331** |
| Forward box-score interaction HPM | 1.198040 | 14.1647 | 14.3921 | 67.68% | 3.5514 | 7.0392 |

### Per-Season Regular Results

| Season | Model | Possession RMSE | Eligible game RMSE |
| --- | --- | ---: | ---: |
| 2023-24 | Forward box-score interaction HPM | 1.193125 | 13.7709 |
| 2023-24 | Forward box-score residual HPM | 1.193126 | 13.7590 |
| 2023-24 | Value-Conditioned Aging HPM | 1.193127 | 13.7577 |
| 2024-25 | Forward box-score interaction HPM | 1.202047 | 14.3191 |
| 2024-25 | Forward box-score residual HPM | 1.202041 | 14.3108 |
| 2024-25 | Value-Conditioned Aging HPM | 1.202038 | 14.3093 |
| 2025-26 | Forward box-score interaction HPM | 1.198771 | 14.3577 |
| 2025-26 | Forward box-score residual HPM | 1.198770 | 14.3548 |
| 2025-26 | Value-Conditioned Aging HPM | 1.198763 | 14.3469 |
<!-- frozen-multiseason-results:end -->
