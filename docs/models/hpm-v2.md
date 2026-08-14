---
last_updated: "2026-08-13"
---

# HIPSTER PM v2: Depth-Aware Shooting

HIPSTER PM v2 is a controlled contextual-feature experiment. It retains the
current Value-Conditioned Aging HPM player prior, exposure-gated cold-start
path, rolling seasonal information boundary, 5th--95th percentile feature
bounds, hierarchical P-spline penalties, and portable-matchup decomposition.
Only the shooting representation changes.

The purpose is to distinguish a lineup with one high-volume shooter from a
lineup whose spacing is supplied by several credible shooters. The model still
uses only prior-season player profiles when it forecasts a season.

## Feature Registry

| Feature family | HPM v1 | HPM v2 status | HPM v2 representation | Rationale |
| --- | --- | --- | --- | --- |
| Raw three-point attempts | Active | Retired | Removed | A single player can dominate a unit total, so this is not a pure combination measure. |
| Raw three-point makes | Active | Retired | Removed | Same single-player dominance problem. |
| Weak-link shooting | `bottom_two_three_pm` | Replaced | `bottom_three_three_pm` | The three weakest shooters carry more information about lineup-wide spacing than the two weakest alone. |
| Credible shooter depth | Active | Retained | Count of players at or above 2.0 prior-season 3PM per 100 possessions | Direct count of players who can contribute meaningful spacing. |
| Capped shooting capacity | None | Added | \(\sum_i\min(\mathrm{3PM}_{i},2.0)\) | Allows broad shooting capacity to matter while capping the marginal credit from any one player. |
| Shooting concentration | None | Added | \(\sum_i s_i^2\), where \(s_i=\mathrm{3PM}_i/\sum_j\mathrm{3PM}_j\) | Separates concentrated, star-driven shooting from distributed shooting. |
| Shooting by usage | `bottom_two_three_pm * usage_concentration` | Replaced | `bottom_three_three_pm * usage_concentration` | Preserves the v1 interaction but makes the shooting side depth-aware. |
| Shooting by passing | Active | Retained | `credible_shooter_count * top_two_assists` | Tests whether shooting depth is more valuable with concentrated creation. |
| Passing, usage, rebounding, steals, blocks, turnover, uncertainty features | Active | Retained | Unchanged | Keeps the ablation isolated to shooting context. |

`bottom_three_three_pm` is the sum of the three lowest player-level prior
3PM-per-100 profiles in the unit. `shooting_concentration` is high when one
player supplies most of the expected shooting and lower when it is distributed.
The model learns all feature weights and nonlinear response shapes; these
features do not impose a hand-set basketball effect or a positive sign.

## Evaluation Contract

Each completed season first fits the same HPM v2 contextual state from the
season's observed stints. That state modifies only the *next* season's RAPM
update. For frozen 2025-26 evaluation, every profile, player prior, spline,
and support bound comes from 2024-25 or earlier. 2025-26 outcomes are never
used to construct the forecast state.

## Recovered-History Frozen Result

The completed artifact is
`forward-hpm-v2-depth-aware-shooting-2025-26-20260813T174513Z-c996ddaf`.
It uses the same recovered historical data and identical 2025-26 frozen
boundary as the current Value-Conditioned Aging HPM artifact, so the two rows
below are a direct ablation comparison. This comparison is separate from the
public Frozen Preseason Leaderboard, whose full candidate set still needs a
common recovery-era rebuild.

| Metric | Value-Conditioned Aging HPM | HPM v2 | Better |
| --- | ---: | ---: | --- |
| Regular possession RMSE | **1.198736** | 1.198758 | HPM v1 |
| Regular eligible game-margin RMSE | **14.3214** | 14.3487 | HPM v1 |
| Regular full-game margin RMSE | **14.5898** | 14.6164 | HPM v1 |
| Regular winner accuracy | **68.46%** | 67.97% | HPM v1 |
| Playoff possession RMSE | **1.192465** | 1.192485 | HPM v1 |
| Playoff eligible game-margin RMSE | **16.4069** | 16.4450 | HPM v1 |
| Team NetRtg RMSE | **3.5858** | 3.6739 | HPM v1 |
| Pythagorean wins RMSE | **7.6300** | 7.9076 | HPM v1 |

This is a useful negative result. The depth-aware representation is more
faithful to the intended interpretation of lineup context, but under this
single declared candidate specification it does not improve forecast accuracy.
HPM v1 remains the website and recovered-history reference model. Future
shooting-context variants should be declared as new rows in the Feature
Registry and evaluated against this same frozen contract.

## Reproduce

```bash
uv run nba-train-hpm-v2 --through-season 2025-26
```

See [the HPM v2 training guide](../guides/train-hpm-v2.md) for the artifact
location and a log-tail command.
