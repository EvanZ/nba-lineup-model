---
last_updated: "2026-08-21"
---

# Assisted Shot Profiles

This mart extends the canonical [shot taxonomy](shot-taxonomy.md) with an
explicit assisted-shot label for **made** field goals. It is a validated data
contract used by the non-promoted NAIL-RAPM v1.3 additive-profile experiment
and retained for future player-profile work.

## Contract

Each final made two- or three-point play-by-play event belongs to one of three
families and exactly one assist-status bucket.

| Dimension | Values |
| --- | --- |
| Shot family | `rim`, `non_rim_two`, `three` |
| Assist status | `assisted`, `unassisted`, `unknown` |

`rim` and `non_rim_two` use the same robust event-subtype rules as the shot
taxonomy mart. The builder marks a make `assisted` only when its source
description explicitly contains an assist marker such as `(A. Player 1 AST)`.
It marks a make `unassisted` only when a non-empty description lacks that
marker. A missing description remains `unknown`; it is never silently treated
as unassisted.

This produces raw counts and possession-native per-100 rates such as
`unassisted_rim_makes_per_100`, `assisted_three_makes_per_100`, and, for
completeness, `assisted_non_rim_two_makes_per_100`.

## Validation

The builder independently aggregates every team-game from the processed player
box scores. It then records four reconciliation checks:

| Check | Play-by-play total | Official comparison |
| --- | --- | --- |
| Field goals made | All classified made twos and threes | Sum of player FGM |
| Two-point makes | Classified made twos | Sum of player 2PM |
| Three-point makes | Classified made threes | Sum of player 3PM |
| Assisted makes | Description carries an `AST` marker | Sum of player assists |

The game-level table preserves deltas and exact-match flags rather than hiding
source disagreement. The season summary reports exact team-game rates, mean
and maximum absolute deltas, missing-box coverage, and the rate of make events
whose assist status is unknown.

Assisted plus unassisted plus unknown makes always equal the mart's total made
shots by construction. The box-score reconciliation is the independent check
that the source play-by-play itself is complete enough to support the split.

## Storage

```text
data/analytical/assisted_shot_taxonomy/
  _manifest.json
  player_season_assisted_shot_profiles.parquet
  season_assisted_shot_coverage.parquet
  game_assisted_shot_reconciliation.parquet
  season_assisted_shot_reconciliation.parquet
```

The atomic manifest hashes the player-season panel, every curated event
manifest, and every published output. It also records the processed player-box
directory used for the independent reconciliation.

## Build

```bash
uv run nba-build-assisted-shot-taxonomy
```

The command uses only local artifacts and makes no NBA API requests. See
[Build assisted shot profiles](../guides/build-assisted-shot-taxonomy.md) for
the validation command.
