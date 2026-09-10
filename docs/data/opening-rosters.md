---
last_updated: "2026-09-08"
---

# Historical Opening Rosters

The opening-roster mart is the preseason membership contract for historical
rotation and minutes experiments. It reconstructs team membership from the
official NBA player-movement feed, then checks that reconstruction against the
player rows in each team's first regular-season box score.

## Source Boundary

The official [NBA player-movement feed](https://stats.nba.com/js/data/playermovement/NBA_Player_Movement.json)
begins on 2015-07-01. Consequently, the mart covers `2015-16` forward. It does
not make an unsupported claim about pre-2015 opening rosters.

For team \(T\) in season \(s\), the snapshot starts from the final observed
regular-season roster in \(s-1\), then applies supported movement records
strictly after that season's final game and strictly before \(T\)'s first
regular-season game date:

\[
R^{open}_{T,s} = \operatorname{apply}(R^{final}_{s-1}, M_{(d^{final}_{s-1},\ d^{open}_{T,s})}).
\]

The strict upper bound is intentional. Movement rows are date-only, so a
transaction reported on opening day cannot be ordered safely against the game.
The source-supported state transition is:

| Movement | State transition |
| --- | --- |
| `Signing`, `ContractConverted`, `AwardOnWaivers` | add player to the listed team |
| `Trade` | move player to the listed destination team |
| `Waive` | remove player when the listed team is their current team |

## Reconciliation and Auditability

When it is available, the opening-game player table defines the published
membership. In modern source seasons it includes inactive players; some older
NBA box-score formats list a narrower game-available group. The mart therefore
does not claim to be a complete 15-player-plus-two-way official roster archive
for every historical season. If an observed opening-game player is missing from
the reconstructed state, the mart adds the player with
`membership_source = opening_game_reconciliation`. The reason for every such
addition is retained. This makes gaps visible rather than silently filling
them.

The movement feed is not a complete training-camp waiver ledger. A
transaction-derived player who is absent from the opening-game roster is
written to the audit-only `transaction_only_candidates.parquet`, not the
published roster. If the opening-game player table is unavailable, the mart
falls back to the transaction state and labels every row
`transaction_state_unvalidated`.

```text
data/raw/player_movement/NBA_Player_Movement.json
data/raw/player_movement/NBA_Player_Movement.meta.json
data/curated/opening_rosters/coverage.parquet
data/curated/opening_rosters/<season>/part-00000.parquet
data/curated/opening_rosters/<season>/validation.parquet
data/curated/opening_rosters/<season>/reconciliation.parquet
data/curated/opening_rosters/<season>/transaction_only_candidates.parquet
data/curated/opening_rosters/<season>/_manifest.json
```

`validation.parquet` reports one row per team: source availability, opening
box-score count, matched reconstruction count, reconciliation count, and the
transaction-state players not observed in the first game.
`reconciliation.parquet` lists player-level additions and their reason.
`transaction_only_candidates.parquet` lists movement-state candidates that did
not pass the opening-game membership check.

`coverage.parquet` is the cross-season quality summary: one row per completed
mart, including source coverage, direct reconstruction, reconciliation,
unvalidated fallback, and excluded-candidate counts.

## Build

```bash
uv run nba-build-opening-rosters 2024-25 2025-26
```

To recover only the official box-score player tables missing for opening games
(without attempting full lineup reconstruction):

```bash
uv run nba-build-opening-rosters 2015-16 2016-17 --recover-missing-opening-games
```

To refresh the official transaction source first:

```bash
uv run nba-build-opening-rosters 2024-25 2025-26 --refresh-movements
```

The mart is roster membership only. It does not infer injuries, availability,
or minutes. A forecast using it must state the snapshot season and must not
include a player whose membership was learned after the declared opening-date
cutoff, except where the explicit opening-game reconciliation proves that the
player was already on that team. For pre-2019 rotation research, downstream
models should treat the mart as an opening-game availability constraint rather
than an exhaustive active-roster list.
