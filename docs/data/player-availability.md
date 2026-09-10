# Player-Game Availability

This mart records whether a player appeared, was available but not selected by
the coach, or was explicitly unavailable for a listed reason. It begins in
2015-16, when the historical official box-score archive provides usable DNP
comments.

## Why This Exists

Games played is an appearance outcome, not an availability label:

\[
P(\mathrm{appears}) = P(\mathrm{available}) \times
P(\mathrm{selected\ by\ coach}\mid\mathrm{available}).
\]

A healthy end-of-bench player can receive a DNP-CD for many games. Rotation
models must not convert those coaching decisions into injury absences.

## Sources and Precedence

For every scheduled regular-season game, the builder prefers the local live
NBA CDN box score when available, because it exposes structured fields:

- `status`
- `notPlayingReason`
- `notPlayingDescription`

Otherwise it falls back to the historical Stats V3 traditional box score,
which preserves the free-text `comment` field such as `DNP - Coach's
Decision` and `DND - Injury/Illness`.

The source path and every raw reason field remain in the published row.

### Source Coverage Caveat

The two official source families do not currently have the same player-table
contract. Modern live box scores enumerate the full roster with a structured
status for each player. Historical Stats V3 traditional box scores generally
enumerate only players listed on that game's box score. A player absent from a
historical Stats V3 table is therefore **not** evidence of either availability
or injury.

Each row and game-coverage record includes `source_player_table_contract`:

| Value | Meaning |
| --- | --- |
| `full_roster_status` | Modern source: valid rostered-game denominator, with structured status. |
| `full_roster_membership` | Historical Stats V3 joined to official Summary V2 `InactivePlayers`: valid rostered-game denominator. An explicitly named inactive player is unavailable even if the older source omits the reason. |
| `boxscore_listed_players` | Historical Stats V3 alone: listed player rows only; not a complete availability denominator. |
| `missing` | No local source document. |

The official `BoxScoreSummaryV2` endpoint recovers historical inactive-player
membership. The approved binary contract labels a player named there as
unavailable even though that endpoint does not include an absence reason.
Joining it to Stats V3 therefore produces a complete historical denominator
without requiring a gamebook-PDF backfill.

### Contradictory Single-Game Statuses

The mart applies one narrow temporal correction after source normalization: an
`available_dnp_coach` row is reclassified as injury/illness unavailable only
when the same player on the same team has explicit injury/illness absences in
the immediately preceding and following team games. This addresses isolated
source-label errors inside a continuous medical absence without changing
ordinary coach decisions.

## Contract

```text
data/curated/player_availability/<season>/part-00000.parquet
data/curated/player_availability/<season>/game_coverage.parquet
data/curated/player_availability/<season>/coverage.parquet
data/curated/player_availability/<season>/state_counts.parquet
data/curated/player_availability/<season>/reason_counts.parquet
data/curated/player_availability/<season>/_manifest.json
data/curated/player_availability/coverage.parquet
```

The canonical row key is `(game_id, team_id, player_id)`. Important columns:

| Column | Meaning |
| --- | --- |
| `availability_state` | Normalized state described below. |
| `available` | The binary target: `true` for played players, coach DNPs, active zero-minute players, and G League/two-way assignments; `false` for explicit injury, rest, personal/not-with-team, ineligible, suspension, and a player explicitly named `Inactive` without a reason; null only for a blank zero-minute box-score row with no inactive designation. |
| `availability_state_known` | `false` only when a zero-minute row has no usable reason. |
| `raw_status` | Structured modern NBA player status when provided. |
| `raw_not_playing_reason` | Structured modern reason code when provided. |
| `raw_not_playing_description` | Structured modern detail, commonly injury body part and condition. |
| `raw_comment` | Historical Stats V3 DNP/DND/NWT text. |
| `source_kind`, `source_path` | Exact provenance for the normalization. |
| `source_player_table_contract` | Whether the source is a full roster-status table or only a listed-player table. |

States are deliberately conservative:

| State | Interpretation |
| --- | --- |
| `played` | Positive minutes or an explicit played flag. |
| `available_dnp_coach` | Explicit coach-decision DNP. |
| `available_dnp_active` | Modern box score explicitly lists the player as active but they receive zero minutes. |
| `available_g_league_assignment` | A listed player is assigned to the G League or on a two-way assignment. This is availability-positive, effectively a team deployment decision rather than a health absence. |
| `unavailable_injury_or_illness` | Explicit injury, illness, concussion, health-and-safety, or return-to-competition status. |
| `unavailable_rest` | Explicit rest designation. |
| `unavailable_personal_or_not_with_team` | Personal, not-with-team, or trade status. |
| `unavailable_suspension` | Suspension status. |
| `unavailable_ineligible` | Explicit ineligible-to-play status. |
| `unavailable_inactive_unspecified` | A player explicitly named on a historical inactive list without an accompanying reason. The binary contract labels this unavailable. |
| `unknown_dnp_reason` | A listed zero-minute player without a defensible reason or inactive designation. This receives no binary label. |

## Build

```bash
uv run nba-build-player-availability 2015-16 2016-17 2017-18
```

Build every completed source season, then the command also refreshes the
cross-season `coverage.parquet` audit. This mart is a data prerequisite for a
future availability model; it does not itself fit one and does not use GP as a
target.
