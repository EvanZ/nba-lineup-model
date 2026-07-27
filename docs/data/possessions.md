# Possessions

Possession reconstruction is a finite-state process over canonical events. It
combines source possession labels with explicit basketball outcomes.

## Source signal and overrides

`source_possession_team_id` is a strong signal, but it can switch before all
companion actions for the previous possession have been emitted. The
reconstruction therefore prioritizes semantic terminal events when the two
disagree.

Expected source overrides are counted on each possession for auditability.

## Terminal reasons

Possessions can end with:

| Reason | Interpretation |
| --- | --- |
| `made_field_goal` | Made two- or three-point field goal |
| `made_final_free_throw` | Made final non-technical free throw |
| `turnover` | Explicit turnover action |
| `defensive_rebound` | Defense controls a missed shot |
| `held_ball` | Opponent recovers a held-ball jump |
| `period_end` | Active possession reaches the period boundary |
| `source_change` | Unexplained source transition retained for diagnosis |
| `feed_end` | Feed ends during an active period |

Made scores and held-ball recoveries remain pending briefly so companion actions
can confirm or supersede the boundary.

## Important feed sequences

### Offensive-foul companions

An offensive foul may carry the next team's source possession label before the
turnover companion action. Both events stay with the original offense.

### Held ball followed by turnover and steal

Some feeds describe one change of control as a held-ball recovery followed by a
turnover and steal at the same clock. The turnover supersedes the pending
held-ball terminal, preventing a phantom possession.

### Technical free throws

A technical free throw can score for the defensive team under the CDN's active
possession label. The point is preserved and marked as
`opponent_technical_free_throw` rather than treated as an accounting failure.

### Post-score loose-ball free throws

Newer feeds can label the prospective next offense immediately after a made
basket, then award a same-clock loose-ball penalty free throw to the scoring
team. The foul and free throw remain continuation context for the scoring
possession.

### Legacy labels

Older live-data feeds use free-throw labels such as `2of2` and may omit
`isFieldGoal`. Canonical event type and shot result provide the stable terminal
signals across these versions.

## Possession contract

`possessions/{game_id}.parquet` contains one row per basketball possession,
including:

- offense and defense team IDs;
- period-local and game-global indexes;
- start and end events, clocks, and elapsed times;
- points by team and offense points;
- start and terminal reasons;
- event count, source mismatch count, and validation flags;
- number of fixed-lineup segments.

## Fixed-lineup segments

`possession_segments/{game_id}.parquet` intersects each possession with atomic
substitution boundaries.

A segment stores one home lineup, one away lineup, points, duration, and its
position within the parent possession. Segment scores and durations must sum
exactly to the possession totals.

!!! success "Accounting invariant"

    For every possession, segment points and elapsed duration must sum exactly
    to the parent row. A mismatch is an audit failure, not a warning.
