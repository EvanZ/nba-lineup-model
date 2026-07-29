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

## Allocation-policy incidence

RAPM requires a rule for possessions that cross a substitution boundary. The
canonical `equal_segments` policy divides one possession equally across all of
its fixed-lineup segments. Sensitivity tests also assign the possession to its
starting lineup, terminal lineup, first and last lineups, or exclude it.

Two related rates quantify how much those choices affect the modeling data:

- **Changed possessions** have a policy-specific exposure vector over distinct
  ten-player lineups that differs from `equal_segments`.
- **Reassigned or removed possession-equivalents** measure how much lineup
  exposure moves. This is total-variation distance with an additional removed
  bucket, so excluding one possession counts as one full possession rather
  than one half.

For example, a two-segment possession has canonical shares `(0.5, 0.5)`.
Assigning it to the starting lineup changes one possession but reassigns only
`0.5` possession-equivalents. `boundary_split` leaves it unchanged.

### 2025-26 regular season

The curated regular season contains **245,772 possessions**. Of those,
**26,962 (10.970%)** cross at least one distinct-lineup boundary and can
therefore depend on allocation policy. There are 25,803 two-segment
possessions and 1,159 possessions with three or more segments.

| Policy | Changed possessions | Percent of all possessions | Reassigned or removed equivalents | Percent of all exposure |
| --- | ---: | ---: | ---: | ---: |
| `equal_segments` | 0 | 0.000% | 0.00 | 0.000% |
| `starting_lineup` | 26,962 | 10.970% | 13,665.98 | 5.560% |
| `terminal_lineup` | 26,962 | 10.970% | 13,666.52 | 5.561% |
| `boundary_split` | 1,158 | 0.471% | 392.67 | 0.160% |
| `exclude_multi_lineup` | 26,962 | 10.970% | 26,962.00 | 10.970% |

`boundary_split` and `equal_segments` agree for ordinary two-segment
possessions: each assigns one half to the first and last lineup. They diverge
only for possessions with additional segments or repeated lineups. Starting
and terminal assignment affect every cross-lineup possession, but they
reassign approximately half as much total exposure as outright exclusion.

Reproduce the summary from the curated possession-segment partition:

```python
import pandas as pd

from nba_lineup_model.modeling.allocation import possession_allocation_summary

segments = pd.read_parquet(
    "data/curated/possession_segments/2025-26/regular"
)
summary = possession_allocation_summary(segments)
```

Future RAPM diagnostic runs persist the same table as
`allocation_summary.parquet`.
