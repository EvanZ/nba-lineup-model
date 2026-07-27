# Events

The canonical event stream is the first project-owned basketball contract. One
NBA source action becomes one `Event`.

## Identity and ordering

| Field | Meaning |
| --- | --- |
| `game_id` | Source game ID stored as a string |
| `event_id` | `{game_id}:{source_order_number}` |
| `event_index` | Dense zero-based index after ordering |
| `source_action_number` | NBA action number |
| `source_order_number` | NBA ordering key |

Source order numbers are integers. Their magnitude is not a timestamp and no
precision is discarded.

## Clocks

The event contract retains both clock representations:

- `source_clock`: source ISO-8601 duration such as `PT10M53.00S`.
- `clock`: canonical display value such as `10:53.00`.
- `seconds_remaining_period`: numeric period clock.
- `elapsed_game_seconds`: elapsed time across regulation and overtime periods.

Regulation periods are 12 minutes. Periods after the fourth are five-minute
overtimes.

## Scores

Every event stores cumulative home and away scores plus deltas from the previous
ordered event. Negative corrections are preserved and marked with validation
flags rather than silently clamped.

## Identifiers

NBA identifiers such as team and player IDs use nullable integer columns in
Pandas and physical `int64` fields in Parquet. They are not converted to floating
point values.

## Source semantics

Fields such as `event_type`, `event_subtype`, `descriptor`, `qualifiers`,
`shot_result`, and `source_possession_team_id` remain close to the NBA response.
Downstream modules interpret combinations of these fields; event normalization
does not attempt to fully classify possession outcomes.

## Output

`data/processed/events/{game_id}.parquet` contains one row per canonical event.
