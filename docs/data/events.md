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

For liveData, `source_order_number` is the NBA-provided `orderNumber`. Stats V3
does not provide that field, can repeat `actionNumber` for related records, and
can append corrections after later actions. The adapter orders by period and
game clock, then uses `actionNumber` and `actionId` as same-clock tie-breakers.
It encodes those fields into a unique integer ordering key while retaining the
original `actionNumber` in `source_action_number`.

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

## Stats V3 adaptation

The V3 feed uses a different event vocabulary and combines each substitution
into one record such as `SUB: Wright FOR Curry`. Before canonical event
normalization, the adapter:

- maps made and missed shots to `2pt` or `3pt`;
- normalizes free-throw sequences and foul descriptors;
- classifies team and player rebounds using the preceding missed attempt;
- restores explicit team IDs on team rebounds and turnovers;
- assigns team heaves from the V3 home/visitor location marker;
- maps separate steal and block records;
- expands each substitution into ordered `out` and `in` events; and
- labels periods after the fourth as overtime.

V3 can omit lineup changes between periods. The adapter infers each period's
opening lineup from player activity and substitution direction, then emits
explicit boundary substitution events with
`descriptor = stats_v3_period_lineup`. These events expose the inference in
processed data and subject it to the existing lineup and minute audits.

## Output

`data/processed/events/{game_id}.parquet` contains one row per canonical event.
