---
last_updated: "2026-09-27"
---

# Possession Start-State Mart

`data/analytical/possession_start_states/{season}/regular/` contains one row
per curated regular-season possession. It records only information available at
the start of that possession: the preceding terminal event, the immediate
handoff sequence, and any intervening stoppage or substitution.

It deliberately does not persist the current possession's points, duration, or
other target-outcome fields.

## Granular states

The root `start_state` is mutually exclusive and exhaustive:

| State | Coarse state | Definition |
| --- | --- | --- |
| `period_start` | dead ball | First possession of a period. |
| `after_made_2pt` | dead ball | Opponent inbounds after a made two-point field goal. |
| `after_made_3pt` | dead ball | Opponent inbounds after a made three-point field goal. |
| `after_final_free_throw` | dead ball | Opponent inbounds after a made final non-technical free throw. |
| `after_live_ball_turnover_steal` | live ball | New offense takes possession through a credited steal. |
| `after_other_live_ball_turnover` | live ball | A bad-pass, lost-ball, or comparable live turnover without a separately recorded steal. |
| `after_dead_ball_turnover` | dead ball | A stoppage turnover such as out-of-bounds, traveling, shot-clock, or offensive-foul turnover. |
| `after_defensive_rebound` | live ball | New offense controls a defensive rebound. |
| `after_held_ball` | dead ball | New offense receives a held-ball recovery. The canonical trigger is a non-opening `jumpball/recovered` event. |
| `audited_fallback` | unknown | A sequence that cannot be classified without guessing. |

The first conditional model will pool these states to `live_ball` and
`dead_ball`, but the mart preserves the granular labels for later tests.

An and-one remains in its original possession through the free-throw sequence:
the next possession is labeled from the made final free throw or the rebound of
a missed final free throw. Technical free throws do not create a new start
state unless the canonical possession reconstruction has independently changed
control.

## Provenance and validation

Each partition contains:

- `part-00000.parquet`: possession-start-state rows;
- `state_coverage.parquet`: season counts and shares for every taxonomy state;
- `game_audit.parquet`: one-to-one possession accounting by game; and
- `fallback_audit.parquet`: every unresolved sequence with its classification
  reason and terminal-event provenance.

`_manifest.json` records input checksums for both the curated event and
possession partitions, output checksums, the builder fingerprint, and the
taxonomy contract.

Build all locally available regular seasons without downloading source data:

```bash
uv run nba-build-possession-start-state-mart
```

Build selected seasons:

```bash
uv run nba-build-possession-start-state-mart --season 2024-25 --season 2025-26
```
