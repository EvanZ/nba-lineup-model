# Lineups

Lineup reconstruction combines boxscore starters with ordered substitution
events.

## Starting state

Exactly five unique starters are required for each team. Player IDs are sorted
to form stable lineup keys; ordering within a five-player tuple has no tactical
meaning.

## Event-level assignments

`event_lineups/{game_id}.parquet` records:

- the home and away lineups before each event;
- the home and away lineups after each event;
- an optional atomic substitution batch ID.

For non-substitution events, before and after states are equal.

## Atomic substitution batches

Consecutive substitution records form one batch. All outgoing and incoming
players are applied together before the next basketball event.

This handles:

- both teams substituting at the same dead ball;
- source ordering that lists all outgoing players before incoming players;
- a player logged in and back out before play resumes, which becomes a net-zero
  change within the batch.

The resulting state must still contain exactly five unique players per team.

## Lineup stints

`lineup_stints/{game_id}.parquet` contains intervals with stable five-on-five
lineups. A stint starts at a period boundary or substitution and ends at a
substitution, period end, or feed end.

Each stint stores:

- time and event boundaries;
- score boundaries and points by team;
- home and away player tuples;
- explicit start and end reasons.

## Validation

Reconstructed player seconds are compared with boxscore minutes. Primary actors
are checked against the lineup except for event classes where an on-court actor
is not required, such as team or bench technical fouls.

Overtime periods use five-minute duration calculations and participate in the
same stint and minute validations.
