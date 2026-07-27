# Architecture Overview

The repository is organized around transformations from immutable source
documents to progressively stronger basketball semantics.

!!! abstract "Boundary rule"

    Every layer strengthens basketball meaning without obscuring the source
    evidence needed to reproduce or challenge that interpretation.

## Boundaries

### Source access

`nba_lineup_model.ingest` owns direct game-feed access.
`nba_lineup_model.season.schedule` owns the season-addressable schedule client.
Both preserve raw response bytes before source-specific normalization.

### Canonical events

`nba_lineup_model.events` converts source actions into ordered, typed events.
This layer standardizes clocks, score deltas, identifiers, and event metadata
while preserving source values that later algorithms may need.

### Lineup reconstruction

`nba_lineup_model.lineups` combines boxscore starters with play-by-play
substitutions. It produces a lineup before and after every event, plus stable
lineup stints.

### Possession reconstruction

`nba_lineup_model.possessions` applies basketball transition rules to canonical
events. It uses the source possession team as a strong signal while allowing
explicit outcomes such as turnovers and defensive rebounds to override it.

### Auditing

`nba_lineup_model.audit` runs the same in-memory reconstruction pipeline over a
versioned game manifest. It records exact invariant failures separately from
diagnostic warnings.

### Modeling

`nba_lineup_model.models` and `nba_lineup_model.evaluation` are deliberately
thin until the data contracts are stable. Initial models will establish ridge
and tree baselines before nonlinear lineup architectures are introduced.

## Core orchestration

`reconstruct_game_payloads` is the shared in-memory pipeline. The single-game
builder persists its outputs, while the audit runner evaluates the same objects
without writing every intermediate table.

This separation prevents the validation path from drifting away from production
processing.

## Failure policy

- Invalid source structure raises an exception.
- Impossible lineup transitions raise a reconstruction exception.
- Recoverable feed anomalies become structured warning or error records.
- Audit execution isolates failures by game so one bad endpoint does not abort
  a multi-season run.
- Exact score and duration failures make an audit game fail.
- Approximate boxscore possession estimates remain diagnostics.
