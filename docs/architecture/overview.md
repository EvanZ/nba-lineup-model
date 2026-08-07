---
last_updated: "2026-07-27"
---

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
`nba_lineup_model.players` owns bulk historical player identity and season bio
access. All preserve raw response bytes before source-specific normalization.

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

### Orchestration

`nba_lineup_model.flows` contains thin Prefect wrappers around project-owned
selection, fetching, validation, and storage functions. Prefect supplies local
concurrency, retries, and observable task state. Raw JSON and Parquet manifests
remain authoritative so orchestration can be changed without changing the data
contracts. Season compaction uses one task per table and season-type partition;
the underlying compactor remains independently callable and testable.

### Modeling

`nba_lineup_model.modeling` owns modeling-table construction, chronological
splits, run manifests, and artifact publication. `nba_lineup_model.models`
owns estimators and sparse signed encodings; `nba_lineup_model.evaluation`
owns weighted and game-aggregated metrics.

The first boundary is a regular-season one-number RAPM benchmark. It compares
an intercept-only mean, schedule-adjusted team strengths, and signed player
ridge coefficients before any offensive/defensive split, player prior, or
nonlinear interaction is introduced.

## Core orchestration

`reconstruct_game_payloads` is the shared in-memory pipeline. The season
processor reconstructs once, evaluates the audit invariants, and only then
persists the six game tables. The single-game builder persists directly, while
the audit runner evaluates the same objects without writing every intermediate
table.

This separation prevents the validation path from drifting away from production
processing.

## Failure policy

- Invalid source structure raises an exception.
- Impossible lineup transitions raise a reconstruction exception.
- Recoverable feed anomalies become structured warning or error records.
- Audit execution isolates failures by game so one bad endpoint does not abort
  a multi-season run.
- Season fetching retries only transient network and source failures, then
  records every terminal game outcome.
- Season processing isolates deterministic reconstruction and quality failures
  by game and checkpoints terminal metadata through a single writer.
- Season compaction rejects missing or mismatched build and quality provenance,
  then atomically publishes only row-conserving partitions.
- Exact score and duration failures make an audit game fail.
- Approximate boxscore possession estimates remain diagnostics.
