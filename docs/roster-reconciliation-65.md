# Durable in-season reconciliation contract (#65)

## Ownership and scope

The existing `reconcile_in_season_rosters` job owns discovering missing season
participants and preparing their inputs. Its existing historical-player rating,
profile, and minutes initialization functions remain the calculation authority.
The daily model owns performance updates. Full trades and trade-history replay
are deferred; unsupported changes are reported without resetting existing state.

Official source captures may be shared. Each environment selects and processes
its own immutable reconciliation result. Production activation is not authorized
by this implementation. The initial validation target is dev, including the
November 4 job that failed on Dalen Terry (1631207).

## Artifact and publication transaction

A reconciliation artifact contains the complete current catalog plus a cumulative
introduction ledger, the original base-prior identity, parent reconciliation
reference, exact output inventory, official source evidence, and initialization
input/rule provenance. Files and identity are checksummed. Source files required
for reproduction belong to the artifact or a verified immutable dependency.
Local filesystem paths are transport details, not durable identities.

Compute the content identity from pinned inputs, parent, and producer version.
Capture/discovery timestamps are pinned once per request; retries reuse them.
Publish all objects and the storage manifest through the existing artifact store,
verify them, and only then advance the environment's reconciliation selection.
A retry always verifies/completes durable publication, even when inputs match.
A local completion marker is never evidence that S3 publication succeeded.

Selection updates are serialized and compare the expected parent. A stale
reconciliation cannot replace a newer accepted result. A failed transaction
preserves the preceding selection. Recovery starts from the recorded artifact
reference and retrieves/validates the complete inventory from S3.

## Introduction ledger and time

An introduction records NBA ID, team, actual discovery/resolution timestamp,
effective date, source evidence, initialization provenance, and any manual
authorization. The ledger is cumulative and immutable within an artifact.
Later artifacts append introductions; corrections create a new selected history.
Existing player initialization and accumulated model state are never reset.

Automatic introductions use captured official roster evidence. Manual requests
enter the same producer and validation path. They must identify saved official
membership evidence, the saved biography/initialization inputs, the effective
date, and the authorizing decision. They do not fabricate an all-team live roster
snapshot or backdate discovery. The two source roles are explicit: membership
evidence establishes the team/date; a historical biography establishes identity
and initialization fields, not current membership.

A late introduction effective after the accepted model boundary can be consumed
by the next chronological run. Changing already accepted dates requires a scoped
impact assessment and historical replay approval. An introduction cannot provide
future performance evidence to earlier initialization or alter frozen pregame
forecasts. The exact discovery-time cutoff for forecast use remains enforced.

## Consumer contract

Every consuming work specification pins its reconciliation artifact reference.
A requested date selects only introductions effective by that date. The original
opening prior remains the model provenance anchor; appending a player is a dated
input transition, not replacement of the opening prior for every past date.

Before model processing, verify complete source-roster/player coverage and adopt
missing introductions. Preserve all accepted rating sufficient statistics,
minutes state, trajectories, and home-court state by NBA identity. Persist the
selected introduction history with the model checkpoint and durable dependencies.
A catalog player missing initialization is detected explicitly.

Causal replay reads the introduction history selected by the season edition,
applying saved initializations chronologically with no live discovery. Accepted
history compatibility includes the date-applicable introductions; a changed
historical introduction cannot silently reuse an incompatible model prefix.
The original season selection remains reproducible.

Forecasting and site publication must use the selected initialized player catalog
and profiles through the canonical release workflow. A missing introduction or
input blocks affected publication; it never drops players/games or invents data.

## Scheduling and acceptance

Register reconciliation in the environment-bound daily instance, using the same
run guard and environment-scoped history. Keep the four daily checks and require
verified reconciliation coverage before dependent daily/forecast work. Dev
scheduling is enabled only for an explicit validation window. Protected raw
collectors remain independent and operational.

Acceptance covers successive additions, no-op/retry behavior, partial uploads,
cold-cache S3 recovery, late/manual introduction on both sides of the accepted
boundary, chronological replay without discovery, and incumbent-state invariance.
The actual November 4 dev job must succeed with full slate/player coverage, then
pass artifact, API, rendered-page, and frozen-forecast comparisons. Its original
failed run remains in history; recovery is proven by success of the affected work.


## Approved forecast-publication extension

The user approved extending the existing forecast publication workflow after
reviewing Terry and the existing cold-start behavior. A forecast release may
publish saved zero-evidence initialization before the next completed-game update.
The accepted daily checkpoint, game-evidence date, incumbent numerical state,
and historical frozen forecasts remain unchanged. The serving information cutoff
and selected introduction reference are explicit. The release validator requires
complete catalog/profile/serving coverage and exact incumbent preservation.

The existing automatic lifecycle sensor performs the four daily reconciliation
checks first (06/09/12/15 Pacific), then considers nightly/forecast work. This
avoids competing schedulers. Manual historical daily recovery uses the selected
verified introduction history and complete official game-roster coverage.
Production captures the official roster once per slot; dev consumes that saved
capture. Missing shared captures fail explicitly.

An introduction affecting already accepted dates can be archived with a precise
impact assessment without moving the accepted reconciliation selection. Selecting
that correction requires a decision bound to the accepted model and affected
start/end dates. The existing causal replay approval gate remains separate;
forecasting rejects a corrected history until the corresponding replay completes.
