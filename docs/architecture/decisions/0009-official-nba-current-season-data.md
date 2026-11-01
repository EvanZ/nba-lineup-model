---
last_updated: "2026-10-31"
---

# ADR-0009: Official NBA Current-Season Data

<p class="adr-status"><strong>Decision status</strong><span>Accepted</span></p>

## Context

The first current-season updater reused ESPN schedule summaries and box scores
that had originally been collected for the public schedule page. That made the
modeling pipeline translate ESPN athlete identifiers and names back to NBA
player identifiers. Rookies, namesakes, recent transactions, DNP rows, and
provider corrections then required matching and repair logic even though the
NBA's own box score already publishes the canonical `personId`.

This was not merely an implementation inconvenience. It allowed a presentation
feed and its identity system to become an authoritative training-data input.

## Decision

All authoritative current-season basketball data comes from official NBA
sources:

- the official NBA schedule catalog defines the exact Pacific-date slate;
- NBA liveData box scores define teams, final scores, starters, player
  identities, game rosters, and actual minutes;
- NBA liveData play-by-play defines events, substitutions, scores, and
  possessions;
- official NBA injury reports define pregame availability; and
- official NBA transaction/player-movement data defines roster changes.

Raw NBA box scores and play-by-play are archived by SHA-256 under immutable S3
keys. Daily observed-data artifacts retain the exact source URI and checksum for
every game. Player names are labels only; joins use the NBA `personId`.

ESPN sports-data APIs, ESPN game IDs, ESPN box scores, ESPN rosters, and
name-based ESPN-to-NBA crosswalks are not authoritative inputs and must not be
used as fallbacks. A missing NBA source blocks that daily artifact instead of
silently switching providers. Betting markets remain separate external market
inputs and do not define basketball facts.

This decision supersedes ADR-0008's choice not to acquire the liveData box
score. ADR-0008's play-by-play archive and canonical-event boundary remain in
force.

## Consequences

The daily chain no longer has a player-identity reconciliation layer between a
box-score provider and the NBA catalog. Historical current-season artifacts
must be replayed from the first regular-season date before any model artifact is
rebuilt. Existing ESPN-backed serving snapshots remain legacy artifacts only
until the schedule-serving migration replaces them; they cannot feed the new
data or model chains.
