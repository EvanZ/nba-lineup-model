---
last_updated: "2026-07-31"
---

# ADR-0006: Stats V3-First Historical Game Archive

<p class="adr-status"><strong>Decision status</strong><span>Accepted</span></p>

## Context

[ADR-0001](0001-direct-nba-source.md) established direct NBA-owned source
clients and byte-preserving raw storage. Its original game-feed wording assumed
that historical game data would be acquired through the NBA CDN live-data
endpoints.

That assumption did not hold across older seasons. Historical liveData
availability was incomplete and unreliable, while the season-parameterized NBA
Stats schedule and game-level V3 endpoints provided a repeatable archival path.
The V3 play-by-play schema differs from liveData, particularly around
substitution representation, so source selection must remain explicit at the
normalization boundary.

## Decision

For historical seasons:

- use NBA Stats `scheduleleaguev2` to define the final game catalog;
- archive `playbyplayv3` and `boxscoretraditionalv3` directly from NBA Stats;
- preserve V3 response bytes, URL, headers, fetch time, and SHA-256 sidecars;
- select a validated Stats V3 artifact before a retained liveData artifact for
  each endpoint;
- use legacy liveData only where the matching V3 endpoint is absent from the
  existing cache;
- do not make new historical liveData acquisition attempts part of the normal
  backfill workflow;
- adapt V3 to the common event boundary in memory without rewriting raw JSON;
- record the selected raw source and digest in game-level build provenance.

This decision supersedes ADR-0001's historical live-data feed selection, but
does not change its direct-source, cache-validation, or identifier-typing
requirements.

## Consequences

Historical source coverage is an endpoint-level property. A historical season
can therefore contain a documented mixture of V3-primary and legacy-liveData
fallback documents, especially 2019-20. Availability remains separate from
processing quality: every selected game still passes lineup, possession, score,
and period invariants before it becomes a modeling input.

The project owns the V3 adapter and its tests, but gains a reproducible
historical archive that does not depend on an unstable live-data backfill path.
Future source repairs must preserve raw bytes and create a new processing run;
they may not overwrite source provenance or silently replace a selected feed.
