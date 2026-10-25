---
last_updated: "2026-10-24"
---

# ADR-0008: Current-Season CDN Play-by-Play Ingestion

<p class="adr-status"><strong>Decision status</strong><span>Accepted</span></p>

## Context

Historical training data is primarily archived from NBA Stats V3, with older
V2 and retained liveData used where required. Direct `stats.nba.com` requests
from the production EC2 address time out because NBA filters many cloud-hosting
IP ranges. The NBA liveData CDN remains reachable from EC2 with current browser
headers and exposes structured in/out substitutions, ordering keys, possession
team IDs, event timestamps, and correction metadata.

The current schedule pipeline already retains final box scores and actual
minutes, so a second box-score acquisition path would duplicate an established
source without filling the missing training-data gap.

## Decision

For current-season nightly acquisition:

- archive final-game CDN `playbyplay` response bytes directly from EC2;
- do not fetch a duplicate CDN box score solely for actual minutes;
- retain every distinct response checksum as an immutable revision;
- re-fetch the prior day's final games at 08:00 America/New_York to capture
  upstream corrections;
- retain Stats V3 acquisition as an optional repair and validation path;
- keep CDN and Stats V3 raw objects in separate source namespaces; and
- reconcile sources only after adapting them to the canonical event contract.

Training builds must never union raw CDN, V3, or V2 actions directly. They
select one source revision per game and endpoint, normalize it to canonical
events, validate lineups, scores, periods, and minutes, and retain the selected
source and SHA-256 digest in build provenance. Canonical event identity is
project-owned; raw source action numbers are lineage fields, not cross-source
join keys.

## Consequences

Production collection no longer depends on a residential proxy. Current-season
CDN rows can be trained beside historical V3/V2 rows only after the same
normalization and validation gates. Cross-source overlap games become a useful
conformance sample, but one game contributes only one selected event stream to
a training build.
