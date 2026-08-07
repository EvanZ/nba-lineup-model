---
last_updated: "2026-07-27"
---

# ADR-0001: Direct NBA Source Data

<p class="adr-status"><strong>Decision status</strong><span>Accepted</span></p>

## Context

Open-source NBA packages can simplify endpoint access, but they introduce
another interpretation layer and can change field types, naming, retries, or
caching behavior. This project needs to investigate feed behavior directly,
including differences across seasons.

## Decision

Use small internal clients for NBA-owned source endpoints. Game feeds come from
NBA CDN live-data endpoints. Historical season discovery comes from the
season-parameterized NBA Stats `scheduleleaguev2` endpoint because the regional
CDN schedule file is not season-addressable.

- Store response bodies byte-for-byte.
- Store URL, fetch time, endpoint, game ID, and SHA-256 in a sidecar.
- Validate cache hashes on read.
- Keep normalization separate from fetching.
- Use source game IDs as strings and NBA numeric identifiers as integers.

## Consequences

The project owns endpoint compatibility and request behavior. In return, source
provenance is inspectable and feed anomalies can be reproduced without depending
on package-specific transformations.
