---
last_updated: "2026-07-26"
---

# ADR-0002: Separate Possessions and Lineup Segments

<p class="adr-status"><strong>Decision status</strong><span>Accepted</span></p>

## Context

A basketball possession can contain substitutions. Free-throw sequences are a
common example: players may enter or leave between attempts without creating a
new possession.

A single table cannot simultaneously mean "one basketball possession" and "one
fixed five-on-five sample."

## Decision

Produce two related contracts:

- `possessions`: one row for the full basketball possession.
- `possession_segments`: one or more rows that split the possession at atomic
  substitution boundaries.

Segment points and duration must sum exactly to their parent possession. Each
segment contains one home lineup and one away lineup.

## Consequences

Possession-level analyses can retain complete outcomes. Lineup models can train
on fixed-lineup segments without pretending substitutions create new
possessions. Multi-segment possessions remain identifiable through
`possession_id` and `possession_segment_index`.
