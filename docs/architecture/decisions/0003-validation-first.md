---
last_updated: "2026-07-26"
---

# ADR-0003: Validation Before Modeling

<p class="adr-status"><strong>Decision status</strong><span>Accepted</span></p>

## Context

Play-by-play feeds contain source-version differences and rare event sequences.
A model can fit apparently plausible coefficients even when score attribution,
lineups, or possession boundaries are wrong.

## Decision

Require a cross-season audit layer before using processed samples for modeling.
The audit must:

- run the production reconstruction pipeline in memory;
- isolate exceptions by game;
- distinguish exact invariant failures from warnings;
- include regular season, playoffs, overtime, and feed-edge samples;
- emit machine-readable game and summary reports.

Exact score, segment score, duration, overtime expectation, and per-period
possession-balance checks determine failure status. The traditional boxscore
possession formula is approximate and remains a warning-only diagnostic.

## Consequences

Parser changes are tested against a versioned historical matrix. New feed
semantics become focused regression fixtures rather than undocumented special
cases. Modeling work can cite a concrete audit result instead of assuming the
data pipeline is correct.
