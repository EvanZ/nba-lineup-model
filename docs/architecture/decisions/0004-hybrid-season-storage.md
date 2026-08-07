---
last_updated: "2026-07-26"
---

# ADR-0004: Hybrid Season Storage

<p class="adr-status"><strong>Decision status</strong><span>Accepted</span></p>

## Context

Per-game files make NBA source responses and reconstruction failures easy to
isolate. They also support independent retries and parallel processing.

Analytics over thousands of small Parquet files repeatedly pays file-open and
metadata costs. A single mutable season file, however, would require every game
build to rewrite shared state and would make isolated retries unsafe.

## Decision

Use three persisted granularities:

- Preserve one raw JSON response and provenance sidecar per endpoint and game.
- Preserve one processed Parquet artifact per contract and game.
- Compact validated game artifacts into curated Parquet datasets partitioned by
  season and season type.

Maintain a canonical game catalog and an append-oriented game build ledger as
separate Parquet contracts. Orchestration systems may mirror build state, but
their databases are not the authoritative data record.

Initial curated partitions may contain one Parquet part. Parts can be split when
their physical size or rewrite cost justifies it; callers must treat each
partition as a dataset rather than assume one file.

## Consequences

Game builds remain idempotent and diagnosable. Modeling scans open a small number
of compact files. Compaction becomes an explicit stage after game validation,
and correcting one game requires rebuilding the affected curated partition.
