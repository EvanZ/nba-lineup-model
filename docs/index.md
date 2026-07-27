<p class="project-kicker">Possession-level modeling infrastructure</p>

# NBA Lineup Model

<p class="project-lead">
A reproducible NBA play-by-play data spine for context-sensitive player and
lineup value. The immediate goal is trustworthy five-on-five possession samples;
the longer horizon is nonlinear lineup modeling with interpretable context.
</p>

<div class="signal-strip">
  <div>
    <strong>Direct source</strong>
    <span>Byte-preserved NBA CDN responses</span>
  </div>
  <div>
    <strong>Exact context</strong>
    <span>Five players per team at every boundary</span>
  </div>
  <div>
    <strong>Validated output</strong>
    <span>Score, duration, and lineup invariants</span>
  </div>
</div>

```mermaid
flowchart LR
    RAW["NBA CDN"] --> EVT["Typed events"]
    EVT --> LUP["Lineups"]
    EVT --> POSS["Possessions"]
    LUP --> SEG["Fixed-lineup segments"]
    POSS --> SEG
    SEG --> MODEL["Contextual models"]
```

## Workflows

<div class="home-paths">
  <a href="guides/build-game/">
    <span class="path-index">01</span>
    <strong>Build one game</strong>
    <span>Fetch, normalize, reconstruct, and persist six Parquet contracts.</span>
  </a>
  <a href="guides/cross-season-audit/">
    <span class="path-index">02</span>
    <strong>Audit across seasons</strong>
    <span>Exercise reconstruction against regulation, playoff, and overtime feeds.</span>
  </a>
  <a href="architecture/overview/">
    <span class="path-index">03</span>
    <strong>Understand the pipeline</strong>
    <span>Review ownership boundaries, decisions, and failure policy.</span>
  </a>
  <a href="data/raw-responses/">
    <span class="path-index">04</span>
    <strong>Inspect data contracts</strong>
    <span>Trace raw source documents through modeling-ready segments.</span>
  </a>
</div>

## Current system

<div class="capability-list" markdown="1">

- Fetch play-by-play and boxscore JSON directly from NBA CDN endpoints.
- Preserve raw response bytes with fetch metadata and SHA-256 digests.
- Normalize source actions into typed canonical events.
- Reconstruct event-level lineups and stable lineup stints.
- Reconstruct basketball possessions with explicit terminal reasons.
- Split possessions when substitutions change either lineup.
- Audit exact invariants across regular-season, playoff, and overtime games.

</div>

## Engineering stance

1. Preserve source information before deriving basketball semantics.
2. Keep identifiers and clocks typed without lossy coercion.
3. Treat feed fields as evidence, not unquestionable ground truth.
4. Surface anomalies through structured issues and audit reports.
5. Establish trustworthy samples before optimizing model complexity.

!!! note "Contract status"

    Processed schemas are explicit and tested, but remain experimental until
    modeling requirements and cross-season feed behavior are better understood.
