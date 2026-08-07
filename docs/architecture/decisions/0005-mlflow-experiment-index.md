---
last_updated: "2026-07-29"
---

# ADR-0005: MLflow Experiment Index

<p class="adr-status"><strong>Decision status</strong><span>Accepted</span></p>

## Context

Immutable model directories provide reproducibility and exact artifact hashes,
but comparing parameters and metrics requires opening manifests and Parquet
tables manually. Neural searches also produce multiple candidate and fold
records that benefit from a searchable run hierarchy.

Making an external tracking service authoritative would weaken local
reproducibility and require a server to be available whenever training runs.

## Decision

Use MLflow as a secondary experiment index over completed immutable project
runs:

- keep typed manifests, Parquet outputs, checkpoints, and hashes authoritative;
- write local MLflow metadata to a project-owned SQLite database;
- copy immutable run artifacts into MLflow's artifact store;
- create season-level model and report experiments;
- represent hyperparameter candidates as child runs;
- identify every primary run by project run ID and manifest SHA-256;
- reject an attempted resynchronization when the same project run ID has a
  different manifest hash;
- allow direct SQLite logging without a running MLflow server;
- use the tracking server only for the UI or remote collaboration.

Completed model and evaluation CLIs index themselves automatically. An
idempotent synchronization command supports backfills and recovery after
tracking failures.

## Consequences

Parameters, metrics, fold histories, and artifacts are browsable in one UI
without changing the model-training contracts. The local artifact copy uses
additional disk space. SQLite is appropriate for one-machine development but
would need replacement with a shared database when concurrent remote writers
become a requirement.
