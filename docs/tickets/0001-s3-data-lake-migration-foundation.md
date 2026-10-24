# Slice 1: Immutable S3 data-lake migration foundation

## Objective

Move the authoritative basketball datasets off a single workstation without
changing the paths consumed by the existing modeling and web pipelines. The
first slice creates a deterministic inventory, an immutable upload contract,
and verification tooling. It does not delete local data or switch readers to
S3.

## Scope

- Classify every top-level `data/` and `artifacts/` dataset as one of:
  `archive`, `regenerate`, `already_archived`, or `defer`.
- Hash every file selected for archival and write a versioned JSON manifest.
- Map archived files beneath `s3://nba-gestalt-data/lake/` without
  flattening their existing dataset layout.
- Require an explicit `--execute` flag before any network write.
- Refuse to overwrite an S3 object and accept retries only when the stored
  SHA-256 metadata and byte count match the local manifest.
- Verify all uploaded objects and upload the migration manifest last.

## Initial policy

Archive raw source responses, catalogs, manifests, quality results, audits,
curated tables, analytical training tables, and external source datasets.
Treat `data/processed` as reproducible from raw data. Existing schedule state
and deployable web releases are already archived by their current publishing
flows. Defer model, rotation, report, and web research artifacts until a later
slice identifies the production runs that must be retained.

## Acceptance criteria

- Repeating a plan against unchanged files produces the same manifest ID.
- Each archived file records its source path, destination URI, byte count, and
  SHA-256 digest.
- Missing sources, overlapping destinations, symlinks, changed local files,
  and mismatched existing S3 objects fail closed.
- Dry-run upload performs no AWS calls.
- Unit tests cover deterministic planning, classification, immutable retry,
  upload, verification, and mismatch rejection.
- No local file is removed by this slice.

## Follow-up slices

1. Execute and verify the initial migration, then add S3-backed DuckDB views.
2. Select and archive production model runs and their lineage.
3. Switch collectors to write immutable raw objects directly to S3.
4. Add retention for reproducible EC2 working data and old application
   releases.
