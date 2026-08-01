# Decision Log

Architecture decisions record choices that affect data interpretation or future
modeling. They explain why a rule exists and which alternatives were rejected.

| Decision | Status | Summary |
| --- | --- | --- |
| [ADR-0001](0001-direct-nba-source.md) | Accepted | Fetch direct NBA JSON and preserve raw bytes. |
| [ADR-0002](0002-possession-segments.md) | Accepted | Separate basketball possessions from fixed-lineup segments. |
| [ADR-0003](0003-validation-first.md) | Accepted | Require exact invariant audits before modeling. |
| [ADR-0004](0004-hybrid-season-storage.md) | Accepted | Separate game build artifacts from curated season partitions. |
| [ADR-0005](0005-mlflow-experiment-index.md) | Accepted | Index immutable model runs in MLflow without making it authoritative. |
| [ADR-0006](0006-historical-stats-v3-first.md) | Accepted | Use Stats V3 as the primary historical game archive and retain liveData only as a legacy fallback. |

New decisions should be added as numbered Markdown files. Existing decisions
should be superseded by a new record rather than rewritten after their context
has changed.
